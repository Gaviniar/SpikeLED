# File: main.py
import argparse
import time

import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from spikenet import dataset, neuron
from spikenet.layers import SAGEAggregator
from spikenet.utils import (RandomWalkSampler, Sampler, add_selfloops,
                            set_seed, tab_printer)
from spikenet.temporal import GroupSparseDelay1D, TemporalSeparableReadout
from spikenet.gates import L2SGate


class SpikeNet(nn.Module):
    def __init__(self, in_features, out_features, hids=[32], alpha=1.0, p=0.5,
                 dropout=0.7, bias=True, aggr='mean', sampler='sage',
                 surrogate='triangle', sizes=[5, 2], concat=False, act='LIF',
                 use_gs_delay=True, delay_groups=8, delay_set=(1, 3, 5),
                 use_learnable_p=True, use_tsr=True):

        super().__init__()

        tau = 1.0
        if sampler == 'rw':
            self.sampler = [RandomWalkSampler(
                add_selfloops(adj_matrix)) for adj_matrix in data.adj]
            self.sampler_t = [RandomWalkSampler(add_selfloops(
                adj_matrix)) for adj_matrix in data.adj_evolve]
        elif sampler == 'sage':
            self.sampler = [Sampler(add_selfloops(adj_matrix))
                            for adj_matrix in data.adj]
            self.sampler_t = [Sampler(add_selfloops(adj_matrix))
                              for adj_matrix in data.adj_evolve]
        else:
            raise ValueError(sampler)

        aggregators, snn = nn.ModuleList(), nn.ModuleList()

        in_dim = in_features
        for hid in hids:
            aggregators.append(SAGEAggregator(in_dim, hid,
                                              concat=concat, bias=bias,
                                              aggr=aggr))

            if act == "IF":
                snn.append(neuron.IF(alpha=alpha, surrogate=surrogate))
            elif act == 'LIF':
                snn.append(neuron.LIF(tau, alpha=alpha, surrogate=surrogate))
            elif act == 'PLIF':
                snn.append(neuron.PLIF(tau, alpha=alpha, surrogate=surrogate))
            else:
                raise ValueError(act)

            in_dim = hid * 2 if concat else hid

        self.aggregators = aggregators
        self.dropout = nn.Dropout(dropout)
        self.snn = snn
        self.sizes = sizes
        self.base_p = p
        self.use_gs_delay = use_gs_delay
        self.use_tsr = use_tsr
        self.use_learnable_p = use_learnable_p

        # 时间统计（不参与梯度）：新增边占比、平均度
        with torch.no_grad():
            stats = []
            for t in range(len(data)):
                # edges: 累计；edges_evolve: 当期新增
                e_now = data.edges_evolve[t].shape[1]
                e_cum = data.edges[t].size(1)
                r_new = float(e_now) / max(1.0, float(e_cum))
                deg_mean = 2.0 * float(e_cum) / float(data.num_nodes)
                stats.append([r_new, deg_mean])
            self.time_stats = torch.tensor(stats, dtype=torch.float32)

        # 可学习 p 门（每层一个标量，输入 F=2）
        if self.use_learnable_p:
            self.gate = L2SGate(num_layers=len(self.sizes), in_features=2, base_p=self.base_p)

        last_dim = in_dim  # 经过所有聚合后 root 节点的维度
        T = len(data)      # 时间步
        if self.use_gs_delay:
            self.delay = GroupSparseDelay1D(D=last_dim, T=T, groups=delay_groups, delays=delay_set)

        if self.use_tsr:
            self.readout = TemporalSeparableReadout(D=last_dim, C=out_features, k=5)
        else:
            self.pooling = nn.Linear(T * last_dim, out_features)

        # 供正则使用的缓存
        self._last_spikes_for_loss = None

    def _layer_forward(self, h_list, sizes):
        """单时间步上的 SAGE+SNN 堆叠（沿用原有实现）。"""
        for i, aggregator in enumerate(self.aggregators):
            self_x = h_list[:-1]
            neigh_x = []
            for j, n_x in enumerate(h_list[1:]):
                neigh_x.append(n_x.view(-1, sizes[j], h_list[0].size(-1)))
            out = self.snn[i](aggregator(self_x, neigh_x))
            if i != len(sizes) - 1:
                out = self.dropout(out)
                # 更新到下一层的多跳特征切片
                # num_nodes 在外层维护，这里由调用方传入切分信息
            h_list = None  # 防止误用
        return out

    def encode(self, nodes: torch.Tensor):
        spikes_per_t = []
        sizes = self.sizes
        for t in range(len(data)):
            snapshot = data[t]
            sampler = self.sampler[t]
            sampler_t = self.sampler_t[t]

            x = snapshot.x
            h = [x[nodes].to(device)]
            num_nodes = [nodes.size(0)]
            nbr = nodes

            # 每层的 p_{l,t}
            if self.use_learnable_p:
                p_vec = self.gate(self.time_stats[t]).cpu().tolist()  # [L]
            else:
                p_vec = [self.base_p for _ in sizes]

            # 逐层扩展采样邻居
            for li, size in enumerate(sizes):
                p_now = float(p_vec[li])
                size_1 = max(int(size * p_now), 1)
                size_2 = size - size_1
                if size_2 > 0:
                    nbr_1 = sampler(nbr, size_1).view(nbr.size(0), size_1)
                    nbr_2 = sampler_t(nbr, size_2).view(nbr.size(0), size_2)
                    nbr = torch.cat([nbr_1, nbr_2], dim=1).flatten()
                else:
                    nbr = sampler(nbr, size_1).view(-1)
                num_nodes.append(nbr.size(0))
                h.append(x[nbr].to(device))

            # SAGE + SNN 堆叠
            for i, aggregator in enumerate(self.aggregators):
                self_x = h[:-1]
                neigh_x = []
                for j, n_x in enumerate(h[1:]):
                    neigh_x.append(n_x.view(-1, sizes[j], h[0].size(-1)))
                out = self.snn[i](aggregator(self_x, neigh_x))
                if i != len(sizes) - 1:
                    out = self.dropout(out)
                    h = torch.split(out, num_nodes[:-(i + 1)])

            spikes_per_t.append(out)  # [N, D]

        # [N, T, D]
        spikes = torch.stack(spikes_per_t, dim=1)
        if self.use_gs_delay:
            spikes = self.delay(spikes)
        self._last_spikes_for_loss = spikes
        neuron.reset_net(self)
        return spikes

    def forward(self, nodes):
        spikes = self.encode(nodes)  # [B, T, D]
        if self.use_tsr:
            return self.readout(spikes)
        else:
            return self.pooling(spikes.flatten(1))


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="DBLP",
                    help="Datasets (DBLP, Tmall, Patent). (default: DBLP)")
parser.add_argument('--sizes', type=int, nargs='+', default=[5, 2], help='Neighborhood sampling size for each layer. (default: [5, 2])')
parser.add_argument('--hids', type=int, nargs='+',
                    default=[128, 10], help='Hidden units for each layer. (default: [128, 10])')
parser.add_argument("--aggr", nargs="?", default="mean",
                    help="Aggregate function ('mean', 'sum'). (default: 'mean')")
parser.add_argument("--sampler", nargs="?", default="sage",
                    help="Neighborhood Sampler, including uniform sampler from GraphSAGE ('sage') and random walk sampler ('rw'). (default: 'sage')")
parser.add_argument("--surrogate", nargs="?", default="sigmoid",
                    help="Surrogate function ('sigmoid', 'triangle', 'arctan', 'mg', 'super'). (default: 'sigmoid')")
parser.add_argument("--neuron", nargs="?", default="LIF",
                    help="Spiking neuron used for training. (IF, LIF, PLIF). (default: LIF")
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Batch size for training. (default: 1024)')
parser.add_argument('--lr', type=float, default=5e-3,
                    help='Learning rate for training. (default: 5e-3)')
parser.add_argument('--train_size', type=float, default=0.4,
                    help='Ratio of nodes for training. (default: 0.4)')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='Smooth factor for surrogate learning. (default: 1.0)')
parser.add_argument('--p', type=float, default=0.5,
                    help='Percentage of sampled neighborhoods for g_t. (default: 0.5)')
parser.add_argument('--dropout', type=float, default=0.7,
                    help='Dropout probability. (default: 0.7)')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs. (default: 100)')
parser.add_argument('--concat', action='store_true',
                    help='Whether to concat node representation and neighborhood representations. (default: False)')
parser.add_argument('--seed', type=int, default=2022,
                    help='Random seed for model. (default: 2022)')
# 新增功能开关
parser.add_argument('--use_gs_delay', type=int, default=1, help='Use Group-Sparse Delay kernel (1|0).')
parser.add_argument('--delay_groups', type=int, default=8, help='Delay groups for GS-Delay.')
parser.add_argument('--delays', type=int, nargs='+', default=[1,3,5], help='Sparse delay set Δ.')
parser.add_argument('--use_tsr', type=int, default=1, help='Use Temporal Separable Readout (1|0).')
parser.add_argument('--use_learnable_p', type=int, default=1, help='Use learnable sampling gate (1|0).')
# 正则与课程
parser.add_argument('--spike_reg', type=float, default=0.0, help='Avg firing-rate regularization λ.')
parser.add_argument('--temp_reg', type=float, default=0.0, help='Temporal consistency regularization λ.')
parser.add_argument('--alpha_warmup', type=float, default=0.3, help='Warm-up ratio of epochs for surrogate alpha (0~1).')

try:
    args = parser.parse_args()
    args.test_size = 1 - args.train_size
    args.train_size = args.train_size - 0.05
    args.val_size = 0.05
    args.split_seed = 42
    tab_printer(args)
except:
    parser.print_help()
    exit(0)

assert len(args.hids) == len(args.sizes), "must be equal!"

if args.dataset.lower() == "dblp":
    data = dataset.DBLP(root='/data4/zhengzhuoyu/data')
elif args.dataset.lower() == "tmall":
    data = dataset.Tmall(root='/data4/zhengzhuoyu/data')
elif args.dataset.lower() == "patent":
    data = dataset.Patent(root='/data4/zhengzhuoyu/data')
else:
    raise ValueError(
        f"{args.dataset} is invalid. Only datasets (dblp, tmall, patent) are available.")

# train:val:test
data.split_nodes(train_size=args.train_size, val_size=args.val_size,
                 test_size=args.test_size, random_state=args.split_seed)

set_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

y = data.y.to(device)

train_loader = DataLoader(data.train_nodes.tolist(), pin_memory=False, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(data.test_nodes.tolist() if data.val_nodes is None else data.val_nodes.tolist(),
                        pin_memory=False, batch_size=200000, shuffle=False)
test_loader = DataLoader(data.test_nodes.tolist(), pin_memory=False, batch_size=200000, shuffle=False)

model = SpikeNet(data.num_features, data.num_classes, alpha=args.alpha,
                 dropout=args.dropout, sampler=args.sampler, p=args.p,
                 aggr=args.aggr, concat=args.concat, sizes=args.sizes, surrogate=args.surrogate,
                 hids=args.hids, act=args.neuron, bias=True,
                 use_gs_delay=bool(args.use_gs_delay), delay_groups=args.delay_groups,
                 delay_set=tuple(args.delays), use_learnable_p=bool(args.use_learnable_p),
                 use_tsr=bool(args.use_tsr)).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()


def set_alpha_with_warmup(net: nn.Module, base_alpha: float, epoch: int, total_epochs: int, warmup_ratio: float):
    if warmup_ratio <= 0:
        cur_alpha = base_alpha
    else:
        warm_epochs = max(1, int(total_epochs * warmup_ratio))
        factor = min(1.0, float(epoch) / float(warm_epochs))
        cur_alpha = max(1e-3, base_alpha * factor)
    # 将 alpha 写入所有神经元模块
    for m in net.modules():
        if hasattr(m, 'alpha') and isinstance(m.alpha, torch.Tensor):
            m.alpha.data.fill_(cur_alpha)


def train():
    model.train()
    for nodes in tqdm(train_loader, desc='Training'):
        optimizer.zero_grad()
        logits = model(nodes)
        ce = loss_fn(logits, y[nodes])
        loss = ce
        # 正则项
        spikes = model._last_spikes_for_loss
        if spikes is not None:
            if args.spike_reg > 0:
                reg_spike = spikes.mean()
                loss = loss + args.spike_reg * reg_spike
            if args.temp_reg > 0 and spikes.size(1) > 1:
                reg_temp = (spikes[:, 1:] - spikes[:, :-1]).pow(2).mean()
                loss = loss + args.temp_reg * reg_temp
        loss.backward()
        optimizer.step()


@torch.no_grad()
def test(loader):
    model.eval()
    logits = []
    labels = []
    for nodes in loader:
        logits.append(model(nodes))
        labels.append(y[nodes])
    logits = torch.cat(logits, dim=0).cpu()
    labels = torch.cat(labels, dim=0).cpu()
    preds = logits.argmax(1)
    metric_macro = metrics.f1_score(labels, preds, average='macro')
    metric_micro = metrics.f1_score(labels, preds, average='micro')
    return metric_macro, metric_micro


best_val_metric = test_metric = 0
start = time.time()
for epoch in range(1, args.epochs + 1):
    # surrogate alpha 课程
    set_alpha_with_warmup(model, args.alpha, epoch, args.epochs, args.alpha_warmup)
    train()
    val_metric, test_metric = test(val_loader), test(test_loader)
    if val_metric[1] > best_val_metric:
        best_val_metric = val_metric[1]
        best_test_metric = test_metric
    end = time.time()
    print(
        f'Epoch: {epoch:03d}, Val: {val_metric[1]:.4f}, Test: {test_metric[1]:.4f}, '
        f'Best: Macro-{best_test_metric[0]:.4f}, Micro-{best_test_metric[1]:.4f}, Time elapsed {end-start:.2f}s')

# # 如需保存二进制脉冲表示（spikes）
# # emb = model.encode(torch.arange(data.num_nodes)).cpu()
# # torch.save(emb, 'emb.pth')
