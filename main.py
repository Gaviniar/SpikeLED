import argparse
import time

import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from spikenet import dataset, neuron
from spikenet.layers import SAGEAggregator
from spikenet.temporal import GroupSparseDelay1D, TemporalSeparableReadout, compute_temporal_stats
from spikenet.gates import L2SGate
from spikenet.utils import (RandomWalkSampler, Sampler, add_selfloops,
                            set_seed, tab_printer)


class SpikeNet(nn.Module):
    def __init__(self, in_features, out_features, hids=[32], alpha=1.0, p=0.5,
                 dropout=0.7, bias=True, aggr='mean', sampler='sage',
                 surrogate='triangle', sizes=[5, 2], concat=False, act='LIF',
                 use_gs_delay=True, delay_groups=8, delay_set=(1, 3, 5),
                 use_learnable_p=True, use_tsr=True, learnable_threshold=False):

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

        for hid in hids:
            aggregators.append(SAGEAggregator(in_features, hid,
                                              concat=concat, bias=bias,
                                              aggr=aggr))

            if act == "IF":
                snn.append(neuron.IF(alpha=alpha, surrogate=surrogate, 
                                   learnable_threshold=learnable_threshold))
            elif act == 'LIF':
                snn.append(neuron.LIF(tau, alpha=alpha, surrogate=surrogate,
                                    learnable_threshold=learnable_threshold))
            elif act == 'PLIF':
                snn.append(neuron.PLIF(tau, alpha=alpha, surrogate=surrogate,
                                     learnable_threshold=learnable_threshold))
            else:
                raise ValueError(act)

            in_features = hid * 2 if concat else hid

        self.aggregators = aggregators
        self.dropout = nn.Dropout(dropout)
        self.snn = snn
        self.sizes = sizes
        self.p = p
        self.use_gs_delay = use_gs_delay
        self.use_tsr = use_tsr
        self.use_learnable_p = use_learnable_p

        # 可学习采样门
        if self.use_learnable_p:
            self.p_gate = L2SGate(input_dim=3)

        # 延迟核
        T = len(data)
        last_dim = in_features
        if self.use_gs_delay:
            self.delay = GroupSparseDelay1D(D=last_dim, T=T, groups=delay_groups, delays=delay_set)

        # 读出层
        if self.use_tsr:
            self.readout = TemporalSeparableReadout(D=last_dim, C=out_features, k=5)
        else:
            self.pooling = nn.Linear(len(data) * in_features, out_features)

        # 用于存储最后一次的spikes用于计算正则项
        self._last_spikes_for_loss = None

    def encode(self, nodes):
        spikes = []
        sizes = self.sizes
        for time_step in range(len(data)):
            snapshot = data[time_step]
            sampler = self.sampler[time_step]
            sampler_t = self.sampler_t[time_step]

            # 可学习采样门
            if self.use_learnable_p:
                temporal_stats = compute_temporal_stats(data.adj, time_step)
                p_t = self.p_gate(temporal_stats.to(device)).item()
            else:
                p_t = self.p

            x = snapshot.x
            h = [x[nodes].to(device)]
            num_nodes = [nodes.size(0)]
            nbr = nodes
            for size in sizes:
                size_1 = max(int(size * p_t), 1)
                size_2 = size - size_1

                if size_2 > 0:
                    nbr_1 = sampler(nbr, size_1).view(nbr.size(0), size_1)
                    nbr_2 = sampler_t(nbr, size_2).view(nbr.size(0), size_2)
                    nbr = torch.cat([nbr_1, nbr_2], dim=1).flatten()
                else:
                    nbr = sampler(nbr, size_1).view(-1)

                num_nodes.append(nbr.size(0))
                h.append(x[nbr].to(device))

            for i, aggregator in enumerate(self.aggregators):
                self_x = h[:-1]
                neigh_x = []
                for j, n_x in enumerate(h[1:]):
                    neigh_x.append(n_x.view(-1, sizes[j], h[0].size(-1)))

                out = self.snn[i](aggregator(self_x, neigh_x))
                if i != len(sizes) - 1:
                    out = self.dropout(out)
                    h = torch.split(out, num_nodes[:-(i + 1)])

            spikes.append(out)

        spikes = torch.stack(spikes, dim=1)  # [B, T, D]
        
        # 应用延迟核
        if self.use_gs_delay:
            spikes = self.delay(spikes)
            
        # 存储用于正则项计算
        self._last_spikes_for_loss = spikes.detach()
        
        neuron.reset_net(self)
        return spikes

    def forward(self, nodes):
        spikes = self.encode(nodes)
        if self.use_tsr:
            return self.readout(spikes)
        else:
            return self.pooling(spikes.flatten(1))


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="DBLP",
                    help="Datasets (DBLP, Tmall, Patent). (default: DBLP)")
parser.add_argument('--sizes', type=int, nargs='+', default=[5, 2], 
                    help='Neighborhood sampling size for each layer. (default: [5, 2])')
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
# 新增的参数
parser.add_argument('--use_gs_delay', action='store_true', default=True,
                    help='Whether to use group sparse delay kernel. (default: True)')
parser.add_argument('--use_tsr', action='store_true', default=True,
                    help='Whether to use temporal separable readout. (default: True)')
parser.add_argument('--use_learnable_p', action='store_true', default=True,
                    help='Whether to use learnable sampling gate. (default: True)')
parser.add_argument('--learnable_threshold', action='store_true', default=False,
                    help='Whether to use learnable threshold parameters. (default: False)')
parser.add_argument('--delay_groups', type=int, default=8,
                    help='Number of groups for delay kernel. (default: 8)')
parser.add_argument('--lambda_spike', type=float, default=1e-4,
                    help='Regularization weight for spike rate. (default: 1e-4)')
parser.add_argument('--lambda_temp', type=float, default=1e-4,
                    help='Regularization weight for temporal consistency. (default: 1e-4)')
parser.add_argument('--alpha_warmup', action='store_true', default=False,
                    help='Whether to use alpha warmup. (default: False)')

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
    data = dataset.DBLP()
elif args.dataset.lower() == "tmall":
    data = dataset.Tmall()
elif args.dataset.lower() == "patent":
    data = dataset.Patent()
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
                 use_gs_delay=args.use_gs_delay, delay_groups=args.delay_groups,
                 use_tsr=args.use_tsr, use_learnable_p=args.use_learnable_p,
                 learnable_threshold=args.learnable_threshold).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()

# Alpha warmup scheduler
def get_current_alpha(epoch, total_epochs, base_alpha, warmup_ratio=0.3):
    if not args.alpha_warmup:
        return base_alpha
    warmup_epochs = int(total_epochs * warmup_ratio)
    if epoch < warmup_epochs:
        return base_alpha * (epoch / warmup_epochs)
    return base_alpha

def train():
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_spike_reg = 0
    total_temp_reg = 0
    
    for nodes in tqdm(train_loader, desc='Training'):
        optimizer.zero_grad()
        logits = model(nodes)
        ce_loss = loss_fn(logits, y[nodes])
        
        # 正则项计算
        reg_spike = 0
        reg_temp = 0
        
        if model._last_spikes_for_loss is not None:
            spikes = model._last_spikes_for_loss  # [B, T, D]
            
            # 平均放电率正则
            reg_spike = spikes.mean()
            
            # 时间一致性正则
            if spikes.size(1) > 1:  # T > 1
                reg_temp = (spikes[:, 1:] - spikes[:, :-1]).pow(2).mean()
        
        # 总损失
        loss = ce_loss + args.lambda_spike * reg_spike + args.lambda_temp * reg_temp
        
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_spike_reg += reg_spike.item() if isinstance(reg_spike, torch.Tensor) else reg_spike
        total_temp_reg += reg_temp.item() if isinstance(reg_temp, torch.Tensor) else reg_temp
    
    return {
        'total_loss': total_loss / len(train_loader),
        'ce_loss': total_ce_loss / len(train_loader),
        'spike_reg': total_spike_reg / len(train_loader),
        'temp_reg': total_temp_reg / len(train_loader)
    }


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
    logits = logits.argmax(1)
    metric_macro = metrics.f1_score(labels, logits, average='macro')
    metric_micro = metrics.f1_score(labels, logits, average='micro')
    return metric_macro, metric_micro


best_val_metric = test_metric = 0
start = time.time()
for epoch in range(1, args.epochs + 1):
    # Alpha warmup
    if args.alpha_warmup:
        current_alpha = get_current_alpha(epoch, args.epochs, args.alpha)
        for module in model.modules():
            if hasattr(module, 'alpha'):
                module.alpha.fill_(current_alpha)
    
    train_stats = train()
    val_metric, test_metric = test(val_loader), test(test_loader)
    if val_metric[1] > best_val_metric:
        best_val_metric = val_metric[1]
        best_test_metric = test_metric
    end = time.time()
    print(
        f'Epoch: {epoch:03d}, Val: {val_metric[1]:.4f}, Test: {test_metric[1]:.4f}, '
        f'Best: Macro-{best_test_metric[0]:.4f}, Micro-{best_test_metric[1]:.4f}, '
        f'CE: {train_stats["ce_loss"]:.4f}, Spike: {train_stats["spike_reg"]:.6f}, '
        f'Temp: {train_stats["temp_reg"]:.6f}, Time: {end-start:.2f}s')

# save binary node embeddings (spikes)
# emb = model.encode(torch.arange(data.num_nodes)).cpu()
# torch.save(emb, 'emb.pth')
