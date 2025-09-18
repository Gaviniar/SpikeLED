# File: main_static.py
import argparse
import os.path as osp
import time

import torch
import torch.nn as nn
from sklearn import metrics
from spikenet import dataset, neuron
from spikenet.layers import SAGEAggregator
from spikenet.utils import (RandomWalkSampler, Sampler, add_selfloops,
                            set_seed, tab_printer)
from torch.utils.data import DataLoader
from torch_geometric.datasets import Flickr, Reddit
from torch_geometric.utils import to_scipy_sparse_matrix
from tqdm import tqdm

from spikenet.temporal import GroupSparseDelay1D, TemporalSeparableReadout
from spikenet.gates import L2SGate


class SpikeNet(nn.Module):
    def __init__(self, in_features, out_features, hids=[32], alpha=1.0, T=5,
                 dropout=0.7, bias=True, aggr='mean', sampler='sage',
                 surrogate='triangle', sizes=[5, 2], concat=False, act='LIF',
                 use_gs_delay=True, delay_groups=8, delay_set=(1,3,5), use_tsr=True):

        super().__init__()

        tau = 1.0
        if sampler == 'rw':
            self.sampler = RandomWalkSampler(add_selfloops(to_scipy_sparse_matrix(data.edge_index)))
        elif sampler == 'sage':
            self.sampler = Sampler(add_selfloops(to_scipy_sparse_matrix(data.edge_index)))
        else:
            raise ValueError(sampler)

        del data.edge_index

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
        self.T = T
        self.use_gs_delay = use_gs_delay
        self.use_tsr = use_tsr

        last_dim = in_dim
        if self.use_gs_delay:
            self.delay = GroupSparseDelay1D(D=last_dim, T=T, groups=delay_groups, delays=delay_set)
        if self.use_tsr:
            self.readout = TemporalSeparableReadout(D=last_dim, C=out_features, k=5)
        else:
            self.pooling = nn.Linear(T * last_dim, out_features)

        # 退化版门控：无时间统计时，等价为每层一个可学习标量（通过 bias 表现）
        self.gate = L2SGate(num_layers=len(self.sizes), in_features=2, base_p=0.5)
        self._zero_stats = torch.zeros(2)

        self._last_spikes_for_loss = None

    def encode(self, nodes):
        spikes = []
        sizes = self.sizes
        x = data.x

        for time_step in range(self.T):
            h = [x[nodes].to(device)]
            num_nodes = [nodes.size(0)]
            nbr = nodes

            p_vec = self.gate(self._zero_stats).cpu().tolist()

            for li, size in enumerate(sizes):
                p_now = float(p_vec[li])
                size_1 = max(int(size * p_now), 1)
                size_2 = size - size_1
                if size_2 > 0:
                    nbr_1 = self.sampler(nbr, size_1)
                    nbr_2 = self.sampler(nbr, size_2)  # 静态图无演化，用同一采样器
                    nbr = torch.cat([nbr_1.view(nbr.size(0), size_1), nbr_2.view(nbr.size(0), size_2)], dim=1).flatten()
                else:
                    nbr = self.sampler(nbr, size_1).view(-1)
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

        spikes = torch.stack(spikes, dim=1)  # [N,T,D]
        if self.use_gs_delay:
            spikes = self.delay(spikes)
        self._last_spikes_for_loss = spikes
        neuron.reset_net(self)
        return spikes

    def forward(self, nodes):
        spikes = self.encode(nodes)
        if self.use_tsr:
            return self.readout(spikes)
        else:
            return self.pooling(spikes.flatten(1))


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="flickr",
                    help="Datasets (Reddit and Flickr only). (default: Flickr)")
parser.add_argument('--sizes', type=int, nargs='+', default=[5, 2],
                    help='Neighborhood sampling size for each layer. (default: [5, 2])')
parser.add_argument('--hids', type=int, nargs='+',
                    default=[512, 10], help='Hidden units for each layer. (default: [128, 10])')
parser.add_argument("--aggr", nargs="?", default="mean",
                    help="Aggregate function ('mean', 'sum'). (default: 'mean')")
parser.add_argument("--sampler", nargs="?", default="sage",
                    help="Neighborhood Sampler, including uniform sampler from GraphSAGE ('sage') and random walk sampler ('rw'). (default: 'sage')")
parser.add_argument("--surrogate", nargs="?", default="sigmoid",
                    help="Surrogate function ('sigmoid', 'triangle', 'arctan', 'mg', 'super'). (default: 'sigmoid')")
parser.add_argument("--neuron", nargs="?", default="LIF",
                    help="Spiking neuron used for training. (IF, LIF, PLIF). (default: LIF")
parser.add_argument('--batch_size', type=int, default=2048,
                    help='Batch size for training. (default: 1024)')
parser.add_argument('--lr', type=float, default=5e-3,
                    help='Learning rate for training. (default: 5e-3)')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='Smooth factor for surrogate learning. (default: 1.0)')
parser.add_argument('--T', type=int, default=15,
                    help='Number of time steps. (default: 15)')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout probability. (default: 0.5)')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs. (default: 100)')
parser.add_argument('--concat', action='store_true',
                    help='Whether to concat node representation and neighborhood representations. (default: False)')
parser.add_argument('--seed', type=int, default=2022,
                    help='Random seed for model. (default: 2022)')
# 新增
parser.add_argument('--use_gs_delay', type=int, default=1)
parser.add_argument('--delay_groups', type=int, default=8)
parser.add_argument('--delays', type=int, nargs='+', default=[1,3,5])
parser.add_argument('--use_tsr', type=int, default=1)
parser.add_argument('--spike_reg', type=float, default=0.0)
parser.add_argument('--temp_reg', type=float, default=0.0)
parser.add_argument('--alpha_warmup', type=float, default=0.3)

try:
    args = parser.parse_args()
    args.split_seed = 42
    tab_printer(args)
except:
    parser.print_help()
    exit(0)

assert len(args.hids) == len(args.sizes), "must be equal!"

root = "data/"  # Specify your root path

if args.dataset.lower() == "reddit":
    dataset_obj = Reddit(osp.join(root, 'Reddit'))
    data = dataset_obj[0]
elif args.dataset.lower() == "flickr":
    dataset_obj = Flickr(osp.join(root, 'Flickr'))
    data = dataset_obj[0]

data.x = torch.nn.functional.normalize(data.x, dim=1)

set_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

y = data.y.to(device)

train_loader = DataLoader(data.train_mask.nonzero().view(-1), pin_memory=False, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(data.val_mask.nonzero().view(-1), pin_memory=False, batch_size=10000, shuffle=False)
test_loader = DataLoader(data.test_mask.nonzero().view(-1), pin_memory=False, batch_size=10000, shuffle=False)

model = SpikeNet(dataset_obj.num_features, dataset_obj.num_classes, alpha=args.alpha,
                 dropout=args.dropout, sampler=args.sampler, T=args.T,
                 aggr=args.aggr, concat=args.concat, sizes=args.sizes, surrogate=args.surrogate,
                 hids=args.hids, act=args.neuron, bias=True,
                 use_gs_delay=bool(args.use_gs_delay), delay_groups=args.delay_groups,
                 delay_set=tuple(args.delays), use_tsr=bool(args.use_tsr)).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()


def set_alpha_with_warmup(net: nn.Module, base_alpha: float, epoch: int, total_epochs: int, warmup_ratio: float):
    if warmup_ratio <= 0:
        cur_alpha = base_alpha
    else:
        warm_epochs = max(1, int(total_epochs * warmup_ratio))
        factor = min(1.0, float(epoch) / float(warm_epochs))
        cur_alpha = max(1e-3, base_alpha * factor)
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
        spikes = model._last_spikes_for_loss
        if spikes is not None:
            if args.spike_reg > 0:
                loss = loss + args.spike_reg * spikes.mean()
            if args.temp_reg > 0 and spikes.size(1) > 1:
                loss = loss + args.temp_reg * (spikes[:,1:] - spikes[:,:-1]).pow(2).mean()
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
    set_alpha_with_warmup(model, args.alpha, epoch, args.epochs, args.alpha_warmup)
    train()
    val_metric, test_metric = test(val_loader), test(test_loader)
    if val_metric[1] > best_val_metric:
        best_val_metric = val_metric[1]
        best_test_metric = test_metric
    end = time.time()
    print(
        f'Epoch: {epoch:03d}, Val: {val_metric[1]:.4f}, Test: {test_metric[1]:.4f}, Best: Macro-{best_test_metric[0]:.4f}, Micro-{best_test_metric[1]:.4f}, Time elapsed {end-start:.2f}s')

# # 保存脉冲嵌入（可选）
# # emb = model.encode(torch.arange(data.num_nodes)).cpu()
# # torch.save(emb, 'emb.pth')
