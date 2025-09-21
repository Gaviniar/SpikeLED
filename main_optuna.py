# File: main_optuna.py
import argparse
import time
import warnings
from typing import Tuple

import optuna
from optuna.trial import TrialState

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import metrics

# === 你项目里的模块 ===
from spikenet import dataset, neuron
from spikenet.layers import SAGEAggregator
from spikenet.utils import (RandomWalkSampler, Sampler, add_selfloops,
                            set_seed)
from spikenet.temporal import GroupSparseDelay1D, TemporalSeparableReadout
from spikenet.gates import L2SGate

warnings.filterwarnings("ignore", category=UserWarning)

# --------- 全局（为兼容你模型里对全局变量 data/device 的引用） ---------
data = None          # will be set in build_data(...)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
y = None


# ========================= 你的模型（与 main.py 基本一致） =========================
class SpikeNet(nn.Module):
    def __init__(self, in_features, out_features, hids=[32], alpha=1.0, p=0.5,
                 dropout=0.7, bias=True, aggr='mean', sampler='sage',
                 surrogate='triangle', sizes=[5, 2], concat=False, act='LIF',
                 use_gs_delay=True, delay_groups=8, delay_set=(1, 3, 5),
                 use_learnable_p=True, use_tsr=True):

        super().__init__()

        tau = 1.0
        if sampler == 'rw':
            self.sampler = [RandomWalkSampler(add_selfloops(adj_matrix)) for adj_matrix in data.adj]
            self.sampler_t = [RandomWalkSampler(add_selfloops(adj_matrix)) for adj_matrix in data.adj_evolve]
        elif sampler == 'sage':
            self.sampler = [Sampler(add_selfloops(adj_matrix)) for adj_matrix in data.adj]
            self.sampler_t = [Sampler(add_selfloops(adj_matrix)) for adj_matrix in data.adj_evolve]
        else:
            raise ValueError(sampler)

        aggregators, snn = nn.ModuleList(), nn.ModuleList()

        in_dim = in_features
        for hid in hids:
            aggregators.append(SAGEAggregator(in_dim, hid, concat=concat, bias=bias, aggr=aggr))

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
                e_now = data.edges_evolve[t].shape[1]
                e_cum = data.edges[t].size(1)
                r_new = float(e_now) / max(1.0, float(e_cum))
                deg_mean = 2.0 * float(e_cum) / float(data.num_nodes)
                stats.append([r_new, deg_mean])
            self.time_stats = torch.tensor(stats, dtype=torch.float32)

        # 可学习 p 门
        if self.use_learnable_p:
            self.gate = L2SGate(num_layers=len(self.sizes), in_features=2, base_p=self.base_p)

        last_dim = in_dim
        T = len(data)
        if self.use_gs_delay:
            self.delay = GroupSparseDelay1D(D=last_dim, T=T, groups=delay_groups, delays=delay_set)

        if self.use_tsr:
            self.readout = TemporalSeparableReadout(D=last_dim, C=out_features, k=5)
        else:
            self.pooling = nn.Linear(T * last_dim, out_features)

        self._last_spikes_for_loss = None

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

        spikes = torch.stack(spikes_per_t, dim=1)  # [N, T, D]
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


# ========================= 训练/评估工具 =========================
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


def train_one_epoch(model, optimizer, loss_fn, train_loader, spike_reg=0.0, temp_reg=0.0):
    model.train()
    for nodes in train_loader:
        optimizer.zero_grad()
        logits = model(nodes)
        ce = loss_fn(logits, y[nodes])
        loss = ce
        # 正则
        spikes = model._last_spikes_for_loss
        if spikes is not None:
            if spike_reg > 0:
                loss = loss + spike_reg * spikes.mean()
            if temp_reg > 0 and spikes.size(1) > 1:
                loss = loss + temp_reg * (spikes[:, 1:] - spikes[:, :-1]).pow(2).mean()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, loader) -> Tuple[float, float]:
    model.eval()
    logits, labels = [], []
    for nodes in loader:
        logits.append(model(nodes))
        labels.append(y[nodes])
    logits = torch.cat(logits, dim=0).cpu()
    labels = torch.cat(labels, dim=0).cpu()
    preds = logits.argmax(1)
    macro = metrics.f1_score(labels, preds, average='macro')
    micro = metrics.f1_score(labels, preds, average='micro')
    return macro, micro


# ========================= 数据 & Loader =========================
def build_data(dataset_name: str, root: str, train_size: float, val_size: float, split_seed: int):
    global data, y
    name = dataset_name.lower()
    if name == "dblp":
        data = dataset.DBLP(root=root)
    elif name == "tmall":
        data = dataset.Tmall(root=root)
    elif name == "patent":
        data = dataset.Patent(root=root)
    else:
        raise ValueError(f"{dataset_name} is invalid. Only datasets (dblp, tmall, patent) are available.")

    # train : val : test
    data.split_nodes(train_size=train_size, val_size=val_size,
                     test_size=1 - train_size - val_size, random_state=split_seed)
    y = data.y.to(device)


def make_loaders(batch_size: int):
    train_loader = DataLoader(data.train_nodes.tolist(), pin_memory=False, batch_size=batch_size, shuffle=True)
    val_nodes = data.test_nodes.tolist() if data.val_nodes is None else data.val_nodes.tolist()
    val_loader = DataLoader(val_nodes, pin_memory=False, batch_size=200000, shuffle=False)
    test_loader = DataLoader(data.test_nodes.tolist(), pin_memory=False, batch_size=200000, shuffle=False)
    return train_loader, val_loader, test_loader


# ========================= Optuna 目标函数 =========================
def build_model_from_trial(trial, in_features, out_features):
    # 层数（保证 len(hids) == len(sizes)）
    n_layers = trial.suggest_int("n_layers", 2, 3)

    # 每层采样邻居数
    sizes = []
    for i in range(n_layers):
        if i == 0:
            sizes.append(trial.suggest_int(f"size_l{i}", 4, 15))
        else:
            sizes.append(trial.suggest_int(f"size_l{i}", 2, 10))

    # ——固定隐藏维度：128——
    hids = [128] * n_layers

    # 其它超参（categorical 一律 tuple）
    dropout = trial.suggest_float("dropout", 0.3, 0.8)
    lr = trial.suggest_float("lr", 1e-5, 5e-2, log=True)
    alpha = trial.suggest_float("alpha", 0.5, 2.0)
    alpha_warmup = trial.suggest_float("alpha_warmup", 0.2, 0.5)
    p = trial.suggest_float("p_base", 0.6, 0.95)

    sampler = trial.suggest_categorical("sampler", ("sage", "rw"))
    aggr = trial.suggest_categorical("aggr", ("mean", "sum"))
    surrogate = trial.suggest_categorical("surrogate", ("sigmoid", "triangle", "arctan", "mg", "super"))
    neuron_type = trial.suggest_categorical("neuron", ("IF", "LIF", "PLIF"))
    concat = trial.suggest_categorical("concat", (False, True))

    use_learnable_p = trial.suggest_categorical("use_learnable_p", (True, False))
    use_gs_delay = trial.suggest_categorical("use_gs_delay", (True, False))
    use_tsr = trial.suggest_categorical("use_tsr", (True, False))

    # 条件参数 - 只有在 use_gs_delay 为 True 时才进行搜索
    if use_gs_delay:
        delay_groups = trial.suggest_int("delay_groups", 4, 16)
        delays_choice = trial.suggest_categorical(
            "delays",
            ((0, 1, 2), (0, 1, 3, 5), (1, 3, 5))
        )
    else:
        delay_groups = 8  
        delays_choice = (1, 3, 5)  


    # 正则相关
    spike_reg = trial.suggest_categorical("spike_reg", (0.0, 1e-5, 1e-4, 5e-4, 1e-3))
    temp_reg = trial.suggest_categorical("temp_reg", (0.0, 1e-6, 5e-6, 1e-5, 5e-5))

    batch_size = 1024

    model = SpikeNet(in_features, out_features,
                     hids=hids, alpha=alpha, p=p,
                     dropout=dropout, bias=True, aggr=aggr, sampler=sampler,
                     surrogate=surrogate, sizes=sizes, concat=concat, act=neuron_type,
                     use_gs_delay=use_gs_delay, delay_groups=delay_groups, delay_set=delays_choice,
                     use_learnable_p=use_learnable_p, use_tsr=use_tsr).to(device)

    # 可选：把条件开关记到面板（不影响优化）
    trial.set_user_attr("gs_delay_enabled", bool(use_gs_delay))
    if use_gs_delay:
        trial.set_user_attr("delay_groups", int(delay_groups))
        trial.set_user_attr("delays", tuple(delays_choice))

    cfg = dict(lr=lr, alpha=alpha, alpha_warmup=alpha_warmup,
               spike_reg=spike_reg, temp_reg=temp_reg, batch_size=batch_size)
    return model, cfg


def objective_factory(args):
    # 数据只加载一次，所有 trial 共享同一拆分，保证公平
    build_data(args.dataset, args.root, train_size=args.train_size, val_size=args.val_size, split_seed=args.split_seed)

    def objective(trial: optuna.trial.Trial):
        set_seed(args.seed + trial.number)

        model, cfg = build_model_from_trial(trial, data.num_features, data.num_classes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
        loss_fn = nn.CrossEntropyLoss()

        # 局部 loader
        train_loader, val_loader, test_loader = make_loaders(cfg["batch_size"])

        best_val_micro = -1.0
        best_test_macro = 0.0
        best_test_micro = 0.0

        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            set_alpha_with_warmup(model, cfg["alpha"], epoch, args.epochs, cfg["alpha_warmup"])
            train_one_epoch(model, optimizer, loss_fn, train_loader,
                            spike_reg=cfg["spike_reg"], temp_reg=cfg["temp_reg"])

            val_macro, val_micro = evaluate(model, val_loader)
            test_macro, test_micro = evaluate(model, test_loader)

            # 向 Optuna 汇报中间结果，并用于裁剪
            trial.report(val_micro, step=epoch)

            # 保存最好验证对应的测试指标
            if val_micro > best_val_micro:
                best_val_micro = val_micro
                best_test_macro = test_macro
                best_test_micro = test_micro

            if trial.should_prune():
                # 把当前最优记录挂到 user_attrs，便于 dashboard 上查看
                trial.set_user_attr("best_test_macro_sofar", float(best_test_macro))
                trial.set_user_attr("best_test_micro_sofar", float(best_test_micro))
                raise optuna.exceptions.TrialPruned()

        # 结束后把关键统计写入 user_attrs
        trial.set_user_attr("best_val_micro", float(best_val_micro))
        trial.set_user_attr("best_test_macro", float(best_test_macro))
        trial.set_user_attr("best_test_micro", float(best_test_micro))
        trial.set_user_attr("epochs", int(args.epochs))
        trial.set_user_attr("train_time_sec", float(time.time() - t0))

        # 目标：最大化验证 Micro-F1
        return best_val_micro

    return objective


# ========================= CLI & Study =========================
def parse_args():
    p = argparse.ArgumentParser("SpikeNet Optuna Search")
    # 数据与拆分
    p.add_argument("--dataset", default="DBLP", help="DBLP | Tmall | Patent")
    p.add_argument("--root", default="/data4/zhengzhuoyu/data")
    p.add_argument("--train_size", type=float, default=0.8)
    p.add_argument("--val_size", type=float, default=0.05)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=2022)

    # 训练轮数
    p.add_argument("--epochs", type=int, default=60)

    # Optuna
    # 用新名字避免旧 Study（含 list choices）影响 Dashboard
    p.add_argument("--study", default="SpikeNet-HPO-v3")
    p.add_argument("--storage", default="sqlite:///s_optuna.db",
                   help="sqlite:///file.db 或 mysql://user:pwd@host/db 等")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--timeout", type=int, default=None)

    # 采样器/裁剪器
    p.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    p.add_argument("--pruner", choices=["median", "sha", "none"], default="median")

    return p.parse_args()


def make_study(args):
    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True, n_startup_trials=10)
    else:
        sampler = optuna.samplers.RandomSampler()

    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(5, args.epochs // 5),
                                             n_min_trials=5)
    elif args.pruner == "sha":
        pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=max(5, args.epochs // 6),
                                                        reduction_factor=3)
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(direction="maximize",
                                study_name=args.study,
                                storage=args.storage,
                                load_if_exists=True,
                                sampler=sampler,
                                pruner=pruner)
    return study


def main():
    args = parse_args()
    print(f"[Info] Using device: {device}")
    objective = objective_factory(args)
    study = make_study(args)

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    pruned = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics:")
    print(f"  Finished trials: {len(study.trials)}")
    print(f"  Pruned trials:   {len(pruned)}")
    print(f"  Complete trials: {len(complete)}")

    print("Best trial:")
    best = study.best_trial
    print(f"  Value (Val Micro-F1): {best.value:.4f}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    # 同时把 Test Macro/Micro 打印出来（如果存在）
    bm = best.user_attrs.get("best_test_macro")
    bmi = best.user_attrs.get("best_test_micro")
    if bm is not None and bmi is not None:
        print(f"  Linked Test (Macro, Micro): ({bm:.4f}, {bmi:.4f})")


if __name__ == "__main__":
    main()
