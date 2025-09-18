# 方案总览（方法名可用：**SpikeNet-LED**：**L**earnable **E**fficient **D**elays）

核心由三块组成，可单独或组合使用：

1) **GS-Delay**（**G**roup-**S**hared Sparse Delay Kernel，组共享稀疏延迟核）用**组共享 + 离散稀疏**的 1D 时间卷积近似 SOTA 的**高斯延迟核**，以**几乎不增加 FLOPs**的代价，显式建模“信息滞后/延迟影响”，并保持梯度稳定；比 SOTA 的逐通道高斯核**更省参/省算**。灵感来自对方的延迟卷积思想与稳定性条件，我们保留其**理论动机**但在结构上做**效率友好的蒸馏**与离散化近似（见“原理”与“代码意见”）。
2) **L2S-Gate**（**L**earnable **L**ocal **2**-source Sampling Gate，可学习时序-结构采样门）将你代码里的常量 `p`（当前累计图 vs. 演化图邻采样比例）变为**超轻量可学习门控** `p_{l,t}`（按层/按时刻），依据**当前时间片统计（新增边占比/平均度/平均放电率）**自动分配“取多少历史/多少当下”。**不对采样操作反传**，因此几乎**零额外开销**，却能明显提升鲁棒性与泛化。
3) **TSR-Readout**（Temporal Separable Readout，可分离时序读出）
   用**Depthwise(时间维) 1D 卷积 + 1×1** 代替 `Linear(T·D, C)` 的大读出层，使**参数/T 解耦**，并显式建模时间模式；与 1) 兼容，可以直接把 GS-Delay 的稀疏核**复用在读出**或作为前置时间滤波。SOTA 里已证明**训练出的时间加权优于简单拼接**，我们给出**更高效的结构化替代**。

**训练层面的两处“低成本稳态增强”**（可选）：

- **SCAT**：自适应阈值 + 替代梯度课程（逐步增大 `alpha`），配合**平均放电率正则**，让梯度更稳、能耗受控；
- **Temporal Consistency**：相邻时刻表示一致性（L2/InfoNCE 的轻量变体），呼应 SOTA 的**时间平滑正则**但略微更强。

---

## 原理：为什么这三件事会有效？

### （1）组共享稀疏延迟核（GS-Delay）

- **效仿但不复制 SOTA**：SOTA 用**高斯核**对历史脉冲加权并“推迟”到将来，缓解信息遗忘并拟合真实传播时延；同时给出了**σ 与核长 Ks 的约束**以避免梯度爆炸/消失，解释了为何延迟在训练上是安全的。
- **我们的改进**：把“每通道可学习高斯核”替换为**组共享（G 组）+ 离散稀疏**的延迟核：仅在有限延迟集合 Δ={1,3,5,…} 上放非零权，且**每组共享同一组核参数**。这样：
  - 复杂度从 **O(D·Ks)** → **O(G·|Δ|)**（D≫G, |Δ|≪Ks），**省参/省算**；
  - 离散核的**L1 归一 + 最大延迟≤Ks** 可直接承接 SOTA 的梯度有界推理思路（把高斯上界替换为离散权的上界），**稳定性不减**且便于实现。
- **与 SNN 的契合**：脉冲稀疏 → 实际有效卷积更少；离散核在稀疏脉冲上更易“事件驱动”地生效，提升**能效/速度**。

### （2）可学习采样门（L2S-Gate）

- **动机**：你已有 `p` 在**累计图**与**演化图**采样之间分配邻居数（一个很棒的时序结构融合点）；不同层/时刻最佳 `p` 不同。
- **做法**：用 **σ(w·z_t+b)** 产出 `p_{l,t}`（`z_t` 为无梯度时间片统计：新增边比例、平均度、上一轮放电率均值等）。只改变**邻居配额**，**不改变采样算子/不反传**，训练开销≈0。
- **收益**：当**突发变化**或**度异质**时，模型自动“多看当前/多看历史”，降低方差、提升稳定性。与 SOTA 的“延迟使历史影响后效”形成**互补**：我们不仅延迟历史，还**自适应决定看多少历史**。

### （3）可分离时序读出（TSR-Readout）

- **动机**：原始 `Linear(T·D,C)` 对 T 敏感、参数大；SOTA 用**可训练时间权重池化**证明“学到的时间权重更好”。
- **做法**：Depthwise 1D（仅时间维）+ Pointwise 1×1，再全局池化 → `Linear(D,C)`。
- **收益**：参数与 T 基本解耦，捕捉**短期与多尺度模式**；与 GS-Delay 共用 1D 卷积框架，工程上简单。

### （4）训练稳态增强（SCAT + Temporal Consistency）

- **SCAT**：把 `gamma, thresh_decay` 变为**可学习且有界**的参数；`alpha` 采用**课程**（前期小、后期大）；再加**平均放电率正则**，可控能耗与梯度噪声。
- **时间一致性**：沿着 SOTA 的**相邻时刻平滑正则**，我们可用更轻的**投影+L2/InfoNCE**，提升低标注/分布漂移时段的鲁棒性，开销很小。

---

# 与你仓库的对接（**只需小改动，多为“增量模块”**）

## 需要新增的文件/模块

- `spikenet/temporal.py`（新）：放 **GS-Delay** 与 **TSR-Readout** 和一个**小工具函数**。
- （可选）`spikenet/gates.py`（新）：放 **L2S-Gate**（也可直接写进 `SpikeNet`）。

## 需要小改动的现有文件

1. **采样 C++ 小修（强烈建议）**：`spikenet/sample_neighber.cpp`你当前 `replace==true` 且 `row_count>num_neighbors` 时逻辑更接近“无放回”，会带来采样偏差。改为真正**有放回**采样（每次 `rand()%row_count` 取一个）即可；代价为常数级，能使实验更公允。
2. **`main.py` / `main_static.py`**：
   - 在 `SpikeNet.__init__` 中注入 **TSR-Readout**（替换 `self.pooling`），注入 **GS-Delay**（在 `encode` 函数收集完 `spikes` 后做时间卷积）；
   - 打开 **L2S-Gate**：把常量 `self.p` 替换为 `p_{l,t}`；
   - 训练循环中加入 **放电率正则** 和（可选）**时间一致性**项；
   - （可选）用**线性 warm-up** 更新 `alpha`；
3. **`spikenet/neuron.py`**：
   - 把全局常量 `gamma, thresh_decay` 变为**`nn.Parameter` + Sigmoid 约束**（保证(0,1)）；
   - 保持向后兼容（不改默认行为）。

---

## 关键代码建议（**骨架级**，便于你交给 AI 直接补全）

### A) `spikenet/temporal.py`

```python
import torch, torch.nn as nn
import torch.nn.functional as F

class GroupSparseDelay1D(nn.Module):
    """
    组共享 + 稀疏离散延迟核（仅沿时间维做 Depthwise 1D）
    输入: x [B, T, D] ；输出: y [B, T, D]
    """
    def __init__(self, D, T, groups=8, delays=(1,3,5), init='gaussian_like'):
        super().__init__()
        assert D % groups == 0
        self.D, self.T, self.G = D, T, groups
        self.delays = list(delays)  # 稀疏位置
        # 每组对每个 delay 一个权重（标量），共享到组内所有通道
        self.weight = nn.Parameter(torch.zeros(self.G, len(self.delays)))
        # 可选：中心/σ 的软约束参数，用于“高斯样式”初始化 + 稳定性约束
        self.register_buffer('mask', self._build_mask(T, self.delays))  # [len(delays), T] 的稀疏移位掩码
        self.reset_parameters(init)

    def _build_mask(self, T, delays):
        # 在时间维做“延迟移位”的 one-hot 卷积核集合（按延迟把1放在对应位置）
        m = torch.zeros(len(delays), T)
        for i, d in enumerate(delays):
            if d < T: m[i, -1-d] = 1.0   # 对齐“右端为当前时刻”
        return m  # 不参与学习

    def reset_parameters(self, init):
        nn.init.normal_(self.weight, mean=0, std=0.02)
        if init == 'gaussian_like':
            # 可按 SOTA 的 σ–Ks 启发，初始化靠近“温和扩散”的形态（稳定起步）  
            with torch.no_grad():
                self.weight.add_(0.01)

    @torch.no_grad()
    def _stability_projection(self):
        # 简单稳定性约束：组内 L1 归一 & 权重幅度裁剪，等价于控制核能量与最大延迟贡献
        w = self.weight.clamp_(-1.0, 1.0)
        w.abs_().div_(w.abs().sum(dim=1, keepdim=True) + 1e-6)  # L1 归一（可改为 softmax）
        self.weight.copy_(torch.sign(self.weight) * w)

    def forward(self, x):
        # x: [B, T, D]  → reshape 为 [B*G, T, D/G] 逐组共享
        B, T, D = x.shape
        gC = D // self.G
        xg = x.view(B, T, self.G, gC).permute(0,2,1,3)  # [B, G, T, gC]
        # 组共享稀疏核：对每组，把 delays 的权重与 mask 做加权叠加，实现“稀疏时间卷积”
        # 等价实现：用 F.conv1d 的 depthwise，提前把稀疏核拼成 [G,1,T]，也行
        # 这里给出向量化思路：y_t = sum_d w_gd * x_{t-d}
        M = self.mask.to(xg)                             # [K,T]
        W = self.weight.softmax(dim=-1).unsqueeze(-1)    # [G,K,1]
        # 收集延迟后的信号
        xs = []
        for i, d in enumerate(self.delays):
            shifted = F.pad(xg, (0,0, d,0))[:, :, :T, :]  # 左 pad d，截到 T
            xs.append(shifted.unsqueeze(2))               # [B,G,1,T,gC]
        Xstk = torch.cat(xs, dim=2)                      # [B,G,K,T,gC]
        y = (W.unsqueeze(0).unsqueeze(-1) * Xstk).sum(dim=2)  # [B,G,T,gC]
        y = y.permute(0,2,1,3).contiguous().view(B, T, D)     # [B,T,D]
        return y

class TemporalSeparableReadout(nn.Module):
    """
    Depthwise(时间) + Pointwise(1×1) + 池化 + FC
    输入 [B, T, D] → 输出 [B, C]
    """
    def __init__(self, D, C, k=5):
        super().__init__()
        self.dw = nn.Conv1d(D, D, kernel_size=k, padding=k//2, groups=D, bias=False)
        self.pw = nn.Conv1d(D, D, kernel_size=1, bias=False)
        self.fc = nn.Linear(D, C)

    def forward(self, spikes):
        # spikes: [B, T, D]
        x = spikes.transpose(1, 2)     # [B, D, T]
        x = self.pw(self.dw(x))        # [B, D, T]
        x = x.mean(dim=-1)             # [B, D]
        return self.fc(x)
```

### B) 在 `main.py` 的最小接入（关键片段）

```python
from spikenet.temporal import GroupSparseDelay1D, TemporalSeparableReadout

class SpikeNet(nn.Module):
    def __init__(..., hids=[128,10], sizes=[5,2], ..., concat=False, act='LIF',
                 use_gs_delay=True, delay_groups=8, delay_set=(1,3,5),
                 use_learnable_p=True, use_tsr=True):
        super().__init__()
        # ... 保留原聚合器与 SNN ...
        self.sizes = sizes
        self.use_gs_delay = use_gs_delay
        self.use_tsr = use_tsr
        self.use_learnable_p = use_learnable_p

        # 可学习 p 门（每层一个标量，也可做“含时刻上下文”的小门）
        if self.use_learnable_p:
            self.p_gate = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in sizes])

        # 计算最后一层通道数（concat 决定）
        last_dim = (hids[-1]*2 if concat else hids[-1])
        T = len(data)  # 动态图时间步
        if self.use_gs_delay:
            self.delay = GroupSparseDelay1D(D=last_dim, T=T, groups=delay_groups, delays=delay_set)

        if self.use_tsr:
            self.readout = TemporalSeparableReadout(D=last_dim, C=out_features, k=5)
        else:
            self.pooling = nn.Linear(T * last_dim, out_features)

    def encode(self, nodes):
        spikes = []
        for t in range(len(data)):
            # …… 原采样逻辑 ……
            # 替换 p：
            p_t = torch.sigmoid(self.p_gate[0]).item() if self.use_learnable_p else self.p
            # 对于多层 sizes，可在每层里单独用 gate；这里演示第一层
            # size1 = max(int(size * p_t), 1); size2 = size - size1
            # …… 原图/演化图邻采样 & SAGE & SNN ……
            spikes.append(out)  # [N, D]，最后一层脉冲
        spikes = torch.stack(spikes, dim=1)  # [N, T, D]
        if self.use_gs_delay:
            spikes = self.delay(spikes)      # [N, T, D]
        neuron.reset_net(self)
        return spikes

    def forward(self, nodes):
        spikes = self.encode(nodes)            # [B, T, D]
        if self.use_tsr:
            return self.readout(spikes)
        else:
            return self.pooling(spikes.flatten(1))
```

### C) 训练处的轻量正则（`main.py`）

```python
# loss_fn = nn.CrossEntropyLoss()
lambda_spike = 1e-4   # 放电率正则
lambda_temp  = 1e-4   # 时间一致性（或 L2 平滑）

def train():
    model.train()
    for nodes in train_loader:
        optimizer.zero_grad()
        logits = model(nodes)                 # 前向里已做 delay/tsr
        ce = loss_fn(logits, y[nodes])

        # 取 encode 的中间产物可通过返回“附加信息”或保存到 model 缓存；示例假设 model 缓存了最近一次 spikes
        spikes = model._last_spikes_for_loss  # [B,T,D]（可按需存）
        reg_spike = spikes.mean()             # 控制平均放电率
        reg_temp  = (spikes[:,1:]-spikes[:,:-1]).pow(2).mean()  # 时间一致性（L2）

        loss = ce + lambda_spike*reg_spike + lambda_temp*reg_temp
        loss.backward()
        optimizer.step()
```

> 说明：若不想改动 `forward` 的返回结构，可在 `encode` 里把 `spikes` 缓存到 `self._last_spikes_for_loss`。

### D) `neuron.py` 的微改（可选）

- 把 `gamma, thresh_decay` 变 `nn.Parameter` 并通过 `torch.sigmoid` 限制在 (0,1)，默认初始化回到原值；
- 训练循环加一个**线性 warm-up** 更新 `alpha`（在 `args.epochs` 前 30% 线性增大到目标值），这是**常见 SNN 训练技巧**，与 SOTA 的稳定性分析一致方向。

---

# 复杂度与资源（定性/可在论文中定量化）

- **参数量**：
  - GS-Delay：从 O(D·Ks) → O(G·|Δ|)。典型 `D=128, Ks=9, G=8, |Δ|=3`，参数削减约一个数量级；
  - TSR：读出层从 `T·D·C` → `D·(k + 1 + C)`，当 `T` 很大（如 Tmall）节省显著。
- **FLOPs/时间**：
  - Depthwise 1D 卷积与稀疏延迟移位开销很小；
  - L2S-Gate 只涉及标量门控，不引入额外大算子；
  - 正则项与课程学习开销可忽略。
- **内存**：
  - 读出层参数显著下降；
  - 不改变 batch 的节点采样路径与张量规模。

---

# 实验计划（最小可验证集 → 可写成主表 + 消融 + 效率表）

**数据**：DBLP/Tmall/Patent（与原设定一致，含 `merge step`）。**主表**：

- Baseline：原 SpikeNet；
- +L2S；+TSR；+GS-Delay；三者组合（Ours）；
- 复现/引用 SOTA（可按文献报告值或你们复现值，强调我们**更高效**）。**指标**：Macro/Micro-F1、单 epoch 时间、显存峰值、参数/FLOPs。**消融**：
- 组数 G、延迟集合 Δ 的大小；
- 是否加时间一致性/放电率正则；
- 用 TSR vs. 原 FC 读出。**可视化**：
- 学到的 `p_{l,t}` 热力图（门控如何在突发时间片提高对“当前图”的权重）；
- 不同数据集上**延迟权重分布**（反映不同领域的时延差异；呼应 SOTA 的可解释性叙述）。

---

# 论文写作结构与创新性亮点

1. **问题与挑战**：动态图中“延迟效应 + 历史遗忘 + 大 T 下读出臃肿”。
2. **方法**：
   - **GS-Delay**：以**组共享稀疏核**近似高斯延迟，**理论上**给出简单稳定性充分条件（L1 归一＋最大延迟约束 ⇒ 梯度有界），与 SOTA 的 σ–Ks 条件同向，**但更高效**；
   - **L2S-Gate**：时序-结构采样配比的**可学习门控**，零额外算子，适配突发变化；
   - **TSR-Readout**：**可分离时间卷积读出**，在 T 大时**降参显著**；
   - **SCAT/Consistency**：稳态训练与鲁棒性增强（两行损失即可）。
3. **复杂度分析**：参数与 FLOPs 对比公式 + 表；
4. **实验**：三数据集 + 大 T 情况占优；
5. **可解释性**：延迟权重、门控 `p_{l,t}` 随时间/层的变化。
6. **结论**：**等效或更优精度 + 更高效率**，能在大规模动态图上稳定工作。

---

## 你可以让 AI 直接做的开发任务清单

- 新增 `spikenet/temporal.py`，按上述骨架补齐（含 mask 构造的更高效实现，如一次性组装 depthwise kernel 用 `Conv1d`）。
- 在 `main.py` 中：
  - 替换 `pooling` → `TemporalSeparableReadout`；
  - `encode` 中插入 `GroupSparseDelay1D`；
  - 将 `self.p` 改为 `p_{l,t}`（实现门控与时间统计的收集）；
  - 训练里加入两项正则与 `alpha` 课程（warm-up）。
- 修复 C++ 采样 `replace` 分支。
- 提供 `--use_gs_delay/--use_tsr/--use_learnable_p` 的命令行开关；默认全开。
- 写三个脚本：`run_dblp.sh / run_tmall.sh / run_patent.sh`（包含 ours 与各消融）。
- 导出报告表格（F1、时间、显存、参数/FLOPs），生成论文表格占位符。

---

## 与 SOTA 的关系与差异化声明（写在论文 related/method 末尾）

- 我们**承认延迟建模的重要性**，并参考其稳定性分析；不同点是我们用**组共享稀疏核**与**可分离读出**达到**近似甚至更好的效果**，同时显著**降低参数与计算**；
- 另外，我们把动态图中的“**历史 vs. 当前**”比例作为**可学习门控**，是对延迟思想的**结构域补强**，与 SOTA 互补。

---

如果你要更细的**补丁式代码改动**（直接对你贴出的文件逐行定位修改），我也可以按上述骨架把每个改动的具体行号与替换块列出来；你把这份方案丢给代码生成 AI，就能非常顺畅地落地实现与跑通实验。
