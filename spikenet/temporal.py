import torch
import torch.nn as nn
import torch.nn.functional as F

# --- replace this class in spikenet/temporal.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupSparseDelay1D(nn.Module):
    """
    组共享 + 稀疏离散延迟核（支持任意 D、groups，不要求整除）
    输入: x [B, T, D] ；输出: y [B, T, D]
    """
    def __init__(self, D: int, T: int, groups: int = 8, delays=(1, 3, 5)):
        super().__init__()
        self.D, self.T = int(D), int(T)
        self.G = max(1, int(groups))
        # 延迟集合
        self.delays = [int(d) for d in delays if int(d) >= 1]
        if len(self.delays) == 0:
            raise ValueError("`delays` must contain at least one positive integer.")

        # 每组一套 delay 权重（共享到被分配到该组的所有通道）
        self.weight = nn.Parameter(torch.zeros(self.G, len(self.delays)))

        # 通道 -> 组 的映射（不要求均匀整除，采用 round-robin 分配）
        ch2g = torch.arange(self.D, dtype=torch.long) % self.G
        self.register_buffer("ch2g", ch2g)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.01, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        if T != self.T or D != self.D:
            raise ValueError(f"Input has shape [B,{T},{D}], but module was built with T={self.T}, D={self.D}")

        # 计算每通道的延迟权重：Wg[G,K] -> Wc[D,K]
        Wg = self.weight.softmax(dim=-1)        # [G, K]
        Wc = Wg.index_select(0, self.ch2g)      # [D, K]

        # 构造按时间维的移位堆栈: xs -> [B, T, D, K]
        xs = []
        for d in self.delays:
            # 在时间维(=1)左侧 pad d，实现 x_{t-d}
            shifted = F.pad(x, (0, 0, d, 0))[:, :T, :]   # [B, T, D]
            xs.append(shifted.unsqueeze(-1))             # [B, T, D, 1]
        Xstk = torch.cat(xs, dim=-1)                     # [B, T, D, K]

        # 通道共享权重按组广播并聚合
        y = (Xstk * Wc.view(1, 1, D, -1)).sum(dim=-1)    # [B, T, D]
        return y



class TemporalSeparableReadout(nn.Module):
    """
    可分离时序读出：Depthwise(时间) + Pointwise(1×1) + 全局均值池化 + FC
    输入 [B, T, D] → 输出 [B, C]
    将原 Linear(T*D, C) 的参数/计算解耦到与 T 基本无关。
    """
    def __init__(self, D: int, C: int, k: int = 5):
        super().__init__()
        pad = k // 2
        # 用 [B, D, T] 的通道为 D 的 depthwise 卷积
        self.dw = nn.Conv1d(D, D, kernel_size=k, padding=pad, groups=D, bias=False)
        self.pw = nn.Conv1d(D, D, kernel_size=1, bias=False)
        self.fc = nn.Linear(D, C)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        # spikes: [B, T, D]
        x = spikes.transpose(1, 2)        # [B, D, T]
        x = self.pw(self.dw(x))           # [B, D, T]
        x = x.mean(dim=-1)                # [B, D]
        return self.fc(x)