import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupSparseDelay1D(nn.Module):
    """
    组共享 + 稀疏离散延迟核（仅沿时间维做 depthwise 1D 卷积的等价移位加权）
    输入: x [B, T, D] ；输出: y [B, T, D]
    复杂度：O(G * |delays| * B * T * (D/G))，D>>G 且 |delays|<<T
    """
    def __init__(self, D: int, T: int, groups: int = 8, delays=(1, 3, 5)):
        super().__init__()
        assert D % groups == 0, "D must be divisible by groups"
        self.D, self.T, self.G = D, T, groups
        self.delays = list(sorted(set(int(d) for d in delays if d >= 1)))
        g = groups
        k = len(self.delays)
        # 每组对每个 delay 一个权重（共享到组内所有通道）
        self.weight = nn.Parameter(torch.zeros(g, k))
        self.reset_parameters()

    def reset_parameters(self):
        # 小幅正值初始化，利于稳定起步
        nn.init.normal_(self.weight, mean=0.01, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        assert T == self.T and D == self.D
        gC = D // self.G
        xg = x.view(B, T, self.G, gC).permute(0, 2, 1, 3)  # [B, G, T, gC]

        # 组共享稀疏核：y_t = sum_d softmax(w_g)[d] * x_{t-d}
        W = self.weight.softmax(dim=-1)  # [G, K]
        xs = []
        for d in self.delays:
            # 左 pad d，再截断到长度 T，实现 x_{t-d}
            shifted = F.pad(xg, (0, 0, d, 0))[:, :, :T, :]  # [B,G,T,gC]
            xs.append(shifted.unsqueeze(2))  # [B,G,1,T,gC]
        Xstk = torch.cat(xs, dim=2)  # [B,G,K,T,gC]
        y = (W.view(1, self.G, -1, 1, 1) * Xstk).sum(dim=2)  # [B,G,T,gC]
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, D)  # [B,T,D]
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