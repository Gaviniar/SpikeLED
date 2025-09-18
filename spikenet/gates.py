# File: spikenet/gates.py
import torch
import torch.nn as nn

class L2SGate(nn.Module):
    """
    Learnable Local 2-source Sampling Gate（按层的极轻量门控），不对采样操作反传。
    输入: stats_t ∈ R^F（如: [新增边占比, 平均度]），输出: p_{l,t} ∈ (0,1) 每层一个。
    """
    def __init__(self, num_layers: int, in_features: int = 2, base_p: float = 0.5):
        super().__init__()
        self.num_layers = num_layers
        self.base_p = base_p
        # 每层一组 (w, b)
        self.w = nn.Parameter(torch.zeros(num_layers, in_features))
        self.b = nn.Parameter(torch.zeros(num_layers))

    @torch.no_grad()
    def forward(self, stats_t: torch.Tensor) -> torch.Tensor:
        # stats_t: [F]（建议放 CPU/GPU 均可，这里不求梯度）
        z = stats_t.float().view(-1)  # [F]
        logits = torch.mv(self.w, z) + self.b  # [L]
        p = torch.sigmoid(logits)              # (0,1)
        # 与 base_p 做一个温和融合，避免训练前期剧烈波动
        p = 0.5 * p + 0.5 * torch.as_tensor(self.base_p).to(p)
        return p  # [L]
