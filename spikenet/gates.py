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
        self.w = nn.Parameter(torch.zeros(num_layers, in_features))
        self.b = nn.Parameter(torch.zeros(num_layers))

    @torch.no_grad()
    def forward(self, stats_t: torch.Tensor) -> torch.Tensor:
        # ⚠️ 关键修复：把输入统计量搬到参数所在设备/精度
        z = stats_t.detach().to(self.w.device, dtype=self.w.dtype).view(-1)  # [F]
        logits = torch.mv(self.w, z) + self.b                               # [L]
        p = torch.sigmoid(logits)                                           # (0,1)
        base = torch.as_tensor(self.base_p, dtype=p.dtype, device=p.device)
        p = 0.5 * p + 0.5 * base                                            # 温和融合
        return p  # [L]
