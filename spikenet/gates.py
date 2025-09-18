import torch
import torch.nn as nn


class L2SGate(nn.Module):
    """
    可学习时序-结构采样门（L2S-Gate）
    根据当前时间片统计自动分配历史vs当前的采样比例
    """
    def __init__(self, input_dim=3, hidden_dim=8):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, temporal_stats):
        """
        Args:
            temporal_stats: [3] tensor with [edge_growth, avg_degree, activity]
        Returns:
            p_t: scalar in [0,1] for sampling allocation
        """
        return self.gate_net(temporal_stats)


class AdaptiveThreshold(nn.Module):
    """自适应阈值参数（SCAT的一部分）"""
    def __init__(self, init_gamma=0.2, init_thresh_decay=0.7):
        super().__init__()
        # 用sigmoid约束在(0,1)
        self.gamma_logit = nn.Parameter(torch.logit(torch.tensor(init_gamma)))
        self.thresh_decay_logit = nn.Parameter(torch.logit(torch.tensor(init_thresh_decay)))
    
    @property
    def gamma(self):
        return torch.sigmoid(self.gamma_logit)
    
    @property    
    def thresh_decay(self):
        return torch.sigmoid(self.thresh_decay_logit)
