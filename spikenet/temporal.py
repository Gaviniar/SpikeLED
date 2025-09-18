import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupSparseDelay1D(nn.Module):
    """
    组共享 + 稀疏离散延迟核（仅沿时间维做 Depthwise 1D）
    输入: x [B, T, D] ；输出: y [B, T, D]
    """
    def __init__(self, D, T, groups=8, delays=(1, 3, 5), init='gaussian_like'):
        super().__init__()
        
        # 自适应调整 groups 数量，确保能被 D 整除
        while D % groups != 0 and groups > 1:
            groups -= 1
        groups = max(groups, 1)  # 至少为1
        
        self.D, self.T, self.G = D, T, groups
        self.delays = list(delays)
        
        print(f"GroupSparseDelay1D: D={D}, T={T}, groups={groups} (adjusted)")
        
        # 每组对每个 delay 一个权重（标量），共享到组内所有通道
        self.weight = nn.Parameter(torch.zeros(self.G, len(self.delays)))
        self.reset_parameters(init)

    def reset_parameters(self, init):
        nn.init.normal_(self.weight, mean=0, std=0.02)
        if init == 'gaussian_like':
            # 按高斯样式初始化，靠近温和扩散
            with torch.no_grad():
                max_delay = max(self.delays)
                for i, delay in enumerate(self.delays):
                    # 基于延迟距离的衰减权重 - 修复：确保计算结果是标量
                    decay_value = (-0.5 * (delay / max_delay) ** 2)
                    decay = float(torch.exp(torch.tensor(decay_value)).item())
                    self.weight[:, i].fill_(decay * 0.1)


    @torch.no_grad()
    def _stability_projection(self):
        """简单稳定性约束：L1 归一 & 权重幅度裁剪"""
        w = self.weight.clamp_(-1.0, 1.0)
        w_abs = w.abs()
        w_norm = w_abs.sum(dim=1, keepdim=True) + 1e-6
        w = w.sign() * (w_abs / w_norm)
        self.weight.copy_(w)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        
        # 如果 groups=1，直接处理整个 D 维度
        if self.G == 1:
            return self._forward_single_group(x)
        
        gC = D // self.G
        
        # Reshape to group dimension: [B, G, T, gC]
        xg = x.view(B, T, self.G, gC).permute(0, 2, 1, 3)
        
        # 应用稳定性投影
        if self.training:
            self._stability_projection()
        
        # Softmax归一化权重
        W = self.weight.softmax(dim=-1)  # [G, K]
        
        # 计算延迟后的加权和
        output = torch.zeros_like(xg)  # [B, G, T, gC]
        
        for i, delay in enumerate(self.delays):
            if delay == 0:
                delayed_x = xg
            else:
                # 向右延迟 delay 步
                delayed_x = torch.zeros_like(xg)
                if delay < T:
                    delayed_x[:, :, delay:, :] = xg[:, :, :-delay, :]
            
            # 按权重累加
            output += W[:, i:i+1].unsqueeze(0).unsqueeze(-1) * delayed_x
        
        # Reshape back: [B, T, D]
        output = output.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return output
    
    def _forward_single_group(self, x):
        """当只有一个组时的简化处理"""
        B, T, D = x.shape
        
        # 应用稳定性投影
        if self.training:
            self._stability_projection()
        
        # Softmax归一化权重
        W = self.weight.softmax(dim=-1).squeeze(0)  # [K]
        
        # 计算延迟后的加权和
        output = torch.zeros_like(x)  # [B, T, D]
        
        for i, delay in enumerate(self.delays):
            if delay == 0:
                delayed_x = x
            else:
                # 向右延迟 delay 步
                delayed_x = torch.zeros_like(x)
                if delay < T:
                    delayed_x[:, delay:, :] = x[:, :-delay, :]
            
            # 按权重累加
            output += W[i] * delayed_x
        
        return output



class TemporalSeparableReadout(nn.Module):
    """
    Depthwise(时间) + Pointwise(1×1) + 池化 + FC
    输入 [B, T, D] → 输出 [B, C]
    """
    def __init__(self, D, C, k=5):
        super().__init__()
        self.dw = nn.Conv1d(D, D, kernel_size=k, padding=k//2, groups=D, bias=False)
        self.pw = nn.Conv1d(D, D, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(D)
        self.fc = nn.Linear(D, C)
        self.dropout = nn.Dropout(0.1)

    def forward(self, spikes):
        # spikes: [B, T, D]
        x = spikes.transpose(1, 2)     # [B, D, T]
        x = self.dw(x)                 # [B, D, T] depthwise temporal conv
        x = self.pw(x)                 # [B, D, T] pointwise conv
        x = x.mean(dim=-1)             # [B, D] temporal pooling
        x = self.norm(x)               # normalization
        x = self.dropout(x)            # dropout
        return self.fc(x)              # [B, C]


def compute_temporal_stats(adj_list, t):
    """计算时间片统计信息用于门控"""
    if t == 0:
        return torch.tensor([0.5, 1.0, 0.1])  # default stats
    
    current_edges = adj_list[t].nnz
    prev_edges = adj_list[t-1].nnz if t > 0 else 1
    
    # 新增边比例
    edge_growth = min(current_edges / (prev_edges + 1e-6), 2.0)
    # 平均度（归一化）
    avg_degree = current_edges / (adj_list[t].shape[0] + 1e-6)
    avg_degree = min(avg_degree / 10.0, 1.0)  # normalize to [0,1]
    # 简单的活跃度指标
    activity = min(current_edges / 1000.0, 1.0)
    
    return torch.tensor([edge_growth, avg_degree, activity])
