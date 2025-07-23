import math
import torch
import torch.nn as nn
import h5py
import numpy as np


class CountLoss(nn.Module):
    """
    Count Loss function based on Poisson Binomial Distribution approximation.
    
    This loss optimizes the log-likelihood of the true emitter count E under a Gaussian
    approximation of the Poisson Binomial distribution. When the number of emitters K is
    large enough, the distribution of predicted emitter count approaches a Gaussian with
    mean μ_count and variance σ²_count.
    
    The loss is defined as:
    L_count = -log P(E|μ_count,σ²_count) = 0.5 * (E-μ_count)²/σ²_count + log(√(2π)σ_count)
    
    where:
    - E is the ground truth total emitter count
    - μ_count is the sum of all pixel probabilities
    - σ²_count is the sum of p*(1-p) for all pixels
    """
    
    def __init__(self, eps: float = 1e-6):
        """
        Initialize the CountLoss.
        
        Args:
            eps: Small constant to prevent division by zero or log of zero
        """
        super().__init__()
        self.eps = eps
        
    def forward(self, prob_map: torch.Tensor, true_count: torch.Tensor) -> torch.Tensor:
        """
        Compute the count loss.
        
        Args:
            prob_map: Predicted probabilities, shape (B, 1, H, W) after sigmoid activation
            true_count: Tensor of shape (B,) with integer ground-truth emitter counts
            
        Returns:
            Scalar loss value (mean across batch)
        """
        B = prob_map.size(0)
        
        # 确保概率值在有效范围内，防止数值不稳定性
        prob_map = torch.clamp(prob_map, min=self.eps, max=1.0-self.eps)
        
        # 确保输入张量有梯度
        if not prob_map.requires_grad:
            prob_map.requires_grad_(True)
        
        # Flatten the probability map to (B, H*W)
        p = prob_map.view(B, -1)
        
        # Calculate mean (μ_count) - sum of all probabilities
        mu = p.sum(dim=1)  # (B,)
        
        # Calculate variance (σ²_count) - sum of p*(1-p)
        # 使用更大的eps值确保方差不会太小
        var = (p * (1.0 - p)).sum(dim=1) + max(self.eps, 1e-4)  # (B,)
        
        # 检查并处理可能的NaN或Inf值
        if torch.isnan(var).any() or torch.isinf(var).any():
            print("Warning: NaN or Inf detected in variance calculation")
            var = torch.where(torch.isnan(var) | torch.isinf(var), 
                             torch.ones_like(var) * max(self.eps, 1e-4), 
                             var)
        
        # 计算差值，并检查是否有异常值
        diff = true_count - mu
        if torch.isnan(diff).any() or torch.isinf(diff).any():
            print("Warning: NaN or Inf detected in count difference calculation")
            diff = torch.where(torch.isnan(diff) | torch.isinf(diff),
                              torch.zeros_like(diff),
                              diff)
        
        # Calculate the loss terms with numerical stability
        term1 = 0.5 * diff.pow(2) / var
        term2 = torch.log(torch.sqrt(var) * math.sqrt(2 * math.pi))
        
        # 检查并处理可能的NaN或Inf值
        if torch.isnan(term1).any() or torch.isinf(term1).any():
            print("Warning: NaN or Inf detected in term1 calculation")
            term1 = torch.where(torch.isnan(term1) | torch.isinf(term1),
                               torch.zeros_like(term1),
                               term1)
        
        if torch.isnan(term2).any() or torch.isinf(term2).any():
            print("Warning: NaN or Inf detected in term2 calculation")
            term2 = torch.where(torch.isnan(term2) | torch.isinf(term2),
                               torch.zeros_like(term2),
                               term2)
        
        # Total loss
        loss = term1 + term2
        
        # 确保损失值是有效的，否则返回一个有效的损失值
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("Warning: NaN or Inf detected in final loss calculation")
            # 创建一个新的有梯度的零张量作为替代损失
            valid_loss = torch.zeros(1, device=prob_map.device, requires_grad=True)
            return valid_loss
        
        return loss.mean()


def load_emitter_count_from_h5(h5_path):
    """
    Load the ground truth emitter count from an H5 file.
    
    Args:
        h5_path: Path to the H5 file containing emitter data
        
    Returns:
        Total number of emitters in the dataset
    """
    with h5py.File(h5_path, 'r') as f:
        # Check if 'emitters' group exists
        if 'emitters' in f:
            # Count the number of emitters in the dataset
            if 'xyz' in f['emitters']:
                # Count based on the first dimension of the xyz dataset
                return len(f['emitters']['xyz'])
            elif 'id' in f['emitters']:
                # Alternative: count based on the id dataset
                return len(f['emitters']['id'])
        
        # If structure is different, try to find any dataset that might contain emitters
        for key in f.keys():
            if isinstance(f[key], h5py.Group) and 'xyz' in f[key]:
                return len(f[key]['xyz'])
    
    raise ValueError(f"Could not find emitter data in {h5_path}")


def get_emitter_count_per_frame(h5_path):
    """
    Get the number of emitters active in each frame from an H5 file.
    
    Args:
        h5_path: Path to the H5 file containing emitter data
        
    Returns:
        Array with emitter counts per frame
    """
    with h5py.File(h5_path, 'r') as f:
        # 首先检查records组中是否有frame_ix
        if 'records' in f and 'frame_ix' in f['records']:
            frame_indices = f['records/frame_ix'][()]
            
            if len(frame_indices) == 0:
                return np.array([0])
            
            # 获取最大帧索引
            max_frame = int(np.max(frame_indices))
            
            # 计算每帧的发射体数量
            counts = np.zeros(max_frame + 1, dtype=int)
            unique, counts_per_frame = np.unique(frame_indices, return_counts=True)
            counts[unique.astype(int)] = counts_per_frame
            
            return counts
        # 如果没有records/frame_ix，检查emitters组
        elif 'emitters' in f and 't0' in f['emitters'] and 'on_time' in f['emitters']:
            # Get frame start times and durations
            t0 = f['emitters']['t0'][...]
            on_time = f['emitters']['on_time'][...]
            
            # Calculate end times
            t_end = t0 + on_time
            
            # Find max frame number
            max_frame = int(np.ceil(np.max(t_end)))
            
            # Count emitters per frame
            counts = np.zeros(max_frame, dtype=int)
            
            for i in range(len(t0)):
                start_frame = int(np.floor(t0[i]))
                end_frame = int(np.ceil(t_end[i]))
                
                for frame in range(start_frame, end_frame):
                    if frame < max_frame:
                        counts[frame] += 1
            
            return counts
        # 最后检查根目录下是否有frame_ix（向后兼容）
        elif 'frame_ix' in f:
            frame_indices = f['frame_ix'][()]
            
            if len(frame_indices) == 0:
                return np.array([0])
            
            # 获取最大帧索引
            max_frame = int(np.max(frame_indices))
            
            # 计算每帧的发射体数量
            counts = np.zeros(max_frame + 1, dtype=int)
            unique, counts_per_frame = np.unique(frame_indices, return_counts=True)
            counts[unique.astype(int)] = counts_per_frame
            
            return counts
    
    raise ValueError(f"Could not extract per-frame emitter counts from {h5_path}")