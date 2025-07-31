import math
import torch
import torch.nn as nn
import h5py
import numpy as np


class LocLoss(nn.Module):
    """
    Localization Loss function based on Gaussian Mixture Model.
    
    This loss optimizes the log-likelihood of the true emitter locations under a mixture
    of Gaussians model. Each pixel predicts a 4D Gaussian distribution (x, y, z, N) and
    the probability of an emitter being present.
    
    The loss is defined as:
    L_loc = -1/E * sum_{e=1}^E log(sum_{k=1}^K (p_k/sum_j p_j) * P(u_e^GT | μ_k, Σ_k))
    
    where:
    - E is the number of ground truth emitters
    - K is the number of pixels
    - p_k is the probability of an emitter at pixel k
    - u_e^GT is the ground truth location and intensity of emitter e
    - μ_k is the predicted mean (location and intensity) at pixel k
    - Σ_k is the predicted covariance matrix at pixel k
    """
    
    def __init__(self, eps: float = 1e-6):
        """
        Initialize the LocLoss.
        
        Args:
            eps: Small constant to prevent division by zero or log of zero
        """
        super().__init__()
        self.eps = eps
        
    def forward(self, 
               pred_detection: torch.Tensor,  # [B, 1, H, W] - detection probability
               pred_dx: torch.Tensor,         # [B, 1, H, W] - x offset
               pred_dy: torch.Tensor,         # [B, 1, H, W] - y offset
               pred_dz: torch.Tensor,         # [B, 1, H, W] - z offset
               pred_photons: torch.Tensor,    # [B, 1, H, W] - photon count
               pred_sigma_x: torch.Tensor,    # [B, 1, H, W] - x uncertainty
               pred_sigma_y: torch.Tensor,    # [B, 1, H, W] - y uncertainty
               pred_sigma_z: torch.Tensor,    # [B, 1, H, W] - z uncertainty
               pred_sigma_photons: torch.Tensor,  # [B, 1, H, W] - photon uncertainty
               gt_emitters: torch.Tensor,     # [B, E, 4] - ground truth emitters (x, y, z, N)
               pixel_centers_x: torch.Tensor,  # [B, 1, H, W] - x coordinates of pixel centers
               pixel_centers_y: torch.Tensor,  # [B, 1, H, W] - y coordinates of pixel centers
               pixel_centers_z: torch.Tensor   # [B, 1, H, W] - z coordinates of pixel centers
              ) -> torch.Tensor:
        """
        Compute the localization loss.
        
        Args:
            pred_detection: Predicted detection probabilities after sigmoid
            pred_dx, pred_dy, pred_dz: Predicted offsets from pixel centers
            pred_photons: Predicted photon counts
            pred_sigma_x, pred_sigma_y, pred_sigma_z, pred_sigma_photons: Predicted uncertainties
            gt_emitters: Ground truth emitter locations and intensities
            pixel_centers_x, pixel_centers_y, pixel_centers_z: Coordinates of pixel centers
            
        Returns:
            Scalar loss value (mean across batch)
        """
        B = pred_detection.size(0)
        device = pred_detection.device
        
        # 确保概率值在有效范围内，防止数值不稳定性
        pred_detection = torch.clamp(pred_detection, min=self.eps, max=1.0-self.eps)
        
        # 确保输入张量有梯度
        if not pred_detection.requires_grad:
            pred_detection.requires_grad_(True)
        
        # 将所有预测展平为 [B, K, 1] 或 [B, K] 形状，其中 K 是像素数
        B, _, H, W = pred_detection.shape
        K = H * W
        
        p_k = pred_detection.view(B, K)  # [B, K]
        
        # 计算归一化的检测概率权重
        p_sum = torch.sum(p_k, dim=1, keepdim=True) + self.eps  # [B, 1]
        w_k = p_k / p_sum  # [B, K]
        log_w_k = torch.log(w_k + self.eps)  # [B, K]
        
        # 构建每个像素的预测均值 μ_k
        mu_x = (pixel_centers_x + pred_dx).view(B, K)  # [B, K]
        mu_y = (pixel_centers_y + pred_dy).view(B, K)  # [B, K]
        mu_z = (pixel_centers_z + pred_dz).view(B, K)  # [B, K]
        mu_photons = pred_photons.view(B, K)  # [B, K]
        
        # 构建每个像素的预测方差（协方差矩阵对角线）
        sigma_x = torch.clamp(pred_sigma_x.view(B, K), min=self.eps)  # [B, K]
        sigma_y = torch.clamp(pred_sigma_y.view(B, K), min=self.eps)  # [B, K]
        sigma_z = torch.clamp(pred_sigma_z.view(B, K), min=self.eps)  # [B, K]
        sigma_photons = torch.clamp(pred_sigma_photons.view(B, K), min=self.eps)  # [B, K]
        
        # 计算方差的倒数，用于高斯概率计算
        sigma_x_inv_sq = 1.0 / (sigma_x ** 2 + self.eps)  # [B, K]
        sigma_y_inv_sq = 1.0 / (sigma_y ** 2 + self.eps)  # [B, K]
        sigma_z_inv_sq = 1.0 / (sigma_z ** 2 + self.eps)  # [B, K]
        sigma_photons_inv_sq = 1.0 / (sigma_photons ** 2 + self.eps)  # [B, K]
        
        # 初始化批次损失
        batch_loss = torch.zeros(B, device=device)
        
        # 对每个样本单独计算损失
        for b in range(B):
            # 获取当前样本的地面真实发射体
            gt_emitters_b = gt_emitters[b]  # [E, 4]
            E = gt_emitters_b.size(0)
            
            if E == 0:  # 如果没有发射体，跳过此样本
                continue
            
            # 初始化发射体损失
            emitter_losses = torch.zeros(E, device=device)
            
            # 对每个发射体计算损失
            for e in range(E):
                gt_x, gt_y, gt_z, gt_photons = gt_emitters_b[e]  # 获取地面真实值
                
                # 计算每个像素的马氏距离平方 (x-μ)^T Σ^-1 (x-μ)
                dx_sq = (gt_x - mu_x[b]) ** 2  # [K]
                dy_sq = (gt_y - mu_y[b]) ** 2  # [K]
                dz_sq = (gt_z - mu_z[b]) ** 2  # [K]
                dphotons_sq = (gt_photons - mu_photons[b]) ** 2  # [K]
                
                # 计算每个像素的四维高斯概率密度的对数
                log_prob_x = -0.5 * dx_sq * sigma_x_inv_sq[b] - 0.5 * torch.log(2 * math.pi * sigma_x[b] ** 2)  # [K]
                log_prob_y = -0.5 * dy_sq * sigma_y_inv_sq[b] - 0.5 * torch.log(2 * math.pi * sigma_y[b] ** 2)  # [K]
                log_prob_z = -0.5 * dz_sq * sigma_z_inv_sq[b] - 0.5 * torch.log(2 * math.pi * sigma_z[b] ** 2)  # [K]
                log_prob_photons = -0.5 * dphotons_sq * sigma_photons_inv_sq[b] - 0.5 * torch.log(2 * math.pi * sigma_photons[b] ** 2)  # [K]
                
                # 计算联合概率密度的对数（假设独立性）
                log_prob = log_prob_x + log_prob_y + log_prob_z + log_prob_photons  # [K]
                
                # 加入检测概率权重
                log_weighted_prob = log_w_k[b] + log_prob  # [K]
                
                # 使用log-sum-exp技巧计算对数和
                max_log_prob = torch.max(log_weighted_prob)
                emitter_loss = max_log_prob + torch.log(torch.sum(torch.exp(log_weighted_prob - max_log_prob)))
                emitter_losses[e] = emitter_loss
            
            # 计算当前样本的平均损失
            batch_loss[b] = -torch.mean(emitter_losses)
        
        # 返回批次平均损失
        return torch.mean(batch_loss)


def get_emitters_from_h5(h5_path: str, frame_idx: int = None) -> np.ndarray:
    """
    从H5文件中获取发射体信息
    
    Args:
        h5_path: H5文件路径
        frame_idx: 帧索引，如果为None则返回所有帧的发射体
        
    Returns:
        包含发射体信息的数组，形状为[E, 4]，每行为[x, y, z, photons]
    """
    with h5py.File(h5_path, 'r') as f:
        if 'xyz' in f and 'photons' in f and 'frame_ix' in f:
            xyz = f['xyz'][()]
            photons = f['photons'][()]
            frame_indices = f['frame_ix'][()]
            
            if frame_idx is not None:
                # 筛选特定帧的发射体
                mask = frame_indices == frame_idx
                xyz_filtered = xyz[mask]
                photons_filtered = photons[mask]
                
                # 组合为[E, 4]数组
                emitters = np.column_stack([xyz_filtered, photons_filtered])
            else:
                # 返回所有发射体
                emitters = np.column_stack([xyz, photons])
                
            return emitters
        else:
            raise ValueError(f"H5文件 {h5_path} 缺少必要的数据集")


def get_emitter_count_per_frame(h5_path: str) -> np.ndarray:
    """
    获取每帧的发射体计数
    
    Args:
        h5_path: H5文件路径
        
    Returns:
        每帧的发射体计数数组
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
            # 获取帧开始时间和持续时间
            t0 = f['emitters/t0'][...]
            on_time = f['emitters/on_time'][...]
            
            # 计算结束时间
            t_end = t0 + on_time
            
            # 找到最大帧号
            max_frame = int(np.ceil(np.max(t_end)))
            
            # 计算每帧的发射体数量
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
        else:
            raise ValueError(f"H5文件 {h5_path} 缺少必要的数据集，无法获取每帧发射体计数")