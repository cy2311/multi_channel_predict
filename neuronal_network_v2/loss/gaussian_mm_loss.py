"""GaussianMMLoss实现

高斯混合模型损失函数，将模型输出解释为高斯混合模型的参数，
计算负对数似然作为损失。适用于SigmaMUNet的10通道输出。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple, Union


class GaussianMMLoss(nn.Module):
    """高斯混合模型损失函数
    
    核心思想：
    - 将模型输出解释为高斯混合模型的参数
    - 计算负对数似然作为损失
    
    输出格式化：
    - p: 检测概率 (通道0)
    - pxyz_mu: 参数均值 (通道1-4)
    - pxyz_sig: 参数标准差 (通道5-8)
    - bg: 背景 (通道9)
    
    Args:
        eps: 小常数，防止数值不稳定
        bg_weight: 背景损失权重
        gmm_weight: GMM损失权重
        min_sigma: 最小标准差值
        max_sigma: 最大标准差值
        use_log_sigma: 是否使用log参数化标准差
    """
    
    def __init__(self,
                 eps: float = 1e-6,
                 bg_weight: float = 1.0,
                 gmm_weight: float = 1.0,
                 min_sigma: float = 0.01,
                 max_sigma: float = 10.0,
                 use_log_sigma: bool = True):
        super().__init__()
        
        self.eps = eps
        self.bg_weight = bg_weight
        self.gmm_weight = gmm_weight
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.use_log_sigma = use_log_sigma
        
        # 背景损失函数
        self.bg_loss_fn = nn.MSELoss(reduction='none')
    
    def forward(self,
                output: torch.Tensor,
                target: torch.Tensor,
                emitter_positions: Optional[torch.Tensor] = None,
                emitter_intensities: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播
        
        Args:
            output: 模型输出，形状为 (B, 10, H, W)
            target: 目标张量，形状为 (B, 10, H, W)
            emitter_positions: 发射器位置，形状为 (B, N, 3) [x, y, z]
            emitter_intensities: 发射器强度，形状为 (B, N)
            
        Returns:
            损失值或损失字典
        """
        if output.size(1) != 10:
            raise ValueError("GaussianMMLoss expects 10-channel input")
        
        # 解析输出
        p, pxyz_mu, pxyz_sig, bg = self._parse_output(output)
        
        # 解析目标
        if target.size(1) == 10:
            target_p, target_pxyz_mu, target_pxyz_sig, target_bg = self._parse_output(target)
        else:
            # 如果目标不是10通道，使用发射器信息构建目标
            target_p, target_pxyz_mu, target_pxyz_sig, target_bg = self._build_target_from_emitters(
                emitter_positions, emitter_intensities, output.shape
            )
        
        # 计算GMM损失
        gmm_loss = self._compute_gmm_loss(p, pxyz_mu, pxyz_sig, target_p, target_pxyz_mu)
        
        # 计算背景损失
        bg_loss = self.bg_loss_fn(bg, target_bg)
        
        # 总损失
        total_loss = self.gmm_weight * gmm_loss + self.bg_weight * bg_loss
        
        return {
            'total_loss': total_loss,
            'gmm_loss': gmm_loss,
            'bg_loss': bg_loss,
            'mean_total_loss': total_loss.mean(),
            'mean_gmm_loss': gmm_loss.mean(),
            'mean_bg_loss': bg_loss.mean()
        }
    
    def _parse_output(self, output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """解析10通道输出"""
        p = output[:, 0:1]  # 检测概率
        pxyz_mu = output[:, 1:5]  # 参数均值 (photon, x, y, z)
        pxyz_sig_raw = output[:, 5:9]  # 参数标准差原始值
        bg = output[:, 9:10]  # 背景
        
        # 处理标准差参数化
        if self.use_log_sigma:
            # log参数化：sigma = exp(log_sigma)
            pxyz_sig = torch.exp(pxyz_sig_raw)
        else:
            # 直接参数化：sigma = softplus(raw) + min_sigma
            pxyz_sig = F.softplus(pxyz_sig_raw) + self.min_sigma
        
        # 限制标准差范围
        pxyz_sig = torch.clamp(pxyz_sig, self.min_sigma, self.max_sigma)
        
        return p, pxyz_mu, pxyz_sig, bg
    
    def _build_target_from_emitters(self,
                                   positions: torch.Tensor,
                                   intensities: torch.Tensor,
                                   output_shape: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """从发射器信息构建目标张量"""
        B, _, H, W = output_shape
        device = positions.device
        
        # 初始化目标张量
        target_p = torch.zeros(B, 1, H, W, device=device)
        target_pxyz_mu = torch.zeros(B, 4, H, W, device=device)
        target_pxyz_sig = torch.ones(B, 4, H, W, device=device) * 0.1  # 默认小标准差
        target_bg = torch.zeros(B, 1, H, W, device=device)  # 假设背景为0
        
        for b in range(B):
            if positions[b].numel() > 0:
                pos = positions[b]  # (N, 3)
                intens = intensities[b]  # (N,)
                
                # 将连续坐标转换为像素坐标
                x_coords = torch.clamp(pos[:, 0].long(), 0, W-1)
                y_coords = torch.clamp(pos[:, 1].long(), 0, H-1)
                
                # 设置检测概率
                target_p[b, 0, y_coords, x_coords] = 1.0
                
                # 设置参数均值
                target_pxyz_mu[b, 0, y_coords, x_coords] = intens  # 光子数
                target_pxyz_mu[b, 1, y_coords, x_coords] = pos[:, 0] - x_coords.float()  # x偏移
                target_pxyz_mu[b, 2, y_coords, x_coords] = pos[:, 1] - y_coords.float()  # y偏移
                if pos.size(1) > 2:
                    target_pxyz_mu[b, 3, y_coords, x_coords] = pos[:, 2]  # z坐标
        
        return target_p, target_pxyz_mu, target_pxyz_sig, target_bg
    
    def _compute_gmm_loss(self,
                         p: torch.Tensor,
                         pxyz_mu: torch.Tensor,
                         pxyz_sig: torch.Tensor,
                         target_p: torch.Tensor,
                         target_pxyz_mu: torch.Tensor) -> torch.Tensor:
        """计算高斯混合模型损失"""
        # 只在有发射器的位置计算损失
        fg_mask = (target_p > 0.5).float()  # (B, 1, H, W)
        
        # 检测概率损失
        p_sigmoid = torch.sigmoid(p)
        detection_loss = F.binary_cross_entropy(p_sigmoid, target_p, reduction='none')
        
        # 回归损失（只在前景位置）
        regression_loss = torch.zeros_like(detection_loss)
        
        if fg_mask.sum() > 0:
            # 计算高斯负对数似然
            diff = pxyz_mu - target_pxyz_mu  # (B, 4, H, W)
            
            # 负对数似然：0.5 * ((x-mu)/sigma)^2 + log(sigma) + 0.5*log(2*pi)
            normalized_diff = diff / (pxyz_sig + self.eps)  # (B, 4, H, W)
            nll = 0.5 * normalized_diff ** 2 + torch.log(pxyz_sig + self.eps) + 0.5 * math.log(2 * math.pi)
            
            # 对4个参数求和
            nll = nll.sum(dim=1, keepdim=True)  # (B, 1, H, W)
            
            # 只在前景位置应用
            regression_loss = nll * fg_mask
        
        # 总损失
        total_loss = detection_loss + regression_loss
        
        return total_loss
    
    def compute_likelihood(self,
                          output: torch.Tensor,
                          emitter_positions: torch.Tensor,
                          emitter_intensities: torch.Tensor) -> torch.Tensor:
        """计算给定发射器的似然"""
        p, pxyz_mu, pxyz_sig, bg = self._parse_output(output)
        
        B, _, H, W = output.shape
        total_likelihood = torch.zeros(B, device=output.device)
        
        for b in range(B):
            if emitter_positions[b].numel() > 0:
                pos = emitter_positions[b]  # (N, 3)
                intens = emitter_intensities[b]  # (N,)
                
                # 获取对应位置的预测
                x_coords = torch.clamp(pos[:, 0].long(), 0, W-1)
                y_coords = torch.clamp(pos[:, 1].long(), 0, H-1)
                
                pred_p = torch.sigmoid(p[b, 0, y_coords, x_coords])  # (N,)
                pred_mu = pxyz_mu[b, :, y_coords, x_coords].t()  # (N, 4)
                pred_sig = pxyz_sig[b, :, y_coords, x_coords].t()  # (N, 4)
                
                # 构建目标
                target_params = torch.zeros_like(pred_mu)
                target_params[:, 0] = intens  # 光子数
                target_params[:, 1] = pos[:, 0] - x_coords.float()  # x偏移
                target_params[:, 2] = pos[:, 1] - y_coords.float()  # y偏移
                if pos.size(1) > 2:
                    target_params[:, 3] = pos[:, 2]  # z坐标
                
                # 计算高斯似然
                diff = pred_mu - target_params
                normalized_diff = diff / (pred_sig + self.eps)
                log_likelihood = -0.5 * (normalized_diff ** 2).sum(dim=1) - \
                               torch.log(pred_sig + self.eps).sum(dim=1) - \
                               2 * math.log(2 * math.pi)
                
                # 加上检测概率
                log_likelihood += torch.log(pred_p + self.eps)
                
                total_likelihood[b] = log_likelihood.sum()
        
        return total_likelihood


class AdaptiveGaussianMMLoss(GaussianMMLoss):
    """自适应高斯混合模型损失
    
    在标准GaussianMMLoss基础上添加：
    - 动态权重调整
    - 温度缩放
    - 不确定性感知
    """
    
    def __init__(self,
                 *args,
                 temperature: float = 1.0,
                 adaptive_temperature: bool = True,
                 uncertainty_weight: float = 0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.register_buffer('temperature', torch.tensor(temperature))
        self.adaptive_temperature = adaptive_temperature
        self.uncertainty_weight = uncertainty_weight
        
        # 温度历史
        self.register_buffer('loss_history', torch.zeros(100))
        self.history_idx = 0
    
    def forward(self,
                output: torch.Tensor,
                target: torch.Tensor,
                emitter_positions: Optional[torch.Tensor] = None,
                emitter_intensities: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """自适应前向传播"""
        # 应用温度缩放
        scaled_output = output.clone()
        scaled_output[:, 0] = scaled_output[:, 0] / self.temperature  # 只对概率应用温度
        
        # 计算基础损失
        loss_dict = super().forward(
            scaled_output, target, emitter_positions, emitter_intensities
        )
        
        # 计算不确定性损失
        uncertainty_loss = self._compute_uncertainty_loss(output)
        
        # 更新总损失
        total_loss = loss_dict['total_loss'] + self.uncertainty_weight * uncertainty_loss
        loss_dict['total_loss'] = total_loss
        loss_dict['uncertainty_loss'] = uncertainty_loss
        loss_dict['mean_uncertainty_loss'] = uncertainty_loss.mean()
        
        # 更新温度
        if self.adaptive_temperature:
            self._update_temperature(loss_dict['mean_total_loss'])
        
        return loss_dict
    
    def _compute_uncertainty_loss(self, output: torch.Tensor) -> torch.Tensor:
        """计算不确定性损失"""
        _, _, pxyz_sig, _ = self._parse_output(output)
        
        # 鼓励合理的不确定性：不要太大也不要太小
        target_sig = torch.ones_like(pxyz_sig) * 0.5  # 目标标准差
        uncertainty_loss = F.mse_loss(pxyz_sig, target_sig, reduction='none')
        
        return uncertainty_loss
    
    def _update_temperature(self, current_loss: torch.Tensor):
        """更新温度参数"""
        # 记录损失历史
        self.loss_history[self.history_idx] = current_loss.detach()
        self.history_idx = (self.history_idx + 1) % 100
        
        # 计算损失趋势
        if self.history_idx == 0:  # 历史已满
            recent_loss = self.loss_history[-10:].mean()
            old_loss = self.loss_history[:10].mean()
            
            # 如果损失在增加，降低温度（增加置信度）
            if recent_loss > old_loss:
                self.temperature = torch.clamp(self.temperature * 0.95, 0.1, 10.0)
            else:
                self.temperature = torch.clamp(self.temperature * 1.05, 0.1, 10.0)


class MultiModalGaussianMMLoss(nn.Module):
    """多模态高斯混合模型损失
    
    支持多种模态的混合损失计算
    """
    
    def __init__(self,
                 num_modes: int = 2,
                 mode_weights: Optional[torch.Tensor] = None,
                 **loss_kwargs):
        super().__init__()
        
        self.num_modes = num_modes
        
        if mode_weights is None:
            mode_weights = torch.ones(num_modes) / num_modes
        
        self.register_buffer('mode_weights', mode_weights)
        
        # 为每个模态创建损失函数
        self.mode_losses = nn.ModuleList([
            GaussianMMLoss(**loss_kwargs) for _ in range(num_modes)
        ])
    
    def forward(self,
                outputs: list,  # 每个模态的输出
                targets: list,  # 每个模态的目标
                **kwargs) -> Dict[str, torch.Tensor]:
        """多模态前向传播"""
        if len(outputs) != self.num_modes or len(targets) != self.num_modes:
            raise ValueError(f"Expected {self.num_modes} outputs and targets")
        
        total_loss = 0.0
        mode_losses = {}
        
        for i, (output, target, loss_fn) in enumerate(zip(outputs, targets, self.mode_losses)):
            mode_loss_dict = loss_fn(output, target, **kwargs)
            mode_loss = mode_loss_dict['mean_total_loss']
            
            total_loss += self.mode_weights[i] * mode_loss
            mode_losses[f'mode_{i}_loss'] = mode_loss
        
        return {
            'total_loss': total_loss,
            **mode_losses
        }