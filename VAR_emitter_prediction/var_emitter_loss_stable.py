import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math
import numpy as np


class StableVAREmitterLoss(nn.Module):
    """
    Stable multi-scale loss function for VAR-based emitter prediction
    Uses PyTorch built-in loss functions for numerical stability
    Follows DECODE's 6-channel unified architecture: [prob_logits, photons, x, y, z, background]
    """
    
    def __init__(self,
                 channel_weights: Optional[List[float]] = None,
                 pos_weight: float = 2.0,
                 scale_weights: Optional[List[float]] = None,
                 eps: float = 1e-4,  # Larger eps for stability
                 warmup_epochs: int = 20):
        super().__init__()
        
        self.eps = eps
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
        # Channel weights [prob, photons, x, y, z, background]
        if channel_weights is not None:
            ch_weights = torch.tensor(channel_weights, dtype=torch.float32)
        else:
            # Default balanced weights with slight emphasis on detection and localization
            ch_weights = torch.tensor([1.5, 1.0, 1.2, 1.2, 0.8, 0.8], dtype=torch.float32)
        
        self.register_buffer('_ch_weight', ch_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        
        # Scale-specific weights (higher weight for higher resolution)
        self.scale_weights = scale_weights or [0.1, 0.3, 0.6, 1.0]
        
        # Probability loss using logits for numerical stability
        self._prob_loss = nn.BCEWithLogitsLoss(
            reduction='none', 
            pos_weight=torch.tensor(pos_weight)
        )
        
        # Other channel losses
        self._photon_loss = nn.MSELoss(reduction='none')
        self._loc_loss = nn.SmoothL1Loss(reduction='none')  # More robust than MSE
        self._background_loss = nn.MSELoss(reduction='none')
        
        # Adaptive channel weights
        self.adaptive_weights = AdaptiveChannelWeights(
            initial_weights=self._ch_weight.squeeze().tolist()
        )
        
        # Device tracking
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def set_epoch(self, epoch: int):
        """Update current epoch for progressive training"""
        self.current_epoch = epoch
        
    def forward(self, 
                predictions: Dict[str, Dict[str, torch.Tensor]],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute stable multi-scale loss
        
        Args:
            predictions: Dictionary of scale predictions with 6-channel outputs
            targets: Ground truth targets with 6-channel format
        
        Returns:
            Dictionary of loss components
        """
        total_loss = 0.0
        loss_components = {}
        channel_losses = []
        
        # Get active scales based on training progress
        active_scales = self._get_active_scales()
        
        for scale_idx, (scale_key, scale_pred) in enumerate(predictions.items()):
            if scale_idx not in active_scales:
                continue
                
            scale_weight = self.scale_weights[min(scale_idx, len(self.scale_weights) - 1)]
            
            # Extract 6-channel predictions
            if 'unified_output' in scale_pred:
                # New unified 6-channel output
                output = scale_pred['unified_output']  # (B, 6, H, W)
                target = targets.get(f'{scale_key}_unified', targets.get('unified', None))
                weight_map = targets.get(f'{scale_key}_weight', targets.get('weight', None))
                
                if target is not None:
                    # Compute unified loss
                    unified_loss = self._compute_unified_loss(output, target, weight_map)
                    scale_loss = unified_loss.sum(dim=(1, 2, 3)).mean()  # Average over batch
                    
                    # Store channel-wise losses for adaptive weighting
                    channel_losses.append(unified_loss.mean(dim=(0, 2, 3)))  # (6,)
                    
                else:
                    # Fallback to legacy format
                    scale_loss = self._compute_legacy_loss(scale_pred, targets, scale_key)
            else:
                # Legacy format
                scale_loss = self._compute_legacy_loss(scale_pred, targets, scale_key)
            
            # Apply scale weight with progressive emphasis
            scale_weight = self._get_progressive_scale_weight(scale_idx)
            weighted_scale_loss = scale_weight * scale_loss
            total_loss += weighted_scale_loss
            
            loss_components[f'{scale_key}_total'] = scale_loss
            loss_components[f'{scale_key}_weighted'] = weighted_scale_loss
        
        # Update adaptive weights
        if channel_losses:
            avg_channel_losses = torch.stack(channel_losses).mean(dim=0)
            self.adaptive_weights.update_weights(avg_channel_losses, self.current_epoch)
            self._ch_weight = self.adaptive_weights.get_weights().to(self.device)
            self._ch_weight = self._ch_weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        loss_components['total_loss'] = total_loss
        loss_components['num_active_scales'] = len(active_scales)
        
        return loss_components
    
    def _compute_unified_loss(self, output: torch.Tensor, target: torch.Tensor, 
                             weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute unified PPXYZBG loss following DECODE style
        
        Args:
            output: (B, 6, H, W) - [prob_logits, photons, x, y, z, background]
            target: (B, 6, H, W) - ground truth
            weight: (B, 6, H, W) - loss weights
        
        Returns:
            (B, 6, H, W) - channel-wise losses
        """
        # Probability loss (channel 0) - use logits
        prob_loss = self._prob_loss(output[:, [0]], target[:, [0]])
        
        # Photon loss (channel 1)
        photon_loss = self._photon_loss(output[:, [1]], target[:, [1]])
        
        # Location losses (channels 2, 3) - use SmoothL1 for robustness
        x_loss = self._loc_loss(output[:, [2]], target[:, [2]])
        y_loss = self._loc_loss(output[:, [3]], target[:, [3]])
        
        # Z loss (channel 4)
        z_loss = self._loc_loss(output[:, [4]], target[:, [4]])
        
        # Background loss (channel 5)
        bg_loss = self._background_loss(output[:, [5]], target[:, [5]])
        
        # Combine losses
        channel_losses = torch.cat([
            prob_loss, photon_loss, x_loss, y_loss, z_loss, bg_loss
        ], dim=1)  # (B, 6, H, W)
        
        # Apply weights
        if weight is not None:
            channel_losses = channel_losses * weight
        
        # Apply channel weights
        channel_losses = channel_losses * self._ch_weight
        
        return channel_losses
    
    def _compute_legacy_loss(self, scale_pred: Dict[str, torch.Tensor], 
                           targets: Dict[str, torch.Tensor], scale_key: str) -> torch.Tensor:
        """
        Compute loss for legacy prediction format
        """
        total_loss = 0.0
        
        # Count loss using stable implementation
        if 'prob_map' in scale_pred and 'count' in targets:
            prob_map = scale_pred['prob_map']
            true_count = targets['count']
            count_loss = self._stable_count_loss(prob_map, true_count)
            total_loss += count_loss
        
        # Localization loss using stable implementation
        if all(k in scale_pred for k in ['loc_pred', 'uncertainty']) and 'locations' in targets:
            loc_pred = scale_pred['loc_pred']
            uncertainty = scale_pred['uncertainty']
            true_locations = targets['locations']
            prob_map = scale_pred.get('prob_map')
            loc_loss = self._stable_localization_loss(loc_pred, uncertainty, true_locations, prob_map)
            total_loss += loc_loss
        
        return total_loss
    
    def _stable_count_loss(self, prob_map: torch.Tensor, true_count: torch.Tensor) -> torch.Tensor:
        """
        Stable count loss using log-sum-exp trick
        """
        B = prob_map.size(0)
        
        # Use log-sum-exp for numerical stability
        log_prob_map = torch.log(torch.clamp(prob_map, min=self.eps, max=1.0 - self.eps))
        log_one_minus_prob = torch.log(torch.clamp(1.0 - prob_map, min=self.eps, max=1.0 - self.eps))
        
        # Flatten probability map
        log_p = log_prob_map.view(B, -1)  # (B, H*W)
        log_one_minus_p = log_one_minus_prob.view(B, -1)
        
        # Calculate expected count using log-sum-exp
        mu = torch.exp(log_p).sum(dim=1)  # Expected count
        
        # Variance calculation with numerical protection
        p_clamped = torch.clamp(prob_map.view(B, -1), min=self.eps, max=1.0 - self.eps)
        var = (p_clamped * (1.0 - p_clamped)).sum(dim=1) + self.eps
        
        # Gaussian negative log-likelihood with numerical protection
        diff = true_count.float() - mu
        log_var = torch.log(var)
        nll = 0.5 * (diff.pow(2) / var + log_var + math.log(2 * math.pi))
        
        return nll.mean()
    
    def _stable_localization_loss(self, loc_pred: torch.Tensor, uncertainty: torch.Tensor,
                                 true_locations: torch.Tensor, prob_map: torch.Tensor) -> torch.Tensor:
        """
        Stable localization loss with improved matching
        """
        B, H, W, _ = loc_pred.shape
        device = loc_pred.device
        
        total_loss = 0.0
        valid_batches = 0
        
        for b in range(B):
            batch_prob = prob_map[b, 0] if prob_map is not None else torch.ones(H, W, device=device)
            batch_loc = loc_pred[b]  # (H, W, 2)
            batch_uncertainty = uncertainty[b, :, :, 0]  # (H, W)
            batch_true_locs = true_locations[b]  # (N, 2)
            
            # Filter valid locations
            valid_mask = (batch_true_locs[:, 0] >= 0) & (batch_true_locs[:, 1] >= 0)
            if not valid_mask.any():
                continue
            
            valid_true_locs = batch_true_locs[valid_mask]
            
            # Find confident predictions with adaptive threshold
            conf_threshold = max(0.3, 0.8 - self.current_epoch * 0.01)  # Adaptive threshold
            conf_mask = batch_prob > conf_threshold
            
            if not conf_mask.any():
                # Penalty for no confident predictions when there are true emitters
                total_loss += 5.0
                valid_batches += 1
                continue
            
            # Get confident prediction locations
            conf_indices = torch.nonzero(conf_mask, as_tuple=False)
            conf_locs = batch_loc[conf_indices[:, 0], conf_indices[:, 1]]
            conf_uncertainties = batch_uncertainty[conf_indices[:, 0], conf_indices[:, 1]]
            
            # Improved distance calculation with uncertainty weighting
            if len(conf_locs) > 0 and len(valid_true_locs) > 0:
                # Distance matrix with uncertainty weighting
                dist_matrix = torch.cdist(conf_locs, valid_true_locs, p=2)
                
                # Uncertainty-aware weighting
                uncertainty_weights = torch.exp(-conf_uncertainties.unsqueeze(1))
                weighted_dist_matrix = dist_matrix * uncertainty_weights
                
                # Hungarian-style assignment (simplified)
                min_distances, _ = weighted_dist_matrix.min(dim=0)
                loc_loss = min_distances.mean()
                
                total_loss += loc_loss
                valid_batches += 1
        
        return total_loss / max(valid_batches, 1)
    
    def _get_active_scales(self) -> List[int]:
        """
        Get active scales based on training progress
        """
        if self.current_epoch < self.warmup_epochs // 4:
            return [0]  # Start with lowest scale
        elif self.current_epoch < self.warmup_epochs // 2:
            return [0, 1]
        elif self.current_epoch < self.warmup_epochs:
            return [0, 1, 2]
        else:
            return list(range(len(self.scale_weights)))  # All scales
    
    def _get_progressive_scale_weight(self, scale_idx: int) -> float:
        """
        Get progressive scale weight based on training progress
        """
        base_weight = self.scale_weights[min(scale_idx, len(self.scale_weights) - 1)]
        
        # Progressive weight adjustment
        progress = min(self.current_epoch / self.warmup_epochs, 1.0)
        
        # Early training: emphasize lower scales
        # Later training: emphasize higher scales
        if scale_idx == 0:
            return base_weight * (1.5 - 0.5 * progress)
        else:
            return base_weight * (0.5 + 0.5 * progress)


class AdaptiveChannelWeights:
    """
    Adaptive channel weight adjustment based on training progress
    """
    
    def __init__(self, initial_weights: List[float]):
        self.weights = torch.tensor(initial_weights)
        self.loss_history = {i: [] for i in range(len(initial_weights))}
        self.update_frequency = 10  # Update every 10 epochs
        
    def update_weights(self, channel_losses: torch.Tensor, epoch: int):
        """
        Update weights based on channel loss trends
        """
        # Record loss history
        for i, loss in enumerate(channel_losses):
            self.loss_history[i].append(loss.item())
        
        # Update weights periodically
        if epoch % self.update_frequency == 0 and epoch > 0:
            for i in range(len(self.weights)):
                if len(self.loss_history[i]) >= self.update_frequency:
                    recent_avg = np.mean(self.loss_history[i][-self.update_frequency:])
                    
                    if len(self.loss_history[i]) >= 2 * self.update_frequency:
                        early_avg = np.mean(self.loss_history[i][-2*self.update_frequency:-self.update_frequency])
                        
                        # Adjust weights based on improvement rate
                        improvement_ratio = recent_avg / (early_avg + 1e-8)
                        
                        if improvement_ratio > 0.95:  # Slow improvement
                            self.weights[i] *= 1.05
                        elif improvement_ratio < 0.85:  # Fast improvement
                            self.weights[i] *= 0.95
                        
                        # Clamp weights to reasonable range
                        self.weights[i] = torch.clamp(self.weights[i], 0.1, 3.0)
    
    def get_weights(self) -> torch.Tensor:
        return self.weights


class UnifiedPPXYZBLoss(nn.Module):
    """
    Unified 6-channel loss following DECODE's PPXYZBLoss implementation
    Channels: [prob_logits, photons, x_offset, y_offset, z_offset, background]
    """
    
    def __init__(self, device: torch.device = None, 
                 channel_weights: Optional[List[float]] = None,
                 pos_weight: float = 2.0,
                 eps: float = 1e-4):
        super().__init__()
        
        self.eps = eps
        
        # Channel weights
        if channel_weights is not None:
            ch_weights = torch.tensor(channel_weights, dtype=torch.float32)
        else:
            ch_weights = torch.tensor([1.5, 1.0, 1.2, 1.2, 0.8, 0.8], dtype=torch.float32)
        
        if device is not None:
            ch_weights = ch_weights.to(device)
        
        self.register_buffer('_ch_weight', ch_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        
        # Loss functions
        pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)
        if device is not None:
            pos_weight_tensor = pos_weight_tensor.to(device)
        
        self._prob_loss = nn.BCEWithLogitsLoss(
            reduction='none',
            pos_weight=pos_weight_tensor
        )
        self._other_loss = nn.MSELoss(reduction='none')
    
    def forward(self, output: torch.Tensor, target: torch.Tensor, 
                weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output: (B, 6, H, W) - [prob_logits, photons, x, y, z, bg]
            target: (B, 6, H, W) - ground truth
            weight: (B, 6, H, W) - loss weights
        
        Returns:
            (B, 6, H, W) - weighted losses
        """
        # Probability loss (channel 0)
        prob_loss = self._prob_loss(output[:, [0]], target[:, [0]])
        
        # Other channel losses (channels 1-5)
        other_losses = self._other_loss(output[:, 1:], target[:, 1:])
        
        # Combine losses
        total_losses = torch.cat([prob_loss, other_losses], dim=1)
        
        # Apply weights
        weighted_losses = total_losses * weight * self._ch_weight
        
        return weighted_losses