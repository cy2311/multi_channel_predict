import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class VAREmitterLoss(nn.Module):
    """
    Multi-scale loss function for VAR-based emitter prediction
    Combines count loss, localization loss, and reconstruction loss across scales
    """
    
    def __init__(self,
                 count_weight: float = 1.0,
                 loc_weight: float = 1.0,
                 recon_weight: float = 0.5,
                 uncertainty_weight: float = 0.1,
                 scale_weights: Optional[List[float]] = None,
                 eps: float = 1e-6):
        super().__init__()
        
        self.count_weight = count_weight
        self.loc_weight = loc_weight
        self.recon_weight = recon_weight
        self.uncertainty_weight = uncertainty_weight
        self.eps = eps
        
        # Scale-specific weights (higher weight for higher resolution)
        self.scale_weights = scale_weights or [0.1, 0.3, 0.6, 1.0]
        
        # Individual loss components
        self.count_loss = CountLoss(eps=eps)
        self.loc_loss = LocalizationLoss()
        self.uncertainty_loss = UncertaintyLoss()
        
    def forward(self, 
                predictions: Dict[str, Dict[str, torch.Tensor]],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-scale loss
        
        Args:
            predictions: Dictionary of scale predictions
            targets: Ground truth targets
                - 'count': (B,) emitter counts
                - 'locations': (B, N, 2) emitter locations
                - 'prob_maps': Dictionary of scale-specific probability maps
        
        Returns:
            Dictionary of loss components
        """
        total_loss = 0.0
        loss_components = {}
        
        num_scales = len(predictions)
        
        for scale_idx, (scale_key, scale_pred) in enumerate(predictions.items()):
            scale_weight = self.scale_weights[min(scale_idx, len(self.scale_weights) - 1)]
            
            # Extract predictions for current scale
            prob_map = scale_pred['prob_map']  # (B, 1, H, W)
            count_pred = scale_pred['count_pred']  # (B, 1)
            loc_pred = scale_pred['loc_pred']  # (B, H, W, 2)
            uncertainty = scale_pred['uncertainty']  # (B, H, W, 1)
            
            # Count loss
            count_loss = self.count_loss(prob_map, targets['count'])
            
            # Localization loss
            loc_loss = self.loc_loss(loc_pred, uncertainty, targets['locations'], prob_map)
            
            # Uncertainty loss (encourage calibrated uncertainty)
            uncertainty_loss = self.uncertainty_loss(uncertainty, prob_map, targets)
            
            # Reconstruction loss (if available)
            recon_loss = 0.0
            if f'scale_{scale_idx}' in targets.get('prob_maps', {}):
                target_prob_map = targets['prob_maps'][f'scale_{scale_idx}']
                recon_loss = F.mse_loss(prob_map, target_prob_map)
            
            # Combine losses for current scale
            scale_loss = (
                self.count_weight * count_loss +
                self.loc_weight * loc_loss +
                self.recon_weight * recon_loss +
                self.uncertainty_weight * uncertainty_loss
            )
            
            # Weight by scale importance
            weighted_scale_loss = scale_weight * scale_loss
            total_loss += weighted_scale_loss
            
            # Store individual components
            loss_components[f'{scale_key}_count'] = count_loss
            loss_components[f'{scale_key}_loc'] = loc_loss
            loss_components[f'{scale_key}_recon'] = recon_loss
            loss_components[f'{scale_key}_uncertainty'] = uncertainty_loss
            loss_components[f'{scale_key}_total'] = scale_loss
        
        loss_components['total_loss'] = total_loss
        return loss_components


class CountLoss(nn.Module):
    """
    Improved count loss with Poisson Binomial approximation
    """
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, prob_map: torch.Tensor, true_count: torch.Tensor) -> torch.Tensor:
        """
        Compute count loss using Gaussian approximation of Poisson Binomial
        
        Args:
            prob_map: Predicted probabilities (B, 1, H, W)
            true_count: Ground truth counts (B,)
        """
        B = prob_map.size(0)
        
        # Ensure probabilities are in valid range
        prob_map = torch.clamp(prob_map, min=self.eps, max=1.0 - self.eps)
        
        # Flatten probability map
        p = prob_map.view(B, -1)  # (B, H*W)
        
        # Calculate mean and variance of Poisson Binomial
        mu = p.sum(dim=1)  # Expected count
        var = (p * (1.0 - p)).sum(dim=1) + self.eps  # Variance
        
        # Gaussian negative log-likelihood
        diff = true_count.float() - mu
        nll = 0.5 * (diff.pow(2) / var + torch.log(2 * math.pi * var))
        
        return nll.mean()


class LocalizationLoss(nn.Module):
    """
    Localization loss with uncertainty weighting
    """
    
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, 
                loc_pred: torch.Tensor,  # (B, H, W, 2)
                uncertainty: torch.Tensor,  # (B, H, W, 1)
                true_locations: torch.Tensor,  # (B, N, 2)
                prob_map: torch.Tensor) -> torch.Tensor:  # (B, 1, H, W)
        """
        Compute localization loss using Hungarian matching
        """
        B, H, W, _ = loc_pred.shape
        device = loc_pred.device
        
        total_loss = 0.0
        
        for b in range(B):
            # Get predictions for current batch item
            batch_prob = prob_map[b, 0]  # (H, W)
            batch_loc = loc_pred[b]  # (H, W, 2)
            batch_uncertainty = uncertainty[b, :, :, 0]  # (H, W)
            batch_true_locs = true_locations[b]  # (N, 2)
            
            # Filter out padding locations (assuming -1 indicates padding)
            valid_mask = (batch_true_locs[:, 0] >= 0) & (batch_true_locs[:, 1] >= 0)
            if not valid_mask.any():
                continue
            
            valid_true_locs = batch_true_locs[valid_mask]  # (M, 2)
            
            # Find high-confidence predictions
            conf_threshold = 0.5
            conf_mask = batch_prob > conf_threshold
            
            if not conf_mask.any():
                # No confident predictions, add penalty
                total_loss += 10.0
                continue
            
            # Get confident prediction locations
            conf_indices = torch.nonzero(conf_mask, as_tuple=False)  # (K, 2)
            conf_locs = batch_loc[conf_indices[:, 0], conf_indices[:, 1]]  # (K, 2)
            conf_uncertainties = batch_uncertainty[conf_indices[:, 0], conf_indices[:, 1]]  # (K,)
            
            # Compute pairwise distances
            if len(conf_locs) > 0 and len(valid_true_locs) > 0:
                # Distance matrix (K, M)
                dist_matrix = torch.cdist(conf_locs, valid_true_locs, p=2)
                
                # Simple assignment: each true location to closest prediction
                min_distances, _ = dist_matrix.min(dim=0)  # (M,)
                
                # Uncertainty-weighted loss
                # Higher uncertainty should lead to lower penalty for errors
                uncertainty_weights = torch.exp(-conf_uncertainties)
                weighted_distances = min_distances * uncertainty_weights[:len(min_distances)]
                
                loc_loss = weighted_distances.mean()
                total_loss += loc_loss
        
        return total_loss / B if B > 0 else torch.tensor(0.0, device=device)


class UncertaintyLoss(nn.Module):
    """
    Loss to encourage well-calibrated uncertainty estimates
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, 
                uncertainty: torch.Tensor,  # (B, H, W, 1)
                prob_map: torch.Tensor,  # (B, 1, H, W)
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encourage uncertainty to correlate with prediction confidence
        """
        B, H, W, _ = uncertainty.shape
        
        # Flatten tensors
        uncertainty_flat = uncertainty.view(B, -1)  # (B, H*W)
        prob_flat = prob_map.view(B, -1)  # (B, H*W)
        
        # Uncertainty should be high where confidence is low
        confidence = torch.abs(prob_flat - 0.5) * 2  # Map [0,1] to [0,1] where 0.5->0, 0/1->1
        target_uncertainty = 1.0 - confidence
        
        # MSE loss between predicted and target uncertainty
        uncertainty_loss = F.mse_loss(uncertainty_flat, target_uncertainty)
        
        return uncertainty_loss


class ProgressiveLoss(nn.Module):
    """
    Progressive training loss that gradually increases resolution
    """
    
    def __init__(self, 
                 base_loss: VAREmitterLoss,
                 warmup_epochs: int = 10,
                 scale_schedule: List[int] = [5, 10, 15, 20]):
        super().__init__()
        
        self.base_loss = base_loss
        self.warmup_epochs = warmup_epochs
        self.scale_schedule = scale_schedule
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """Update current epoch for progressive training"""
        self.current_epoch = epoch
    
    def get_active_scales(self) -> List[int]:
        """Get currently active scales based on training progress"""
        if self.current_epoch < self.warmup_epochs:
            return [0]  # Only lowest resolution during warmup
        
        # Gradually add higher resolution scales
        active_scales = [0]
        for i, epoch_threshold in enumerate(self.scale_schedule):
            if self.current_epoch >= epoch_threshold:
                active_scales.append(i + 1)
        
        return active_scales
    
    def forward(self, 
                predictions: Dict[str, Dict[str, torch.Tensor]],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute progressive loss using only active scales
        """
        active_scales = self.get_active_scales()
        
        # Filter predictions to only include active scales
        filtered_predictions = {
            f'scale_{i}': predictions[f'scale_{i}']
            for i in active_scales
            if f'scale_{i}' in predictions
        }
        
        return self.base_loss(filtered_predictions, targets)