import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Union
import numpy as np
from functools import partial
from var_emitter_loss import CountLoss


class EmitterVectorQuantizer(nn.Module):
    """
    Vector Quantizer specifically designed for emitter prediction
    Implements VAR's core residual accumulation mechanism
    """
    
    def __init__(self, 
                 vocab_size: int = 8192,
                 embed_dim: int = 256,
                 patch_nums: Tuple[int, ...] = (10, 20, 40, 80),
                 quant_resi: float = 0.5,
                 beta: float = 0.25):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.patch_nums = patch_nums
        self.beta = beta
        
        # Embedding table for quantization
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.uniform_(self.embedding.weight, -1/vocab_size, 1/vocab_size)
        
        # Phi residual networks for each scale
        self.quant_resi = self._build_phi_networks(embed_dim, quant_resi, len(patch_nums))
        
    def _build_phi_networks(self, embed_dim: int, quant_resi: float, num_scales: int):
        """Build Phi residual networks for different scales"""
        phi_networks = nn.ModuleList()
        for i in range(num_scales):
            phi = PhiResidual(embed_dim, quant_resi)
            phi_networks.append(phi)
        return phi_networks
    
    def forward(self, f_BChw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard VQ forward pass"""
        B, C, H, W = f_BChw.shape
        
        # Flatten spatial dimensions
        f_flat = f_BChw.permute(0, 2, 3, 1).contiguous().view(-1, C)  # BHW, C
        
        # Calculate distances to embedding vectors
        distances = torch.sum(f_flat**2, dim=1, keepdim=True) + \
                   torch.sum(self.embedding.weight**2, dim=1) - \
                   2 * torch.matmul(f_flat, self.embedding.weight.t())
        
        # Get closest embedding indices
        indices = torch.argmin(distances, dim=1).unsqueeze(1)  # BHW, 1
        indices_flat = indices.view(-1)
        
        # Get quantized vectors
        quantized_flat = self.embedding(indices_flat)
        quantized = quantized_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        # Calculate VQ loss
        commitment_loss = F.mse_loss(f_BChw.detach(), quantized)
        embedding_loss = F.mse_loss(f_BChw, quantized.detach())
        vq_loss = embedding_loss + self.beta * commitment_loss
        
        # Straight-through estimator
        quantized = f_BChw + (quantized - f_BChw).detach()
        
        return quantized, vq_loss, indices.view(B, H, W)
    
    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """VAR's core residual accumulation mechanism"""
        max_HW = self.patch_nums[-1]
        
        if si != SN - 1:
            # Apply Phi residual network after upsampling
            h = self.quant_resi[si](F.interpolate(h_BChw, size=(max_HW, max_HW), mode='bicubic'))
            f_hat = f_hat + h  # Residual accumulation (non-inplace)
            # Prepare next scale input
            next_input = F.interpolate(f_hat, size=(self.patch_nums[si+1], self.patch_nums[si+1]), mode='area')
            return f_hat, next_input
        else:
            # Final scale
            h = self.quant_resi[si](h_BChw)
            f_hat = f_hat + h  # Non-inplace operation
            return f_hat, f_hat
    
    def embed_to_fhat(self, ms_h_BChw: List[torch.Tensor]) -> torch.Tensor:
        """Convert multi-scale embeddings to final feature map through residual accumulation"""
        B = ms_h_BChw[0].shape[0]
        H = W = self.patch_nums[-1]
        SN = len(self.patch_nums)
        
        f_hat = ms_h_BChw[0].new_zeros(B, self.embed_dim, H, W, dtype=torch.float32)
        
        for si, pn in enumerate(self.patch_nums):
            h_BChw = ms_h_BChw[si]
            if si < len(self.patch_nums) - 1:
                h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
            h_BChw = self.quant_resi[si](h_BChw)
            f_hat = f_hat + h_BChw  # Residual accumulation (non-inplace)
        
        return f_hat


class PhiResidual(nn.Conv2d):
    """Phi residual network from VAR"""
    
    def __init__(self, embed_dim: int, quant_resi: float):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, 
                        kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BChw: torch.Tensor) -> torch.Tensor:
        return h_BChw.mul(1 - self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class AdaLNSelfAttn(nn.Module):
    """Adaptive Layer Norm Self-Attention from VAR"""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, 
                 dropout: float = 0.1, norm_eps: float = 1e-6):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Adaptive Layer Norm
        self.ln1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.ln2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        
        # Self-attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Adaptive conditioning
        self.ada_lin = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim)
        )
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Adaptive conditioning
        ada_params = self.ada_lin(cond).view(-1, 1, 6, self.embed_dim)  # B, 1, 6, C
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_params.unbind(2)
        
        # Self-attention with adaptive norm
        x_norm = self.ln1(x)
        x_norm = x_norm * (1 + scale_msa) + shift_msa
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa * attn_out
        
        # MLP with adaptive norm
        x_norm = self.ln2(x)
        x_norm = x_norm * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp * mlp_out
        
        return x


class TrueVAREmitterPredictor(nn.Module):
    """
    True VAR-based Emitter Predictor implementing:
    1. Progressive residual accumulation
    2. Next-scale autoregressive prediction
    3. Multi-scale feature fusion
    """
    
    def __init__(self,
                 patch_nums: Tuple[int, ...] = (10, 20, 40, 80),  # Progressive scales: 10x10 -> 80x80
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 24,
                 vocab_size: int = 8192,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 input_channels: int = 1):
        super().__init__()
        
        self.patch_nums = patch_nums
        self.embed_dim = embed_dim
        self.num_scales = len(patch_nums)
        self.vocab_size = vocab_size
        
        # Calculate sequence lengths for each scale
        self.seq_lens = [pn * pn for pn in patch_nums]
        self.total_len = sum(self.seq_lens)
        
        # Input projection (no downsampling!)
        self.input_proj = nn.Conv2d(input_channels, embed_dim, 3, 1, 1)
        
        # Vector quantizer with residual accumulation
        self.quantizer = EmitterVectorQuantizer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            patch_nums=patch_nums
        )
        
        # Word embedding for tokens
        self.word_embed = nn.Linear(embed_dim, embed_dim)
        
        # Position embeddings for each scale
        self.pos_embeds = nn.ParameterList()
        for pn in patch_nums:
            pos_embed = nn.Parameter(torch.zeros(1, pn * pn, embed_dim))
            nn.init.trunc_normal_(pos_embed, std=0.02)
            self.pos_embeds.append(pos_embed)
        
        # Level embeddings to distinguish scales
        self.level_embed = nn.Embedding(self.num_scales, embed_dim)
        
        # Scale conditioning embedding
        self.scale_cond = nn.Embedding(self.num_scales, embed_dim)
        
        # VAR Transformer blocks
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Output heads for each scale
        self.output_heads = nn.ModuleList([
            nn.Linear(embed_dim, vocab_size) for _ in range(self.num_scales)
        ])
        
        # Emitter-specific prediction heads
        self.emitter_prob_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(self.num_scales)
        ])
        
        self.emitter_loc_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, 2),
                nn.Tanh()  # Normalized coordinates [-1, 1]
            ) for _ in range(self.num_scales)
        ])
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights following VAR's initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode_input(self, x: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """Encode input at specific scale without downsampling"""
        target_size = self.patch_nums[scale_idx]
        
        # Resize input to target scale
        x_resized = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)
        
        # Project to embedding space (no downsampling!)
        features = self.input_proj(x_resized)  # B, C, H, W
        
        return features
    
    def forward(self, x: torch.Tensor, target_scale: int = -1) -> Dict[str, torch.Tensor]:
        """
        VAR forward pass with progressive residual accumulation
        
        Args:
            x: Input tensor (B, 1, H, W) - can be any resolution
            target_scale: Target scale index (-1 for highest)
        """
        if target_scale == -1:
            target_scale = self.num_scales - 1
        
        batch_size = x.size(0)
        device = x.device
        
        # Initialize accumulated features at maximum resolution
        max_size = self.patch_nums[-1]
        f_hat = torch.zeros(batch_size, self.embed_dim, max_size, max_size, device=device)
        
        scale_outputs = {}
        
        # Progressive scale prediction (VAR's core mechanism)
        prev_quantized = None
        for si in range(target_scale + 1):
            current_size = self.patch_nums[si]
            
            if si == 0:
                # Initial scale: encode input directly
                features = self.encode_input(x, si)  # B, C, H, W
                
                # Quantize features
                quantized, vq_loss, indices = self.quantizer(features)
                prev_quantized = quantized  # Store for next iteration
                
                # Convert to tokens
                tokens = quantized.flatten(2).transpose(1, 2)  # B, H*W, C
                tokens = self.word_embed(tokens)
                
            else:
                # Higher scales: use VAR's autoregressive mechanism
                # Get next autoregressive input from accumulated features
                f_hat, next_input = self.quantizer.get_next_autoregressive_input(
                    si-1, self.num_scales, f_hat, prev_quantized
                )
                
                # Convert accumulated features to tokens
                current_features = F.interpolate(f_hat, size=(current_size, current_size), mode='area')
                tokens = current_features.flatten(2).transpose(1, 2)  # B, H*W, C
                tokens = self.word_embed(tokens)
                
                # Encode current scale input for residual
                current_input_features = self.encode_input(x, si)
                quantized, vq_loss, indices = self.quantizer(current_input_features)
                prev_quantized = quantized  # Store for next iteration
            
            # Add positional and level embeddings
            pos_embed = self.pos_embeds[si]
            level_embed = self.level_embed(torch.tensor(si, device=device)).unsqueeze(0).unsqueeze(0)
            scale_cond = self.scale_cond(torch.tensor(si, device=device)).unsqueeze(0)
            
            tokens = tokens + pos_embed + level_embed
            
            # Apply VAR transformer blocks with scale conditioning
            for block in self.blocks:
                tokens = block(tokens, scale_cond)
            
            # Generate predictions for current scale
            logits = self.output_heads[si](tokens)  # B, H*W, vocab_size
            emitter_prob = self.emitter_prob_head[si](tokens)  # B, H*W, 1
            emitter_loc = self.emitter_loc_head[si](tokens)   # B, H*W, 2
            
            # Reshape to spatial dimensions
            prob_map = emitter_prob.view(batch_size, current_size, current_size, 1).permute(0, 3, 1, 2)
            loc_map = emitter_loc.view(batch_size, current_size, current_size, 2).permute(0, 3, 1, 2)
            
            scale_outputs[f'scale_{si}'] = {
                'logits': logits,
                'prob_map': prob_map,
                'loc_map': loc_map,
                'indices': indices,
                'vq_loss': vq_loss if si == 0 else torch.tensor(0.0, device=device),
                'tokens': tokens,
                'resolution': current_size
            }
        
        return scale_outputs
    
    @torch.no_grad()
    def autoregressive_inference(self, x: torch.Tensor, target_resolution: int = 160) -> Dict[str, torch.Tensor]:
        """
        VAR-style autoregressive inference from low-res to high-res
        
        Args:
            x: Low resolution input (B, 1, 40, 40)
            target_resolution: Target high resolution (e.g., 160)
        """
        # Find target scale index
        target_scale_idx = -1
        for i, pn in enumerate(self.patch_nums):
            if pn >= target_resolution:
                target_scale_idx = i
                break
        
        if target_scale_idx == -1:
            target_scale_idx = self.num_scales - 1
        
        # Progressive inference
        outputs = self.forward(x, target_scale_idx)
        
        # Return highest resolution output
        highest_scale_key = f'scale_{target_scale_idx}'
        return outputs[highest_scale_key]
    
    def predict_super_resolution(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict at highest resolution (super-resolution mode)
        Input: 40x40, Output: 80x80 (or highest configured resolution)
        """
        return self.autoregressive_inference(x, self.patch_nums[-1])


class VAREmitterLoss(nn.Module):
    """Loss function for VAR emitter prediction"""
    
    def __init__(self, 
                 prob_weight: float = 1.0,
                 loc_weight: float = 1.0, 
                 vq_weight: float = 0.1,
                 count_weight: float = 1.0,
                 scale_weights: Optional[List[float]] = None):
        super().__init__()
        
        self.prob_weight = prob_weight
        self.loc_weight = loc_weight
        self.vq_weight = vq_weight
        self.count_weight = count_weight
        self.scale_weights = scale_weights or [0.5, 0.7, 0.9, 1.0]  # Increasing weights for higher scales
        
        # Initialize count loss
        self.count_loss = CountLoss()
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate multi-scale VAR loss
        
        Args:
            predictions: Model outputs for each scale
            targets: Ground truth targets for each scale (should include 'emitter_count')
        """
        device = next(iter(predictions.values()))['prob_map'].device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        losses = {}
        
        # Extract emitter count from targets if available
        emitter_count = targets.get('emitter_count', None)
        
        # Create mapping from prediction scales to target scales
        # predictions: scale_0, scale_1, scale_2, scale_3 (resolutions: 10, 20, 40, 80)
        # targets: scale_0, scale_1, scale_2, scale_3 (resolutions: 10, 20, 40, 80)
        scale_mapping = {
            'scale_0': 'scale_0',  # 10x10
            'scale_1': 'scale_1',  # 20x20
            'scale_2': 'scale_2',  # 40x40
            'scale_3': 'scale_3'   # 80x80
        }
        
        for pred_scale_key in predictions.keys():
            if pred_scale_key.startswith('scale_'):
                scale_idx = int(pred_scale_key.split('_')[1])
                scale_weight = self.scale_weights[min(scale_idx, len(self.scale_weights)-1)]
                
                pred = predictions[pred_scale_key]
                target_scale_key = scale_mapping.get(pred_scale_key)
                
                if target_scale_key and target_scale_key in targets:
                    target = targets[target_scale_key]
                    
                    # Probability map loss (BCE)
                    prob_loss = F.binary_cross_entropy(
                        pred['prob_map'], target['prob_map']
                    )
                    
                    # Count loss (probability map sum should match emitter count)
                    count_loss = torch.tensor(0.0, device=device)
                    if emitter_count is not None:
                        if isinstance(emitter_count, (int, float)):
                            # Convert single value to tensor
                            emitter_count_tensor = torch.tensor([emitter_count], device=device, dtype=torch.float32)
                        else:
                            # Assume it's already a tensor or list
                            emitter_count_tensor = torch.tensor(emitter_count, device=device, dtype=torch.float32)
                        
                        # Ensure batch dimension matches
                        if emitter_count_tensor.dim() == 0:
                            emitter_count_tensor = emitter_count_tensor.unsqueeze(0)
                        if len(emitter_count_tensor) == 1 and pred['prob_map'].size(0) > 1:
                            emitter_count_tensor = emitter_count_tensor.expand(pred['prob_map'].size(0))
                        
                        count_loss = self.count_loss(pred['prob_map'], emitter_count_tensor)
                    
                    # Location loss (MSE, only where emitters exist)
                    emitter_mask = target['prob_map'] > 0.5  # (B, 1, H, W)
                    if emitter_mask.sum() > 0:
                        # Expand mask to match loc_map dimensions (B, 2, H, W)
                        emitter_mask_expanded = emitter_mask.expand_as(target['loc_map'])
                        loc_loss = F.mse_loss(
                            pred['loc_map'][emitter_mask_expanded],
                            target['loc_map'][emitter_mask_expanded]
                        )
                    else:
                        loc_loss = torch.tensor(0.0, device=pred['prob_map'].device)
                    
                    # VQ loss
                    vq_loss = pred['vq_loss']
                    
                    # Weighted scale loss
                    scale_loss = scale_weight * (
                        self.prob_weight * prob_loss +
                        self.loc_weight * loc_loss +
                        self.vq_weight * vq_loss +
                        self.count_weight * count_loss
                    )
                    
                    total_loss = total_loss + scale_loss
                    losses[f'{pred_scale_key}_prob'] = prob_loss
                    losses[f'{pred_scale_key}_loc'] = loc_loss
                    losses[f'{pred_scale_key}_vq'] = vq_loss
                    losses[f'{pred_scale_key}_count'] = count_loss
                    losses[f'{pred_scale_key}_total'] = scale_loss
        
        losses['total'] = total_loss
        return losses