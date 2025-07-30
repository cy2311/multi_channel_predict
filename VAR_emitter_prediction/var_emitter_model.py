import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np


class MultiScaleVQVAE(nn.Module):
    """
    Multi-scale Vector Quantized VAE for emitter representation
    Handles different resolution levels for progressive prediction
    """
    
    def __init__(self, 
                 input_channels: int = 1,
                 base_channels: int = 64,
                 codebook_size: int = 1024,
                 embedding_dim: int = 256,
                 num_scales: int = 4):
        super().__init__()
        
        self.num_scales = num_scales
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        
        # Multi-scale encoders
        self.encoders = nn.ModuleList()
        for i in range(num_scales):
            scale_channels = base_channels * (2 ** i)
            encoder = nn.Sequential(
                nn.Conv2d(input_channels, scale_channels // 4, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(scale_channels // 4, scale_channels // 2, 3, 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(scale_channels // 2, scale_channels, 3, 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(scale_channels, embedding_dim, 1)
            )
            self.encoders.append(encoder)
        
        # Vector quantization codebooks for each scale
        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, embedding_dim) 
            for _ in range(num_scales)
        ])
        
        # Multi-scale decoders
        self.decoders = nn.ModuleList()
        for i in range(num_scales):
            scale_channels = base_channels * (2 ** i)
            decoder = nn.Sequential(
                nn.Conv2d(embedding_dim, scale_channels, 1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(scale_channels, scale_channels // 2, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(scale_channels // 2, scale_channels // 4, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(scale_channels // 4, 1, 3, 1, 1),  # Output probability map
                nn.Sigmoid()
            )
            self.decoders.append(decoder)
    
    def encode(self, x: torch.Tensor, scale_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input at specific scale"""
        z = self.encoders[scale_idx](x)
        z_q, indices = self.quantizers[scale_idx](z)
        return z_q, indices
    
    def decode(self, z_q: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """Decode quantized features at specific scale"""
        return self.decoders[scale_idx](z_q)
    
    def forward(self, x: torch.Tensor, scale_idx: int) -> Dict[str, torch.Tensor]:
        z_q, indices = self.encode(x, scale_idx)
        recon = self.decode(z_q, scale_idx)
        
        return {
            'reconstruction': recon,
            'quantized': z_q,
            'indices': indices
        }


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Convert quantized from BHWC -> BCHW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        return quantized, encoding_indices.view(input_shape[:-1])


class ProgressiveEmitterTransformer(nn.Module):
    """
    Progressive Transformer for multi-scale emitter prediction
    Inspired by VAR's next-scale prediction mechanism
    """
    
    def __init__(self,
                 vqvae: MultiScaleVQVAE,
                 patch_nums: Tuple[int, ...] = (5, 10, 20, 40),  # Progressive patch numbers
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        
        self.vqvae = vqvae
        self.patch_nums = patch_nums
        self.num_scales = len(patch_nums)
        self.embed_dim = embed_dim
        
        # Token embedding for each scale
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(vqvae.codebook_size, embed_dim)
            for _ in range(self.num_scales)
        ])
        
        # Position embeddings for each scale
        # Initialize with a reasonable size, will be interpolated to match actual encoder output
        self.pos_embeddings = nn.ParameterList()
        for patch_num in patch_nums:
            # Use a base size that can be interpolated
            base_size = max(8, patch_num // 4)  # Minimum 8x8 for interpolation
            pos_emb = nn.Parameter(torch.randn(1, base_size * base_size, embed_dim))
            self.pos_embeddings.append(pos_emb)
        
        # Scale embeddings
        self.scale_embeddings = nn.Embedding(self.num_scales, embed_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Output heads for each scale
        self.output_heads = nn.ModuleList([
            nn.Linear(embed_dim, vqvae.codebook_size)
            for _ in range(self.num_scales)
        ])
        
        # Emitter-specific output heads
        self.emitter_count_head = nn.Linear(embed_dim, 1)  # Count prediction
        self.emitter_loc_head = nn.Linear(embed_dim, 2)    # Location prediction (x, y)
        self.uncertainty_head = nn.Linear(embed_dim, 1)    # Uncertainty estimation
        
    def forward(self, 
                low_res_input: torch.Tensor,  # 40x40 input
                target_scale: int = -1) -> Dict[str, torch.Tensor]:
        """
        Progressive prediction from low-resolution input to high-resolution output
        
        Args:
            low_res_input: Low resolution input (B, 1, 40, 40)
            target_scale: Target scale for prediction (-1 for highest scale)
        """
        batch_size = low_res_input.size(0)
        device = low_res_input.device
        
        if target_scale == -1:
            target_scale = self.num_scales - 1
        
        # Start with lowest resolution encoding
        current_tokens = None
        scale_outputs = {}
        
        for scale_idx in range(target_scale + 1):
            if scale_idx == 0:
                # Initial encoding from low-res input
                # Resize input to match current scale
                current_size = self.patch_nums[scale_idx]
                resized_input = F.interpolate(low_res_input, 
                                            size=(current_size, current_size), 
                                            mode='bilinear', 
                                            align_corners=False)
                
                # Encode with VQVAE
                vqvae_output = self.vqvae(resized_input, scale_idx)
                indices = vqvae_output['indices']  # (B, H, W)
                
                # Convert to tokens
                current_tokens = self.token_embeddings[scale_idx](indices.flatten(1))  # (B, H*W, D)
                
            else:
                # Progressive prediction for higher scales
                # Use previous scale tokens to predict current scale
                prev_tokens = current_tokens
                current_size = self.patch_nums[scale_idx]
                
                # Expand previous tokens to current scale size
                # Calculate actual spatial sizes from token sequence lengths
                prev_seq_len = prev_tokens.size(1)
                prev_spatial_size = int(prev_seq_len ** 0.5)
                
                # Calculate current spatial size from patch_nums
                current_spatial_size = self.patch_nums[scale_idx]
                
                if current_spatial_size > prev_spatial_size:
                    # Upsample previous tokens
                    prev_tokens_2d = prev_tokens.view(batch_size, prev_spatial_size, prev_spatial_size, -1)
                    prev_tokens_2d = prev_tokens_2d.permute(0, 3, 1, 2)  # (B, D, H, W)
                    upsampled_tokens = F.interpolate(prev_tokens_2d, 
                                                   size=(current_spatial_size, current_spatial_size),
                                                   mode='bilinear', 
                                                   align_corners=False)
                    current_tokens = upsampled_tokens.permute(0, 2, 3, 1).flatten(1, 2)  # (B, H*W, D)
                else:
                    current_tokens = prev_tokens
            
            # Add positional and scale embeddings
            pos_emb = self.pos_embeddings[scale_idx]
            scale_emb = self.scale_embeddings(torch.tensor(scale_idx, device=device))

            
            # Ensure pos_emb matches current_tokens sequence length
            if pos_emb.size(1) != current_tokens.size(1):
                # Interpolate position embeddings to match current sequence length
                seq_len = current_tokens.size(1)
                spatial_size = int(seq_len ** 0.5)
                pos_emb_2d = pos_emb.view(1, int(pos_emb.size(1) ** 0.5), int(pos_emb.size(1) ** 0.5), -1)
                pos_emb_2d = pos_emb_2d.permute(0, 3, 1, 2)  # (1, D, H, W)
                pos_emb_resized = F.interpolate(pos_emb_2d, size=(spatial_size, spatial_size), mode='bilinear', align_corners=False)
                pos_emb = pos_emb_resized.permute(0, 2, 3, 1).flatten(1, 2)  # (1, H*W, D)
            
            current_tokens = current_tokens + pos_emb + scale_emb.unsqueeze(0).unsqueeze(0)
            
            # Apply transformer layers
            for layer in self.transformer_layers:
                current_tokens = layer(current_tokens)
            
            # Generate outputs for current scale
            logits = self.output_heads[scale_idx](current_tokens)  # (B, H*W, vocab_size)
            
            # Emitter-specific predictions
            count_logits = self.emitter_count_head(current_tokens.mean(dim=1))  # Global pooling
            loc_preds = self.emitter_loc_head(current_tokens)  # (B, H*W, 2)
            uncertainty = self.uncertainty_head(current_tokens)  # (B, H*W, 1)
            
            # Convert logits to probability maps
            prob_map = F.softmax(logits, dim=-1)
            # Use the "emitter present" probability (assuming last token is emitter)
            # Calculate actual spatial size from current token sequence length
            seq_len = current_tokens.size(1)
            spatial_size = int(seq_len ** 0.5)
            emitter_prob = prob_map[:, :, -1].view(batch_size, 1, spatial_size, spatial_size)
            
            scale_outputs[f'scale_{scale_idx}'] = {
                'logits': logits,
                'prob_map': emitter_prob,
                'count_pred': count_logits,
                'loc_pred': loc_preds.view(batch_size, spatial_size, spatial_size, 2),
                'uncertainty': uncertainty.view(batch_size, spatial_size, spatial_size, 1),
                'tokens': current_tokens
            }
        
        return scale_outputs


class TransformerBlock(nn.Module):
    """
    Standard Transformer block with multi-head attention and MLP
    """
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class VAREmitterPredictor(nn.Module):
    """
    Complete VAR-based Emitter Prediction System
    """
    
    def __init__(self,
                 input_size: int = 40,  # Low-res input size
                 target_sizes: List[int] = [40, 80, 160, 320],  # Progressive target sizes
                 base_channels: int = 64,
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 12):
        super().__init__()
        
        self.input_size = input_size
        self.target_sizes = target_sizes
        
        # Calculate patch numbers for each scale based on encoder output
        # The encoder reduces spatial dimensions by factor of 4 (2 stride-2 convolutions)
        # Handle both [size1, size2, ...] and [[h1, w1], [h2, w2], ...] formats
        if isinstance(target_sizes[0], (list, tuple)):
            sizes = [size[0] for size in target_sizes]  # Use height (assuming square)
        else:
            sizes = target_sizes
        patch_nums = tuple(size // 4 for size in sizes)  # Encoder reduces by factor of 4
        
        # Initialize components
        self.vqvae = MultiScaleVQVAE(
            input_channels=1,
            base_channels=base_channels,
            num_scales=len(target_sizes)
        )
        
        self.transformer = ProgressiveEmitterTransformer(
            vqvae=self.vqvae,
            patch_nums=patch_nums,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
    
    def forward(self, x: torch.Tensor, target_scale: int = -1) -> Dict[str, torch.Tensor]:
        """
        Forward pass for emitter prediction
        
        Args:
            x: Input tensor (B, 1, 40, 40)
            target_scale: Target resolution scale
        
        Returns:
            Dictionary containing predictions at different scales
        """
        return self.transformer(x, target_scale)
    
    def predict_super_resolution(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict at highest resolution (super-resolution mode)
        
        Args:
            x: Low-resolution input (B, 1, 40, 40)
        
        Returns:
            High-resolution predictions
        """
        outputs = self.forward(x, target_scale=-1)
        highest_scale = f'scale_{len(self.target_sizes) - 1}'
        return outputs[highest_scale]