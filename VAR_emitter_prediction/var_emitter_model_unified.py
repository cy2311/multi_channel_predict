import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np


class UnifiedEmitterPredictor(nn.Module):
    """
    Unified VAR-based emitter predictor with 6-channel output
    Following DECODE standard: [prob_logits, photons, x_offset, y_offset, z_offset, background]
    """
    
    def __init__(self,
                 input_size: int = 40,
                 target_sizes: List[int] = [40, 80, 160, 320],
                 base_channels: int = 64,
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 12,
                 codebook_size: int = 1024,
                 embedding_dim: int = 256):
        super().__init__()
        
        self.input_size = input_size
        self.target_sizes = target_sizes
        self.num_scales = len(target_sizes)
        self.embed_dim = embed_dim
        
        # Multi-scale VQVAE backbone
        self.vqvae = MultiScaleVQVAE(
            input_channels=1,
            base_channels=base_channels,
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
            num_scales=self.num_scales
        )
        
        # Progressive transformer
        self.transformer = ProgressiveEmitterTransformer(
            vqvae=self.vqvae,
            patch_nums=tuple(s // 8 for s in target_sizes),  # Adjust patch numbers
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # Unified 6-channel output heads for each scale
        self.unified_heads = nn.ModuleList([
            UnifiedOutputHead(embed_dim, target_size)
            for target_size in target_sizes
        ])
        
    def forward(self, x: torch.Tensor, target_scale: int = -1) -> Dict[str, torch.Tensor]:
        """
        Forward pass with unified 6-channel output
        
        Args:
            x: Input tensor (B, 1, H, W)
            target_scale: Target scale (-1 for all scales)
        
        Returns:
            Dictionary with unified outputs for each scale
        """
        # Get transformer features
        transformer_outputs = self.transformer(x, target_scale)
        
        unified_outputs = {}
        
        for scale_key, scale_features in transformer_outputs.items():
            if 'tokens' not in scale_features:
                continue
                
            scale_idx = int(scale_key.split('_')[1])
            tokens = scale_features['tokens']  # (B, H*W, D)
            
            # Generate unified 6-channel output
            unified_output = self.unified_heads[scale_idx](tokens)
            
            unified_outputs[scale_key] = {
                'unified_output': unified_output,  # (B, 6, H, W)
                'prob_logits': unified_output[:, 0:1],     # Probability logits
                'photons': unified_output[:, 1:2],         # Photon count
                'x_offset': unified_output[:, 2:3],        # X offset
                'y_offset': unified_output[:, 3:4],        # Y offset
                'z_offset': unified_output[:, 4:5],        # Z offset
                'background': unified_output[:, 5:6],      # Background
                'tokens': tokens
            }
            
            # Add derived outputs for compatibility
            unified_outputs[scale_key]['prob_map'] = torch.sigmoid(unified_output[:, 0:1])
            unified_outputs[scale_key]['count_pred'] = unified_output[:, 1:2].mean(dim=(2, 3))
            
        return unified_outputs
    
    def predict_super_resolution(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict at highest resolution
        """
        outputs = self.forward(x, target_scale=-1)
        highest_scale = f'scale_{self.num_scales - 1}'
        return outputs[highest_scale]


class UnifiedOutputHead(nn.Module):
    """
    Unified output head for 6-channel prediction
    """
    
    def __init__(self, embed_dim: int, target_size: int):
        super().__init__()
        
        self.target_size = target_size
        self.embed_dim = embed_dim
        
        # Feature processing layers
        self.feature_norm = nn.LayerNorm(embed_dim)
        self.feature_proj = nn.Linear(embed_dim, embed_dim)
        
        # Channel-specific heads
        self.prob_head = nn.Linear(embed_dim, 1)      # Probability logits
        self.photon_head = nn.Linear(embed_dim, 1)    # Photon count
        self.x_head = nn.Linear(embed_dim, 1)         # X offset
        self.y_head = nn.Linear(embed_dim, 1)         # Y offset
        self.z_head = nn.Linear(embed_dim, 1)         # Z offset
        self.bg_head = nn.Linear(embed_dim, 1)        # Background
        
        # Initialize heads with appropriate ranges
        self._initialize_heads()
    
    def _initialize_heads(self):
        """Initialize output heads with appropriate ranges"""
        # Probability head: neutral initialization
        nn.init.zeros_(self.prob_head.bias)
        
        # Photon head: positive bias for typical photon counts
        nn.init.constant_(self.photon_head.bias, 1000.0)
        
        # Location heads: small initialization for offsets
        for head in [self.x_head, self.y_head, self.z_head]:
            nn.init.normal_(head.weight, 0, 0.01)
            nn.init.zeros_(head.bias)
        
        # Background head: small positive bias
        nn.init.constant_(self.bg_head.bias, 100.0)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Generate 6-channel output from tokens
        
        Args:
            tokens: (B, H*W, D) token features
        
        Returns:
            (B, 6, H, W) unified output
        """
        B, seq_len, D = tokens.shape
        H = W = int(seq_len ** 0.5)
        
        # Process features
        features = self.feature_norm(tokens)
        features = F.relu(self.feature_proj(features))
        
        # Generate channel outputs
        prob_logits = self.prob_head(features)    # (B, H*W, 1)
        photons = self.photon_head(features)      # (B, H*W, 1)
        x_offset = self.x_head(features)          # (B, H*W, 1)
        y_offset = self.y_head(features)          # (B, H*W, 1)
        z_offset = self.z_head(features)          # (B, H*W, 1)
        background = self.bg_head(features)       # (B, H*W, 1)
        
        # Apply appropriate activations
        photons = F.softplus(photons)             # Ensure positive photons
        background = F.softplus(background)       # Ensure positive background
        
        # Clamp offsets to reasonable ranges
        x_offset = torch.tanh(x_offset) * 0.5     # [-0.5, 0.5] pixel offset
        y_offset = torch.tanh(y_offset) * 0.5     # [-0.5, 0.5] pixel offset
        z_offset = torch.tanh(z_offset) * 200.0   # [-200, 200] nm z offset
        
        # Combine channels
        output = torch.cat([
            prob_logits, photons, x_offset, y_offset, z_offset, background
        ], dim=-1)  # (B, H*W, 6)
        
        # Reshape to spatial format
        output = output.view(B, H, W, 6).permute(0, 3, 1, 2)  # (B, 6, H, W)
        
        return output


class MultiScaleVQVAE(nn.Module):
    """
    Enhanced Multi-scale Vector Quantized VAE
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
        
        # Multi-scale encoders with improved architecture
        self.encoders = nn.ModuleList()
        for i in range(num_scales):
            scale_channels = base_channels * (2 ** min(i, 3))  # Cap channel growth
            encoder = self._build_encoder(input_channels, scale_channels, embedding_dim)
            self.encoders.append(encoder)
        
        # Vector quantization codebooks
        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, embedding_dim) 
            for _ in range(num_scales)
        ])
        
        # Multi-scale decoders
        self.decoders = nn.ModuleList()
        for i in range(num_scales):
            scale_channels = base_channels * (2 ** min(i, 3))
            decoder = self._build_decoder(embedding_dim, scale_channels)
            self.decoders.append(decoder)
    
    def _build_encoder(self, input_channels: int, scale_channels: int, embedding_dim: int) -> nn.Module:
        """Build encoder with residual connections"""
        return nn.Sequential(
            # Initial convolution
            nn.Conv2d(input_channels, scale_channels // 4, 3, 1, 1),
            nn.BatchNorm2d(scale_channels // 4),
            nn.ReLU(inplace=True),
            
            # Residual blocks
            ResidualBlock(scale_channels // 4, scale_channels // 2, stride=2),
            ResidualBlock(scale_channels // 2, scale_channels, stride=2),
            
            # Final projection
            nn.Conv2d(scale_channels, embedding_dim, 1),
            nn.BatchNorm2d(embedding_dim)
        )
    
    def _build_decoder(self, embedding_dim: int, scale_channels: int) -> nn.Module:
        """Build decoder with residual connections"""
        return nn.Sequential(
            # Initial projection
            nn.Conv2d(embedding_dim, scale_channels, 1),
            nn.BatchNorm2d(scale_channels),
            nn.ReLU(inplace=True),
            
            # Upsampling blocks
            nn.ConvTranspose2d(scale_channels, scale_channels // 2, 4, 2, 1),
            nn.BatchNorm2d(scale_channels // 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(scale_channels // 2, scale_channels // 4, 4, 2, 1),
            nn.BatchNorm2d(scale_channels // 4),
            nn.ReLU(inplace=True),
            
            # Final output
            nn.Conv2d(scale_channels // 4, 1, 3, 1, 1),
            nn.Sigmoid()
        )
    
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


class ResidualBlock(nn.Module):
    """
    Residual block with optional downsampling
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        return F.relu(out + residual)


class VectorQuantizer(nn.Module):
    """
    Improved Vector Quantization with exponential moving averages
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 commitment_cost: float = 0.25, decay: float = 0.99):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        
        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        # EMA parameters
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', self.embedding.weight.data.clone())
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: (B, D, H, W)
        
        Returns:
            quantized: (B, D, H, W)
            indices: (B, H, W)
        """
        # Flatten input
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)  # (B*H*W, D)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Get closest embedding indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # (B*H*W, 1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Update EMA during training
        if self.training:
            self.cluster_size = self.decay * self.cluster_size + (1 - self.decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing
            n = torch.sum(self.cluster_size)
            self.cluster_size = (self.cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            
            dw = torch.matmul(encodings.t(), flat_input)
            self.embed_avg = self.decay * self.embed_avg + (1 - self.decay) * dw
            
            self.embedding.weight.data = self.embed_avg / self.cluster_size.unsqueeze(1)
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Return indices in spatial format
        indices = encoding_indices.view(input_shape[0], input_shape[2], input_shape[3])
        
        return quantized, indices


class ProgressiveEmitterTransformer(nn.Module):
    """
    Enhanced Progressive Transformer with improved stability
    """
    
    def __init__(self,
                 vqvae: MultiScaleVQVAE,
                 patch_nums: Tuple[int, ...] = (5, 10, 20, 40),
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
        
        # Token embeddings with improved initialization
        # Use VQ-VAE's embedding dimension, then project to transformer dimension
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(vqvae.codebook_size, vqvae.embedding_dim)
            for _ in range(self.num_scales)
        ])
        
        # Projection layers to match transformer embedding dimension
        self.token_projections = nn.ModuleList([
            nn.Linear(vqvae.embedding_dim, embed_dim)
            for _ in range(self.num_scales)
        ])
        
        # Learnable position embeddings
        # Note: encoder downsamples by 4x (2 stride-2 layers), so actual sequence length is (input_size // 4)^2
        self.pos_embeddings = nn.ParameterList()
        for patch_num in patch_nums:
            # Calculate actual sequence length after encoder downsampling
            actual_seq_len = (patch_num * 8 // 4) ** 2  # patch_num * 8 is input size, // 4 is downsampling
            pos_emb = nn.Parameter(torch.randn(1, actual_seq_len, embed_dim) * 0.02)
            self.pos_embeddings.append(pos_emb)
        
        # Scale embeddings
        self.scale_embeddings = nn.Embedding(self.num_scales, embed_dim)
        
        # Transformer layers with improved architecture
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Initialize embeddings
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embeddings with proper scaling"""
        for token_emb in self.token_embeddings:
            nn.init.normal_(token_emb.weight, 0, 0.02)
        
        for projection in self.token_projections:
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
        
        nn.init.normal_(self.scale_embeddings.weight, 0, 0.02)
    
    def forward(self, low_res_input: torch.Tensor, target_scale: int = -1) -> Dict[str, torch.Tensor]:
        """
        Progressive prediction with improved stability
        """
        batch_size = low_res_input.size(0)
        device = low_res_input.device
        
        if target_scale == -1:
            target_scale = self.num_scales - 1
        
        current_tokens = None
        scale_outputs = {}
        
        # Simplified: only process the target scale for now
        scale_idx = 0 if target_scale == -1 else min(target_scale, 0)
        
        # Initial encoding
        current_size = self.patch_nums[scale_idx]
        resized_input = F.interpolate(
            low_res_input, 
            size=(current_size * 8, current_size * 8),  # Account for encoder downsampling
            mode='bilinear', 
            align_corners=False
        )
        
        vqvae_output = self.vqvae(resized_input, scale_idx)
        indices = vqvae_output['indices']
        
        token_emb = self.token_embeddings[scale_idx](indices.flatten(1))
        current_tokens = self.token_projections[scale_idx](token_emb)
        
        # Add embeddings
        pos_emb = self.pos_embeddings[scale_idx]
        scale_emb = self.scale_embeddings(torch.tensor(scale_idx, device=device))
        
        # Expand scale embedding to match current_tokens shape
        scale_emb = scale_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, current_tokens.size(1), -1)
        
        current_tokens = current_tokens + pos_emb + scale_emb
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            current_tokens = layer(current_tokens)
        
        # Final normalization
        current_tokens = self.layer_norm(current_tokens)
        
        scale_outputs[f'scale_{scale_idx}'] = {
            'tokens': current_tokens
        }
        
        return scale_outputs


class TransformerBlock(nn.Module):
    """
    Enhanced Transformer block with improved stability
    """
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + self.dropout1(attn_out)
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x