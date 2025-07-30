#!/usr/bin/env python3
"""
Demo script for TrueVAR Emitter Predictor
Demonstrates the core VAR functionality without requiring trained weights
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from var_emitter_model_true import TrueVAREmitterPredictor
import json


def create_demo_image(size=(160, 160), num_emitters=8):
    """
    Create a synthetic demo image with known emitter positions
    """
    image = np.random.normal(0, 0.05, size)  # Low noise background
    emitter_positions = []
    
    # Add bright spots (emitters)
    for i in range(num_emitters):
        # Random position with some margin
        x = np.random.randint(20, size[1] - 20)
        y = np.random.randint(20, size[0] - 20)
        
        # Add Gaussian spot
        sigma = np.random.uniform(2.0, 4.0)
        amplitude = np.random.uniform(0.7, 1.0)
        
        yy, xx = np.ogrid[:size[0], :size[1]]
        gaussian = amplitude * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
        image += gaussian
        
        emitter_positions.append([x, y])
    
    # Normalize to [0, 1]
    image = np.clip(image, 0, 1)
    
    return image, np.array(emitter_positions)


def demonstrate_progressive_prediction():
    """
    Demonstrate VAR's progressive multi-scale prediction
    """
    print("🔬 Demonstrating Progressive Multi-Scale Prediction")
    print("=" * 60)
    
    # Initialize model
    model = TrueVAREmitterPredictor(
        patch_nums=(10, 20, 40, 80),
        embed_dim=384,  # Smaller for demo
        num_heads=6,
        num_layers=12,
        dropout=0.0  # No dropout for consistent demo
    )
    model.eval()
    
    # Create demo image
    demo_image, gt_positions = create_demo_image()
    print(f"Created demo image: {demo_image.shape}")
    print(f"Ground truth emitters: {len(gt_positions)}")
    
    # Preprocess input
    input_tensor = torch.from_numpy(demo_image).float().unsqueeze(0).unsqueeze(0)
    input_tensor = (input_tensor - input_tensor.mean()) / (input_tensor.std() + 1e-8)
    
    # Progressive prediction
    print("\n🚀 Running progressive VAR prediction...")
    with torch.no_grad():
        outputs = model(input_tensor)
    
    # Visualize results
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # Show input
    axes[0, 0].imshow(demo_image, cmap='gray')
    axes[0, 0].set_title('Input Image\n160x160')
    axes[0, 0].scatter(gt_positions[:, 0], gt_positions[:, 1], c='red', s=30, marker='x')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    # Show predictions at each scale
    for i, (scale_key, output) in enumerate(outputs.items()):
        col = i + 1
        scale_idx = int(scale_key.split('_')[1])
        resolution = model.patch_nums[scale_idx]
        
        prob_map = output['prob_map'][0, 0].numpy()
        loc_map = output['loc_map'][0].numpy()
        
        # Probability map
        im1 = axes[0, col].imshow(prob_map, cmap='hot', vmin=0, vmax=1)
        axes[0, col].set_title(f'Scale {scale_idx}\nProb Map {resolution}x{resolution}')
        axes[0, col].axis('off')
        plt.colorbar(im1, ax=axes[0, col], fraction=0.046)
        
        # Extract detections
        threshold = 0.3  # Lower threshold for demo
        peaks = prob_map > threshold
        if peaks.sum() > 0:
            peak_coords = np.where(peaks)
            detections = []
            for y, x in zip(peak_coords[0], peak_coords[1]):
                # Sub-pixel position
                offset_x = (loc_map[0, y, x] + 1) / 2
                offset_y = (loc_map[1, y, x] + 1) / 2
                pos_x = x + offset_x
                pos_y = y + offset_y
                detections.append([pos_x, pos_y])
            
            detections = np.array(detections)
            
            # Show detections
            axes[1, col].imshow(prob_map, cmap='gray')
            axes[1, col].scatter(detections[:, 0], detections[:, 1], 
                               c='red', s=50, marker='x', linewidths=2)
            axes[1, col].set_title(f'Detections: {len(detections)}')
        else:
            axes[1, col].imshow(prob_map, cmap='gray')
            axes[1, col].set_title('No Detections')
        
        axes[1, col].axis('off')
        
        print(f"  Scale {scale_idx} ({resolution}x{resolution}): {peaks.sum()} detections")
    
    plt.tight_layout()
    plt.savefig('demo_progressive_prediction.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Progressive prediction demo completed!")
    print("📊 Visualization saved as 'demo_progressive_prediction.png'")


def demonstrate_super_resolution():
    """
    Demonstrate VAR's super-resolution capability: 40x40 -> 80x80
    """
    print("\n🔍 Demonstrating Super-Resolution Capability")
    print("=" * 60)
    
    # Initialize model
    model = TrueVAREmitterPredictor(
        patch_nums=(10, 20, 40, 80),
        embed_dim=384,
        num_heads=6,
        num_layers=12,
        dropout=0.0
    )
    model.eval()
    
    # Create high-res demo image
    high_res_image, gt_positions = create_demo_image((160, 160), num_emitters=6)
    
    # Downsample to 40x40 for input
    low_res_tensor = torch.from_numpy(high_res_image).float().unsqueeze(0).unsqueeze(0)
    low_res_tensor = F.interpolate(low_res_tensor, size=(40, 40), mode='bilinear', align_corners=False)
    low_res_image = low_res_tensor.squeeze().numpy()
    
    # Normalize
    low_res_tensor = (low_res_tensor - low_res_tensor.mean()) / (low_res_tensor.std() + 1e-8)
    
    print(f"Input: 40x40, Target: 80x80")
    print(f"Ground truth emitters: {len(gt_positions)}")
    
    # Super-resolution inference
    print("\n🚀 Running super-resolution inference...")
    with torch.no_grad():
        result = model.autoregressive_inference(low_res_tensor, target_resolution=80)
    
    # Extract results
    prob_map_80 = result['prob_map'][0, 0].numpy()
    loc_map_80 = result['loc_map'][0].numpy()
    
    # Extract detections
    threshold = 0.3
    peaks = prob_map_80 > threshold
    detections_80 = []
    if peaks.sum() > 0:
        peak_coords = np.where(peaks)
        for y, x in zip(peak_coords[0], peak_coords[1]):
            offset_x = (loc_map_80[0, y, x] + 1) / 2
            offset_y = (loc_map_80[1, y, x] + 1) / 2
            pos_x = x + offset_x
            pos_y = y + offset_y
            detections_80.append([pos_x, pos_y])
        detections_80 = np.array(detections_80)
    
    # Visualize super-resolution results
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original high-res
    axes[0].imshow(high_res_image, cmap='gray')
    axes[0].scatter(gt_positions[:, 0], gt_positions[:, 1], c='red', s=30, marker='x')
    axes[0].set_title('Original 160x160\n(Ground Truth)')
    axes[0].axis('off')
    
    # Low-res input
    axes[1].imshow(low_res_image, cmap='gray')
    axes[1].set_title('Input 40x40\n(Downsampled)')
    axes[1].axis('off')
    
    # Super-res probability map
    im2 = axes[2].imshow(prob_map_80, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('VAR Output 80x80\n(Probability Map)')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    # Super-res detections
    axes[3].imshow(prob_map_80, cmap='gray')
    if len(detections_80) > 0:
        axes[3].scatter(detections_80[:, 0], detections_80[:, 1], 
                       c='red', s=50, marker='x', linewidths=2)
    axes[3].set_title(f'Detections: {len(detections_80)}')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_super_resolution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Super-resolution demo completed!")
    print(f"📊 Input: 40x40 -> Output: 80x80")
    print(f"🎯 Detected {len(detections_80)} emitters at high resolution")
    print("📊 Visualization saved as 'demo_super_resolution.png'")


def demonstrate_var_architecture():
    """
    Demonstrate VAR's core architectural components
    """
    print("\n🏗️ Demonstrating VAR Architecture Components")
    print("=" * 60)
    
    # Initialize model
    model = TrueVAREmitterPredictor()
    
    print(f"📐 Model Architecture:")
    print(f"  - Patch numbers (scales): {model.patch_nums}")
    print(f"  - Embedding dimension: {model.embed_dim}")
    print(f"  - Number of scales: {model.num_scales}")
    print(f"  - Transformer layers: {len(model.blocks)}")
    print(f"  - Attention heads: {model.blocks[0].num_heads}")
    
    # Count parameters by component
    quantizer_params = sum(p.numel() for p in model.quantizer.parameters())
    transformer_params = sum(p.numel() for p in model.blocks.parameters())
    output_params = sum(p.numel() for p in model.output_heads.parameters()) + \
                   sum(p.numel() for p in model.emitter_prob_head.parameters()) + \
                   sum(p.numel() for p in model.emitter_loc_head.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n📊 Parameter Distribution:")
    print(f"  - Vector Quantizer: {quantizer_params:,} ({quantizer_params/total_params*100:.1f}%)")
    print(f"  - Transformer Blocks: {transformer_params:,} ({transformer_params/total_params*100:.1f}%)")
    print(f"  - Output Heads: {output_params:,} ({output_params/total_params*100:.1f}%)")
    print(f"  - Total: {total_params:,}")
    
    # Demonstrate residual accumulation
    print(f"\n🔄 VAR Residual Accumulation Mechanism:")
    device = torch.device('cpu')
    batch_size = 1
    
    # Initialize accumulated features
    f_hat = torch.zeros(batch_size, model.embed_dim, 80, 80)
    
    print(f"  Initial f_hat shape: {f_hat.shape}")
    
    # Simulate residual accumulation for each scale
    for si, pn in enumerate(model.patch_nums[:-1]):
        # Simulate features at current scale
        h_BChw = torch.randn(batch_size, model.embed_dim, pn, pn)
        
        # Apply VAR residual accumulation
        f_hat_new, next_input = model.quantizer.get_next_autoregressive_input(
            si, len(model.patch_nums), f_hat, h_BChw
        )
        
        print(f"  Scale {si} ({pn}x{pn}):")
        print(f"    - Input h_BChw: {h_BChw.shape}")
        print(f"    - Updated f_hat: {f_hat_new.shape}")
        print(f"    - Next input: {next_input.shape}")
        
        f_hat = f_hat_new
    
    print(f"\n✅ VAR architecture demonstration completed!")


def main():
    """
    Run all demonstrations
    """
    print("🎯 TrueVAR Emitter Predictor - Interactive Demo")
    print("=" * 60)
    print("This demo showcases the core VAR functionality:")
    print("1. Progressive multi-scale prediction (10x10 -> 20x20 -> 40x40 -> 80x80)")
    print("2. Super-resolution capability (40x40 -> 80x80)")
    print("3. VAR architecture components")
    print("\nNote: This demo uses untrained weights for illustration purposes.")
    print("Real performance requires proper training on emitter data.\n")
    
    try:
        # Demonstrate progressive prediction
        demonstrate_progressive_prediction()
        
        # Demonstrate super-resolution
        demonstrate_super_resolution()
        
        # Demonstrate architecture
        demonstrate_var_architecture()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("\n📝 Summary:")
        print("  ✅ Progressive multi-scale prediction working")
        print("  ✅ Super-resolution capability demonstrated")
        print("  ✅ VAR residual accumulation mechanism functional")
        print("  ✅ Model ready for training with real data")
        
        print("\n🚀 Next Steps:")
        print("  1. Prepare your emitter dataset")
        print("  2. Run: python train_true_var.py --config config_true_var.json")
        print("  3. Use trained model for inference: python inference_true_var.py")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()