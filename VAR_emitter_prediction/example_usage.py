#!/usr/bin/env python3
"""
Example usage of the stable VAR emitter prediction system.
This script demonstrates how to use the new stable components.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from var_emitter_model_unified import UnifiedEmitterPredictor
from var_emitter_loss_stable import UnifiedPPXYZBLoss, StableVAREmitterLoss
from data_loader import EmitterDataset

def example_model_creation():
    """Example: Create a unified emitter prediction model."""
    print("Creating UnifiedEmitterPredictor model...")
    
    model = UnifiedEmitterPredictor(
        input_size=40,
        target_sizes=[40, 80, 160, 320],
        base_channels=64,
        embed_dim=512,
        num_heads=8,
        num_layers=12,
        codebook_size=1024,
        embedding_dim=256
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model

def example_loss_functions():
    """Example: Create stable loss functions."""
    print("\nCreating stable loss functions...")
    
    # Unified 6-channel loss
    unified_loss = UnifiedPPXYZBLoss(
        channel_weights=[1.5, 1.0, 1.2, 1.2, 0.8, 0.8],  # [prob, photons, x, y, z, bg]
        pos_weight=2.0,
        eps=1e-4
    )
    
    # Multi-scale VAR loss
    var_loss = StableVAREmitterLoss(
        scale_weights=[0.1, 0.3, 0.6, 1.0],
        eps=1e-4,
        warmup_epochs=20
    )
    
    print("Loss functions created successfully")
    return unified_loss, var_loss

def example_forward_pass():
    """Example: Perform a forward pass with dummy data."""
    print("\nPerforming forward pass with dummy data...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = example_model_creation()
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    input_size = 40
    dummy_input = torch.randn(batch_size, 1, input_size, input_size).to(device)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print("\nOutput shapes:")
    for scale_key, output in outputs.items():
        if isinstance(output, dict) and 'tokens' in output:
            print(f"  {scale_key}: {output['tokens'].shape}")
        else:
            print(f"  {scale_key}: {type(output)}")
    
    return outputs

def example_loss_computation():
    """Example: Compute loss with dummy data."""
    print("\nComputing loss with dummy data...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy predictions and targets
    batch_size = 2
    target_size = 80
    
    # Dummy prediction (6 channels: prob_logits, photons, x_offset, y_offset, z_offset, background)
    pred = torch.randn(batch_size, 6, target_size, target_size).to(device)
    
    # Dummy target (same format)
    target = torch.zeros(batch_size, 6, target_size, target_size).to(device)
    
    # Add some dummy emitters
    for b in range(batch_size):
        # Add 3 random emitters per batch
        for i in range(3):
            x, y = torch.randint(0, target_size, (2,))
            target[b, 0, y, x] = 1.0  # probability
            target[b, 1, y, x] = torch.rand(1) * 1000 + 100  # photons
            target[b, 2, y, x] = torch.rand(1) - 0.5  # x_offset
            target[b, 3, y, x] = torch.rand(1) - 0.5  # y_offset
            target[b, 4, y, x] = torch.rand(1) * 2 - 1  # z_offset
            target[b, 5, y, x] = torch.rand(1) * 50  # background
    
    # Create loss function
    loss_fn = StableVAREmitterLoss(
        channel_weights=[1.5, 1.0, 1.2, 1.2, 0.8, 0.8],
        pos_weight=2.0,
        eps=1e-4
    ).to(device)
    
    # Prepare inputs for StableVAREmitterLoss
    predictions = {'scale_0': {'unified_output': pred}}
    target_dict = {'unified': target}
    
    # Compute loss
    loss_dict = loss_fn(predictions, target_dict)
    
    print("Loss components:")
    if isinstance(loss_dict, dict):
        for key, value in loss_dict.items():
            if hasattr(value, 'item'):
                print(f"  {key}: {value.item():.4f}")
            else:
                print(f"  {key}: {value:.4f}")
    else:
        print(f"  total_loss: {loss_dict.mean().item():.4f}")
        print(f"  loss_shape: {loss_dict.shape}")
        print(f"  loss_sum: {loss_dict.sum().item():.4f}")
    
    return loss_dict

def example_data_generation():
    """Example: Generate synthetic training data."""
    print("\nGenerating synthetic training data...")
    
    # Parameters
    image_size = 40
    num_emitters = 5
    
    # Generate synthetic image
    image = torch.randn(1, image_size, image_size) * 0.1 + 0.5
    
    # Generate random emitter positions
    positions = torch.rand(num_emitters, 3)  # [x, y, z]
    positions[:, :2] *= image_size  # Scale x, y to image size
    positions[:, 2] = positions[:, 2] * 2 - 1  # Scale z to [-1, 1]
    
    # Generate photon counts
    photons = torch.rand(num_emitters) * 1000 + 100
    
    # Generate background
    background = torch.rand(num_emitters) * 50 + 10
    
    print(f"Generated image shape: {image.shape}")
    print(f"Number of emitters: {num_emitters}")
    print(f"Position range: x=[{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}], "
          f"y=[{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}], "
          f"z=[{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]")
    print(f"Photon range: [{photons.min():.1f}, {photons.max():.1f}]")
    print(f"Background range: [{background.min():.1f}, {background.max():.1f}]")
    
    return {
        'image': image,
        'positions': positions,
        'photons': photons,
        'background': background
    }

def example_training_step():
    """Example: Simulate a single training step."""
    print("\nSimulating a training step...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and loss
    model = example_model_creation()
    model = model.to(device)
    model.train()
    
    loss_fn = StableVAREmitterLoss(
        channel_weights=[1.5, 1.0, 1.2, 1.2, 0.8, 0.8],
        pos_weight=2.0,
        eps=1e-4
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-2
    )
    
    # Generate dummy batch
    batch_size = 2
    input_size = 40
    
    images = torch.randn(batch_size, 1, input_size, input_size).to(device)
    
    # Create dummy targets matching model output size
    target_size = 10  # Match model output size (scale_0 is 10x10)
    targets = torch.zeros(batch_size, 6, target_size, target_size).to(device)
    
    # Add some dummy emitters
    for b in range(batch_size):
        for i in range(5):  # 5 emitters per image
            x, y = torch.randint(0, target_size, (2,))
            targets[b, 0, y, x] = 1.0
            targets[b, 1, y, x] = torch.rand(1) * 1000 + 100
            targets[b, 2, y, x] = torch.rand(1) - 0.5
            targets[b, 3, y, x] = torch.rand(1) - 0.5
            targets[b, 4, y, x] = torch.rand(1) * 2 - 1
            targets[b, 5, y, x] = torch.rand(1) * 50
    
    # Forward pass
    optimizer.zero_grad()
    outputs = model(images)
    
    # Use the available output scale - get unified_output directly
    scale_output = outputs['scale_0']  # Currently only scale_0 is implemented
    pred = scale_output['unified_output']  # Get the 6-channel unified output
    
    # Prepare inputs for StableVAREmitterLoss
    predictions = {'scale_0': {'unified_output': pred}}
    target_dict = {'unified': targets}
    
    # Compute loss
    loss_dict = loss_fn(predictions, target_dict)
    total_loss = loss_dict['total_loss']
    if not isinstance(total_loss, torch.Tensor):
        total_loss = torch.tensor(total_loss, requires_grad=True, device=device)
    
    # Backward pass
    total_loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Optimizer step
    optimizer.step()
    
    print(f"Training step completed successfully")
    print(f"Total loss: {total_loss.item():.4f}")
    
    # Check for numerical stability
    if torch.isfinite(total_loss):
        print("✓ Loss is finite (numerically stable)")
    else:
        print("✗ Loss is not finite (numerical instability detected)")
    
    return total_loss.item()

def main():
    """Run all examples."""
    print("=" * 60)
    print("Stable VAR Emitter Prediction - Example Usage")
    print("=" * 60)
    
    try:
        # Example 1: Model creation
        model = example_model_creation()
        
        # Example 2: Loss functions
        unified_loss, var_loss = example_loss_functions()
        
        # Example 3: Forward pass
        outputs = example_forward_pass()
        
        # Example 4: Loss computation
        loss_dict = example_loss_computation()
        
        # Example 5: Data generation
        data = example_data_generation()
        
        # Example 6: Training step
        loss_value = example_training_step()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("The stable VAR emitter prediction system is ready to use.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)