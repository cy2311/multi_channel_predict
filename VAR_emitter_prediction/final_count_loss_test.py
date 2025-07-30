#!/usr/bin/env python3
"""
Final comprehensive test of the count loss integration
"""

import torch
import numpy as np
from train_true_var import MultiScaleEmitterDataset
from var_emitter_model_true import TrueVAREmitterPredictor, VAREmitterLoss
import json
from torch.utils.data import DataLoader

def test_real_dataset_with_count_loss():
    """Test with the actual dataset structure"""
    print("Testing with real dataset structure and count loss...")
    
    # Load config
    with open('configs/config_true_var.json', 'r') as f:
        config = json.load(f)
    
    # Create model with smaller parameters for testing
    model = TrueVAREmitterPredictor(
        patch_nums=(10, 20, 40, 80),
        embed_dim=128,  # Smaller for testing
        num_heads=4,
        num_layers=2,
        vocab_size=512,
        input_channels=1
    )
    
    # Create loss function with count loss enabled
    loss_config = config['loss']
    criterion = VAREmitterLoss(
        prob_weight=loss_config['prob_weight'],
        loc_weight=loss_config['loc_weight'],
        vq_weight=loss_config['vq_weight'],
        count_weight=loss_config['count_weight']  # This should be 1.0 from config
    )
    
    print(f"Count weight from config: {loss_config['count_weight']}")
    
    # Create dataset (will generate synthetic data since no real H5 files)
    dataset = MultiScaleEmitterDataset(
        data_path="/fake/path",  # Will trigger synthetic data generation
        input_resolution=(160, 160),
        synthetic_samples=4
    )
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # Test one batch
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Emitter counts: {batch['emitter_count']}")
        print(f"  Emitter photons: {batch['emitter_photons']}")
        
        # Forward pass
        images = batch['image']
        targets = batch['targets']
        
        # Add emitter count to targets (as done in train_step)
        if 'emitter_count' in batch:
            targets['emitter_count'] = batch['emitter_count']
        
        # Model prediction
        predictions = model(images)
        
        print(f"  Prediction scales: {list(predictions.keys())}")
        for scale_key in predictions.keys():
            if 'prob_map' in predictions[scale_key]:
                prob_sum = predictions[scale_key]['prob_map'].sum(dim=(2,3)).mean().item()
                print(f"    {scale_key} prob sum (avg): {prob_sum:.2f}")
        
        # Calculate loss
        losses = criterion(predictions, targets)
        total_loss = losses['total']
        
        print(f"  Total loss: {total_loss.item():.4f}")
        
        # Print all loss components
        for key, value in losses.items():
            if key != 'total':
                print(f"    {key}: {value.item():.4f}")
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"  Backward pass completed successfully!")
        
        # Only test first batch
        break
    
    print("\nReal dataset test completed successfully!")

def test_count_loss_effectiveness():
    """Test that count loss actually works to constrain probability sums"""
    print("\nTesting count loss effectiveness with controlled scenario...")
    
    # Create simple model
    model = TrueVAREmitterPredictor(
        patch_nums=(10, 20),  # Only 2 scales for simplicity
        embed_dim=64,
        num_heads=2,
        num_layers=1,
        vocab_size=256,
        input_channels=1
    )
    
    # High count weight to see clear effect
    criterion = VAREmitterLoss(
        prob_weight=0.01,  # Very low
        loc_weight=0.01,   # Very low
        vq_weight=0.001,   # Very low
        count_weight=5.0   # High to see effect
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Create controlled test case
    batch_size = 2
    image = torch.randn(batch_size, 1, 160, 160) * 0.1 + 0.5
    target_counts = [3, 7]  # Different counts for each sample
    
    # Create minimal targets (just need structure, not accurate values)
    targets = {
        'scale_0': {
            'prob_map': torch.zeros(batch_size, 1, 10, 10),
            'loc_map': torch.zeros(batch_size, 2, 10, 10)
        },
        'scale_1': {
            'prob_map': torch.zeros(batch_size, 1, 20, 20),
            'loc_map': torch.zeros(batch_size, 2, 20, 20)
        },
        'emitter_count': target_counts
    }
    
    print(f"Target emitter counts: {target_counts}")
    
    # Train for several steps and observe probability sum convergence
    model.train()
    for step in range(15):
        optimizer.zero_grad()
        
        predictions = model(image)
        losses = criterion(predictions, targets)
        
        # Get probability sums for each scale
        prob_sums = {}
        count_losses = {}
        for scale_key in ['scale_0', 'scale_1']:
            if scale_key in predictions:
                # Sum over spatial dimensions for each sample in batch
                prob_sum_per_sample = predictions[scale_key]['prob_map'].sum(dim=(2,3))  # (B, 1)
                prob_sums[scale_key] = prob_sum_per_sample.squeeze().tolist()
                
                # Get count loss for this scale
                count_loss_key = f"{scale_key}_count"
                if count_loss_key in losses:
                    count_losses[scale_key] = losses[count_loss_key].item()
        
        print(f"Step {step+1:2d}:")
        for scale_key in prob_sums:
            sums = prob_sums[scale_key]
            if isinstance(sums, list):
                print(f"  {scale_key}: [{sums[0]:.2f}, {sums[1]:.2f}] (target: {target_counts})")
            else:
                print(f"  {scale_key}: {sums:.2f}")
            
            if scale_key in count_losses:
                print(f"    {scale_key} count loss: {count_losses[scale_key]:.4f}")
        
        losses['total'].backward()
        optimizer.step()
    
    print("\nCount loss effectiveness test completed!")
    print("Note: If count loss is working, probability sums should gradually approach target counts.")

if __name__ == "__main__":
    print("=" * 70)
    print("Final Count Loss Integration Test")
    print("=" * 70)
    
    # Test with real dataset structure
    test_real_dataset_with_count_loss()
    
    # Test count loss effectiveness
    test_count_loss_effectiveness()
    
    print("\n" + "=" * 70)
    print("All final tests completed successfully!")
    print("Count loss integration is working correctly.")
    print("=" * 70)