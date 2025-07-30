#!/usr/bin/env python3
"""
Test script to verify count loss functionality
"""

import torch
import numpy as np
from var_emitter_loss import CountLoss
from var_emitter_model_true import VAREmitterLoss

def test_count_loss():
    """Test CountLoss functionality"""
    print("Testing CountLoss...")
    
    # Create test data
    batch_size = 2
    height, width = 40, 40
    
    # Create probability maps with known sums
    prob_map1 = torch.zeros(1, 1, height, width)
    prob_map1[0, 0, 10:15, 10:15] = 0.8  # Should sum to ~20
    prob_map1[0, 0, 20:25, 20:25] = 0.6  # Should sum to ~15
    # Total expected: ~35
    
    prob_map2 = torch.zeros(1, 1, height, width)
    prob_map2[0, 0, 5:10, 5:10] = 0.9   # Should sum to ~22.5
    # Total expected: ~22.5
    
    prob_maps = torch.cat([prob_map1, prob_map2], dim=0)  # (2, 1, H, W)
    
    # True counts
    true_counts = torch.tensor([35.0, 22.5])  # (2,)
    
    # Test CountLoss
    count_loss = CountLoss()
    loss = count_loss(prob_maps, true_counts)
    
    print(f"Probability map 1 sum: {prob_map1.sum().item():.2f}")
    print(f"Probability map 2 sum: {prob_map2.sum().item():.2f}")
    print(f"True counts: {true_counts.tolist()}")
    print(f"Count loss: {loss.item():.4f}")
    
    return loss

def test_var_emitter_loss():
    """Test VAREmitterLoss with count loss"""
    print("\nTesting VAREmitterLoss with count loss...")
    
    # Create test predictions
    batch_size = 2
    predictions = {
        'scale_0': {
            'prob_map': torch.rand(batch_size, 1, 80, 80) * 0.1,  # Low probability
            'loc_map': torch.rand(batch_size, 2, 80, 80),
            'vq_loss': torch.tensor(0.1)
        }
    }
    
    # Create test targets
    targets = {
        'scale_1': {
            'prob_map': torch.rand(batch_size, 1, 80, 80) * 0.1,
            'loc_map': torch.rand(batch_size, 2, 80, 80)
        },
        'emitter_count': [15, 25]  # Different counts for each batch item
    }
    
    # Test VAREmitterLoss
    var_loss = VAREmitterLoss(
        prob_weight=1.0,
        loc_weight=1.0,
        vq_weight=0.1,
        count_weight=1.0
    )
    
    losses = var_loss(predictions, targets)
    
    print(f"Total loss: {losses['total'].item():.4f}")
    print(f"Scale 0 count loss: {losses['scale_0_count'].item():.4f}")
    print(f"Scale 0 prob loss: {losses['scale_0_prob'].item():.4f}")
    print(f"Scale 0 loc loss: {losses['scale_0_loc'].item():.4f}")
    print(f"Scale 0 vq loss: {losses['scale_0_vq'].item():.4f}")
    
    return losses

def test_realistic_scenario():
    """Test with more realistic emitter scenario"""
    print("\nTesting realistic emitter scenario...")
    
    batch_size = 1
    height, width = 80, 80
    
    # Create probability map with sparse emitters
    prob_map = torch.zeros(batch_size, 1, height, width)
    
    # Add 5 emitters at different locations
    emitter_locations = [(20, 20), (30, 50), (60, 30), (70, 70), (10, 60)]
    for x, y in emitter_locations:
        # Create Gaussian-like probability around each emitter
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if 0 <= x+dx < width and 0 <= y+dy < height:
                    distance = np.sqrt(dx**2 + dy**2)
                    prob_map[0, 0, y+dy, x+dx] = max(0, 0.8 * np.exp(-distance**2 / 2))
    
    true_count = torch.tensor([5.0])  # 5 emitters
    
    count_loss = CountLoss()
    loss = count_loss(prob_map, true_count)
    
    print(f"Probability map sum: {prob_map.sum().item():.2f}")
    print(f"True count: {true_count.item()}")
    print(f"Count loss: {loss.item():.4f}")
    
    return loss

if __name__ == "__main__":
    print("=" * 50)
    print("Count Loss Testing")
    print("=" * 50)
    
    # Test individual CountLoss
    test_count_loss()
    
    # Test VAREmitterLoss with count loss
    test_var_emitter_loss()
    
    # Test realistic scenario
    test_realistic_scenario()
    
    print("\n=" * 50)
    print("All tests completed!")
    print("=" * 50)