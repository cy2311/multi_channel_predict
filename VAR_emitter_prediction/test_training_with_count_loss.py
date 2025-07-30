#!/usr/bin/env python3
"""
Test training with count loss to verify the complete pipeline
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from var_emitter_model_true import TrueVAREmitterPredictor, VAREmitterLoss
import json

class MockEmitterDataset(Dataset):
    """Mock dataset for testing"""
    
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
        self.input_resolution = (160, 160)
        # Match the patch_nums from the model: (10, 20, 40, 80)
        self.target_resolutions = {
            'scale_0': (10, 10),
            'scale_1': (20, 20),
            'scale_2': (40, 40), 
            'scale_3': (80, 80)
        }
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random image
        image = torch.randn(1, *self.input_resolution) * 0.1 + 0.5
        
        # Generate random emitter count and positions
        emitter_count = np.random.randint(3, 15)
        emitter_positions = np.random.rand(emitter_count, 2)
        
        # Generate targets for each scale
        targets = {}
        for scale_key, (h, w) in self.target_resolutions.items():
            prob_map, loc_map = self._generate_target_maps(emitter_positions, h, w)
            targets[scale_key] = {
                'prob_map': prob_map,
                'loc_map': loc_map
            }
        
        # Pad emitter positions to fixed size for batching
        max_emitters = 20
        padded_positions = np.zeros((max_emitters, 2))
        if len(emitter_positions) > 0:
            num_to_copy = min(len(emitter_positions), max_emitters)
            padded_positions[:num_to_copy] = emitter_positions[:num_to_copy]
        
        return {
            'image': image,
            'targets': targets,
            'emitter_count': emitter_count,
            'emitter_positions': torch.from_numpy(padded_positions).float(),
            'actual_emitter_count': min(emitter_count, max_emitters)
        }
    
    def _generate_target_maps(self, emitter_positions, target_h, target_w):
        """Generate probability and location maps"""
        prob_map = torch.zeros(1, target_h, target_w)
        loc_map = torch.zeros(2, target_h, target_w)
        
        for pos in emitter_positions:
            # Convert normalized position to pixel coordinates
            x = int(pos[0] * target_w)
            y = int(pos[1] * target_h)
            
            # Ensure coordinates are within bounds
            x = max(0, min(x, target_w - 1))
            y = max(0, min(y, target_h - 1))
            
            # Set probability (with some Gaussian spread)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < target_w and 0 <= ny < target_h:
                        distance = np.sqrt(dx**2 + dy**2)
                        prob_map[0, ny, nx] = max(prob_map[0, ny, nx], 0.8 * np.exp(-distance**2 / 2))
                        
                        # Set location offset
                        loc_map[0, ny, nx] = pos[0] * target_w - nx  # x offset
                        loc_map[1, ny, nx] = pos[1] * target_h - ny  # y offset
        
        return prob_map, loc_map

def test_training_step():
    """Test a single training step with count loss"""
    print("Testing training step with count loss...")
    
    # Create model
    model = TrueVAREmitterPredictor(
        patch_nums=(10, 20, 40, 80),
        embed_dim=128,  # Smaller for testing
        num_heads=4,
        num_layers=2,
        vocab_size=512,
        input_channels=1
    )
    
    # Create loss function with count loss
    criterion = VAREmitterLoss(
        prob_weight=1.0,
        loc_weight=1.0,
        vq_weight=0.1,
        count_weight=1.0  # Enable count loss
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create dataset and dataloader
    dataset = MockEmitterDataset(num_samples=4)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Training step
    model.train()
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Emitter counts: {batch['emitter_count'].tolist()}")
        
        # Forward pass
        images = batch['image']
        targets = batch['targets']
        
        # Add emitter count to targets
        targets['emitter_count'] = batch['emitter_count']
        
        # Model prediction
        predictions = model(images)
        
        print(f"  Prediction scales: {list(predictions.keys())}")
        
        # Calculate loss
        losses = criterion(predictions, targets)
        total_loss = losses['total']
        
        print(f"  Total loss: {total_loss.item():.4f}")
        
        # Print individual loss components
        for key, value in losses.items():
            if 'count' in key:
                print(f"  {key}: {value.item():.4f}")
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"  Backward pass completed successfully!")
        
        # Only test first batch
        break
    
    print("\nTraining step test completed successfully!")

def test_count_constraint():
    """Test that count loss actually constrains probability sum"""
    print("\nTesting count constraint effectiveness...")
    
    # Create simple model for testing
    model = TrueVAREmitterPredictor(
        patch_nums=(20, 40),  # Only 2 scales for simplicity
        embed_dim=64,
        num_heads=2,
        num_layers=1,
        vocab_size=256,
        input_channels=1
    )
    
    criterion = VAREmitterLoss(
        prob_weight=0.1,  # Lower prob weight
        loc_weight=0.1,   # Lower loc weight
        vq_weight=0.01,   # Lower vq weight
        count_weight=10.0  # High count weight to see effect
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Create fixed test case
    image = torch.randn(1, 1, 160, 160) * 0.1 + 0.5
    target_count = 5  # Fixed count
    
    # Create targets
    targets = {
        'scale_1': {
            'prob_map': torch.zeros(1, 1, 40, 40),
            'loc_map': torch.zeros(1, 2, 40, 40)
        },
        'emitter_count': [target_count]
    }
    
    print(f"Target emitter count: {target_count}")
    
    # Train for a few steps and observe probability sum
    model.train()
    for step in range(10):
        optimizer.zero_grad()
        
        predictions = model(image)
        losses = criterion(predictions, targets)
        
        # Get probability sum for scale_0 (maps to scale_1 target)
        if 'scale_0' in predictions:
            prob_sum = predictions['scale_0']['prob_map'].sum().item()
            count_loss = losses.get('scale_0_count', torch.tensor(0)).item()
            
            print(f"Step {step+1}: Prob sum = {prob_sum:.2f}, Count loss = {count_loss:.4f}")
        
        losses['total'].backward()
        optimizer.step()
    
    print("Count constraint test completed!")

if __name__ == "__main__":
    print("=" * 60)
    print("Training with Count Loss Test")
    print("=" * 60)
    
    # Test training step
    test_training_step()
    
    # Test count constraint
    test_count_constraint()
    
    print("\n" + "=" * 60)
    print("All training tests completed successfully!")
    print("=" * 60)