#!/usr/bin/env python3
"""
Evaluation script for stable VAR emitter prediction model.
This script evaluates trained models on test datasets and computes metrics.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from var_emitter_model_unified import UnifiedEmitterPredictor
from data_loader import EmitterDataLoader

class EmitterEvaluator:
    """Evaluator for emitter prediction models."""
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
        
    def evaluate_dataset(self, data_loader, threshold: float = 0.5) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        all_metrics = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch_metrics = self._evaluate_batch(batch, threshold)
                all_metrics.append(batch_metrics)
                
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx}/{len(data_loader)} batches")
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        return aggregated_metrics
    
    def _evaluate_batch(self, batch: Dict[str, torch.Tensor], threshold: float) -> Dict[str, float]:
        """Evaluate a single batch."""
        # Move batch to device
        images = batch['image'].to(self.device)
        targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
        
        # Forward pass
        predictions = self.model(images)
        
        # Evaluate each scale
        scale_metrics = []
        for scale_idx, scale_key in enumerate(targets.keys()):
            if scale_key in predictions:
                pred = predictions[scale_key]
                target = targets[scale_key]
                
                metrics = self._compute_scale_metrics(pred, target, threshold)
                scale_metrics.append(metrics)
        
        # Average across scales
        if scale_metrics:
            batch_metrics = {}
            for key in scale_metrics[0].keys():
                batch_metrics[key] = np.mean([m[key] for m in scale_metrics])
            return batch_metrics
        else:
            return {}
    
    def _compute_scale_metrics(self, pred: torch.Tensor, target: torch.Tensor, 
                              threshold: float) -> Dict[str, float]:
        """Compute metrics for a single scale."""
        batch_size = pred.shape[0]
        metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'ap': 0.0,
            'localization_error': 0.0,
            'photon_error': 0.0,
            'background_error': 0.0
        }
        
        for b in range(batch_size):
            pred_b = pred[b]  # [6, H, W]
            target_b = target[b]  # [6, H, W]
            
            # Extract probability maps
            pred_prob = torch.sigmoid(pred_b[0])  # Apply sigmoid to logits
            target_prob = target_b[0]
            
            # Compute detection metrics
            pred_binary = (pred_prob > threshold).float()
            
            # Flatten for metric computation
            pred_flat = pred_prob.flatten().cpu().numpy()
            target_flat = target_prob.flatten().cpu().numpy()
            pred_binary_flat = pred_binary.flatten().cpu().numpy()
            
            # Precision, Recall, F1
            if target_flat.sum() > 0:  # Only if there are positive targets
                precision = np.sum(pred_binary_flat * target_flat) / (np.sum(pred_binary_flat) + 1e-8)
                recall = np.sum(pred_binary_flat * target_flat) / (np.sum(target_flat) + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                # Average Precision
                ap = average_precision_score(target_flat, pred_flat)
                
                metrics['precision'] += precision
                metrics['recall'] += recall
                metrics['f1'] += f1
                metrics['ap'] += ap
            
            # Localization error (only for detected emitters)
            if target_prob.sum() > 0:
                loc_error = self._compute_localization_error(pred_b, target_b, threshold)
                metrics['localization_error'] += loc_error
            
            # Photon and background errors
            photon_error = F.mse_loss(pred_b[1], target_b[1]).item()
            background_error = F.mse_loss(pred_b[5], target_b[5]).item()
            
            metrics['photon_error'] += photon_error
            metrics['background_error'] += background_error
        
        # Average across batch
        for key in metrics:
            metrics[key] /= batch_size
        
        return metrics
    
    def _compute_localization_error(self, pred: torch.Tensor, target: torch.Tensor, 
                                   threshold: float) -> float:
        """Compute localization error using Hungarian matching."""
        # Extract detected positions
        pred_prob = torch.sigmoid(pred[0])
        pred_mask = pred_prob > threshold
        target_mask = target[0] > 0.5
        
        if not pred_mask.any() or not target_mask.any():
            return 0.0
        
        # Get coordinates
        pred_coords = torch.nonzero(pred_mask, as_tuple=False).float()  # [N_pred, 2]
        target_coords = torch.nonzero(target_mask, as_tuple=False).float()  # [N_target, 2]
        
        # Add sub-pixel offsets
        for i, coord in enumerate(pred_coords):
            y, x = coord.long()
            pred_coords[i, 0] += pred[3, y, x]  # y_offset
            pred_coords[i, 1] += pred[2, y, x]  # x_offset
        
        for i, coord in enumerate(target_coords):
            y, x = coord.long()
            target_coords[i, 0] += target[3, y, x]  # y_offset
            target_coords[i, 1] += target[2, y, x]  # x_offset
        
        # Compute distance matrix
        dist_matrix = torch.cdist(pred_coords, target_coords)  # [N_pred, N_target]
        
        # Hungarian matching
        if dist_matrix.numel() > 0:
            cost_matrix = dist_matrix.cpu().numpy()
            pred_indices, target_indices = linear_sum_assignment(cost_matrix)
            
            if len(pred_indices) > 0:
                matched_distances = cost_matrix[pred_indices, target_indices]
                return np.mean(matched_distances)
        
        return 0.0
    
    def _aggregate_metrics(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across all batches."""
        if not all_metrics:
            return {}
        
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            if values:
                aggregated[key] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
        
        return aggregated
    
    def visualize_predictions(self, data_loader, num_samples: int = 5, 
                            save_path: str = None) -> None:
        """Visualize model predictions."""
        self.model.eval()
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        with torch.no_grad():
            for sample_idx, batch in enumerate(data_loader):
                if sample_idx >= num_samples:
                    break
                
                # Take first item from batch
                image = batch['image'][0:1].to(self.device)
                target = {k: v[0:1].to(self.device) for k, v in batch['targets'].items()}
                
                # Forward pass
                predictions = self.model(image)
                
                # Use the highest resolution scale for visualization
                scale_key = list(predictions.keys())[-1]
                pred = predictions[scale_key][0]  # [6, H, W]
                tgt = target[scale_key][0]  # [6, H, W]
                
                # Plot input image
                axes[sample_idx, 0].imshow(image[0, 0].cpu().numpy(), cmap='gray')
                axes[sample_idx, 0].set_title('Input Image')
                axes[sample_idx, 0].axis('off')
                
                # Plot target probability
                axes[sample_idx, 1].imshow(tgt[0].cpu().numpy(), cmap='hot')
                axes[sample_idx, 1].set_title('Target Probability')
                axes[sample_idx, 1].axis('off')
                
                # Plot predicted probability
                pred_prob = torch.sigmoid(pred[0]).cpu().numpy()
                axes[sample_idx, 2].imshow(pred_prob, cmap='hot')
                axes[sample_idx, 2].set_title('Predicted Probability')
                axes[sample_idx, 2].axis('off')
                
                # Plot photon prediction
                axes[sample_idx, 3].imshow(pred[1].cpu().numpy(), cmap='viridis')
                axes[sample_idx, 3].set_title('Predicted Photons')
                axes[sample_idx, 3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

def load_model(checkpoint_path: str, config: dict, device: torch.device) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    model = UnifiedEmitterPredictor(
        input_size=config['model']['input_size'],
        target_sizes=config['model']['target_sizes'],
        base_channels=config['model']['base_channels'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        codebook_size=config['model']['codebook_size'],
        embedding_dim=config['model']['embedding_dim']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate VAR emitter prediction model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Detection threshold')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load model
        print("Loading model...")
        model = load_model(args.checkpoint, config, device)
        print("Model loaded successfully")
        
        # Create data loader
        print("Creating data loader...")
        config['data']['val_path'] = args.data_path  # Override with test data path
        data_loader = EmitterDataLoader(config)
        test_loader = data_loader.get_val_loader()
        print(f"Test samples: {len(test_loader.dataset)}")
        
        # Create evaluator
        evaluator = EmitterEvaluator(model, device)
        
        # Evaluate model
        print("Evaluating model...")
        metrics = evaluator.evaluate_dataset(test_loader, args.threshold)
        
        # Print results
        print("\nEvaluation Results:")
        print("=" * 50)
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        # Save results
        results_path = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {results_path}")
        
        # Generate visualizations
        if args.visualize:
            print("Generating visualizations...")
            viz_path = os.path.join(args.output_dir, 'predictions_visualization.png')
            evaluator.visualize_predictions(test_loader, num_samples=5, save_path=viz_path)
        
        print("Evaluation completed successfully")
        
    except Exception as e:
        print(f"Evaluation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()