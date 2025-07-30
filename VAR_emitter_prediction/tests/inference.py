import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from var_emitter_model import VAREmitterPredictor
from var_dataset import InferenceDataset


class VAREmitterInference:
    """
    Inference class for VAR-based emitter prediction
    """
    
    def __init__(self,
                 model: VAREmitterPredictor,
                 device: str = 'cuda',
                 use_amp: bool = True):
        
        self.model = model
        self.device = device
        self.use_amp = use_amp
        
        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.eval()
        
        print(f"Inference initialized on device: {device}")
        print(f"AMP enabled: {use_amp}")
    
    def predict_single(self, 
                      input_image: torch.Tensor,
                      return_all_scales: bool = False) -> Dict[str, torch.Tensor]:
        """
        Predict emitters for a single input image
        
        Args:
            input_image: Input tensor of shape [1, C, H, W] or [C, H, W]
            return_all_scales: Whether to return predictions for all scales
        
        Returns:
            Dictionary containing predictions
        """
        
        # Ensure input has batch dimension
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
        
        # Move to device
        input_image = input_image.to(self.device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                predictions = self.model(input_image)
        
        # Process predictions
        result = {}
        
        if return_all_scales:
            # Return all scale predictions
            for scale_idx, scale_pred in enumerate(predictions['multi_scale']):
                scale_key = f'scale_{scale_idx}'
                result[scale_key] = {
                    'prob_map': torch.sigmoid(scale_pred['prob_logits']).cpu(),
                    'loc_map': scale_pred['locations'].cpu(),
                    'uncertainty': torch.sigmoid(scale_pred['uncertainty']).cpu() if 'uncertainty' in scale_pred else None
                }
        
        # Always return the highest resolution prediction
        final_pred = predictions['multi_scale'][-1]  # Last scale is highest resolution
        result['final'] = {
            'prob_map': torch.sigmoid(final_pred['prob_logits']).cpu(),
            'loc_map': final_pred['locations'].cpu(),
            'uncertainty': torch.sigmoid(final_pred['uncertainty']).cpu() if 'uncertainty' in final_pred else None,
            'count_estimate': predictions['count_estimate'].cpu()
        }
        
        return result
    
    def predict_batch(self,
                     dataloader,
                     output_dir: str,
                     save_visualizations: bool = True,
                     save_raw_outputs: bool = True) -> List[Dict]:
        """
        Predict emitters for a batch of images
        
        Args:
            dataloader: DataLoader containing input images
            output_dir: Directory to save outputs
            save_visualizations: Whether to save visualization images
            save_raw_outputs: Whether to save raw prediction tensors
        
        Returns:
            List of prediction results
        """
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if save_visualizations:
            vis_dir = output_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True)
        
        if save_raw_outputs:
            raw_dir = output_dir / 'raw_outputs'
            raw_dir.mkdir(exist_ok=True)
        
        all_results = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Inference')):
            input_images = batch['image']
            batch_size = input_images.shape[0]
            
            # Move to device
            input_images = input_images.to(self.device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    predictions = self.model(input_images)
            
            # Process each image in the batch
            for i in range(batch_size):
                img_idx = batch_idx * dataloader.batch_size + i
                
                # Extract single image predictions
                single_pred = {
                    'multi_scale': [],
                    'count_estimate': predictions['count_estimate'][i:i+1]
                }
                
                for scale_pred in predictions['multi_scale']:
                    single_scale = {}
                    for key, value in scale_pred.items():
                        single_scale[key] = value[i:i+1]
                    single_pred['multi_scale'].append(single_scale)
                
                # Process predictions
                result = self._process_single_prediction(single_pred, img_idx)
                
                # Add metadata
                if 'filename' in batch:
                    result['filename'] = batch['filename'][i]
                if 'original_size' in batch:
                    result['original_size'] = batch['original_size'][i]
                
                all_results.append(result)
                
                # Save outputs
                if save_raw_outputs:
                    self._save_raw_output(result, raw_dir / f'prediction_{img_idx:06d}.h5')
                
                if save_visualizations:
                    self._save_visualization(
                        input_images[i].cpu(),
                        result,
                        vis_dir / f'visualization_{img_idx:06d}.png'
                    )
        
        # Save summary
        self._save_summary(all_results, output_dir / 'summary.json')
        
        return all_results
    
    def _process_single_prediction(self, predictions: Dict, img_idx: int) -> Dict:
        """Process predictions for a single image"""
        
        result = {
            'image_idx': img_idx,
            'scales': [],
            'count_estimate': predictions['count_estimate'].item()
        }
        
        # Process each scale
        for scale_idx, scale_pred in enumerate(predictions['multi_scale']):
            prob_map = torch.sigmoid(scale_pred['prob_logits']).squeeze().cpu().numpy()
            loc_map = scale_pred['locations'].squeeze().cpu().numpy()
            
            scale_result = {
                'scale_idx': scale_idx,
                'resolution': prob_map.shape,
                'prob_map': prob_map,
                'loc_map': loc_map,
                'max_prob': float(prob_map.max()),
                'mean_prob': float(prob_map.mean()),
                'estimated_count': float(prob_map.sum())
            }
            
            if 'uncertainty' in scale_pred:
                uncertainty = torch.sigmoid(scale_pred['uncertainty']).squeeze().cpu().numpy()
                scale_result['uncertainty'] = uncertainty
                scale_result['mean_uncertainty'] = float(uncertainty.mean())
            
            # Detect peaks (potential emitter locations)
            peaks = self._detect_peaks(prob_map, threshold=0.5, min_distance=2)
            scale_result['detected_peaks'] = peaks
            scale_result['num_peaks'] = len(peaks)
            
            result['scales'].append(scale_result)
        
        # Use highest resolution scale as final result
        result['final'] = result['scales'][-1]
        
        return result
    
    def _detect_peaks(self, prob_map: np.ndarray, threshold: float = 0.5, min_distance: int = 2) -> List[Tuple[int, int]]:
        """Detect peaks in probability map"""
        from scipy.ndimage import maximum_filter
        from scipy.ndimage import label
        
        # Apply threshold
        binary_map = prob_map > threshold
        
        # Find local maxima
        local_maxima = maximum_filter(prob_map, size=min_distance) == prob_map
        peaks_map = binary_map & local_maxima
        
        # Get peak coordinates
        peak_coords = np.where(peaks_map)
        peaks = list(zip(peak_coords[0], peak_coords[1]))
        
        return peaks
    
    def _save_raw_output(self, result: Dict, filepath: Path):
        """Save raw prediction outputs to HDF5 file"""
        
        with h5py.File(filepath, 'w') as f:
            # Save metadata
            f.attrs['image_idx'] = result['image_idx']
            f.attrs['count_estimate'] = result['count_estimate']
            f.attrs['num_scales'] = len(result['scales'])
            
            if 'filename' in result:
                f.attrs['filename'] = result['filename']
            
            # Save each scale
            for scale_idx, scale_data in enumerate(result['scales']):
                scale_group = f.create_group(f'scale_{scale_idx}')
                
                scale_group.attrs['resolution'] = scale_data['resolution']
                scale_group.attrs['max_prob'] = scale_data['max_prob']
                scale_group.attrs['mean_prob'] = scale_data['mean_prob']
                scale_group.attrs['estimated_count'] = scale_data['estimated_count']
                scale_group.attrs['num_peaks'] = scale_data['num_peaks']
                
                # Save arrays
                scale_group.create_dataset('prob_map', data=scale_data['prob_map'])
                scale_group.create_dataset('loc_map', data=scale_data['loc_map'])
                
                if 'uncertainty' in scale_data:
                    scale_group.create_dataset('uncertainty', data=scale_data['uncertainty'])
                    scale_group.attrs['mean_uncertainty'] = scale_data['mean_uncertainty']
                
                # Save detected peaks
                if scale_data['detected_peaks']:
                    peaks_array = np.array(scale_data['detected_peaks'])
                    scale_group.create_dataset('detected_peaks', data=peaks_array)
    
    def _save_visualization(self, input_image: torch.Tensor, result: Dict, filepath: Path):
        """Save visualization of predictions"""
        
        # Prepare input image for visualization
        if input_image.dim() == 3 and input_image.shape[0] == 1:
            input_img = input_image.squeeze(0).numpy()
        else:
            input_img = input_image.numpy()
        
        # Normalize input image
        input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
        
        # Create figure
        num_scales = len(result['scales'])
        fig, axes = plt.subplots(2, num_scales + 1, figsize=(4 * (num_scales + 1), 8))
        
        if num_scales == 0:
            axes = axes.reshape(2, -1)
        
        # Plot input image
        axes[0, 0].imshow(input_img, cmap='gray')
        axes[0, 0].set_title('Input Image\n(40x40)')
        axes[0, 0].axis('off')
        
        axes[1, 0].text(0.5, 0.5, f'Count Estimate:\n{result["count_estimate"]:.2f}', 
                        ha='center', va='center', transform=axes[1, 0].transAxes,
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue'))
        axes[1, 0].axis('off')
        
        # Plot each scale
        for i, scale_data in enumerate(result['scales']):
            col_idx = i + 1
            
            # Probability map
            im1 = axes[0, col_idx].imshow(scale_data['prob_map'], cmap='hot', vmin=0, vmax=1)
            axes[0, col_idx].set_title(f'Scale {i}\n{scale_data["resolution"]}\nPeaks: {scale_data["num_peaks"]}')
            axes[0, col_idx].axis('off')
            
            # Mark detected peaks
            for peak in scale_data['detected_peaks']:
                axes[0, col_idx].plot(peak[1], peak[0], 'b+', markersize=8, markeredgewidth=2)
            
            plt.colorbar(im1, ax=axes[0, col_idx], fraction=0.046, pad=0.04)
            
            # Uncertainty map (if available)
            if 'uncertainty' in scale_data:
                im2 = axes[1, col_idx].imshow(scale_data['uncertainty'], cmap='viridis', vmin=0, vmax=1)
                axes[1, col_idx].set_title(f'Uncertainty\nMean: {scale_data["mean_uncertainty"]:.3f}')
                plt.colorbar(im2, ax=axes[1, col_idx], fraction=0.046, pad=0.04)
            else:
                # Show location map instead
                loc_magnitude = np.sqrt(scale_data['loc_map'][0]**2 + scale_data['loc_map'][1]**2)
                im2 = axes[1, col_idx].imshow(loc_magnitude, cmap='viridis')
                axes[1, col_idx].set_title('Location Magnitude')
                plt.colorbar(im2, ax=axes[1, col_idx], fraction=0.046, pad=0.04)
            
            axes[1, col_idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_summary(self, results: List[Dict], filepath: Path):
        """Save summary statistics"""
        
        summary = {
            'total_images': len(results),
            'statistics': {}
        }
        
        # Collect statistics
        count_estimates = [r['count_estimate'] for r in results]
        
        summary['statistics']['count_estimates'] = {
            'mean': float(np.mean(count_estimates)),
            'std': float(np.std(count_estimates)),
            'min': float(np.min(count_estimates)),
            'max': float(np.max(count_estimates))
        }
        
        # Scale-wise statistics
        for scale_idx in range(len(results[0]['scales'])):
            scale_key = f'scale_{scale_idx}'
            
            max_probs = [r['scales'][scale_idx]['max_prob'] for r in results]
            mean_probs = [r['scales'][scale_idx]['mean_prob'] for r in results]
            num_peaks = [r['scales'][scale_idx]['num_peaks'] for r in results]
            
            summary['statistics'][scale_key] = {
                'resolution': results[0]['scales'][scale_idx]['resolution'],
                'max_prob': {
                    'mean': float(np.mean(max_probs)),
                    'std': float(np.std(max_probs))
                },
                'mean_prob': {
                    'mean': float(np.mean(mean_probs)),
                    'std': float(np.std(mean_probs))
                },
                'num_peaks': {
                    'mean': float(np.mean(num_peaks)),
                    'std': float(np.std(num_peaks))
                }
            }
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to {filepath}")
        print(f"Processed {summary['total_images']} images")
        print(f"Mean count estimate: {summary['statistics']['count_estimates']['mean']:.2f} ± {summary['statistics']['count_estimates']['std']:.2f}")


def load_model(checkpoint_path: str, config_path: str, device: str = 'cuda') -> VAREmitterPredictor:
    """Load trained model from checkpoint"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model
    model = VAREmitterPredictor(
        input_size=config['model']['input_size'],
        target_sizes=config['model']['target_sizes'],
        base_channels=config['model']['base_channels'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation loss: {checkpoint.get('val_loss', 'unknown')}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='VAR-based Emitter Prediction Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input TIFF files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--no_visualizations', action='store_true', help='Skip saving visualizations')
    parser.add_argument('--no_raw_outputs', action='store_true', help='Skip saving raw outputs')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, args.config, device)
    
    # Create inference dataset
    print("Creating dataset...")
    dataset = InferenceDataset(
        tiff_dir=args.input_dir,
        input_size=(40, 40)  # Always use 40x40 for inference as specified
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Found {len(dataset)} images for inference")
    
    # Create inference object
    inference = VAREmitterInference(
        model=model,
        device=device,
        use_amp=True
    )
    
    # Run inference
    print("Running inference...")
    results = inference.predict_batch(
        dataloader=dataloader,
        output_dir=args.output_dir,
        save_visualizations=not args.no_visualizations,
        save_raw_outputs=not args.no_raw_outputs
    )
    
    print(f"Inference completed! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()