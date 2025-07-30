import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import argparse
from var_emitter_model_true import TrueVAREmitterPredictor


class VAREmitterInference:
    """
    Inference engine for VAR Emitter Prediction
    Demonstrates progressive resolution enhancement from 40x40 to 160x160
    """
    
    def __init__(self, model_path: str, config_path: str, device: str = 'auto'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
        print(f"Model patch_nums: {self.model.patch_nums}")
    
    def _load_model(self, model_path: str) -> TrueVAREmitterPredictor:
        """Load trained model"""
        # Initialize model
        model_config = self.config['model']
        model = TrueVAREmitterPredictor(**model_config)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    @torch.no_grad()
    def predict_progressive(self, input_image: np.ndarray, 
                          visualize: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Progressive VAR inference from low-res to high-res
        
        Args:
            input_image: Input image (H, W) or (H, W, C)
            visualize: Whether to create visualization
        
        Returns:
            Dictionary containing predictions at each scale
        """
        # Preprocess input
        input_tensor = self._preprocess_input(input_image)
        
        # Progressive inference through all scales
        predictions = self.model(input_tensor)
        
        # Convert to numpy for visualization
        results = {}
        for scale_key, pred in predictions.items():
            scale_idx = int(scale_key.split('_')[1])
            resolution = self.model.patch_nums[scale_idx]
            
            prob_map = pred['prob_map'].cpu().numpy()[0, 0]  # (H, W)
            loc_map = pred['loc_map'].cpu().numpy()[0]        # (2, H, W)
            
            results[f'scale_{resolution}x{resolution}'] = {
                'prob_map': prob_map,
                'loc_map': loc_map,
                'resolution': resolution,
                'emitter_positions': self._extract_emitter_positions(prob_map, loc_map)
            }
        
        if visualize:
            self._visualize_progressive_results(input_image, results)
        
        return results
    
    @torch.no_grad()
    def predict_super_resolution(self, input_image: np.ndarray, 
                               target_resolution: int = 80,
                               visualize: bool = True) -> Dict[str, np.ndarray]:
        """
        Super-resolution inference: 40x40 -> target_resolution
        
        Args:
            input_image: Low resolution input (will be resized to 40x40)
            target_resolution: Target high resolution
            visualize: Whether to create visualization
        
        Returns:
            High-resolution prediction results
        """
        # Resize input to 40x40 for super-resolution demo
        input_40x40 = self._resize_image(input_image, (40, 40))
        input_tensor = self._preprocess_input(input_40x40)
        
        # Super-resolution inference
        result = self.model.autoregressive_inference(input_tensor, target_resolution)
        
        # Convert to numpy
        prob_map = result['prob_map'].cpu().numpy()[0, 0]  # (H, W)
        loc_map = result['loc_map'].cpu().numpy()[0]        # (2, H, W)
        
        emitter_positions = self._extract_emitter_positions(prob_map, loc_map)
        
        results = {
            'prob_map': prob_map,
            'loc_map': loc_map,
            'emitter_positions': emitter_positions,
            'input_40x40': input_40x40,
            'target_resolution': target_resolution
        }
        
        if visualize:
            self._visualize_super_resolution(results)
        
        return results
    
    def _preprocess_input(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess input image"""
        # Ensure single channel
        if len(image.shape) == 3:
            image = image.mean(axis=2)
        
        # Convert to tensor
        tensor = torch.from_numpy(image).float()
        
        # Add batch and channel dimensions
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # Normalize
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
        
        return tensor.to(self.device)
    
    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image using torch interpolation"""
        if len(image.shape) == 3:
            image = image.mean(axis=2)
        
        tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
        
        return resized.squeeze().numpy()
    
    def _extract_emitter_positions(self, prob_map: np.ndarray, 
                                 loc_map: np.ndarray, 
                                 threshold: float = 0.5) -> np.ndarray:
        """
        Extract emitter positions from probability and location maps
        
        Args:
            prob_map: (H, W) probability map
            loc_map: (2, H, W) location offset map
            threshold: Probability threshold for detection
        
        Returns:
            (N, 2) array of emitter positions in pixel coordinates
        """
        # Find peaks above threshold
        peaks = prob_map > threshold
        peak_coords = np.where(peaks)
        
        if len(peak_coords[0]) == 0:
            return np.array([]).reshape(0, 2)
        
        # Get sub-pixel positions
        positions = []
        for y, x in zip(peak_coords[0], peak_coords[1]):
            # Get sub-pixel offsets (convert from [-1, 1] to [0, 1])
            offset_x = (loc_map[0, y, x] + 1) / 2
            offset_y = (loc_map[1, y, x] + 1) / 2
            
            # Calculate final position
            pos_x = x + offset_x
            pos_y = y + offset_y
            
            positions.append([pos_x, pos_y])
        
        return np.array(positions)
    
    def _visualize_progressive_results(self, input_image: np.ndarray, 
                                     results: Dict[str, Dict[str, np.ndarray]]):
        """Visualize progressive resolution enhancement"""
        num_scales = len(results)
        fig, axes = plt.subplots(2, num_scales + 1, figsize=(4 * (num_scales + 1), 8))
        
        # Show input
        axes[0, 0].imshow(input_image, cmap='gray')
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        
        # Show results for each scale
        for i, (scale_name, result) in enumerate(results.items()):
            col = i + 1
            resolution = result['resolution']
            
            # Probability map
            axes[0, col].imshow(result['prob_map'], cmap='hot', vmin=0, vmax=1)
            axes[0, col].set_title(f'Prob Map {resolution}x{resolution}')
            axes[0, col].axis('off')
            
            # Emitter positions overlay
            axes[1, col].imshow(result['prob_map'], cmap='gray')
            if len(result['emitter_positions']) > 0:
                positions = result['emitter_positions']
                axes[1, col].scatter(positions[:, 0], positions[:, 1], 
                                   c='red', s=20, marker='x')
            axes[1, col].set_title(f'Detections {resolution}x{resolution}')
            axes[1, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('progressive_results.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def _visualize_super_resolution(self, results: Dict[str, np.ndarray]):
        """Visualize super-resolution results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Input 40x40
        axes[0].imshow(results['input_40x40'], cmap='gray')
        axes[0].set_title('Input 40x40')
        axes[0].axis('off')
        
        # High-res probability map
        target_res = results['target_resolution']
        axes[1].imshow(results['prob_map'], cmap='hot', vmin=0, vmax=1)
        axes[1].set_title(f'Prob Map {target_res}x{target_res}')
        axes[1].axis('off')
        
        # Detected emitters
        axes[2].imshow(results['prob_map'], cmap='gray')
        if len(results['emitter_positions']) > 0:
            positions = results['emitter_positions']
            axes[2].scatter(positions[:, 0], positions[:, 1], 
                          c='red', s=30, marker='x', linewidths=2)
        axes[2].set_title(f'Detections {target_res}x{target_res}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('super_resolution_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        num_detections = len(results['emitter_positions'])
        max_prob = results['prob_map'].max()
        print(f"\nSuper-resolution Results:")
        print(f"Input resolution: 40x40")
        print(f"Output resolution: {target_res}x{target_res}")
        print(f"Number of detections: {num_detections}")
        print(f"Maximum probability: {max_prob:.3f}")


def create_synthetic_test_image(size: Tuple[int, int] = (160, 160), 
                              num_emitters: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic test image with known emitter positions
    
    Returns:
        image: Synthetic image
        emitter_positions: Ground truth positions
    """
    image = np.random.normal(0, 0.1, size)  # Background noise
    
    # Add random emitters
    emitter_positions = []
    for _ in range(num_emitters):
        x = np.random.randint(10, size[1] - 10)
        y = np.random.randint(10, size[0] - 10)
        
        # Add Gaussian spot
        sigma = np.random.uniform(1.5, 3.0)
        amplitude = np.random.uniform(0.8, 1.2)
        
        yy, xx = np.ogrid[:size[0], :size[1]]
        gaussian = amplitude * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
        image += gaussian
        
        emitter_positions.append([x, y])
    
    # Normalize
    image = np.clip(image, 0, 1)
    
    return image, np.array(emitter_positions)


def main():
    parser = argparse.ArgumentParser(description='VAR Emitter Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='../configs/config_true_var.json',
                       help='Path to config file')
    parser.add_argument('--input', type=str, default=None,
                       help='Path to input image (optional, will use synthetic if not provided)')
    parser.add_argument('--mode', type=str, choices=['progressive', 'super_resolution', 'both'],
                       default='both', help='Inference mode')
    parser.add_argument('--target_res', type=int, default=80,
                       help='Target resolution for super-resolution mode')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = VAREmitterInference(args.model, args.config, args.device)
    
    # Load or create test image
    if args.input:
        # Load real image
        if args.input.endswith('.npy'):
            image = np.load(args.input)
        else:
            from PIL import Image
            image = np.array(Image.open(args.input).convert('L')) / 255.0
        print(f"Loaded image from {args.input}")
    else:
        # Create synthetic test image
        image, gt_positions = create_synthetic_test_image()
        print(f"Created synthetic test image with {len(gt_positions)} emitters")
        print(f"Ground truth positions: {gt_positions}")
    
    print(f"Input image shape: {image.shape}")
    
    # Run inference
    if args.mode in ['progressive', 'both']:
        print("\n=== Progressive Resolution Enhancement ===")
        progressive_results = inference.predict_progressive(image, visualize=True)
        
        # Print results summary
        for scale_name, result in progressive_results.items():
            num_detections = len(result['emitter_positions'])
            resolution = result['resolution']
            print(f"{scale_name}: {num_detections} detections")
    
    if args.mode in ['super_resolution', 'both']:
        print(f"\n=== Super-Resolution: 40x40 -> {args.target_res}x{args.target_res} ===")
        sr_results = inference.predict_super_resolution(
            image, target_resolution=args.target_res, visualize=True
        )
    
    print("\nInference completed! Check the generated visualization images.")


if __name__ == "__main__":
    main()