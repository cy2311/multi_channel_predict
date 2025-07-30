import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import tifffile
import cv2
from scipy import ndimage


class MultiScaleEmitterDataset(Dataset):
    """
    Multi-scale dataset for VAR-based emitter prediction
    Supports training with high-resolution data and inference with low-resolution input
    """
    
    def __init__(self,
                 tiff_dir: str,
                 emitter_dir: str,
                 low_res_size: int = 40,  # Input resolution (physical size consistent)
                 high_res_sizes: List[int] = [40, 80, 160, 320],  # Training resolutions
                 pixel_size_nm: float = 100.0,  # Physical pixel size in nm
                 frame_window: int = 3,
                 max_emitters_per_frame: int = 100,
                 train_mode: bool = True,
                 augment: bool = True):
        
        self.tiff_dir = Path(tiff_dir)
        self.emitter_dir = Path(emitter_dir)
        self.low_res_size = low_res_size
        self.high_res_sizes = high_res_sizes
        self.pixel_size_nm = pixel_size_nm
        self.frame_window = frame_window
        self.max_emitters_per_frame = max_emitters_per_frame
        self.train_mode = train_mode
        self.augment = augment
        
        # Get all TIFF files (search recursively in subdirectories)
        self.tiff_files = sorted(list(self.tiff_dir.glob("**/*.ome.tiff")))
        
        # Build sample index
        self.samples = self._build_sample_index()
        
        print(f"Loaded {len(self.samples)} samples from {len(self.tiff_files)} TIFF files")
    
    def _build_sample_index(self) -> List[Dict]:
        """Build index of all available samples"""
        samples = []
        
        for tiff_idx, tiff_file in enumerate(self.tiff_files):
            # Check for corresponding emitter file in the same directory as TIFF
            tiff_parent = tiff_file.parent
            emitter_file = tiff_parent / "emitters.h5"
            if not emitter_file.exists():
                print(f"Warning: Emitter file {emitter_file} not found")
                continue
            
            # Get number of frames
            try:
                with tifffile.TiffFile(str(tiff_file)) as f:
                    n_frames = len(f.pages)
                
                # Create samples for valid frame windows
                for start_frame in range(n_frames - self.frame_window + 1):
                    samples.append({
                        'tiff_file': tiff_file,
                        'emitter_file': emitter_file,
                        'start_frame': start_frame,
                        'tiff_idx': tiff_idx
                    })
            except Exception as e:
                print(f"Error processing {tiff_file}: {e}")
                continue
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load TIFF frames
        frames = self._load_tiff_frames(sample)
        
        # Load emitter data
        emitter_data = self._load_emitter_data(sample)
        
        # Process multi-scale data
        processed_data = self._process_multiscale_data(frames, emitter_data)
        
        return processed_data
    
    def _load_tiff_frames(self, sample: Dict) -> np.ndarray:
        """Load TIFF frames for the sample"""
        tiff_file = sample['tiff_file']
        start_frame = sample['start_frame']
        
        with tifffile.TiffFile(str(tiff_file)) as f:
            frames = []
            for i in range(self.frame_window):
                frame_idx = start_frame + i
                if frame_idx < len(f.pages):
                    frame = f.pages[frame_idx].asarray()
                    frames.append(frame)
                else:
                    # Pad with last frame if needed
                    frames.append(frames[-1])
        
        return np.stack(frames, axis=0)  # (T, H, W)
    
    def _load_emitter_data(self, sample: Dict) -> Dict:
        """Load emitter data from H5 file"""
        emitter_file = sample['emitter_file']
        start_frame = sample['start_frame']
        
        emitter_data = {'locations': [], 'counts': []}
        
        try:
            with h5py.File(emitter_file, 'r') as f:
                for i in range(self.frame_window):
                    frame_idx = start_frame + i
                    frame_key = f'frame_{frame_idx}'
                    
                    if frame_key in f:
                        frame_group = f[frame_key]
                        if 'emitters' in frame_group:
                            emitters = frame_group['emitters'][:]
                            # Assuming emitters format: [x, y, intensity, ...]
                            locations = emitters[:, :2]  # x, y coordinates
                            emitter_data['locations'].append(locations)
                            emitter_data['counts'].append(len(locations))
                        else:
                            emitter_data['locations'].append(np.empty((0, 2)))
                            emitter_data['counts'].append(0)
                    else:
                        emitter_data['locations'].append(np.empty((0, 2)))
                        emitter_data['counts'].append(0)
        except Exception as e:
            print(f"Error loading emitter data from {emitter_file}: {e}")
            # Return empty data
            for i in range(self.frame_window):
                emitter_data['locations'].append(np.empty((0, 2)))
                emitter_data['counts'].append(0)
        
        return emitter_data
    
    def _process_multiscale_data(self, frames: np.ndarray, emitter_data: Dict) -> Dict[str, torch.Tensor]:
        """Process data for multi-scale training"""
        T, H, W = frames.shape
        
        # Use middle frame as reference
        ref_frame = frames[T // 2]
        ref_locations = emitter_data['locations'][T // 2]
        ref_count = emitter_data['counts'][T // 2]
        
        # Normalize frame
        ref_frame = self._normalize_frame(ref_frame)
        
        # Create low-resolution input (always 40x40)
        low_res_input = self._resize_frame(ref_frame, self.low_res_size)
        
        # Create multi-scale targets if in training mode
        multiscale_data = {
            'low_res_input': torch.from_numpy(low_res_input).unsqueeze(0).float(),  # (1, H, W)
            'count': torch.tensor(ref_count, dtype=torch.long),
        }
        
        if self.train_mode:
            # Create high-resolution targets
            prob_maps = {}
            processed_locations = {}
            
            for scale_idx, target_size in enumerate(self.high_res_sizes):
                # Resize frame to target resolution
                high_res_frame = self._resize_frame(ref_frame, target_size)
                
                # Handle both int and [h, w] formats for scale calculation
                if isinstance(target_size, (list, tuple)):
                    target_h, target_w = target_size
                    scale_factor = min(target_h, target_w) / min(H, W)
                    # Calculate prob_map_size based on actual encoder output
                    # The encoder has two stride-2 convolutions, so output is input_size // 4
                    # But we need to match the actual model output size
                    # For 40x40 input, encoder outputs 5x5 (not 10x10 as expected)
                    # This suggests additional downsampling in the model
                    prob_map_size = min(target_h, target_w) // 16  # Match actual model output
                else:
                    scale_factor = target_size / min(H, W)
                    # Calculate prob_map_size based on actual encoder output
                    # The encoder has two stride-2 convolutions, so output is input_size // 4
                    # But we need to match the actual model output size
                    # For 40x40 input, encoder outputs 5x5 (not 10x10 as expected)
                    # This suggests additional downsampling in the model
                    prob_map_size = target_size // 16  # Match actual model output
                
                scaled_locations = ref_locations * scale_factor
                # Scale locations to match the prob_map coordinate system
                scaled_locations = scaled_locations * prob_map_size / (scale_factor * min(H, W))
                
                # Create probability map
                prob_map = self._create_probability_map(scaled_locations, prob_map_size)
                
                prob_maps[f'scale_{scale_idx}'] = torch.from_numpy(prob_map).unsqueeze(0).float()
                processed_locations[f'scale_{scale_idx}'] = torch.from_numpy(scaled_locations).float()
            
            multiscale_data['prob_maps'] = prob_maps
            multiscale_data['locations_by_scale'] = processed_locations
        
        # Pad locations for consistent tensor size
        padded_locations = self._pad_locations(ref_locations)
        multiscale_data['locations'] = torch.from_numpy(padded_locations).float()
        
        # Apply augmentation if enabled
        if self.augment and self.train_mode:
            multiscale_data = self._apply_augmentation(multiscale_data)
        
        return multiscale_data
    
    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame to [0, 1] range"""
        frame = frame.astype(np.float32)
        frame_min, frame_max = frame.min(), frame.max()
        if frame_max > frame_min:
            frame = (frame - frame_min) / (frame_max - frame_min)
        return frame
    
    def _resize_frame(self, frame: np.ndarray, target_size) -> np.ndarray:
        """Resize frame to target size"""
        # Handle both int and [h, w] formats
        if isinstance(target_size, (list, tuple)):
            h, w = target_size
            return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            return cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    def _create_probability_map(self, locations: np.ndarray, size: int, sigma: float = 1.0) -> np.ndarray:
        """Create Gaussian probability map from emitter locations"""
        prob_map = np.zeros((size, size), dtype=np.float32)
        
        if len(locations) == 0:
            return prob_map
        
        # Create Gaussian blobs at emitter locations
        for loc in locations:
            x, y = loc
            if 0 <= x < size and 0 <= y < size:
                # Create small Gaussian around location
                xx, yy = np.meshgrid(np.arange(size), np.arange(size))
                gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
                prob_map = np.maximum(prob_map, gaussian)
        
        return prob_map
    
    def _pad_locations(self, locations: np.ndarray) -> np.ndarray:
        """Pad locations array to fixed size"""
        if len(locations) == 0:
            return np.full((self.max_emitters_per_frame, 2), -1.0, dtype=np.float32)
        
        padded = np.full((self.max_emitters_per_frame, 2), -1.0, dtype=np.float32)
        n_emitters = min(len(locations), self.max_emitters_per_frame)
        padded[:n_emitters] = locations[:n_emitters]
        
        return padded
    
    def _apply_augmentation(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply data augmentation"""
        # Random horizontal flip
        if torch.rand(1) < 0.5:
            data['low_res_input'] = torch.flip(data['low_res_input'], dims=[2])
            
            if 'prob_maps' in data:
                for key in data['prob_maps']:
                    data['prob_maps'][key] = torch.flip(data['prob_maps'][key], dims=[2])
            
            # Flip locations
            locations = data['locations']
            valid_mask = locations[:, 0] >= 0
            # Handle both int and [h, w] formats for low_res_size
            if isinstance(self.low_res_size, (list, tuple)):
                size_w = self.low_res_size[1]  # Use width for horizontal flip
            else:
                size_w = self.low_res_size
            locations[valid_mask, 0] = size_w - 1 - locations[valid_mask, 0]
        
        # Random vertical flip
        if torch.rand(1) < 0.5:
            data['low_res_input'] = torch.flip(data['low_res_input'], dims=[1])
            
            if 'prob_maps' in data:
                for key in data['prob_maps']:
                    data['prob_maps'][key] = torch.flip(data['prob_maps'][key], dims=[1])
            
            # Flip locations
            locations = data['locations']
            valid_mask = locations[:, 0] >= 0
            # Handle both int and [h, w] formats for low_res_size
            if isinstance(self.low_res_size, (list, tuple)):
                size_h = self.low_res_size[0]  # Use height for vertical flip
            else:
                size_h = self.low_res_size
            locations[valid_mask, 1] = size_h - 1 - locations[valid_mask, 1]
        
        # Random rotation (90 degree increments)
        if torch.rand(1) < 0.5:
            k = torch.randint(1, 4, (1,)).item()
            data['low_res_input'] = torch.rot90(data['low_res_input'], k, dims=[1, 2])
            
            if 'prob_maps' in data:
                for key in data['prob_maps']:
                    data['prob_maps'][key] = torch.rot90(data['prob_maps'][key], k, dims=[1, 2])
        
        return data


def create_dataloaders(tiff_dir: str,
                      emitter_dir: str,
                      batch_size: int = 4,
                      num_workers: int = 4,
                      train_val_split: float = 0.8,
                      **dataset_kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders
    
    Args:
        tiff_dir: Directory containing TIFF files
        emitter_dir: Directory containing emitter H5 files
        batch_size: Batch size for training
        num_workers: Number of worker processes
        train_val_split: Fraction of data for training
        **dataset_kwargs: Additional arguments for dataset
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    # Create full dataset
    full_dataset = MultiScaleEmitterDataset(
        tiff_dir=tiff_dir,
        emitter_dir=emitter_dir,
        train_mode=True,
        augment=True,
        **dataset_kwargs
    )
    
    # Split into train/val
    total_size = len(full_dataset)
    train_size = int(train_val_split * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create validation dataset without augmentation
    val_dataset_no_aug = MultiScaleEmitterDataset(
        tiff_dir=tiff_dir,
        emitter_dir=emitter_dir,
        train_mode=True,
        augment=False,
        **dataset_kwargs
    )
    
    # Update validation dataset
    val_indices = val_dataset.indices
    val_dataset = torch.utils.data.Subset(val_dataset_no_aug, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


class InferenceDataset(Dataset):
    """
    Dataset for inference with low-resolution input
    """
    
    def __init__(self, 
                 tiff_dir: str,
                 input_size: int = 40,
                 frame_window: int = 3):
        
        self.tiff_dir = Path(tiff_dir)
        self.input_size = input_size
        self.frame_window = frame_window
        
        self.tiff_files = sorted(list(self.tiff_dir.glob("*.ome.tiff")))
        self.samples = self._build_sample_index()
    
    def _build_sample_index(self) -> List[Dict]:
        samples = []
        
        for tiff_idx, tiff_file in enumerate(self.tiff_files):
            try:
                with tifffile.TiffFile(str(tiff_file)) as f:
                    n_frames = len(f.pages)
                
                for start_frame in range(n_frames - self.frame_window + 1):
                    samples.append({
                        'tiff_file': tiff_file,
                        'start_frame': start_frame,
                        'tiff_idx': tiff_idx
                    })
            except Exception as e:
                print(f"Error processing {tiff_file}: {e}")
                continue
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load frames
        frames = self._load_tiff_frames(sample)
        T, H, W = frames.shape
        
        # Use middle frame
        ref_frame = frames[T // 2]
        
        # Normalize and resize
        ref_frame = self._normalize_frame(ref_frame)
        low_res_input = self._resize_frame(ref_frame, self.input_size)
        
        return {
            'input': torch.from_numpy(low_res_input).unsqueeze(0).float(),
            'sample_info': sample
        }
    
    def _load_tiff_frames(self, sample: Dict) -> np.ndarray:
        """Load TIFF frames"""
        tiff_file = sample['tiff_file']
        start_frame = sample['start_frame']
        
        with tifffile.TiffFile(str(tiff_file)) as f:
            frames = []
            for i in range(self.frame_window):
                frame_idx = start_frame + i
                if frame_idx < len(f.pages):
                    frame = f.pages[frame_idx].asarray()
                    frames.append(frame)
                else:
                    frames.append(frames[-1])
        
        return np.stack(frames, axis=0)
    
    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame"""
        frame = frame.astype(np.float32)
        frame_min, frame_max = frame.min(), frame.max()
        if frame_max > frame_min:
            frame = (frame - frame_min) / (frame_max - frame_min)
        return frame
    
    def _resize_frame(self, frame: np.ndarray, target_size) -> np.ndarray:
        """Resize frame to target size"""
        # Handle both int and [h, w] formats
        if isinstance(target_size, (list, tuple)):
            h, w = target_size
            return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            return cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)