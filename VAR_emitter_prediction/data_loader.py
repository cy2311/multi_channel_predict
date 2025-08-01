import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import h5py
import os

class EmitterDataset(Dataset):
    """Dataset for emitter prediction with DECODE-style 6-channel targets."""
    
    def __init__(self, data_path: str, transform=None, target_sizes: List[int] = [40, 80, 160, 320]):
        self.data_path = data_path
        self.transform = transform
        self.target_sizes = target_sizes
        self.data_files = self._load_data_files()
        
    def _load_data_files(self) -> List[str]:
        """Load list of data files."""
        if os.path.isfile(self.data_path):
            return [self.data_path]
        elif os.path.isdir(self.data_path):
            files = []
            for f in os.listdir(self.data_path):
                if f.endswith(('.h5', '.hdf5')):
                    files.append(os.path.join(self.data_path, f))
            return sorted(files)
        else:
            raise ValueError(f"Data path {self.data_path} does not exist")
    
    def __len__(self) -> int:
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample."""
        file_path = self.data_files[idx]
        
        with h5py.File(file_path, 'r') as f:
            # Load input image
            image = torch.from_numpy(f['image'][:]).float()
            
            # Load emitter positions and properties
            positions = torch.from_numpy(f['positions'][:]).float()  # [N, 3] (x, y, z)
            photons = torch.from_numpy(f['photons'][:]).float()      # [N]
            background = torch.from_numpy(f['background'][:]).float() if 'background' in f else torch.zeros_like(photons)
            
        # Apply transforms if provided
        if self.transform:
            image, positions, photons, background = self.transform(image, positions, photons, background)
        
        # Generate multi-scale targets
        targets = self._generate_multiscale_targets(positions, photons, background, image.shape[-2:])
        
        return {
            'image': image,
            'targets': targets,
            'positions': positions,
            'photons': photons,
            'background': background
        }
    
    def _generate_multiscale_targets(self, positions: torch.Tensor, photons: torch.Tensor, 
                                   background: torch.Tensor, image_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """Generate multi-scale 6-channel targets following DECODE standard."""
        targets = {}
        
        for scale_idx, target_size in enumerate(self.target_sizes):
            # Calculate scale factor
            scale_factor = target_size / max(image_size)
            
            # Scale positions
            scaled_positions = positions * scale_factor
            
            # Initialize 6-channel target: [prob, photons, x_offset, y_offset, z_offset, background]
            target = torch.zeros(6, target_size, target_size)
            
            # Filter positions within bounds
            valid_mask = (
                (scaled_positions[:, 0] >= 0) & (scaled_positions[:, 0] < target_size) &
                (scaled_positions[:, 1] >= 0) & (scaled_positions[:, 1] < target_size)
            )
            
            if valid_mask.sum() > 0:
                valid_positions = scaled_positions[valid_mask]
                valid_photons = photons[valid_mask]
                valid_background = background[valid_mask]
                
                # Get integer pixel coordinates
                pixel_coords = valid_positions[:, :2].long()
                
                # Set probability channel (channel 0)
                target[0, pixel_coords[:, 1], pixel_coords[:, 0]] = 1.0
                
                # Set photon channel (channel 1)
                target[1, pixel_coords[:, 1], pixel_coords[:, 0]] = valid_photons
                
                # Set offset channels (channels 2, 3, 4)
                offsets = valid_positions - pixel_coords.float()
                target[2, pixel_coords[:, 1], pixel_coords[:, 0]] = offsets[:, 0]  # x_offset
                target[3, pixel_coords[:, 1], pixel_coords[:, 0]] = offsets[:, 1]  # y_offset
                if valid_positions.shape[1] > 2:  # z coordinate available
                    target[4, pixel_coords[:, 1], pixel_coords[:, 0]] = valid_positions[:, 2]  # z_offset
                
                # Set background channel (channel 5)
                target[5, pixel_coords[:, 1], pixel_coords[:, 0]] = valid_background
            
            targets[f'scale_{scale_idx}'] = target
        
        return targets

class EmitterDataLoader:
    """Data loader wrapper for emitter prediction."""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def get_train_loader(self) -> DataLoader:
        """Get training data loader."""
        dataset = EmitterDataset(
            data_path=self.config['data']['train_path'],
            transform=self.config['data'].get('train_transform'),
            target_sizes=self.config['model']['target_sizes']
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True,
            drop_last=True
        )
    
    def get_val_loader(self) -> DataLoader:
        """Get validation data loader."""
        dataset = EmitterDataset(
            data_path=self.config['data']['val_path'],
            transform=self.config['data'].get('val_transform'),
            target_sizes=self.config['model']['target_sizes']
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config['data']['val_batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching emitter data."""
    images = torch.stack([item['image'] for item in batch])
    
    # Collect targets for each scale
    targets = {}
    for scale_key in batch[0]['targets'].keys():
        targets[scale_key] = torch.stack([item['targets'][scale_key] for item in batch])
    
    # Pad positions, photons, and background to same length
    max_emitters = max(len(item['positions']) for item in batch)
    
    positions_list = []
    photons_list = []
    background_list = []
    
    for item in batch:
        pos = item['positions']
        phot = item['photons']
        bg = item['background']
        
        # Pad to max_emitters
        if len(pos) < max_emitters:
            pad_size = max_emitters - len(pos)
            pos = torch.cat([pos, torch.zeros(pad_size, pos.shape[1])], dim=0)
            phot = torch.cat([phot, torch.zeros(pad_size)], dim=0)
            bg = torch.cat([bg, torch.zeros(pad_size)], dim=0)
        
        positions_list.append(pos)
        photons_list.append(phot)
        background_list.append(bg)
    
    return {
        'image': images,
        'targets': targets,
        'positions': torch.stack(positions_list),
        'photons': torch.stack(photons_list),
        'background': torch.stack(background_list)
    }