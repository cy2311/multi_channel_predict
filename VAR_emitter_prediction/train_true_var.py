import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import wandb
import tifffile
import h5py
from var_emitter_model_true import TrueVAREmitterPredictor, VAREmitterLoss
# Removed NPZ dataset import - using only MultiScaleEmitterDataset


class MultiScaleEmitterDataset(Dataset):
    """
    Dataset for multi-scale emitter prediction training
    Generates targets at multiple resolutions for VAR training
    """
    
    def __init__(self, 
                 data_path: str,
                 input_resolution: Tuple[int, int] = (160, 160),
                 target_resolutions: Dict[str, Tuple[int, int]] = None,
                 augment: bool = True,
                 synthetic_samples: int = 1000):
        super().__init__()
        
        self.data_path = Path(data_path)
        self.input_resolution = input_resolution
        self.target_resolutions = target_resolutions or {
            'target_10': (10, 10),
            'target_20': (20, 20),
            'target_40': (40, 40),
            'target_80': (80, 80)
        }
        self.augment = augment
        
        # Try to load sample directories, fallback to synthetic data
        if self.data_path.exists():
            # Look for sample directories (sample_160_xxx format)
            self.sample_dirs = [d for d in self.data_path.iterdir() 
                              if d.is_dir() and d.name.startswith('sample_')]
        else:
            self.sample_dirs = []
        
        # Use synthetic data if no real data available
        if len(self.sample_dirs) == 0:
            self.use_synthetic = True
            self.synthetic_samples = synthetic_samples
            print(f"No sample directories found at {data_path}, using {synthetic_samples} synthetic samples")
        else:
            self.use_synthetic = False
            print(f"Found {len(self.sample_dirs)} sample directories")
        
    def __len__(self):
        if self.use_synthetic:
            return self.synthetic_samples
        return len(self.sample_dirs)
    
    def __getitem__(self, idx):
        if self.use_synthetic:
            # Generate synthetic data
            np.random.seed(idx)  # For reproducibility
            
            # Generate random image
            image = np.random.randn(*self.input_resolution) * 0.1 + 0.5
            
            # Generate fixed number of emitter positions for consistent batching
            emitter_count = 4  # Fixed number for consistent batching
            emitter_positions = np.random.rand(emitter_count, 2)  # Normalized coordinates
            emitter_photons = np.random.exponential(1000, emitter_count)
        else:
            # Load real data from sample directory
            sample_dir = self.sample_dirs[idx]
            image, emitter_positions, emitter_count, emitter_photons = self._load_sample_data(sample_dir)
        
        # Convert to tensor and resize if needed
        image = torch.from_numpy(image).float().unsqueeze(0)  # (1, H, W)
        if image.shape[1:] != self.input_resolution:
            image = F.interpolate(image.unsqueeze(0), size=self.input_resolution, mode='bilinear', align_corners=False)
            image = image.squeeze(0)  # (1, H, W)
        
        # Normalize image
        image = (image - image.mean()) / (image.std() + 1e-8)
        
        # Generate multi-scale targets
        targets = {}
        for target_key, (target_h, target_w) in self.target_resolutions.items():
            prob_map, loc_map = self._generate_target_maps(
                emitter_positions, target_h, target_w
            )
            # Convert target_XX to scale_X format to match model output
            if target_key.startswith('target_'):
                resolution = int(target_key.split('_')[1])
                if resolution == 10:
                    scale_key = 'scale_0'
                elif resolution == 20:
                    scale_key = 'scale_1'
                elif resolution == 40:
                    scale_key = 'scale_2'
                elif resolution == 80:
                    scale_key = 'scale_3'
                else:
                    scale_key = target_key
            else:
                scale_key = target_key
                
            targets[scale_key] = {
                'prob_map': prob_map,
                'loc_map': loc_map
            }
        
        return {
            'image': image,
            'targets': targets,
            'emitter_positions': torch.from_numpy(emitter_positions).float(),
            'emitter_count': emitter_count,
            'emitter_photons': torch.from_numpy(emitter_photons).float()
        }
    
    def _load_sample_data(self, sample_dir: Path) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        """
        Load image and emitter data from a sample directory
        
        Args:
            sample_dir: Path to sample directory
            
        Returns:
            image: (H, W) numpy array
            emitter_positions: (N, 2) normalized coordinates
            emitter_count: Total number of emitters
            emitter_photons: (N,) photon counts for each emitter
        """
        
        # Find TIFF files
        tiff_files = list(sample_dir.glob('*.tif')) + list(sample_dir.glob('*.tiff')) + list(sample_dir.glob('*.ome.tiff'))
        
        if not tiff_files:
            # If no TIFF files, create a random image
            print(f"Warning: No TIFF files found in {sample_dir}, using random image")
            image = np.random.randn(*self.input_resolution) * 0.1 + 0.5
        else:
            # Load the first TIFF file
            tiff_file = tiff_files[0]
            try:
                img_data = tifffile.imread(tiff_file)
                
                # Handle different image formats
                if img_data.ndim == 3:
                    # Multi-frame TIFF, take middle frame
                    image = img_data[img_data.shape[0] // 2]
                elif img_data.ndim == 2:
                    # Single frame
                    image = img_data
                else:
                    raise ValueError(f"Unexpected image dimensions: {img_data.shape}")
                
                # Ensure single channel
                if image.ndim == 3:
                    image = image.mean(axis=2)
                
                # Convert to float32
                image = image.astype(np.float32)
                
            except Exception as e:
                print(f"Error loading {tiff_file}: {e}, using random image")
                image = np.random.randn(*self.input_resolution) * 0.1 + 0.5
        
        # Find H5 files for emitter data
        h5_files = list(sample_dir.glob('*.h5')) + list(sample_dir.glob('*.hdf5'))
        
        if not h5_files:
            # If no H5 files, generate random emitter data
            num_emitters = np.random.randint(1, 20)  # Random number of emitters
            emitter_positions = np.random.rand(num_emitters, 2)
            emitter_photons = np.random.exponential(1000, num_emitters)  # Random photon counts
        else:
            # Load emitter data from H5 file
            h5_file = h5_files[0]
            try:
                with h5py.File(h5_file, 'r') as f:
                    if 'emitters' in f:
                        # Check if emitters is a group or dataset
                        if isinstance(f['emitters'], h5py.Group):
                            # It's a group, look for xyz and phot datasets
                            if 'xyz' in f['emitters']:
                                emitters_xyz = f['emitters']['xyz'][:]
                                # Format: [x, y, z], we only need x, y
                                positions = emitters_xyz[:, :2]  # x, y coordinates
                                
                                # Try to get photon counts
                                if 'phot' in f['emitters']:
                                    emitter_photons = f['emitters']['phot'][:]
                                else:
                                    # Default photon counts if not available
                                    emitter_photons = np.ones(len(positions)) * 1000
                            else:
                                # No xyz dataset, use random data
                                num_emitters = np.random.randint(1, 20)
                                emitter_positions = np.random.rand(num_emitters, 2)
                                emitter_photons = np.random.exponential(1000, num_emitters)
                                return image, emitter_positions, num_emitters, emitter_photons
                        else:
                            # It's a dataset, use directly
                            emitters = f['emitters'][:]
                            # Assume format: [x, y, intensity, ...]
                            positions = emitters[:, :2]  # x, y coordinates
                            if emitters.shape[1] > 2:
                                emitter_photons = emitters[:, 2]  # intensity/photon counts
                            else:
                                emitter_photons = np.ones(len(positions)) * 1000
                        
                        # Normalize coordinates to [0, 1]
                        if positions.size > 0:
                            # Assume coordinates are in pixel units, normalize by image size
                            h, w = image.shape
                            positions[:, 0] /= w  # x coordinates
                            positions[:, 1] /= h  # y coordinates
                            
                            # Clip to [0, 1] range
                            positions = np.clip(positions, 0, 1)
                            
                            # Use actual number of emitters from data
                            num_emitters = len(positions)
                            emitter_positions = positions
                        else:
                            # No valid positions found
                            num_emitters = np.random.randint(1, 20)
                            emitter_positions = np.random.rand(num_emitters, 2)
                            emitter_photons = np.random.exponential(1000, num_emitters)
                    else:
                        # No emitters in H5 file
                        num_emitters = np.random.randint(1, 20)
                        emitter_positions = np.random.rand(num_emitters, 2)
                        emitter_photons = np.random.exponential(1000, num_emitters)
                        
            except Exception as e:
                print(f"Error loading {h5_file}: {e}, using random emitter data")
                num_emitters = np.random.randint(1, 20)
                emitter_positions = np.random.rand(num_emitters, 2)
                emitter_photons = np.random.exponential(1000, num_emitters)
        
        return image, emitter_positions, num_emitters, emitter_photons
    
    def _generate_target_maps(self, emitter_positions: np.ndarray, 
                            target_h: int, target_w: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate probability and location maps for given resolution
        
        Args:
            emitter_positions: (N, 2) normalized coordinates in [0, 1]
            target_h, target_w: Target resolution
        
        Returns:
            prob_map: (1, target_h, target_w) binary probability map
            loc_map: (2, target_h, target_w) location offset map
        """
        prob_map = torch.zeros(1, target_h, target_w)
        loc_map = torch.zeros(2, target_h, target_w)
        
        if len(emitter_positions) == 0:
            return prob_map, loc_map
        
        # Convert normalized coordinates to pixel coordinates
        pixel_coords = emitter_positions * np.array([target_w, target_h])
        
        for px, py in pixel_coords:
            # Get integer pixel coordinates
            ix, iy = int(px), int(py)
            
            # Ensure within bounds
            if 0 <= ix < target_w and 0 <= iy < target_h:
                prob_map[0, iy, ix] = 1.0
                
                # Store sub-pixel offsets
                offset_x = px - ix  # [0, 1)
                offset_y = py - iy  # [0, 1)
                
                # Normalize offsets to [-1, 1] for tanh output
                loc_map[0, iy, ix] = 2 * offset_x - 1
                loc_map[1, iy, ix] = 2 * offset_y - 1
        
        return prob_map, loc_map


class VARTrainer:
    """
    Trainer for VAR Emitter Prediction Model
    Implements VAR-specific training procedures
    """
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Use GPU if available, fallback to CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using device: {self.device} ({torch.cuda.get_device_name()})")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU memory: {total_memory:.1f} GB")
            print(f"Current batch size: {self.config['training']['batch_size']}")
            print("If you encounter CUDA out of memory errors, reduce batch_size in config file")
        else:
            self.device = torch.device('cpu')
            print(f"Using device: {self.device} (CUDA not available)")
        self.setup_logging()
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        self.setup_loss()
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Set warmup steps
        self.warmup_steps = self.config['training'].get('warmup_steps', 1000)
    
    def setup_logging(self):
        """Setup logging and wandb"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Ensure outputs directory exists
        Path('outputs').mkdir(exist_ok=True)
        
        # Check if wandb should be disabled
        use_wandb = self.config['logging'].get('use_wandb', True)
        wandb_project = self.config['logging'].get('wandb_project')
        
        if use_wandb and wandb_project:
            try:
                wandb.init(
                    project=wandb_project,
                    entity=self.config['logging'].get('wandb_entity'),
                    config=self.config,
                    mode='offline' if self.config['logging'].get('wandb_offline', False) else 'online',
                    dir='outputs'  # Set wandb output directory
                )
                self.logger.info("Wandb initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize wandb: {e}")
                self.logger.info("Continuing without wandb logging")
        else:
            self.logger.info("Wandb logging disabled")
    
    def setup_model(self):
        """Initialize model"""
        model_config = self.config['model'].copy()
        # Remove 'type' field if present as it's not a model parameter
        model_config.pop('type', None)
        self.model = TrueVAREmitterPredictor(**model_config)
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def setup_data(self):
        """Setup data loaders"""
        data_config = self.config['data']
        train_config = self.config['training']
        
        # Use MultiScaleEmitterDataset
        self.logger.info("Using MultiScaleEmitterDataset")
        
        # Training dataset
        self.train_dataset = MultiScaleEmitterDataset(
            data_path=data_config['train_path'],
            input_resolution=tuple(train_config['input_resolution']),
            target_resolutions=train_config.get('target_resolutions', {
                'scale_1': (80, 80),
                'scale_2': (40, 40),
                'scale_3': (20, 20)
            }),
            augment=True
        )
        
        # Validation dataset
        self.val_dataset = MultiScaleEmitterDataset(
            data_path=data_config['val_path'],
            input_resolution=tuple(train_config['input_resolution']),
            target_resolutions=train_config.get('target_resolutions', {
                'scale_1': (80, 80),
                'scale_2': (40, 40),
                'scale_3': (20, 20)
            }),
            augment=False
        )
        
        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory'],
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory']
        )
        
        self.logger.info(f"Training samples: {len(self.train_dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_dataset)}")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        train_config = self.config['training']
        opt_config = self.config.get('optimizer', {})
        
        # Optimizer
        optimizer_type = opt_config.get('type', 'AdamW')
        if optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=train_config['learning_rate'],
                weight_decay=train_config.get('weight_decay', 0.05),
                betas=opt_config.get('betas', [0.9, 0.95]),
                eps=opt_config.get('eps', 1e-8)
            )
        
        # Simple scheduler (optional)
        self.scheduler = None
        if 'scheduler' in self.config:
            sched_config = self.config['scheduler']
            if sched_config.get('type') == 'cosine' and 'T_max' in sched_config:
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=sched_config['T_max'],
                    eta_min=sched_config.get('min_lr', 1e-6)
                )
        
        # Set default scheduler if none configured
        if self.scheduler is None:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        
        # Warmup steps
        self.warmup_steps = train_config.get('warmup_steps', 0)
    
    def setup_loss(self):
        """Setup loss function"""
        loss_config = self.config['loss']
        self.criterion = VAREmitterLoss(**loss_config)
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step with CUDA memory error handling"""
        self.model.train()
        
        try:
            images = batch['image'].to(self.device)
            targets = {}
            for key, target_dict in batch['targets'].items():
                targets[key] = {
                    'prob_map': target_dict['prob_map'].to(self.device),
                    'loc_map': target_dict['loc_map'].to(self.device)
                }
            
            # Add emitter count to targets if available
            if 'emitter_count' in batch:
                targets['emitter_count'] = batch['emitter_count']
            
            # Forward pass
            predictions = self.model(images)
            
            # Calculate loss
            losses = self.criterion(predictions, targets)
            total_loss = losses['total']
            
            # Debug: Print loss components for first few steps
            if self.global_step < 5:
                print(f"\nStep {self.global_step} Debug:")
                print(f"Predictions keys: {list(predictions.keys())}")
                print(f"Targets keys: {list(targets.keys())}")
                if 'scale_0' in predictions:
                    pred_prob = predictions['scale_0']['prob_map']
                    print(f"Pred prob_map shape: {pred_prob.shape}, min: {pred_prob.min():.6f}, max: {pred_prob.max():.6f}, mean: {pred_prob.mean():.6f}")
                if 'scale_0' in targets:
                    target_prob = targets['scale_0']['prob_map']
                    print(f"Target prob_map shape: {target_prob.shape}, min: {target_prob.min():.6f}, max: {target_prob.max():.6f}, mean: {target_prob.mean():.6f}")
                print(f"Emitter count: {targets.get('emitter_count', 'Not found')}")
                print(f"Loss components: {losses}")
                print(f"Total loss: {total_loss.item():.6f}")
            
            # Backward pass
            total_loss.backward()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                self.logger.error(f"CUDA out of memory error: {e}")
                self.logger.error("Try reducing batch_size in config file")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise e
            else:
                raise e
        
        # Gradient clipping
        if self.config['training'].get('gradient_clip'):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training']['gradient_clip']
            )
        
        # Optimizer step
        if (self.global_step + 1) % self.config['training']['accumulation_steps'] == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Learning rate scheduling
            if self.global_step < self.warmup_steps:
                # Linear warmup
                lr = self.config['training']['learning_rate'] * (self.global_step + 1) / self.warmup_steps
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                self.scheduler.step()
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
    
    @torch.no_grad()
    def val_step(self, batch: Dict) -> Dict[str, float]:
        """Single validation step"""
        self.model.eval()
        
        images = batch['image'].to(self.device)
        targets = {}
        for key, target_dict in batch['targets'].items():
            targets[key] = {
                'prob_map': target_dict['prob_map'].to(self.device),
                'loc_map': target_dict['loc_map'].to(self.device)
            }
        
        # Add emitter count to targets if available
        if 'emitter_count' in batch:
            targets['emitter_count'] = batch['emitter_count']
        
        # Forward pass
        predictions = self.model(images)
        
        # Calculate loss
        losses = self.criterion(predictions, targets)
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting VAR training...")
        
        self.optimizer.zero_grad()
        
        for epoch in range(1000):  # Large number, will stop based on max_steps
            epoch_losses = []
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            for batch in pbar:
                # Training step
                losses = self.train_step(batch)
                epoch_losses.append(losses)
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config['logging']['log_every'] == 0:
                    avg_loss = np.mean([l['total'] for l in epoch_losses[-100:]])
                    lr = self.optimizer.param_groups[0]['lr']
                    
                    pbar.set_postfix({
                        'loss': f"{avg_loss:.4f}",
                        'lr': f"{lr:.2e}",
                        'step': self.global_step
                    })
                    
                    if wandb.run:
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/lr': lr,
                            'step': self.global_step
                        })
                
                # Validation
                if self.global_step % self.config['logging']['eval_every'] == 0:
                    val_losses = self.validate()
                    
                    if wandb.run:
                        wandb.log({
                            'val/loss': val_losses['total'],
                            'step': self.global_step
                        })
                    
                    # Save best model
                    if val_losses['total'] < self.best_val_loss:
                        self.best_val_loss = val_losses['total']
                        self.save_checkpoint('models/best_model.pt')
                
                # Save checkpoint
                if self.global_step % self.config['checkpointing']['save_every'] == 0:
                    self.save_checkpoint(f'models/checkpoint_{self.global_step}.pt')
                
                # Check if training is complete
                if self.global_step >= self.config['training']['max_steps']:
                    self.logger.info("Training completed!")
                    return
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation loop"""
        self.model.eval()
        
        val_losses = []
        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            losses = self.val_step(batch)
            val_losses.append(losses)
        
        # Average losses
        avg_losses = {}
        for key in val_losses[0].keys():
            avg_losses[key] = np.mean([l[key] for l in val_losses])
        
        self.logger.info(f"Validation loss: {avg_losses['total']:.4f}")
        return avg_losses
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        # Ensure the directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, filename)
        self.logger.info(f"Checkpoint saved: {filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_true_var.json',
                       help='Path to config file')
    args = parser.parse_args()
    
    trainer = VARTrainer(args.config)
    trainer.train()