#!/usr/bin/env python3
"""
Stable VAR Emitter Training Script
Implements numerical stability improvements and unified 6-channel architecture
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from var_emitter_model_unified import UnifiedEmitterPredictor
from var_emitter_loss_stable import StableVAREmitterLoss, UnifiedPPXYZBLoss
from var_dataset import VARDataset


class StableVARTrainer:
    """
    Stable trainer for VAR emitter prediction with numerical stability improvements
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Setup logging
        self.setup_logging()
        
        # Initialize model
        self.model = self.build_model()
        
        # Initialize loss function
        self.criterion = self.build_loss_function()
        
        # Initialize optimizer with gradient clipping
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self.build_data_loaders()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Tensorboard logging
        self.writer = SummaryWriter(config.get('log_dir', 'logs/tensorboard'))
        
        # Gradient clipping
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
        self.logger.info(f"Initialized trainer on device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'training_{self.config.get("run_id", "stable")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def build_model(self) -> nn.Module:
        """Build unified emitter predictor model"""
        model_config = self.config.get('model', {})
        
        model = UnifiedEmitterPredictor(
            input_size=model_config.get('input_size', 40),
            target_sizes=model_config.get('target_sizes', [40, 80, 160, 320]),
            base_channels=model_config.get('base_channels', 64),
            embed_dim=model_config.get('embed_dim', 512),
            num_heads=model_config.get('num_heads', 8),
            num_layers=model_config.get('num_layers', 12),
            codebook_size=model_config.get('codebook_size', 1024),
            embedding_dim=model_config.get('embedding_dim', 256)
        )
        
        model = model.to(self.device)
        
        # Load pretrained weights if specified
        if 'pretrained_path' in model_config:
            self.load_pretrained_weights(model, model_config['pretrained_path'])
        
        return model
    
    def build_loss_function(self) -> nn.Module:
        """Build stable loss function"""
        loss_config = self.config.get('loss', {})
        
        if loss_config.get('use_unified_loss', True):
            # Use unified PPXYZBG loss
            criterion = UnifiedPPXYZBLoss(
                device=self.device,
                channel_weights=loss_config.get('channel_weights'),
                pos_weight=loss_config.get('pos_weight', 2.0)
            )
        else:
            # Use stable multi-scale loss
            criterion = StableVAREmitterLoss(
                device=self.device,
                channel_weights=loss_config.get('channel_weights'),
                pos_weight=loss_config.get('pos_weight', 2.0),
                scale_weights=loss_config.get('scale_weights'),
                eps=loss_config.get('eps', 1e-4),
                warmup_epochs=loss_config.get('warmup_epochs', 20)
            )
        
        return criterion
    
    def build_optimizer(self) -> optim.Optimizer:
        """Build optimizer with improved stability"""
        optim_config = self.config.get('optimizer', {})
        
        optimizer_type = optim_config.get('type', 'adamw')
        lr = optim_config.get('lr', 1e-4)
        weight_decay = optim_config.get('weight_decay', 1e-2)
        
        if optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=optim_config.get('betas', (0.9, 0.999)),
                eps=optim_config.get('eps', 1e-8)
            )
        elif optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=optim_config.get('betas', (0.9, 0.999)),
                eps=optim_config.get('eps', 1e-8)
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        return optimizer
    
    def build_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Build learning rate scheduler"""
        sched_config = self.config.get('scheduler', {})
        
        if not sched_config.get('enabled', True):
            return None
        
        scheduler_type = sched_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('training', {}).get('epochs', 100),
                eta_min=sched_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 30),
                gamma=sched_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=sched_config.get('factor', 0.5),
                patience=sched_config.get('patience', 10),
                min_lr=sched_config.get('min_lr', 1e-6)
            )
        else:
            return None
        
        return scheduler
    
    def build_data_loaders(self) -> tuple:
        """Build training and validation data loaders"""
        data_config = self.config.get('data', {})
        
        # Training dataset
        train_dataset = VARDataset(
            data_path=data_config.get('train_path'),
            mode='train',
            transform=data_config.get('train_transform'),
            unified_format=True  # Use unified 6-channel format
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=data_config.get('batch_size', 8),
            shuffle=True,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True
        )
        
        # Validation dataset
        val_dataset = VARDataset(
            data_path=data_config.get('val_path'),
            mode='val',
            transform=data_config.get('val_transform'),
            unified_format=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=data_config.get('val_batch_size', data_config.get('batch_size', 8)),
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True
        )
        
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        # Update loss function epoch
        if hasattr(self.criterion, 'set_epoch'):
            self.criterion.set_epoch(self.current_epoch)
        
        epoch_losses = []
        epoch_metrics = {}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            inputs = batch['input'].to(self.device)
            targets = {k: v.to(self.device) if torch.is_tensor(v) else v 
                      for k, v in batch.items() if k != 'input'}
            
            # Forward pass
            self.optimizer.zero_grad()
            
            try:
                predictions = self.model(inputs)
                
                # Compute loss
                if isinstance(self.criterion, UnifiedPPXYZBLoss):
                    # Unified loss computation
                    total_loss = 0.0
                    for scale_key, scale_pred in predictions.items():
                        if 'unified_output' in scale_pred:
                            output = scale_pred['unified_output']
                            target = targets.get(f'{scale_key}_unified', targets.get('unified'))
                            weight = targets.get(f'{scale_key}_weight', targets.get('weight'))
                            
                            if target is not None and weight is not None:
                                loss = self.criterion(output, target, weight)
                                total_loss += loss.sum()
                    
                    loss_dict = {'total_loss': total_loss}
                else:
                    # Multi-scale loss computation
                    loss_dict = self.criterion(predictions, targets)
                    total_loss = loss_dict['total_loss']
                
                # Check for NaN/Inf
                if not torch.isfinite(total_loss):
                    self.logger.warning(f"Non-finite loss detected: {total_loss}")
                    continue
                
                # Backward pass with gradient clipping
                total_loss.backward()
                
                # Gradient clipping for stability
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Record loss
                epoch_losses.append(total_loss.item())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                # Log to tensorboard
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/batch_loss', total_loss.item(), global_step)
                
                # Log additional metrics
                for key, value in loss_dict.items():
                    if key != 'total_loss' and torch.is_tensor(value):
                        self.writer.add_scalar(f'train/{key}', value.item(), global_step)
                
            except RuntimeError as e:
                self.logger.error(f"Training error at batch {batch_idx}: {e}")
                continue
        
        # Compute epoch metrics
        epoch_metrics['loss'] = np.mean(epoch_losses) if epoch_losses else float('inf')
        epoch_metrics['lr'] = self.optimizer.param_groups[0]['lr']
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        epoch_losses = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                inputs = batch['input'].to(self.device)
                targets = {k: v.to(self.device) if torch.is_tensor(v) else v 
                          for k, v in batch.items() if k != 'input'}
                
                try:
                    # Forward pass
                    predictions = self.model(inputs)
                    
                    # Compute loss
                    if isinstance(self.criterion, UnifiedPPXYZBLoss):
                        total_loss = 0.0
                        for scale_key, scale_pred in predictions.items():
                            if 'unified_output' in scale_pred:
                                output = scale_pred['unified_output']
                                target = targets.get(f'{scale_key}_unified', targets.get('unified'))
                                weight = targets.get(f'{scale_key}_weight', targets.get('weight'))
                                
                                if target is not None and weight is not None:
                                    loss = self.criterion(output, target, weight)
                                    total_loss += loss.sum()
                    else:
                        loss_dict = self.criterion(predictions, targets)
                        total_loss = loss_dict['total_loss']
                    
                    if torch.isfinite(total_loss):
                        epoch_losses.append(total_loss.item())
                    
                    pbar.set_postfix({'val_loss': f'{total_loss.item():.4f}'})
                    
                except RuntimeError as e:
                    self.logger.error(f"Validation error at batch {batch_idx}: {e}")
                    continue
        
        val_metrics = {
            'loss': np.mean(epoch_losses) if epoch_losses else float('inf')
        }
        
        return val_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'models'))
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with validation loss: {self.best_val_loss:.4f}")
    
    def load_pretrained_weights(self, model: nn.Module, pretrained_path: str):
        """Load pretrained weights"""
        try:
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.logger.info(f"Loaded pretrained weights from {pretrained_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load pretrained weights: {e}")
    
    def train(self):
        """Main training loop"""
        num_epochs = self.config.get('training', {}).get('epochs', 100)
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            
            # Validation
            val_metrics = self.validate_epoch()
            self.val_losses.append(val_metrics['loss'])
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Logging
            self.logger.info(
                f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, LR: {train_metrics['lr']:.2e}"
            )
            
            # Tensorboard logging
            self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/lr', train_metrics['lr'], epoch)
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            if epoch % self.config.get('save_frequency', 10) == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        self.logger.info(f"Training completed. Best validation loss: {self.best_val_loss:.4f}")
        self.writer.close()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Stable VAR Emitter Training')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--run_id', type=str, default=None,
                       help='Run identifier for logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.device:
        config['device'] = args.device
    if args.run_id:
        config['run_id'] = args.run_id
    
    # Initialize and start training
    trainer = StableVARTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()