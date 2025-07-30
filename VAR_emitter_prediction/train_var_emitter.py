import os
import sys
import time
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

# Import our modules
from var_emitter_model import VAREmitterPredictor
from var_emitter_loss import VAREmitterLoss, ProgressiveLoss
from var_dataset import create_dataloaders, InferenceDataset


class VAREmitterTrainer:
    """
    Trainer for VAR-based emitter prediction
    """
    
    def __init__(self,
                 model: VAREmitterPredictor,
                 train_loader,
                 val_loader,
                 loss_fn: VAREmitterLoss,
                 optimizer: optim.Optimizer,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cuda',
                 output_dir: str = './outputs',
                 use_amp: bool = True,
                 log_interval: int = 10,
                 save_interval: int = 5):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.use_amp = use_amp
        self.log_interval = log_interval
        self.save_interval = save_interval
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize AMP scaler
        self.scaler = GradScaler() if use_amp else None
        
        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir=self.output_dir / 'tensorboard')
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Move model to device
        self.model.to(device)
        
        print(f"Trainer initialized. Output directory: {self.output_dir}")
        print(f"Using device: {device}")
        print(f"AMP enabled: {use_amp}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        # Update loss function epoch for progressive training
        if hasattr(self.loss_fn, 'set_epoch'):
            self.loss_fn.set_epoch(self.current_epoch)
        
        epoch_losses = {}
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                predictions = self.model(batch['low_res_input'])
                
                # Prepare targets
                targets = {
                    'count': batch['count'],
                    'locations': batch['locations']
                }
                
                if 'prob_maps' in batch:
                    targets['prob_maps'] = batch['prob_maps']
                
                # Compute loss
                loss_dict = self.loss_fn(predictions, targets)
                total_loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()
            
            # Update global step
            self.global_step += 1
            
            # Accumulate losses
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value.item() if torch.is_tensor(value) else value)
            
            # Log to tensorboard
            if batch_idx % self.log_interval == 0:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}', 
                                         value.item() if torch.is_tensor(value) else value, 
                                         self.global_step)
                
                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/learning_rate', current_lr, self.global_step)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Average losses over epoch
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        val_losses = {}
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                with autocast(enabled=self.use_amp):
                    predictions = self.model(batch['low_res_input'])
                    
                    # Prepare targets
                    targets = {
                        'count': batch['count'],
                        'locations': batch['locations']
                    }
                    
                    if 'prob_maps' in batch:
                        targets['prob_maps'] = batch['prob_maps']
                    
                    # Compute loss
                    loss_dict = self.loss_fn(predictions, targets)
                
                # Accumulate losses
                for key, value in loss_dict.items():
                    if key not in val_losses:
                        val_losses[key] = []
                    val_losses[key].append(value.item() if torch.is_tensor(value) else value)
        
        # Average losses
        avg_val_losses = {key: np.mean(values) for key, values in val_losses.items()}
        
        # Log to tensorboard
        for key, value in avg_val_losses.items():
            self.writer.add_scalar(f'val/{key}', value, self.current_epoch)
        
        return avg_val_losses
    
    def train(self, num_epochs: int):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total_loss'])
                else:
                    self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"Train Loss: {train_losses['total_loss']:.4f}")
            print(f"Val Loss: {val_losses['total_loss']:.4f}")
            
            # Save checkpoint
            is_best = val_losses['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total_loss']
            
            if epoch % self.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best, val_losses['total_loss'])
            
            # Log epoch summary to tensorboard
            self.writer.add_scalars('epoch/loss', {
                'train': train_losses['total_loss'],
                'val': val_losses['total_loss']
            }, epoch)
        
        print("Training completed!")
        self.writer.close()
    
    def save_checkpoint(self, epoch: int, is_best: bool, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model saved with val_loss: {val_loss:.4f}")
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.global_step = checkpoint.get('global_step', 0)
        
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {epoch}")
        
        return epoch
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch data to device"""
        device_batch = {}
        
        for key, value in batch.items():
            if torch.is_tensor(value):
                device_batch[key] = value.to(self.device, non_blocking=True)
            elif isinstance(value, dict):
                device_batch[key] = {
                    k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                    for k, v in value.items()
                }
            else:
                device_batch[key] = value
        
        return device_batch


def create_model_and_optimizer(config: Dict) -> tuple:
    """Create model and optimizer from config"""
    
    # Create model
    model = VAREmitterPredictor(
        input_size=config['model']['input_size'],
        target_sizes=config['model']['target_sizes'],
        base_channels=config['model']['base_channels'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers']
    )
    
    # Create loss function
    if config['training']['progressive']:
        base_loss = VAREmitterLoss(
            count_weight=config['loss']['count_weight'],
            loc_weight=config['loss']['loc_weight'],
            recon_weight=config['loss']['recon_weight'],
            uncertainty_weight=config['loss']['uncertainty_weight']
        )
        loss_fn = ProgressiveLoss(
            base_loss=base_loss,
            warmup_epochs=config['training']['warmup_epochs'],
            scale_schedule=config['training']['scale_schedule']
        )
    else:
        loss_fn = VAREmitterLoss(
            count_weight=config['loss']['count_weight'],
            loc_weight=config['loss']['loc_weight'],
            recon_weight=config['loss']['recon_weight'],
            uncertainty_weight=config['loss']['uncertainty_weight']
        )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['learning_rate'],
        weight_decay=config['optimizer']['weight_decay'],
        betas=config['optimizer']['betas']
    )
    
    # Create scheduler
    scheduler = None
    if config['scheduler']['type'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=config['scheduler']['min_lr']
        )
    elif config['scheduler']['type'] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['scheduler']['factor'],
            patience=config['scheduler']['patience'],
            min_lr=config['scheduler']['min_lr']
        )
    
    return model, loss_fn, optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(description='Train VAR-based Emitter Predictor')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--tiff_dir', type=str, required=True, help='Directory containing TIFF files')
    parser.add_argument('--emitter_dir', type=str, required=True, help='Directory containing emitter H5 files')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded config from {args.config}")
    print(json.dumps(config, indent=2))
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        tiff_dir=args.tiff_dir,
        emitter_dir=args.emitter_dir,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        train_val_split=config['training']['train_val_split'],
        low_res_size=config['model']['input_size'],
        high_res_sizes=config['model']['target_sizes']
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model and optimizer
    print("Creating model and optimizer...")
    model, loss_fn, optimizer, scheduler = create_model_and_optimizer(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = VAREmitterTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        use_amp=config['training']['use_amp'],
        log_interval=config['training']['log_interval'],
        save_interval=config['training']['save_interval']
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume) + 1
        print(f"Resuming training from epoch {start_epoch}")
    
    # Save config
    config_path = Path(args.output_dir) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Start training
    remaining_epochs = config['training']['num_epochs'] - start_epoch
    if remaining_epochs > 0:
        trainer.current_epoch = start_epoch
        trainer.train(remaining_epochs)
    else:
        print("Training already completed!")


if __name__ == '__main__':
    main()