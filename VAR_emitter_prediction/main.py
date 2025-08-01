#!/usr/bin/env python3
"""
Main training script for stable VAR emitter prediction.
This script integrates all components and provides a command-line interface.
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from train_var_emitter_stable import StableVARTrainer
from data_loader import EmitterDataLoader
from var_emitter_model_unified import UnifiedEmitterPredictor
from var_emitter_loss_stable import StableVAREmitterLoss, UnifiedPPXYZBLoss

def setup_logging(log_dir: str, run_id: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('VAR_Training')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    log_file = os.path.join(log_dir, f'{run_id}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def validate_config(config: dict) -> None:
    """Validate configuration parameters."""
    required_keys = [
        'model', 'loss', 'optimizer', 'training', 'data', 'logging'
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate model config
    model_keys = ['input_size', 'target_sizes', 'base_channels', 'embed_dim']
    for key in model_keys:
        if key not in config['model']:
            raise ValueError(f"Missing required model config key: {key}")
    
    # Validate data paths
    if not os.path.exists(config['data']['train_path']):
        raise ValueError(f"Training data path does not exist: {config['data']['train_path']}")
    
    if not os.path.exists(config['data']['val_path']):
        raise ValueError(f"Validation data path does not exist: {config['data']['val_path']}")

def create_model(config: dict, device: torch.device) -> torch.nn.Module:
    """Create and initialize the model."""
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
    
    model = model.to(device)
    
    # Enable mixed precision if available
    if hasattr(torch.cuda, 'amp') and device.type == 'cuda':
        model = torch.jit.script(model) if config.get('jit_compile', False) else model
    
    return model

def create_loss_function(config: dict, device: torch.device) -> torch.nn.Module:
    """Create loss function based on configuration."""
    if config['loss'].get('use_unified_loss', True):
        loss_fn = UnifiedPPXYZBLoss(
            channel_weights=config['loss']['channel_weights'],
            pos_weight=config['loss']['pos_weight'],
            eps=config['loss']['eps']
        )
    else:
        loss_fn = StableVAREmitterLoss(
            scale_weights=config['loss']['scale_weights'],
            eps=config['loss']['eps'],
            warmup_epochs=config['loss']['warmup_epochs']
        )
    
    return loss_fn.to(device)

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train stable VAR emitter prediction model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--dry-run', action='store_true',
                       help='Perform a dry run without actual training')
    
    args = parser.parse_args()
    
    # Load and validate configuration
    config = load_config(args.config)
    validate_config(config)
    
    # Setup device
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Setup logging
    logger = setup_logging(config['logging']['log_dir'], config['run_id'])
    logger.info(f"Starting training with config: {args.config}")
    logger.info(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed(42)
        cudnn.deterministic = True
        cudnn.benchmark = False
    
    try:
        # Create model
        logger.info("Creating model...")
        model = create_model(config, device)
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create loss function
        logger.info("Creating loss function...")
        loss_fn = create_loss_function(config, device)
        
        # Create data loaders
        logger.info("Creating data loaders...")
        data_loader = EmitterDataLoader(config)
        train_loader = data_loader.get_train_loader()
        val_loader = data_loader.get_val_loader()
        
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = StableVARTrainer(
            model=model,
            loss_fn=loss_fn,
            config=config,
            device=device,
            logger=logger
        )
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            start_epoch = trainer.load_checkpoint(args.resume)
        
        # Dry run check
        if args.dry_run:
            logger.info("Performing dry run...")
            trainer.validate(val_loader, epoch=0)
            logger.info("Dry run completed successfully")
            return
        
        # Start training
        logger.info("Starting training...")
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            start_epoch=start_epoch
        )
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()