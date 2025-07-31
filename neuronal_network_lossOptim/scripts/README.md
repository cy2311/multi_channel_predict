# Training and Inference Scripts

This directory contains shell scripts for training and inference:

## Training Scripts
- `train_network.sh` - Basic training script
- `train_network_fix.sh` - Fixed version training script
- `train_network_loc.sh` - Training with localization
- `train_network_loc_back.sh` - Training with localization and background

## Inference Scripts
- `inference_loc.sh` - Inference with localization

## Usage
Make sure the scripts are executable:
```bash
chmod +x *.sh
```

Then run the desired script:
```bash
./train_network.sh
```