# Training Scripts

This folder contains scripts for training the SRCNN (Super-Resolution Convolutional Neural Network) model.

## Available Training Scripts

- `train_srcnn.py` - Standard SRCNN training with cropped patches
- `train_srcnn_full_resolution.py` - SRCNN training with full resolution images

## Directory Structure

- `core/` - Training configuration and trainer classes for SRCNN
- `inference/` - Inference utilities for trained SRCNN models
- `managers/` - Checkpoint management
- `steps/` - Metric tracking utilities
- `train_utils/` - Model architecture, loss functions, and training utilities

## Usage

For detailed usage instructions, see `train_utils/SRCNN_README.md`.