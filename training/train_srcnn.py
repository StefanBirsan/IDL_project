"""
Complete SRCNN Face Super-Resolution Training Script

This script implements the full training pipeline for SRCNN:
- Loads face datasets (FFHQ or similar)
- Implements 9-5-5 SRCNN architecture
- Uses layer-specific learning rates (1e-4 for early layers, 1e-5 for reconstruction)
- MSE loss for PSNR optimization
- Checkpointing and ONNX export

Usage:
    uv run training/train_srcnn.py --data-dir dataset/ffhq --batch-size 64 --num-epochs 100
    
    Or with custom parameters:
    uv run training/train_srcnn.py --data-dir dataset/ffhq --scale-factor 4 --crop-size 33 \\
        --batch-size 128 --num-epochs 200 --lr-early 1e-4 --lr-recon 1e-5
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.core.config_srcnn import SRCNNTrainingConfig
from training.core.trainer_srcnn import SRCNNTrainer
from training.train_utils.face_sr_dataset import get_face_sr_dataloaders


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='SRCNN Face Super-Resolution Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default settings
  python training/train_srcnn.py --data-dir dataset/ffhq
  
  # 4x super-resolution with custom batch size
  python training/train_srcnn.py --data-dir dataset/ffhq --scale-factor 4 --batch-size 32
  
  # Resume from checkpoint
  python training/train_srcnn.py --data-dir dataset/ffhq --resume checkpoints/srcnn/checkpoint_epoch_0050.pth
        """
    )
    
    # ============ DATA ARGUMENTS ============
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to face dataset (FFHQ or similar)')
    parser.add_argument('--crop-size', type=int, default=33,
                        help='Training patch size (default: 33x33)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # ============ MODEL ARGUMENTS ============
    parser.add_argument('--scale-factor', type=int, default=2, choices=[2, 4, 8],
                        help='Super-resolution scale factor (default: 2)')
    parser.add_argument('--intermediate-channels', type=int, default=32,
                        help='Channels in middle layer (default: 32)')
    
    # ============ TRAINING ARGUMENTS ============
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    
    # ============ LEARNING RATE ARGUMENTS ============
    parser.add_argument('--lr-early', type=float, default=1e-4,
                        help='Learning rate for layers 1-2 (default: 1e-4)')
    parser.add_argument('--lr-recon', type=float, default=1e-5,
                        help='Learning rate for layer 3 (default: 1e-5)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    
    # ============ VALIDATION ARGUMENTS ============
    parser.add_argument('--val-interval', type=int, default=5,
                        help='Validate every N epochs (default: 5)')
    
    # ============ CHECKPOINTING ARGUMENTS ============
    parser.add_argument('--save-dir', type=str, default='checkpoints/srcnn',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--export-onnx', action='store_true', default=True,
                        help='Export model to ONNX after training')
    
    # ============ HARDWARE ARGUMENTS ============
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to train on')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # ============ LOGGING ARGUMENTS ============
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log metrics every N batches')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print training progress')
    
    return parser.parse_args()


def create_config_from_args(args) -> SRCNNTrainingConfig:
    """Create training config from command-line arguments"""
    config = SRCNNTrainingConfig(
        # Model
        scale_factor=args.scale_factor,
        intermediate_channels=args.intermediate_channels,
        crop_size=args.crop_size,
        
        # Training
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr_early_layers=args.lr_early,
        lr_reconstruction_layer=args.lr_recon,
        momentum=args.momentum,
        
        # Data
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        
        # Validation
        val_interval=args.val_interval,
        
        # Checkpointing
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        export_final_model=args.export_onnx,
        
        # Hardware
        device=args.device,
        seed=args.seed,
        
        # Logging
        log_interval=args.log_interval,
        verbose=args.verbose
    )
    
    return config


def main():
    """Main training script"""
    args = parse_arguments()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Using CPU instead.")
        args.device = 'cpu'
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Verify dataset exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}")
    
    print("\n" + "="*70)
    print("SRCNN FACE SUPER-RESOLUTION TRAINING")
    print("="*70)
    print(config.summary())
    
    # Load data
    print("Loading datasets...")
    train_loader, val_loader = get_face_sr_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        scale_factor=args.scale_factor,
        crop_size=args.crop_size,
        num_workers=args.num_workers,
        normalize=True
    )
    print(f"Train batches: {len(train_loader)}")
    if val_loader is not None:
        print(f"Val batches:   {len(val_loader)}")
    
    # Create trainer
    trainer = SRCNNTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer, checkpoint = SRCNNTrainer.load_checkpoint(args.resume, device=args.device)
    
    # Train
    try:
        training_history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader
        )
        print("\n✓ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer._save_checkpoint(is_best=False)
    except Exception as e:
        print(f"\n✗ Training failed with error: {str(e)}")
        raise


if __name__ == '__main__':
    main()
