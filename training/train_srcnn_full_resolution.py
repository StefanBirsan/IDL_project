"""
SRCNN Full-Resolution Training Script

Trains SRCNN on complete high-resolution images instead of random patches.

Setup:
    1. Generate LR images from HR images:
       python training/train_utils/create_lr_images.py --input-dir dataset/images1024x1024 --output-dir dataset --scale-factor 2
    
    2. Train on full-resolution images:
       python training/train_srcnn_full_resolution.py --data-root dataset --batch-size 4 --num-epochs 100

This approach:
- Preserves spatial information across the entire image
- Better for training on faces at native resolution
- Requires more GPU memory (use smaller batch sizes)
- Uses fewer training iterations (no patch extraction)
"""

import argparse
import sys
from pathlib import Path
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from training.train_utils.srcnn import SRCNN
from training.train_utils.full_resolution_dataset import get_full_resolution_dataloaders
from training.core.config_srcnn import SRCNNTrainingConfig


class FullResolutionTrainer:
    """Trainer for full-resolution SRCNN."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: SRCNNTrainingConfig,
                 device: str = 'cuda'):
        """
        Initialize trainer.
        
        Args:
            model: SRCNN model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Training configuration
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Setup optimizer with layer-specific learning rates
        self.optimizer = self._setup_optimizer()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf'),
        }
    
    def _setup_optimizer(self):
        """Setup optimizer with layer-specific learning rates."""
        # Group parameters by layer
        layer1_params = list(self.model.layer1.parameters())
        layer2_params = list(self.model.layer2.parameters())
        layer3_params = list(self.model.layer3.parameters())
        
        optimizer = optim.SGD([
            {'params': layer1_params, 'lr': self.config.lr_early_layers},
            {'params': layer2_params, 'lr': self.config.lr_early_layers},
            {'params': layer3_params, 'lr': self.config.lr_reconstruction_layer}
        ], momentum=self.config.momentum)
        
        return optimizer
    
    def train_epoch(self) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            lr_image = batch['lr_image'].to(self.device)
            hr_image = batch['hr_image'].to(self.device)
            
            # Forward pass
            sr_image = self.model(lr_image)
            loss = self.criterion(sr_image, hr_image)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validate on validation set. Returns average loss."""
        if self.val_loader is None:
            return float('nan')
        
        self.model.eval()
        total_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc="Validation", leave=False)
        for batch in pbar:
            lr_image = batch['lr_image'].to(self.device)
            hr_image = batch['hr_image'].to(self.device)
            
            sr_image = self.model(lr_image)
            loss = self.criterion(sr_image, hr_image)
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self, num_epochs: int, save_dir: str):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Starting Full-Resolution SRCNN Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Model: {self.config.model_name}")
        print(f"Scale factor: {self.config.scale_factor}x")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Training epochs: {num_epochs}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Save dir: {save_path}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1:3d}/{num_epochs}]")
            
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            if self.val_loader:
                val_loss = self.validate()
                self.history['val_loss'].append(val_loss)
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss:   {val_loss:.6f}")
                
                # Save best model
                if val_loss < self.history['best_val_loss']:
                    self.history['best_val_loss'] = val_loss
                    best_path = save_path / 'best_model.pth'
                    torch.save(self.model.state_dict(), best_path)
                    print(f"  ✓ Best model saved")
            else:
                print(f"  Train Loss: {train_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                ckpt_path = save_path / f'checkpoint_epoch_{epoch:04d}.pth'
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"  ✓ Checkpoint saved: {ckpt_path.name}")
        
        # Save training history
        history_path = save_path / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining complete!")
        print(f"History saved to: {history_path}")


def main():
    parser = argparse.ArgumentParser(
        description='SRCNN Full-Resolution Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training on full-resolution images
  python training/train_srcnn_full_resolution.py --data-root dataset --num-epochs 100
  
  # 4x super-resolution with smaller batch size
  python training/train_srcnn_full_resolution.py --data-root dataset --scale-factor 4 --batch-size 2 --num-epochs 100
  
  # Resume from checkpoint
  python training/train_srcnn_full_resolution.py --data-root dataset --resume checkpoints/srcnn_full_res/checkpoint_epoch_0050.pth --num-epochs 100
        """
    )
    
    # Data arguments
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory with hr_images/ and lr_images_Nx/ subdirectories')
    
    # Model arguments
    parser.add_argument('--scale-factor', type=int, default=2, choices=[2, 4, 8],
                        help='Super-resolution scale factor (default: 2)')
    parser.add_argument('--intermediate-channels', type=int, default=32,
                        help='Channels in middle layer (default: 32)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (default: 4, use smaller for full images)')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    
    # Learning rate arguments
    parser.add_argument('--lr-early', type=float, default=1e-4,
                        help='Learning rate for layers 1-2 (default: 1e-4)')
    parser.add_argument('--lr-recon', type=float, default=1e-5,
                        help='Learning rate for layer 3 (default: 1e-5)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    
    # Validation arguments
    parser.add_argument('--val-interval', type=int, default=5,
                        help='Validate every N epochs (default: 5)')
    parser.add_argument('--train-fraction', type=float, default=0.8,
                        help='Fraction of data for training (default: 0.8)')
    
    # Checkpointing arguments
    parser.add_argument('--save-dir', type=str, default='checkpoints/srcnn_full_resolution',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to train on')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of data loading workers (default: 2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create config
    config = SRCNNTrainingConfig(
        model_name='srcnn_full_resolution',
        scale_factor=args.scale_factor,
        intermediate_channels=args.intermediate_channels,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr_early_layers=args.lr_early,
        lr_reconstruction_layer=args.lr_recon,
        momentum=args.momentum,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
    )
    
    # Create model
    model = SRCNN(
        in_channels=3,
        intermediate_channels=args.intermediate_channels,
        scale_factor=args.scale_factor
    )
    
    # Load checkpoint if resuming
    if args.resume:
        print(f"Loading checkpoint from: {args.resume}")
        state_dict = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(state_dict)
    
    # Create dataloaders
    print(f"Loading data from: {args.data_root}")
    train_loader, val_loader = get_full_resolution_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        scale_factor=args.scale_factor,
        num_workers=args.num_workers,
        train_fraction=args.train_fraction
    )
    
    # Create trainer
    trainer = FullResolutionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device
    )
    
    # Train
    trainer.train(
        num_epochs=args.num_epochs,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
