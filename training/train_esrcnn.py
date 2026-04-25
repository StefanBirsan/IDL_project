"""
Training script for Enhanced SRCNN (ESRCNN) - Face Super-Resolution

This script implements deep residual SR with perceptual loss for high-quality face upscaling.

Usage:
    python training/train_esrcnn.py --data-dir dataset/ffhq --scale-factor 2
    
Advanced usage:
    python training/train_esrcnn.py \
        --data-dir dataset/ffhq \
        --scale-factor 2 \
        --num-residual-blocks 10 \
        --batch-size 16 \
        --use-perceptual-loss \
        --perceptual-weight 0.1
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.train_utils.esrcnn import EnhancedSRCNN, PerceptualLoss
from training.train_utils.losses_sr import CharbonnierLoss, SSIMLoss
from training.core.config_esrcnn import ESRCNNConfig
from training.train_utils.face_sr_dataset import get_face_sr_dataloaders
from training.managers import CheckpointManager


class ESRCNNTrainer:
    """Trainer for Enhanced SRCNN with perceptual loss"""
    
    def __init__(self, config: ESRCNNConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Create model
        self.model = EnhancedSRCNN(
            in_channels=config.input_channels,
            num_features=config.num_features,
            num_residual_blocks=config.num_residual_blocks,
            scale_factor=config.scale_factor,
            use_global_skip=config.use_global_skip
        ).to(self.device)
        
        # Setup optimizer
        if config.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.learning_rate,
                betas=config.betas,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                betas=config.betas,
                weight_decay=config.weight_decay
            )
        
        # Setup learning rate scheduler
        if config.lr_scheduler == 'step':
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=config.lr_decay_epochs,
                gamma=config.lr_decay_factor
            )
        elif config.lr_scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs
            )
        elif config.lr_scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.lr_decay_factor,
                patience=10
            )
        
        # Setup losses
        if config.pixel_loss_type == 'l1':
            self.pixel_loss = CharbonnierLoss()
        else:
            self.pixel_loss = nn.MSELoss()
        
        if config.use_perceptual_loss:
            self.perceptual_loss = PerceptualLoss(
                feature_layers=config.perceptual_layers
            ).to(self.device)
        else:
            self.perceptual_loss = None
        
        if config.loss_ssim_weight > 0:
            self.ssim_loss = SSIMLoss()
        else:
            self.ssim_loss = None
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(config.save_dir)
        
        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_psnr': [],
            'val_ssim': [],
            'epoch': []
        }
        
        self._print_info()
    
    def _print_info(self):
        """Print model and training info"""
        print(self.model.summary())
        print(self.config.summary())
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Compute combined loss
        
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        
        # Pixel loss
        pixel_loss = self.pixel_loss(pred, target)
        losses['pixel'] = pixel_loss * self.config.loss_pixel_weight
        
        # Perceptual loss
        if self.perceptual_loss is not None:
            percep_loss = self.perceptual_loss(pred, target)
            losses['perceptual'] = percep_loss * self.config.loss_perceptual_weight
        
        # SSIM loss
        if self.ssim_loss is not None:
            ssim_loss = self.ssim_loss(pred, target)
            losses['ssim'] = ssim_loss * self.config.loss_ssim_weight
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, (lr_images, hr_images) in enumerate(pbar):
            lr_images = lr_images.to(self.device)
            hr_images = hr_images.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    pred = self.model(lr_images)
                    losses = self.compute_loss(pred, hr_images)
                    loss = losses['total']
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(lr_images)
                losses = self.compute_loss(pred, hr_images)
                loss = losses['total']
                
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            if batch_idx % self.config.log_interval == 0:
                loss_str = f"Loss: {loss.item():.6f}"
                if 'perceptual' in losses:
                    loss_str += f" | Percep: {losses['perceptual'].item():.6f}"
                pbar.set_postfix_str(loss_str)
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = 0
        
        for lr_images, hr_images in tqdm(val_loader, desc="Validation"):
            lr_images = lr_images.to(self.device)
            hr_images = hr_images.to(self.device)
            
            pred = self.model(lr_images)
            losses = self.compute_loss(pred, hr_images)
            
            # Compute metrics
            mse = F.mse_loss(pred, hr_images)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            
            total_loss += losses['total'].item()
            total_psnr += psnr.item()
            num_batches += 1
        
        metrics = {
            'val_loss': total_loss / max(num_batches, 1),
            'val_psnr': total_psnr / max(num_batches, 1),
            'val_ssim': 0.0  # Placeholder
        }
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Full training loop"""
        print(f"\n{'='*70}")
        print(f"Starting Enhanced SRCNN Training")
        print(f"{'='*70}\n")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['epoch'].append(epoch)
            
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Validate
            if val_loader is not None and (epoch + 1) % self.config.val_interval == 0:
                val_metrics = self.validate(val_loader)
                self.training_history['val_loss'].append(val_metrics['val_loss'])
                self.training_history['val_psnr'].append(val_metrics['val_psnr'])
                
                print(f"  Val Loss: {val_metrics['val_loss']:.6f}")
                print(f"  Val PSNR: {val_metrics['val_psnr']:.2f} dB")
                
                # Save best model
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self._save_checkpoint(is_best=True)
                    print(f"  ✅ New best model saved!")
            
            # Learning rate scheduling
            if self.config.lr_scheduler == 'plateau' and val_loader is not None:
                self.scheduler.step(val_metrics['val_loss'])
            else:
                self.scheduler.step()
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint(is_best=False)
        
        print(f"\n{'='*70}")
        print(f"Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"{'='*70}\n")
        
        # Export to ONNX
        if self.config.export_final_model:
            self.export_to_onnx()
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
        }
        
        save_path = Path(self.config.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if is_best:
            checkpoint_path = save_path / 'best_model.pth'
            print(f"  💾 Saving best model to {checkpoint_path}")
        else:
            checkpoint_path = save_path / f'checkpoint_epoch_{self.current_epoch:04d}.pth'
        
        torch.save(checkpoint, checkpoint_path)
    
    def export_to_onnx(self):
        """Export model to ONNX format"""
        save_path = Path(self.config.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        onnx_path = save_path / 'esrcnn_face.onnx'
        
        self.model.eval()
        
        dummy_input = torch.randn(
            1,
            self.config.input_channels,
            self.config.crop_size,
            self.config.crop_size,
            device=self.device
        )
        
        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                str(onnx_path),
                input_names=['low_res_image'],
                output_names=['super_resolved_image'],
                dynamic_axes={
                    'low_res_image': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'super_resolved_image': {0: 'batch_size', 2: 'height', 3: 'width'}
                },
                opset_version=13,
                do_constant_folding=True,
                verbose=False
            )
            
            print(f"\n{'='*70}")
            print(f"ONNX Model Exported Successfully!")
            print(f"{'='*70}")
            print(f"Path: {onnx_path}")
            print(f"{'='*70}\n")
            
            return True
        except Exception as e:
            print(f"Error exporting to ONNX: {e}")
            return False


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Enhanced SRCNN Training',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to face dataset')
    parser.add_argument('--crop-size', type=int, default=48,
                        help='Training patch size')
    
    # Model
    parser.add_argument('--scale-factor', type=int, default=2, choices=[2, 4, 8],
                        help='Super-resolution scale factor')
    parser.add_argument('--num-residual-blocks', type=int, default=10,
                        help='Number of residual blocks')
    parser.add_argument('--num-features', type=int, default=64,
                        help='Number of feature channels')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=150,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    
    # Loss
    parser.add_argument('--use-perceptual-loss', action='store_true', default=True,
                        help='Use perceptual loss')
    parser.add_argument('--perceptual-weight', type=float, default=0.1,
                        help='Perceptual loss weight')
    parser.add_argument('--pixel-loss-type', type=str, default='l1',
                        choices=['l1', 'mse'], help='Pixel loss type')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                        help='Use mixed precision training')
    
    # Checkpointing
    parser.add_argument('--save-dir', type=str, default='checkpoints/esrcnn',
                        help='Save directory')
    parser.add_argument('--val-interval', type=int, default=5,
                        help='Validation interval')
    
    return parser.parse_args()


def main():
    """Main training entry point"""
    args = parse_args()
    
    # Create config
    config = ESRCNNConfig(
        data_dir=args.data_dir,
        crop_size=args.crop_size,
        scale_factor=args.scale_factor,
        num_residual_blocks=args.num_residual_blocks,
        num_features=args.num_features,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        use_perceptual_loss=args.use_perceptual_loss,
        loss_perceptual_weight=args.perceptual_weight,
        pixel_loss_type=args.pixel_loss_type,
        device=args.device,
        mixed_precision=args.mixed_precision,
        save_dir=args.save_dir,
        val_interval=args.val_interval,
    )
    
    config.validate()
    
    # Create trainer
    trainer = ESRCNNTrainer(config)
    
    # Create data loaders
    print("Loading dataset...")
    train_loader, val_loader = get_face_sr_dataloaders(
        data_dir=config.data_dir,
        scale_factor=config.scale_factor,
        crop_size=config.crop_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Train
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
