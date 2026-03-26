"""
Training script for Physics-Informed Masked Vision Transformer
Implements the full training loop with masking and flux-guided reconstruction
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import tqdm
from typing import Dict, Tuple, Optional
import argparse

from training.train_utils import create_physics_informed_mae
from utils.numpy_dataset import get_dataloader


class TrainingConfig:
    """Training configuration"""
    def __init__(self,
                 # Model params
                 img_size: int = 64,
                 patch_size: int = 4,
                 embed_dim: int = 768,
                 encoder_depth: int = 12,
                 decoder_depth: int = 8,
                 num_heads: int = 12,
                 mask_ratio: float = 0.75,
                 
                 # Training params
                 batch_size: int = 32,
                 num_epochs: int = 100,
                 learning_rate: float = 1.5e-4,
                 weight_decay: float = 0.05,
                 lambda_flux: float = 0.01,
                 
                 # Data params
                 data_dir: str = 'dataset/data',
                 num_workers: int = 4,
                 
                 # Hardware params
                 device: str = 'cuda',
                 seed: int = 42,
                 
                 # Logging
                 save_dir: str = 'checkpoints',
                 save_interval: int = 10,
                 log_interval: int = 10):
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.num_heads = num_heads
        self.mask_ratio = mask_ratio
        
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lambda_flux = lambda_flux
        
        self.data_dir = data_dir
        self.num_workers = num_workers
        
        self.device = device
        self.seed = seed
        
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.log_interval = log_interval
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig':
        """Create from dictionary"""
        return cls(**config_dict)


class Trainer:
    """Training manager for Physics-Informed MAE"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set seed
        torch.manual_seed(config.seed)
        
        # Create model
        self.model = create_physics_informed_mae(
            img_size=config.img_size,
            embed_dim=config.embed_dim,
            encoder_depth=config.encoder_depth,
            decoder_depth=config.decoder_depth,
            num_heads=config.num_heads,
            mask_ratio=config.mask_ratio,
        ).to(self.device)
        
        # Optimizer (AdamW with specified hyperparameters)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler (cosine annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.train_losses = []
        self.eval_losses = []
        self.global_step = 0
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_loss_recon = 0.0
        total_loss_flux = 0.0
        num_batches = 0
        
        progress_bar = tqdm.tqdm(train_loader, desc="Training")
        
        for batch_idx, (images, metadata) in enumerate(progress_bar):
            images = images.to(self.device)  # (B, C, H, W)
            
            # Forward pass
            reconstructed, aux_outputs = self.model(images)
            
            # Compute loss
            loss, loss_dict = self.model.get_loss(
                inputs=images,
                reconstructed=reconstructed,
                aux_outputs=aux_outputs,
                flux_map_target=None,
                lambda_flux=self.config.lambda_flux
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track loss
            total_loss += loss_dict['loss_total']
            total_loss_recon += loss_dict['loss_recon']
            total_loss_flux += loss_dict['loss_flux']
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_loss = total_loss / num_batches
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'loss_recon': f'{total_loss_recon / num_batches:.4f}',
                    'loss_flux': f'{total_loss_flux / num_batches:.4f}',
                })
        
        return {
            'loss_total': total_loss / num_batches,
            'loss_recon': total_loss_recon / num_batches,
            'loss_flux': total_loss_flux / num_batches,
        }
    
    @torch.no_grad()
    def eval_epoch(self, eval_loader: DataLoader) -> Dict[str, float]:
        """Evaluate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_loss_recon = 0.0
        total_loss_flux = 0.0
        num_batches = 0
        
        progress_bar = tqdm.tqdm(eval_loader, desc="Evaluating")
        
        for images, metadata in progress_bar:
            images = images.to(self.device)
            
            # Forward pass
            reconstructed, aux_outputs = self.model(images)
            
            # Compute loss
            loss, loss_dict = self.model.get_loss(
                inputs=images,
                reconstructed=reconstructed,
                aux_outputs=aux_outputs,
                flux_map_target=None,
                lambda_flux=self.config.lambda_flux
            )
            
            total_loss += loss_dict['loss_total']
            total_loss_recon += loss_dict['loss_recon']
            total_loss_flux += loss_dict['loss_flux']
            num_batches += 1
        
        return {
            'loss_total': total_loss / num_batches,
            'loss_recon': total_loss_recon / num_batches,
            'loss_flux': total_loss_flux / num_batches,
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'global_step': self.global_step,
        }
        
        # Regular checkpoint
        ckpt_path = Path(self.config.save_dir) / f'checkpoint_epoch_{epoch:04d}.pt'
        torch.save(checkpoint, ckpt_path)
        
        # Best checkpoint
        if is_best:
            best_path = Path(self.config.save_dir) / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']
    
    def export_to_onnx(self, output_dir: str = 'models'):
        """Export trained model to ONNX format"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create dummy input
        dummy_input = torch.randn(
            1,
            self.config.img_size,
            self.config.img_size,
            device=self.device
        ).unsqueeze(0)  # (1, 1, H, W)
        
        # Set model to eval mode
        self.model.eval()
        
        # Export to ONNX
        onnx_path = output_path / 'physics_informed_mae.onnx'
        
        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                str(onnx_path),
                input_names=['image'],
                output_names=['reconstructed'],
                dynamic_axes={
                    'image': {0: 'batch_size'},
                    'reconstructed': {0: 'batch_size'}
                },
                opset_version=17,
                verbose=False
            )
            print(f"\n{'='*60}")
            print(f"ONNX model exported successfully!")
            print(f"Path: {onnx_path}")
            print(f"{'='*60}\n")
            return True
        except Exception as e:
            print(f"\nError exporting to ONNX: {e}")
            print(f"Make sure opset version 17 or higher is supported.\n")
            return False
    
    def train(self,
              train_loader: DataLoader,
              eval_loader: Optional[DataLoader] = None):
        """Full training loop"""
        print(f"\n{'='*60}")
        print("Starting Training: Physics-Informed Masked Vision Transformer")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Mask ratio: {self.config.mask_ratio}")
        print(f"Lambda (flux): {self.config.lambda_flux}")
        print(f"{'='*60}\n")
        
        best_eval_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch [{epoch+1}/{self.config.num_epochs}]")
            
            # Train
            train_loss_dict = self.train_epoch(train_loader)
            self.train_losses.append(train_loss_dict)
            
            print(f"Train Loss: {train_loss_dict['loss_total']:.4f}")
            print(f"  - Recon: {train_loss_dict['loss_recon']:.4f}")
            print(f"  - Flux:  {train_loss_dict['loss_flux']:.4f}")
            
            # Evaluate
            if eval_loader is not None:
                eval_loss_dict = self.eval_epoch(eval_loader)
                self.eval_losses.append(eval_loss_dict)
                
                print(f"Eval Loss:  {eval_loss_dict['loss_total']:.4f}")
                print(f"  - Recon: {eval_loss_dict['loss_recon']:.4f}")
                print(f"  - Flux:  {eval_loss_dict['loss_flux']:.4f}")
                
                is_best = eval_loss_dict['loss_total'] < best_eval_loss
                if is_best:
                    best_eval_loss = eval_loss_dict['loss_total']
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch + 1, is_best=(eval_loader is not None and is_best))
            
            # Update scheduler
            self.scheduler.step()
        
        # Save final model
        self.save_checkpoint(self.config.num_epochs, is_best=False)
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
        }
        history_path = Path(self.config.save_dir) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Export to ONNX
        print(f"\nExporting trained model to ONNX format...")
        self.export_to_onnx(output_dir='models')
        
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Checkpoints saved to: {self.config.save_dir}")
        print(f"ONNX model saved to: models/")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train Physics-Informed Masked Vision Transformer"
    )
    parser.add_argument('--data-dir', type=str, default='dataset/data',
                        help='Path to dataset directory')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1.5e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--lambda-flux', type=float, default=0.01,
                        help='Flux loss weight')
    parser.add_argument('--mask-ratio', type=float, default=0.75,
                        help='Masking ratio (0.75-0.90)')
    parser.add_argument('--img-size', type=int, default=64,
                        help='Image size')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lambda_flux=args.lambda_flux,
        mask_ratio=args.mask_ratio,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        save_dir=args.save_dir,
    )
    
    # Create trainer
    trainer = Trainer(config)
    
    # Create data loaders
    train_loader = get_dataloader(
        args.data_dir,
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )
    
    eval_loader = get_dataloader(
        args.data_dir,
        split='eval',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )
    
    # Train model
    trainer.train(train_loader, eval_loader)


if __name__ == '__main__':
    main()
