"""
Trainer for SRCNN Face Super-Resolution Model
Handles training loop, validation, checkpointing, and model export
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from training.train_utils.srcnn import SRCNN
from training.core.config_srcnn import SRCNNTrainingConfig
from training.managers import CheckpointManager, ModelExporter


class SRCNNTrainer:
    """
    Trainer for SRCNN model with layer-specific learning rates,
    checkpointing, and ONNX export support.
    """
    
    def __init__(self, config: SRCNNTrainingConfig):
        """
        Initialize SRCNN trainer.
        
        Args:
            config: SRCNNTrainingConfig with all training parameters
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seed for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Create model
        self.model = SRCNN(
            in_channels=config.input_channels,
            intermediate_channels=config.intermediate_channels,
            scale_factor=config.scale_factor
        ).to(self.device)
        
        # Setup optimizers with layer-specific learning rates
        self.optimizer = self._setup_optimizer()
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Managers
        self.checkpoint_manager = CheckpointManager(config.save_dir)
        self.model_exporter = ModelExporter()
        
        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        # Initialize best loss with maximum float value
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_psnr': [],
            'val_ssim': [],
            'epoch': []
        }
        
        self._print_info()
    
    def _setup_optimizer(self) -> optim.SGD:
        """
        Setup SGD optimizer with layer-specific learning rates.
        Layer 1-2: lr=1e-4, Layer 3: lr=1e-5
        
        Returns:
            SGD optimizer with parameter groups
        """
        param_groups = [
            {
                'params': self.model.layer1.parameters(),
                'lr': self.config.lr_early_layers,
                'name': 'layer1'
            },
            {
                'params': self.model.layer2.parameters(),
                'lr': self.config.lr_early_layers,
                'name': 'layer2'
            },
            {
                'params': self.model.layer3.parameters(),
                'lr': self.config.lr_reconstruction_layer,
                'name': 'layer3'
            }
        ]
        
        optimizer = optim.SGD(
            param_groups,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        return optimizer
    
    def _print_info(self) -> None:
        """Print training configuration and model info"""
        print("\n" + "="*70)
        print("SRCNN TRAINER INITIALIZED")
        print("="*70)
        print(self.config.summary())
        print(self.model.summary())
        print(f"Device: {self.device}")
        print(f"Model Parameters: {self.model.num_parameters:,}")
        print("="*70 + "\n")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch+1}/{self.config.num_epochs}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            lr_image = batch['lr_image'].to(self.device)  # (B, 3, H, W) - bicubic upscaled
            hr_image = batch['hr_image'].to(self.device)  # (B, 3, H, W) - ground truth
            
            # Reset optimizer gradients
            self.optimizer.zero_grad()
            # Forward pass on input LR
            output = self.model(lr_image)
            
            # Compute loss (MSE between output and HR)
            loss = self.criterion(output, hr_image)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix(loss=f'{loss.item():.6f}')
            
            # Log to terminal periodically
            if (batch_idx + 1) % self.config.log_interval == 0 and self.config.verbose:
                avg_loss = total_loss / num_batches
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {avg_loss:.6f}")
        
        avg_epoch_loss = total_loss / max(num_batches, 1)
        return avg_epoch_loss
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics (loss, PSNR, SSIM)
        """
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = 0
        
        for batch in val_loader:
            lr_image = batch['lr_image'].to(self.device)
            hr_image = batch['hr_image'].to(self.device)
            
            output = self.model(lr_image)
            loss = self.criterion(output, hr_image)
            
            total_loss += loss.item()
            
            # Calculate PSNR
            mse = loss.item()
            if mse > 0:
                psnr = peak_signal_noise_ratio(hr_image.cpu().numpy(), output.cpu().numpy(), data_range=1.0)
                total_psnr += psnr
            
            # Calculate SSIM
            ssim = structural_similarity(hr_image.cpu().numpy(), output.cpu().numpy(), data_range=1.0)
            total_ssim += ssim
            
            num_batches += 1
        
        metrics = {
            'val_loss': total_loss / max(num_batches, 1),
            'val_psnr': total_psnr / max(num_batches, 1),
            'val_ssim': total_ssim / max(num_batches, 1)
        }
        
        return metrics

    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None) -> Dict[str, list]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            
        Returns:
            Training history dictionary
        """
        print(f"\nStarting training for {self.config.num_epochs} epochs...\n")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['epoch'].append(epoch)
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs} - Train Loss: {train_loss:.6f}")
            
            # Validate
            val_metrics = {}
            if val_loader is not None and (epoch + 1) % self.config.val_interval == 0:
                val_metrics = self.validate(val_loader)
                self.training_history['val_loss'].append(val_metrics['val_loss'])
                self.training_history['val_psnr'].append(val_metrics['val_psnr'])
                self.training_history['val_ssim'].append(val_metrics['val_ssim'])
                
                print(f"  Val Loss: {val_metrics['val_loss']:.6f} | "
                      f"Val PSNR: {val_metrics['val_psnr']:.2f} | "
                      f"Val SSIM: {val_metrics['val_ssim']:.4f}")
                
                # Check if loss improved, if so save checkpoint
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.checkpoint_manager.save(
                        epoch=self.current_epoch,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=None,
                        config=self.config,
                        global_step=self.global_step,
                        is_best=True
                    )
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.checkpoint_manager.save(
                    epoch=self.current_epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=None,
                    config=self.config.to_dict(),
                    global_step=self.global_step,
                )
        
        print(f"\nTraining completed!")
        self._save_training_history()
        
        # Export to ONNX
        if self.config.export_final_model:
            self.export_to_onnx()
        
        return self.training_history
    
    def _save_checkpoint(self, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best checkpoint
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
        }
        
        if is_best:
            checkpoint_path = Path(self.config.save_dir) / 'best_model.pth'
            print(f"  Saving best model to {checkpoint_path}")
        else:
            checkpoint_path = Path(self.config.save_dir) / f'checkpoint_epoch_{self.current_epoch:04d}.pth'
        
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
    
    def _save_training_history(self) -> None:
        """Save training history to JSON file"""
        history_path = Path(self.config.save_dir) / 'training_history.json'
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)
        
        # Convert numpy values to Python float for JSON serialization
        history_for_json = {
            k: [float(v) for v in v_list]
            for k, v_list in self.training_history.items()
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_for_json, f, indent=2)
        
        print(f"Training history saved to {history_path}")
    
    def export_to_onnx(self, output_dir: Optional[str] = None) -> bool:
        """
        Export trained model to ONNX format.
        
        Args:
            output_dir: Output directory (uses config.save_dir if None)
            
        Returns:
            True if export successful
        """
        if output_dir is None:
            output_dir = self.config.save_dir
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        onnx_path = Path(output_dir) / 'srcnn_face.onnx'
        
        # Set model to eval mode
        self.model.eval()
        
        # Create dummy input with dynamic shape
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
            
            print("\n" + "="*70)
            print("ONNX Model Exported Successfully!")
            print("="*70)
            print(f"Path:                    {onnx_path}")
            print(f"Input:                   low_res_image (batch, 3, H, W)")
            print(f"Output:                  super_resolved_image (batch, 3, H, W)")
            print(f"Opset Version:           13")
            print(f"Dynamic Axes:            Batch, Height, Width")
            print(f"Scale Factor:            {self.config.scale_factor}x")
            print("="*70 + "\n")
            
            return True
        
        except Exception as e:
            print(f"\nError exporting to ONNX: {str(e)}\n")
            return False
    
    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, device: str = 'cuda') -> Tuple['SRCNNTrainer', dict]:
        """
        Load trainer from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model to
            
        Returns:
            Tuple of (trainer, checkpoint_dict)
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config_dict = checkpoint['config']
        config = SRCNNTrainingConfig(**config_dict)
        
        trainer = cls(config)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.current_epoch = checkpoint['epoch'] + 1
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        trainer.global_step = checkpoint.get('global_step', 0)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {trainer.current_epoch}")
        
        return trainer, checkpoint
