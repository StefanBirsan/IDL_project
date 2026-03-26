"""
Main trainer class for Physics-Informed Masked Vision Transformer
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from typing import Optional

from training.train_utils import create_physics_informed_mae
from training.core.config import TrainingConfig
from training.managers import CheckpointManager, ModelExporter
from training.steps import train_one_epoch, eval_one_epoch, MetricTracker


class Trainer:
    """Training manager for Physics-Informed MAE"""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer
        
        Args:
            config: Training configuration
        """
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
        
        # Initialize managers
        self.checkpoint_manager = CheckpointManager(config.save_dir)
        self.model_exporter = ModelExporter()
        
        # Metric tracking
        self.metrics = MetricTracker()
        self.global_step = 0
    
    def train(self,
              train_loader: DataLoader,
              eval_loader: Optional[DataLoader] = None):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            eval_loader: Evaluation data loader (optional)
        """
        self._print_training_info()
        
        best_eval_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch [{epoch+1}/{self.config.num_epochs}]")
            
            # Train step
            train_loss_dict = train_one_epoch(
                model=self.model,
                train_loader=train_loader,
                optimizer=self.optimizer,
                device=self.device,
                lambda_flux=self.config.lambda_flux,
                log_interval=self.config.log_interval
            )
            self.metrics.add_train_loss(train_loss_dict)
            self.global_step += len(train_loader)
            
            print(f"Train Loss: {train_loss_dict['loss_total']:.4f}")
            print(f"  - Recon: {train_loss_dict['loss_recon']:.4f}")
            print(f"  - Flux:  {train_loss_dict['loss_flux']:.4f}")
            
            # Evaluation step
            if eval_loader is not None:
                eval_loss_dict = eval_one_epoch(
                    model=self.model,
                    eval_loader=eval_loader,
                    device=self.device,
                    lambda_flux=self.config.lambda_flux
                )
                self.metrics.add_eval_loss(eval_loss_dict)
                
                print(f"Eval Loss:  {eval_loss_dict['loss_total']:.4f}")
                print(f"  - Recon: {eval_loss_dict['loss_recon']:.4f}")
                print(f"  - Flux:  {eval_loss_dict['loss_flux']:.4f}")
                
                is_best = eval_loss_dict['loss_total'] < best_eval_loss
                if is_best:
                    best_eval_loss = eval_loss_dict['loss_total']
            else:
                is_best = False
            
            # Add current learning rate to tracking
            current_lr = self.optimizer.param_groups[0]['lr']
            self.metrics.add_learning_rate(current_lr)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.checkpoint_manager.save(
                    epoch=epoch + 1,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    config=self.config.to_dict(),
                    global_step=self.global_step,
                    is_best=is_best
                )
            
            # Update scheduler
            self.scheduler.step()
        
        # Save final model
        self.checkpoint_manager.save(
            epoch=self.config.num_epochs,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.config.to_dict(),
            global_step=self.global_step,
            is_best=False
        )
        
        # Save metrics
        self._save_metrics()
        
        # Export to ONNX
        print(f"\nExporting trained model to ONNX format...")
        self.model_exporter.export_to_onnx(
            model=self.model,
            output_dir='models',
            img_size=self.config.img_size,
            device=str(self.device)
        )
        
        self._print_training_complete()
    
    def _print_training_info(self):
        """Print training information"""
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
    
    def _save_metrics(self):
        """Save training metrics to file"""
        history_path = Path(self.config.save_dir) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.metrics.to_dict(), f, indent=2)
        print(f"Saved training history to {history_path}")
    
    def _print_training_complete(self):
        """Print training completion message"""
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Checkpoints saved to: {self.config.save_dir}")
        print(f"Metrics saved to: {Path(self.config.save_dir) / 'training_history.json'}")
        print(f"ONNX model saved to: models/physics_informed_mae.onnx")
        print(f"{'='*60}\n")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
        Returns:
            Epoch of loaded checkpoint
        """
        return self.checkpoint_manager.load(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=str(self.device)
        )
