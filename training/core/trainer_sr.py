"""
Super-Resolution Trainer for Physics-Informed Masked Vision Transformer
Optimized for true LR -> HR upscaling with advanced loss composition
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from typing import Optional

from training.train_utils import create_physics_informed_mae
from training.core.config_sr import SRTrainingConfig
from training.managers import CheckpointManager, ModelExporter
from training.steps import MetricTracker
from training.steps.train_step_sr import train_one_epoch_sr
from training.steps.eval_step_sr import eval_one_epoch_sr


class SRTrainer:
    """
    Training manager for Physics-Informed MAE with Super-Resolution
    - Handles LR -> HR upscaling (true super-resolution)
    - Advanced loss composition (MSE + L1 + SSIM + FFT + Flux)
    - Mask-aware MAE reconstruction loss
    """
    
    def __init__(self, config: SRTrainingConfig):
        """
        Initialize SR trainer
        
        Args:
            config: SRTrainingConfig instance (SR-specific parameters)
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Set seed for reproducibility
        torch.manual_seed(config.seed)
        
        # Create model with SR parameters
        # The model will be initialized in SR mode
        self.model = create_physics_informed_mae(
            img_size=config.img_size,
            embed_dim=config.embed_dim,
            encoder_depth=config.encoder_depth,
            decoder_depth=config.decoder_depth,
            num_heads=config.num_heads,
            mask_ratio=config.mask_ratio,
            scale_factor=config.scale_factor,  # NEW: Pass scale factor for SR upsampling
            sr_mode=True,                        # NEW: Enable SR mode in model
        ).to(self.device)
        
        # Optimizer (AdamW with configurable betas)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2)
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
        Full training loop for SR mode
        
        Args:
            train_loader: Training data loader (LR/HR pairs)
            eval_loader: Evaluation data loader (optional)
        """
        self._print_training_info()
        
        best_eval_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch [{epoch+1}/{self.config.num_epochs}]")

            def _save_mid_epoch(batch_index: int):
                """Save checkpoint at mid-epoch intervals"""
                self.checkpoint_manager.save(
                    epoch=epoch + 1,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    config=self.config.to_dict(),
                    global_step=self.global_step,
                    is_best=False
                )
            
            # SR Training step
            train_loss_dict = train_one_epoch_sr(
                model=self.model,
                train_loader=train_loader,
                optimizer=self.optimizer,
                device=self.device,
                config=self.config,
                log_interval=self.config.log_interval,
                save_every_batches=self.config.save_every_batches,
                on_batch_checkpoint_callback_fn=_save_mid_epoch if self.config.save_every_batches > 0 else None
            )
            self.metrics.add_train_loss(train_loss_dict)
            self.global_step += len(train_loader)
            
            # Print training metrics
            self._print_train_metrics(train_loss_dict)
            
            # SR Evaluation step
            if eval_loader is not None:
                eval_loss_dict = eval_one_epoch_sr(
                    model=self.model,
                    eval_loader=eval_loader,
                    device=self.device,
                    config=self.config
                )
                self.metrics.add_eval_loss(eval_loss_dict)
                
                # Print evaluation metrics
                self._print_eval_metrics(eval_loss_dict)
                
                is_best = eval_loss_dict['loss_total'] < best_eval_loss
                if is_best:
                    best_eval_loss = eval_loss_dict['loss_total']
            else:
                is_best = False
            
            # Add current learning rate to tracking
            current_lr = self.optimizer.param_groups[0]['lr']
            self.metrics.add_learning_rate(current_lr)
            
            # Save checkpoint at intervals
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
            
            # Update learning rate scheduler
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
        """Print SR training configuration information"""
        print(f"\n{'='*70}")
        print("Starting Super-Resolution Training (SR MODE)")
        print("Physics-Informed Masked Vision Transformer -> LR to HR Upscaling")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Scale Factor: {self.config.scale_factor}x (LR {self.config.img_size} -> HR {self.config.hr_img_size})")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Mask ratio: {self.config.mask_ratio}")
        print(f"--- Loss Weights ---")
        print(f"  Reconstruction (MSE): {self.config.lambda_recon}")
        print(f"  L1 (edges): {self.config.lambda_l1}")
        print(f"  SSIM (perceptual): {self.config.lambda_ssim}")
        print(f"  FFT (structure): {self.config.lambda_fft}")
        print(f"  Flux ({self.config.flux_loss_mode}): {self.config.lambda_flux}")
        if self.config.enable_multiscale:
            print(f"Multi-scale supervision: ENABLED with weights {self.config.multiscale_weights}")
        print(f"{'='*70}\n")
    
    def _print_train_metrics(self, loss_dict: dict):
        """Print training metrics nicely"""
        print(f"Train Loss: {loss_dict['loss_total']:.4f}")
        print(f"  - Recon: {loss_dict.get('loss_recon', 0):.4f}")
        if 'loss_l1' in loss_dict:
            print(f"  - L1: {loss_dict['loss_l1']:.4f}")
        if 'loss_ssim' in loss_dict:
            print(f"  - SSIM: {loss_dict['loss_ssim']:.4f}")
        if 'loss_fft' in loss_dict:
            print(f"  - FFT: {loss_dict['loss_fft']:.4f}")
        print(f"  - Flux: {loss_dict.get('loss_flux', 0):.4f}")
    
    def _print_eval_metrics(self, loss_dict: dict):
        """Print evaluation metrics nicely"""
        print(f"Eval Loss:  {loss_dict['loss_total']:.4f}")
        print(f"  - Recon: {loss_dict.get('loss_recon', 0):.4f}")
        if 'loss_l1' in loss_dict:
            print(f"  - L1: {loss_dict['loss_l1']:.4f}")
        if 'loss_ssim' in loss_dict:
            print(f"  - SSIM: {loss_dict['loss_ssim']:.4f}")
        if 'loss_fft' in loss_dict:
            print(f"  - FFT: {loss_dict['loss_fft']:.4f}")
        print(f"  - Flux: {loss_dict.get('loss_flux', 0):.4f}")
    
    def _save_metrics(self):
        """Save training metrics to file"""
        history_path = Path(self.config.save_dir) / 'training_history_sr.json'
        with open(history_path, 'w') as f:
            json.dump(self.metrics.to_dict(), f, indent=2)
        print(f"Saved SR training history to {history_path}")
    
    def _print_training_complete(self):
        """Print training completion message"""
        print(f"\n{'='*70}")
        print("Super-Resolution Training completed!")
        print(f"Checkpoints saved to: {self.config.save_dir}")
        print(f"Metrics saved to: {Path(self.config.save_dir) / 'training_history_sr.json'}")
        print(f"ONNX model saved to: models/physics_informed_mae.onnx")
        print(f"Output HR size: {self.config.hr_img_size}x{self.config.hr_img_size}")
        print(f"{'='*70}\n")
    
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
