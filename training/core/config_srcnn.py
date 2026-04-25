"""
Training configuration for SRCNN Face Super-Resolution
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class SRCNNTrainingConfig:
    """
    Complete training configuration for SRCNN.
    Implements the exact parameters from Dong et al. paper optimized for face super-resolution.
    """
    
    # ============ MODEL ARCHITECTURE ============
    model_name: str = 'srcnn_face'
    scale_factor: int = 2                    # Super-resolution upscaling factor (2x, 4x, etc)
    intermediate_channels: int = 32          # Channels in middle layer (optimized for structural info)
    
    # ============ INPUT/PREPROCESSING ============
    input_channels: int = 3                  # RGB color channels
    crop_size: int = 33                      # Sub-image crop size for training (prevents border effects)
    upscale_method: str = 'bicubic'          # Upscaling method for LR to target size
    
    # ============ TRAINING HYPERPARAMETERS ============
    batch_size: int = 64                     # Batch size for training
    num_epochs: int = 100                    # Total training epochs
    
    # Learning rates - Layer specific (crucial for SRCNN convergence)
    lr_early_layers: float = 1e-4            # Learning rate for layers 1-2 (patch extraction & non-linear mapping)
    lr_reconstruction_layer: float = 1e-5    # Learning rate for layer 3 (reconstruction) - 10x smaller
    
    # Optimizer configuration
    momentum: float = 0.9                    # SGD momentum
    weight_decay: float = 0.0                # Weight decay (L2 regularization)
    
    # Loss configuration
    loss_function: str = 'mse'              # Loss function: MSE optimizes for PSNR
    
    # ============ DATA PARAMETERS ============
    data_dir: str = 'dataset/ffhq'           # Dataset directory (FFHQ for high-quality faces)
    train_subset_fraction: float = 1.0       # Use fraction of training set (for quick testing use <1.0)
    num_workers: int = 4                     # Data loading workers
    
    # ============ VALIDATION & EVALUATION ============
    val_interval: int = 1                    # Validate every N epochs
    eval_metrics: List[str] = field(default_factory=lambda: ['psnr', 'ssim', 'mae'])
    
    # ============ CHECKPOINTING & EXPORT ============
    save_dir: str = 'checkpoints/srcnn'      # Directory to save checkpoints
    save_interval: int = 10                  # Save checkpoint every N epochs
    keep_best_only: bool = False             # Keep only the best checkpoint
    export_final_model: bool = True          # Export model to ONNX after training
    
    # ============ LOGGING ============
    log_interval: int = 10                   # Log metrics every N batches
    log_dir: str = 'logs/srcnn'              # Directory for tensorboard/logs
    verbose: bool = True                     # Print training progress
    
    # ============ HARDWARE ============
    device: str = 'cuda'                     # Device: 'cuda' or 'cpu'
    seed: int = 42                           # Random seed for reproducibility
    mixed_precision: bool = False            # Use automatic mixed precision training
    
    # ============ AUGMENTATION ============
    augmentation_enabled: bool = True        # Enable data augmentation
    augmentation_types: List[str] = field(default_factory=lambda: ['rotate', 'flip', 'brightness'])
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'SRCNNTrainingConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    def summary(self) -> str:
        """Pretty print configuration summary"""
        summary = f"""
          {'='*70}
          SRCNN Training Configuration
          {'='*70}

          MODEL:
            - Name:                    {self.model_name}
            - Scale Factor:            {self.scale_factor}x
            - Intermediate Channels:   {self.intermediate_channels}

          TRAINING:
            - Batch Size:              {self.batch_size}
            - Epochs:                  {self.num_epochs}
            - Optimizer:               SGD with momentum={self.momentum}
            - Learning Rates:          Layer 1-2: {self.lr_early_layers}, Layer 3: {self.lr_reconstruction_layer}
            - Loss:                    {self.loss_function.upper()}

          DATA:
            - Dataset Directory:       {self.data_dir}
            - Crop Size:               {self.crop_size}x{self.crop_size}
            - Upscale Method:          {self.upscale_method}
            - Data Workers:            {self.num_workers}

          CHECKPOINTING:
            - Save Directory:          {self.save_dir}
            - Save Interval:           Every {self.save_interval} epochs
            - Export ONNX:             {self.export_final_model}

          HARDWARE:
            - Device:                  {self.device}
            - Seed:                    {self.seed}
            - Mixed Precision:         {self.mixed_precision}

          {'='*70}
        """
        return summary
