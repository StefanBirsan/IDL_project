"""
Training configuration for Enhanced SRCNN (ESRCNN)
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ESRCNNConfig:
    """
    Complete training configuration for Enhanced SRCNN.
    Optimized for high-quality face super-resolution.
    """
    
    # ============ MODEL ARCHITECTURE ============
    model_name: str = 'esrcnn_face'
    scale_factor: int = 2                    # Super-resolution upscaling factor (2, 4, 8)
    num_features: int = 64                   # Number of feature channels
    num_residual_blocks: int = 10            # Number of residual blocks (8-16 recommended)
    use_global_skip: bool = True             # Global skip connection from input to output
    
    # ============ INPUT/PREPROCESSING ============
    input_channels: int = 3                  # RGB color channels
    crop_size: int = 48                      # Training patch size (larger than SRCNN for more context)
    
    # ============ TRAINING HYPERPARAMETERS ============
    batch_size: int = 16                     # Smaller batch due to larger model
    num_epochs: int = 150                    # More epochs for deeper network
    
    # Learning rate - Using Adam instead of SGD for better convergence
    learning_rate: float = 1e-4              # Initial learning rate
    lr_scheduler: str = 'step'               # 'step', 'cosine', or 'plateau'
    lr_decay_epochs: List[int] = field(default_factory=lambda: [50, 100, 130])  # Epochs to decay LR
    lr_decay_factor: float = 0.5             # LR decay factor
    
    # Optimizer configuration
    optimizer: str = 'adam'                  # 'adam' or 'adamw'
    weight_decay: float = 1e-4               # Weight decay (L2 regularization)
    betas: tuple = (0.9, 0.999)              # Adam betas
    
    # ============ LOSS CONFIGURATION ============
    # Multiple losses for better perceptual quality
    loss_pixel_weight: float = 1.0           # Weight for pixel-wise loss (MSE/L1)
    loss_perceptual_weight: float = 0.1      # Weight for perceptual (VGG) loss
    loss_ssim_weight: float = 0.0            # Weight for SSIM loss (optional)
    use_perceptual_loss: bool = True         # Enable perceptual loss
    pixel_loss_type: str = 'l1'              # 'l1' or 'mse'
    
    # Perceptual loss settings
    perceptual_layers: List[int] = field(default_factory=lambda: [3, 8, 17, 26])
    
    # ============ DATA PARAMETERS ============
    data_dir: str = 'dataset/ffhq'           # Dataset directory
    train_subset_fraction: float = 1.0       # Use fraction of training set
    num_workers: int = 4                     # Data loading workers
    
    # ============ VALIDATION & EVALUATION ============
    val_interval: int = 5                    # Validate every N epochs
    eval_metrics: List[str] = field(default_factory=lambda: ['psnr', 'ssim', 'lpips'])
    
    # ============ CHECKPOINTING & EXPORT ============
    save_dir: str = 'checkpoints/esrcnn'     # Directory to save checkpoints
    save_interval: int = 10                  # Save checkpoint every N epochs
    keep_best_only: bool = False             # Keep only the best checkpoint
    export_final_model: bool = True          # Export model to ONNX after training
    
    # ============ LOGGING ============
    log_interval: int = 10                   # Log metrics every N batches
    log_dir: str = 'logs/esrcnn'             # Directory for tensorboard/logs
    verbose: bool = True                     # Print training progress
    use_tensorboard: bool = False            # Use tensorboard logging
    
    # ============ HARDWARE ============
    device: str = 'cuda'                     # Device: 'cuda' or 'cpu'
    seed: int = 42                           # Random seed for reproducibility
    mixed_precision: bool = True             # Use automatic mixed precision training (faster!)
    
    # ============ AUGMENTATION ============
    augmentation_enabled: bool = True        # Enable data augmentation
    augmentation_types: List[str] = field(default_factory=lambda: ['rotate', 'flip'])
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ESRCNNConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    def summary(self) -> str:
        """Pretty print configuration summary"""
        return f"""
{'='*70}
Enhanced SRCNN (ESRCNN) Training Configuration
{'='*70}

MODEL:
  - Name:                    {self.model_name}
  - Scale Factor:            {self.scale_factor}x
  - Num Features:            {self.num_features}
  - Residual Blocks:         {self.num_residual_blocks}
  - Global Skip:             {self.use_global_skip}

TRAINING:
  - Batch Size:              {self.batch_size}
  - Epochs:                  {self.num_epochs}
  - Optimizer:               {self.optimizer.upper()}
  - Learning Rate:           {self.learning_rate}
  - LR Scheduler:            {self.lr_scheduler}
  - Weight Decay:            {self.weight_decay}
  - Mixed Precision:         {self.mixed_precision}

LOSSES:
  - Pixel Loss:              {self.pixel_loss_type.upper()} (weight={self.loss_pixel_weight})
  - Perceptual Loss:         {'Enabled' if self.use_perceptual_loss else 'Disabled'} (weight={self.loss_perceptual_weight})
  - SSIM Loss:               weight={self.loss_ssim_weight}

DATA:
  - Dataset Directory:       {self.data_dir}
  - Crop Size:               {self.crop_size}x{self.crop_size}
  - Workers:                 {self.num_workers}
  - Augmentation:            {self.augmentation_enabled}

CHECKPOINTING:
  - Save Directory:          {self.save_dir}
  - Save Interval:           Every {self.save_interval} epochs
  - Validation Interval:     Every {self.val_interval} epochs
  - Export ONNX:             {self.export_final_model}

HARDWARE:
  - Device:                  {self.device}
  - Mixed Precision:         {self.mixed_precision}
  - Random Seed:             {self.seed}
{'='*70}
"""
    
    def validate(self):
        """Validate configuration parameters"""
        assert self.scale_factor in [2, 4, 8], "Scale factor must be 2, 4, or 8"
        assert self.num_residual_blocks >= 1, "Need at least 1 residual block"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.pixel_loss_type in ['l1', 'mse'], "Pixel loss must be 'l1' or 'mse'"
        assert self.optimizer in ['adam', 'adamw'], "Optimizer must be 'adam' or 'adamw'"
