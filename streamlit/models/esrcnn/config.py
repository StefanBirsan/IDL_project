"""
Configuration for Enhanced SRCNN (ESRCNN) Model
"""
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ESRCNNConfig:
    """Configuration for ESRCNN (Enhanced Super-Resolution CNN with Residual Blocks)"""
    
    # Model Parameters
    name: str = "ESRCNN"
    description: str = "Enhanced SRCNN with Deep Residual Blocks and Perceptual Loss for Face Super-Resolution"
    
    # Architecture
    scale_factor: int = 2
    num_features: int = 64
    num_residual_blocks: int = 10
    use_global_skip: bool = True
    crop_size: int = 48
    
    # Training
    batch_size: int = 16
    num_epochs: int = 150
    learning_rate: float = 1e-4
    optimizer: str = 'adam'
    weight_decay: float = 1e-4
    lr_scheduler: str = 'step'
    
    # Loss Configuration
    loss_pixel_weight: float = 1.0
    loss_perceptual_weight: float = 0.1
    loss_ssim_weight: float = 0.0
    use_perceptual_loss: bool = True
    pixel_loss_type: str = 'l1'
    
    # Validation
    val_interval: int = 5
    
    # Advanced
    mixed_precision: bool = True
    
    # Inference
    device: str = "cpu"
    
    # Paths
    checkpoint_name: str = "esrcnn_best.pt"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'scale_factor': self.scale_factor,
            'num_features': self.num_features,
            'num_residual_blocks': self.num_residual_blocks,
            'use_global_skip': self.use_global_skip,
            'crop_size': self.crop_size,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer,
            'weight_decay': self.weight_decay,
            'loss_pixel_weight': self.loss_pixel_weight,
            'loss_perceptual_weight': self.loss_perceptual_weight,
            'use_perceptual_loss': self.use_perceptual_loss,
            'pixel_loss_type': self.pixel_loss_type,
            'val_interval': self.val_interval,
            'mixed_precision': self.mixed_precision,
        }


# Create instance
MODEL_CONFIG = ESRCNNConfig()
