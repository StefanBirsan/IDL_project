"""
Configuration for SRCNN Model
"""
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SRCNNConfig:
    """Configuration for SRCNN (Super-Resolution Convolutional Neural Network)"""
    
    # Model Parameters
    name: str = "SRCNN"
    description: str = "Super-Resolution Convolutional Neural Network for Face Image Enhancement"
    
    # Architecture
    scale_factor: int = 2
    intermediate_channels: int = 32
    crop_size: int = 33
    
    # Training
    batch_size: int = 64
    num_epochs: int = 100
    lr_early_layers: float = 1e-4
    lr_reconstruction_layer: float = 1e-5
    momentum: float = 0.9
    
    # Validation
    val_interval: int = 5
    
    # Inference
    device: str = "cpu"
    
    # Paths
    checkpoint_name: str = "srcnn_best.pt"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'scale_factor': self.scale_factor,
            'intermediate_channels': self.intermediate_channels,
            'crop_size': self.crop_size,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'lr_early_layers': self.lr_early_layers,
            'lr_reconstruction_layer': self.lr_reconstruction_layer,
            'momentum': self.momentum,
            'val_interval': self.val_interval,
        }


# Create instance
MODEL_CONFIG = SRCNNConfig()
