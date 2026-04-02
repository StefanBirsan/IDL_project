"""
Configuration for Physics-Informed MAE Model
"""
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class PhysicsInformedMAEConfig:
    """Configuration for Physics-Informed MAE"""
    
    # Model Parameters
    name: str = "Physics-Informed MAE"
    description: str = "Masked Autoencoder with Physics-Informed Preprocessing for Image Reconstruction"
    
    # Architecture
    img_size: int = 64
    patch_size: int = 4
    embed_dim: int = 768
    encoder_depth: int = 12
    decoder_depth: int = 8
    num_heads: int = 12
    mask_ratio: float = 0.75
    
    # Training
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1.5e-4
    weight_decay: float = 0.05
    lambda_flux: float = 0.01
    
    # Inference
    device: str = "cpu"
    
    # Paths
    checkpoint_name: str = "physics_informed_mae_best.pt"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'encoder_depth': self.encoder_depth,
            'decoder_depth': self.decoder_depth,
            'num_heads': self.num_heads,
            'mask_ratio': self.mask_ratio,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'lambda_flux': self.lambda_flux,
        }


# Create instance
MODEL_CONFIG = PhysicsInformedMAEConfig()
