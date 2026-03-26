"""
Training configuration
"""
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model params
    img_size: int = 64
    patch_size: int = 4
    embed_dim: int = 768
    encoder_depth: int = 12
    decoder_depth: int = 8
    num_heads: int = 12
    mask_ratio: float = 0.75
    
    # Training params
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1.5e-4
    weight_decay: float = 0.05
    lambda_flux: float = 0.01
    
    # Data params
    data_dir: str = 'dataset/data'
    num_workers: int = 4
    
    # Hardware params
    device: str = 'cuda'
    seed: int = 42
    
    # Logging
    save_dir: str = 'checkpoints'
    save_interval: int = 10
    log_interval: int = 10
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig':
        """Create from dictionary"""
        return cls(**config_dict)
