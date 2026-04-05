"""
Super-Resolution Training Configuration (SRTrainingConfig)
Extends the base training configuration for true SR upscaling (LR -> HR)
Default behavior is now SR mode instead of legacy reconstruction
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass
class SRTrainingConfig:
    """
    Super-Resolution Training Configuration
    Optimized for true LR->HR upscaling with advanced reconstruction losses
    """
    # ============ SUPER-RESOLUTION PARAMETERS ============
    
    # Upscaling factor (2x, 4x, etc.)
    scale_factor: int = 2
    
    # ============ MODEL ARCHITECTURE ============
    img_size: int = 64
    patch_size: int = 4
    embed_dim: int = 768
    encoder_depth: int = 12
    decoder_depth: int = 8
    num_heads: int = 12
    mask_ratio: float = 0.75
    
    # ============ RECONSTRUCTION LOSS WEIGHTS ============
    # Loss composition: L_total = L_recon + L_l1 + L_ssim + L_fft + L_flux
    
    # Reconstruction loss (MSE on masked patches)
    lambda_recon: float = 1.0
    
    # L1 (Charbonnier) loss for sharper edges
    lambda_l1: float = 0.5
    
    # SSIM loss for perceptual quality
    lambda_ssim: float = 0.3
    
    # Frequency domain (FFT magnitude) loss for global structure
    lambda_fft: float = 0.1
    
    # Flux consistency/sparsity loss
    lambda_flux: float = 0.01
    
    # ============ MAE MASK PARAMETERS ============
    
    # Weight applied to visible (unmasked) patches in reconstruction loss
    # Default 0.0 = masked-only (true MAE), >0 = include visible patches
    masked_visible_weight: float = 0.0
    
    # ============ FLUX LOSS MODE ============
    
    # Flux loss mode: "sparsity" (default) or "target"
    # - "sparsity": regularize flux map to be sparse (sum of flux_weights^2)
    # - "target": compute consistency loss against provided flux_map_target
    flux_loss_mode: str = "sparsity"
    
    # ============ MULTI-SCALE SUPERVISION ============
    
    # Enable multi-scale loss supervision (1x, 2x, 4x, etc.)
    enable_multiscale: bool = False
    
    # Weights for each scale (will be normalized)
    # Default: equal weight for all scales that are produced
    multiscale_weights: Optional[List[float]] = None
    
    # ============ TRAINING PARAMETERS ============
    
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1.5e-4
    weight_decay: float = 0.05
    
    # Optimizer betas (AdamW)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    
    # ============ DATA PARAMETERS ============
    
    data_dir: str = 'dataset/data'
    num_workers: int = 4
    
    # ============ HARDWARE PARAMETERS ============
    
    device: str = 'cuda'
    seed: int = 42
    
    # ============ LOGGING & CHECKPOINTING ============
    
    save_dir: str = 'checkpoints'
    save_interval: int = 10
    log_interval: int = 10
    save_every_batches: int = 0
    
    # ============ DEFAULTS ============
    # These mirror legacy config but should not be edited
    # Only provided for compatibility
    in_channels: int = field(default=1, init=False)
    mlp_ratio: float = field(default=4.0, init=False)
    attn_drop: float = field(default=0.0, init=False)
    drop_path: float = field(default=0.1, init=False)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.scale_factor not in [1, 2, 4, 8]:
            raise ValueError(f"scale_factor must be in [1, 2, 4, 8], got {self.scale_factor}")
        
        if self.flux_loss_mode not in ["sparsity", "target"]:
            raise ValueError(f"flux_loss_mode must be 'sparsity' or 'target', got {self.flux_loss_mode}")
        
        if self.scale_factor == 1 and self.enable_multiscale:
            raise ValueError("enable_multiscale requires scale_factor > 1")
        
        # Initialize default multiscale weights if not provided
        if self.enable_multiscale and self.multiscale_weights is None:
            # Default: equal weight for output scale
            self.multiscale_weights = [1.0]
    
    @property
    def hr_img_size(self) -> int:
        """Compute HR image size based on scale factor"""
        return self.img_size * self.scale_factor
    
    @property
    def num_upsamples(self) -> int:
        """Compute number of upsampling stages needed"""
        import math
        if self.scale_factor == 1:
            return 0
        return int(math.log2(self.scale_factor))
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'SRTrainingConfig':
        """Create from dictionary"""
        return cls(**config_dict)
