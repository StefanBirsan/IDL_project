"""
Configuration for Streamlit documentation app
"""
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class AppConfig:
    """Configuration for the streamlit app"""
    
    # Page configuration
    page_title: str = "Model Documentation Hub"
    page_icon: str = "🚀"
    layout: str = "wide"
    
    # Model parameters
    img_size: int = 64
    patch_size: int = 4
    embed_dim: int = 768
    encoder_depth: int = 12
    decoder_depth: int = 8
    num_heads: int = 12
    mask_ratio: float = 0.75
    
    # Paths (relative to project root)
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "dataset/data"
    output_dir: str = "streamlit/outputs"
    
    # Inference settings
    default_device: str = "cpu"  # Change to 'cuda' if GPU available
    inference_batch_size: int = 1
    
    # Visualization settings
    colormap: str = "gray"
    error_colormap: str = "RdYlGn_r"
    figure_dpi: int = 100
    
    # Available models
    available_models: Dict[str, str] = None
    
    def __post_init__(self):
        if self.available_models is None:
            self.available_models = {
                "Physics-Informed MAE": "physics_informed_mae_best.pt",
                "Baseline MAE": "baseline_mae_best.pt",
                "Latest Training": "latest_checkpoint.pt",
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'page_title': self.page_title,
            'page_icon': self.page_icon,
            'layout': self.layout,
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'encoder_depth': self.encoder_depth,
            'decoder_depth': self.decoder_depth,
            'num_heads': self.num_heads,
            'mask_ratio': self.mask_ratio,
            'checkpoint_dir': self.checkpoint_dir,
            'data_dir': self.data_dir,
            'output_dir': self.output_dir,
            'default_device': self.default_device,
        }
