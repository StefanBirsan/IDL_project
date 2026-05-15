from dataclasses import dataclass
import torch
import os
from pathlib import Path

@dataclass
class FSRCNNConfig:
    """Configuration for FSRCNN (Fast Super-Resolution CNN)"""
    
    # Model Parameters
    name: str = "FSRCNN"
    description: str = "Fast Super-Resolution CNN"
    
    # Architecture
    scale_factor: int = 4
    num_channels = 3

    # Inference
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    project_root = Path(os.getcwd())
    onnx_file = r"fsrcnn_best.onnx"
    onnx_file_path = project_root / "models" / onnx_file


# Create instance
MODEL_CONFIG = FSRCNNConfig()
