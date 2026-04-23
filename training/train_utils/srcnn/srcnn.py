"""
SRCNN (Super-Resolution Convolutional Neural Network) for Face Super-Resolution
Based on "Image Super-Resolution Using Very Deep Convolutional Networks for Photorealistic Results" by Dong et al.

Architecture: 9-5-5 configuration optimized for face images
- Layer 1 (Patch Extraction): Conv2d(3, 64, 9x9) + ReLU
- Layer 2 (Non-linear Mapping): Conv2d(64, 32, 5x5) + ReLU  
- Layer 3 (Reconstruction): Conv2d(32, 3, 5x5) [Linear activation]
"""
import torch
import torch.nn as nn

class SRCNN(nn.Module):
    """
    SRCNN model for single-image super-resolution on face images.
    
    This model takes a low-resolution image that has been upscaled to target size
    using bicubic interpolation and refines it to produce sharper output.
    
    Architecture:
        Input (3, H, W) -> Patch Extraction -> Non-linear Mapping -> Reconstruction -> Output (3, H, W)
    """
    
    def __init__(self, 
                 in_channels: int = 3,
                 intermediate_channels: int = 32,
                 scale_factor: int = 2):
        """
        Initialize SRCNN model.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            intermediate_channels: Number of channels in second layer (optimization for structural info)
            scale_factor: Upscaling factor (2, 4, 8, etc) - used for documentation
        """
        super(SRCNN, self).__init__()
        
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        
        # Layer 1: Patch Extraction and Representation
        # Input: (batch, 3, H, W)
        # Output: (batch, 64, H-8, W-8)
        # 9x9 kernel extracts 9x9 patches and maps them to 64-dim representations
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                     out_channels=64, 
                     kernel_size=9, 
                     padding=4,  # padding to maintain spatial dimensions
                     bias=True),
            nn.ReLU(inplace=True)
        )
        
        # Layer 2: Non-linear Mapping
        # Input: (batch, 64, H, W)
        # Output: (batch, 32, H-4, W-4)
        # Maps 64-dim representations to 32-dim representation space
        # 32-dim is more compact for structural information
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, 
                     out_channels=intermediate_channels, 
                     kernel_size=5, 
                     padding=2,  # padding to maintain spatial dimensions
                     bias=True),
            nn.ReLU(inplace=True)
        )
        
        # Layer 3: Reconstruction
        # Input: (batch, 32, H, W)
        # Output: (batch, 3, H, W)
        # Reconstructs the final RGB image without activation (linear)
        # This is crucial - no activation allows output to vary beyond [0,1] before clipping
        self.layer3 = nn.Conv2d(in_channels=intermediate_channels, 
                               out_channels=in_channels, 
                               kernel_size=5, 
                               padding=2,
                               bias=True)
        
        # Initialize weights from Gaussian distribution with mean=0, std=0.001
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """
        Initialize network weights with Gaussian distribution N(0, 0.001).
        This initialization is crucial for SRCNN convergence.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SRCNN.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
               Should be bicubic-upscaled low-resolution image
               
        Returns:
            Output tensor of shape (batch_size, 3, height, width)
            Super-resolved image
        """
        # Patch Extraction
        out = self.layer1(x)  # (batch, 64, H, W)
        
        # Non-linear Mapping
        out = self.layer2(out)  # (batch, 32, H, W)
        
        # Reconstruction
        out = self.layer3(out)  # (batch, 3, H, W)
        
        # The network stores only the residual connection, i.e. the changes made to the input
        out = x + out
        
        return out
    
    def get_layer_parameters(self) -> dict:
        """
        Return grouped parameters by layer for layer-specific learning rates.
        
        Returns:
            Dictionary with 'early_layers', 'middle_layer', 'reconstruction_layer' parameter groups
        """
        return {
            'early_layers': list(self.layer1.parameters()),
            'middle_layer': list(self.layer2.parameters()),
            'reconstruction_layer': list(self.layer3.parameters()),
        }
    
    @property
    def num_parameters(self) -> int:
        """Return total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self) -> str:
        """Return model architecture summary"""
        summary = f"""
{'='*70}
SRCNN Model Summary (9-5-5 Configuration for Face Super-Resolution)
{'='*70}
Scale Factor: {self.scale_factor}x
Total Parameters: {self.num_parameters:,}

Architecture:
  Layer 1 (Patch Extraction):     Conv(3 -> 64, 9x9) + ReLU
  Layer 2 (Non-linear Mapping):   Conv(64 -> 32, 5x5) + ReLU
  Layer 3 (Reconstruction):       Conv(32 -> 3, 5x5) [Linear + Residual]
  
Input:  (batch_size, 3, H, W) - bicubic upscaled low-res face
Output: (batch_size, 3, H, W) - super-resolved face
{'='*70}
        """
        return summary
