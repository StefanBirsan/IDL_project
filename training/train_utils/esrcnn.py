"""
Enhanced SRCNN (ESRCNN) for Face Super-Resolution
A deeper, more powerful architecture for high-quality face upscaling

Key improvements over classic SRCNN:
- Deeper network: 10+ layers with residual blocks
- Perceptual features via feature extraction head
- Larger receptive field
- Batch normalization for training stability
- Skip connections for gradient flow
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ResidualBlock(nn.Module):
    """
    Residual Block with Batch Normalization
    Used as the building block for deep SR networks
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection"""
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual connection
        out = out + identity
        out = self.relu(out)
        
        return out


class UpsampleBlock(nn.Module):
    """
    Efficient Sub-Pixel Convolution Upsampling
    Also known as PixelShuffle - more efficient than deconv
    """
    def __init__(self, in_channels: int, scale_factor: int):
        super(UpsampleBlock, self).__init__()
        
        # PixelShuffle requires scale_factor^2 more channels
        self.conv = nn.Conv2d(
            in_channels, 
            in_channels * (scale_factor ** 2), 
            kernel_size=3, 
            padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x


class EnhancedSRCNN(nn.Module):
    """
    Enhanced SRCNN (ESRCNN) for Face Super-Resolution
    
    Architecture:
    - Initial feature extraction (64 channels)
    - Multiple residual blocks (8-16 blocks)
    - Global skip connection
    - Efficient upsampling via sub-pixel convolution
    - Final reconstruction layer
    
    This architecture is significantly more powerful than classic SRCNN
    while remaining relatively lightweight (~2-5M parameters).
    """
    
    def __init__(
        self, 
        in_channels: int = 3,
        num_features: int = 64,
        num_residual_blocks: int = 10,
        scale_factor: int = 2,
        use_global_skip: bool = True
    ):
        """
        Initialize Enhanced SRCNN
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            num_features: Number of feature channels (64 recommended)
            num_residual_blocks: Number of residual blocks (8-16 recommended)
            scale_factor: Upscaling factor (2, 4, 8)
            use_global_skip: Use global residual connection from input to output
        """
        super(EnhancedSRCNN, self).__init__()
        
        self.scale_factor = scale_factor
        self.use_global_skip = use_global_skip
        
        # ========== FEATURE EXTRACTION ==========
        # Initial convolution to extract low-level features
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # ========== RESIDUAL BLOCKS ==========
        # Stack of residual blocks for non-linear mapping
        residual_layers = []
        for _ in range(num_residual_blocks):
            residual_layers.append(ResidualBlock(num_features, kernel_size=3))
        self.residual_blocks = nn.Sequential(*residual_layers)
        
        # ========== BOTTLENECK ==========
        # Additional convolution after residual blocks
        self.bottleneck = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features)
        )
        
        # ========== UPSAMPLING ==========
        # Efficient upsampling using sub-pixel convolution
        upsampling_layers = []
        if scale_factor in [2, 4, 8]:
            # For scale factors that are powers of 2, stack multiple 2x upsamplers
            num_upsample_blocks = int(torch.log2(torch.tensor(scale_factor)))
            for _ in range(num_upsample_blocks):
                upsampling_layers.append(UpsampleBlock(num_features, scale_factor=2))
        else:
            raise ValueError(f"Scale factor {scale_factor} not supported. Use 2, 4, or 8.")
        
        self.upsampling = nn.Sequential(*upsampling_layers)
        
        # ========== RECONSTRUCTION ==========
        # Final convolution to produce RGB output
        self.reconstruction = nn.Conv2d(
            num_features, 
            in_channels, 
            kernel_size=3, 
            padding=1
        )
        
        # ========== INITIALIZATION ==========
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Low-resolution input (B, C, H, W)
               Can be either native LR or bicubic-upscaled
        
        Returns:
            High-resolution output (B, C, H*scale, W*scale)
        """
        # Store input for global skip connection
        if self.use_global_skip:
            # If input is not at target size, upsample it
            if x.shape[-2:] != (x.shape[-2] * self.scale_factor, x.shape[-1] * self.scale_factor):
                x_upsampled = F.interpolate(
                    x, 
                    scale_factor=self.scale_factor, 
                    mode='bicubic', 
                    align_corners=False
                )
            else:
                x_upsampled = x
        
        # Feature extraction
        features = self.feature_extraction(x)
        
        # Residual blocks
        residual_output = self.residual_blocks(features)
        
        # Bottleneck (with skip connection from feature extraction)
        bottleneck_output = self.bottleneck(residual_output)
        bottleneck_output = bottleneck_output + features  # Skip connection
        
        # Upsampling
        upsampled = self.upsampling(bottleneck_output)
        
        # Reconstruction
        output = self.reconstruction(upsampled)
        
        # Global skip connection (add upsampled input to output)
        if self.use_global_skip:
            output = output + x_upsampled
        
        return output
    
    @property
    def num_parameters(self) -> int:
        """Return total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self) -> str:
        """Return model architecture summary"""
        return f"""
{'='*70}
Enhanced SRCNN (ESRCNN) Model Summary
{'='*70}
Scale Factor:           {self.scale_factor}x
Total Parameters:       {self.num_parameters:,}
Global Skip Connection: {self.use_global_skip}

Architecture:
  1. Feature Extraction:   Conv(3→64) + ReLU
  2. Residual Blocks:      {len(self.residual_blocks)} blocks
  3. Bottleneck:           Conv(64→64) + BatchNorm + Skip
  4. Upsampling:           {int(torch.log2(torch.tensor(self.scale_factor)))}x Sub-Pixel Conv
  5. Reconstruction:       Conv(64→3)

Expected Improvements:
  - Deeper network captures more complex features
  - Residual connections enable better gradient flow
  - Sub-pixel upsampling is more efficient than bicubic pre-upscaling
  - Batch normalization improves training stability
{'='*70}
"""


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG19 features
    Crucial for face super-resolution to preserve facial structure
    """
    def __init__(
        self, 
        feature_layers: list = [3, 8, 17, 26],  # relu1_2, relu2_2, relu3_4, relu4_4
        use_input_norm: bool = True
    ):
        """
        Initialize Perceptual Loss
        
        Args:
            feature_layers: Which VGG19 layers to use for loss
            use_input_norm: Normalize inputs to ImageNet stats
        """
        super(PerceptualLoss, self).__init__()
        
        # Load pretrained VGG19
        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        except:
            from torchvision.models import vgg19
            vgg = vgg19(pretrained=True)
        
        self.feature_layers = feature_layers
        self.use_input_norm = use_input_norm
        
        # Extract features from specified layers
        self.feature_extractors = nn.ModuleList()
        current_layer = 0
        for layer_idx in feature_layers:
            self.feature_extractors.append(
                nn.Sequential(*list(vgg.features.children())[current_layer:layer_idx+1])
            )
            current_layer = layer_idx + 1
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # ImageNet normalization
        if use_input_norm:
            self.register_buffer(
                'mean', 
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                'std', 
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss
        
        Args:
            pred: Predicted image (B, 3, H, W)
            target: Target image (B, 3, H, W)
        
        Returns:
            Perceptual loss (scalar)
        """
        # Normalize inputs if needed
        if self.use_input_norm:
            pred = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std
        
        # Extract features and compute loss
        loss = 0.0
        pred_features = pred
        target_features = target
        
        for extractor in self.feature_extractors:
            pred_features = extractor(pred_features)
            target_features = extractor(target_features)
            
            # L1 loss on features
            loss += F.l1_loss(pred_features, target_features)
        
        return loss / len(self.feature_extractors)


# ========== HELPER FUNCTIONS ==========

def create_esrcnn(
    scale_factor: int = 2,
    num_residual_blocks: int = 10,
    num_features: int = 64,
    **kwargs
) -> EnhancedSRCNN:
    """
    Factory function to create Enhanced SRCNN
    
    Args:
        scale_factor: Upscaling factor (2, 4, 8)
        num_residual_blocks: Number of residual blocks
        num_features: Number of feature channels
        **kwargs: Additional arguments
    
    Returns:
        EnhancedSRCNN model
    """
    return EnhancedSRCNN(
        in_channels=3,
        num_features=num_features,
        num_residual_blocks=num_residual_blocks,
        scale_factor=scale_factor,
        use_global_skip=True
    )


if __name__ == "__main__":
    # Test the model
    model = create_esrcnn(scale_factor=2, num_residual_blocks=10)
    print(model.summary())
    
    # Test forward pass
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Expected shape: (1, 3, {64*2}, {64*2})")
