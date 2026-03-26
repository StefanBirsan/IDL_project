"""
Core modules for Physics-Informed Masked Vision Transformer
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PhysicsInformedPreprocessing(nn.Module):
    """
    Physics-Informed Preprocessing Module
    - Performs double differentiation for edge detection
    - Applies Tanh normalization for HDR stability
    """
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        
        # Sobel filters for edge detection (double differentiation)
        self.register_buffer('sobel_x', torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ]).view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1., -2., -1.],
            [0., 0., 0.],
            [1., 2., 1.]
        ]).view(1, 1, 3, 3))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input image tensor (B, C, H, W)
        Returns:
            normalized_x: Tanh-normalized image
            edge_map: Edge detection map
        """
        B, C, H, W = x.shape
        
        # Normalize input
        x_normalized = torch.tanh(x)
        
        # Compute gradients (first differentiation)
        if C == 1:
            gx = F.conv2d(x, self.sobel_x, padding=1)
            gy = F.conv2d(x, self.sobel_y, padding=1)
        else:
            # For multi-channel, apply to each channel
            gx_list = []
            gy_list = []
            for i in range(C):
                gx_list.append(F.conv2d(x[:, i:i+1, :, :], self.sobel_x, padding=1))
                gy_list.append(F.conv2d(x[:, i:i+1, :, :], self.sobel_y, padding=1))
            gx = torch.cat(gx_list, dim=1)
            gy = torch.cat(gy_list, dim=1)
        
        # Compute magnitude (second differentiation → edge strength)
        edge_map = torch.sqrt(gx**2 + gy**2 + 1e-8)
        
        # Normalize edge map
        edge_map = torch.tanh(edge_map / (edge_map.max() + 1e-8))
        
        return x_normalized, edge_map


class PatchEmbedding(nn.Module):
    """
    Asymmetric Patch Partition and Embedding
    Converts image to non-overlapping patches and embeds them
    """
    def __init__(self, 
                 img_size: int = 64,
                 patch_size: int = 4,
                 in_channels: int = 1,
                 embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        
        self.proj = nn.Linear(self.patch_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image (B, C, H, W)
        Returns:
            patches: Embedded patches (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        
        # Reshape to patches
        x = x.reshape(
            B,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size
        )
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, -1, self.patch_dim)
        
        # Project and normalize
        x = self.proj(x)
        x = self.norm(x)
        return x


class MaskedPatchEmbedding(nn.Module):
    """
    Masked patch embedding with configurable masking ratio
    """
    def __init__(self,
                 img_size: int = 64,
                 patch_size: int = 4,
                 in_channels: int = 1,
                 embed_dim: int = 768,
                 mask_ratio: float = 0.75):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.mask_ratio = mask_ratio
        self.num_patches = self.patch_embed.num_patches
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input image (B, C, H, W)
        Returns:
            visible_patches: Patches that are not masked (B, num_visible, embed_dim)
            mask: Binary mask (B, num_patches)
            ids_restore: Indices to restore original order (B, num_patches)
        """
        # Get patch embeddings
        patches = self.patch_embed(x)  # (B, num_patches, embed_dim)
        B, N, D = patches.shape
        
        # Generate random mask
        num_mask = int(N * self.mask_ratio)
        noise = torch.rand(B, N, device=patches.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Create mask: 1 for masked, 0 for visible
        mask = torch.ones((B, N), device=patches.device)
        mask[:, :num_mask] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        # Get visible patches
        ids_keep = ids_shuffle[:, num_mask:]
        visible_patches = torch.gather(patches, dim=1, 
                                      index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        return visible_patches, mask, ids_restore


class FluxGuidanceGeneration(nn.Module):
    """
    Flux Guidance Generation Module
    - Detects celestial objects via edge maps
    - Generates flux maps using rotatable Gaussian kernels
    """
    def __init__(self,
                 img_size: int = 64,
                 patch_size: int = 4,
                 num_scales: int = 4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_scales = num_scales
        self.feature_size = img_size // patch_size
        
    def gaussian_kernel_2d(self, 
                          kernel_size: int,
                          sigma: float,
                          angle: float = 0.0) -> torch.Tensor:
        """Generate rotatable 2D Gaussian kernel"""
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        
        # Rotation matrix
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        xx_rot = cos_a * xx - sin_a * yy
        yy_rot = sin_a * xx + cos_a * yy
        
        kernel = torch.exp(-(xx_rot**2 + yy_rot**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel
    
    def forward(self, 
               edge_map: torch.Tensor,
               flux_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            edge_map: Edge detection map from preprocessing (B, C, H, W)
            flux_weights: Pre-calculated flux values per patch (optional)
        Returns:
            flux_map: Flux map for guided attention (B, 1, feature_size, feature_size)
        """
        B, C, H, W = edge_map.shape
        
        # Aggregate edge map to patch level
        edge_patches = F.max_pool2d(
            edge_map,
            kernel_size=self.patch_size,
            stride=self.patch_size
        ).squeeze(1)  # (B, feature_size, feature_size)
        
        # Create flux map - assume flux weights are provided or use edge intensity
        if flux_weights is None:
            flux_map = edge_patches.unsqueeze(1)  # (B, 1, feature_size, feature_size)
        else:
            flux_map = flux_weights.view(B, 1, self.feature_size, self.feature_size)
        
        # Normalize flux map
        flux_map = torch.tanh(flux_map)
        
        return flux_map


class RotaryPositionalEmbedding(nn.Module):
    """
    Sine-Cosine Positional Embeddings with support for 2D patches
    """
    def __init__(self, embed_dim: int, max_seq_len: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Pre-compute position embeddings
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * 
            (-math.log(10000.0) / embed_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings (B, N, embed_dim)
        Returns:
            x + positional embeddings
        """
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention with proper scaling"""
    def __init__(self, embed_dim: int, num_heads: int = 8, attn_drop: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input (B, N, embed_dim)
            mask: Attention mask (optional)
        Returns:
            Output (B, N, embed_dim)
        """
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer Block"""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0,
                 drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class FluxGuidanceController(nn.Module):
    """
    Flux Guidance Controller (FGC)
    Injects flux maps into encoder features via global attention
    """
    def __init__(self, embed_dim: int, num_scales: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_scales = num_scales
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.control_mlp = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim)
        )
    
    def forward(self, 
               features: torch.Tensor,
               flux_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Encoder features (B, N, embed_dim)
            flux_map: Flux guidance map (B, 1, H_p, W_p)
        Returns:
            Modulated features
        """
        B, N, D = features.shape
        
        # Global average pooling on flux map
        flux_context = self.pool(flux_map).view(B, 1)  # (B, 1)
        
        # Generate control signal
        control = self.control_mlp(flux_context)  # (B, embed_dim)
        
        # Apply modulation (multiplicative + additive)
        modulated = features * (1.0 + control.unsqueeze(1))
        
        return modulated


class PixelShuffleUpsample(nn.Module):
    """Sub-pixel upsampling using PixelShuffle"""
    def __init__(self, in_channels: int, out_channels: int, upscale_factor: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * (upscale_factor ** 2),
            kernel_size=3,
            padding=1
        )
        self.shuffle = nn.PixelShuffle(upscale_factor)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.shuffle(x)
        return x


class CNNRefinementHead(nn.Module):
    """Lightweight CNN refinement head with ReLU activations"""
    def __init__(self, 
                 in_channels: int = 768,
                 hidden_channels: int = 256,
                 out_channels: int = 16):  # patch_size^2
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DropPath(nn.Module):
    """Drop path (stochastic depth) as described in `Deep Networks with Stochastic Depth`"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.bernoulli(torch.ones(shape) * keep_prob)
        return x * random_tensor / keep_prob
