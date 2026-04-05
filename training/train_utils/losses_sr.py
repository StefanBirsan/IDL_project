"""
Loss functions for Super-Resolution training
Includes L1, SSIM, FFT, and other SR-specific losses
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (smoother L1 for better gradients)
    L_C = sqrt((x - y)^2 + epsilon^2)
    """
    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W)
            target: (B, C, H, W)
        Returns:
            scalar loss
        """
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.epsilon ** 2)
        return loss.mean()


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Loss
    Uses windowed SSIM computation
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5, reduction: str = 'mean'):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.reduction = reduction
        
        # Create Gaussian kernel
        kernel = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2
        kernel = torch.exp(-kernel.pow(2.0) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel_2d = kernel.view(-1, 1) * kernel.view(1, -1)
        self.register_buffer('kernel', kernel_2d.unsqueeze(0).unsqueeze(0))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
        """
        Compute SSIM loss (1 - SSIM)
        
        Args:
            pred: (B, C, H, W)
            target: (B, C, H, W)
            data_range: dynamic range of data (default 1.0 for normalized images)
        Returns:
            scalar loss
        """
        B, C, H, W = pred.shape
        
        # Extend kernel to match number of channels
        kernel = self.kernel.repeat(C, 1, 1, 1)
        
        # Compute local means
        mu1 = F.conv2d(pred, kernel, padding=self.window_size // 2, groups=C)
        mu2 = F.conv2d(target, kernel, padding=self.window_size // 2, groups=C)
        
        # Compute local variances and covariance
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred ** 2, kernel, padding=self.window_size // 2, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, kernel, padding=self.window_size // 2, groups=C) - mu2_sq
        sigma12 = F.conv2d(pred * target, kernel, padding=self.window_size // 2, groups=C) - mu1_mu2
        
        # SSIM formula constants
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        
        # Compute SSIM
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        # Return loss (1 - SSIM)
        loss = 1 - ssim_map.mean()
        return loss


class FFTLoss(nn.Module):
    """
    Frequency Domain Loss using FFT
    Compares magnitude spectra of predictions and targets
    Helps preserve global structure and fine details
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute FFT loss on magnitude spectrum
        
        Args:
            pred: (B, C, H, W)
            target: (B, C, H, W)
        Returns:
            scalar loss
        """
        # Compute FFT
        pred_fft = torch.fft.rfft2d(pred)
        target_fft = torch.fft.rfft2d(target)
        
        # Get magnitude spectra
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # L1 distance on magnitude spectrum
        loss = F.l1_loss(pred_mag, target_mag, reduction=self.reduction)
        
        return loss


class MaskedReconstructionLoss(nn.Module):
    """
    MAE Reconstruction Loss - computed on masked patches only
    Optionally includes visible patches with reduced weight
    """
    def __init__(self, reduction: str = 'mean', visible_weight: float = 0.0):
        super().__init__()
        self.reduction = reduction
        self.visible_weight = visible_weight  # 0.0 = masked only (true MAE)
    
    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor,
                patch_size: int) -> torch.Tensor:
        """
        Compute MAE reconstruction loss on patches
        
        Args:
            pred: (B, C, H, W) predicted patches (flattened)
            target: (B, C, H, W) target patches (flattened)
            mask: (B, num_patches) binary mask (1 = masked, 0 = visible)
            patch_size: size of patch
        Returns:
            scalar loss
        """
        B, C, H, W = target.shape
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        
        # Extract to patches
        target_patches = target.view(B, C, num_patches_h, patch_size,
                                      num_patches_w, patch_size)
        target_patches = target_patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        target_patches = target_patches.view(B, -1, C * patch_size * patch_size)
        
        pred_patches = pred.view(B, C, num_patches_h, patch_size,
                                  num_patches_w, patch_size)
        pred_patches = pred_patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        pred_patches = pred_patches.view(B, -1, C * patch_size * patch_size)
        
        # Reconstruction loss
        recon_loss = (pred_patches - target_patches) ** 2
        
        # Apply mask: weight masked patches higher
        # mask shape: (B, num_patches)
        mask_expanded = mask.unsqueeze(-1)  # (B, num_patches, 1)
        
        if self.visible_weight > 0:
            # Include visible patches with reduced weight
            weights = mask_expanded.float() + self.visible_weight * (1 - mask_expanded.float())
        else:
            # Masked-only (true MAE)
            weights = mask_expanded.float()
        
        weighted_loss = recon_loss * weights
        loss = weighted_loss.mean()
        
        return loss


def compute_multiresolution_losses(pred: torch.Tensor,
                                    target: torch.Tensor,
                                    scales: list = [1, 2, 4]) -> dict:
    """
    Compute losses at multiple resolutions
    Useful for multi-scale supervision
    
    Args:
        pred: (B, C, H, W) predicted HR image
        target: (B, C, H, W) target HR image
        scales: downsampling scales to evaluate (1 = full res)
    Returns:
        dict with loss_scale_{i} keys
    """
    losses = {}
    
    for scale in scales:
        if scale == 1:
            pred_scaled = pred
            target_scaled = target
        else:
            # Downsample to lower resolution
            pred_scaled = F.avg_pool2d(pred, kernel_size=scale, stride=scale)
            target_scaled = F.avg_pool2d(target, kernel_size=scale, stride=scale)
        
        # L1 loss at this scale
        loss_l1 = F.l1_loss(pred_scaled, target_scaled)
        losses[f'loss_l1_scale_{scale}'] = loss_l1
    
    return losses
