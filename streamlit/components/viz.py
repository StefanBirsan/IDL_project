"""
Inference and visualization utilities for Streamlit app
"""
import torch
import numpy as np
import streamlit as st
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches


@st.cache_resource
def load_model_checkpoint(checkpoint_path: str, device: str = 'cpu'):
    """Load model checkpoint with caching"""
    try:
        from training.train_utils import create_physics_informed_mae
        from training.inference import Inference
        
        model = create_physics_informed_mae(
            img_size=64,
            patch_size=4,
            embed_dim=768,
            encoder_depth=12,
            decoder_depth=8,
            num_heads=12,
            mask_ratio=0.75
        )
        
        inference_engine = Inference(
            model=model,
            checkpoint_path=checkpoint_path,
            device=device
        )
        return inference_engine
    except Exception as e:
        st.error(f"Error loading checkpoint: {e}")
        return None


def load_npy_file(file_path: str) -> np.ndarray:
    """Load numpy file with error handling"""
    try:
        data = np.load(file_path)
        return data
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def normalize_image(img: np.ndarray, mode: str = 'tanh') -> np.ndarray:
    """
    Normalize image based on mode
    
    Args:
        img: Input image
        mode: 'tanh' (-1 to 1) or 'minmax' (0 to 1)
    
    Returns:
        Normalized image
    """
    img = np.array(img, dtype=np.float32)
    
    if mode == 'tanh':
        # Normalize to 0-1 first, then apply tanh
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        return np.tanh(img)
    else:  # minmax
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            return (img - vmin) / (vmax - vmin)
        return img


def denormalize_image(img: np.ndarray, original_img: np.ndarray) -> np.ndarray:
    """
    Denormalize image back to original range
    
    Args:
        img: Normalized image
        original_img: Original image for reference
    
    Returns:
        Denormalized image
    """
    vmin, vmax = original_img.min(), original_img.max()
    if vmax > vmin:
        return img * (vmax - vmin) + vmin
    return img


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute PSNR between two images
    
    Args:
        img1: Reference image
        img2: Comparison image
    
    Returns:
        PSNR value
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
    return psnr


def compute_ssim(img1: np.ndarray, img2: np.ndarray, window_size: int = 11) -> float:
    """
    Approximate SSIM computation
    
    Args:
        img1: Reference image
        img2: Comparison image
        window_size: Size of local window
    
    Returns:
        SSIM value (0-1)
    """
    from scipy.ndimage import uniform_filter, uniform_filter1d
    
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    
    # Compute means
    mu1 = uniform_filter(img1, window_size)
    mu2 = uniform_filter(img2, window_size)
    
    # Compute variances
    mu1_sq = uniform_filter(img1 ** 2, window_size)
    mu2_sq = uniform_filter(img2 ** 2, window_size)
    sigma12 = uniform_filter(img1 * img2, window_size)
    
    sigma1_sq = mu1_sq - mu1 ** 2
    sigma2_sq = mu2_sq - mu2 ** 2
    sigma12 = sigma12 - mu1 * mu2
    
    # Compute SSIM
    num = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denom = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
    
    return np.mean(num / denom)


def visualize_patches(img: np.ndarray, patch_size: int = 4) -> plt.Figure:
    """
    Visualize image patches grid
    
    Args:
        img: Input image
        patch_size: Size of patches
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(
        img.shape[0] // patch_size,
        img.shape[1] // patch_size,
        figsize=(10, 10)
    )
    
    for i in range(img.shape[0] // patch_size):
        for j in range(img.shape[1] // patch_size):
            ax = axes[i, j]
            patch = img[
                i*patch_size:(i+1)*patch_size,
                j*patch_size:(j+1)*patch_size
            ]
            ax.imshow(patch, cmap='gray')
            ax.set_title(f"Patch ({i},{j})")
            ax.axis('off')
    
    plt.tight_layout()
    return fig


def visualize_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    edge_map: Optional[np.ndarray] = None,
    title: str = "Input/Output Comparison"
) -> plt.Figure:
    """
    Create comparison visualization
    
    Args:
        original: Original image
        reconstructed: Reconstructed image
        edge_map: Edge map (optional)
        title: Figure title
    
    Returns:
        Matplotlib figure
    """
    num_plots = 3 if edge_map is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))
    
    # Original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    # Reconstructed
    axes[1].imshow(reconstructed, cmap='gray')
    axes[1].set_title("Reconstructed Image")
    axes[1].axis('off')
    
    # Edge map
    if edge_map is not None and num_plots > 2:
        axes[2].imshow(edge_map, cmap='hot')
        axes[2].set_title("Edge Map")
        axes[2].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_error_map(
    original: np.ndarray,
    reconstructed: np.ndarray,
    colormap: str = 'RdYlGn_r'
) -> plt.Figure:
    """
    Visualize error between original and reconstructed
    
    Args:
        original: Original image
        reconstructed: Reconstructed image
        colormap: Colormap name
    
    Returns:
        Matplotlib figure
    """
    error = np.abs(original - reconstructed)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Error map
    im = axes[0].imshow(error, cmap=colormap)
    axes[0].set_title("Absolute Error Map")
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0])
    
    # Sorted error
    error_sorted = np.sort(error.flatten())
    axes[1].plot(error_sorted)
    axes[1].set_title("Error Distribution")
    axes[1].set_xlabel("Pixel Index (sorted)")
    axes[1].set_ylabel("Absolute Error")
    axes[1].grid(True, alpha=0.3)
    
    # Error histogram
    axes[2].hist(error.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[2].set_title("Error Histogram")
    axes[2].set_xlabel("Absolute Error")
    axes[2].set_ylabel("Frequency")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_metrics_summary(
    original: np.ndarray,
    reconstructed: np.ndarray
) -> dict:
    """
    Compute comprehensive metrics
    
    Args:
        original: Original image
        reconstructed: Reconstructed image
    
    Returns:
        Dictionary of metrics
    """
    # Ensure same shape
    if original.shape != reconstructed.shape:
        reconstructed = reconstructed[:original.shape[0], :original.shape[1]]
    
    mae = np.mean(np.abs(original - reconstructed))
    mse = np.mean((original - reconstructed) ** 2)
    rmse = np.sqrt(mse)
    psnr = compute_psnr(original, reconstructed)
    
    # Approximate SSIM
    try:
        ssim = compute_ssim(original, reconstructed)
    except:
        ssim = 0.0
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'PSNR': psnr,
        'SSIM': ssim,
    }
