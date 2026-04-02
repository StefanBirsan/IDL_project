"""
Example usage of the Streamlit app utilities
Shows how to:
- Load models and run inference
- Create custom visualizations
- Compute metrics
- Batch process images
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from streamlit.components import (
    load_model_checkpoint,
    load_npy_file,
    normalize_image,
    denormalize_image,
    visualize_comparison,
    visualize_error_map,
    visualize_patches,
    create_metrics_summary,
)
from streamlit.config import AppConfig


def example_1_basic_inference():
    """
    Example 1: Load a checkpoint and run basic inference
    """
    print("=" * 60)
    print("Example 1: Basic Inference")
    print("=" * 60)
    
    # Load model checkpoint
    checkpoint_path = project_root / "checkpoints" / "best_model.pt"
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        print("Please train the model first or provide a valid checkpoint path")
        return
    
    print(f"✓ Loading checkpoint from {checkpoint_path}")
    inference = load_model_checkpoint(str(checkpoint_path), device='cpu')
    
    if inference is None:
        print("❌ Failed to load inference engine")
        return
    
    # Load sample image
    sample_image_path = project_root / "dataset" / "data" / "x2" / "train_hr_patch" / "hst_10152_05_acs_wfc_f814w_j90i05_drc_padded_hr_hr_patch_10.npy"
    
    if not sample_image_path.exists():
        print(f"❌ Sample image not found at {sample_image_path}")
        return
    
    print(f"✓ Loading sample image from {sample_image_path}")
    original_image = load_npy_file(str(sample_image_path))
    
    if original_image is None:
        print("❌ Failed to load image")
        return
    
    print(f"  Image shape: {original_image.shape}")
    print(f"  Image range: [{original_image.min():.4f}, {original_image.max():.4f}]")
    
    # Normalize image
    normalized_image = normalize_image(original_image, mode='tanh')
    print(f"✓ Image normalized to range: [{normalized_image.min():.4f}, {normalized_image.max():.4f}]")
    
    # Run inference
    print("✓ Running inference...")
    reconstructed = inference.infer(normalized_image)
    reconstructed_np = reconstructed.numpy()[0, 0]  # Remove batch and channel dims
    
    print(f"✓ Reconstruction complete")
    print(f"  Output shape: {reconstructed_np.shape}")
    print(f"  Output range: [{reconstructed_np.min():.4f}, {reconstructed_np.max():.4f}]")
    
    # Compute metrics
    print("\n✓ Computing metrics...")
    # Ensure same shape and normalize to same range
    if original_image.shape != reconstructed_np.shape:
        reconstructed_np = reconstructed_np[:original_image.shape[0], :original_image.shape[1]]
    
    original_normalized = normalize_image(original_image)
    metrics = create_metrics_summary(original_normalized, reconstructed_np)
    
    print("\n📊 Metrics Summary:")
    for metric_name, value in metrics.items():
        if metric_name in ['PSNR']:
            print(f"  {metric_name:6s}: {value:8.2f}")
        elif metric_name in ['SSIM']:
            print(f"  {metric_name:6s}: {value:8.4f}")
        else:
            print(f"  {metric_name:6s}: {value:8.6f}")
    
    return original_normalized, reconstructed_np, metrics


def example_2_visualization():
    """
    Example 2: Create comparison visualizations
    """
    print("\n" + "=" * 60)
    print("Example 2: Visualization")
    print("=" * 60)
    
    # Run inference first
    result = example_1_basic_inference()
    if result is None:
        return
    
    original, reconstructed, metrics = result
    
    # Create comparison plot
    print("\n✓ Creating comparison visualization...")
    fig = visualize_comparison(
        original,
        reconstructed,
        title="Physics-Informed MAE: Input vs Reconstruction"
    )
    plt.savefig(
        project_root / "streamlit" / "outputs" / "comparison.png",
        dpi=100,
        bbox_inches='tight'
    )
    print(f"  Saved to: streamlit/outputs/comparison.png")
    plt.close()
    
    # Create error map visualization
    print("✓ Creating error analysis visualization...")
    fig = visualize_error_map(original, reconstructed)
    plt.savefig(
        project_root / "streamlit" / "outputs" / "error_analysis.png",
        dpi=100,
        bbox_inches='tight'
    )
    print(f"  Saved to: streamlit/outputs/error_analysis.png")
    plt.close()


def example_3_batch_processing():
    """
    Example 3: Process multiple images in batch
    """
    print("\n" + "=" * 60)
    print("Example 3: Batch Processing")
    print("=" * 60)
    
    # Find all patch files
    patch_dir = project_root / "dataset" / "data" / "x2" / "train_hr_patch"
    
    if not patch_dir.exists():
        print(f"❌ Patch directory not found at {patch_dir}")
        return
    
    patch_files = sorted(list(patch_dir.glob("*.npy")))[:5]  # First 5 patches
    
    if not patch_files:
        print(f"❌ No patch files found in {patch_dir}")
        return
    
    print(f"✓ Found {len(patch_files)} patch files")
    
    # Load checkpoint
    checkpoint_path = project_root / "checkpoints" / "best_model.pt"
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return
    
    inference = load_model_checkpoint(str(checkpoint_path), device='cpu')
    
    # Process each patch
    results = []
    for i, patch_path in enumerate(patch_files):
        print(f"\n  Processing patch {i+1}/{len(patch_files)}: {patch_path.name}")
        
        # Load and normalize
        original = load_npy_file(str(patch_path))
        if original is None:
            continue
        
        normalized = normalize_image(original)
        
        # Infer
        reconstructed = inference.infer(normalized)
        reconstructed_np = reconstructed.numpy()[0, 0]
        
        # Compute metrics
        if original.shape != reconstructed_np.shape:
            reconstructed_np = reconstructed_np[:original.shape[0], :original.shape[1]]
        
        metrics = create_metrics_summary(normalized, reconstructed_np)
        
        results.append({
            'filename': patch_path.name,
            'original': original,
            'reconstructed': reconstructed_np,
            'metrics': metrics
        })
        
        print(f"    PSNR: {metrics['PSNR']:.2f}, SSIM: {metrics['SSIM']:.4f}, MAE: {metrics['MAE']:.6f}")
    
    # Summary statistics
    print(f"\n✓ Batch processing complete")
    print(f"\nSummary Statistics (across {len(results)} images):")
    
    for metric_name in ['PSNR', 'SSIM', 'MAE', 'MSE']:
        values = [r['metrics'][metric_name] for r in results]
        print(f"  {metric_name:6s}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")
    
    # Save batch results
    print(f"\n✓ Saving batch results visualization...")
    fig, axes = plt.subplots(len(results), 3, figsize=(15, 5*len(results)))
    
    if len(results) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(results):
        axes[idx, 0].imshow(result['original'], cmap='gray')
        axes[idx, 0].set_title(f"Input: {result['filename']}")
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(result['reconstructed'], cmap='gray')
        axes[idx, 1].set_title(f"PSNR: {result['metrics']['PSNR']:.2f}")
        axes[idx, 1].axis('off')
        
        error = np.abs(result['original'] - result['reconstructed'])
        im = axes[idx, 2].imshow(error, cmap='hot')
        axes[idx, 2].set_title(f"MAE: {result['metrics']['MAE']:.6f}")
        axes[idx, 2].axis('off')
        plt.colorbar(im, ax=axes[idx, 2])
    
    plt.tight_layout()
    plt.savefig(
        project_root / "streamlit" / "outputs" / "batch_results.png",
        dpi=100,
        bbox_inches='tight'
    )
    print(f"  Saved to: streamlit/outputs/batch_results.png")
    plt.close()


def example_4_custom_metrics():
    """
    Example 4: Compute custom metrics and analysis
    """
    print("\n" + "=" * 60)
    print("Example 4: Custom Metrics & Analysis")
    print("=" * 60)
    
    # Run inference
    result = example_1_basic_inference()
    if result is None:
        return
    
    original, reconstructed, metrics = result
    
    # Additional custom metrics
    print("\n✓ Computing additional metrics...")
    
    # Local statistics
    window_size = 8
    local_errors = []
    
    for i in range(0, original.shape[0] - window_size, window_size):
        for j in range(0, original.shape[1] - window_size, window_size):
            patch_orig = original[i:i+window_size, j:j+window_size]
            patch_recon = reconstructed[i:i+window_size, j:j+window_size]
            
            local_mae = np.mean(np.abs(patch_orig - patch_recon))
            local_errors.append(local_mae)
    
    print(f"\n  Local Error Statistics (8×8 patches):")
    print(f"    Mean Local MAE: {np.mean(local_errors):.6f}")
    print(f"    Std Local MAE:  {np.std(local_errors):.6f}")
    print(f"    Min Local MAE:  {np.min(local_errors):.6f}")
    print(f"    Max Local MAE:  {np.max(local_errors):.6f}")
    
    # Gradient preservation
    from scipy.ndimage import sobel
    
    grad_orig = np.sqrt(sobel(original, axis=0)**2 + sobel(original, axis=1)**2)
    grad_recon = np.sqrt(sobel(reconstructed, axis=0)**2 + sobel(reconstructed, axis=1)**2)
    
    grad_error = np.mean(np.abs(grad_orig - grad_recon))
    
    print(f"\n  Gradient Preservation:")
    print(f"    Gradient MAE: {grad_error:.6f}")
    
    # Frequency analysis
    from scipy.fft import fft2, fftshift
    
    fft_orig = fftshift(fft2(original))
    fft_recon = fftshift(fft2(reconstructed))
    
    spektrum_orig = np.abs(fft_orig)
    spektrum_recon = np.abs(fft_recon)
    
    spektrum_error = np.mean(np.abs(spektrum_orig - spektrum_recon) / (spektrum_orig + 1e-8))
    
    print(f"\n  Frequency Domain:")
    print(f"    Spectrum MAPE: {spektrum_error:.6f}")


def create_outputs_directory():
    """Create outputs directory if it doesn't exist"""
    output_dir = project_root / "streamlit" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Streamlit App Examples")
    print("=" * 60)
    
    # Create output directory
    create_outputs_directory()
    
    # Run examples
    try:
        example_1_basic_inference()
        example_2_visualization()
        example_3_batch_processing()
        example_4_custom_metrics()
        
        print("\n" + "=" * 60)
        print("✓ All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during examples: {e}")
        import traceback
        traceback.print_exc()
