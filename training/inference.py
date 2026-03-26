"""
Inference script for Physics-Informed Masked Vision Transformer
Demonstrates loading a trained model and performing reconstruction
"""
import torch
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

from training.train_utils import create_physics_informed_mae
from utils.numpy_dataset import get_dataset


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str = 'cuda'):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model


def reconstruct_image(model: torch.nn.Module,
                     image: torch.Tensor,
                     device: str = 'cuda') -> torch.Tensor:
    """
    Reconstruct an image using the trained model
    
    Args:
        model: Physics-Informed MAE model
        image: Input image tensor (1, C, H, W)
        device: Computation device
    Returns:
        reconstructed: Reconstructed image
    """
    model.eval()
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        reconstructed, aux_outputs = model(image)
    
    return reconstructed.squeeze(0).cpu(), aux_outputs


def visualize_reconstruction(original: np.ndarray,
                           reconstructed: np.ndarray,
                           flux_map: np.ndarray = None,
                           edge_map: np.ndarray = None,
                           save_path: str = None):
    """Visualize original vs reconstructed"""
    
    n_cols = 2
    if flux_map is not None:
        n_cols += 1
    if edge_map is not None:
        n_cols += 1
    
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    
    if n_cols == 1:
        axes = [axes]
    
    # Original
    im0 = axes[0].imshow(original.squeeze(), cmap='viridis')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0])
    
    # Reconstructed
    im1 = axes[1].imshow(reconstructed.squeeze(), cmap='viridis')
    axes[1].set_title('Reconstructed Image')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    col_idx = 2
    
    # Flux map
    if flux_map is not None:
        im2 = axes[col_idx].imshow(flux_map.squeeze(), cmap='hot')
        axes[col_idx].set_title('Flux Map')
        axes[col_idx].axis('off')
        plt.colorbar(im2, ax=axes[col_idx])
        col_idx += 1
    
    # Edge map
    if edge_map is not None:
        im3 = axes[col_idx].imshow(edge_map.squeeze(), cmap='gray')
        axes[col_idx].set_title('Edge Map')
        axes[col_idx].axis('off')
        plt.colorbar(im3, ax=axes[col_idx])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def compute_metrics(original: np.ndarray,
                   reconstructed: np.ndarray) -> dict:
    """Compute reconstruction metrics"""
    
    # MSE
    mse = np.mean((original - reconstructed) ** 2)
    
    # PSNR
    max_val = np.max(np.abs(original))
    if max_val == 0:
        max_val = 1.0
    psnr = 20 * np.log10(max_val / np.sqrt(mse + 1e-8))
    
    # SSIM (simplified)
    # For full SSIM, use skimage.metrics.structural_similarity
    
    return {
        'mse': float(mse),
        'psnr': float(psnr),
    }


class InferenceEngine:
    """Inference engine for Physics-Informed MAE"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device)
        self.model = None
        self.load_model(model_path)
    
    def load_model(self, model_path: str, img_size: int = 64):
        """Load trained model"""
        self.model = create_physics_informed_mae(img_size=img_size).to(self.device)
        self.model = load_checkpoint(self.model, model_path, device=str(self.device))
    
    def infer(self, image: torch.Tensor) -> dict:
        """
        Run inference on an image
        
        Args:
            image: Input image tensor (C, H, W)
        Returns:
            results: Dict with reconstructed image and metrics
        """
        original = image.cpu().numpy()
        reconstructed, aux_outputs = reconstruct_image(
            self.model, image, device=str(self.device)
        )
        reconstructed = reconstructed.numpy()
        
        # Compute metrics
        metrics = compute_metrics(original, reconstructed)
        
        # Extract auxiliary outputs
        flux_map = aux_outputs['flux_map'].cpu().squeeze().numpy()
        edge_map = aux_outputs['edge_map'].cpu().squeeze().numpy()
        mask = aux_outputs['mask'].cpu().numpy()
        
        return {
            'original': original,
            'reconstructed': reconstructed,
            'flux_map': flux_map,
            'edge_map': edge_map,
            'mask': mask,
            'metrics': metrics,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with Physics-Informed MAE"
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint')
    parser.add_argument('--data-dir', type=str, default='dataset/data',
                        help='Path to dataset')
    parser.add_argument('--sample-idx', type=int, default=0,
                        help='Sample index to reconstruct')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}")
    dataset = get_dataset(args.data_dir, split='eval', img_size=64)
    
    # Get sample
    sample_image, metadata = dataset[args.sample_idx]
    print(f"Loaded sample: {metadata['filename']}")
    print(f"Image shape: {sample_image.shape}")
    
    # Initialize inference engine
    print(f"\nLoading model from checkpoint: {args.checkpoint}")
    engine = InferenceEngine(args.checkpoint, device=args.device)
    
    # Run inference
    print("\nRunning inference...")
    results = engine.infer(sample_image)
    
    # Print metrics
    print(f"\nReconstruction Metrics:")
    print(f"  MSE:  {results['metrics']['mse']:.6f}")
    print(f"  PSNR: {results['metrics']['psnr']:.2f} dB")
    
    # Visualize
    print(f"\nVisualizing results...")
    save_path = output_dir / f"reconstruction_{metadata['filename'].replace('.npy', '.png')}"
    visualize_reconstruction(
        results['original'],
        results['reconstructed'],
        flux_map=results['flux_map'],
        edge_map=results['edge_map'],
        save_path=str(save_path)
    )
    
    print(f"\nInference complete!")


if __name__ == '__main__':
    main()
