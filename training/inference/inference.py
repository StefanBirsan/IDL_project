"""
Post-training inference engine for Physics-Informed MAE
Handles model loading, inference execution, and result visualization
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Literal
from PIL import Image
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)

class Inference:
    """Inference engine for Physics-Informed MAE models"""
    
    def __init__(self,
                 model: nn.Module,
                 checkpoint_path: Optional[str] = None,
                 device: Literal['cpu', 'cuda'] = 'cpu'):
        """
        Initialize inference engine
        
        Args:
            model: PyTorch model
            checkpoint_path: Path to model checkpoint (optional)
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()
        
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as exc:
            logger.warning(f"Failed to load checkpoint in strict mode: {exc}")
            logger.info("Retrying with strict=False for leniency")
            incompatible = self.model.load_state_dict(state_dict, strict=False)
            if incompatible.missing_keys:
                logger.info(f"Missing keys: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                logger.info(f"Unexpected keys: {incompatible.unexpected_keys}")
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    @staticmethod
    def _extract_prediction(model_output):
        """Extract image prediction when model returns either tensor or (tensor, aux)."""
        if isinstance(model_output, (tuple, list)):
            return model_output[0]
        return model_output

    def _forward_without_masking(self, x: torch.Tensor):
        """
        Run forward pass with MAE masking disabled.
        Masking: intentionally hiding a large fraction of image patches
        during forward pass then reconstructing missing content using
        the remaining visible patches
        This is good in training, however in inference
        we want full-information output from the whole input image
        thus we disable masking
        """
        original_model_mask_ratio = getattr(self.model, 'mask_ratio', None)
        original_patch_mask_ratio = getattr(getattr(self.model, 'patch_embed', None), 'mask_ratio', None)

        if original_model_mask_ratio is not None:
            self.model.mask_ratio = 0.0
        if original_patch_mask_ratio is not None:
            self.model.patch_embed.mask_ratio = 0.0

        try:
            return self.model(x)
        finally:
            if original_model_mask_ratio is not None:
                self.model.mask_ratio = original_model_mask_ratio
            if original_patch_mask_ratio is not None:
                self.model.patch_embed.mask_ratio = original_patch_mask_ratio

    @torch.no_grad()
    def infer(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Run inference on input image
        
        Args:
            image: Input image (numpy array or torch tensor)
            Shape: (C, H, W) or (1, C, H, W) batch
        Returns:
            Reconstructed image as torch tensor on CPU
        """
        # Convert to tensor if needed
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Move to device
        image = image.to(self.device)
        
        # Run inference
        with torch.no_grad():
            model_output = self._forward_without_masking(image)
            reconstructed = self._extract_prediction(model_output)
        
        return reconstructed.cpu()
    
    @torch.no_grad()
    def infer_batch(self,
                   images: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Run inference on batch of images
        
        Args:
            images: Batch of images
            Shape: (B, C, H, W)
        Returns:
            Reconstructed batch as torch tensor on CPU
        """
        # Convert to tensor if needed
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        
        # Move to device
        images = images.to(self.device)
        
        # Run inference
        with torch.no_grad():
            model_output = self._forward_without_masking(images)
            reconstructed = self._extract_prediction(model_output)
        
        return reconstructed.cpu()
    
    def visualize_inference(self,
                            image: np.ndarray,
                            ground_truth: Optional[np.ndarray] = None,
                            save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference and visualize results
        
        Args:
            image: Input image (H, W) or (C, H, W)
            ground_truth: Optional ground truth image for visual comparison (H, W) or (C, H, W)
            save_path: Optional path to save comparison image
        Returns:
            Tuple of (input_image, reconstructed_image)
        """
        # Convert to tensor format if needed
        if image.ndim == 2:
            image = np.expand_dims(image, 0)
        # Do the same for ground truth if provided
        if ground_truth is not None and ground_truth.ndim == 2:
            ground_truth = np.expand_dims(ground_truth, 0)

        # Run inference
        reconstructed = self.infer(image)  # Returns (1, C, H, W)

        # Convert back to numpy
        input_np = image[0] if image.ndim == 3 else image  # Remove channel dim for 2D
        output_np = reconstructed[0, 0].numpy() if reconstructed.shape[1] == 1 else reconstructed[0].numpy()

        # Create visualization
        num_cols = 3 if ground_truth is not None else 2
        fig, axes = plt.subplots(1, num_cols, figsize=(12, 5))

        axes[0].imshow(input_np, cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        axes[1].imshow(output_np, cmap='gray')
        axes[1].set_title('Reconstructed Image')
        axes[1].axis('off')

        if ground_truth is not None:
            axes[2].imshow(ground_truth[0], cmap='gray')
            axes[2].set_title('Ground Truth')
            axes[2].axis('off')

        plt.tight_layout()
        # save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        plt.show()
        
        return input_np, output_np
    
    @staticmethod
    def compute_metrics(input_image: np.ndarray,
                       reconstructed: np.ndarray) -> Dict[str, float]:
        """
        Compute inference quality metrics
        
        Args:
            input_image: Original image
            reconstructed: Reconstructed image
        Returns:
            Dictionary with metrics (MSE, MAE, PSNR)
        """
        mse = np.mean((input_image - reconstructed) ** 2)
        mae = np.mean(np.abs(input_image - reconstructed))
        
        # PSNR (assuming max pixel value is 1.0)
        max_val = 1.0
        psnr = 20 * np.log10(max_val / np.sqrt(mse + 1e-10))
        
        return {
            'MSE': float(mse),
            'MAE': float(mae),
            'PSNR': float(psnr),
        }
