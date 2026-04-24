import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2

from training.train_utils.srcnn import SRCNN
from training.core.config_srcnn import SRCNNTrainingConfig


class SRCNNInference:
    """Inference manager for trained SRCNN models"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize inference with trained model.
        
        Args:
            model_path: Path to checkpoint (.pth) or ONNX model
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.model.eval()
    
    def _load_model(self, model_path: str) -> SRCNN:
        """Load trained SRCNN model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract config
        config_dict = checkpoint.get('config', {})
        config = SRCNNTrainingConfig(**config_dict)
        
        # Create model
        model = SRCNN(
            in_channels=config.input_channels,
            intermediate_channels=config.intermediate_channels,
            scale_factor=config.scale_factor
        ).to(self.device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded SRCNN model from {model_path}")
        print(f"Scale factor: {config.scale_factor}x")
        print(f"Model parameters: {model.num_parameters:,}")
        
        return model
    
    # TODO: This method should accept an already created LR image
    #  not create one from a given HR image
    @torch.no_grad()
    def infer(self,
              image: Union[np.ndarray, str, Path],
              scale_factor: int = 2) -> np.ndarray:
        """
        Apply super-resolution to an image.
        
        Args:
            image: LR input image (as numpy array or file path)
            scale_factor: Super-resolution upscaling factor
        Returns:
            Model super-resolution output (numpy array, normalized to [0, 1])
        """
        # Load image
        #  and process into floating-point format [0,1]
        if isinstance(image, (str, Path)):
            img_pil = Image.open(image).convert('RGB')
            img_array = np.array(img_pil, dtype=np.float32) / 255.0
        elif isinstance(image, np.ndarray):
            img_array = image.astype(np.float32)
            if img_array.max() > 1.0:
                img_array = img_array / 255.0
        else:
            raise ValueError("Image must be file path or numpy array")
        
        # Convert to tensor (C, H, W)
        h, w = img_array.shape[:2]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Downscale
        h_lr = (h // scale_factor) * scale_factor
        w_lr = (w // scale_factor) * scale_factor
        img_tensor = img_tensor[:, :, :h_lr, :w_lr]
        
        # Create LR version and bicubic upscale
        img_pil = Image.fromarray((img_tensor[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        
        lr_pil = img_pil.resize((w_lr // scale_factor, h_lr // scale_factor), Image.BICUBIC)
        lr_upscaled = lr_pil.resize((w_lr, h_lr), Image.BICUBIC)

        # Convert to tensor and normalize
        lr_tensor = torch.from_numpy(
            np.array(lr_upscaled, dtype=np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        # Run SRCNN
        lr_tensor = lr_tensor.to(self.device)
        sr_tensor = self.model(lr_tensor)

        # Clip to [0, 1] and convert to numpy
        sr_array = sr_tensor[0].permute(1, 2, 0).cpu().numpy()
        sr_array = np.clip(sr_array, 0, 1)

        return sr_array
    
    @torch.no_grad()
    def batch_super_resolve(self,
                           image_dir: Union[str, Path],
                           output_dir: Union[str, Path],
                           scale_factor: int = 2) -> None:
        """
        Apply super-resolution to all images in a directory.
        
        Args:
            image_dir: Directory containing images to process
            output_dir: Directory to save super-resolved images
            scale_factor: Super-resolution upscaling factor
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = sorted(
            list(image_dir.glob('*.jpg')) +
            list(image_dir.glob('*.jpeg')) +
            list(image_dir.glob('*.png')) +
            list(image_dir.glob('*.PNG'))
        )
        
        print(f"\nProcessing {len(image_files)} images...")
        
        for i, img_path in enumerate(image_files, 1):
            try:
                # Super-resolve
                sr_array = self.infer(img_path, scale_factor=scale_factor)
                
                # Save
                sr_uint8 = (sr_array * 255).astype(np.uint8)
                output_path = output_dir / f"sr_{img_path.stem}.png"
                Image.fromarray(sr_uint8).save(output_path)
                
                print(f"  [{i}/{len(image_files)}] {img_path.name} -> {output_path.name}")
            
            except Exception as e:
                print(f"  [{i}/{len(image_files)}] Error processing {img_path.name}: {str(e)}")


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        img1, img2: Images normalized to [0, 1]
        
    Returns:
        PSNR value in dB
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    return psnr


def compute_ssim(img1: np.ndarray, img2: np.ndarray, window_size: int = 11) -> float:
    """
    Compute Structural Similarity Index (SSIM) - simplified version.
    For production, use skimage.metrics.structural_similarity instead.
    
    Args:
        img1, img2: Images normalized to [0, 1]
        window_size: Window size for computation
        
    Returns:
        SSIM value in range [-1, 1]
    """
    # Convert to grayscale for simplified SSIM
    if img1.ndim == 3:
        img1_gray = np.mean(img1, axis=2)
        img2_gray = np.mean(img2, axis=2)
    else:
        img1_gray = img1
        img2_gray = img2
    
    # Compute local means
    kernel = cv2.getGaussianKernel(window_size, 1.5)
    kernel = kernel @ kernel.T
    
    mu1 = cv2.filter2D(img1_gray, -1, kernel)
    mu2 = cv2.filter2D(img2_gray, -1, kernel)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1_gray ** 2, -1, kernel) - mu1_sq
    sigma2_sq = cv2.filter2D(img2_gray ** 2, -1, kernel) - mu2_sq
    sigma12 = cv2.filter2D(img1_gray * img2_gray, -1, kernel) - mu1_mu2
    
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return np.mean(ssim_map)


# ============ Example usage =============

def example_single_image_sr():
    """Example: Apply super-resolution to a single image"""
    print("\n" + "="*70)
    print("SRCNN Single Image Super-Resolution Example")
    print("="*70)
    
    # Paths
    checkpoint_path = 'checkpoints/srcnn/best_model.pth'
    input_image = 'sample_images/low_res_face.jpg'
    output_path = 'sample_images/sr_face.png'
    
    # Initialize
    sr = SRCNNInference(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Super-resolve
    sr_image = sr.infer(input_image, scale_factor=2)
    
    # Save
    sr_uint8 = (sr_image * 255).astype(np.uint8)
    Image.fromarray(sr_uint8).save(output_path)
    
    print(f"Super-resolved image saved to {output_path}")


def example_batch_processing():
    """Example: Batch process images from a directory"""
    print("\n" + "="*70)
    print("SRCNN Batch Processing Example")
    print("="*70)
    
    checkpoint_path = 'checkpoints/srcnn/best_model.pth'
    input_dir = 'dataset/ffhq/test'
    output_dir = 'results/srcnn_sr'
    
    sr = SRCNNInference(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    sr.batch_super_resolve(input_dir, output_dir, scale_factor=2)


if __name__ == '__main__':
    # Uncomment to run examples
    # example_single_image_sr()
    # example_batch_processing()
    pass
