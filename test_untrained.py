from typing import Any, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from training.train_utils import create_physics_informed_mae
import torch.nn.functional as F
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.visualization import ZScaleInterval, ImageNormalize

from utils import NumpyAstronomicalDataset, get_dataset

"""
This script tests the untrained model 
on a random Low-Resolution (LR) and High-Resolution (HR)
sample pair from the dataset.
"""

# configuration
DATA_DIR = Path("dataset/data/x2")
DATASET_SPLIT = "train"  # or "eval"
IMG_SIZE = 128  # since we are using the x2 dataset
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_random_sample_images(
    dataset: NumpyAstronomicalDataset,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a random evaluation sample from the dataset and
    return the LR and HR images as numpy arrays (images).
    - LR image shape: (1, H, W)
    - HR image shape: (1, H, W)
    """

    lr_tensor, hr_tensor = get_random_sample_tensors(dataset)

    # convert tensors to numpy arrays
    lr_image = lr_tensor.numpy()
    hr_image = hr_tensor.numpy()

    return lr_image, hr_image


def get_random_sample_tensors(
    dataset: NumpyAstronomicalDataset,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get a random evaluation sample from the dataset and return the LR and HR images as tensors.
     - LR image tensor shape: (1, H, W)
     - HR image tensor shape: (1, H, W)
    """

    # get random sample
    tensors = dataset.get_random_sample()

    # convert tensors to numpy arrays for normalization
    lr_image = tensors["lr_image"].numpy()
    hr_image = tensors["hr_image"].numpy()

    # TODO: if this normalization method works well,
    #  use it directly in the dataset class
    lr_image = normalize_image(lr_image)
    hr_image = normalize_image(hr_image)

    # convert numpy arrays to tensors
    lr_tensor = torch.from_numpy(lr_image)
    hr_tensor = torch.from_numpy(hr_image)

    return lr_tensor, hr_tensor


# helper functions
def resize_image_to_shape(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """
    Resize 2D image to target (H, W).
    Uses area for downscaling, bicubic for upscaling.
    """
    t = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    in_h, in_w = image.shape
    out_h, out_w = target_hw

    if out_h < in_h or out_w < in_w:
        resized = F.interpolate(t, size=target_hw, mode="area")
    else:
        resized = F.interpolate(t, size=target_hw, mode="bicubic", align_corners=False)

    return resized.squeeze(0).squeeze(0).cpu().numpy()


def get_hr_filename_from_lr(lr_filename: str) -> str:
    """
    Get the corresponding HR image filename from the LR image filename.
    """
    # Replace _hr_lr_patch_ with _hr_hr_patch_
    hr_filename = lr_filename.replace("_hr_lr_patch_", "_hr_hr_patch_")
    return hr_filename


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [-1, 1] (matches dataset.py)"""
    min_val = image.min()
    max_val = image.max()

    if max_val - min_val > 0:
        image = 2.0 * (image - min_val) / (max_val - min_val) - 1.0
    else:
        image = image - min_val

    return image


def z_scale_normalize(image: np.ndarray, contrast: float = 0.25) -> np.ndarray:
    """
        Apply Z-scale normalization for astronomical images.
        This technique enhances faint features while preventing bright stars from saturating.

        Args:
            image: Input astronomical image
            contrast: Contrast parameter (default 0.25, lower = more contrast)

        Returns:
            Normalized image suitable for visualization
        """
    # Remove NaN and Inf values
    image_clean = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    interval = ZScaleInterval(contrast=contrast)
    vmin, vmax = interval.get_limits(image_clean)
    norm = ImageNormalize(vmin=vmin, vmax=vmax)
    return norm(image_clean)

def compute_metrics(ground_truth: np.ndarray, prediction: np.ndarray) -> dict:
    """Compute MSE, MAE, PSNR"""
    mse = np.mean((ground_truth - prediction) ** 2)
    mae = np.mean(np.abs(ground_truth - prediction))

    # PSNR (assuming max pixel value is 1.0)
    max_val = 1.0
    psnr = 20 * np.log10(max_val / np.sqrt(mse + 1e-10))

    return {
        "MSE": float(mse),
        "MAE": float(mae),
        "PSNR": float(psnr),
    }


def main():

    dataset = get_dataset(str(DATA_DIR), DATASET_SPLIT, img_size=IMG_SIZE, normalize=False)

    lr_image, hr_image = get_random_sample_images(dataset)

    # Add channel dimension if needed
    # NOTE: what is this?
    if lr_image.ndim == 2:
        lr_image = lr_image[np.newaxis, ...]  # (1, H, W)
    if hr_image.ndim == 2:
        hr_image = hr_image[np.newaxis, ...]

    print(f"\n📊 Raw Data Shapes:")
    print(f"  LR shape: {lr_image.shape}")
    print(f"  HR shape: {hr_image.shape}")

    # First-stage normalization for model/metrics scale.
    lr_normalized = normalize_image(lr_image)
    hr_normalized = normalize_image(hr_image)

    # Second-stage normalization for display contrast enhancement only.
    lr_vis = z_scale_normalize(lr_normalized.squeeze())
    hr_vis = z_scale_normalize(hr_normalized.squeeze())

    print(f"\n📊 Normalized Data Ranges:")
    print(f"  LR range: [{lr_normalized.min():.4f}, {lr_normalized.max():.4f}]")
    print(f"  HR range: [{hr_normalized.min():.4f}, {hr_normalized.max():.4f}]")

    # Create untrained model
    print(f"\n🤖 Creating untrained model (img_size={IMG_SIZE})...")
    model = create_physics_informed_mae(img_size=IMG_SIZE)
    model = model.to(DEVICE)
    model.eval()

    # Convert to tensor and add batch dimension
    lr_tensor = torch.from_numpy(lr_normalized).unsqueeze(0).to(DEVICE)  # (1, 1, H, W)

    print(f"   Input tensor shape: {lr_tensor.shape}")

    # Run inference
    print(f"\n⚙️  Running inference on untrained model...")
    with torch.no_grad():
        model_output, aux_outputs = model(lr_tensor)

    # Convert output to numpy
    model_output_np = model_output.squeeze().cpu().numpy()  # Remove batch & channel

    print(f"   Output shape: {model_output_np.shape}")

    # Compute metrics against downscaled HR target
    hr_for_metrics = hr_normalized.squeeze()
    if hr_for_metrics.shape != model_output_np.shape:
        hr_for_metrics = resize_image_to_shape(hr_for_metrics, model_output_np.shape)

    model_output_vis = z_scale_normalize(model_output_np)

    print(f"\n📈 Comparing Model Output vs Downscaled Ground Truth HR:")
    metrics = compute_metrics(hr_for_metrics, model_output_np)
    print(f"  MSE:  {metrics['MSE']:.6f}")
    print(f"  MAE:  {metrics['MAE']:.6f}")
    print(f"  PSNR: {metrics['PSNR']:.2f} dB")

    # ============ VISUALIZATION ============

    print(f"\n🎨 Creating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # LR input
    axes[0].imshow(lr_vis, cmap="gist_gray")
    axes[0].set_title(
        f"Input (LR)\n{lr_image.shape[1:]}", fontsize=12, fontweight="bold"
    )
    axes[0].axis("off")

    # Untrained model output
    axes[1].imshow(model_output_vis, cmap="gist_gray")
    axes[1].set_title(
        f"Untrained Model Output\n{model_output_np.shape}",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].axis("off")

    # Ground truth HR
    axes[2].imshow(hr_vis, cmap="gist_gray")
    axes[2].set_title(
        f"Ground Truth (HR)\n{hr_image.shape[1:]}", fontsize=12, fontweight="bold"
    )
    axes[2].axis("off")

    plt.tight_layout()
    save_path = "results_untrained_vs_hr.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved visualization to {save_path}")
    plt.show()

    print("\n" + "=" * 70)
    print("🎯 KEY INSIGHT:")
    print("   Untrained model produces RANDOM noise (no training yet).")
    print("   After training 1 epoch, this should improve & match HR better!")
    print("=" * 70)


def get_random_file_paths() -> tuple[Path, Path]:
    lr_dir = DATA_DIR / f"{DATASET_SPLIT}_lr_patch"
    hr_dir = DATA_DIR / f"{DATASET_SPLIT}_hr_patch"

    lr_files = sorted([f for f in lr_dir.iterdir() if f.suffix == ".npy"])

    if len(lr_files) == 0:
        print(f"[WARNING]: No LR files found in {lr_dir}")
        exit(1)

    # Pick random LR file
    random_lr_file_path = random.choice(lr_files)
    random_hr_file_path = hr_dir / get_hr_filename_from_lr(random_lr_file_path.name)

    return random_hr_file_path, random_lr_file_path


def get_random_index() -> int:
    lr_dir = DATA_DIR / f"{DATASET_SPLIT}_lr_patch"
    hr_dir = DATA_DIR / f"{DATASET_SPLIT}_hr_patch"

    lr_files = sorted([f for f in lr_dir.iterdir() if f.suffix == ".npy"])

    if len(lr_files) == 0:
        print(f"[WARNING]: No LR files found in {lr_dir}")
        exit(1)

    # pick a random LR file index
    random_index = random.randint(0, len(lr_files) - 1)

    return random_index


if __name__ == "__main__":
    main()
