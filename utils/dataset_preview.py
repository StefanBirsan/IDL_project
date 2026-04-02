import random
from pathlib import Path
import numpy as np
import os
from astropy.visualization import ZScaleInterval, ImageNormalize
import matplotlib.pyplot as plt


def z_scale_normalize(image, contrast=0.9):
    """
    Code from https://huggingface.co/datasets/KUOCHENG/STAR
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


def visualize_sample(hr_path, lr_path):
    # Load data
    hr_data = np.load(hr_path, allow_pickle=True).item()
    lr_data = np.load(lr_path, allow_pickle=True).item()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    hr_image_vis = z_scale_normalize(hr_data["image"])
    lr_image_vis = z_scale_normalize(lr_data["image"])

    # HR visualizations
    axes[0, 0].imshow(hr_image_vis, cmap="gray")
    axes[0, 0].set_title("HR Image (256x256)")

    axes[0, 1].imshow(hr_data["mask"], cmap="binary")
    axes[0, 1].set_title("HR Mask (Valid Regions)")

    axes[0, 2].imshow(hr_data["attn_map"], cmap="hot")
    axes[0, 2].set_title("HR Attention Map (Detected Sources)")

    # LR visualizations
    axes[1, 0].imshow(lr_image_vis, cmap="gray")
    axes[1, 0].set_title(
        f"LR Image ({lr_data['image'].shape[0]}x{lr_data['image'].shape[1]})"
    )

    axes[1, 1].imshow(lr_data["mask"], cmap="binary")
    axes[1, 1].set_title("LR Mask")

    axes[1, 2].imshow(lr_data["attn_map"], cmap="hot")
    axes[1, 2].set_title("LR Attention Map")

    plt.tight_layout()
    plt.show()


"""
Chooses a random low-resolution and high-resolution pair from the LOCALLY-DOWNLOADED (!) STAR dataset 
and displays them both
"""
if __name__ == "__main__":
    LR_PATH = Path("dataset/data/x2/eval_lr_patch")
    HR_PATH = Path("dataset/data/x2/eval_hr_patch")

    visualize_sample(
        hr_path=HR_PATH / random.choice(os.listdir(HR_PATH)),
        lr_path=LR_PATH / random.choice(os.listdir(LR_PATH)),
    )
