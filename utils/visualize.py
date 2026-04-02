"""
Methods to visualize inference results
for demonstration purposes
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

from astropy.visualization import ZScaleInterval, ImageNormalize

def normalize_STAR(image: np.ndarray, contrast: float = 0.25) -> np.ndarray:
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

def visualize_result(
    input_image: np.ndarray,
    output_image: np.ndarray,
    ground_truth_image: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    show_plot: Optional[bool] = False,
):
    """
    Visualize input, output, and optionally ground truth images side by side.

    Args:
        input_image: Input image (C, H, W) or (H, W, C)
        output_image: Model output image (C, H, W) or (H, W, C)
        ground_truth_image: Optional ground truth image (C, H, W) or (H, W, C)
        save_path: Optional path to save the visualization
        show_plot: Whether to show the plot
    """

    def to_display(image: np.ndarray) -> np.ndarray:
        """Convert CHW/HW/HWC arrays to a shape accepted by matplotlib."""
        arr = np.asarray(image)

        if arr.ndim == 2:
            return arr

        if arr.ndim == 3:
            # Channel-first: (C, H, W)
            if arr.shape[0] in (1, 3):
                if arr.shape[0] == 1:
                    return arr[0]
                return np.transpose(arr, (1, 2, 0))

            # Channel-last: (H, W, C)
            if arr.shape[-1] in (1, 3):
                if arr.shape[-1] == 1:
                    return np.squeeze(arr, axis=-1)
                return arr

        if arr.ndim == 4:
            # Batch of images: (B, C, H, W)
            return np.squeeze(arr)

        raise ValueError(f"Unsupported image shape for visualization: {arr.shape}")

    input_image = to_display(input_image)
    output_image = to_display(output_image)

    if ground_truth_image is not None:
        ground_truth_image = to_display(ground_truth_image)

    # Create figure
    num_images = 2 + (ground_truth_image is not None)
    fig, axes = plt.subplots(1, num_images, figsize=(10, 5))

    plot_items = [
        (input_image, "Input Image"),
        (output_image, "Model Output"),
        (ground_truth_image, "Ground Truth")
        if ground_truth_image is not None
        else None,
    ]

    plot_items = [item for item in plot_items if item is not None]

    """
    NOTE: Some results are better visualized with different colormaps:
    - `gray_r` displays the details in black over a white background
    thus fainter details are more visible.
    - `Greys_r`, is similar to `gray`, but the colormap is sequential
    and more friendly to the human eye.
    - `Greys` is similar to `gray_r`, with the same difference stated above.
    """

    for ax, (image, title) in zip(axes, plot_items):
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()

    # Save and show the figure
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")

    if not show_plot:
        plt.close(fig)
        return

    plt.show()
