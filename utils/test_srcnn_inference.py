"""Run SRCNN checkpoint inference and visualize LR/SR/HR results.

This utility mirrors utils/test_inference.py but targets the SRCNN pipeline.
It can use:
- a specific input image via --input-image
- a random sample from a local dataset folder via --data-dir

Expected local dataset layout examples:
1) data_dir/*.jpg|png
2) data_dir/train/*.jpg|png and data_dir/val/*.jpg|png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import Tensor

from training.inference.srcnn.srcnn_inference import SRCNNInference
from training.train_utils.srcnn.face_sr_dataset import FaceSuperResolutionDataset
from utils import visualize_result


def define_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize SRCNN inference from a trained checkpoint",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to SRCNN checkpoint (.pth)",
    )

    parser.add_argument(
        "--input-image",
        type=Path,
        default=None,
        help="Path to an input face image (.jpg/.png/.jpeg)",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Optional dataset root. If --input-image is missing, a random image is "
            "sampled from this folder (or from data-dir/<split>)."
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "eval"],
        help="Split folder to use under --data-dir if it exists",
    )

    parser.add_argument(
        "--scale-factor",
        type=int,
        default=2,
        choices=[2, 4, 8],
        help="Downsample/upsample factor used to build LR input",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        choices=["cpu", "cuda"],
        help="Device for inference",
    )

    parser.add_argument(
        "--save-figure",
        type=Path,
        default=None,
        help="Optional path to save the visualization",
    )

    parser.add_argument(
        "--show-plot",
        action="store_true",
        default=False,
        help="Display matplotlib window",
    )

    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="marcosv/ffhq-dataset",
        help="Hugging Face dataset id used when neither --input-image nor --data-dir is provided",
    )

    parser.add_argument(
        "--stream-sample-limit",
        type=int,
        default=1024,
        help="Number of streamed samples to inspect for random reservoir sampling",
    )

    return parser


def _normalize_image(image: Tensor) -> np.ndarray:
    """Convert a tensor image to a normalized RGB numpy array in [0, 1]."""
    return np.asarray(image, dtype=np.float32) / 255.0

def _load_rgb_image(image: np.ndarray) -> np.ndarray:
    """Load an image as float32 HWC normalized to [0, 1]."""
    return np.asarray(
        Image.fromarray(image.astype(np.uint8), mode="RGB"), dtype=np.float32
    ) / 255.0


def _build_lr_upscaled(hr_image: np.ndarray, scale_factor: int) -> np.ndarray:
    """Create SRCNN input: bicubic downsample then bicubic upsample."""
    h, w = hr_image.shape[:2]
    h_aligned = (h // scale_factor) * scale_factor
    w_aligned = (w // scale_factor) * scale_factor

    if h_aligned <= 0 or w_aligned <= 0:
        raise ValueError(
            "Input image is smaller than scale factor. "
            f"Image shape={hr_image.shape}, scale_factor={scale_factor}"
        )

    hr_cropped = hr_image[:h_aligned, :w_aligned, :]
    hr_uint8 = (hr_cropped * 255.0).astype(np.uint8)
    bicubic = Image.Resampling.BICUBIC

    hr_pil = Image.fromarray(hr_uint8)
    lr_pil = hr_pil.resize(
        (w_aligned // scale_factor, h_aligned // scale_factor), bicubic
    )
    lr_upscaled_pil = lr_pil.resize((w_aligned, h_aligned), bicubic)

    return np.asarray(lr_upscaled_pil, dtype=np.float32) / 255.0


def _get_search_dir(data_dir: Path, split: str) -> Path:
    split_dir = data_dir / split
    return split_dir if split_dir.exists() else data_dir


def _get_random_tensor_pairs(data_dir: Path):
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    dataset = FaceSuperResolutionDataset(data_dir=str(data_dir))

    sample = dataset.get_random_sample()

    lr_tensor = sample["lr_image"]
    hr_tensor = sample["hr_image"]

    return lr_tensor, hr_tensor

def _sample_random_image_stream(
    split: str
) -> tuple[str, np.ndarray]:
    """Reservoir-sample one image from a streamed HF dataset split."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required for streaming mode. Install it with: uv add datasets"
        ) from exc

    dataset_id = "marcosv/ffhq-dataset"
    sample_limit = 162000

    ds = load_dataset(
        path=dataset_id,
        cache_dir="dataset/.cache",
        split=split,
        streaming=True,
    )

    print("Done loading!")

    shuffled_dataset = ds.shuffle(seed=42)

    selected_sample = next(iter(shuffled_dataset))

    selected_name, selected_image = _extract_hf_image(selected_sample)

    if selected_image is None or selected_name is None:
        raise RuntimeError(
            f"Could not read images from streaming dataset '{dataset_id}' split '{split}'"
        )

    return selected_name, selected_image


def _extract_hf_image(sample: dict[str, Any]) -> tuple[str, np.ndarray]:
    """Extract a RGB numpy image from a HF sample with common schemas."""
    if "image" in sample:
        image_obj = sample["image"]
    else:
        raise KeyError(
            "Streaming sample does not contain an 'image' field. "
            f"Available keys: {list(sample.keys())}"
        )

    # Most image datasets decode to PIL.Image.Image objects.
    if isinstance(image_obj, Image.Image):
        print("Image.Image")
        image = np.asarray(image_obj.convert("RGB"), dtype=np.float32) / 255.0
    elif isinstance(image_obj, np.ndarray):
        print("np.ndarray")
        image = image_obj.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
    else:
        raise TypeError(
            "Unsupported image payload type in streaming sample: "
            f"{type(image_obj).__name__}"
        )

    name = "stream_random_sample"
    return name, image

def _compute_metrics(reference_hr: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    """Compute PSNR and SSIM on RGB images in [0, 1]."""
    # data_range is the difference between the max and min possible pixel values
    drange = reference_hr.max() - reference_hr.min()

    psnr = peak_signal_noise_ratio(reference_hr, prediction, data_range=drange)

    # set channel_axis=2 since shape is H,W,C
    ssim = structural_similarity(
        reference_hr,
        prediction,
        channel_axis=2,
        data_range=drange,
    )

    return float(psnr), float(ssim)


def main() -> None:
    parser = define_args()
    args = parser.parse_args()

    # Turn on cuDNN optimization and check for best algorithms
    cudnn.benchmark = True

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this machine")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # if args.input_image is None and args.data_dir is None:
    #     raise ValueError("Provide either --input-image or --data-dir")

    if args.input_image is not None and not args.input_image.exists():
        raise FileNotFoundError(f"Input image not found: {args.input_image}")

    selected_image = args.input_image
    lr_tensor, hr_tensor = None, None

    if selected_image is None:
        # select a random tensor pair from the dataset
        if args.data_dir is not None:
            lr_tensor, hr_tensor = _get_random_tensor_pairs(args.data_dir, args.scale_factor)
    else:
        # If an input image is provided, load it and build the LR/HR pair
        #  then convert to tensors
        pass

    # Convert obtained tensors to numpy arrays for inference and visualization
    hr_image = hr_tensor.permute(1, 2, 0).cpu().numpy()
    lr_image = lr_tensor.permute(1, 2, 0).cpu().numpy()

    inference_engine = SRCNNInference(str(args.checkpoint), device=device)
    output_numpy = inference_engine.infer(lr_image)

    # Crop HR image to match output size
    hr_aligned = hr_image[: output_numpy.shape[0], : output_numpy.shape[1], :]

    psnr_bicubic, ssim_bicubic = _compute_metrics(hr_aligned, lr_image)
    psnr_srcnn, ssim_srcnn = _compute_metrics(hr_aligned, output_numpy)

    print("Comparing basic bicubic upscaling to SRCNN:")
    print("PSNR: The higher the better")
    print("SSIM: The closer to 1.0 the better")
    print("BICUBIC (basic resizing in image editing software)")
    print(f"PSNR: {psnr_bicubic:.2f} dB")
    print(f"SSIM: {ssim_bicubic:.4f}")
    print("SRCNN")
    print(f"PSNR: {psnr_srcnn:.2f} dB")
    print(f"SSIM: {ssim_srcnn:.4f}")

    import matplotlib.pyplot as plt
    residual = np.abs(output_numpy - hr_aligned).mean(axis=2)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(residual, cmap='hot')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pixel Intensity Difference', rotation=270, labelpad=15)

    ax.set_title('SRCNN Residual Heatmap')
    plt.show()

    visualize_result(
        input_image=lr_image,
        output_image=output_numpy,
        ground_truth_image=hr_aligned,
        save_path=args.save_figure,
        show_plot=args.show_plot,
    )


if __name__ == "__main__":
    main()
