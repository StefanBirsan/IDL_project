"""Evaluate checkpoint inference and optionally compare with ONNX output."""

import argparse
from pathlib import Path
from typing import Dict, Tuple, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from test_untrained import get_random_sample_images, load_image_from_npy
from training.train_utils import create_physics_informed_mae
from training.inference import Inference, ONNXInference
from utils import NumpyAstronomicalDataset, visualize_result
DATA_DIR = Path("dataset/data/x2")
DATASET_SPLIT = "eval"
IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_npy_image(path: Path) -> np.ndarray:
    """Load image payload from .npy file that may contain dict or raw ndarray."""
    payload = np.load(path, allow_pickle=True)

    if isinstance(payload, np.ndarray) and payload.dtype == object:
        payload = payload.item()

    if isinstance(payload, dict):
        if "image" not in payload:
            raise KeyError(f"Missing 'image' key in {path}")
        image = payload["image"]
    else:
        image = payload

    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)

    if image.ndim != 3:
        raise ValueError(f"Expected image with shape (C,H,W), got {image.shape}")

    return image


def minmax_normalize(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] while handling NaN/Inf safely."""
    arr = np.asarray(image, dtype=np.float32)
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return np.zeros_like(arr, dtype=np.float32)

    finite_vals = arr[finite_mask]
    min_val = float(finite_vals.min())
    max_val = float(finite_vals.max())
    if max_val <= min_val:
        return np.zeros_like(arr, dtype=np.float32)

    arr = np.nan_to_num(arr, nan=min_val, posinf=max_val, neginf=min_val)
    return ((arr - min_val) / (max_val - min_val + 1e-8)).astype(np.float32)


def get_inference_engine(checkpoint_path: Path, device: Literal["cpu", "cuda"]):
    """Create inference engine from PyTorch checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # get config from checkpoint
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    # create model object
    model = create_physics_informed_mae(
        img_size=int(cfg.get("img_size", 128)),
        patch_size=int(cfg.get("patch_size", 4)),
        embed_dim=int(cfg.get("embed_dim", 768)),
        encoder_depth=int(cfg.get("encoder_depth", 12)),
        decoder_depth=int(cfg.get("decoder_depth", 8)),
        num_heads=int(cfg.get("num_heads", 12)),
        mask_ratio=float(cfg.get("mask_ratio", 0.75)),
    )

    # initialize inference engine to load state dict into model
    inference_engine = Inference(model=model, checkpoint_path=str(checkpoint_path), device=device)
    return inference_engine

    # state_dict = checkpoint.get("model_state_dict", checkpoint)
    # model.load_state_dict(state_dict)
    # model.to(device)
    # model.eval()
    # return model


def compute_metrics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    """Compute MSE, MAE, PSNR."""
    target = np.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)
    prediction = np.nan_to_num(prediction, nan=0.0, posinf=1.0, neginf=0.0)
    mse = float(np.mean((target - prediction) ** 2))
    mae = float(np.mean(np.abs(target - prediction)))
    psnr = float(20.0 * np.log10(1.0 / np.sqrt(mse + 1e-10)))
    return {"MSE": mse, "MAE": mae, "PSNR": psnr}


def resize_to_shape(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize single-channel image (C,H,W) to requested spatial shape."""
    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError(f"Expected image shape (1,H,W), got {image.shape}")

    _, in_h, in_w = image.shape
    out_h, out_w = target_hw
    tensor = torch.from_numpy(image).unsqueeze(0).float()

    if out_h < in_h or out_w < in_w:
        resized = F.interpolate(tensor, size=target_hw, mode="area")
    else:
        resized = F.interpolate(tensor, size=target_hw, mode="bicubic", align_corners=False)

    return resized.squeeze(0).cpu().numpy().astype(np.float32)


def run_checkpoint_inference(
    checkpoint_path: Path,
    input_image: np.ndarray,
    ground_truth_image: np.ndarray | None,
    device: Literal["cpu", "cuda"],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float] | None]:
    """
    Run PyTorch checkpoint inference and optional target metrics.
    """
    dataset = NumpyAstronomicalDataset(data_dir=str(DATA_DIR), split=DATASET_SPLIT, img_size=IMG_SIZE)
    inference_engine = get_inference_engine(checkpoint_path, device=device)

    input_image = minmax_normalize(input_image)

    # convert numpy arrays to tensors
    # add batch dimension and convert to float tensor on device: (1, C, H, W)
    input_tensor = torch.from_numpy(input_image).unsqueeze(0).float().to(device)

    output_tensor = inference_engine.infer(input_tensor)

    # convert output tensor to numpy array and remove batch dimension: (C, H, W)
    output_image = output_tensor.squeeze().cpu().numpy()

    # alt_output = None
    # with torch.no_grad():
    #     alt_output = model(lr_tensor)[0]
    # alt_output = alt_output.squeeze().cpu().numpy()

    metrics = None

    # NOTE: ground truth image is used only for metrics computation
    if ground_truth_image is not None:
        # normalize ground truth image
        ground_truth_image = minmax_normalize(ground_truth_image)

        if ground_truth_image.shape != output_image.shape:
            # match spatial shape of output for metrics computation
            ground_truth_image = resize_to_shape(ground_truth_image, output_image.shape[-2:])
        # metrics = compute_metrics(target, output)

    return input_image, output_image, metrics


def run_onnx_inference(
    onnx_path: Path,
    input_image: np.ndarray,
    target_image_path: Path | None,
) -> Tuple[np.ndarray, Dict[str, float] | None]:
    """Run ONNX inference and optional target metrics."""
    onnx_engine = ONNXInference(str(onnx_path))
    onnx_output = onnx_engine.infer(input_image).squeeze(0)

    metrics = None
    if target_image_path is not None:
        target = minmax_normalize(load_npy_image(target_image_path))
        if target.shape != onnx_output.shape:
            target = resize_to_shape(target, onnx_output.shape[-2:])
        metrics = onnx_engine.compute_metrics(target, onnx_output)

    return onnx_output, metrics


def visualize_results(
    input_image: np.ndarray,
    target_image: np.ndarray | None,
    ckpt_output: np.ndarray,
    onnx_output: np.ndarray | None,
    save_path: Path,
    show_plot: bool = True,
) -> None:
    """Visualize LR input, HR ground truth, and model outputs."""
    def to_display_image(image: np.ndarray) -> np.ndarray:
        """Apply robust contrast stretching for visualization."""

        return image

        arr = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros_like(arr, dtype=np.float32)

        p_low, p_high = np.percentile(finite, [0.5, 99.5])
        if p_high <= p_low:
            p_low = float(np.min(finite))
            p_high = float(np.max(finite))

        if p_high <= p_low:
            return np.zeros_like(arr, dtype=np.float32)

        arr = np.clip(arr, p_low, p_high)
        arr = (arr - p_low) / (p_high - p_low + 1e-8)
        return arr

    lr_vis = to_display_image(input_image.squeeze())

    target_vis = None
    if target_image is not None:
        target = minmax_normalize(target_image)
        target_vis = to_display_image(target.squeeze())

    ckpt_vis = to_display_image(ckpt_output.squeeze())
    onnx_vis = None
    if onnx_output is not None:
        onnx_vis = to_display_image(onnx_output.squeeze())

    if onnx_vis is not None:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        plot_items = [
            (lr_vis, "Input (LR)"),
            (target_vis if target_vis is not None else lr_vis, "Ground Truth (HR)" if target_vis is not None else "Ground Truth (HR) unavailable"),
            (ckpt_vis, "Checkpoint Output"),
            (onnx_vis, "ONNX Output"),
        ]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        plot_items = [
            (lr_vis, "Input (LR)"),
            (target_vis if target_vis is not None else lr_vis, "Ground Truth (HR)" if target_vis is not None else "Ground Truth (HR) unavailable"),
            (ckpt_vis, "Checkpoint Output"),
        ]

    for ax, (image, title) in zip(axes, plot_items):
        ax.imshow(image, cmap="gist_gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    # save_path.parent.mkdir(parents=True, exist_ok=True)
    # plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # print(f"Saved visualization to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Test inference with a checkpoint and optional ONNX model"
    )

    # which checkpoint file to use
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to .pt checkpoint",
    )

    # which LR input image to use
    parser.add_argument(
        "--input-image",
        type=Path,
        required=False,
        help="Optional path to LR .npy patch to apply inference on",
    )

    # TODO: automatically find the HR image based on LR
    parser.add_argument(
        "--target-image",
        type=Path,
        default=None,
        help="Optional path to HR .npy patch for metrics",
    )

    # path to ONNX model for comparison
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("models/physics_informed_mae.onnx"),
        help="Path to ONNX model for optional comparison",
    )

    # whether to run ONNX inference and compare with PyTorch output
    parser.add_argument(
        "--run-onnx",
        action="store_true",
        help="Also run ONNX inference and compare with PyTorch output",
    )

    # cuda or cpu for inference
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        choices=["cpu", "cuda"],
        help="Device for PyTorch checkpoint inference",
    )
    parser.add_argument(
        "--save-figure",
        type=Path,
        default=Path("results/checkpoint_inference_visualization.png"),
        help="Path to save the matplotlib visualization",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Show matplotlib window in addition to saving figure",
    )
    args = parser.parse_args()
    device = cast(Literal["cpu", "cuda"], args.device)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    input_image = args.input_image
    target_image = args.target_image

    if not input_image:
        print("No LR input image provided, using random sample from dataset")
        # get a random sample from the dataset
        dataset = NumpyAstronomicalDataset(data_dir=str(DATA_DIR), split=DATASET_SPLIT, img_size=IMG_SIZE)
        input_image, target_image = get_random_sample_images(dataset)


    input_image, ckpt_output, ckpt_metrics = run_checkpoint_inference(
        checkpoint_path=args.checkpoint,
        input_image=input_image,
        ground_truth_image=target_image,
        device=device,
    )

    print(f"  Output shape: {ckpt_output.shape}")
    if not np.isfinite(ckpt_output).all():
        print("  Warning: checkpoint output contains NaN/Inf values (sanitized for metrics).")
    if ckpt_metrics is not None:
        print("  Checkpoint metrics (vs target):")
        print(f"    MSE:  {ckpt_metrics['MSE']:.6f}")
        print(f"    MAE:  {ckpt_metrics['MAE']:.6f}")
        print(f"    PSNR: {ckpt_metrics['PSNR']:.2f} dB")

    if args.run_onnx:
        if not args.onnx.exists():
            raise FileNotFoundError(f"ONNX model not found: {args.onnx}")

        print("Running ONNX inference")
        print(f"  ONNX model: {args.onnx}")

        onnx_output, onnx_metrics = run_onnx_inference(
            onnx_path=args.onnx,
            input_image=input_image,
            target_image_path=args.target_image,
        )

        if not np.isfinite(onnx_output).all():
            print("  Warning: ONNX output contains NaN/Inf values (sanitized for metrics).")

        delta = np.abs(
            np.nan_to_num(ckpt_output, nan=0.0, posinf=1.0, neginf=0.0)
            - np.nan_to_num(onnx_output, nan=0.0, posinf=1.0, neginf=0.0)
        )
        print("  ONNX comparison (vs checkpoint output):")
        print(f"    Mean abs diff: {delta.mean():.8f}")
        print(f"    Max abs diff:  {delta.max():.8f}")

        if onnx_metrics is not None:
            print("  ONNX metrics (vs target):")
            print(f"    MSE:  {onnx_metrics['MSE']:.6f}")
            print(f"    MAE:  {onnx_metrics['MAE']:.6f}")
            print(f"    PSNR: {onnx_metrics['PSNR']:.2f} dB")

        visualize_results(
            input_image=input_image,
            target_image=target_image,
            ckpt_output=ckpt_output,
            onnx_output=onnx_output,
            save_path=args.save_figure,
            show_plot=args.show_plot,
        )
    else:
        visualize_result(
            input_image=input_image,
            output_image=ckpt_output,
            ground_truth_image=target_image,
            save_path=args.save_figure,
        )

def simple():

    dataset = NumpyAstronomicalDataset(data_dir=str(DATA_DIR), split=DATASET_SPLIT, img_size=IMG_SIZE)
    input_image, target_image = get_random_sample_images(dataset)

    inference_engine = get_inference_engine(
        Path('checkpoints/checkpoint_epoch_0005.pt'),
        'cpu'
    )

    inference_engine.visualize_inference(input_image)


if __name__ == "__main__":
    main()
