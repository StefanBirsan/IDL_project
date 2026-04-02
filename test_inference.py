"""Evaluate checkpoint inference and optionally compare with ONNX output."""

import argparse
from pathlib import Path
from typing import Literal, cast

import torch

from test_untrained import get_random_sample_images
from training.inference import Inference, ONNXInference
from training.train_utils import create_physics_informed_mae
from utils import NumpyAstronomicalDataset, get_dataset, visualize_result

DATA_DIR = Path("dataset/data/x2")
DATASET_SPLIT = "eval"
IMG_SIZE = 128


def define_args():
    parser = argparse.ArgumentParser(
        description="Visualize inference results from checkpoint and/or ONNX model"
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to .pt checkpoint",
    )

    parser.add_argument(
        "--onnx",
        type=Path,
        help="Path to ONNX model.",
    )

    parser.add_argument(
        "--input-image",
        type=Path,
        required=False,
        help="Optional path to LR .npy patch to apply inference on. Random sample selected if not provided.",
    )

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
        default=None,
        help="Path to save the matplotlib visualization",
    )

    parser.add_argument(
        "--show-plot",
        action="store_true",
        default=False,
        help="Show matplotlib window in addition to saving figure",
    )

    parser.add_argument(
        "--show-ground-truth",
        action="store_true",
        default=False,
        help="Show ground truth image for comparison",
    )

    return parser


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
    inference_engine = Inference(
        model=model, checkpoint_path=str(checkpoint_path), device=device
    )
    return inference_engine


def main():

    parser = define_args()
    args = parser.parse_args()
    device = cast(Literal["cpu", "cuda"], args.device)

    # === ARGUMENT VALIDATION ===
    # check if either checkpoint or ONNX is provided
    if not args.checkpoint and not args.onnx:
        raise ValueError(
            "One of --checkpoint or --onnx must be provided for visualization."
        )
    # both can't be provided
    if args.checkpoint and args.onnx:
        raise ValueError(
            "Only one of --checkpoint or --onnx can be provided for visualization."
        )

    # check if checkpoint or ONNX provided
    #  and create inference engines
    # TODO: checkpoint inference engines and onnx inference engine
    #  can inherit a base class and inference_engine can be a
    #  single instance to avoid conditional logic later on
    checkpoint_inference_engine = None
    onnx_inference_engine = None
    if args.checkpoint:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        checkpoint_inference_engine = get_inference_engine(
            args.checkpoint, device=device
        )
    if args.onnx:
        if not args.onnx.exists():
            raise FileNotFoundError(f"ONNX model not found: {args.onnx}")
        # TODO: ONNX inference needs testing
        onnx_inference_engine = ONNXInference(str(args.onnx))

    # check if input image exists, if provided
    dataset = get_dataset(
        data_dir=str(DATA_DIR), split=DATASET_SPLIT, img_size=IMG_SIZE
    )
    input_image = None
    ground_truth_image = None
    if not args.input_image:
        print("No LR input image provided, using random sample from dataset")
        input_image, ground_truth_image = get_random_sample_images(dataset)
        # keep ground_truth_image None if not specified by user
        if not args.show_ground_truth:
            ground_truth_image = None
    else:
        # first check if input image file exists
        if not args.input_image.exists():
            raise FileNotFoundError(f"Input image not found: {args.input_image}")

        # load npy array from path
        input_image = NumpyAstronomicalDataset.load_image_from_npy(args.input_image)

        # if user specified ground truth image,
        # load the correct one, the HR counterpart of the LR input
        if args.show_ground_truth:
            input_image_name = args.input_image.name
            ground_truth_image_name = NumpyAstronomicalDataset.get_hr_filename_from_lr(
                input_image_name
            )
            ground_truth_image_path = dataset.hr_dir / ground_truth_image_name
            ground_truth_image = NumpyAstronomicalDataset.load_image_from_npy(
                ground_truth_image_path
            )

    # check plot arguments
    if not args.show_plot and not args.save_figure:
        print(
            "Warning: neither --show-plot nor --save-figure specified, visualization will not be visible."
        )
        # NOTE: i kept this case and did not treat it as an error as this method
        #  can be used in automated scripts where saving or showing may not be needed

    # === INFERENCE ===

    output_image = None
    if checkpoint_inference_engine:
        output_image = checkpoint_inference_engine.infer(input_image)
    if onnx_inference_engine:
        output_image = onnx_inference_engine.infer(input_image)

    # === RESULT VISUALIZATION ===
    visualize_result(
        input_image=input_image,
        output_image=output_image,
        ground_truth_image=ground_truth_image,
        save_path=args.save_figure,
        show_plot=args.show_plot,
    )


if __name__ == "__main__":
    main()
