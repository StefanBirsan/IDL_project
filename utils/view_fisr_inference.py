"""Visualize FISR checkpoint inference on dataset samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from training.train_utils import create_fisr_x2, create_fisr_x4
from utils import get_dataset, visualize_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View FISR inference results")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data-dir", type=str, default="dataset/data/x2")
    parser.add_argument("--split", type=str, default="eval", choices=["train", "eval"])
    parser.add_argument("--variant", type=str, default=None, choices=["x2", "x4", None])
    parser.add_argument("--index", type=int, default=None, help="Sample index. Random if omitted.")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--save-figure", type=str, default=None)
    parser.add_argument("--show-plot", action="store_true", default=False)
    return parser.parse_args()


def _to_star_flux_map(flux_map: torch.Tensor) -> torch.Tensor:
    if flux_map.ndim == 4 and flux_map.shape[1] == 1:
        return flux_map.squeeze(1)
    return flux_map


def _build_model_from_checkpoint(checkpoint: dict, device: torch.device, variant_override: str | None):
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    variant = variant_override or cfg.get("variant", "x2")

    kwargs = {
        "inp_channels": int(cfg.get("inp_channels", 1)),
        "out_channels": int(cfg.get("out_channels", 1)),
        "dim": int(cfg.get("dim", 48)),
        "num_blocks": list(cfg.get("num_blocks", (4, 6, 6, 8))),
        "num_refinement_blocks": int(cfg.get("num_refinement_blocks", 4)),
        "heads": list(cfg.get("heads", (1, 2, 4, 8))),
        "ffn_expansion_factor": float(cfg.get("ffn_expansion_factor", 2.66)),
        "bias": bool(cfg.get("bias", True)),
        "layer_norm_type": str(cfg.get("layer_norm_type", "WithBias")),
        "decoder": bool(cfg.get("decoder", True)),
        "use_loss": str(cfg.get("use_loss", "L1")),
        "use_attention": bool(cfg.get("use_attention", False)),
    }

    if variant == "x4":
        model = create_fisr_x4(**kwargs)
    else:
        model = create_fisr_x2(**kwargs)

    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def _prepare_for_plot(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model = _build_model_from_checkpoint(checkpoint, device, args.variant)

    dataset = get_dataset(args.data_dir, split=args.split)
    sample_idx = args.index if args.index is not None else np.random.randint(0, len(dataset))
    sample = dataset[sample_idx]

    lr = sample["lr_image"].unsqueeze(0).to(device)
    hr = sample["hr_image"].unsqueeze(0)
    lr_mask = sample["lr_mask"].unsqueeze(0).to(device)

    targets = {"flux_map": _to_star_flux_map(lr_mask)}

    with torch.no_grad():
        output = model(lr, targets)
        pred = output["pred_img"].cpu()

    input_np = _prepare_for_plot(lr.cpu().numpy())
    pred_np = _prepare_for_plot(pred.numpy())
    hr_np = _prepare_for_plot(hr.numpy())

    visualize_result(
        input_image=input_np,
        output_image=pred_np,
        ground_truth_image=hr_np,
        save_path=args.save_figure,
        show_plot=args.show_plot,
    )

    print(f"Visualized sample index: {sample_idx}")


if __name__ == "__main__":
    main()
