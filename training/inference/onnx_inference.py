"""ONNX inference utilities for Physics-Informed MAE."""

from pathlib import Path
from typing import Dict, Optional, Tuple
import importlib

import numpy as np


class ONNXInference:
    """Inference engine backed by ONNX Runtime."""

    def __init__(
        self,
        onnx_model_path: str,
        providers: Optional[list[str]] = None,
    ):
        try:
            ort = importlib.import_module("onnxruntime")
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ONNX inference. "
                "Install with: pip install onnxruntime"
            ) from exc

        self._ort = ort
        model_path = Path(onnx_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")

        if providers is None:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Run ONNX inference.

        Args:
            image: Input image in shape (C, H, W) or (B, C, H, W)
        Returns:
            Model output in shape (B, C, H, W)
        """
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        if image.ndim != 4:
            raise ValueError(
                f"Expected input shape (C,H,W) or (B,C,H,W), got {image.shape}"
            )

        image = image.astype(np.float32, copy=False)
        output = self.session.run([self.output_name], {self.input_name: image})[0]
        return output

    @staticmethod
    def compute_metrics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
        """Compute MSE, MAE, and PSNR for model output."""
        target = np.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)
        prediction = np.nan_to_num(prediction, nan=0.0, posinf=1.0, neginf=0.0)
        mse = float(np.mean((target - prediction) ** 2))
        mae = float(np.mean(np.abs(target - prediction)))
        psnr = float(20.0 * np.log10(1.0 / np.sqrt(mse + 1e-10)))
        return {"MSE": mse, "MAE": mae, "PSNR": psnr}

    def infer_and_compare(
        self,
        image: np.ndarray,
        target: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
        """Run inference and optionally compute metrics against target."""
        output = self.infer(image)
        metrics = None
        if target is not None:
            if target.ndim == 3:
                target = np.expand_dims(target, axis=0)
            metrics = self.compute_metrics(target.astype(np.float32), output)
        return output, metrics
