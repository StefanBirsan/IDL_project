"""
Model export functionality (ONNX, TorchScript, etc.)
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


class ModelExporter:
    """Exports trained models to various formats"""
    
    @staticmethod
    def export_to_onnx(model: nn.Module,
                      output_dir: str = 'models',
                      img_size: int = 64,
                      device: str = 'cuda') -> bool:
        """
        Export model to ONNX format
        
        Args:
            model: Model to export
            output_dir: Directory to save ONNX model
            img_size: Input image size
            device: Device model is on
        Returns:
            True if export successful, False otherwise
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create dummy input
        dummy_input = torch.randn(
            1,
            1,
            img_size,
            img_size,
            device=device
        )
        
        # Set model to eval mode
        model.eval()
        
        # Export to ONNX
        onnx_path = output_path / 'physics_informed_mae.onnx'
        
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                input_names=['image'],
                output_names=['reconstructed'],
                dynamic_axes={
                    'image': {0: 'batch_size'},
                    'reconstructed': {0: 'batch_size'}
                },
                opset_version=17,
                verbose=False,
                do_constant_folding=True
            )
            print(f"\n{'='*60}")
            print(f"ONNX model exported successfully!")
            print(f"Path: {onnx_path}")
            print(f"Format: ONNX (opset 17)")
            print(f"Input: image (batch_size, 1, {img_size}, {img_size})")
            print(f"Output: reconstructed (batch_size, 1, {img_size}, {img_size})")
            print(f"{'='*60}\n")
            return True
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"Error exporting to ONNX:")
            print(f"  {str(e)}")
            print(f"Make sure opset version 17 or higher is supported.")
            print(f"{'='*60}\n")
            return False
    
    @staticmethod
    def export_to_torchscript(model: nn.Module,
                             output_dir: str = 'models',
                             img_size: int = 64,
                             device: str = 'cuda') -> bool:
        """
        Export model to TorchScript format
        
        Args:
            model: Model to export
            output_dir: Directory to save TorchScript model
            img_size: Input image size
            device: Device model is on
        Returns:
            True if export successful, False otherwise
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create dummy input
        dummy_input = torch.randn(
            1,
            1,
            img_size,
            img_size,
            device=device
        )
        
        # Set model to eval mode
        model.eval()
        
        # Export to TorchScript
        torchscript_path = output_path / 'physics_informed_mae.pt'
        
        try:
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(str(torchscript_path))
            print(f"\n{'='*60}")
            print(f"TorchScript model exported successfully!")
            print(f"Path: {torchscript_path}")
            print(f"{'='*60}\n")
            return True
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"Error exporting to TorchScript:")
            print(f"  {str(e)}")
            print(f"{'='*60}\n")
            return False
