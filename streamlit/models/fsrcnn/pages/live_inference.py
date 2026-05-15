import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import sys
from dataclasses import dataclass
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

@dataclass
class FSRCNNConfig:
    """Configuration for FSRCNN (Fast Super-Resolution CNN)"""

    # Model Parameters
    name: str = "FSRCNN"
    description: str = "Fast Super-Resolution CNN"

    # Architecture
    scale_factor: int = 4
    num_channels = 3

    # Inference
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    project_root = Path(os.getcwd())
    onnx_file = r"fsrcnn_best.onnx"
    onnx_file_path = project_root / "models" / onnx_file
    print(onnx_file_path)


# Create instance
MODEL_CONFIG = FSRCNNConfig()

@st.cache_resource
def load_fsrcnn_model(config):
    """Load FSRCNN ONNX model with caching"""
    try:
        import onnxruntime as ort
        
        # Path to ONNX model
        model_path = MODEL_CONFIG.onnx_file_path

        print(f"Model path: {model_path}")
        
        if not model_path.exists():
            st.error(f"Model file not found: {model_path}")
            return None, None
        
        # Check for GPU providers
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(str(model_path), providers=providers)
        device = "GPU" if 'CUDAExecutionProvider' in session.get_providers() else "CPU"
        
        return session, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def preprocess_image(image: Image.Image, scale_factor: int = 2) -> tuple:
    """
    Preprocess a high-resolution image into a low-resolution one for FSRCNN inference.
    """
    try:
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to tensor and normalize
        transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )

        tensor = transform(image).unsqueeze(0)

        return image, tensor
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None, None, None


def postprocess_output(hr_tensor: torch.Tensor) -> Image.Image:
    """Postprocess model output back to PIL Image"""
    try:
        # Clamp values
        hr_tensor = hr_tensor.squeeze(0).cpu().detach()
        hr_tensor = torch.clamp(hr_tensor, 0, 1)

        # Convert to PIL Image
        hr_image = T.ToPILImage()(hr_tensor)
        return hr_image
    except Exception as e:
        st.error(f"Error postprocessing output: {e}")
        return None


def run_inference(session, input_tensor, config) -> tuple:
    """Run inference with ONNX model"""
    try:
        # The ONNX runtime expects a numpy array
        lr_numpy = input_tensor.numpy()
        # Batch dimension is expected (B, C, H, W)
        lr_numpy_with_batch = np.expand_dims(lr_numpy, axis=0)

        # Run ONNX inference
        sr_numpy = session.run(
            output_names=["sr"], input_feed={"lr": lr_numpy}
        )[0]

        # Convert to PIL image
        # Clip to [0, 1] values and tranpose to (H, W, C)
        sr_numpy = sr_numpy[0]
        sr_hwc = np.clip(sr_numpy.transpose(1, 2, 0), 0.0, 1.0)
        sr_image = Image.fromarray((sr_hwc * 255.0).round().astype(np.uint8))

        return sr_image, sr_numpy
    except Exception as e:
        st.error(f"Error during inference: {e}")
        return None, None


def compute_metrics(lr_image: np.ndarray, hr_image: np.ndarray) -> dict:
    """Compute quality metrics"""
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity

        # Convert to numpy arrays if needed
        if isinstance(lr_image, Image.Image):
            lr_array = np.array(lr_image).astype(np.float32) / 255.0
        else:
            lr_array = lr_image

        if isinstance(hr_image, Image.Image):
            hr_array = np.array(hr_image).astype(np.float32) / 255.0
        else:
            hr_array = hr_image

        # Compute PSNR
        psnr = peak_signal_noise_ratio(lr_array, hr_array, data_range=1.0)

        # Compute SSIM (handle different channel counts)
        if len(lr_array.shape) == 3:
            ssim = structural_similarity(
                lr_array, hr_array, channel_axis=2, data_range=1.0
            )
        else:
            ssim = structural_similarity(lr_array, hr_array, data_range=1.0)

        return {
            "psnr": psnr,
            "ssim": ssim,
        }
    except Exception as e:
        st.warning(f"Could not compute metrics: {e}")
        return None


def render_live_inference_page(config):

    st.markdown(
        '<div class="section-header">🎬 Live Inference Demo</div>',
        unsafe_allow_html=True,
    )

    st.info(f"""
    📸 **Upload an image, and see FSRCNN in action.**
    
    - **Scale Factor:** {MODEL_CONFIG.scale_factor}x
    """)

    st.markdown("**Upload Image for Super-Resolution**")
    image_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="image_upload",
    )

    # Initialize ONNX inference session
    session, device = load_fsrcnn_model(MODEL_CONFIG)
    
    if session is None:
        st.error("❌ Failed to load model")
        return

    # Load image
    if image_file is not None:
        input_image = Image.open(image_file)
        image_name = image_file.name

        st.success(f"✅ Loaded: {image_name}")

        # Preprocess
        lr_image, lr_tensor = preprocess_image(input_image, MODEL_CONFIG.scale_factor)

        if lr_image is None or lr_tensor is None:
            st.error("Failed to preprocess image")
            return

        # Run inference
        st.info("🚀 Running inference...")
        progress_bar = st.progress(0)

        output_image, output_numpy = run_inference(session, lr_tensor, MODEL_CONFIG)

        progress_bar.progress(100)

        if output_image is None:
            st.error("Inference failed")
            return

        st.success(":white_check_mark: Inference complete")

        # Display input and output images side by side
        st.markdown("---")
        st.markdown("### Inference Results")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Image")
            st.image(input_image, caption=f"Original: {image_name}", use_container_width=True)

        with col2:
            st.subheader("Super-Resolved Output")
            output_caption = f"SR Output: {image_name}"
            st.image(output_image, caption=output_caption, use_container_width=True)

        # Download button
        st.markdown("---")
        st.markdown("### Download Results")

        import io

        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        st.download_button(
            label="⬇️ Download SR Image",
            data=img_byte_arr,
            file_name=f"srcnn_sr_{image_name}",
            mime="image/png",
        )


__all__ = ["render_live_inference_page"]
