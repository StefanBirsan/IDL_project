"""
Live Inference Demo Page for SRCNN
Real-time image super-resolution with classic SRCNN architecture
"""
import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import sys
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@st.cache_resource
def load_srcnn_model(config):
    """Load SRCNN model with caching"""
    try:
        from training.train_utils.srcnn.srcnn import SRCNN
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create model
        model = SRCNN(
            scale_factor=config.scale_factor,
            intermediate_channels=config.intermediate_channels
        )
        
        model = model.to(device)
        model.eval()
        
        st.success(f"✅ Model loaded on {device.upper()}")
        return model, device
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None


def load_onnx_model_from_upload_srcnn(uploaded_file):
    """Load ONNX model from uploaded file for SRCNN"""
    try:
        import onnxruntime as ort
        import tempfile
        import os
        
        # Create temporary directory to save the uploaded file
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        try:
            # Check for GPU providers
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(tmp_path, providers=providers)
            device = "GPU" if 'CUDAExecutionProvider' in session.get_providers() else "CPU"
            
            st.success(f"✅ ONNX Model loaded on {device}")
            return session, device
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except ImportError:
        st.error("❌ onnxruntime not installed. Please install it: pip install onnxruntime")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading ONNX model: {e}")
        return None, None


def preprocess_image_srcnn(image: Image.Image, scale_factor: int = 2) -> tuple:
    """Preprocess input image for SRCNN inference"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get original size
        orig_width, orig_height = image.size
        
        # Downsample to create LR image
        lr_width = orig_width // scale_factor
        lr_height = orig_height // scale_factor
        
        # Ensure dimensions are even
        lr_width = (lr_width // 2) * 2
        lr_height = (lr_height // 2) * 2
        
        lr_image = image.resize((lr_width, lr_height), Image.Resampling.LANCZOS)
        
        # Bicubic upsampling (SRCNN expects bicubic pre-upsampled input)
        hr_width = lr_width * scale_factor
        hr_height = lr_height * scale_factor
        upsampled_image = lr_image.resize((hr_width, hr_height), Image.Resampling.BICUBIC)
        
        # Convert to tensor and normalize
        transform = T.Compose([
            T.ToTensor(),
        ])
        
        upsampled_tensor = transform(upsampled_image).unsqueeze(0)
        
        return lr_image, upsampled_image, upsampled_tensor
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


def run_inference(model, input_tensor, device: str, model_type: str = "pytorch") -> tuple:
    """Run inference on pre-upsampled image"""
    try:
        start_time = time.time()
        
        if model_type == "onnx":
            # ONNX inference
            input_name = model.get_inputs()[0].name
            input_data = input_tensor.numpy().astype(np.float32)
            outputs = model.run(None, {input_name: input_data})
            hr_output = torch.from_numpy(outputs[0]).float()
        else:
            # PyTorch inference
            with torch.no_grad():
                input_tensor = input_tensor.to(device)
                hr_output = model(input_tensor)
        
        inference_time = time.time() - start_time
        return hr_output, inference_time
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
                lr_array, hr_array, 
                channel_axis=2, 
                data_range=1.0
            )
        else:
            ssim = structural_similarity(
                lr_array, hr_array, 
                data_range=1.0
            )
        
        return {
            'psnr': psnr,
            'ssim': ssim,
        }
    except Exception as e:
        st.warning(f"Could not compute metrics: {e}")
        return None


def render_live_inference_page(config):
    """Render the live inference demo page for SRCNN"""
    
    st.markdown('<div class="section-header">🎬 Live Inference Demo</div>', unsafe_allow_html=True)
    
    st.info("""
    📸 **Upload a model and image, then watch SRCNN enhance it in real-time!**
    
    - **Scale Factor:** {}×
    - **Architecture:** Classic 3-layer CNN with {} intermediate channels
    - **Device:** GPU (if available) or CPU
    - **Input:** Bicubic pre-upsampled images
    """.format(
        config.scale_factor,
        config.intermediate_channels
    ))
    
    # Create tabs for different inference modes
    tab1, tab2, tab3 = st.tabs(["📤 Upload Model & Image", "📊 Batch Processing", "ℹ️ About"])
    
    with tab1:
        st.markdown("### Model & Image Upload")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Step 1: Upload Model**")
            model_file = st.file_uploader(
                "Choose a model file (.onnx or .pth)",
                type=['onnx', 'pth', 'pt'],
                label_visibility="collapsed",
                key="model_upload_srcnn"
            )
            
            use_default_model = st.checkbox(
                "Or use default SRCNN model",
                value=not model_file,
                key="use_default_model_srcnn"
            )
            
        with col2:
            st.markdown("**Step 2: Upload Image for Upscaling**")
            image_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
                label_visibility="collapsed",
                key="image_upload_srcnn"
            )
            
            use_example = st.checkbox("Or use an example image", value=False)
        
        # Load model
        model = None
        device = None
        model_source = None
        model_type = None
        
        if use_default_model or model_file is None:
            # Load default PyTorch model
            st.info("📦 Loading default SRCNN model...")
            model, device = load_srcnn_model(config)
            model_source = "Default SRCNN model"
            model_type = "pytorch"
        elif model_file is not None:
            # Detect model type by extension
            file_ext = model_file.name.lower().split('.')[-1]
            
            if file_ext == 'onnx':
                # Load ONNX model
                from streamlit.components.viz import validate_onnx_model
                
                st.info("🔍 Validating uploaded ONNX model...")
                if validate_onnx_model(model_file):
                    st.info("📦 Loading uploaded ONNX model...")
                    model, device = load_onnx_model_from_upload_srcnn(model_file)
                    model_source = f"Custom ONNX model: {model_file.name}"
                    model_type = "onnx"
                else:
                    st.error("❌ Invalid ONNX model file. Please upload a valid ONNX model.")
                    model = None
                    device = None
            else:
                st.error(f"❌ Unsupported model format: {file_ext}. Please upload .onnx, .pth, or .pt files.")
                model = None
                device = None
        
        if model is None or device is None:
            st.error("Failed to load model. Please check your configuration.")
            return
        
        st.success(f"✅ Model loaded: {model_source}")
        
        # Process image
        if image_file is not None or use_example:
            try:
                # Load image
                if image_file is not None:
                    input_image = Image.open(image_file)
                    image_name = image_file.name
                else:
                    # Create a sample image if example is selected
                    input_image = Image.new('RGB', (128, 128), color='blue')
                    image_name = "sample_image.png"
                
                st.success(f"✅ Loaded: {image_name}")
                
                # Preprocess
                st.info("🔄 Preprocessing image...")
                lr_image, upsampled_image, upsampled_tensor = preprocess_image_srcnn(
                    input_image, config.scale_factor
                )
                
                if lr_image is None or upsampled_tensor is None:
                    st.error("Failed to preprocess image")
                    return
                
                lr_width, lr_height = lr_image.size
                hr_width, hr_height = upsampled_image.size
                
                st.info(f"Original: {lr_width}×{lr_height} → Bicubic upsampled: {hr_width}×{hr_height}")
                
                # Run inference
                st.info("🚀 Running inference...")
                progress_bar = st.progress(0)
                
                hr_output, inference_time = run_inference(model, upsampled_tensor, device, model_type=model_type)
                
                progress_bar.progress(100)
                
                if hr_output is None:
                    st.error("Inference failed")
                    return
                
                st.success(f"✅ Inference complete in {inference_time:.3f}s")
                
                # Postprocess
                hr_image = postprocess_output(hr_output)
                
                if hr_image is None:
                    st.error("Failed to postprocess output")
                    return
                
                # Create visualization
                st.markdown("---")
                st.markdown("### Results Comparison")
                
                col_bicubic, col_srcnn = st.columns(2)
                
                with col_bicubic:
                    st.markdown("**Bicubic Interpolation**")
                    st.image(upsampled_image, use_column_width=True)
                    st.caption("Baseline: Bicubic upsampling")
                
                with col_srcnn:
                    st.markdown("**SRCNN Enhanced**")
                    st.image(hr_image, use_column_width=True)
                    st.caption(f"Size: {hr_width}×{hr_height}")
                
                # Show metrics
                show_metrics = st.checkbox("Show quality metrics", value=True, key="show_metrics_srcnn")
                if show_metrics:
                    st.markdown("---")
                    st.markdown("### Quality Metrics")
                    
                    metrics = compute_metrics(upsampled_image, hr_image)
                    
                    if metrics is not None:
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.metric("PSNR (dB)", f"{metrics['psnr']:.2f}")
                        with col_m2:
                            st.metric("SSIM (0-1)", f"{metrics['ssim']:.4f}")
                
                # Download button
                st.markdown("---")
                st.markdown("### Download Results")
                
                import io
                img_byte_arr = io.BytesIO()
                hr_image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                st.download_button(
                    label="⬇️ Download SR Image",
                    data=img_byte_arr,
                    file_name=f"srcnn_sr_{image_name}",
                    mime="image/png"
                )
                
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                traceback.print_exc()
    
    with tab2:
        st.markdown("### Batch Processing")
        st.info("Upload multiple images for batch processing (coming soon)")
        
        st.markdown("""
        **Features:**
        - Process multiple images at once
        - Export results as ZIP
        - Generate comparison reports
        """)
    
    with tab3:
        st.markdown("### About SRCNN Live Inference")
        
        st.markdown(f"""
        **Model Information:**
        - **Name:** {config.name}
        - **Description:** {config.description}
        - **Scale Factor:** {config.scale_factor}×
        - **Intermediate Channels:** {config.intermediate_channels}
        
        **Architecture:**
        - **Patch Extraction:** Extracts overlapping patches from input
        - **Non-linear Mapping:** Maps low-res features to high-res space
        - **Reconstruction:** Aggregates patches into final image
        
        **Key Characteristics:**
        - Requires bicubic pre-upsampling
        - Simple yet effective 3-layer architecture
        - Fast inference speed
        - Good baseline for super-resolution tasks
        
        **Device Information:**
        - Current Device: **{device.upper()}**
        - CUDA Available: {"Yes ✅" if torch.cuda.is_available() else "No ❌"}
        """)


__all__ = ['render_live_inference_page']
