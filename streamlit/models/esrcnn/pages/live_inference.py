"""
Live Inference Demo Page for ESRCNN
Real-time image super-resolution with interactive upload and visualization
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
def load_esrcnn_model(config):
    """Load ESRCNN ONNX model with caching"""
    try:
        import onnxruntime as ort
        
        # Path to ONNX model
        model_path = Path(__file__).parent.parent.parent.parent.parent / "models" / "esrcnn_epoch_0029.onnx"
        
        if not model_path.exists():
            st.error(f"❌ Model file not found: {model_path}")
            return None, None
        
        # Check for GPU providers
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(str(model_path), providers=providers)
        device = "GPU" if 'CUDAExecutionProvider' in session.get_providers() else "CPU"
        
        st.success(f"✅ ONNX Model loaded on {device}")
        return session, device
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def preprocess_image(image: Image.Image, scale_factor: int = 2) -> tuple:
    """Preprocess input image for inference"""
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
        
        # Convert to tensor and normalize
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        lr_tensor = transform(lr_image).unsqueeze(0)
        
        # Convert to numpy array for ONNX runtime (float32)
        lr_array = lr_tensor.numpy().astype(np.float32)
        
        return lr_image, lr_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None, None


def postprocess_output(hr_tensor: torch.Tensor) -> Image.Image:
    """Postprocess model output back to PIL Image"""
    try:
        # Denormalize
        hr_tensor = hr_tensor.squeeze(0).cpu().detach()
        hr_tensor = (hr_tensor + 1) / 2  # [-1, 1] -> [0, 1]
        
        # Clamp values
        hr_tensor = torch.clamp(hr_tensor, 0, 1)
        
        # Convert to PIL Image
        hr_image = T.ToPILImage()(hr_tensor)
        return hr_image
    except Exception as e:
        st.error(f"Error postprocessing output: {e}")
        return None


def run_inference(session, input_data: np.ndarray, device: str) -> tuple:
    """Run ONNX inference on input data"""
    try:
        start_time = time.time()
        
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Run inference
        outputs = session.run(None, {input_name: input_data})
        
        inference_time = time.time() - start_time
        
        # Return first output (HR image)
        return torch.from_numpy(outputs[0]).float(), inference_time
    except Exception as e:
        st.error(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
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
    """Render the live inference demo page"""
    
    st.markdown('<div class="section-header">🎬 Live Inference Demo</div>', unsafe_allow_html=True)
    
    st.info("""
    📸 **Upload a model and image, then watch ESRCNN enhance it in real-time!**
    
    - **Scale Factor:** {}×
    - **Architecture:** {} residual blocks with {} features
    - **Device:** GPU (if available) or CPU
    """.format(
        config.scale_factor,
        config.num_residual_blocks,
        config.num_features
    ))
    
    # Create tabs for different inference modes
    tab1, tab2, tab3 = st.tabs(["📤 Upload Model & Image", "📊 Batch Processing", "ℹ️ About"])
    
    with tab1:
        st.markdown("### Model & Image Upload")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Step 1: Upload ONNX Model**")
            model_file = st.file_uploader(
                "Choose an ONNX model file",
                type=['onnx'],
                label_visibility="collapsed",
                key="model_upload"
            )
            
            use_default_model = st.checkbox(
                "Or use default ESRCNN model",
                value=not model_file,
                key="use_default_model"
            )
            
        with col2:
            st.markdown("**Step 2: Upload Image for Upscaling**")
            image_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
                label_visibility="collapsed",
                key="image_upload"
            )
            
            use_example = st.checkbox("Or use an example image", value=False)
        
        # Load model
        model = None
        device = None
        model_source = None
        
        if use_default_model or model_file is None:
            # Load default model
            st.info("📦 Loading default ESRCNN model...")
            model, device = load_esrcnn_model(config)
            model_source = "Default ESRCNN model"
        elif model_file is not None:
            # Validate and load uploaded model
            st.info("🔍 Validating uploaded model...")
            from streamlit.components.viz import validate_onnx_model, load_onnx_model_from_upload
            
            if validate_onnx_model(model_file):
                st.info("📦 Loading uploaded ONNX model...")
                model, device = load_onnx_model_from_upload(model_file)
                model_source = f"Custom model: {model_file.name}"
            else:
                st.error("❌ Invalid ONNX model file. Please upload a valid ONNX model.")
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
                    input_image = Image.new('RGB', (128, 128), color='red')
                    image_name = "sample_image.png"
                
                st.success(f"✅ Loaded: {image_name}")
                
                # Preprocess
                st.info("🔄 Preprocessing image...")
                lr_image, lr_array = preprocess_image(input_image, config.scale_factor)
                
                if lr_image is None or lr_array is None:
                    st.error("Failed to preprocess image")
                    return
                
                lr_width, lr_height = lr_image.size
                hr_width = lr_width * config.scale_factor
                hr_height = lr_height * config.scale_factor
                
                st.info(f"Input LR size: {lr_width}×{lr_height} → HR size: {hr_width}×{hr_height}")
                
                # Run inference
                st.info("🚀 Running inference...")
                progress_bar = st.progress(0)
                
                hr_output, inference_time = run_inference(model, lr_array, device)
                
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
                st.markdown("### Results")
                
                col_lr, col_hr = st.columns(2)
                
                with col_lr:
                    st.markdown("**Input (Low Resolution)**")
                    st.image(lr_image, use_column_width=True)
                    lr_width, lr_height = lr_image.size
                    st.caption(f"Size: {lr_width}×{lr_height}")
                
                with col_hr:
                    st.markdown("**Output (Super-Resolution)**")
                    st.image(hr_image, use_column_width=True)
                    hr_width, hr_height = hr_image.size
                    st.caption(f"Size: {hr_width}×{hr_height}")
                
                # Show metrics
                show_metrics = st.checkbox("Show quality metrics", value=True, key="show_metrics")
                if show_metrics:
                    st.markdown("---")
                    st.markdown("### Quality Metrics")
                    
                    # Resize LR to match HR for metric computation
                    lr_resized = lr_image.resize((hr_width, hr_height), Image.Resampling.BILINEAR)
                    metrics = compute_metrics(lr_resized, hr_image)
                    
                    if metrics is not None:
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.metric("PSNR (dB)", f"{metrics['psnr']:.2f}")
                        with col_m2:
                            st.metric("SSIM (0-1)", f"{metrics['ssim']:.4f}")
                
                # Download button
                st.markdown("---")
                st.markdown("### Download Results")
                
                col_d1, col_d2 = st.columns(2)
                
                with col_d1:
                    # Convert PIL image to bytes
                    import io
                    img_byte_arr = io.BytesIO()
                    hr_image.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    
                    st.download_button(
                        label="⬇️ Download HR Image",
                        data=img_byte_arr,
                        file_name=f"sr_{image_name}",
                        mime="image/png"
                    )
                
                with col_d2:
                    # Also provide LR for reference
                    img_byte_arr_lr = io.BytesIO()
                    lr_image.save(img_byte_arr_lr, format='PNG')
                    img_byte_arr_lr.seek(0)
                    
                    st.download_button(
                        label="⬇️ Download LR Image",
                        data=img_byte_arr_lr,
                        file_name=f"lr_{image_name}",
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
        st.markdown("### About ESRCNN Live Inference")
        
        st.markdown(f"""
        **Model Information:**
        - **Name:** {config.name}
        - **Description:** {config.description}
        - **Scale Factor:** {config.scale_factor}×
        - **Residual Blocks:** {config.num_residual_blocks}
        - **Feature Channels:** {config.num_features}
        
        **Architecture Highlights:**
        - Deep residual learning for robust feature extraction
        - Global skip connections for improved gradient flow
        - Perceptual loss for realistic facial details
        - Sub-pixel convolution for upsampling
        
        **Performance:**
        - PSNR: 28-32 dB
        - SSIM: 0.85-0.92
        - Parameters: ~930K
        
        **Device Information:**
        - Current Device: **{device.upper()}**
        - CUDA Available: {"Yes ✅" if torch.cuda.is_available() else "No ❌"}
        """)


__all__ = ['render_live_inference_page']
