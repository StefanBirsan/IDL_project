"""
Live Inference Demo Page for Physics-Informed MAE
Real-time image reconstruction with masked autoencoder
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
def load_pim_model(config):
    """Load Physics-Informed MAE model with caching"""
    try:
        from training.train_utils.fisr.physics_informed_mae import PhysicsInformedMAE
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create model
        model = PhysicsInformedMAE(
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            encoder_depth=config.encoder_depth,
            decoder_depth=config.decoder_depth,
            num_heads=config.num_heads,
            mask_ratio=config.mask_ratio
        )
        
        model = model.to(device)
        model.eval()
        
        st.success(f"✅ Model loaded on {device.upper()}")
        return model, device
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None


def preprocess_image_pim(image: Image.Image, img_size: int = 64) -> tuple:
    """Preprocess input image for Physics-Informed MAE inference"""
    try:
        # Convert to grayscale (for astronomical images)
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to model input size
        image = image.resize((img_size, img_size), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize to [-1, 1]
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
        
        img_tensor = transform(image).unsqueeze(0)
        
        return image, img_tensor
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None, None


def postprocess_output(hr_tensor: torch.Tensor) -> Image.Image:
    """Postprocess model output back to PIL Image"""
    try:
        # Denormalize from [-1, 1] to [0, 1]
        hr_tensor = hr_tensor.squeeze(0).cpu().detach()
        hr_tensor = (hr_tensor + 1) / 2
        
        # Clamp values
        hr_tensor = torch.clamp(hr_tensor, 0, 1)
        
        # Convert to PIL Image
        hr_image = T.ToPILImage(mode='L')(hr_tensor)
        return hr_image
    except Exception as e:
        st.error(f"Error postprocessing output: {e}")
        return None


def run_inference(model, img_tensor: torch.Tensor, device: str) -> tuple:
    """Run inference on image"""
    try:
        with torch.no_grad():
            img_tensor = img_tensor.to(device)
            start_time = time.time()
            
            # Get reconstructed image and mask
            reconstructed, mask = model(img_tensor)
            
            inference_time = time.time() - start_time
        
        return reconstructed, mask, inference_time
    except Exception as e:
        st.error(f"Error during inference: {e}")
        return None, None, None


def compute_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """Compute quality metrics"""
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        
        # Convert to numpy arrays if needed
        if isinstance(original, Image.Image):
            orig_array = np.array(original).astype(np.float32) / 255.0
        else:
            orig_array = original
        
        if isinstance(reconstructed, Image.Image):
            recon_array = np.array(reconstructed).astype(np.float32) / 255.0
        else:
            recon_array = reconstructed
        
        # Compute PSNR
        psnr = peak_signal_noise_ratio(orig_array, recon_array, data_range=1.0)
        
        # Compute SSIM (handle different channel counts)
        if len(orig_array.shape) == 3:
            ssim = structural_similarity(
                orig_array, recon_array, 
                channel_axis=2, 
                data_range=1.0
            )
        else:
            ssim = structural_similarity(
                orig_array, recon_array, 
                data_range=1.0
            )
        
        return {
            'psnr': psnr,
            'ssim': ssim,
        }
    except Exception as e:
        st.warning(f"Could not compute metrics: {e}")
        return None


def visualize_mask(mask: np.ndarray, img_size: int = 64) -> Image.Image:
    """Visualize the mask pattern"""
    try:
        # Reshape mask to image dimensions
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        mask = mask.reshape(img_size // 4, img_size // 4)
        mask = np.repeat(np.repeat(mask, 4, axis=0), 4, axis=1)
        
        # Create visualization (red for masked, green for visible)
        mask_vis = np.zeros((img_size, img_size, 3))
        mask_vis[mask == 1] = [1, 0, 0]  # Red for masked
        mask_vis[mask == 0] = [0, 1, 0]  # Green for visible
        
        mask_img = Image.fromarray((mask_vis * 255).astype(np.uint8))
        return mask_img
    except Exception as e:
        st.warning(f"Could not visualize mask: {e}")
        return None


def render_live_inference_page(config):
    """Render the live inference demo page for Physics-Informed MAE"""
    
    st.markdown('<div class="section-header">🎬 Live Inference Demo</div>', unsafe_allow_html=True)
    
    st.info("""
    📸 **Upload a model and image, then watch Physics-Informed MAE reconstruct it!**
    
    - **Image Size:** {}×{}
    - **Architecture:** Masked Autoencoder with Physics-Informed Preprocessing
    - **Mask Ratio:** {}%
    - **Device:** GPU (if available) or CPU
    """.format(
        config.img_size,
        config.img_size,
        int(config.mask_ratio * 100)
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
                key="model_upload_pim"
            )
            
            use_default_model = st.checkbox(
                "Or use default Physics-Informed MAE model",
                value=not model_file,
                key="use_default_model_pim"
            )
            
        with col2:
            st.markdown("**Step 2: Upload Image for Reconstruction**")
            image_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
                label_visibility="collapsed",
                key="image_upload_pim"
            )
            
            use_example = st.checkbox("Or use an example image", value=False)
        
        # Load model
        model = None
        device = None
        model_source = None
        model_type = None
        
        if use_default_model or model_file is None:
            # Load default PyTorch model
            st.info("📦 Loading default Physics-Informed MAE model...")
            model, device = load_pim_model(config)
            model_source = "Default Physics-Informed MAE model"
            model_type = "pytorch"
        elif model_file is not None:
            # Detect model type by extension
            file_ext = model_file.name.lower().split('.')[-1]
            
            if file_ext == 'onnx':
                # Load ONNX model
                from streamlit.components.viz import validate_onnx_model, load_onnx_model_from_upload
                
                st.info("🔍 Validating uploaded ONNX model...")
                if validate_onnx_model(model_file):
                    st.info("📦 Loading uploaded ONNX model...")
                    model, device = load_onnx_model_from_upload(model_file)
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
                    input_image = Image.new('L', (128, 128), color=128)
                    image_name = "sample_image.png"
                
                st.success(f"✅ Loaded: {image_name}")
                
                # Preprocess
                st.info("🔄 Preprocessing image...")
                processed_image, img_tensor = preprocess_image_pim(
                    input_image, config.img_size
                )
                
                if processed_image is None or img_tensor is None:
                    st.error("Failed to preprocess image")
                    return
                
                st.info(f"Resized to: {config.img_size}×{config.img_size} (required for model)")
                
                # Run inference
                st.info("🚀 Running inference...")
                progress_bar = st.progress(0)
                
                if model_type == "pytorch":
                    reconstructed, mask, inference_time = run_inference(model, img_tensor, device)
                else:
                    # ONNX inference
                    start_time = time.time()
                    input_name = model.get_inputs()[0].name
                    input_data = img_tensor.numpy().astype(np.float32)
                    outputs = model.run(None, {input_name: input_data})
                    reconstructed = torch.from_numpy(outputs[0]).float()
                    mask = torch.from_numpy(outputs[1]).float() if len(outputs) > 1 else None
                    inference_time = time.time() - start_time
                
                progress_bar.progress(100)
                
                if reconstructed is None:
                    st.error("Inference failed")
                    return
                
                st.success(f"✅ Inference complete in {inference_time:.3f}s")
                
                # Postprocess
                reconstructed_image = postprocess_output(reconstructed)
                
                if reconstructed_image is None:
                    st.error("Failed to postprocess output")
                    return
                
                # Create visualization
                st.markdown("---")
                st.markdown("### Results")
                
                col_input, col_recon = st.columns(2)
                
                with col_input:
                    st.markdown("**Input Image**")
                    st.image(processed_image, use_column_width=True)
                    st.caption(f"Size: {config.img_size}×{config.img_size}")
                
                with col_recon:
                    st.markdown("**Reconstructed Image**")
                    st.image(reconstructed_image, use_column_width=True)
                    st.caption(f"From {int(config.mask_ratio*100)}% masked patches")
                
                # Show mask visualization
                show_mask = st.checkbox("Show mask visualization", value=True, key="show_mask_pim")
                if show_mask and mask is not None:
                    st.markdown("---")
                    st.markdown("### Mask Pattern Visualization")
                    
                    mask_img = visualize_mask(mask, config.img_size)
                    
                    if mask_img is not None:
                        col_mask_info, col_mask_vis = st.columns([1, 1])
                        
                        with col_mask_info:
                            st.markdown("""
                            **Mask Legend:**
                            - 🟢 Green: Visible patches
                            - 🔴 Red: Masked patches
                            
                            The model learns to reconstruct
                            the red (masked) regions from
                            the green (visible) patches.
                            """)
                        
                        with col_mask_vis:
                            st.image(mask_img, use_column_width=True)
                
                # Show metrics
                if show_metrics:
                    st.markdown("---")
                    st.markdown("### Quality Metrics")
                    
                    metrics = compute_metrics(processed_image, reconstructed_image)
                    
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
                reconstructed_image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                st.download_button(
                    label="⬇️ Download Reconstructed Image",
                    data=img_byte_arr,
                    file_name=f"pim_reconstructed_{image_name}",
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
        st.markdown("### About Physics-Informed MAE Live Inference")
        
        st.markdown(f"""
        **Model Information:**
        - **Name:** {config.name}
        - **Description:** {config.description}
        - **Image Size:** {config.img_size}×{config.img_size}
        - **Patch Size:** {config.patch_size}
        - **Mask Ratio:** {int(config.mask_ratio * 100)}%
        
        **Architecture Details:**
        - **Embedding Dimension:** {config.embed_dim}
        - **Encoder Depth:** {config.encoder_depth} layers
        - **Decoder Depth:** {config.decoder_depth} layers
        - **Attention Heads:** {config.num_heads}
        
        **Key Characteristics:**
        - Masked autoencoder for self-supervised learning
        - Physics-informed preprocessing for astronomical data
        - Learns to reconstruct masked image regions
        - Supports multi-wavelength data integration
        
        **Device Information:**
        - Current Device: **{device.upper()}**
        - CUDA Available: {"Yes ✅" if torch.cuda.is_available() else "No ❌"}
        """)


__all__ = ['render_live_inference_page']
