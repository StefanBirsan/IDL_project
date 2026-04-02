"""
Examples and inference page for Physics-Informed MAE
"""
import streamlit as st
import numpy as np
from pathlib import Path


def render_examples_page(config):
    """Render examples page for Physics-Informed MAE"""
    
    st.info("""
    💡 **Note:** Connect training checkpoints and dataset to run live inference.
    This section shows the inference framework.
    """)
    
    st.markdown('<div class="section-header">Example Showcase</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Input Image")
        st.markdown(f"""
        The original input image:
        - Shape: {config.img_size} x {config.img_size} pixels
        - Single channel (grayscale)
        - Normalized to [-1, 1]
        
        **Features:**
        - Various edge profiles
        - Different intensity levels
        - Natural astronomical data
        """)
    
    with col2:
        st.markdown("### Reconstructed Image")
        st.markdown(f"""
        Model's reconstruction output:
        - Same shape as input: {config.img_size} x {config.img_size}
        - Predicted from {int(config.mask_ratio*100)}% masked patches
        - Physics-informed constraints applied
        
        **Quality Metrics:**
        - PSNR: Peak Signal-to-Noise Ratio
        - SSIM: Structural Similarity Index
        - MAE: Pixel-level accuracy
        """)
    
    st.markdown('<div class="section-header">Inference Setup</div>', unsafe_allow_html=True)
    
    with st.expander("How to Run Inference", expanded=True):
        st.code("""
from training.inference import Inference
from training.train_utils import create_physics_informed_mae

# Load model
model = create_physics_informed_mae(
    img_size=64,
    patch_size=4,
    embed_dim=768,
    encoder_depth=12,
    decoder_depth=8,
    num_heads=12,
    mask_ratio=0.75
)

# Create inference engine
inference = Inference(
    model=model,
    checkpoint_path='checkpoints/best_model.pt',
    device='cuda'
)

# Run inference
output = inference.infer(input_image)

# Visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))
plt.subplot(121); plt.imshow(input_image[0,0], cmap='gray')
plt.subplot(122); plt.imshow(output[0,0].numpy(), cmap='gray')
plt.show()
        """, language="python")
    
    st.markdown('<div class="section-header">Visualization Techniques</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Edge Maps
        Visualize the physics-informed preprocessing:
        - Sobel-filtered gradients
        - Double differentiation
        - Edge strength magnitude
        """)
    
    with col2:
        st.markdown("""
        ### Attention Maps
        Understand what the model focuses on:
        - Multi-head attention visualization
        - Patch importance scores
        - Feature activation patterns
        """)
    
    with col3:
        st.markdown("""
        ### Residual Analysis
        Compare input vs output:
        - Pixel-level difference maps
        - Error distribution
        - Performance metrics
        """)


def upload_and_infer():
    """Section for uploading images and running inference"""
    st.subheader("Interactive Inference")
    
    uploaded_file = st.file_uploader("Upload a .npy image", type=['npy'])
    
    if uploaded_file is not None:
        try:
            # Load the uploaded numpy file
            image = np.load(uploaded_file)
            st.info(f"Loaded image with shape: {image.shape}")
            
            # Show upload options
            col1, col2 = st.columns(2)
            with col1:
                st.write("Original Image Statistics:")
                st.write(f"Min: {image.min():.4f}")
                st.write(f"Max: {image.max():.4f}")
                st.write(f"Mean: {image.mean():.4f}")
                st.write(f"Std: {image.std():.4f}")
            
            with col2:
                st.write("Image Preview:")
                st.image(image, clamp=True, use_column_width=True)
            
            # Inference button
            if st.button("Run Inference", key="inference_button"):
                st.info("Inference would run here with loaded checkpoint...")
                try:
                    import torch
                    st.success("PyTorch available for inference")
                except ImportError:
                    st.warning("PyTorch not installed - install with: pip install torch")
                
        except Exception as e:
            st.error(f"Error loading file: {e}")
