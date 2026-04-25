import streamlit as st


def render_documentation_page(config):
    st.markdown('<div class="section-header">Model Specifications</div>', unsafe_allow_html=True)
    
    with st.expander("Training Configuration", expanded=True):
        hyperparameters_col, loss_col = st.columns(2)
        
        with hyperparameters_col:
            st.markdown("**Hyperparameters**")
            st.code(f"""
Optimizer: SGD with Momentum
  momentum: {config.momentum}
  batch_size: {config.batch_size}

Learning Rates:
  Layers 1-2: {config.lr_early_layers:.1e}
  Layer 3: {config.lr_reconstruction_layer:.1e}
            """)
        
        with loss_col:
            st.markdown("**Loss Configuration**")
            st.code(f"""
Primary Loss:
  L_MSE = Mean Squared Error = nn.MSELoss()
  
Applied to:
  - Training: patches of size {config.crop_size}
  - Validation: full-resolution images
            """)
    
    st.markdown('<div class="section-header">Layer-Specific Learning Rates</div>', unsafe_allow_html=True)

    layer1_2_col, layer3_col = st.columns(2)

    with layer1_2_col:
        st.markdown(f"""
        ### Layers 1-2
        **Learning Rate:** {config.lr_early_layers:.1e}
        
        - Feature extraction
        - Learning general patterns
        - Higher capacity for change
        """)
    
    with layer3_col:
        st.markdown(f"""
        ### Layer 3 (Reconstruction)
        **Learning Rate:** {config.lr_reconstruction_layer:.1e}
        
        - Final output prediction
        - Refined reconstruction
        - Lower learning rate to preserve learned features
        """)
    
    st.markdown('<div class="section-header">Training Pipeline</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Data Preparation
    1. **Load Dataset**: Load high-resolution (HR) images from training directory
    4. **Downscale**: Downscale the HR images by the scale factor (e.g., 2x, 4x) using bicubic interpolation
    5. **Upscale**: Upscale the downscaled images back to original HR size using bicubic interpolation (to create LR input)
    
    ### Training Loop
    1. **Batch Formation**: Sample random patches from training set
    2. **Forward Pass**: Pass LR patches through SRCNN
    3. **Loss Computation**: Calculate MSE between output and HR patches
    4. **Backward Pass**: Compute gradients
    5. **Optimization**: Update weights using SGD with layer-specific LR
    6. **Logging**: Track loss metrics
    
    ### Validation
    - Run on full-resolution images
    - Compute PSNR and SSIM metrics
    
    ### Checkpointing
    - Save best model based on validation PSNR
    - Save periodic checkpoints for recovery
    """)
