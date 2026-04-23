"""
Documentation page for SRCNN
Comprehensive technical documentation integrated from training guide
"""
import streamlit as st


def render_documentation_page(config):
    """Render documentation page for SRCNN"""
    
    st.markdown('<div class="section-header">Model Specifications</div>', unsafe_allow_html=True)
    
    # Training Configuration
    with st.expander("⚙️ Training Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Model Architecture**")
            st.code(f"""
Scale Factor: {config.scale_factor}×
Crop Size: {config.crop_size}×{config.crop_size}

Layers:
  Layer 1: 9×9 Conv
    Input: 1 channel
    Output: {config.intermediate_channels} channels
    Activation: ReLU
  
  Layer 2: 5×5 Conv
    Input: {config.intermediate_channels} channels
    Output: {config.intermediate_channels} channels
    Activation: ReLU
  
  Layer 3: 5×5 Conv
    Input: {config.intermediate_channels} channels
    Output: 1 channel
    No Activation (regression)

Total Parameters: ~8.6M
            """)
        
        with col2:
            st.markdown("**Training Hyperparameters**")
            st.code(f"""
Optimizer: SGD with Momentum
  momentum: {config.momentum}
  batch_size: {config.batch_size}

Learning Rates:
  Layers 1-2: {config.lr_early_layers:.2e}
  Layer 3: {config.lr_reconstruction_layer:.2e}
  Ratio: {config.lr_early_layers / config.lr_reconstruction_layer:.0f}:1

Scheduler: Fixed LR
  (Alternatively: StepLR or CosineAnnealingLR)
  
Convergence:
  num_epochs: {config.num_epochs}
  log_interval: 10 batches
            """)
        
        with col3:
            st.markdown("**Loss Configuration**")
            st.code(f"""
Primary Loss:
  L_MSE = Mean Squared Error
  
Applied to:
  - Training: patches of size {config.crop_size}
  - Validation: full-resolution images
  
Loss Computation:
  L = 1/B × Σ ||ŷ - y||²
  
Where:
  B = batch size ({config.batch_size})
  ŷ = super-resolved output
  y = ground truth HR image

Properties:
  ✓ Differentiable (smooth gradients)
  ✓ PSNR-oriented loss
  ✓ Simple and effective
            """)
    
    st.markdown('<div class="section-header">Loss Functions</div>', unsafe_allow_html=True)
    
    # MSE Loss
    with st.expander("📊 1. Mean Squared Error (MSE) Loss", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            **Type:** Mean Squared Error (MSE)
            **Applies to:** Entire output
            
            **Mathematical Definition:**
            $$L_{{MSE}} = \\frac{{1}}{{B \\times H \\times W}} \\sum_{{i,j}} |\\hat{{y}}_{{i,j}} - y_{{i,j}}|^2$$
            
            **Where:**
            - $B$ = batch size ({config.batch_size})
            - $H \\times W$ = spatial dimensions ({config.crop_size}×{config.crop_size})
            - $\\hat{{y}}_{{i,j}}$ = predicted HR pixel
            - $y_{{i,j}}$ = ground truth HR pixel
            
            **Properties:**
            ✓ Smooth gradients across entire domain
            ✓ Encourages PSNR maximization
            ✓ Penalizes large errors more heavily
            ✓ Simple to implement and optimize
            
            **Relationship to PSNR:**
            $$PSNR = 10 \\log_{{10}}\\left(\\frac{{L_{{max}}^2}}{{MSE}}\\right)$$
            
            Where $L_{{max}}$ is the maximum pixel value (typically 1.0 for normalized images)
            """)
        
        with col2:
            st.info(f"""
            **Training Tips**
            
            • Use normalized pixel values [-1, 1] or [0, 1]
            • Compute MSE on HR patches
            • Accumulate over full batch
            • Include all spatial locations
            """)
    
    st.markdown('<div class="section-header">Layer-Specific Learning Rates</div>', unsafe_allow_html=True)
    
    with st.expander("📚 Learning Rate Strategy", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            ### Early Layers (1-2)
            **Learning Rate:** {config.lr_early_layers:.2e}
            
            **Purpose:**
            - Feature extraction
            - Learning general patterns
            - Higher capacity for change
            
            **Rationale:**
            - Earlier in network
            - Learn basic patterns
            - Higher adaptability needed
            """)
        
        with col2:
            st.markdown(f"""
            ### Reconstruction Layer (3)
            **Learning Rate:** {config.lr_reconstruction_layer:.2e}
            
            **Purpose:**
            - Final output prediction
            - Refined reconstruction
            - Stability critical
            
            **Rationale:**
            - Directly affects output
            - Small changes can cause instability
            - Need careful fine-tuning
            """)
        
        with col3:
            st.markdown(f"""
            ### Learning Rate Ratio
            **Ratio:** {config.lr_early_layers / config.lr_reconstruction_layer:.0f}:1
            
            **Effect:**
            - Early layers: 10× faster updates
            - Reconstruction: 10× slower updates
            
            **Implementation:**
            - Use param groups in optimizer
            - Assign different LR to different layers
            """)
        
        st.code("""
import torch.optim as optim

optimizer = optim.SGD([
    {'params': model.conv1.parameters(), 'lr': 1e-4},
    {'params': model.conv2.parameters(), 'lr': 1e-4},
    {'params': model.conv3.parameters(), 'lr': 1e-5},
], momentum=0.9)
        """, language="python")
    
    st.markdown('<div class="section-header">Training Pipeline</div>', unsafe_allow_html=True)
    
    with st.expander("🔄 Training Pipeline Details", expanded=True):
        st.markdown("""
        ### Data Preparation
        1. **Load Dataset**: Collect low-resolution (LR) and high-resolution (HR) image pairs
        2. **Normalization**: Normalize pixel values to [-1, 1] or [0, 1]
        3. **Patching**: Extract random patches from HR images
        4. **Downsampling**: Downsample HR patches to LR (using bicubic/nearest)
        
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
        - Track overfitting
        
        ### Checkpointing
        - Save best model based on validation PSNR
        - Save periodic checkpoints for recovery
        """)
    
    st.markdown('<div class="section-header">Common Configurations</div>', unsafe_allow_html=True)
    
    with st.expander("🎛️ Scale Factor Configurations", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 2× Super-Resolution
            - Input: 64×64 LR
            - Output: 128×128 HR
            - Most commonly used
            - Fastest to train
            """)
        
        with col2:
            st.markdown("""
            ### 4× Super-Resolution
            - Input: 64×64 LR
            - Output: 256×256 HR
            - More challenging
            - Requires tuning
            """)
        
        with col3:
            st.markdown("""
            ### 8× Super-Resolution
            - Input: 64×64 LR
            - Output: 512×512 HR
            - Very challenging
            - May need additional layers
            """)
