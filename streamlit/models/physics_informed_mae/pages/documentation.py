"""
Documentation page for Physics-Informed MAE
"""
import streamlit as st


def render_documentation_page(config):
    """Render documentation page for Physics-Informed MAE"""
    
    st.markdown('<div class="section-header">Model Details</div>', unsafe_allow_html=True)
    
    # Training Configuration
    with st.expander("⚙️ Training Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Model Parameters**")
            st.code(f"""
img_size: {config.img_size}
patch_size: {config.patch_size}
embed_dim: {config.embed_dim}
encoder_depth: {config.encoder_depth}
decoder_depth: {config.decoder_depth}
num_heads: {config.num_heads}
mask_ratio: {int(config.mask_ratio*100)}%
            """)
        
        with col2:
            st.markdown("**Training Parameters**")
            st.code(f"""
batch_size: {config.batch_size}
num_epochs: {config.num_epochs}
learning_rate: {config.learning_rate:.1e}
weight_decay: {config.weight_decay}
optimizer: AdamW
betas: (0.9, 0.95)
            """)
        
        with col3:
            st.markdown("**Loss Weights**")
            st.code(f"""
lambda_flux: {config.lambda_flux}

Loss Function:
L_total = L_MAE + 
         lambda_flux * L_flux
            """)
    
    # Physics-Informed Loss
    with st.expander("🔬 Physics-Informed Loss", expanded=True):
        st.markdown(f"""
        The model uses a combined loss function:
        
        **1. Reconstruction Loss (MAE)**
        - Compares reconstructed and original patches
        - Applied only to masked regions
        - Measures pixel-level accuracy
        
        **2. Flux Loss (Physics)**
        - Preserves physical quantities across edges
        - Uses edge maps from preprocessing
        - Enforces smoothness in regions of interest
        
        **Formula:**
        ```
        L_total = ||x - x̂||² + λ_flux * L_flux(E, x̂)
        ```
        
        Where:
        - `x` = original image
        - `x̂` = reconstructed image
        - `E` = edge map
        - `λ_flux` = physics loss weight ({config.lambda_flux})
        """)
    
    # Data Format
    with st.expander("📁 Data Format & Processing", expanded=True):
        st.markdown(f"""
        **Input Data:**
        - File Format: NumPy (.npy)
        - Shape: (H, W) or (C, H, W)
        - Size: {config.img_size}×{config.img_size}
        - Range: [0, 1] or normalized
        
        **Preprocessing:**
        1. Load from .npy file
        2. Convert to tensor
        3. Normalize to [-1, 1]
        4. Physics-informed preprocessing:
           - Tanh normalization
           - Edge detection via Sobel
        5. Patch embedding
        
        **Batching:**
        - Batch size: {config.batch_size}
        - Shuffled during training
        - Multiple workers: 4
        """)
    
    st.markdown('<div class="section-header">Key Innovations</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Asymmetric Design
        - Deeper encoder (12 layers)
        - Shallow decoder (8 layers)
        - Efficient encoding of information
        - Less computation during decoding
        """)
    
    with col2:
        st.markdown(f"""
        ### Masking Strategy
        - Random patch masking ({int(config.mask_ratio*100)}%)
        - [MASK] tokens for missing patches
        - Learned representations for hidden regions
        - Symmetric masking pattern
        """)
    
    with col3:
        st.markdown("""
        ### Physics Constraint
        - Preserves image gradients
        - Edge-aware reconstruction
        - Flux conservation loss
        - Applicable to scientific images
        """)
