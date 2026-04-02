"""
Architecture documentation page for Physics-Informed MAE
"""
import streamlit as st


def render_architecture_page(config):
    """Render architecture page for Physics-Informed MAE"""
    
    st.markdown('<div class="section-header">System Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("""
        ### Physics-Informed MAE Pipeline
        
        The model consists of several key components:
        
        1. **Input Image** → Physics-Informed Preprocessing
        2. **Normalization & Edge Detection** → Patch Embedding
        3. **Vision Transformer Encoder** → Masking
        4. **Vision Transformer Decoder** → Reconstruction
        5. **Output Image** ← Loss Computation (MAE + Physics Loss)
        """)
    
    with col2:
        st.info(f"""
        **Model Configuration**
        - Image Size: {config.img_size}×{config.img_size}
        - Patch Size: {config.patch_size}×{config.patch_size}
        - Embedding: {config.embed_dim}-dim
        - Heads: {config.num_heads}
        - Mask Ratio: {int(config.mask_ratio*100)}%
        """)
    
    st.markdown('<div class="section-header">Component Architecture</div>', unsafe_allow_html=True)
    
    # Physics-Informed Preprocessing
    with st.expander("🔬 Physics-Informed Preprocessing", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Purpose:** Extract physical features from input images
            
            **Operations:**
            - **Tanh Normalization**: Handles HDR dynamic range
            - **Sobel Filtering**: Double differentiation for edges
            - **Edge Detection**: Computes gradient magnitude
            
            **Output:** 
            - Normalized image (B, C, H, W)
            - Edge map (B, C, H, W)
            """)
        
        with col2:
            st.code("""
# Input: x (B, C, H, W)
x_normalized = tanh(x)
gx = Conv2D(x, sobel_x)
gy = Conv2D(x, sobel_y)
edge_map = sqrt(gx² + gy²)
# Output: normalized image, edge map
            """, language="python")
    
    # Patch Embedding
    with st.expander("🧩 Patch Embedding", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Purpose:** Convert images to patches and embed them
            
            **Process:**
            - Partition image into non-overlapping patches
            - Flatten each patch to 1D vector
            - Project to embedding dimension ({config.embed_dim}-dim)
            - Apply layer normalization
            
            **Dimensions:**
            - Input: (B, C, {config.img_size}, {config.img_size})
            - Patches: {config.img_size//config.patch_size}×{config.img_size//config.patch_size} = {(config.img_size//config.patch_size)**2} patches
            - Output: (B, {(config.img_size//config.patch_size)**2}, {config.embed_dim})
            """)
        
        with col2:
            st.code(f"""
# Reshape to patches
B, C, H, W = x.shape
patch_size = {config.patch_size}
patches = x.reshape(
    B, C*patch_size², H//patch_size, W//patch_size
)
patches = patches.reshape(B, -1, C*patch_size²)

# Embed patches
embeddings = Linear(C*{config.patch_size**2}, {config.embed_dim})(patches)
embeddings = LayerNorm(embeddings)
            """, language="python")
    
    # Encoder
    with st.expander("🔢 Vision Transformer Encoder", expanded=True):
        st.markdown(f"""
        **Architecture:** {config.encoder_depth}-layer Transformer with self-attention
        
        **Configuration for each layer:**
        - Multi-Head Attention: {config.num_heads} heads
        - Hidden Dimension: {config.embed_dim * 4} (4× embedding dim)
        - Activation: GELU
        - Layer Normalization: Pre-norm
        
        **Input:** (B, {(config.img_size//config.patch_size)**2}, {config.embed_dim}) embedded patches
        **Output:** (B, {(config.img_size//config.patch_size)**2}, {config.embed_dim}) encoded features
        """)
    
    # Decoder
    with st.expander("🔄 Vision Transformer Decoder", expanded=True):
        st.markdown(f"""
        **Architecture:** {config.decoder_depth}-layer Transformer (shallow decoder)
        
        **Key aspects:**
        - Uses masked patch tokens for unobserved regions
        - Encoder output tokens prepended to decoder tokens
        - Asymmetric design: {config.encoder_depth}-layer encoder, {config.decoder_depth}-layer decoder
        - Learnable [MASK] tokens for hidden patches
        
        **Process:**
        1. Receive encoder output
        2. Add masked patch embeddings
        3. Add positional encoding
        4. Apply {config.decoder_depth} transformer layers
        5. Project to pixel reconstruction
        """)
