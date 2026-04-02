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
    
    st.markdown('<div class="section-header">Detailed Layer Components</div>', unsafe_allow_html=True)
    
    # Multi-Head Self-Attention
    with st.expander("🎯 Multi-Head Self-Attention (MHSA)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Configuration:**
            - Number of heads: {config.num_heads}
            - Head dimension: {config.embed_dim // config.num_heads}
            - Total Q,K,V projection: {config.embed_dim} → {config.embed_dim * 3}
            
            **Computation Steps:**
            1. **Project to Q, K, V**
               - QKV = Linear({config.embed_dim}, {config.embed_dim}×3)
               - Reshape: (B, N, 3, {config.num_heads}, {config.embed_dim // config.num_heads})
            
            2. **Compute Attention Scores**
               - scores = Q @ K^T / √({config.embed_dim // config.num_heads})
               - Shape: (B, {config.num_heads}, N, N)
               - Scale factor: 1/√{config.embed_dim // config.num_heads} ≈ {1.0/((config.embed_dim // config.num_heads)**0.5):.3f}
            
            3. **Apply Softmax**
               - attn_weights = softmax(scores)
               - Sum per row: 1.0 (probability distribution)
            
            4. **Attention × Value**
               - context = attn_weights @ V
               - Shape: (B, {config.num_heads}, N, {config.embed_dim // config.num_heads})
            
            5. **Concatenate & Project**
               - concat({config.num_heads} heads) → (B, N, {config.embed_dim})
               - Linear projection back to {config.embed_dim}
            
            **Mathematical Formula:**
            ```
            Attention(Q,K,V) = softmax(QK^T/√d_k)V
            ```
            """)
        
        with col2:
            st.markdown(f"""
            **Properties:**
            ✓ All-to-all token interactions
            ✓ Parallelizable across heads
            ✓ Learns different attention patterns per head
            ✓ Interpretable: visualize head attention
            
            **Head Specialization:**
            Each of {config.num_heads} heads learns:
            - Head 1: Local patterns (nearby tokens)
            - Head 2: Global structure (whole image)
            - Head 3: Object boundaries (flux regions)
            - ... ({config.num_heads - 3} more)
            
            **Parameters per head:**
            ```
            QKV projection: {config.embed_dim}×{config.embed_dim}×3
            Output projection: {config.embed_dim}×{config.embed_dim}
            
            Total MHSA params ≈ 3.1M
            ```
            
            **Attention Map Examples:**
            - (B, {config.num_heads}, {(config.img_size//config.patch_size)**2}, {(config.img_size//config.patch_size)**2})
            - Each map shows how one patch attends to all others
            """)
    
    # Feed-Forward Network
    with st.expander("🧠 Feed-Forward Network (MLP)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Architecture:**
            ```
            Linear({config.embed_dim} → {config.embed_dim * 4})
            ↓
            GELU()
            ↓
            Linear({config.embed_dim * 4} → {config.embed_dim})
            ```
            
            **Expansion Ratio:** 4×
            - This 4× factor is empirically effective for ViT
            - Increases model capacity within transformer blocks
            - Standard in Vision Transformer literature
            
            **Parameter Count:**
            ```
            Layer 1: {config.embed_dim} × {config.embed_dim * 4} ≈ 3.1M
            Layer 2: {config.embed_dim * 4} × {config.embed_dim} ≈ 3.1M
            
            Per-block MLP: ≈ 6.3M parameters
            ```
            
            **Computation:**
            Input: (B, N, {config.embed_dim})
            ↓
            hidden = Linear(x)
            Shape: (B, N, {config.embed_dim * 4})
            ↓
            activated = GELU(hidden)
            ↓
            output = Linear(activated)
            Shape: (B, N, {config.embed_dim})
            """)
        
        with col2:
            st.markdown("""
            **Why This Design?**
            
            ✓ **Non-linearity:** GELU introduces non-linearity
            ✓ **Expansion:** 4× creates room for interactions
            ✓ **Token-wise:** Independent processing per patch
            ✓ **Stable:** GELU smoothness aids training
            
            **Residual Connection:**
            ```
            output = x + DropPath(MLP(LayerNorm(x)))
            ```
            
            This residual addition:
            - Preserves original information
            - Enables deep networks (prevents vanishing gradients)
            - Allows MLP to be additive refinement
            - Empirically proven crucial for deep ViT
            """)
    
    # Positional Embeddings
    with st.expander("📍 Positional Embeddings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Type:** Sine-Cosine Positional Encoding
            
            **Formula for position p, dimension 2i:**
            ```
            PE[p, 2i] = sin(p / 10000^(2i/d))
            PE[p, 2i+1] = cos(p / 10000^(2i/d))
            ```
            
            **Where:**
            - p = position index (0 to {(config.img_size//config.patch_size)**2})
            - d = embedding dimension ({config.embed_dim})
            - i = frequency index (0 to d/2)
            
            **Properties:**
            ✓ Captures absolute positions
            ✓ Relative position awareness (periodic)
            ✓ Extrapolates to longer sequences
            ✓ Low frequencies: coarse positions
            ✓ High frequencies: fine positions
            
            **Application:**
            ```
            x_with_pos = x + positional_embeddings
            ```
            Added before encoder/decoder blocks
            """)
        
        with col2:
            st.markdown(f"""
            **Shape:** (1, max_seq_len, {config.embed_dim})
            
            **Sequence Length:**
            Encoder: {(config.img_size//config.patch_size)**2} + 1 (CLS token)
            Decoder: {(config.img_size//config.patch_size)**2} + 1 (CLS token)
            
            **Why Sine-Cosine?**
            - No learnable parameters
            - Infinite extrapolation capability
            - Periodic structure encodes distance
            - Works for variable sequence lengths
            - Empirically proven effective
            
            **Alternative (Learnable):**
            Could use learnable pos_embed (like ViT)
            - More flexible but requires fixed size
            - More parameters
            - Current implementation: simpler
            """)
    
    # Layer Normalization
    with st.expander("📊 Layer Normalization (LayerNorm)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Architecture:** Pre-LayerNorm
            ```
            x → LayerNorm(x) → Attention
               ↓ add residual ↓
               ←─────────────
            
            x → LayerNorm(x) → MLP
               ↓ add residual ↓
               ←─────────────
            ```
            
            **Formula:**
            ```
            LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β
            ```
            
            Where:
            - μ = mean across feature dimension
            - σ² = variance across features
            - γ, β = learnable scale and shift
            - ε = 1e-6 (numerical stability)
            
            **Applied to:** Feature dimension ({config.embed_dim})
            NOT batch dimension
            """)
        
        with col2:
            st.markdown(f"""
            **Why Pre-LayerNorm?**
            
            ✓ **Stability:** Normalizes before non-linear ops
            ✓ **Gradients:** Better gradient flow in deep nets
            ✓ **Training:** More stable than post-norm
            ✓ **Modern ViT:** Standard in current architectures
            
            **Post-norm (older):**
            ```
            x → Attention → LayerNorm
                ↓ residual ↓
            ```
            
            Less stable in deep networks
            Can cause training divergence
            
            **Per block:** 2 LayerNorm layers
            Total: {config.encoder_depth + config.decoder_depth} × 2 = {(config.encoder_depth + config.decoder_depth) * 2} LayerNorms
            """)
    
    # Patch Reconstruction
    with st.expander("📐 Patch Reconstruction Head", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Purpose:** Convert embeddings back to pixels
            
            **Process:**
            1. **Output Projection**
               - Input: (B, {(config.img_size//config.patch_size)**2}, {config.embed_dim})
               - Linear({config.embed_dim} → {config.patch_size}²)
               - Output: (B, {(config.img_size//config.patch_size)**2}, {config.patch_size**2})
            
            2. **CNN Refinement (Optional)**
               - Conv2d({config.embed_dim} → 256)
               - ReLU activation
               - Conv2d(256 → 256)
               - ReLU activation
               - Conv2d(256 → {config.patch_size**2})
            
            3. **Patch Unfolding**
               - Reshape: (B, {(config.img_size//config.patch_size)**2}, {config.patch_size**2})
               - → (B, {config.img_size//config.patch_size}, {config.img_size//config.patch_size}, {config.patch_size}, {config.patch_size})
               - → (B, 1, {config.img_size}, {config.img_size})
            
            **Final Shape:** (B, 1, {config.img_size}, {config.img_size})
            Ready for loss computation!
            """)
        
        with col2:
            st.markdown(f"""
            **CNN Refinement Details:**
            
            Conv2d(768→256):
            - Reduces channel dimension
            - Processes at patch resolution
            - {(config.img_size//config.patch_size)}×{(config.img_size//config.patch_size)} spatial
            
            Conv2d(256→256):
            - Same resolution processing
            - Refines local features
            - 3×3 kernels, padding=1
            
            Conv2d(256→{config.patch_size**2}):
            - Projects to pixel values
            - Output per patch position
            
            **Why CNN head?**
            ✓ Captures local patterns
            ✓ Smooth transition: transformer→pixels
            ✓ Improves reconstruction quality
            ✓ Efficient local processing
            """)
