"""
Home page (shared across all models)
"""
import streamlit as st


def render_home_page():
    """Render the home page"""
    st.markdown('<div class="main-header">Model Documentation Hub</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the **Physics-Informed Model Documentation Hub**!
    
    This platform provides comprehensive documentation, architecture details, and 
    interactive inference capabilities for our machine learning models.
    """)
    
    st.markdown("### How to Use")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **1️⃣ Select a Model**
        
        Use the dropdown in the sidebar to choose which model you want to explore.
        Each model has its own complete documentation.
        """)
    
    with col2:
        st.markdown("""
        **2️⃣ Browse Pages**
        
        Once a model is selected, navigate through:
        - Architecture
        - Documentation
        - Metrics
        - Examples
        """)
    
    with col3:
        st.markdown("""
        **3️⃣ Run Inference**
        
        Upload images and test the models directly in the 
        Examples page (when checkpoints are available).
        """)
    
    st.markdown("### Available Models")
    
    # Load models using file path to avoid module naming conflicts
    import importlib.util
    from pathlib import Path
    models_path = Path(__file__).parent.parent / 'models' / '__init__.py'
    spec = importlib.util.spec_from_file_location('models_loader', models_path)
    models_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models_module)
    get_available_models = models_module.get_available_models
    
    available_models = get_available_models()
    
    if not available_models:
        st.warning("No models found. Please check the models directory.")
    else:
        for model_key, model_info in available_models.items():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown(f"**{model_info['display_name']}**")
            
            with col2:
                st.markdown(f"`{model_info['config'].description}`")
    
    st.markdown("### Quick Stats")
    
    if available_models:
        # Show stats for first model as example
        first_model = next(iter(available_models.values()))
        config = first_model['config']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Image Size", f"{config.img_size}×{config.img_size}")
        
        with col2:
            st.metric("Patch Size", f"{config.patch_size}×{config.patch_size}")
        
        with col3:
            st.metric("Embedding Dim", config.embed_dim)
        
        with col4:
            st.metric("Mask Ratio", f"{int(config.mask_ratio*100)}%")
    
    st.markdown('<div class="section-header">Training Overview</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Architecture", "Training", "Loss Function", "Key Facts"])
    
    with tab1:
        st.markdown("""
        ### Physics-Informed Masked Autoencoder (MAE)
        
        **Core Pipeline:**
        1. **Preprocessing** → Tanh normalization + Sobel edge detection
        2. **Masking** → Random 75-90% patch masking (asymmetric MAE)
        3. **Encoding** → Vision Transformer with 12 layers
        4. **Decoding** → Shallow decoder with 8 layers
        5. **Reconstruction** → CNN refinement head → image reconstruction
        6. **Loss** → MSE + Flux consistency loss
        """)
        
        if available_models:
            first_model = next(iter(available_models.values()))
            config = first_model['config']
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **Model Dimensions:**
                - Input: (B, 1, {config.img_size}, {config.img_size})
                - Patches: {(config.img_size//config.patch_size)**2} total
                - Visible: {int((config.img_size//config.patch_size)**2 * (1-config.mask_ratio))} patches
                - Embedding: {config.embed_dim}-dimensional
                - Heads: {config.num_heads}
                - Head dim: {config.embed_dim // config.num_heads}
                """)
            
            with col2:
                st.markdown(f"""
                **Processing:**
                - Encoder depth: {config.encoder_depth} blocks
                - Decoder depth: {config.decoder_depth} blocks
                - Asymmetric ratio: {config.encoder_depth}/{config.decoder_depth}
                - Parameter count: ~130M
                - Estimated memory: 2.3 GB (training)
                """)
    
    with tab2:
        st.markdown("""
        ### Training Configuration
        """)
        
        if available_models:
            first_model = next(iter(available_models.values()))
            config = first_model['config']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Optimizer: AdamW**
                - Learning rate: {config.learning_rate:.2e}
                - β₁ (momentum): 0.9
                - β₂ (second moment): 0.95
                - Weight decay: {config.weight_decay}
                
                **Scheduler: Cosine Annealing**
                - Type: CosineAnnealingLR
                - Total: {config.num_epochs} epochs
                - Starts high, decays smoothly to ~0
                
                **Regularization:**
                - Gradient clipping: max_norm=1.0
                - DropPath (stochastic depth): 0.0-0.1
                """)
            
            with col2:
                st.markdown(f"""
                **Data Processing**
                - Batch size: {config.batch_size}
                - Data workers: 4
                - Device: cuda (configurable)
                - Random seed: 42
                
                **Checkpointing**
                - Save interval: every 10 epochs
                - Best model tracking: validation loss
                - Intra-epoch saves: optional
                
                **Training Loop**
                - Input: LR image → HR target
                - Loss computed: only on masked patches
                - Backward: full backpropagation
                - Updates: per batch
                """)
    
    with tab3:
        st.markdown("""
        ### Combined Loss Function
        
        **Total Loss:**
        ```
        L_total = L_recon + λ_flux · L_flux
        ```
        """)
        
        if available_models:
            first_model = next(iter(available_models.values()))
            config = first_model['config']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **1. Reconstruction Loss**
                - Type: Mean Squared Error (MSE)
                - Computed on: {int(config.mask_ratio*100)}% masked patches only
                - Formula: ||ŷ - y||²
                - Typical range: 0.05-1.0
                
                **2. Flux Loss**
                - Type: Sparse regularization
                - Formula: Σ (flux_weight)²
                - Weight: λ = {config.lambda_flux}
                - Encourages sparse attention
                
                **3. Total Loss**
                - Combined weighted sum
                - 99% reconstruction, 1% physics
                - Stable, predictable training
                """)
            
            with col2:
                st.markdown(f"""
                **Loss Characteristics:**
                
                **Reconstruction Loss:**
                - Pixel-level accuracy
                - Direct image quality
                - Primary learning signal
                - Smooth gradients
                
                **Flux Loss:**
                - Physics alignment
                - Educational component
                - Flux concentration
                - Secondary loss
                
                **Training Dynamics:**
                - Both losses decrease
                - L_recon dominates initially
                - L_flux stabilizes learning
                - Smooth convergence
                """)
    
    with tab4:
        st.markdown("""
        ### Key Facts & Innovations
        """)
        
        fact1, fact2, fact3 = st.columns(3)
        
        with fact1:
            st.markdown("""
            #### 🎭 Asymmetric MAE
            
            - Encoder: 12 blocks
            - Decoder: 8 blocks  
            - Only 25% visible
            - 4× encoder speedup
            - 30% faster overall
            
            **Why?** Information bottleneck forces learning.
            """)
        
        with fact2:
            st.markdown("""
            #### 🔬 Physics-Informed
            
            - Sobel edge detection
            - Flux guidance generation
            - Edge-aware reconstruction
            - Domain knowledge integrated
            - Astronomical images
            
            **Why?** Guides learning to relevant features.
            """)
        
        with fact3:
            st.markdown("""
            #### ⚙️ Self-Supervised
            
            - No labels required
            - Reconstruction-based learning
            - High masking ratio (75%)
            - Reconstruction of masked
            - Learns rich representations
            
            **Why?** Maximum information compression.
            """)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Activation Functions
            
            - **GELU** in MLP: Smooth, probabilistic
            - **Tanh** in preprocessing: HDR bounded
            - **Softmax** in attention: Probability
            - **ReLU** in CNN: Sparse, efficient
            """)
        
        with col2:
            st.markdown("""
            ### Layer Components
            
            - **LayerNorm:** Pre-norm for stability
            - **MHSA:** 12 heads, all-to-all
            - **Residual:** Every block, enable depth
            - **DropPath:** Stochastic depth regularization
            """)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Physics-Informed Model Documentation Hub | v0.2.0</p>
        <p><small>Built with Streamlit | Powered by PyTorch</small></p>
    </div>
    """, unsafe_allow_html=True)
