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
    
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Physics-Informed Model Documentation Hub | v0.2.0</p>
        <p><small>Built with Streamlit | Powered by PyTorch</small></p>
    </div>
    """, unsafe_allow_html=True)
