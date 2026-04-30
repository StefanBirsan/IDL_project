"""
Home page (shared across all models)
"""
import streamlit as st


def render_home_page():
    """Render the home page"""
    st.markdown('<div class="main-header">Model Documentation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This page documents the architecture, training, metrics and results of our super-resolution models. Instead of making boring slides, we made a Streamlit app :)
    """)
    
    st.markdown("### Usage")

    st.markdown("""
    For each model, you can view, wherever applicable:
    - Architecture
    - Documentation
    - Metrics
    - Examples
    """)
    
    st.markdown("### Our models so far")
    
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