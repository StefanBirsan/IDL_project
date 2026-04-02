"""
Main Streamlit App - Multi-Model Documentation Hub
Refactored for easier model management and navigation
"""
import streamlit as st
import sys
from pathlib import Path
import importlib.util

# Add project root to path (for importing from training module)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load pages module using file path (avoid naming conflicts)
pages_path = Path(__file__).parent / 'pages' / '__init__.py'
spec = importlib.util.spec_from_file_location('home_pages', pages_path)
home_pages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(home_pages)
render_home_page = home_pages.render_home_page

# Import from models
from models import get_available_models

# Configure page
st.set_page_config(
    page_title="Model Documentation Hub",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #2ca02c;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .model-selector {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## 📚 Model Documentation Hub")
st.sidebar.markdown("---")

# Get available models
available_models = get_available_models()

if not available_models:
    st.error("❌ No models found in streamlit/models/")
    st.stop()

# Model selector
model_names = list(available_models.keys())
display_names = [available_models[m]['display_name'] for m in model_names]

selected_display = st.sidebar.selectbox(
    "Select Model:",
    display_names,
    help="Choose a model to view its documentation"
)

# Get the selected model key
selected_model_key = model_names[display_names.index(selected_display)]
selected_model = available_models[selected_model_key]
model_config = selected_model['config']
model_pages = selected_model['get_pages']()

st.sidebar.markdown("---")

# Navigation
if selected_model_key:
    page_names = ['Home'] + list(model_pages.keys())
    page = st.sidebar.radio(
        "Navigate:",
        page_names,
        label_visibility="collapsed"
    )
else:
    page = st.sidebar.radio(
        "Navigate:",
        ['Home'],
        disabled=True,
        label_visibility="collapsed"
    )

st.sidebar.markdown("---")

# Footer info
st.sidebar.markdown("### Model Info")
st.sidebar.info(f"""
**{selected_model['display_name']}**

{model_config.description}
""")

# Page routing
if page == 'Home':
    render_home_page()
else:
    # Remove the icon from page name to get the key
    page_key = page
    
    if page_key in model_pages:
        page_config = model_pages[page_key]
        render_func = page_config['render']
        
        # Render the page with model config
        render_func(model_config)
    else:
        st.error(f"Page not found: {page_key}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Physics-Informed Model Documentation Hub | v0.2.0</p>
</div>
""", unsafe_allow_html=True)
