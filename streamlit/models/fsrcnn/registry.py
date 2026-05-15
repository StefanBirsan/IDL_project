"""
Page registry for ESRCNN Model
Centralized page configuration and loader
"""
from pathlib import Path
import importlib.util
import sys

# Load pages module using file path
pages_path = Path(__file__).parent / 'pages' / '__init__.py'
spec = importlib.util.spec_from_file_location('fsrcnn_pages', pages_path)
pages = importlib.util.module_from_spec(spec)
sys.modules['fsrcnn_pages'] = pages
spec.loader.exec_module(pages)

render_architecture_page = pages.render_architecture_page
render_documentation_page = pages.render_documentation_page
render_examples_page = pages.render_examples_page
render_metrics_page = pages.render_metrics_page
render_live_inference_page = pages.render_live_inference_page


def get_pages():
    """
    Get all available pages for FSRCNN model
    
    Returns:
        dict: Mapping of page names to render functions and metadata
    """
    return {
        '🏗️ Architecture': {
            'render': render_architecture_page,
            'description': 'Deep residual architecture with perceptual loss',
        },
        '📖 Documentation': {
            'render': render_documentation_page,
            'description': 'Technical specifications and training guide',
        },
        '📊 Metrics': {
            'render': render_metrics_page,
            'description': 'Training metrics and performance comparison',
        },
        '🖼️ Examples': {
            'render': render_examples_page,
            'description': 'Face super-resolution examples',
        },
        '🎬 Live Demo': {
            'render': render_live_inference_page,
            'description': 'Real-time image super-resolution inference',
        },
    }


__all__ = ['get_pages']
