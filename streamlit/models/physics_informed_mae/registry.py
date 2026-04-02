"""
Page registry for Physics-Informed MAE Model
Centralized page configuration and loader
"""
from pathlib import Path
import importlib.util
import sys

# Load pages module using file path
pages_path = Path(__file__).parent / 'pages' / '__init__.py'
spec = importlib.util.spec_from_file_location('pages', pages_path)
pages = importlib.util.module_from_spec(spec)
sys.modules['pages'] = pages
spec.loader.exec_module(pages)

render_architecture_page = pages.render_architecture_page
render_documentation_page = pages.render_documentation_page
render_examples_page = pages.render_examples_page
render_metrics_page = pages.render_metrics_page


def get_pages():
    """
    Get all available pages for this model
    
    Returns:
        dict: Mapping of page names to render functions and metadata
    """
    return {
        '🏗️ Architecture': {
            'render': render_architecture_page,
            'description': 'Detailed model architecture breakdown',
        },
        '📖 Documentation': {
            'render': render_documentation_page,
            'description': 'Technical specifications and configuration',
        },
        '📊 Metrics': {
            'render': render_metrics_page,
            'description': 'Training metrics and performance tracking',
        },
        '🖼️ Examples': {
            'render': render_examples_page,
            'description': 'Inference examples and visualizations',
        },
    }


__all__ = ['get_pages']
