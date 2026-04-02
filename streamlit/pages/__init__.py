"""
Shared pages for the model documentation hub
"""
from pathlib import Path
import importlib.util

# Load home module using file path
home_path = Path(__file__).parent / 'home.py'
spec = importlib.util.spec_from_file_location('home_page', home_path)
home_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(home_module)
render_home_page = home_module.render_home_page

__all__ = ['render_home_page']
