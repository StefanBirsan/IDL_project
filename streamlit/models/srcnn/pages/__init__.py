"""
Model pages module for SRCNN
"""
from pathlib import Path
import importlib.util
import sys

# Load all page modules using file paths to avoid relative import issues
pages_dir = Path(__file__).parent

# Load architecture module
arch_path = pages_dir / 'architecture.py'
spec = importlib.util.spec_from_file_location('srcnn_architecture', arch_path)
arch_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arch_module)
render_architecture_page = arch_module.render_architecture_page

# Load documentation module
doc_path = pages_dir / 'documentation.py'
spec = importlib.util.spec_from_file_location('srcnn_documentation', doc_path)
doc_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(doc_module)
render_documentation_page = doc_module.render_documentation_page

# Load examples module
ex_path = pages_dir / 'examples.py'
spec = importlib.util.spec_from_file_location('srcnn_examples', ex_path)
ex_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex_module)
render_examples_page = ex_module.render_examples_page
upload_and_infer = ex_module.upload_and_infer

# Load metrics module
met_path = pages_dir / 'metrics.py'
spec = importlib.util.spec_from_file_location('srcnn_metrics', met_path)
met_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(met_module)
render_metrics_page = met_module.render_metrics_page


__all__ = ['render_architecture_page', 'render_documentation_page', 'render_examples_page', 'render_metrics_page', 'upload_and_infer']
