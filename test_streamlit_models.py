"""Test streamlit models loading"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import from local streamlit folder
import importlib.util
models_path = project_root / 'streamlit' / 'models' / '__init__.py'
spec = importlib.util.spec_from_file_location('models_loader', models_path)
models_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models_module)
get_available_models = models_module.get_available_models

# Get available models
models = get_available_models()

print(f"✓ Available models: {list(models.keys())}")
print()

for key, info in models.items():
    print(f"Model: {key}")
    print(f"  Display Name: {info['display_name']}")
    print(f"  Description: {info['config'].description}")
    print(f"  Pages: {list(info['get_pages']().keys())}")
    print()

print(f"✓ All {len(models)} models loaded successfully!")
