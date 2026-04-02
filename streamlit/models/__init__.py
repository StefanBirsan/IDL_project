"""
Model registry and management
"""
from pathlib import Path
import importlib.util
import sys


def get_available_models():
    """
    Dynamically discover and load available models
    
    Returns:
        dict: Mapping of model names to their config and pages
    """
    models_dir = Path(__file__).parent
    available_models = {}
    
    for model_folder in models_dir.iterdir():
        if model_folder.is_dir() and not model_folder.name.startswith('_'):
            try:
                # Load config module
                config_path = model_folder / 'config.py'
                spec = importlib.util.spec_from_file_location(
                    f"{model_folder.name}_config", config_path
                )
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
                
                # Load registry module  
                registry_path = model_folder / 'registry.py'
                spec = importlib.util.spec_from_file_location(
                    f"{model_folder.name}_registry", registry_path
                )
                registry_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(registry_module)
                
                # Get config and pages
                config = config_module.MODEL_CONFIG
                get_pages = registry_module.get_pages
                
                available_models[model_folder.name] = {
                    'display_name': config.name,
                    'config': config,
                    'get_pages': get_pages,
                }
                print(f"[OK] Loaded model: {model_folder.name}", flush=True)
            except Exception as e:
                print(f"[ERROR] Failed to load model {model_folder.name}:", flush=True)
                print(f"  {type(e).__name__}: {e}", flush=True)
                import traceback
                traceback.print_exc()
    
    return available_models


__all__ = ['get_available_models']
