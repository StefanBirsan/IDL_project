"""
Physics-Informed MAE Model Documentation Module
"""
from .config import MODEL_CONFIG, PhysicsInformedMAEConfig
from .registry import get_pages

__all__ = ['MODEL_CONFIG', 'PhysicsInformedMAEConfig', 'get_pages']
