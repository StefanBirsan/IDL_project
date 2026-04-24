"""ESRCNN Pages Module"""
from .architecture import render_architecture_page
from .documentation import render_documentation_page
from .examples import render_examples_page
from .metrics import render_metrics_page

__all__ = [
    'render_architecture_page',
    'render_documentation_page',
    'render_examples_page',
    'render_metrics_page',
]
