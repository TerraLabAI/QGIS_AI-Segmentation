"""
AI Segmentation core modules

This package contains the core functionality for AI-powered segmentation:
- dependency_manager: Handles checking and installing required packages
- model_manager: Manages SAM model files (download, paths)
- sam_model: High-level SAM model interface
- sam_encoder: Image encoding for SAM
- sam_decoder: Mask decoding from prompts
- image_utils: Image processing utilities
- polygon_exporter: Convert masks to vector polygons
- lazy_loader: Utilities for lazy/safe imports

IMPORTANT: Modules that depend on numpy/onnxruntime (sam_model, sam_encoder,
sam_decoder, image_utils, polygon_exporter) should NOT be imported at package
level. Instead, import them lazily when needed to avoid crashes if dependencies
are not installed.

Usage:
    # Safe to import at module level (no numpy/onnxruntime dependency):
    from .core.dependency_manager import all_dependencies_installed
    from .core.model_manager import models_exist
    
    # Import lazily when needed (depends on numpy/onnxruntime):
    if all_dependencies_installed():
        from .core.sam_model import SAMModel
"""

# Only export dependency and model management - these don't require numpy/onnxruntime
# Other modules should be imported directly when needed

__all__ = [
    # Safe imports (no numpy/onnxruntime dependency)
    'dependency_manager',
    'model_manager',
    'lazy_loader',
    'cleanup',
    # Modules that require numpy/onnxruntime (import lazily)
    'sam_model',
    'sam_encoder', 
    'sam_decoder',
    'image_utils',
    'polygon_exporter',
]
