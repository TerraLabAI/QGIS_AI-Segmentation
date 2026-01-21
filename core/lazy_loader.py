"""
Lazy Package Loader for AI Segmentation

Provides utilities to lazily import packages that may not be installed yet.
This prevents the plugin from crashing at startup if dependencies are missing.

Inspired by the approach used in the Deepness QGIS plugin.
"""

import importlib
from typing import Optional, Any


class LazyPackageLoader:
    """
    Wraps a Python package into a lazy version that loads on first use.
    
    This allows the plugin to start even if external dependencies like
    numpy or onnxruntime are not installed. The actual import happens
    only when the package is actually used.
    
    Usage:
        np = LazyPackageLoader('numpy')  # No import yet
        ...
        arr = np.array([1, 2, 3])  # Import happens here
    
    Attributes:
        _package_name: Name of the package to import
        _package: Cached reference to the imported module (None until first use)
    """
    
    def __init__(self, package_name: str):
        """
        Initialize the lazy loader.
        
        Args:
            package_name: The name of the package to lazily import
        """
        # Use object.__setattr__ to avoid triggering __getattr__
        object.__setattr__(self, '_package_name', package_name)
        object.__setattr__(self, '_package', None)
    
    def __getattr__(self, name: str) -> Any:
        """
        Get an attribute from the lazily loaded package.
        
        On first access, imports the package and caches it.
        
        Args:
            name: The attribute name to retrieve
            
        Returns:
            The requested attribute from the package
            
        Raises:
            ImportError: If the package cannot be imported
            AttributeError: If the attribute doesn't exist on the package
        """
        package = object.__getattribute__(self, '_package')
        if package is None:
            package_name = object.__getattribute__(self, '_package_name')
            package = importlib.import_module(package_name)
            object.__setattr__(self, '_package', package)
        return getattr(package, name)
    
    def __setattr__(self, name: str, value: Any):
        """Prevent setting attributes on the loader itself."""
        if name in ('_package_name', '_package'):
            object.__setattr__(self, name, value)
        else:
            package = object.__getattribute__(self, '_package')
            if package is None:
                package_name = object.__getattribute__(self, '_package_name')
                package = importlib.import_module(package_name)
                object.__setattr__(self, '_package', package)
            setattr(package, name, value)
    
    def is_available(self) -> bool:
        """
        Check if the package can be imported without actually loading it.
        
        Returns:
            True if the package is available, False otherwise
        """
        try:
            package = object.__getattribute__(self, '_package')
            if package is not None:
                return True
            package_name = object.__getattribute__(self, '_package_name')
            spec = importlib.util.find_spec(package_name)
            return spec is not None
        except Exception:
            return False
    
    def get_module(self) -> Optional[Any]:
        """
        Get the actual module, or None if not available.
        
        This forces the import if not already done.
        
        Returns:
            The imported module, or None if import fails
        """
        try:
            package = object.__getattribute__(self, '_package')
            if package is None:
                package_name = object.__getattribute__(self, '_package_name')
                package = importlib.import_module(package_name)
                object.__setattr__(self, '_package', package)
            return package
        except ImportError:
            return None


def check_package_available(package_name: str) -> bool:
    """
    Check if a package is available for import.
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        True if the package can be imported
    """
    try:
        spec = importlib.util.find_spec(package_name)
        return spec is not None
    except Exception:
        return False


def safe_import(package_name: str, fallback: Any = None) -> Any:
    """
    Safely import a package, returning a fallback if not available.
    
    Args:
        package_name: Name of the package to import
        fallback: Value to return if import fails (default: None)
        
    Returns:
        The imported module or the fallback value
    """
    try:
        return importlib.import_module(package_name)
    except ImportError:
        return fallback
