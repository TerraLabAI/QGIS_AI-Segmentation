

import importlib
from typing import Optional, Any


class LazyPackageLoader:
    
    
    def __init__(self, package_name: str):
        
        object.__setattr__(self, '_package_name', package_name)
        object.__setattr__(self, '_package', None)
    
    def __getattr__(self, name: str) -> Any:
        
        package = object.__getattribute__(self, '_package')
        if package is None:
            package_name = object.__getattribute__(self, '_package_name')
            package = importlib.import_module(package_name)
            object.__setattr__(self, '_package', package)
        return getattr(package, name)
    
    def __setattr__(self, name: str, value: Any):
        
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
    
    try:
        spec = importlib.util.find_spec(package_name)
        return spec is not None
    except Exception:
        return False


def safe_import(package_name: str, fallback: Any = None) -> Any:
    
    try:
        return importlib.import_module(package_name)
    except ImportError:
        return fallback
