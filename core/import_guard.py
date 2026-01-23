import sys
import os
from typing import Tuple, Dict, Optional, Any


PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBS_DIR = os.path.join(PLUGIN_DIR, 'libs')


def verify_package_source(package_name: str, module: Optional[Any] = None) -> Tuple[bool, str]:
    """Check if package is from libs/ directory.

    Returns:
        (is_isolated, source_path)
    """
    if module is None:
        if package_name not in sys.modules:
            return False, "Not imported"
        module = sys.modules[package_name]

    try:
        if not hasattr(module, '__file__'):
            return False, "No __file__ attribute (built-in module?)"

        module_path = os.path.abspath(module.__file__)

        is_isolated = module_path.startswith(LIBS_DIR)

        return is_isolated, module_path

    except Exception as e:
        return False, f"Error checking source: {str(e)}"


def assert_package_isolated(package_name: str, module: Optional[Any] = None):
    """Raise ImportError if package is NOT from libs/.

    Use this at module level to fail-fast if isolation is broken.
    """
    is_isolated, source = verify_package_source(package_name, module)

    if not is_isolated:
        raise ImportError(
            f"Package '{package_name}' is NOT isolated!\n"
            f"Expected source: {LIBS_DIR}\n"
            f"Actual source: {source}\n"
            f"This indicates dependency isolation failed. "
            f"Please reinstall dependencies or check plugin installation."
        )


def get_isolation_report() -> Dict[str, Dict[str, Any]]:
    """Generate diagnostic report of all package sources.

    Returns dict with format:
        {
            'package_name': {
                'isolated': bool,
                'source': str
            },
            ...
        }
    """
    packages_to_check = [
        'numpy',
        'rasterio',
        'pandas',
        'torch',
        'segment_anything'
    ]

    report = {}

    for pkg_name in packages_to_check:
        is_isolated, source = verify_package_source(pkg_name)
        report[pkg_name] = {
            'isolated': is_isolated,
            'source': source
        }

    return report
