import sys
import os
from typing import Tuple, Dict, Optional, Any, List


PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBS_DIR = os.path.join(PLUGIN_DIR, 'libs')


def _get_valid_install_dirs() -> List[str]:
    """Return all valid install directories (libs/ and venv site-packages)."""
    dirs = [LIBS_DIR]
    try:
        from .venv_manager import get_venv_site_packages, get_venv_dir
        venv_sp = get_venv_site_packages(get_venv_dir())
        if venv_sp:
            dirs.append(venv_sp)
    except Exception:
        pass
    return dirs


def verify_package_source(package_name: str, module: Optional[Any] = None) -> Tuple[bool, str]:
    """Check if package is from libs/ or venv site-packages directory.

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

        module_path = os.path.normcase(os.path.abspath(module.__file__))

        for install_dir in _get_valid_install_dirs():
            if module_path.startswith(os.path.normcase(install_dir)):
                return True, module_path

        return False, module_path

    except Exception as e:
        return False, "Error checking source: {}".format(str(e))


def assert_package_isolated(package_name: str, module: Optional[Any] = None):
    """Raise ImportError if package is NOT from libs/ or venv.

    Use this at module level to fail-fast if isolation is broken.
    """
    is_isolated, source = verify_package_source(package_name, module)

    if not is_isolated:
        install_dirs = _get_valid_install_dirs()
        raise ImportError(
            "Package '{}' is NOT isolated!\n"
            "Expected source: {}\n"
            "Actual source: {}\n"
            "This indicates dependency isolation failed. "
            "Please reinstall dependencies or check plugin installation.".format(
                package_name, install_dirs, source)
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
