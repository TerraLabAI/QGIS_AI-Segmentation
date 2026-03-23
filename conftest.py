"""Root conftest: make tests/ importable without QGIS runtime.

The root __init__.py does a relative import from .src.core.venv_manager
which fails outside QGIS.  We pre-register the root package as a plain
namespace so that `import src.shared.constants` works without loading
the real __init__.py.
"""
import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

# 1. Mock qgis modules
for mod_name in ("qgis", "qgis.core", "qgis.gui", "qgis.PyQt",
                 "qgis.PyQt.QtCore", "qgis.PyQt.QtWidgets",
                 "qgis.PyQt.QtGui"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# 2. Make `src` importable directly (without going through root __init__.py)
plugin_dir = Path(__file__).parent
src_dir = plugin_dir / "src"
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

# Pre-register src as a simple namespace package
src_mod = types.ModuleType("src")
src_mod.__path__ = [str(src_dir)]
src_mod.__package__ = "src"
sys.modules["src"] = src_mod

# Pre-register src.shared
shared_dir = src_dir / "shared"
shared_mod = types.ModuleType("src.shared")
shared_mod.__path__ = [str(shared_dir)]
shared_mod.__package__ = "src.shared"
sys.modules["src.shared"] = shared_mod

# Tell pytest to ignore root __init__.py
collect_ignore = ["__init__.py"]
