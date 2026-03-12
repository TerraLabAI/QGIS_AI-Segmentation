"""Smoke test: verify the plugin loads inside a real QGIS Python environment."""
import sys
from unittest.mock import MagicMock

from qgis.core import QgsApplication

app = QgsApplication([], False)
app.initQgis()

try:
    plugin_parent = "/plugins"
    if plugin_parent not in sys.path:
        sys.path.insert(0, plugin_parent)

    import ai_seg  # noqa: E402

    mock_iface = MagicMock()
    plugin = ai_seg.classFactory(mock_iface)

    assert hasattr(plugin, "initGui"), "Plugin missing initGui method"
    assert hasattr(plugin, "unload"), "Plugin missing unload method"

    print(f"OK: classFactory returned {type(plugin).__name__} with initGui and unload")
finally:
    app.exitQgis()
