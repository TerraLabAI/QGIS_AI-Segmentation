"""Smoke test: verify the plugin loads inside a real QGIS Python environment."""
import importlib
import signal
import sys


def _timeout_handler(signum, frame):
    print("FAIL: smoke test timed out after 60 seconds", file=sys.stderr)
    sys.exit(1)


signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(60)

from qgis.core import QgsApplication  # noqa: E402

app = QgsApplication([], False)
app.initQgis()

try:
    plugin_parent = "/plugins"
    if plugin_parent not in sys.path:
        sys.path.insert(0, plugin_parent)

    mod = importlib.import_module("ai_seg")

    assert hasattr(mod, "classFactory"), "Plugin missing classFactory"
    assert callable(mod.classFactory), "classFactory is not callable"

    print(f"OK: ai_seg module loaded, classFactory found")
finally:
    app.exitQgis()
