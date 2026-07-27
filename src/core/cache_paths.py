"""The plugin's on-disk cache root, and the temp dir contained inside it.

Everything the install writes (the isolated venv, the weights, the package
caches, the run history) lives under one root. It is defined here once so
``AI_SEGMENTATION_CACHE_DIR`` moves all of it together: a second copy of the
expression that forgot the variable would strand one consumer on the system
drive while the rest moved to the roomy one.

Pure stdlib, no Qt and no plugin imports, so the installer modules can import
it before anything else is available.
"""
from __future__ import annotations

import os

# Normalised at the source, so every consumer inherits one spelling. On Windows
# expanduser rejoins the tail after "~" verbatim, which yields a path with mixed
# separators; that exact string is shown in the account dialog and pasted by IT
# into an application-control or antivirus rule, where a stray separator makes
# the rule miss and the install keeps failing for the same reason.
PLUGIN_CACHE_DIR = os.path.normpath(
    os.environ.get("AI_SEGMENTATION_CACHE_DIR") or os.path.expanduser("~/.qgis_ai_segmentation")
)


def plugin_cache_tmp_dir() -> str | None:
    """Containment temp dir on the cache volume, mirroring venv_manager's
    _apply_cache_containment. A download temp file must land on the same volume
    the install uses, or a full SYSTEM drive still ENOSPCs the download even
    when AI_SEGMENTATION_CACHE_DIR points at a roomy secondary drive. Returns
    None on any failure so the caller falls back to the system default temp
    (``dir=None`` == the old behaviour)."""
    tmp_dir = os.path.join(PLUGIN_CACHE_DIR, "tmp")
    try:
        os.makedirs(tmp_dir, exist_ok=True)
        return tmp_dir
    except OSError:
        return None
