"""What the install progress line says, in words a user can act on.

The installer writes its own English lines and names its own packages. Those
belong in the log. On screen the same fact is said once, in the user's
language, and a component is named for what it does rather than by the
package it ships as.
"""
from __future__ import annotations

import re

from .i18n import tr

#: Packages whose own name says nothing to a user. Everything else keeps its
#: name: a person reading "Installing rasterio" can at least search for it.
_ENGINE_PACKAGES = frozenset({"torch", "torchvision"})
_MODEL_PACKAGES = frozenset({"sam2", "segment-anything"})

_SIZE_IN_STATUS = re.compile(r"\(([\d.]+\s*(?:kB|KB|MB|GB))\)")


def install_display_name(package_name: str) -> str:
    """The name to show for a package being installed."""
    name = (package_name or "").strip().lower()
    if name in _ENGINE_PACKAGES:
        return tr("the AI engine")
    if name in _MODEL_PACKAGES:
        return tr("the AI model")
    return package_name


def download_size_of(status_line: str) -> str:
    """The download size the installer named in its own line, or "".

    Only the size travels to the screen. The rest of the line is the
    installer's English and stays in the log.
    """
    match = _SIZE_IN_STATUS.search(status_line or "")
    return match.group(1) if match else ""
