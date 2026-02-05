






from __future__ import annotations

import re

from .i18n import tr



_ENGINE_PACKAGES = frozenset({"torch", "torchvision"})
_MODEL_PACKAGES = frozenset({"sam2", "segment-anything"})

_SIZE_IN_STATUS = re.compile(r"\(([\d.]+\s*(?:kB|KB|MB|GB))\)")


def install_display_name(package_name: str) -> str:

    name = (package_name or "").strip().lower()
    if name in _ENGINE_PACKAGES:
        return tr("the AI engine")
    if name in _MODEL_PACKAGES:
        return tr("the AI model")
    return package_name


def download_size_of(status_line: str) -> str:





    match = _SIZE_IN_STATUS.search(status_line or "")
    return match.group(1) if match else ""
