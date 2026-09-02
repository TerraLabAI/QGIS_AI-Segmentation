"""Hold the last "is the on-device model installed" answer while it stays true.

The question is asked from the dock build, from every review refresh and from
the click path, and answering it walks the whole site-packages directory. The
answer can only change when that directory changes, so it is kept until its
timestamp moves, and dropped outright when an install or a removal runs.
"""
from __future__ import annotations

import os

# (site_packages, mtime, answer). One environment at a time, so one slot is
# enough.
_cache: tuple[str, float, tuple[bool, str]] | None = None


def _stamp(site_packages: str) -> float | None:
    try:
        return os.stat(site_packages).st_mtime
    except OSError:
        return None


def cached_answer(site_packages: str) -> tuple[bool, str] | None:
    """The remembered answer for this directory, or None to go and look."""
    held = _cache
    if held is None:
        return None
    stamp = _stamp(site_packages)
    if stamp is None or held[0] != site_packages or held[1] != stamp:
        return None
    return held[2]


def remember(site_packages: str, answer: tuple[bool, str]) -> None:
    """Keep an answer against the directory's current timestamp."""
    global _cache
    stamp = _stamp(site_packages)
    _cache = None if stamp is None else (site_packages, stamp, answer)


def invalidate() -> None:
    """Forget the answer. An install or a removal just changed it."""
    global _cache
    _cache = None
