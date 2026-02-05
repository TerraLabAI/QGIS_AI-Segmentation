






from __future__ import annotations

import os



_cache: tuple[str, tuple[int, int, int], tuple[bool, str]] | None = None


def _stamp(site_packages: str) -> tuple[int, int, int] | None:
    try:
        info = os.stat(site_packages)
        return info.st_mtime_ns, info.st_ino, info.st_dev
    except OSError:
        return None


def cached_answer(site_packages: str) -> tuple[bool, str] | None:

    held = _cache
    if held is None:
        return None
    stamp = _stamp(site_packages)
    if stamp is None or held[0] != site_packages or held[1] != stamp:
        return None
    return held[2]


def remember(site_packages: str, answer: tuple[bool, str]) -> None:

    global _cache
    stamp = _stamp(site_packages)
    _cache = None if stamp is None or not answer[0] else (site_packages, stamp, answer)


def invalidate() -> None:

    global _cache
    _cache = None
