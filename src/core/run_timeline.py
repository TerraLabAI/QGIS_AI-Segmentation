











from __future__ import annotations

import os
import threading
import time

_ENV = "AI_SEGMENTATION_RUN_TIMELINE"

_t0: float | None = None
_marks: list = []
_once: set = set()


def enabled() -> bool:
    return os.environ.get(_ENV, "") == "1"


def begin(name: str = "detect_click") -> None:

    global _t0, _marks, _once
    if not enabled():
        return
    _t0 = time.perf_counter()
    _marks = []
    _once = set()
    _marks.append((0.0, name, threading.current_thread().name))


def mark(name: str) -> None:

    t0 = _t0
    if t0 is None or not enabled():
        return
    _marks.append((time.perf_counter() - t0, name, threading.current_thread().name))


def mark_once(name: str) -> None:

    if _t0 is None or name in _once or not enabled():
        return
    _once.add(name)
    mark(name)


def marks() -> list:

    return sorted(_marks)


def summary_line() -> str:

    seen: dict = {}
    for at, name, _thread in marks():
        seen.setdefault(name, at)
    return ", ".join(f"{name}={at:.3f}" for name, at in seen.items())
