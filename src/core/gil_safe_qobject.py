




























from __future__ import annotations


def _noop(*_args) -> None:
    return None


def prime(obj):

    if obj is None:
        return obj
    try:
        signal = obj.destroyed
        signal.connect(_noop)
        signal.disconnect(_noop)
    except (AttributeError, RuntimeError, TypeError):
        pass  # nosec B110
    return obj


__all__ = ["prime"]
