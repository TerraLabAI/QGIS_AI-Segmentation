
from __future__ import annotations

import math


def confidence_value(value) -> float | None:

    if value is None or isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return round(score, 3) if math.isfinite(score) and 0.0 <= score <= 1.0 else None
