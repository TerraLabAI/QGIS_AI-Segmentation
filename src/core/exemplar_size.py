"""Pure geometry predicate for the Automatic example-size warning.

No QGIS import on purpose: kept separate from ``src/ui/plugin/exemplars.py``
(which imports ``qgis.core`` at module load) so the arithmetic can be unit
tested in a plain Python process. The warning itself is dynamic: it must be
re-checked every time the Detail slider or the example set changes, not
computed once when the example is drawn (a bigger detail level renders the
same drawn rect at more pixels, which can clear the warning).
"""
from __future__ import annotations


def exemplar_min_side_px(rect_w: float, rect_h: float, run_mupp: float) -> float:
    """Smaller side of a drawn exemplar rect, in pixels, at ``run_mupp`` ground
    units per pixel. Returns -1.0 when the resolution is unusable (<= 0): the
    caller's signal to skip this entry rather than treat 0 px as a real size.
    """
    if run_mupp <= 0:
        return -1.0
    return min(rect_w, rect_h) / run_mupp


def exemplar_too_small(rect_w: float, rect_h: float, run_mupp: float, floor: float) -> bool:
    """True when a drawn exemplar rect renders below ``floor`` px on its
    smaller side at ``run_mupp`` ground units per pixel. An unusable
    resolution (<= 0) never counts as too small: there is nothing to compare
    against yet, so the caller should treat that entry as unknown, not small.
    """
    side = exemplar_min_side_px(rect_w, rect_h, run_mupp)
    if side < 0:
        return False
    return side < floor


def exemplar_at_max_detail(current_detail: int, max_detail: int) -> bool:
    """True when the Detail slider is already at (or past) its useful
    maximum for the current layer/zone. The too-small warning must not tell
    the user to "zoom the detail slider finer" when there is no finer level
    left to move to; the caller picks its message wording from this."""
    return current_detail >= max_detail
