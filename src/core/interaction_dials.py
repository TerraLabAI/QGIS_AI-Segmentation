"""Server dials for the interactive surfaces: the restore path, the hover
preview, the review's reslice and rescue budgets, the run watchdogs, the
Correct step, the crop encode lock and the install wait.

Each getter takes the module constant its caller ships as the fallback, bounds
the served value, and fails open to that constant. Everything here is
cache-only and pure Python: no network, no disk, no Qt at import time, so the
GUI thread and a worker can both call it. Read a dial at call time, never at
import time, because the cache fills after import.
"""
from __future__ import annotations

from .server_dials import dial_in_range, dial_url, read_value

# Restore ---------------------------------------------------------------------


def restore_confidence_floor(fallback: float) -> float:
    """A stored run threshold at or under this is the recall floor, not the
    cutoff the user reviewed at, so the restore opens at the default instead."""
    return dial_in_range("detection_policy.review.restore_confidence_floor",
                         fallback, 0.0, 1.0)


def restore_align_max_objects(fallback: int) -> int:
    """How many objects a restore runs the footprint alignment pass over."""
    return dial_in_range("detection_policy.auto_regularize.restore_max_objects",
                         fallback, 0, 100000)


def restore_align_budget_s(fallback: float) -> float:
    """Wall clock the restore's alignment pass may spend."""
    return dial_in_range("detection_policy.auto_regularize.restore_budget_s",
                         fallback, 0.0, 60.0)


# Hover preview --------------------------------------------------------------


def hover_recall_one_object_coverage(fallback: float) -> float:
    """Share of the crop under which a remembered mask counts as one object."""
    return dial_in_range("network.hover.recall_one_object_coverage",
                         fallback, 0.0, 1.0)


def hover_shape_max_coverage(fallback: float) -> float:
    """Share of the crop a preview mask may cover and still be shaped."""
    return dial_in_range("network.hover.shape_max_coverage", fallback, 0.0, 1.0)


def hover_refusal_quiet_s(fallback: float) -> float:
    """How long the preview loop stands down after one refusal."""
    return dial_in_range("network.hover.refusal_quiet_s", fallback, 0.0, 600.0)


def hover_shape_budget_ms(fallback: float) -> float:
    """The most one preview answer may spend shaping its mask on the GUI thread."""
    return dial_in_range("network.hover.shape_budget_ms", fallback, 50.0, 10000.0)


def route_memo_ms(fallback: float) -> float:
    """How long the route gates (served switch, account store, auth header) are
    answered from the last look. One key for the hover preview and the review's
    AI fix route, so a sign-out stops both in the same breath."""
    return dial_in_range("network.route_memo_ms", fallback, 0.0, 60000.0)


# Live preview and the finalize pump -----------------------------------------


def live_repaint_ms(fallback: int) -> int:
    """How often the live preview asks the canvas to redraw."""
    return dial_in_range("ui.live_repaint_ms", fallback, 50, 5000)


def live_frame_cost_ratio(fallback: float) -> float:
    """How many times its own frame cost the live preview sits out between frames."""
    return dial_in_range("ui.live_frame_cost_ratio", fallback, 1.0, 20.0)


def live_repaint_max_ms(fallback: int) -> int:
    """However slow one frame gets, the preview still repaints this often."""
    return dial_in_range("ui.live_repaint_max_ms", fallback, 500, 60000)


def auto_pump_budget_s(fallback: float) -> float:
    """Seconds one finalize or reslice turn works before yielding to the loop."""
    return dial_in_range("ui.auto_pump_budget_s", fallback, 0.001, 1.0)


# Review ---------------------------------------------------------------------


def reslice_screen_first_min_objects(fallback: int) -> int:
    """Below this many objects a reslice is not partitioned by what is on screen."""
    return dial_in_range("detection_policy.review.reslice_screen_first_min_objects",
                         fallback, 0, 100000)


def review_reslice_parked_keys_max(fallback: int) -> int:
    """How many superseded shape keys keep their refined geometry."""
    return dial_in_range("ui.review.reslice_parked_keys_max", fallback, 1, 20)


def review_reslice_parked_geoms_max(fallback: int) -> int:
    """How many geometries all parked keys together may hold."""
    return dial_in_range("ui.review.reslice_parked_geoms_max", fallback, 100, 1000000)


def rescue_refine_budget_s(fallback: float) -> float:
    """How long the safety-net export may spend shaping hidden objects."""
    return dial_in_range("ui.review.rescue_refine_budget_s", fallback, 0.0, 60.0)


def live_refiner_memo_max(fallback: int) -> int:
    """How many run refiners the review keeps alive at once."""
    return dial_in_range("ui.review.live_refiner_memo_max", fallback, 1, 64)


def ground_scale_band_deg(fallback: float) -> float:
    """Latitude span, in degrees, inside which one refiner's ground dials hold."""
    return dial_in_range("export_policy.ground_scale_band_deg", fallback, 0.01, 10.0)


def ground_scale_band_m(fallback: float) -> float:
    """Northing span, in Mercator metres, inside which one refiner's ground dials hold."""
    return dial_in_range("export_policy.ground_scale_band_m", fallback, 100.0, 1000000.0)


# Run watchdogs --------------------------------------------------------------


def lost_terminal_grace_s(fallback: float) -> float:
    """Seconds after a worker exits within which its terminal must arrive."""
    return dial_in_range("detection_policy.network.lost_terminal_grace_s",
                         fallback, 0.0, 300.0)


def cancel_watchdog_ms(fallback: int) -> int:
    """How long a cooperative stop may go unconfirmed before the UI leaves the run."""
    return dial_in_range("detection_policy.network.cancel_watchdog_ms",
                         fallback, 500, 60000)


# Correct step ---------------------------------------------------------------


def correct_fold_look_ms(fallback: int) -> int:
    """Pause between two looks at the fold's reslice before a session re-opens."""
    return dial_in_range("network.correct.fold_look_ms", fallback, 10, 1000)


def correct_fold_max_looks(fallback: int) -> int:
    """How many looks a method switch waits for the fold's reslice."""
    return dial_in_range("network.correct.fold_max_looks", fallback, 1, 1000)


def confirm_reset_ms(fallback: int) -> int:
    """How long an armed two-stage confirm stays armed. One key for the retry
    link and the Clear all link."""
    return dial_in_range("ui.confirm_reset_ms", fallback, 1000, 30000)


# Crop encode lock -----------------------------------------------------------

# Seconds the lock ceiling keeps above the served click transport wait, so a
# served raise of the wait can never strand a lock that is still legitimately
# held by a slow answer.
_ENCODE_CEILING_MARGIN_S = 60.0


def _served_click_wait_s() -> float:
    """The served click transport wait in seconds, or 0.0 when none is served."""
    value = read_value("detection_policy.network.click_wait_max_ms")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value) / 1000.0
    return 0.0


def encode_watchdog_interval_ms(fallback: int) -> int:
    """Beat interval of the transport-lock watchdog."""
    return dial_in_range("network.encode_watchdog_interval_ms", fallback, 500, 60000)


def encode_lock_ceiling_s(fallback: float) -> float:
    """Seconds past which a held encode lock counts as stranded. Never below
    the served click wait plus a margin, whatever the dial says."""
    ceiling = dial_in_range("network.encode_lock_ceiling_s", fallback, 30.0, 900.0)
    return max(float(ceiling), _served_click_wait_s() + _ENCODE_CEILING_MARGIN_S)


# Install and warm-up --------------------------------------------------------


def install_pipe_wait_turns(fallback: int) -> int:
    """How many turns the install waits for the engine pipe to go idle."""
    return dial_in_range("install.pipe_wait_turns", fallback, 1, 60)


def warm_recent_manual_days(fallback: int) -> int:
    """Days since the last Semi-Auto session within which the engine pre-warms."""
    return dial_in_range("ui.warm_recent_manual_days", fallback, 0, 365)


def vcredist_url(fallback: str) -> str:
    """Where the Windows runtime download lives."""
    return dial_url("install.vcredist_url", fallback)
