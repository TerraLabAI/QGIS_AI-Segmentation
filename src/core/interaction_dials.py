









from __future__ import annotations

from .server_dials import dial_in_range, dial_url




def restore_confidence_floor(fallback: float) -> float:


    return dial_in_range("detection_policy.review.restore_confidence_floor",
                         fallback, 0.0, 1.0)


def restore_align_max_objects(fallback: int) -> int:

    return dial_in_range("detection_policy.auto_regularize.restore_max_objects",
                         fallback, 0, 100000)


def restore_align_budget_s(fallback: float) -> float:

    return dial_in_range("detection_policy.auto_regularize.restore_budget_s",
                         fallback, 0.0, 60.0)





def hover_recall_one_object_coverage(fallback: float) -> float:

    return dial_in_range("network.hover.recall_one_object_coverage",
                         fallback, 0.0, 1.0)


def hover_shape_max_coverage(fallback: float) -> float:

    return dial_in_range("network.hover.shape_max_coverage", fallback, 0.0, 1.0)


def hover_refusal_quiet_s(fallback: float) -> float:

    return dial_in_range("network.hover.refusal_quiet_s", fallback, 0.0, 600.0)


def hover_transient_quiet_s(fallback: float) -> float:






    return dial_in_range("network.hover.transient_quiet_s", fallback, 0.0, 30.0)


def hover_shape_budget_ms(fallback: float) -> float:

    return dial_in_range("network.hover.shape_budget_ms", fallback, 50.0, 10000.0)


def route_memo_ms(fallback: float) -> float:



    return dial_in_range("network.route_memo_ms", fallback, 0.0, 60000.0)





def live_repaint_ms(fallback: int) -> int:

    return dial_in_range("ui.live_repaint_ms", fallback, 50, 5000)


def live_frame_cost_ratio(fallback: float) -> float:

    return dial_in_range("ui.live_frame_cost_ratio", fallback, 1.0, 20.0)


def live_repaint_max_ms(fallback: int) -> int:

    return dial_in_range("ui.live_repaint_max_ms", fallback, 500, 60000)


def auto_pump_budget_s(fallback: float) -> float:

    return dial_in_range("ui.auto_pump_budget_s", fallback, 0.001, 1.0)





def reslice_screen_first_min_objects(fallback: int) -> int:

    return dial_in_range("detection_policy.review.reslice_screen_first_min_objects",
                         fallback, 0, 100000)


def review_reslice_parked_keys_max(fallback: int) -> int:

    return dial_in_range("ui.review.reslice_parked_keys_max", fallback, 1, 20)


def review_reslice_parked_geoms_max(fallback: int) -> int:

    return dial_in_range("ui.review.reslice_parked_geoms_max", fallback, 100, 1000000)


def rescue_refine_budget_s(fallback: float) -> float:

    return dial_in_range("ui.review.rescue_refine_budget_s", fallback, 0.0, 60.0)


def live_refiner_memo_max(fallback: int) -> int:

    return dial_in_range("ui.review.live_refiner_memo_max", fallback, 1, 64)


def ground_scale_band_deg(fallback: float) -> float:

    return dial_in_range("export_policy.ground_scale_band_deg", fallback, 0.01, 10.0)


def ground_scale_band_m(fallback: float) -> float:

    return dial_in_range("export_policy.ground_scale_band_m", fallback, 100.0, 1000000.0)





def lost_terminal_grace_s(fallback: float) -> float:

    return dial_in_range("detection_policy.network.lost_terminal_grace_s",
                         fallback, 0.0, 300.0)


def cancel_watchdog_ms(fallback: int) -> int:

    return dial_in_range("detection_policy.network.cancel_watchdog_ms",
                         fallback, 500, 60000)





def correct_fold_look_ms(fallback: int) -> int:

    return dial_in_range("network.correct.fold_look_ms", fallback, 10, 1000)


def correct_fold_max_looks(fallback: int) -> int:

    return dial_in_range("network.correct.fold_max_looks", fallback, 1, 1000)


def confirm_reset_ms(fallback: int) -> int:


    return dial_in_range("ui.confirm_reset_ms", fallback, 1000, 30000)







_ENCODE_CEILING_MARGIN_S = 60.0


def _served_click_wait_s() -> float:

    return dial_in_range("detection_policy.network.click_wait_max_ms", 0.0, 5000.0, 300_000.0) / 1000.0


def encode_watchdog_interval_ms(fallback: int) -> int:

    return dial_in_range("network.encode_watchdog_interval_ms", fallback, 500, 60000)


def encode_lock_ceiling_s(fallback: float) -> float:


    ceiling = dial_in_range("network.encode_lock_ceiling_s", fallback, 30.0, 900.0)
    return max(float(ceiling), _served_click_wait_s() + _ENCODE_CEILING_MARGIN_S)





def install_pipe_wait_turns(fallback: int) -> int:

    return dial_in_range("install.pipe_wait_turns", fallback, 1, 60)


def warm_recent_manual_days(fallback: int) -> int:

    return dial_in_range("ui.warm_recent_manual_days", fallback, 0, 365)


def vcredist_url(fallback: str) -> str:

    return dial_url("install.vcredist_url", fallback)
