






from __future__ import annotations

from .detection_policy_core import (
    _is_finite_policy_value,
    network_policy,
)


def _net_float(
    key: str, fallback: float, policy: dict | None, high: float | None = None,
    low: float | None = None,
) -> float:















    val = network_policy(policy).get(key)
    if _is_finite_policy_value(val) and val > 0:
        if high is not None and val > high:
            return fallback
        if low is not None and val < low:
            return fallback
        return float(val)
    return fallback


def max_rate_limit_retries(fallback: int, policy: dict | None = None) -> int:

    return int(_net_float("max_rate_limit_retries", float(fallback), policy, high=50.0))


def queue_retry_budget_s(fallback: float, policy: dict | None = None) -> float:


    return _net_float("queue_retry_budget_s", fallback, policy, high=1800.0)


def midrun_offline_streak(fallback: int, policy: dict | None = None) -> int:


    return int(_net_float("midrun_offline_streak", float(fallback), policy,
                          high=100.0))


def backend_unavailable_retries(fallback: int, policy: dict | None = None) -> int:


    return int(_net_float("backend_unavailable_retries", float(fallback), policy, high=50.0))


def backend_unavailable_delay_s(fallback: float, policy: dict | None = None) -> float:


    return _net_float("backend_unavailable_delay_s", fallback, policy, high=120.0)







_STALL_TIMEOUT_FLOOR_S = 120.0


def stall_timeout_s(fallback: float, policy: dict | None = None) -> float:




    return _net_float("stall_timeout_s", fallback, policy, high=3600.0,
                      low=_STALL_TIMEOUT_FLOOR_S)


def busy_jitter(
    fallback: tuple[float, float], policy: dict | None = None
) -> tuple[float, float]:





    low = _net_float("busy_jitter_min", fallback[0], policy, high=10.0)
    high = _net_float("busy_jitter_max", fallback[1], policy, high=10.0)
    return (low, high) if low <= high else fallback


def prefetch_depth(fallback: int, policy: dict | None = None) -> int:


    return int(_net_float("prefetch_depth", float(fallback), policy, high=64.0))


def tile_fetch_parallel(fallback: int, policy: dict | None = None) -> int:



    return int(_net_float("tile_fetch_parallel", float(fallback), policy, high=64.0))


def convert_workers(fallback: int, policy: dict | None = None) -> int:



    return int(_net_float("convert_workers", float(fallback), policy, high=64.0))


def prefetch_holdoff_s(fallback: float, policy: dict | None = None) -> float:


    return _net_float("prefetch_holdoff_s", fallback, policy, high=300.0)


def slow_notice_s(fallback: float, policy: dict | None = None) -> float:



    return _net_float("slow_notice_s", fallback, policy, high=3600.0)


def render_slow_s(fallback: float, policy: dict | None = None) -> float:


    return _net_float("render_slow_s", fallback, policy, high=300.0)


def tile_render_timeout_ms(fallback: int, policy: dict | None = None) -> int:


    return int(_net_float(
        "tile_render_timeout_ms", float(fallback), policy, high=600_000.0))


def aimd_start(fallback: int, policy: dict | None = None) -> int:


    return int(_net_float("aimd_start", float(fallback), policy))


def max_consecutive_tile_fatals(fallback: int, policy: dict | None = None) -> int:


    return int(_net_float("max_consecutive_tile_fatals", float(fallback), policy))


def empty_tiles_before_notice(fallback: int, policy: dict | None = None) -> int:



    return int(_net_float("empty_tiles_before_notice", float(fallback), policy))


def render_retry_max(fallback: int, policy: dict | None = None) -> int:


    return int(_net_float("render_retry_max", float(fallback), policy))


def render_retry_delay_s(fallback: float, policy: dict | None = None) -> float:


    return _net_float("render_retry_delay_s", fallback, policy, high=120.0)


def gate_scan_render_tries(fallback: int, policy: dict | None = None) -> int:


    return int(_net_float("gate_scan_render_tries", float(fallback), policy))


def poll_interval_s(fallback: float, policy: dict | None = None) -> float:


    return _net_float("poll_interval_s", fallback, policy, high=60.0)


def poll_max_wait_s(fallback: float, policy: dict | None = None) -> float:


    return _net_float("poll_max_wait_s", fallback, policy, high=1800.0, low=5.0)


def min_poll_backoff_s(fallback: float, policy: dict | None = None) -> float:



    return _net_float("min_poll_backoff_s", fallback, policy, high=30.0)


def window_hint_max(fallback: int, policy: dict | None = None) -> int:




    return int(_net_float("window_hint_max", float(fallback), policy, high=32.0))


def aimd_min(fallback: int, policy: dict | None = None) -> int:


    return int(_net_float("aimd_min", float(fallback), policy, high=32.0))


def convert_workers_ceiling(fallback: int, policy: dict | None = None) -> int:



    return int(_net_float("convert_workers_ceiling", float(fallback), policy, high=64.0))


def convert_backlog_per_worker(fallback: int, policy: dict | None = None) -> int:



    return int(_net_float("convert_backlog_per_worker", float(fallback), policy, high=64.0))


def convert_drain_budget_s(fallback: float, policy: dict | None = None) -> float:



    return _net_float("convert_drain_budget_s", fallback, policy, high=600.0, low=5.0)


def stop_drain_budget_s(fallback: float, policy: dict | None = None) -> float:



    return _net_float("stop_drain_budget_s", fallback, policy, high=60.0, low=0.5)


def gate_render_cache_max(fallback: int, policy: dict | None = None) -> int:



    return int(_net_float("gate_render_cache_max", float(fallback), policy, high=4096.0))




_SUBMIT_TIMEOUT_FLOOR_MS = 5_000


def submit_timeout_ms(fallback: int, policy: dict | None = None) -> int:









    val = int(_net_float("submit_timeout_ms", float(fallback), policy, high=300_000.0))
    return val if val >= _SUBMIT_TIMEOUT_FLOOR_MS else fallback


def max_geojson_bytes(fallback: int, policy: dict | None = None) -> int:




    val = int(_net_float("max_geojson_bytes", float(fallback), policy))
    return val if val > 0 else fallback


def max_wkb_bytes(fallback: int, policy: dict | None = None) -> int:




    val = int(_net_float("max_wkb_bytes", float(fallback), policy))
    return val if val > 0 else fallback
