













from __future__ import annotations

import math

from .server_dials import dial_in_range, dial_str, read_value




_MAX_HEADER_CHARS = 120


def _served_positive_numbers(path: str, fallback: tuple[float, ...],
                             low: float, high: float,
                             max_len: int = 16) -> tuple[float, ...]:





    try:
        value = read_value(path)
        if not isinstance(value, (list, tuple)) or not 0 < len(value) <= max_len:
            return fallback
        out: list[float] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                return fallback
            if not math.isfinite(item) or not low <= item <= high:
                return fallback
            out.append(float(item))
        return tuple(out)
    except Exception:  # noqa: BLE001  # nosec B110
        return fallback


def _served_header_text(path: str, fallback: str) -> str:

    value = dial_str(path, fallback)
    if value is fallback:
        return fallback
    if len(value) > _MAX_HEADER_CHARS or any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        return fallback
    return value







def xyz_parallel_max(fallback: int) -> int:

    return dial_in_range("detection_policy.network.tile_fetch_parallel_max", fallback, 1, 128)


def xyz_parallel(fallback: int) -> int:

    return dial_in_range("detection_policy.network.tile_fetch_parallel", fallback,
                         1, xyz_parallel_max(128))


def xyz_attempts(fallback: int) -> int:

    return dial_in_range("network.xyz.attempts", fallback, 1, 10)


def xyz_timeout_s(fallback: float) -> float:

    return dial_in_range("network.xyz.timeout_s", fallback, 1, 60)


def xyz_backoff_s(fallback: tuple[float, ...]) -> tuple[float, ...]:

    return _served_positive_numbers("network.xyz.backoff_s", fallback, 0.01, 60.0)


def xyz_backoff_spread(fallback: float) -> float:

    return dial_in_range("network.xyz.backoff_spread", fallback, 0, 1)


def xyz_deadline_cap_s(fallback: float) -> float:

    return dial_in_range("network.xyz.deadline_cap_s", fallback, 5, 300)


def xyz_max_retry_after_s(fallback: float) -> float:

    return dial_in_range("network.xyz.max_retry_after_s", fallback, 0, 60)


def xyz_max_tiles_per_crop(fallback: int) -> int:

    return dial_in_range("network.xyz.max_tiles_per_crop", fallback, 1, 1024)


def xyz_max_tile_bytes(fallback: int) -> int:

    return dial_in_range("network.xyz.max_tile_bytes", fallback, 65536, 16777216)


def xyz_failures_before_giving_up(fallback: int) -> int:

    return dial_in_range("network.xyz.failures_before_giving_up", fallback, 1, 20)


def xyz_retry_blocked_source_after_s(fallback: float) -> float:

    return dial_in_range("network.xyz.retry_blocked_source_after_s", fallback, 10, 3600)


def xyz_cache_max_bytes(fallback: int) -> int:

    return dial_in_range("network.xyz.cache_max_bytes", fallback, 1_000_000, 1_000_000_000)


def xyz_cache_max_entries(fallback: int) -> int:

    return dial_in_range("network.xyz.cache_max_entries", fallback, 100, 100_000)


def xyz_user_agent(fallback: str) -> str:

    return _served_header_text("network.xyz.user_agent", fallback)




def api_timeout_ms(fallback: int) -> int:

    return dial_in_range("network.timeouts.api_ms", fallback, 2000, 120_000)


def interactive_timeout_ms(fallback: int) -> int:

    return dial_in_range("network.timeouts.interactive_ms", fallback, 2000, 60_000)


def poll_detection_timeout_ms(fallback: int) -> int:

    return dial_in_range("network.timeouts.poll_detection_ms", fallback, 2000, 120_000)


def translate_timeout_ms(fallback: int) -> int:

    return dial_in_range("network.timeouts.translate_ms", fallback, 2000, 60_000)


def run_export_timeout_ms(fallback: int) -> int:

    return dial_in_range("network.timeouts.run_export_ms", fallback, 5000, 300_000)




def retry_after_max_s(fallback: float) -> float:

    return dial_in_range("network.retry.after_max_s", fallback, 0, 120)


def retry_after_hint_max_s(fallback: float) -> float:

    return dial_in_range("network.retry.after_hint_max_s", fallback, 0, 3600)


def retry_pause_window_s(fallback: tuple[float, float]) -> tuple[float, float]:


    low = dial_in_range("network.retry.pause_min_s", fallback[0], 0, 10)
    high = dial_in_range("network.retry.pause_max_s", fallback[1], 0, 30)
    return (low, high) if low <= high else fallback


def server_contact_ttl_s(fallback: float) -> float:

    return dial_in_range("network.retry.server_contact_ttl_s", fallback, 10, 3600)


def window_hint_ceiling(fallback: int) -> int:



    return dial_in_range("detection_policy.network.window_hint_max", fallback, 1, 32)




def prompt_translate_max_chars(fallback: int) -> int:

    return dial_in_range("prompt.translate_max_chars", fallback, 10, 500)


def prompt_translate_max_words(fallback: int) -> int:

    return dial_in_range("prompt.translate_max_words", fallback, 1, 50)


def prompt_translate_cache_max(fallback: int) -> int:

    return dial_in_range("prompt.translate_cache_max", fallback, 10, 10_000)




def telemetry_flush_interval_s(fallback: int) -> int:

    return dial_in_range("telemetry.flush_interval_s", fallback, 5, 3600)


def telemetry_retry_backoff_s(fallback: float) -> float:

    return dial_in_range("telemetry.retry_backoff_s", fallback, 0.5, 60)


def telemetry_pending_pre_auth_max(fallback: int) -> int:

    return dial_in_range("telemetry.pending_pre_auth_max", fallback, 0, 1000)


def telemetry_batch_hard_max(fallback: int) -> int:

    return dial_in_range("telemetry.batch_hard_max", fallback, 1, 2000)


def telemetry_post_max_bytes(fallback: int) -> int:

    return dial_in_range("telemetry.post_max_bytes", fallback, 4096, 4_194_304)




def stitch_raw_fragment_retain_cap(fallback: int) -> int:

    return dial_in_range("detection_policy.network.stitch.raw_fragment_retain_cap",
                         fallback, 1000, 1_000_000)


def stitch_rescale_min_change(fallback: float) -> float:

    return dial_in_range("detection_policy.network.stitch.rescale_min_change", fallback, 0, 1)


def stitch_shape_max_wait(fallback: int) -> int:

    return dial_in_range("detection_policy.network.stitch.shape_max_wait", fallback, 0, 60)


def stitch_shape_workers(fallback: int) -> int:

    return dial_in_range("detection_policy.network.stitch.shape_workers", fallback, 0, 8)


def stitch_shape_min_batch(fallback: int) -> int:

    return dial_in_range("detection_policy.network.stitch.shape_min_batch",
                         fallback, 1, 10_000)




def aimd_cooldown_cycles(fallback: int) -> int:

    return dial_in_range("detection_policy.network.aimd.cooldown_cycles", fallback, 0, 20)


def aimd_failure_threshold(fallback: int) -> int:

    return dial_in_range("detection_policy.network.aimd.failure_threshold", fallback, 1, 20)


def convert_pool_default_max(fallback: int) -> int:

    return dial_in_range("detection_policy.network.convert.default_max", fallback, 1, 32)


def convert_pool_spare_cores(fallback: int) -> int:

    return dial_in_range("detection_policy.network.convert.spare_cores", fallback, 0, 16)


def handoff_min_delay_s(fallback: float) -> float:

    return dial_in_range("detection_policy.network.handoff_min_delay_s", fallback, 0, 30)


def handoff_open_window_max_s(fallback: float) -> float:

    return dial_in_range("detection_policy.network.handoff_open_window_max_s", fallback, 0, 60)




def catalog_ttl_s(fallback: int) -> int:

    return dial_in_range("library.catalog_ttl_s", fallback, 60, 86_400)


def catalog_neg_ttl_s(fallback: int) -> int:

    return dial_in_range("library.catalog_neg_ttl_s", fallback, 10, 3600)


def merge_compact_min_live(fallback: int) -> int:

    return dial_in_range("detection_policy.review.merge.compact_min_live", fallback, 1, 10_000)


def merge_absorbed_pool_mult(fallback: int) -> int:

    return dial_in_range("detection_policy.review.merge.absorbed_pool_mult", fallback, 1, 64)
