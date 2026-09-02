"""Served dials for the transport layer: tile fetching, the API client, prompt
translation, telemetry, the streaming worker, the catalogue cache, the venv
scan and the merger's housekeeping.

Every getter here takes the shipped constant as its ``fallback`` and returns
it whenever the cache is empty, the key is absent, or the served value is out
of its band. Read a dial at call time (or once per run or object), never at
import time: the cache fills after import. Cache-only, no network, safe on
the GUI thread and from a worker thread. Pure Python, no Qt at import time.

A guard travels with the value it bounds, so a ceiling lives in the same
section as the number it caps. No getter here clamps a served value against a
shipped constant: the band is the dial's own range.
"""
from __future__ import annotations

import math

from .server_dials import dial_in_range, dial_str, read_value

# -- shared shapes -----------------------------------------------------------

# A served string that ends up in a request header: one line, printable.
_MAX_HEADER_CHARS = 120


def _served_positive_numbers(path: str, fallback: tuple[float, ...],
                             low: float, high: float,
                             max_len: int = 16) -> tuple[float, ...]:
    """A served list of finite numbers, each inside ``[low, high]``.

    The whole list is refused when it is empty, too long, or when any entry is
    not a number in band, so a ladder is never half replaced.
    """
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
    except Exception:  # noqa: BLE001 -- config is best-effort  # nosec B110
        return fallback


def _served_header_text(path: str, fallback: str) -> str:
    """A served one-line header value, or the shipped one."""
    value = dial_str(path, fallback)
    if value is fallback:
        return fallback
    if len(value) > _MAX_HEADER_CHARS or any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        return fallback
    return value


# -- network.xyz: basemap tiles fetched straight from the source --------------
# The pool width itself is already served as
# ``detection_policy.network.tile_fetch_parallel`` and applied once per run;
# its ceiling lives beside it so the two never disagree.

def xyz_parallel_max(fallback: int) -> int:
    """Widest tile pool a served width may open."""
    return dial_in_range("detection_policy.network.tile_fetch_parallel_max", fallback, 1, 128)


def xyz_parallel(fallback: int) -> int:
    """Default tile pool width before a run names one. Same key as the run's."""
    return dial_in_range("detection_policy.network.tile_fetch_parallel", fallback,
                         1, xyz_parallel_max(128))


def xyz_attempts(fallback: int) -> int:
    """How many times one tile is asked for."""
    return dial_in_range("network.xyz.attempts", fallback, 1, 10)


def xyz_timeout_s(fallback: float) -> float:
    """Seconds one tile request may take."""
    return dial_in_range("network.xyz.timeout_s", fallback, 1, 60)


def xyz_backoff_s(fallback: tuple[float, ...]) -> tuple[float, ...]:
    """Base wait per attempt, in seconds, before a tile is asked for again."""
    return _served_positive_numbers("network.xyz.backoff_s", fallback, 0.01, 60.0)


def xyz_backoff_spread(fallback: float) -> float:
    """Random spread added on top of a back-off wait, as a fraction."""
    return dial_in_range("network.xyz.backoff_spread", fallback, 0, 1)


def xyz_deadline_cap_s(fallback: float) -> float:
    """Longest one crop may spend fetching its tiles, in seconds."""
    return dial_in_range("network.xyz.deadline_cap_s", fallback, 5, 300)


def xyz_max_retry_after_s(fallback: float) -> float:
    """Longest wait a tile host may name that one crop still obeys."""
    return dial_in_range("network.xyz.max_retry_after_s", fallback, 0, 60)


def xyz_max_tiles_per_crop(fallback: int) -> int:
    """Most tiles one crop may fetch before it zooms out or gives up."""
    return dial_in_range("network.xyz.max_tiles_per_crop", fallback, 1, 1024)


def xyz_max_tile_bytes(fallback: int) -> int:
    """Largest tile body read off the wire."""
    return dial_in_range("network.xyz.max_tile_bytes", fallback, 65536, 16777216)


def xyz_failures_before_giving_up(fallback: int) -> int:
    """Failed crops on one source before the direct path is paused."""
    return dial_in_range("network.xyz.failures_before_giving_up", fallback, 1, 20)


def xyz_retry_blocked_source_after_s(fallback: float) -> float:
    """Seconds a paused source waits before the direct path is tried again."""
    return dial_in_range("network.xyz.retry_blocked_source_after_s", fallback, 10, 3600)


def xyz_cache_max_bytes(fallback: int) -> int:
    """Memory the session tile cache may hold."""
    return dial_in_range("network.xyz.cache_max_bytes", fallback, 1_000_000, 1_000_000_000)


def xyz_cache_max_entries(fallback: int) -> int:
    """Tiles the session tile cache may hold."""
    return dial_in_range("network.xyz.cache_max_entries", fallback, 100, 100_000)


def xyz_user_agent(fallback: str) -> str:
    """User-Agent sent to a tile host."""
    return _served_header_text("network.xyz.user_agent", fallback)


# -- network.timeouts: the API client's per-route deadlines, in ms ----------

def api_timeout_ms(fallback: int) -> int:
    """Default deadline for an ordinary API round trip."""
    return dial_in_range("network.timeouts.api_ms", fallback, 2000, 120_000)


def interactive_timeout_ms(fallback: int) -> int:
    """Deadline for the small GETs a screen waits on (config, usage, account)."""
    return dial_in_range("network.timeouts.interactive_ms", fallback, 2000, 60_000)


def poll_detection_timeout_ms(fallback: int) -> int:
    """Deadline for one detection status poll."""
    return dial_in_range("network.timeouts.poll_detection_ms", fallback, 2000, 120_000)


def translate_timeout_ms(fallback: int) -> int:
    """Deadline for a prompt translation, which a Detect waits on."""
    return dial_in_range("network.timeouts.translate_ms", fallback, 2000, 60_000)


def run_export_timeout_ms(fallback: int) -> int:
    """Deadline for the final-output upload."""
    return dial_in_range("network.timeouts.run_export_ms", fallback, 5000, 300_000)


# -- network.retry: the API client's one retry and its hints -----------------

def retry_after_max_s(fallback: float) -> float:
    """Longest wait a rate limiter may name that one blocking read obeys."""
    return dial_in_range("network.retry.after_max_s", fallback, 0, 120)


def retry_after_hint_max_s(fallback: float) -> float:
    """Ceiling on a parsed Retry-After before it travels to whoever paces."""
    return dial_in_range("network.retry.after_hint_max_s", fallback, 0, 3600)


def retry_pause_window_s(fallback: tuple[float, float]) -> tuple[float, float]:
    """``(min, max)`` seconds the one retry is spread over. Served as two keys
    in the same section; a crossed pair falls back whole."""
    low = dial_in_range("network.retry.pause_min_s", fallback[0], 0, 10)
    high = dial_in_range("network.retry.pause_max_s", fallback[1], 0, 30)
    return (low, high) if low <= high else fallback


def server_contact_ttl_s(fallback: float) -> float:
    """Seconds a real answer keeps the link judged alive."""
    return dial_in_range("network.retry.server_contact_ttl_s", fallback, 10, 3600)


def window_hint_ceiling(fallback: int) -> int:
    """Widest in-flight width a per-answer header may name. Same key the
    worker reads (``detection_policy.network.window_hint_max``), same band,
    so the client's parse and the worker's clamp cannot disagree."""
    return dial_in_range("detection_policy.network.window_hint_max", fallback, 1, 32)


# -- prompt: the translation lookup ------------------------------------------

def prompt_translate_max_chars(fallback: int) -> int:
    """Longest prompt sent for translation."""
    return dial_in_range("prompt.translate_max_chars", fallback, 10, 500)


def prompt_translate_max_words(fallback: int) -> int:
    """Most words a prompt sent for translation may hold."""
    return dial_in_range("prompt.translate_max_words", fallback, 1, 50)


def prompt_translate_cache_max(fallback: int) -> int:
    """Translations kept on disk."""
    return dial_in_range("prompt.translate_cache_max", fallback, 10, 10_000)


# -- telemetry: batching and delivery ----------------------------------------

def telemetry_flush_interval_s(fallback: int) -> int:
    """Seconds between two batch flushes."""
    return dial_in_range("telemetry.flush_interval_s", fallback, 5, 3600)


def telemetry_retry_backoff_s(fallback: float) -> float:
    """Wait before the one retry of a failed batch."""
    return dial_in_range("telemetry.retry_backoff_s", fallback, 0.5, 60)


def telemetry_pending_pre_auth_max(fallback: int) -> int:
    """Events kept before the account is known."""
    return dial_in_range("telemetry.pending_pre_auth_max", fallback, 0, 1000)


def telemetry_batch_hard_max(fallback: int) -> int:
    """Events one batch may hold before the oldest are dropped."""
    return dial_in_range("telemetry.batch_hard_max", fallback, 1, 2000)


def telemetry_post_max_bytes(fallback: int) -> int:
    """Body size at which a batch is split over several posts."""
    return dial_in_range("telemetry.post_max_bytes", fallback, 4096, 4_194_304)


# -- detection_policy.network.stitch: the live stitch thread -----------------

def stitch_raw_fragment_retain_cap(fallback: int) -> int:
    """Raw fragments kept for the review override before the list is freed."""
    return dial_in_range("detection_policy.network.stitch.raw_fragment_retain_cap",
                         fallback, 1000, 1_000_000)


def stitch_rescale_min_change(fallback: float) -> float:
    """Relative pixel-size change under which the stitch keeps its scale."""
    return dial_in_range("detection_policy.network.stitch.rescale_min_change", fallback, 0, 1)


def stitch_shape_max_wait(fallback: int) -> int:
    """Cycles a growing object waits before it is shaped anyway."""
    return dial_in_range("detection_policy.network.stitch.shape_max_wait", fallback, 0, 60)


def stitch_shape_workers(fallback: int) -> int:
    """Threads the stitcher spreads one cycle's shape pass over. 0 = in place."""
    return dial_in_range("detection_policy.network.stitch.shape_workers", fallback, 0, 8)


def stitch_shape_min_batch(fallback: int) -> int:
    """Objects a cycle must carry before its shape pass is worth spreading."""
    return dial_in_range("detection_policy.network.stitch.shape_min_batch",
                         fallback, 1, 10_000)


# -- detection_policy.network.aimd / convert / handoff: the run worker -------

def aimd_cooldown_cycles(fallback: int) -> int:
    """Clean cycles after a setback before the window grows again."""
    return dial_in_range("detection_policy.network.aimd.cooldown_cycles", fallback, 0, 20)


def aimd_failure_threshold(fallback: int) -> int:
    """Hard connectivity failures in a row that end a run early."""
    return dial_in_range("detection_policy.network.aimd.failure_threshold", fallback, 1, 20)


def convert_pool_default_max(fallback: int) -> int:
    """Most converter threads when the pool sizes itself from the machine."""
    return dial_in_range("detection_policy.network.convert.default_max", fallback, 1, 32)


def convert_pool_spare_cores(fallback: int) -> int:
    """Cores the self-sized converter pool leaves free."""
    return dial_in_range("detection_policy.network.convert.spare_cores", fallback, 0, 16)


def handoff_min_delay_s(fallback: float) -> float:
    """Floor under a wait the service names on a 429/503."""
    return dial_in_range("detection_policy.network.handoff_min_delay_s", fallback, 0, 30)


def handoff_open_window_max_s(fallback: float) -> float:
    """Longest named wait still read as capacity arriving, not overload."""
    return dial_in_range("detection_policy.network.handoff_open_window_max_s", fallback, 0, 60)


# -- library / install / review.merge -----------------------------------------

def catalog_ttl_s(fallback: int) -> int:
    """Seconds a fetched catalogue stays fresh."""
    return dial_in_range("library.catalog_ttl_s", fallback, 60, 86_400)


def catalog_neg_ttl_s(fallback: int) -> int:
    """Seconds a failed catalogue fetch is not retried."""
    return dial_in_range("library.catalog_neg_ttl_s", fallback, 10, 3600)


def foreign_venv_keep_days(fallback: int) -> int:
    """Days a venv written to by another window is left alone."""
    return dial_in_range("install.foreign_venv_keep_days", fallback, 1, 3650)


def merge_compact_min_live(fallback: int) -> int:
    """Live objects under which the merger never rebuilds its index."""
    return dial_in_range("detection_policy.review.merge.compact_min_live", fallback, 1, 10_000)


def merge_absorbed_pool_mult(fallback: int) -> int:
    """Absorbed children one keeper may hold, as a multiple of the minimum."""
    return dial_in_range("detection_policy.review.merge.absorbed_pool_mult", fallback, 1, 64)
