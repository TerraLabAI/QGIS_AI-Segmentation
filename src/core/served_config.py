

















from __future__ import annotations

import functools
import math
import time
from itertools import islice
from typing import Any, Callable, Iterable




REQUIRED_CONFIG_SCHEMA = 2



_MAX_SERVED_LIST_ENTRIES = 64
_MAX_SERVED_ENTRY_CHARS = 128

_POLICY_PREFIX = "detection_policy."







REQUIRED_SERVED_KEYS: dict[str, str] = {
    "detection_policy.auto_regularize.min_angle_split_deg": "number",
    "detection_policy.exemplar.context_pad": "number",
    "detection_policy.exemplar.context_pad_px": "number",
    "detection_policy.exemplar.min_paste_scale": "number",
    "detection_policy.gate.group": "number",
    "detection_policy.gate.max_group": "number",
    "detection_policy.gate.min_pixels": "number",
    "detection_policy.gate.min_tiles": "number",
    "detection_policy.gate.prefilter.band_eps": "number",
    "detection_policy.review.boundary_snap.keywords": "list",
    "detection_policy.review.boundary_snap.max_area_change": "number",
    "detection_policy.review.boundary_snap.max_objects": "number",
    "detection_policy.review.boundary_snap.min_keep_share": "number",
    "detection_policy.review.boundary_snap.tolerance_m": "number",
    "detection_policy.review.canopy_hint_tokens": "list",
    "detection_policy.review.closed_canopy_advice.max_raw_per_tile": "number",
    "detection_policy.review.closed_canopy_advice.min_span_dropped": "number",
    "detection_policy.review.closed_canopy_advice.min_tiles": "number",
    "detection_policy.review.merge.cover_threshold": "number",
    "detection_policy.review.merge.dedup_ios": "number",
    "detection_policy.review.merge.dup_centroid_frac": "number",
    "detection_policy.review.merge.dup_ios_floor": "number",
    "detection_policy.review.merge.ios_threshold": "number",
    "detection_policy.review.merge.jitter_area_frac": "number",
    "detection_policy.review.merge.jitter_erode_px": "number",
    "detection_policy.review.merge.map_likeness_min_share": "number",
    "detection_policy.review.merge.merge_ios": "number",
    "detection_policy.review.merge.part_cover_frac": "number",
    "detection_policy.review.merge.part_inside": "number",
    "detection_policy.review.merge.part_max_frac": "number",
    "detection_policy.review.merge.part_min_children": "number",
    "detection_policy.review.merge.part_sibling_ios": "number",
    "detection_policy.review.merge.score_floor_frac": "number",
    "detection_policy.review.merge.seam_span_ios": "number",
    "detection_policy.review.merge.seam_span_tol": "number",
    "detection_policy.seed.detail_coarse_travel_ratio": "number",
    "detection_policy.seed.detail_fine_travel_ratio": "number",
    "detection_policy.seed.drawn_object_tile_frac": "number",
    "detection_policy.seed.max_object_tile_frac": "number",
    "detection_policy.seed.object_min_px": "number",
    "detection_policy.seed.recall_floor": "number",
    "detection_policy.seed.recall_floor_exemplar_only": "number",
    "detection_policy.seed.saturation.compact_min_fill": "number",
    "detection_policy.seed.saturation.hard_tile_coverage": "number",
    "detection_policy.seed.saturation.max_masks_per_tile": "number",
    "detection_policy.seed.saturation.max_tile_coverage": "number",
    "detection_policy.seed.saturation.resplit_time_ratio": "number",
    "detection_policy.seed.saturation.subdiv_max_depth": "number",
    "detection_policy.seed.saturation.subdivide_min_parent_px": "number",
    "detection_policy.seed.saturation.subdivide_overlap_fraction": "number",
    "detection_policy.seed.split_risk_tile_frac": "number",
    "detection_policy.seed.sweet_spot_max_mupp": "number",
    "detection_policy.seed.tile_plan.slider_half_steps": "number",
    "detection_policy.seed.tile_plan.slider_step_ratio": "number",
    "detection_policy.seed.zone_seed_mupp": "number",
    "tuning.review.flat_score_tolerance": "number",
}




_RUN_ONLY_PREFIXES = (
    "detection_policy.auto_regularize.",
    "detection_policy.gate.",
    "detection_policy.review.boundary_snap.",
    "detection_policy.review.closed_canopy_advice.",
    "detection_policy.review.merge.",
    "detection_policy.seed.recall_floor",
    "detection_policy.seed.saturation.",
)


REQUIRED_CONFIG_KEYS: dict[str, str] = {
    key: kind for key, kind in REQUIRED_SERVED_KEYS.items()
    if not key.startswith(_RUN_ONLY_PREFIXES)
}


REQUIRED_RUN_KEYS: dict[str, str] = {
    key: kind for key, kind in REQUIRED_SERVED_KEYS.items()
    if key.startswith(_RUN_ONLY_PREFIXES)
}


class ServedConfigMissing(Exception):


    def __init__(self, key: str):
        super().__init__(key)
        self.key = key


def _served_config_root() -> dict:
    try:
        from .config_cache import get_config

        config = get_config()
    except Exception:  # noqa: BLE001  # nosec B110
        return {}
    return config if isinstance(config, dict) else {}


def _served_lookup(key: str) -> Any:


    try:
        if key.startswith(_POLICY_PREFIX):
            from .detection_policy_core import get_detection_policy

            value: Any = get_detection_policy()
            parts = key[len(_POLICY_PREFIX):].split(".")
        else:
            value = _served_config_root()
            parts = key.split(".")
        for part in parts:
            if not isinstance(value, dict):
                return None
            value = value.get(part)
    except Exception:  # noqa: BLE001  # nosec B110
        return None
    return value


def _is_finite_served_number(value: Any) -> bool:
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def require_served_number(key: str, low: float, high: float) -> float:

    value = _served_lookup(key)
    if not _is_finite_served_number(value) or not low <= value <= high:
        raise ServedConfigMissing(key)
    return float(value)


def require_served_int(key: str, low: int, high: int) -> int:


    value = _served_lookup(key)
    if not _is_finite_served_number(value) or float(value) != int(value) or not low <= value <= high:
        raise ServedConfigMissing(key)
    return int(value)


def require_served_bool(key: str) -> bool:

    value = _served_lookup(key)
    if not isinstance(value, bool):
        raise ServedConfigMissing(key)
    return value


def require_served_str(key: str, allowed: Iterable[str] | None = None) -> str:

    value = _served_lookup(key)
    if not isinstance(value, str) or len(value) > _MAX_SERVED_ENTRY_CHARS * 8:
        raise ServedConfigMissing(key)
    value = value.strip()
    if not value or (allowed is not None and value not in allowed):
        raise ServedConfigMissing(key)
    return value


def require_served_list(key: str) -> tuple[str, ...]:


    value = _served_lookup(key)
    if not isinstance(value, (list, tuple)):
        raise ServedConfigMissing(key)
    entries = []
    for item in islice(value, _MAX_SERVED_LIST_ENTRIES):
        if isinstance(item, str) and len(item) <= _MAX_SERVED_ENTRY_CHARS and item.strip():
            entries.append(item.strip())
    return tuple(entries)


def require_served_table(key: str, validate: Callable[[dict], bool] | None = None) -> dict:



    value = _served_lookup(key)
    if not isinstance(value, dict):
        raise ServedConfigMissing(key)
    try:
        if validate is not None and not validate(value):
            raise ServedConfigMissing(key)
    except ServedConfigMissing:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ServedConfigMissing(key) from exc
    return value


def served_config_schema() -> int:

    value = _served_config_root().get("config_schema")
    if (_is_finite_served_number(value) and value >= 0
            and float(value) == int(value)):
        return int(value)
    return 0


def served_config_state() -> str:


    try:
        from .config_cache import SOURCE_DISK, SOURCE_LIVE, config_source

        source = config_source()
    except Exception:  # noqa: BLE001  # nosec B110
        return "missing"
    if not _served_config_root():
        return "missing"
    if source == SOURCE_LIVE:
        return "live"
    if source == SOURCE_DISK:
        return "cached"
    return "missing"


def served_config_age_s() -> float | None:

    try:
        from .config_cache import config_fetched_at

        fetched_at = config_fetched_at()
    except Exception:  # noqa: BLE001  # nosec B110
        return None
    if fetched_at is None:
        return None
    return max(0.0, time.time() - fetched_at)


def _served_kind_matches(kind: str, value: Any) -> bool:
    if kind == "number":
        return _is_finite_served_number(value)
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "list":
        return isinstance(value, (list, tuple))
    return value is not None


def _missing_keys(root: Any, keys: dict[str, str], strip: str = "") -> list[str]:
    gaps: list[str] = []
    for key, kind in keys.items():
        value: Any = root
        for part in key[len(strip):].split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if not _served_kind_matches(kind, value):
            gaps.append(key)
    return gaps


def missing_served_values(config: Any) -> tuple[str, ...]:





    if not isinstance(config, dict):
        return ("config_schema", *REQUIRED_CONFIG_KEYS)
    gaps: list[str] = []
    schema = config.get("config_schema")
    if (not _is_finite_served_number(schema) or schema < REQUIRED_CONFIG_SCHEMA
            or float(schema) != int(schema)):
        gaps.append("config_schema")
    gaps.extend(_missing_keys(config, REQUIRED_CONFIG_KEYS))
    return tuple(gaps)


def missing_run_policy_values(policy: Any) -> tuple[str, ...]:




    keys = {key: kind for key, kind in REQUIRED_SERVED_KEYS.items()
            if key.startswith(_POLICY_PREFIX)}
    if not isinstance(policy, dict):
        return tuple(keys)
    return tuple(_missing_keys(policy, keys, _POLICY_PREFIX))




_completeness_memo: tuple[Any, bool] = (None, False)


def served_config_ready() -> bool:

    global _completeness_memo
    if served_config_state() == "missing":
        return False
    config = _served_config_root()
    memo_config, memo_complete = _completeness_memo
    if memo_config is config:
        return memo_complete
    complete = not missing_served_values(config)
    _completeness_memo = (config, complete)
    return complete


def quiet_without_served_config(func: Callable[..., Any]) -> Callable[..., Any]:




    @functools.wraps(func)
    def _quiet(*args: Any, **kwargs: Any) -> Any:
        if not served_config_ready():
            return None
        try:
            return func(*args, **kwargs)
        except ServedConfigMissing as err:
            try:
                from qgis.core import Qgis, QgsMessageLog

                QgsMessageLog.logMessage(
                    f"{func.__name__}: served setting missing ({err.key})",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
            except Exception:  # noqa: BLE001  # nosec B110
                pass
            return None
    return _quiet
