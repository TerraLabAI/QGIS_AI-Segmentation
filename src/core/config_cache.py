




































from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import threading
import time
from typing import NamedTuple

from .cache_paths import PLUGIN_CACHE_DIR
from .file_replace_retry import replace_file_with_retry

CONFIG_FILENAME = "server_config.json"



_MAX_BYTES = 2 * 1024 * 1024



_FILE_SOURCE = "live_fetch"



SOURCE_NONE = "none"
SOURCE_DISK = "disk"
SOURCE_LIVE = "live"


class _Snapshot(NamedTuple):


    config: dict
    fetched_at: float | None
    source: str




    etag: str | None = None




    lossless: bool = False


    raw: dict | None = None


_EMPTY = _Snapshot({}, None, SOURCE_NONE)



_ETAG_MAX_CHARS = 300


def _sanitize_etag(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > _ETAG_MAX_CHARS:
        return None


    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        return None
    return value


def _parse_config_json(payload) -> object:

    def reject_constant(value):
        raise ValueError(f"Non-finite configuration number: {value}")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Configuration number is outside the finite range")
        return number

    return json.loads(payload, parse_constant=reject_constant, parse_float=finite_float)




_state: _Snapshot = _EMPTY

_override: dict | None = None
_override_state = {"loaded": False}
_publish_lock = threading.RLock()





def config_cache_path() -> str:

    return os.path.join(PLUGIN_CACHE_DIR, CONFIG_FILENAME)


def save_config(config: dict, etag: str | None = None) -> bool:










    if not isinstance(config, dict) or not config:
        return False
    path = config_cache_path()
    tmp_path = None
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload_obj = {"source": _FILE_SOURCE, "fetched_at": time.time(), "config": config}
        clean_etag = _sanitize_etag(etag)
        if clean_etag:
            payload_obj["etag"] = clean_etag
        payload = json.dumps(payload_obj, allow_nan=False)
        if len(payload.encode("utf-8")) > _MAX_BYTES:
            return False
        handle, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(path), prefix=".server_config-", suffix=".tmp"
        )
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(payload)


        replace_file_with_retry(tmp_path, path)
    except Exception:  # noqa: BLE001  # nosec B110
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass  # nosec B110
        return False
    return True








_FAIL_CLOSED_FEATURES = frozenset({
    "crop_webp",
    "gzip_request_bodies",
    "hover_preview",
    "hover_preview_click_reuse",
    "map_hypothesis_nms",
    "tree_self_exemplar",
})


def _without_kill_switches(config: dict) -> dict:









    out = dict(config)
    if out.get("automatic_mode_enabled") is not True:
        out.pop("automatic_mode_enabled", None)
    features = out.get("features")
    if isinstance(features, dict):
        kept = {name: on for name, on in features.items()
                if on is True and name not in _FAIL_CLOSED_FEATURES}
        if len(kept) != len(features):
            out["features"] = kept
    elif features is not None:


        out.pop("features", None)
    return out






_INSTALL_KEYS_THAT_RUN_CODE = ("packages", "torch_index_url", "checkpoint")
_INSTALL_SUBKEYS_THAT_RUN_CODE = {


    "python": ("release_tag",),
    "uv": ("version",),
}


def _without_code_execution_dials(config: dict) -> dict:


    install = config.get("install")
    if not isinstance(install, dict):
        return config
    out = dict(config)
    safe = {k: v for k, v in install.items() if k not in _INSTALL_KEYS_THAT_RUN_CODE}
    for parent, subkeys in _INSTALL_SUBKEYS_THAT_RUN_CODE.items():
        child = safe.get(parent)
        if isinstance(child, dict):
            safe[parent] = {k: v for k, v in child.items() if k not in subkeys}
    out["install"] = safe
    return out


def _load_parsed() -> tuple[dict, float | None, str | None]:







    path = config_cache_path()
    try:
        if not os.path.isfile(path) or os.path.getsize(path) > _MAX_BYTES:
            return {}, None, None
        with open(path, "rb") as fh:
            payload = fh.read(_MAX_BYTES + 1)
        if len(payload) > _MAX_BYTES:
            return {}, None, None

        data = _parse_config_json(payload)
    except Exception:  # noqa: BLE001  # nosec B110
        return {}, None, None
    if not isinstance(data, dict) or data.get("source") != _FILE_SOURCE:
        return {}, None, None
    config = data.get("config")
    if not isinstance(config, dict) or not config:
        return {}, None, None
    fetched_at = data.get("fetched_at")
    if isinstance(fetched_at, int) and fetched_at.bit_length() > 64:
        fetched_at = None
    if (not isinstance(fetched_at, (int, float)) or isinstance(fetched_at, bool)
            or not math.isfinite(fetched_at) or not 0 <= fetched_at <= time.time()):
        fetched_at = None
    etag = _sanitize_etag(data.get("etag"))
    return config, fetched_at, etag








_RUN_ONLY_TOP_KEYS = ("run_decisions",)
_RUN_ONLY_POLICY_DROPS = {
    "seed": ("object_tiers", "exemplar_size_ladder"),
    "review": ("class_keywords", "category_to_class", "min_size_m2",
               "adaptive_off_prompts", "land_cover"),
}

_RUN_ONLY_POLICY_EMPTIED = {
    ("review", "regularize", "keywords"): list,
    ("review", "boundary_snap", "keywords"): list,
    ("review", "fp_filter"): dict,
    ("gate", "class_map"): list,
    ("gate", "classes"): dict,
    ("auto_regularize", "classes"): list,
}


def _with_emptied_table(block: object, path: tuple, kind: type) -> object:


    if not isinstance(block, dict) or path[0] not in block:
        return block
    if len(path) == 1:
        return block if block[path[0]] == kind() else {**block, path[0]: kind()}
    child = _with_emptied_table(block[path[0]], path[1:], kind)
    return block if child is block[path[0]] else {**block, path[0]: child}


def _without_run_tables(config: dict) -> dict:


    out = {k: v for k, v in config.items() if k not in _RUN_ONLY_TOP_KEYS}
    policy = out.get("detection_policy")
    if not isinstance(policy, dict):
        return out
    policy = dict(policy)
    for section, keys in _RUN_ONLY_POLICY_DROPS.items():
        block = policy.get(section)
        if isinstance(block, dict) and any(k in block for k in keys):
            policy[section] = {k: v for k, v in block.items() if k not in keys}
    review = policy.get("review")
    if isinstance(review, dict) and isinstance(review.get("class_settings"), dict):
        settings = review["class_settings"]
        if any(k != "default" for k in settings):
            policy["review"] = dict(review, class_settings={
                k: v for k, v in settings.items() if k == "default"})
    for path, kind in _RUN_ONLY_POLICY_EMPTIED.items():
        policy = _with_emptied_table(policy, path, kind)
    out["detection_policy"] = policy
    return out






_DISK_POLICY_MAX_AGE_DAYS = 30.0


def _disk_policy_max_age_s(config: dict) -> float:
    tuning = config.get("tuning")
    section = tuning.get("config") if isinstance(tuning, dict) else None
    days = section.get("disk_policy_max_age_days") if isinstance(section, dict) else None
    if (not isinstance(days, (int, float)) or isinstance(days, bool)
            or not math.isfinite(days) or not 1 <= days <= 365):
        days = _DISK_POLICY_MAX_AGE_DAYS
    return float(days) * 86400.0


def _without_aged_policy(config: dict, fetched_at: float | None) -> dict:


    if "detection_policy" not in config and "run_decisions" not in config:
        return config
    if fetched_at is not None and time.time() - fetched_at <= _disk_policy_max_age_s(config):
        return config
    return {k: v for k, v in config.items() if k not in ("detection_policy", "run_decisions")}


def _strip_read_back(config: dict) -> dict:
    return _without_run_tables(
        _without_code_execution_dials(_without_kill_switches(config)))


def load_config() -> tuple[dict, float | None, str | None]:









    config, fetched_at, etag = _load_parsed()
    if not config:
        return {}, None, None
    return _without_aged_policy(_strip_read_back(config), fetched_at), fetched_at, etag


def clear_config() -> None:






    global _state, _override
    with _publish_lock:
        _state = _Snapshot({}, None, SOURCE_NONE)
        _override = None
        _override_state["loaded"] = False
        try:
            os.unlink(config_cache_path())
        except Exception:  # noqa: BLE001  # nosec B110
            pass


def forget_account_policy() -> None:







    global _state
    with _publish_lock:
        state = _state
        drop = ("detection_policy",) + _RUN_ONLY_TOP_KEYS
        if not any(k in state.config for k in drop) and not (
                isinstance(state.raw, dict) and any(k in state.raw for k in drop)):
            return
        config = {k: v for k, v in state.config.items() if k not in drop}
        raw = ({k: v for k, v in state.raw.items() if k not in drop}
               if isinstance(state.raw, dict) else None)
        _state = state._replace(config=config, raw=raw, etag=None, lossless=False)
        try:
            os.unlink(config_cache_path())
        except Exception:  # noqa: BLE001  # nosec B110
            pass





def override_path() -> str:

    plugin_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(plugin_root, ".debug", "dev_policy.json")


def get_override() -> dict:







    with _publish_lock:
        return _load_override_once()


def _load_override_once() -> dict:
    global _override
    if _override_state["loaded"]:
        return _override or {}
    try:
        path = override_path()
        if os.path.isfile(path):



            with open(path, encoding="utf-8-sig") as fh:
                raw = fh.read(_MAX_BYTES + 1)
            data = _parse_config_json(raw) if len(raw) <= _MAX_BYTES else None
            if isinstance(data, dict):
                _override = data
    except Exception:  # noqa: BLE001  # nosec B110
        _override = None
    _override_state["loaded"] = True
    return _override or {}


def deep_merge(base: dict, override: dict) -> dict:

    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = val
    return out





def _resolve_with_override(config: dict) -> dict:

    override = get_override()
    return deep_merge(config, override) if override else config


def prime_from_disk() -> bool:










    global _state
    initial = _state
    if initial.source != SOURCE_NONE:
        return False
    try:
        raw, fetched_at, etag = _load_parsed()
        config = _without_aged_policy(_strip_read_back(raw), fetched_at)


        lossless = bool(raw) and config == raw
        resolved = _resolve_with_override(config)
    except Exception:  # noqa: BLE001  # nosec B110
        return False
    if not resolved:
        return False
    with _publish_lock:
        if _state is not initial:
            return False
        _state = _Snapshot(resolved, fetched_at, SOURCE_DISK, etag, lossless, config)
        return True


def _loses_required_values(candidate: dict) -> bool:





    try:
        from .served_config import missing_served_values

        gaps = missing_served_values(candidate)
        if not gaps or missing_served_values(_state.config):
            return False
    except Exception:  # noqa: BLE001
        return False
    shown = ", ".join(gaps[:3]) + (f" and {len(gaps) - 3} more" if len(gaps) > 3 else "")
    try:
        from qgis.core import Qgis, QgsMessageLog

        QgsMessageLog.logMessage(
            f"Server settings answer lacks required values ({shown}); "
            "keeping the last complete settings", "AI Segmentation",
            level=Qgis.MessageLevel.Warning)
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return True


def set_config(config: dict, etag: str | None = None) -> None:



















    global _state
    if not isinstance(config, dict) or not config:
        return
    try:
        owned = _without_run_tables(copy.deepcopy(config))


        payload = json.dumps(owned, allow_nan=False)
        if len(payload.encode("utf-8")) > _MAX_BYTES:
            return
    except Exception:  # noqa: BLE001
        return
    with _publish_lock:
        try:
            resolved = _resolve_with_override(owned)
        except Exception:  # noqa: BLE001
            resolved = owned
        if _loses_required_values(resolved):



            _state = _state._replace(etag=None)
            return
        resolved_etag = _sanitize_etag(etag) if etag is not None else _state.etag
        _state = _Snapshot(resolved, time.time(), SOURCE_LIVE, resolved_etag, False, owned)
        try:
            save_config(owned, resolved_etag)
        except TypeError:




            save_config(owned)



_DOWNGRADED_SCOPES = frozenset({"public"})
_DOWNGRADED_AUTH_STATES = frozenset({"public", "degraded"})


def _is_account_answer(config: dict) -> bool:
    return config.get("policy_scope") == "account" or config.get("auth_state") == "account"


def _is_downgraded_answer(config: dict) -> bool:
    return (config.get("policy_scope") in _DOWNGRADED_SCOPES
            or config.get("auth_state") in _DOWNGRADED_AUTH_STATES)


def keep_account_sections(config: dict, holds_key: bool) -> dict:









    try:
        if not holds_key or not isinstance(config, dict) or not _is_downgraded_answer(config):
            return config
        cached = _state.raw
        if not isinstance(cached, dict) or not _is_account_answer(cached):
            return config
        merged = dict(config)
        if isinstance(cached.get("detection_policy"), dict):
            merged["detection_policy"] = copy.deepcopy(cached["detection_policy"])
        for key, value in cached.items():
            if key not in merged and key not in _RUN_ONLY_TOP_KEYS:
                merged[key] = copy.deepcopy(value)

        merged["policy_scope"] = cached.get("policy_scope", "account")
        if "auth_state" in cached:
            merged["auth_state"] = cached["auth_state"]
        return merged
    except Exception:  # noqa: BLE001
        return config


def config_etag() -> str | None:


















    state = _state
    if state.source == SOURCE_LIVE or (state.source == SOURCE_DISK and state.lossless):
        return state.etag
    return None


def remember_etag(etag: str | None) -> None:












    global _state
    clean = _sanitize_etag(etag)
    with _publish_lock:
        _state = _state._replace(etag=clean)


def get_config() -> dict:











    return _state.config


def config_source() -> str:






    return _state.source


def config_fetched_at() -> float | None:

    return _state.fetched_at
