



































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

CONFIG_FILENAME = "server_config.json"



_MAX_BYTES = 2 * 1024 * 1024



_FILE_SOURCE = "live_fetch"




_MAX_DISK_AGE_S = 21 * 24 * 60 * 60



SOURCE_NONE = "none"
SOURCE_DISK = "disk"
SOURCE_LIVE = "live"


class _Snapshot(NamedTuple):


    config: dict
    fetched_at: float | None
    source: str




    etag: str | None = None


_EMPTY = _Snapshot({}, None, SOURCE_NONE)



_ETAG_MAX_CHARS = 300


def _sanitize_etag(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > _ETAG_MAX_CHARS:
        return None


    if any(ord(ch) < 32 for ch in value):
        return None
    return value




_state: _Snapshot = _EMPTY

_override: dict | None = None
_override_state = {"loaded": False}
_publish_lock = threading.RLock()





def config_cache_path() -> str:

    return os.path.join(PLUGIN_CACHE_DIR, CONFIG_FILENAME)






_REPLACE_ATTEMPTS = 5
_REPLACE_DELAY_S = 0.2


def _move_config_into_place(tmp_path: str, path: str) -> None:





    for attempt in range(1, _REPLACE_ATTEMPTS + 1):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS:
                raise
            time.sleep(_REPLACE_DELAY_S)


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
        _move_config_into_place(tmp_path, path)
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
    "manual_cloud_route",
    "map_hypothesis_nms",
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


def load_config() -> tuple[dict, float | None, str | None]:










    path = config_cache_path()
    try:
        if not os.path.isfile(path) or os.path.getsize(path) > _MAX_BYTES:
            return {}, None, None
        with open(path, "rb") as fh:
            payload = fh.read(_MAX_BYTES + 1)
        if len(payload) > _MAX_BYTES:
            return {}, None, None

        def reject_constant(value):
            raise ValueError(f"Non-finite configuration number: {value}")

        data = json.loads(payload, parse_constant=reject_constant)
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
            or not math.isfinite(fetched_at) or fetched_at > time.time()):
        fetched_at = None
    elif time.time() - float(fetched_at) > _MAX_DISK_AGE_S:
        return {}, None, None
    etag = _sanitize_etag(data.get("etag"))
    return _without_code_execution_dials(_without_kill_switches(config)), fetched_at, etag


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
            data = json.loads(raw) if len(raw) <= _MAX_BYTES else None
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
        config, fetched_at, etag = load_config()
        resolved = _resolve_with_override(config)
    except Exception:  # noqa: BLE001  # nosec B110
        return False
    if not resolved:
        return False
    with _publish_lock:
        if _state is not initial:
            return False
        _state = _Snapshot(resolved, fetched_at, SOURCE_DISK, etag)
        return True


def set_config(config: dict, etag: str | None = None) -> None:















    global _state
    if not isinstance(config, dict) or not config:
        return
    try:
        owned = copy.deepcopy(config)
    except Exception:  # noqa: BLE001
        return
    with _publish_lock:
        try:
            resolved = _resolve_with_override(owned)
        except Exception:  # noqa: BLE001
            resolved = owned
        resolved_etag = _sanitize_etag(etag) if etag is not None else _state.etag
        _state = _Snapshot(resolved, time.time(), SOURCE_LIVE, resolved_etag)
        try:
            save_config(owned, resolved_etag)
        except TypeError:




            save_config(owned)


def config_etag() -> str | None:













    state = _state
    return state.etag if state.source == SOURCE_LIVE else None


def remember_etag(etag: str | None) -> None:











    global _state
    clean = _sanitize_etag(etag)
    if clean is None:
        return
    with _publish_lock:
        _state = _state._replace(etag=clean)


def get_config() -> dict:











    return _state.config


def config_source() -> str:






    return _state.source


def config_age_s() -> float | None:





    fetched_at = _state.fetched_at
    if fetched_at is None:
        return None
    return max(0.0, time.time() - fetched_at)
