
































from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path

from qgis.PyQt.QtNetwork import QNetworkReply

from ..core.server_dials import dial_in_range






_REVALIDATE_AFTER_SECONDS = 6 * 3600



_MAX_VALIDATOR_CHARS = 200


def _clean_validator(value) -> str:

    if not isinstance(value, str) or len(value) > _MAX_VALIDATOR_CHARS:
        return ""
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        return ""
    return value.strip()


def validator_from_reply(reply: QNetworkReply) -> tuple[str, str]:

    def header(name: bytes) -> str:
        try:
            return bytes(reply.rawHeader(name)).decode("latin-1", "replace").strip()
        except (TypeError, ValueError, UnicodeError):
            return ""
    return header(b"ETag"), header(b"Last-Modified")


def validator_path(image_path: Path) -> Path:

    return image_path.with_name(image_path.name + ".meta")


def read_validator(image_path: Path) -> dict | None:

    try:
        with open(validator_path(image_path), encoding="utf-8") as f:
            raw = f.read(4097)
        if len(raw) > 4096:
            return None
        data = json.loads(raw)
    except (OSError, ValueError, RecursionError):
        return None
    if not isinstance(data, dict):
        return None
    etag = _clean_validator(data.get("etag"))
    modified = _clean_validator(data.get("last_modified"))
    if not etag and not modified:
        return None
    return {"etag": etag, "last_modified": modified, "checked": data.get("checked", 0.0)}


def write_validator(image_path: Path, etag: str = "", last_modified: str = "") -> None:





    etag = _clean_validator(etag)
    last_modified = _clean_validator(last_modified)
    if not etag and not last_modified:
        try:
            validator_path(image_path).unlink(missing_ok=True)
        except OSError:
            pass  # nosec B110
        return
    _write_json(
        validator_path(image_path),
        {"etag": etag, "last_modified": last_modified, "checked": time.time()},
    )


def mark_validator_checked(image_path: Path) -> None:

    meta = read_validator(image_path)
    if not meta:
        return
    meta["checked"] = time.time()
    _write_json(validator_path(image_path), meta)


def should_revalidate(meta: dict | None) -> bool:

    if not meta or not (meta.get("etag") or meta.get("last_modified")):
        return False
    try:
        checked = float(meta.get("checked") or 0.0)
    except (TypeError, ValueError, OverflowError):
        return True
    now = time.time()
    if not math.isfinite(checked) or checked < 0 or checked > now:
        return True
    interval = dial_in_range(
        "tuning.library.demo_revalidate_after_s", _REVALIDATE_AFTER_SECONDS, 300, 86400)
    return (now - checked) >= interval


def conditional_headers(meta: dict | None) -> dict[str, str]:











    if not meta:
        return {}
    etag = _clean_validator(meta.get("etag"))
    if etag:
        return {"If-None-Match": etag, "X-If-None-Match": etag}
    last_modified = _clean_validator(meta.get("last_modified"))
    if last_modified:
        return {"If-Modified-Since": last_modified,
                "X-If-Modified-Since": last_modified}
    return {}


def _write_json(path: Path, payload: dict) -> None:

    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".validator-", suffix=".tmp", delete=False) as f:
            tmp = Path(f.name)
            json.dump(payload, f, allow_nan=False)
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError):
        try:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
        except OSError:
            pass  # nosec B110
