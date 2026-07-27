
































from __future__ import annotations

import json
import os
import time
from pathlib import Path

from qgis.PyQt.QtNetwork import QNetworkReply

from ..core.server_dials import dial_in_range






_REVALIDATE_AFTER_SECONDS = 6 * 3600



_MAX_VALIDATOR_CHARS = 200


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
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_validator(image_path: Path, etag: str = "", last_modified: str = "") -> None:





    etag = (etag or "")[:_MAX_VALIDATOR_CHARS]
    last_modified = (last_modified or "")[:_MAX_VALIDATOR_CHARS]
    if not etag and not last_modified:
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
    except (TypeError, ValueError):
        return True
    interval = dial_in_range(
        "tuning.library.demo_revalidate_after_s", _REVALIDATE_AFTER_SECONDS, 300, 86400)
    return (time.time() - checked) >= interval


def conditional_headers(meta: dict | None) -> dict[str, str]:











    if not meta:
        return {}
    etag = str(meta.get("etag") or "")[:_MAX_VALIDATOR_CHARS]
    if etag:
        return {"If-None-Match": etag, "X-If-None-Match": etag}
    last_modified = str(meta.get("last_modified") or "")[:_MAX_VALIDATOR_CHARS]
    if last_modified:
        return {"If-Modified-Since": last_modified,
                "X-If-Modified-Since": last_modified}
    return {}


def _write_json(path: Path, payload: dict) -> None:

    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError):
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass  # nosec B110
