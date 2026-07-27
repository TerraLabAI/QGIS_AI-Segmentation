"""Remembers each cached image's HTTP validator, so it can be revalidated.

Template demo images are re-seeded on the server between plugin releases, and
a user has to see the new image without updating the plugin. Expiry alone is a
poor way to do that: it makes the user wait for a full re-download of bytes
that are usually identical. A conditional request does it properly. Keep the
``ETag`` (or ``Last-Modified``) the server sent with the bytes, send it back as
``If-None-Match`` (or ``If-Modified-Since``), and a 304 means the file on disk
is still the current one.

The image URL redirects to storage, so the validator that matters is the one
on the FINAL response. That is exactly what a finished ``QNetworkReply``
exposes, and the server sends the same string storage does, so one value works
whoever answers. The intermediate 302 is never an answer.

Sending it back is the awkward part. A standard ``If-None-Match`` does not
survive the trip in either direction, so the same value goes out under a
prefixed name as well (see ``conditional_headers``).

The validator lives in a sidecar next to the image (``before.jpg`` ->
``before.jpg.meta``, a few dozen bytes of JSON) rather than in one index for
the whole cache. Two loaders are alive at once (the library grid and the detail
popup): a shared index would mean a read-modify-write race between them and a
full rewrite per image, while a sidecar has exactly one writer, is deleted with
its image by the LRU sweep, and is inert if it is ever orphaned (the entry just
falls back to plain expiry until it is fetched again).

Entries the caller declared immutable never get a sidecar: their bytes cannot
change, so a conditional request for them is pure waste.

Everything here is silent: a cache that cannot be written is not an error
worth a message log line on every image.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from qgis.PyQt.QtNetwork import QNetworkReply

# How long a cached image is trusted before the next conditional request. Six
# hours means a re-seeded demo reaches an already-running QGIS the same working
# day, while a user who opens the library ten times in an afternoon spends one
# conditional request per image instead of ten. Shorter turns into noise on a
# thin link, longer makes a fresh demo look like it never shipped.
_REVALIDATE_AFTER_SECONDS = 6 * 3600

# Enough for any real ETag or HTTP date. Anything longer is junk we refuse to
# store, and refuse to send back to the server.
_MAX_VALIDATOR_CHARS = 200


def validator_from_reply(reply: QNetworkReply) -> tuple[str, str]:
    """A response's ETag and Last-Modified, as plain strings ("" when absent)."""
    def header(name: bytes) -> str:
        try:
            return bytes(reply.rawHeader(name)).decode("latin-1", "replace").strip()
        except (TypeError, ValueError, UnicodeError):
            return ""
    return header(b"ETag"), header(b"Last-Modified")


def validator_path(image_path: Path) -> Path:
    """Sidecar holding the validator of one cached image."""
    return image_path.with_name(image_path.name + ".meta")


def read_validator(image_path: Path) -> dict | None:
    """Return the stored validator, or None when there is none to trust."""
    try:
        with open(validator_path(image_path), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_validator(image_path: Path, etag: str = "", last_modified: str = "") -> None:
    """Store the validator that came with these bytes.

    No validator from the server means no sidecar, and the entry keeps its
    plain expiry behaviour.
    """
    etag = (etag or "")[:_MAX_VALIDATOR_CHARS]
    last_modified = (last_modified or "")[:_MAX_VALIDATOR_CHARS]
    if not etag and not last_modified:
        return
    _write_json(
        validator_path(image_path),
        {"etag": etag, "last_modified": last_modified, "checked": time.time()},
    )


def mark_validator_checked(image_path: Path) -> None:
    """Record that a 304 just confirmed the cached bytes, validator unchanged."""
    meta = read_validator(image_path)
    if not meta:
        return
    meta["checked"] = time.time()
    _write_json(validator_path(image_path), meta)


def should_revalidate(meta: dict | None) -> bool:
    """True when this entry carries a validator and is due for a check."""
    if not meta or not (meta.get("etag") or meta.get("last_modified")):
        return False
    try:
        checked = float(meta.get("checked") or 0.0)
    except (TypeError, ValueError):
        return True
    return (time.time() - checked) >= _REVALIDATE_AFTER_SECONDS


def conditional_headers(meta: dict | None) -> dict[str, str]:
    """Headers that ask the server for the image only if it has changed.

    Sent twice under two names. The standard header is dropped in transit (the
    host strips it before its own code runs, and it does not survive the
    redirect either), so the validator also travels prefixed, which nothing
    along the way claims to own. Measured: with the standard header alone every
    check answered 200 and re-downloaded the whole image.

    Plain ``str`` keys and values, because they are merged into the caller's
    own header dict and encoded there.
    """
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
    """Write a small JSON file atomically. Failure is silent by design."""
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
