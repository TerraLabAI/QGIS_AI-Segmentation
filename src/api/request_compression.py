
















from __future__ import annotations

import gzip





_COMPRESS_FLOOR_BYTES = 8 * 1024




_COMPRESS_LEVEL = 1








_BODY_REFUSED_STATUS = 400






_gzip_refused = False


def gzip_request_refused() -> bool:

    return _gzip_refused


def note_gzip_request_refused() -> None:

    global _gzip_refused
    _gzip_refused = True


def gzip_requests_allowed() -> bool:







    if _gzip_refused:
        return False
    try:
        from ..core.server_dials import gzip_request_bodies_enabled

        return gzip_request_bodies_enabled()
    except Exception:  # noqa: BLE001  # nosec B110
        return False


def packed_request_body(body: bytes) -> tuple[bytes, bool]:








    try:
        from ..core.server_dials import dial_in_range

        floor_bytes = dial_in_range(
            "tuning.network.compress_floor_bytes", _COMPRESS_FLOOR_BYTES, 1024, 1_000_000)
    except Exception:  # noqa: BLE001
        return body, False
    if not body or len(body) < floor_bytes:
        return body, False
    if not gzip_requests_allowed():
        return body, False
    try:
        packed = gzip.compress(body, _COMPRESS_LEVEL, mtime=0)
    except Exception:  # noqa: BLE001
        return body, False
    if len(packed) >= len(body):
        return body, False
    return packed, True


def answer_refused_the_body(http_status: int | None) -> bool:










    return http_status == _BODY_REFUSED_STATUS
