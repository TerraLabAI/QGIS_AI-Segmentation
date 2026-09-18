







from __future__ import annotations

import math

from ..core import transport_dials as _td
from .terralab_client_primitives import (
    _http_status_of,
    _log_warning,
)




_WORTH_ASKING_AGAIN_CODES = ("TIMEOUT", "NO_INTERNET")




_RATE_LIMITED_STATUS = 429
_RETRY_AFTER_MAX_S = 10.0




_HANDOFF_STATUSES = (429, 503)


_RETRY_AFTER_HINT_MAX_S = 300.0



_RETRY_PAUSE_MIN_S = 0.3
_RETRY_PAUSE_MAX_S = 0.8


def _worth_asking_again(answer, http_status: int | None) -> bool:


    if http_status is not None and 500 <= http_status < 600:
        return True
    if http_status == _RATE_LIMITED_STATUS:
        return True
    if http_status is not None:
        return False
    return (isinstance(answer, dict)
            and answer.get("code") in _WORTH_ASKING_AGAIN_CODES)


def _parse_retry_after_header(raw: str) -> float | None:



    text = (raw or "").strip()
    if not text:
        return None
    hint_max = _td.retry_after_hint_max_s(_RETRY_AFTER_HINT_MAX_S)
    try:
        seconds = float(text)
        if not math.isfinite(seconds):
            return None
        return max(0.0, min(seconds, hint_max))
    except (TypeError, ValueError):
        pass
    try:
        import datetime  # noqa: PLC0415
        from email.utils import parsedate_to_datetime  # noqa: PLC0415

        when = parsedate_to_datetime(text)
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=datetime.timezone.utc)
        left = (when - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        return max(0.0, min(left, hint_max))
    except Exception:  # noqa: BLE001
        return None


def _retry_after_hint(reply) -> float | None:

    try:
        raw = bytes(reply.rawHeader(b"Retry-After")).decode("ascii", "ignore")
    except Exception:  # noqa: BLE001
        return None
    return _parse_retry_after_header(raw)


def _note_retry_after(answer, reply):













    if not isinstance(answer, dict) or "error" not in answer:
        return answer
    if _http_status_of(reply) not in _HANDOFF_STATUSES:
        return answer
    hint = _retry_after_hint(reply)
    if hint is None:
        return answer
    answer = dict(answer)
    answer["retry_after_header"] = True
    try:
        from_body = float(answer.get("retry_after") or 0.0)
    except (TypeError, ValueError, OverflowError):
        from_body = 0.0
    if not math.isfinite(from_body) or from_body <= 0.0:
        answer["retry_after"] = hint
    return answer






_WINDOW_HINT_MIN = 1
_WINDOW_HINT_MAX = 16


def _window_hint(reply) -> int | None:






    try:
        raw = bytes(reply.rawHeader(b"X-Window-Hint")).decode("ascii", "ignore")
    except Exception:  # noqa: BLE001
        return None
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        return None
    if _WINDOW_HINT_MIN <= value <= _td.window_hint_ceiling(_WINDOW_HINT_MAX):
        return value
    return None


def _note_window_hint(answer, reply):









    if not isinstance(answer, dict) or "error" in answer:
        return answer
    if _http_status_of(reply) != 200:
        return answer
    hint = _window_hint(reply)
    if hint is None:
        return answer
    answer = dict(answer)
    answer["window_hint"] = hint
    return answer


def _retry_after_s(reply) -> float:


    hint = _retry_after_hint(reply)
    if hint is None:
        return 0.0
    return min(hint, _td.retry_after_max_s(_RETRY_AFTER_MAX_S))


def _retry_pause_s() -> float:

    import random  # noqa: PLC0415

    return random.uniform(  # nosec B311
        *_td.retry_pause_window_s((_RETRY_PAUSE_MIN_S, _RETRY_PAUSE_MAX_S)))


def _note_skipped_tuning(what: str, answer) -> None:







    if not isinstance(answer, dict) or "error" not in answer:
        return
    _log_warning(f"Ran without the {what} "
                 f"({answer.get('code') or 'no code'})")
