










from __future__ import annotations

import math

from ..core.qt_compat import reply_http_status



SERVER_TIMING_NAMES = ("total", "queue", "claim", "decode", "infer", "post")

_SERVER_TIMING_MAX_CHARS = 2000

_SERVER_TIMING_MAX_MS = 3_600_000.0


def _split_outside_quotes(text: str, sep: str) -> list[str]:


    parts: list[str] = []
    start, quoted, escaped = 0, False, False
    for i, ch in enumerate(text):
        if escaped:
            escaped = False
        elif ch == "\\" and quoted:
            escaped = True
        elif ch == '"':
            quoted = not quoted
        elif ch == sep and not quoted:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def parse_server_timing(raw: str | None) -> dict[str, float]:








    if not raw or len(raw) > _SERVER_TIMING_MAX_CHARS:
        return {}
    out: dict[str, float] = {}
    for metric in _split_outside_quotes(raw, ","):
        fields = _split_outside_quotes(metric, ";")
        name = fields[0].strip().lower()
        if name not in SERVER_TIMING_NAMES or name in out:
            continue
        for param in fields[1:]:
            key, eq, value = param.partition("=")
            if not eq or key.strip().lower() != "dur":
                continue
            try:
                ms = float(value.strip().strip('"'))
            except ValueError:
                break
            if math.isfinite(ms) and 0.0 <= ms <= _SERVER_TIMING_MAX_MS:
                out[name] = ms
            break
    return out


def note_server_timing(answer, reply):








    if not isinstance(answer, dict) or "error" in answer:
        return answer
    if reply_http_status(reply) != 200:
        return answer
    try:
        raw = bytes(reply.rawHeader(b"Server-Timing")).decode("ascii", "ignore")
    except Exception:  # noqa: BLE001
        return answer
    timing = parse_server_timing(raw)
    if not timing:
        return answer
    answer = dict(answer)
    answer["server_timing"] = timing
    return answer
