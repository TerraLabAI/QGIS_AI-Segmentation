























from __future__ import annotations

from itertools import islice



_MAX_REASON_CHARS = 160



_MAX_REWRITE_CHARS = 160


def _clean_served_phrase(value: str, max_chars: int) -> str:






    try:
        from .server_dials import clean_served_text

        return clean_served_text(value, max_chars) or ""
    except Exception:  # noqa: BLE001
        return ""


def sanitize_attribute_filters(raw: object) -> list[dict[str, str]]:






    out: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    seen: set[tuple[str, str]] = set()
    for item in islice(raw, 64):
        if not isinstance(item, dict):
            continue
        attr = item.get("attribute")
        val = item.get("value")
        if not isinstance(attr, str) or not isinstance(val, str):
            continue
        attr = _clean_served_phrase(attr, _MAX_REWRITE_CHARS)
        val = _clean_served_phrase(val, _MAX_REWRITE_CHARS)
        if not attr or not val or (attr, val) in seen:
            continue
        seen.add((attr, val))
        out.append({"attribute": attr, "value": val})
    return out


def parse_prompt_rewrite(block: object) -> tuple[str, str, list[dict[str, str]]]:











    filters: list[dict[str, str]] = []
    if not isinstance(block, dict):
        return "none", "", filters
    filters = sanitize_attribute_filters(block.get("attribute_filters"))
    rewritten = block.get("rewritten")
    if isinstance(rewritten, str) and rewritten.strip():
        phrase = _clean_served_phrase(rewritten, _MAX_REWRITE_CHARS)
        if phrase:
            return "rewrite", phrase, filters
    if block.get("decline") is True:
        reason = block.get("reason")
        reason = _clean_served_phrase(reason, _MAX_REASON_CHARS) if isinstance(reason, str) else ""
        return "decline", reason, filters
    return "none", "", filters
