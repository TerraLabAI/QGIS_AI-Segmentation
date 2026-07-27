"""Pure parsing of the optional server ``prompt_rewrite`` block.

The POST /api/plugin/seg-run-plan response MAY carry an additive
``prompt_rewrite`` object; older servers and language-model timeouts omit it,
and absence must be exactly today's behavior. This module turns the raw block
into a small validated decision the Automatic flow acts on, with no Qt/QGIS
dependency so the parsing is unit-testable headless.

Block shape (all keys optional, any may be missing or wrongly typed):
    {
        "rewritten": str | None,
        "alternates": [str, ...],
        "attribute_filters": [{"attribute": str, "value": str}, ...],
        "decline": bool,
        "reason": str,
    }

Product rules baked in here:
  - a rewrite is a plain phrase applied verbatim (the server preserves any
    attributes it contains); this module only reports the phrase, the visible
    swap-and-tell UX is the sole place it is applied.
  - ``decline`` is advisory: it yields a short reason to show, never a block.
  - ``attribute_filters`` are informational in this pass (stored, not used).
"""
from __future__ import annotations

# Server-authored reason strings are shown verbatim; cap the length so a long
# model reply cannot overflow the one-line note under the prompt box.
_MAX_REASON_CHARS = 160


def sanitize_attribute_filters(raw: object) -> list[dict[str, str]]:
    """Keep only well-formed ``{"attribute": str, "value": str}`` entries.

    Non-list input, non-dict entries, and entries missing a non-empty string
    attribute or value are dropped; the two fields are stripped. Returns a
    fresh list (possibly empty) so a caller can store it directly. Never
    raises."""
    out: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        attr = item.get("attribute")
        val = item.get("value")
        if not isinstance(attr, str) or not isinstance(val, str):
            continue
        attr = attr.strip()
        val = val.strip()
        if not attr or not val:
            continue
        out.append({"attribute": attr, "value": val})
    return out


def parse_prompt_rewrite(block: object) -> tuple[str, str, list[dict[str, str]]]:
    """Normalize a raw ``prompt_rewrite`` block into ``(action, payload, filters)``.

    ``action`` is one of:
      - ``"rewrite"``: ``payload`` is the non-empty phrase to run verbatim.
      - ``"decline"``: ``payload`` is the (truncated) reason to show, may be "".
      - ``"none"``: ``payload`` is "" (absent or malformed block, or nothing
        to act on) -- exactly today's behavior.

    ``filters`` is the always-parsed attribute-filter list, returned for every
    action so the caller can store it regardless. A rewrite takes precedence
    over a decline. Never raises."""
    filters: list[dict[str, str]] = []
    if not isinstance(block, dict):
        return "none", "", filters
    filters = sanitize_attribute_filters(block.get("attribute_filters"))
    rewritten = block.get("rewritten")
    if isinstance(rewritten, str) and rewritten.strip():
        return "rewrite", rewritten.strip(), filters
    if block.get("decline") is True:
        reason = block.get("reason")
        reason = reason.strip()[:_MAX_REASON_CHARS] if isinstance(reason, str) else ""
        return "decline", reason, filters
    return "none", "", filters
