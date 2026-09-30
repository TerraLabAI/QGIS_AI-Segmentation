





from __future__ import annotations

from .detection_policy_core import (
    prompt_policy,
)
from .prompt_taxonomy import iter_keywords, longest_keyword_match, normalize_prompt
from .server_dials import clean_served_text


def prompt_hint_for(
    prompt: str, policy: dict | None = None
) -> tuple[str, str] | None:
















    text = normalize_prompt(prompt)
    if not text:
        return None
    hints = prompt_policy(policy).get("hints")
    if not isinstance(hints, list):
        return None
    usable = [
        h for h in hints
        if isinstance(h, dict)
        and clean_served_text(h.get("hint"))
        and next(iter_keywords(h), None) is not None
    ]
    entry = longest_keyword_match(text, usable)
    if entry is None:
        return None
    first = next(iter_keywords(entry))
    slug = "".join(c if c.isalnum() else "_" for c in first.lower()).strip("_")
    hint_text = clean_served_text(entry["hint"])
    if hint_text is None:
        return None
    return f"prompt_hint_{slug}", hint_text
