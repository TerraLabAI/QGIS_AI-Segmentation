




























from __future__ import annotations

import re
from functools import lru_cache
from itertools import islice
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


_MAX_KEYWORD_CHARS = 512
_MAX_ENTRY_KEYWORDS = 256


def normalize_prompt(prompt: str) -> str:



    if not isinstance(prompt, str):
        return ""
    text = prompt.strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", text)


@lru_cache(maxsize=2048)
def _pattern(keyword: str) -> re.Pattern[str]:

    return re.compile(r"\b" + re.escape(keyword) + r"(?:e?s)?\b")


def keyword_matches(text: str, keyword: str) -> bool:

    if (not isinstance(keyword, str) or not isinstance(text, str)
            or not keyword or not text or len(keyword) > _MAX_KEYWORD_CHARS):
        return False
    return _pattern(keyword).search(text) is not None


def iter_keywords(entry: object) -> Iterator[str]:




    if not isinstance(entry, dict):
        return
    keywords = entry.get("keywords")
    if not isinstance(keywords, (list, tuple)):
        return
    seen: set[str] = set()
    for kw in islice(keywords, _MAX_ENTRY_KEYWORDS):
        if not isinstance(kw, str) or len(kw) > _MAX_KEYWORD_CHARS:
            continue
        normalized = normalize_prompt(kw)
        if normalized and normalized not in seen:
            seen.add(normalized)
            yield normalized


def first_entry_match(text: str, entries: Iterable[T]) -> T | None:




    for entry in entries:
        for kw in iter_keywords(entry):
            if keyword_matches(text, kw):
                return entry
    return None


def longest_keyword_match(text: str, entries: Iterable[T]) -> T | None:





    best: T | None = None
    best_len = -1
    for entry in entries:
        for kw in iter_keywords(entry):
            if len(kw) > best_len and keyword_matches(text, kw):
                best, best_len = entry, len(kw)
    return best
