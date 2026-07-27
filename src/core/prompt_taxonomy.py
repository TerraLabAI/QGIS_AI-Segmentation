"""Shared prompt-matching primitives for every server-delivered keyword table.

Four independent policy tables classify the SAME user prompt: the seed object
tiers (which ground resolution to render), the review shape classes (which
geometry regularizers), the coarse mask-grid routing, and the scan gate's own
class map. They answer different questions and each keeps its own resolution
STRATEGY (see below), but they must agree on the one thing that is not a
product decision: what counts as a keyword hit.

Before this module each table carried its own matcher. Two semantics coexisted,
and the looser one (a bare word-start prefix) matched the keyword inside longer
unrelated words: "bus" hit "bush", "car" hit "carpet", "dam" hit "damaged roof"
(seeding a roof as an 80 m dam), "tree" hit "treeline". Normalization also
differed, so "solar_panel" resolved a shape class but no seed tier.

The matcher here is the strict one: a keyword hits only as a whole word, with a
trailing plural tolerated. Compounds that ARE genuine members of a class
("hedgerow", "carpark") are then a server-side data question (add the keyword to
the table) instead of a silent side effect of the regex, which is the right
place for a product decision.

Resolution strategy stays with each caller, because the intents genuinely
differ:
  - seed tiers: FINEST tier first. A multi-object prompt must be rendered fine
    enough for its smallest object, so "car parking lot" seeds at car scale.
  - shape / mask-grid: LONGEST keyword first, so a phrase beats another class's
    component word and the table's order never decides a tie.
  - scan gate: first entry wins, an explicitly ordered override list.
"""
from __future__ import annotations

import re
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")

# Keyword -> compiled matcher. Bounded by the policy's keyword count (~1k), and
# every lookup reuses it, so a run's thousands of matches compile nothing.
_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}


def normalize_prompt(prompt: str) -> str:
    """Canonical form every table matches against: lowercase, underscores as
    word separators (so "solar_panel" reads as two words) and collapsed
    whitespace."""
    text = (prompt or "").strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", text)


def _pattern(keyword: str) -> re.Pattern[str]:
    pat = _PATTERN_CACHE.get(keyword)
    if pat is None:
        # Whole word, trailing plural tolerated ("buildings" hits "building").
        # Anchored on both sides, so a keyword never matches inside a longer
        # word: "tree" must not hit "treeline", "bus" must not hit "bush".
        pat = re.compile(r"\b" + re.escape(keyword) + r"(?:e?s)?\b")
        _PATTERN_CACHE[keyword] = pat
    return pat


def keyword_matches(text: str, keyword: str) -> bool:
    """Whether ``keyword`` hits ``text`` (already normalized) as a whole word."""
    if not keyword or not text:
        return False
    return _pattern(keyword).search(text) is not None


def iter_keywords(entry: object) -> Iterator[str]:
    """The non-empty string keywords of a policy entry's ``keywords`` list.

    Server tables are untrusted input: a malformed entry yields nothing rather
    than raising, so one bad row can never break a run."""
    if not isinstance(entry, dict):
        return
    for kw in entry.get("keywords") or []:
        if isinstance(kw, str) and kw:
            yield kw


def first_entry_match(text: str, entries: Iterable[T]) -> T | None:
    """The first entry with a matching keyword, in the table's own order.

    For tables whose order encodes a priority the server author chose (seed
    tiers ordered finest-first, the gate's override list)."""
    for entry in entries:
        for kw in iter_keywords(entry):
            if keyword_matches(text, kw):
                return entry
    return None


def longest_keyword_match(text: str, entries: Iterable[T]) -> T | None:
    """The entry owning the LONGEST matching keyword, ties broken by table order.

    For tables where a longer phrase must beat another entry's component word
    ("swimming pool" over "pool") and the entry order must not decide the
    outcome."""
    best: T | None = None
    best_len = -1
    for entry in entries:
        for kw in iter_keywords(entry):
            if len(kw) > best_len and keyword_matches(text, kw):
                best, best_len = entry, len(kw)
    return best
