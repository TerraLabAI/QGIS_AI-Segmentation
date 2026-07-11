"""Prompt guard rail for the Automatic (text prompt) input box."""
from __future__ import annotations

import difflib
import re
import unicodedata

# ---------------------------------------------------------------------------
# Prompt guard rail for the Automatic (cloud) text box.
#
# the cloud model segments a CONCEPT named by a short English noun phrase ("solar panel").
# Its paper says it is NOT for sentences, questions, or referring expressions,
# and Esri / Wherobots confirm it wants clear, countable objects. It also only
# understands ENGLISH: a prompt in any other language silently returns garbage,
# which reads as "the AI is bad" instead of "the prompt was off the rails".
#
# validate_prompt blocks the off-rails cases (sentences, referring expressions,
# abstract/subjective words, several objects at once, non-English input) and
# routes the user back to a 1-2 word English object, suggesting the closest
# known token - or the English translation - when it can. It also silently
# repairs what it CAN fix itself: known objects typed in another language and
# near-miss typos of known tokens ('buildin'), both via the (True, "translated",
# token) return.
#
# The tuned word sets and the offline foreign->English lexicon come from the
# server-delivered prompt policy (core.detection_policy.prompt_policy). Without
# a policy the guard keeps only its generic English fallbacks and validation
# gets MORE permissive (a word in none of the lists is never rejected for it):
# non-English prompts flow to the commit-time server translation fallback and
# the typo corrector falls back to the catalogue tokens alone.
# ---------------------------------------------------------------------------

# Fallback maxima (used when the policy omits them). One head noun plus at most
# one modifier; longer reads as a phrase the cloud model rejects. Counted after
# harmless quantifiers/articles are stripped.
_PROMPT_MAX_WORDS_FALLBACK = 2
_PROMPT_MAX_CHARS_FALLBACK = 30

# Generic English function words kept as fallbacks: these are safe to ship as
# plain defaults.
#
# Harmless quantifiers/articles: stripped, never blocked ("the buildings",
# "all cars" and "building" behave identically).
_PROMPT_STRIP_WORDS_FALLBACK = {
    "a", "an", "the", "all", "every", "each", "any", "my",
}
# Sentence / command / question markers - the cloud model fails on these.
_PROMPT_COMMAND_WORDS_FALLBACK = {
    "please", "find", "show", "detect", "segment", "give", "want", "need",
    "can", "could", "would", "select", "identify", "locate", "highlight",
    "where", "how", "what", "which", "extract", "get", "map", "mark",
    "draw", "outline", "make", "generate", "create", "count", "list",
    "i", "me", "you",
}

# Separators that mean "several objects at once" - the cloud model grounds ONE
# concept per run, so "building, tree" quietly biases toward garbage.
_MULTI_OBJECT_RE = re.compile(r"[,;/+&]| and | or ")

# Leading articles in the supported languages, stripped before the silent
# translation lookup so "la piscine" resolves like "piscine".
_LEAD_ARTICLES = {
    "the", "a", "an", "le", "la", "les", "l", "un", "une", "des", "du",
    "el", "los", "las", "una", "unos", "unas", "o", "os", "um", "uma",
    "uns", "umas", "il", "lo", "gli", "i", "der", "die", "das", "ein",
    "eine", "de", "d",
    # Dutch (de is already listed above): het, een
    "het", "een",
}


# ---------------------------------------------------------------------------
# Policy-derived word sets. Built once per distinct policy dict object and
# memoized on its identity, so the per-keystroke validate_prompt stays flat.
# ---------------------------------------------------------------------------

def _build_prompt_tables(policy: dict) -> dict:
    """Derive the guard's word sets / maps / maxima from a prompt policy dict.
    Missing or malformed entries fall back to the generic defaults (empty for
    the policy-supplied lists, so validation only gets more permissive)."""
    def _as_set(key: str) -> set[str]:
        v = policy.get(key)
        return {str(w).lower() for w in v} if isinstance(v, list) else set()

    def _as_map(key: str) -> dict[str, str]:
        v = policy.get(key)
        if not isinstance(v, dict):
            return {}
        return {str(k).lower(): str(val) for k, val in v.items()}

    def _as_int(key: str, fallback: int) -> int:
        v = policy.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
        return fallback

    def _as_steer(key: str) -> dict[str, str]:
        """Flatten the steer entries into {trigger word -> better term}. An
        empty ``suggest`` is kept verbatim ('' = 'point at the Library')."""
        v = policy.get(key)
        if not isinstance(v, list):
            return {}
        out: dict[str, str] = {}
        for entry in v:
            if not isinstance(entry, dict):
                continue
            suggest = entry.get("suggest")
            suggest = suggest if isinstance(suggest, str) else ""
            for kw in entry.get("keywords") or []:
                if isinstance(kw, str) and kw:
                    out[kw.lower()] = suggest
        return out

    return {
        "strip": _as_set("strip_words") or set(_PROMPT_STRIP_WORDS_FALLBACK),
        "command": _as_set("command_words") or set(_PROMPT_COMMAND_WORDS_FALLBACK),
        "abstract": _as_set("abstract"),
        "subjective": _as_set("subjective"),
        "referential": _as_set("referential"),
        "foreign_stopwords": _as_set("foreign_stopwords"),
        "foreign_to_english": _as_map("foreign_to_english"),
        "english_object_words": _as_set("english_object_words"),
        "steer": _as_steer("steer"),
        "max_words": _as_int("max_words", _PROMPT_MAX_WORDS_FALLBACK),
        "max_chars": _as_int("max_chars", _PROMPT_MAX_CHARS_FALLBACK),
    }


# Shared no-policy tables: only the generic strip/command fallbacks, everything
# else empty. A stable singleton so the no-policy path never rebuilds or churns
# the memo (a fresh {} would have a new id() each call).
_EMPTY_TABLES = _build_prompt_tables({})

_TABLES_CACHE: dict | None = None
_TABLES_CACHE_POLICY_ID: int | None = None


def _prompt_tables() -> dict:
    """The guard's word sets for the current policy. Rebuilt only when the
    policy dict object changes identity (never per call), so this is cheap
    enough to call on every keystroke."""
    global _TABLES_CACHE, _TABLES_CACHE_POLICY_ID
    try:
        from ...core.detection_policy import prompt_policy

        policy = prompt_policy()
    except Exception:  # noqa: BLE001 -- policy is best-effort; fail permissive
        return _EMPTY_TABLES
    if not policy:
        return _EMPTY_TABLES
    pid = id(policy)
    if _TABLES_CACHE is not None and _TABLES_CACHE_POLICY_ID == pid:
        return _TABLES_CACHE
    _TABLES_CACHE = _build_prompt_tables(policy)
    _TABLES_CACHE_POLICY_ID = pid
    return _TABLES_CACHE


def _fold_ascii(text: str) -> str:
    """Accent-fold to plain ASCII ('bâtiment' -> 'batiment')."""
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def _prompt_known_tokens() -> list[str]:
    try:
        from ...core.presets.segmentation_presets import known_tokens
        return known_tokens()
    except Exception:  # noqa: BLE001
        return []


def _prompt_suggestion(norm: str, words: list[str]) -> str | None:
    """Closest known object token for an off-rails prompt, or None."""
    tokens = _prompt_known_tokens()
    if not tokens:
        return None
    word_set = set(words)
    # A multi-word token present verbatim (e.g. "swimming pool") wins outright.
    for tok in tokens:
        if " " in tok and tok in norm:
            return tok
    # A single-word token present as a whole word (avoid "car" in "cargo").
    for tok in tokens:
        if " " not in tok and tok in word_set:
            return tok
    # Otherwise the closest fuzzy match on any word.
    best, best_ratio = None, 0.0
    for w in words:
        for m in difflib.get_close_matches(w, tokens, n=1, cutoff=0.72):
            ratio = difflib.SequenceMatcher(None, w, m).ratio()
            if ratio > best_ratio:
                best, best_ratio = m, ratio
    return best


def _english_suggestion(folded: str, words: list[str]) -> str | None:
    """English translation for a non-English prompt, or None. The whole
    phrase is tried first ("panneau solaire"), then each word."""
    foreign = _prompt_tables()["foreign_to_english"]
    phrase = foreign.get(folded)
    if phrase:
        return phrase
    for w in words:
        hit = foreign.get(w)
        if hit:
            return hit
    return None


def _localized_label_index() -> dict[str, str]:
    try:
        from ...core.presets.segmentation_presets import token_by_localized_label
        return token_by_localized_label()
    except Exception:  # noqa: BLE001
        return {}


def _lookup_variants(phrase: str) -> list[str]:
    """The phrase itself plus a naive singular (trailing s/x stripped per
    word), so "piscines" and "panneaux solaires" both resolve."""
    words = phrase.split(" ")
    singular = " ".join(
        w[:-1] if len(w) > 3 and w[-1] in "sx" else w for w in words)
    return [phrase] if singular == phrase else [phrase, singular]


def english_token_for(text: str) -> str | None:
    """Silent translation: resolve a prompt typed in a supported UI language
    to its English cloud-model token, or None when unknown.

    Sources, in order: the catalogue's own localized labels (en/fr/es/pt -
    scales with the library, nothing to maintain) then the server-delivered
    common-word lexicon. Tolerant to case, accents, extra spaces, a leading
    article, and naive plurals.
    """
    foreign = _prompt_tables()["foreign_to_english"]
    norm = re.sub(r"\s+", " ", (text or "")).strip().lower().strip("?.!,;:")
    folded = _fold_ascii(norm)
    words = [w for w in folded.split(" ") if w]
    while words and words[0] in _LEAD_ARTICLES:
        words = words[1:]
    if not words:
        return None
    candidate = " ".join(words)
    index = _localized_label_index()
    for probe in _lookup_variants(candidate):
        hit = index.get(probe) or foreign.get(probe)
        if hit:
            return hit
    return None


def resolve_object_token(text: str) -> str:
    """The English cloud-model token for a possibly-localized prompt.

    A thin wrapper over ``english_token_for``: returns the English token when
    the offline lexicon (or catalogue label index) resolves the prompt, and the
    prompt itself otherwise (already English, or a word the offline lexicon does
    not cover). Synchronous and cheap, so it is safe on the debounced prompt
    commit; the async server fallback lives at the caller. Never rewrites what
    the user sees, only what the policy lookups key on.
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    return english_token_for(raw) or raw


_VOCAB_CACHE: set[str] | None = None
_VOCAB_CACHE_POLICY_ID: int | None = None


def _known_vocabulary() -> set[str]:
    """English words the typo corrector treats as already correct: catalogue
    tokens, every translation target of the lexicon, the curated English
    object list, and each individual word of the multi-word phrases. Cached
    once the catalogue tokens are importable; rebuilt when the policy changes."""
    global _VOCAB_CACHE, _VOCAB_CACHE_POLICY_ID
    tables = _prompt_tables()
    pid = id(tables)
    if _VOCAB_CACHE is not None and _VOCAB_CACHE_POLICY_ID == pid:
        return _VOCAB_CACHE
    vocab = set(tables["english_object_words"])
    vocab.update(tables["foreign_to_english"].values())
    tokens = _prompt_known_tokens()
    vocab.update(tokens)
    for phrase in list(vocab):
        vocab.update(phrase.split(" "))
    if tokens:  # a failed catalogue import stays retryable next call
        _VOCAB_CACHE = vocab
        _VOCAB_CACHE_POLICY_ID = pid
    return vocab


def _word_is_known(word: str, vocab: set[str]) -> bool:
    """The word, or its naive singular, is a recognized English object word."""
    strip = _prompt_tables()["strip"]
    if word in vocab or word in strip:
        return True
    return len(word) > 3 and word[-1] in "sx" and word[:-1] in vocab


def is_known_object(text: str) -> bool:
    """True when every core word of the prompt is a recognized English object
    word. The commit path uses this to decide whether a VALID-looking prompt
    still deserves the one-off server translation lookup (a language the
    offline lexicon does not cover, or a rare English word: both pass the
    guard, only the former needs rewriting)."""
    strip = _prompt_tables()["strip"]
    norm = re.sub(r"\s+", " ", (text or "")).strip().lower().strip("?.!,;:")
    words = [w for w in _fold_ascii(norm).split(" ") if w]
    core = [w for w in words if w not in strip] or words
    if not core:
        return True
    vocab = _known_vocabulary()
    return all(_word_is_known(w, vocab) for w in core)


def _typo_correction(words: list[str]) -> str | None:
    """Silent repair for a committed prompt that LOOKS valid but is not a
    known object word: 'buildin' -> 'building' (fuzzy), 'sol' -> 'solar
    panel' (unique prefix), 'batimen' -> 'building' (fuzzy against the
    foreign lexicon keys). Returns the token to run, or None when the prompt
    is already fine or no safe repair exists. Known English words are never
    rewritten ('cart' must stay 'cart')."""
    tables = _prompt_tables()
    foreign = tables["foreign_to_english"]
    vocab = _known_vocabulary()
    core = [w for w in words if w not in tables["strip"]] or words
    if len(core) > tables["max_words"]:
        return None
    if all(_word_is_known(w, vocab) for w in core):
        return None
    candidate = " ".join(core)
    pool = sorted(set(_prompt_known_tokens()) | set(foreign.values()))
    if not pool:
        return None
    if len(candidate) >= 3:
        prefixed = [t for t in pool if t.startswith(candidate)]
        if len(prefixed) == 1:
            return prefixed[0]
    close = difflib.get_close_matches(candidate, pool, n=1, cutoff=0.8)
    if close:
        return close[0]
    close = difflib.get_close_matches(
        candidate, list(foreign), n=1, cutoff=0.84)
    if close:
        return foreign[close[0]]
    return None


def _looks_foreign(raw_norm: str, folded: str, folded_words: list[str]) -> bool:
    """True when the prompt is written in another language or script."""
    tables = _prompt_tables()
    foreign = tables["foreign_to_english"]
    stopwords = tables["foreign_stopwords"]
    # Whole known phrase first ("panneaux solaires").
    if folded in foreign:
        return True
    # Any letter beyond the extended-Latin block (Cyrillic, CJK, Arabic,
    # Greek...) is never an English object word.
    if any(c.isalpha() and ord(c) > 0x024F for c in raw_norm):
        return True
    # Accented Latin letters: English cloud-model tokens are pure ASCII, and the
    # accented word almost always has a translation in the map above.
    if any(c.isalpha() and ord(c) > 0x7F for c in raw_norm):
        return True
    # Pure-ASCII prompts in fr/es/pt/de/it: caught by their function words
    # ("los edificios") or by a known object word ("batiment").
    if any(w in stopwords for w in folded_words):
        return True
    return any(w in foreign for w in folded_words)


def _steer_suggestion(words: list[str]) -> str | None:
    """A better single object term for a valid but weak-from-above prompt, or
    ``None`` when nothing should be steered.

    Returns the term to nudge toward (e.g. 'building' for 'wall'), an empty
    string ``""`` when the concept has no clean aerial term and the user should
    be pointed at the Library, or ``None`` when the prompt is fine.

    Matches the WHOLE stripped phrase (plus its naive singular), never a single
    word inside a longer prompt: 'wall' is steered, but a valid compound like
    'sea wall' or 'forest floor' is left alone. The trigger set is
    server-curated and high-precision, so a normal object prompt is never
    nudged."""
    steer = _prompt_tables()["steer"]
    if not steer:
        return None
    strip = _prompt_tables()["strip"]
    core = [w for w in words if w not in strip] or words
    candidate = " ".join(core)
    for probe in _lookup_variants(candidate):
        if probe in steer:
            term = steer[probe]
            return term if term != probe else None
    return None


def validate_prompt(text: str) -> tuple[bool, str | None, str | None]:
    """Validate the committed cloud-model prompt.

    Returns ``(ok, reason, suggestion)``. ``reason`` is a short key the dock
    maps to a localized hint ("empty" / "weird" / "language" / "multi" /
    "sentence" / "referential" / "subjective" / "abstract" / "too_long");
    ``suggestion`` is the closest known object token (or the English
    translation for a non-English prompt) when one is obvious.

    Special case: ``(True, "translated", token)`` means the prompt is a KNOWN
    object typed in another supported language (or a naive plural) - the run
    should silently use ``token`` instead.

    Special case: ``(True, "steer", term)`` means the prompt is valid English
    but a weak choice from a top-down view ('wall'); the run still proceeds,
    the dock just shows a light non-blocking nudge toward ``term`` (or, when
    ``term`` is ``""``, toward the Library). Otherwise ``ok`` is True only for
    a clean 1-2 word English object the model can ground.
    """
    tables = _prompt_tables()
    strip = tables["strip"]
    raw = (text or "").strip()
    if not raw:
        return (False, "empty", None)
    norm = re.sub(r"\s+", " ", raw).strip().lower().strip("?.!,;:")
    if not norm:
        return (False, "empty", None)
    words = [w for w in norm.split(" ") if w]
    folded = _fold_ascii(norm)
    folded_words = [w for w in folded.split(" ") if w]

    # Silent translation first: a known object word in the user's language is
    # VALID - the run just sends the English token (returned as ``suggestion``
    # under the "translated" reason; the dock swaps it in and tells the user).
    token = english_token_for(raw)
    if token and token != norm:
        return (True, "translated", token)

    # Untranslatable non-English input: every later hint assumes English
    # vocabulary, so explain the language rule instead of misdiagnosing.
    if _looks_foreign(norm, folded, folded_words):
        suggestion = _english_suggestion(folded, folded_words)
        if suggestion:
            # Snap the translation to the catalogue token when one matches.
            suggestion = _prompt_suggestion(
                suggestion, suggestion.split(" ")) or suggestion
        return (False, "language", suggestion)

    letters = sum(c.isalpha() for c in norm)
    if letters < max(2, (len(norm) + 1) // 2):
        return (False, "weird", _prompt_suggestion(norm, words))
    # Vowel-less letter mashes ("df", "sdfk") are keyboard noise, not objects.
    # Catalogue tokens are exempt so a legitimate acronym preset still passes.
    if norm not in set(_prompt_known_tokens()) and any(
            len(w) >= 2 and not any(c in "aeiouy" for c in w) for w in words):
        return (False, "weird", _prompt_suggestion(norm, words))
    # Several objects at once ("building, tree" / "cars and trucks").
    if _MULTI_OBJECT_RE.search(" " + norm + " "):
        first = re.split(r"[,;/+&]| and | or ", norm)[0].strip()
        first_words = [w for w in first.split(" ") if w]
        return (False, "multi", _prompt_suggestion(first, first_words))
    if "?" in raw or any(w in tables["command"] for w in words):
        return (False, "sentence", _prompt_suggestion(norm, words))
    if any(w in tables["referential"] for w in words):
        return (False, "referential", _prompt_suggestion(norm, words))
    if any(w in tables["subjective"] for w in words):
        return (False, "subjective", _prompt_suggestion(norm, words))
    if any(w in tables["abstract"] for w in words):
        return (False, "abstract", _prompt_suggestion(norm, words))
    # Quantifiers/articles are free: "the buildings" == "buildings".
    core_words = [w for w in words if w not in strip] or words
    if len(core_words) > tables["max_words"] or len(norm) > tables["max_chars"]:
        return (False, "too_long", _prompt_suggestion(norm, words))
    # Last: a valid-LOOKING prompt may still be a typo of a known token
    # ('buildin', 'solar panle'). Repair it through the same silent-swap
    # channel as the language translation, so the user sees what will run.
    correction = _typo_correction(folded_words)
    if correction and correction != norm:
        return (True, "translated", correction)
    # Valid English, but a weak choice from a top-down view ('wall' -> the
    # building). Non-blocking: the run still proceeds, the dock just shows a
    # light nudge toward the term that works best.
    steer = _steer_suggestion(words)
    if steer is not None:
        return (True, "steer", steer)
    return (True, None, None)
