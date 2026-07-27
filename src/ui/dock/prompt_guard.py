"""Prompt guard rail for the Automatic (text prompt) input box."""
from __future__ import annotations

import difflib
import re
import time
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

    def _as_ratio(key: str, fallback: float) -> float:
        """A fuzzy-match cutoff in (0, 1], else the fallback literal. Keeps the
        default behaviour identical when the policy omits the key."""
        v = policy.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 < float(v) <= 1.0:
            return float(v)
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
        # Words never singularized even when their singular is a known object
        # (empty fallback, so the bare-plural rewrite is off without a policy).
        "plural_keep": _as_set("plural_keep"),
        # Concept phrase -> concrete object word the cloud model can ground
        # ("wheat" -> "crop field"). Empty fallback, never a shipped table.
        "aliases": _as_map("aliases"),
        # Object tokens whose runs benefit from an auto-attached example crop.
        "exemplar_boost": _as_set("exemplar_boost"),
        "max_words": _as_int("max_words", _PROMPT_MAX_WORDS_FALLBACK),
        "max_chars": _as_int("max_chars", _PROMPT_MAX_CHARS_FALLBACK),
        # Fuzzy-match cutoffs. Defaults reproduce the prior hardcoded behaviour.
        "typo_cutoff": _as_ratio("typo_cutoff", 0.8),
        "typo_cutoff_foreign": _as_ratio("typo_cutoff_foreign", 0.84),
        "suggest_cutoff": _as_ratio("suggest_cutoff", 0.72),
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
    if _TABLES_CACHE is not None and pid == _TABLES_CACHE_POLICY_ID:
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
    cutoff = _prompt_tables()["suggest_cutoff"]
    best, best_ratio = None, 0.0
    for w in words:
        for m in difflib.get_close_matches(w, tokens, n=1, cutoff=cutoff):
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


# The catalogue behind the label index changes rarely (a background prefetch
# task, at most), while this lookup fires on every keystroke of the prompt
# box. A short TTL keeps the per-keystroke cost flat without needing a cheap
# identity to peg on (the source re-parses JSON on each read, so a fresh
# object comes back even when the content is unchanged).
_LABEL_INDEX_CACHE_TTL_S = 2.0
_LABEL_INDEX_CACHE: dict[str, str] | None = None
_LABEL_INDEX_CACHE_TIME: float = 0.0


def _localized_label_index() -> dict[str, str]:
    global _LABEL_INDEX_CACHE, _LABEL_INDEX_CACHE_TIME
    now = time.monotonic()
    if _LABEL_INDEX_CACHE is not None and (now - _LABEL_INDEX_CACHE_TIME) < _LABEL_INDEX_CACHE_TTL_S:
        return _LABEL_INDEX_CACHE
    try:
        from ...core.presets.segmentation_presets import token_by_localized_label
        index = token_by_localized_label()
    except Exception:  # noqa: BLE001
        index = {}
    _LABEL_INDEX_CACHE = index
    _LABEL_INDEX_CACHE_TIME = now
    return _LABEL_INDEX_CACHE


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

    Builds on ``english_token_for``: resolves a localized word to its English
    token (else the prompt itself), then applies the same two silent rewrites
    the run uses so every policy lookup keys on the token that will actually be
    sent, a bare plural of a known object collapses to its singular
    ("buildings" -> "building") and a server-aliased concept maps to its
    concrete object ("wheat" -> "crop field"). Synchronous and cheap, so it is
    safe on the debounced prompt commit; the async server fallback lives at the
    caller. Never rewrites what the user sees, only what the lookups key on.
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    token = english_token_for(raw) or raw
    return _apply_alias(_singular_token(token))


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
    if _VOCAB_CACHE is not None and pid == _VOCAB_CACHE_POLICY_ID:
        return _VOCAB_CACHE
    vocab = set(tables["english_object_words"])
    vocab.update(tables["foreign_to_english"].values())
    # Alias keys AND values are recognized words: the key ('wheat') so it is
    # never typo-corrected away before the alias fires, the value ('crop
    # field') so is_known_object accepts the concrete object it maps to.
    aliases = tables["aliases"]
    vocab.update(aliases.keys())
    vocab.update(aliases.values())
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
    close = difflib.get_close_matches(candidate, pool, n=1, cutoff=tables["typo_cutoff"])
    if close:
        return close[0]
    close = difflib.get_close_matches(
        candidate, list(foreign), n=1, cutoff=tables["typo_cutoff_foreign"])
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


# ---------------------------------------------------------------------------
# Bare-plural singularization and concept aliasing. Both are silent rewrites:
# the cloud model grounds a singular concrete object far more reliably than a
# bare plural or an ungroundable concept word, so the run keys on the rewritten
# token while the box keeps the user's own words.
# ---------------------------------------------------------------------------

def _singular_candidates(word: str) -> list[str]:
    """Candidate singular spellings for a bare plural, in priority order. A
    word ending in 'ss' is left alone (so 'grass'/'glass' are never stripped)."""
    cands: list[str] = []
    if len(word) < 4:
        return cands
    if word.endswith("s") and not word.endswith("ss"):
        cands.append(word[:-1])
    if word.endswith("es"):
        cands.append(word[:-2])
    if word.endswith("ies"):
        cands.append(word[:-3] + "y")
    return cands


def _bare_plural_of(word: str, vocab: set[str] | None = None) -> str | None:
    """The singular of ``word`` when it is a bare plural whose singular is a
    known object word, else None. A word already recognized as-is is kept (so a
    known object that happens to end in 's' is not stripped), and the server
    ``plural_keep`` list is honored."""
    if vocab is None:
        vocab = _known_vocabulary()
    if word in vocab or word in _prompt_tables()["plural_keep"]:
        return None
    for cand in _singular_candidates(word):
        if cand in vocab:
            return cand
    return None


def _singular_token(token: str) -> str:
    """Collapse a token whose LAST word is a bare plural of a known object to
    its singular ('solar panels' -> 'solar panel'). Unknown or already-singular
    tokens pass through unchanged."""
    words = token.split(" ")
    if not words:
        return token
    sing = _bare_plural_of(words[-1])
    if sing is None:
        return token
    return " ".join(words[:-1] + [sing])


def _singularize_bare_plural(words: list[str]) -> str | None:
    """Rewrite a committed prompt whose LAST core word is a bare plural of a
    known object into its singular ('buildings' -> 'building'), or None. Earlier
    core words must already be recognized object words; the last is validated by
    its singular. The vocab gate is what keeps this safe with no client table."""
    strip = _prompt_tables()["strip"]
    vocab = _known_vocabulary()
    core = [w for w in words if w not in strip] or words
    if not core:
        return None
    sing = _bare_plural_of(core[-1], vocab)
    if sing is None:
        return None
    if not all(_word_is_known(w, vocab) for w in core[:-1]):
        return None
    return " ".join(core[:-1] + [sing])


def _apply_alias(token: str) -> str:
    """Map a resolved concept token to its concrete object word when the server
    alias table carries it ('wheat' -> 'crop field'), else the token unchanged.
    Naive singular tolerated, so an aliased key resolves in its plural too."""
    aliases = _prompt_tables()["aliases"]
    if not aliases:
        return token
    for probe in _lookup_variants(token.strip().lower()):
        if probe in aliases:
            return aliases[probe]
    return token


def _alias_for(words: list[str]) -> str | None:
    """The concrete object a plain English concept prompt aliases to, or None.
    Matches the whole stripped core phrase (plus its naive singular)."""
    aliases = _prompt_tables()["aliases"]
    if not aliases:
        return None
    strip = _prompt_tables()["strip"]
    core = [w for w in words if w not in strip] or words
    candidate = " ".join(core)
    for probe in _lookup_variants(candidate):
        if probe in aliases:
            return aliases[probe]
    return None


def _swap_result(token: str, base_reason: str) -> tuple[bool, str, str]:
    """A silent-swap return for a resolved token, applying the concept alias so
    a translation/typo/plural that lands on an alias key resolves in one step.
    ``base_reason`` is used when no alias applies ('translated' or 'plural')."""
    aliased = _apply_alias(token)
    if aliased != token:
        return (True, "alias", aliased)
    return (True, base_reason, token)


def is_exemplar_boost_prompt(text: str) -> bool:
    """True when ``text`` names an object whose runs benefit from an auto
    example crop (server ``exemplar_boost`` list). Matches the stripped core
    phrase (accent-folded, leading article dropped) or its naive singular.
    Cheap and never raises: False on an empty list or any problem."""
    try:
        boost = _prompt_tables()["exemplar_boost"]
        if not boost:
            return False
        norm = re.sub(r"\s+", " ", (text or "")).strip().lower().strip("?.!,;:")
        words = [w for w in _fold_ascii(norm).split(" ") if w]
        while words and words[0] in _LEAD_ARTICLES:
            words = words[1:]
        strip = _prompt_tables()["strip"]
        core = [w for w in words if w not in strip] or words
        if not core:
            return False
        return any(probe in boost for probe in _lookup_variants(" ".join(core)))
    except Exception:  # noqa: BLE001 -- best-effort UI hint, never blocks
        return False


def validate_prompt(text: str) -> tuple[bool, str | None, str | None]:
    """Validate the committed cloud-model prompt.

    Returns ``(ok, reason, suggestion)``. ``reason`` is a short key the dock
    maps to a localized hint ("empty" / "weird" / "language" / "multi" /
    "sentence" / "referential" / "subjective" / "abstract" / "too_long");
    ``suggestion`` is the closest known object token (or the English
    translation for a non-English prompt) when one is obvious.

    Silent-swap reasons ``{"translated", "plural", "alias"}`` all mean the same
    thing to the dock: run ``suggestion`` instead of the typed text, and tell
    the user. ``"translated"`` covers a KNOWN object typed in another supported
    language and near-miss typo repair; ``"plural"`` a bare plural of a known
    object collapsed to its singular ('buildings' -> 'building'); ``"alias"`` a
    concept word the server maps to a concrete object ('wheat' -> 'crop field',
    applied on top of a translation/typo/plural so it resolves in one step).

    Special case: ``(True, "steer", term)`` means the prompt is valid English
    but a weak choice from a top-down view ('wall'); the run still proceeds,
    the dock just shows a light non-blocking nudge toward ``term`` (or, when
    ``term`` is ``""``, toward the Library).

    Special case: ``(True, "multi_first", token)`` means the prompt names
    several objects ("buildings and roads") whose first object is itself a
    clean prompt: the run proceeds on ``token`` and the dock nudges the user
    to run the others separately. Otherwise ``ok`` is True only for a clean
    1-2 word English object the model can ground.
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
    # A token the server aliases to a concrete object resolves in the same step.
    token = english_token_for(raw)
    if token and token != norm:
        return _swap_result(token, "translated")

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
    # Several objects at once ("building, tree" / "cars and trucks"): the
    # cloud model grounds ONE concept per run. Refusing outright loses the
    # user, so when the FIRST named object stands on its own the run proceeds
    # on it - (True, "multi_first", token) - and the dock shows a light
    # non-blocking nudge to run the rest separately. The hard block remains
    # only when no usable leading object can be extracted.
    if _MULTI_OBJECT_RE.search(" " + norm + " "):
        # Split the SAME padded string the search matched: a mid-typing
        # trailing connector ("cars and") only matches with the padding, so
        # splitting the bare norm would return the whole prompt unchanged and
        # the recursion below would never terminate.
        first = re.split(r"[,;/+&]| and | or ", " " + norm + " ")[0].strip()
        first_words = [w for w in first.split(" ") if w]
        if first and first != norm:
            f_ok, f_reason, f_sugg = validate_prompt(first)
            if f_ok:
                swapped = f_reason in ("translated", "plural", "alias")
                token = f_sugg if (swapped and f_sugg) else first
                return (True, "multi_first", token)
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
    # A valid-LOOKING prompt may still be a typo of a known token ('buildin',
    # 'solar panle'). Repair it through the same silent-swap channel as the
    # language translation, so the user sees what will run. A repair that lands
    # on a server-aliased concept resolves to its concrete object in one step.
    correction = _typo_correction(folded_words)
    if correction and correction != norm:
        return _swap_result(correction, "translated")
    # A committed, all-known prompt whose last word is a bare plural runs on the
    # singular ('buildings' -> 'building'), which the cloud model grounds far
    # more reliably. Skipped when the singular is itself a weak, steered word
    # (the steer nudge below wins, e.g. 'walls' -> steer 'building').
    steer = _steer_suggestion(words)
    plural = _singularize_bare_plural(words)
    if plural is not None and plural != norm and steer is None:
        return _swap_result(plural, "plural")
    # A plain English concept the server maps to a concrete object ('wheat' ->
    # 'crop field'): swap it in so the run keys on the object the model grounds.
    alias = _alias_for(words)
    if alias is not None:
        return (True, "alias", alias)
    # Valid English, but a weak choice from a top-down view ('wall' -> the
    # building). Non-blocking: the run still proceeds, the dock just shows a
    # light nudge toward the term that works best.
    if steer is not None:
        return (True, "steer", steer)
    return (True, None, None)
