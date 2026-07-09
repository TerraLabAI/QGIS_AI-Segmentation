
from __future__ import annotations

import difflib
import re
import time
import unicodedata

from ...core.surface_dials import multi_object_pattern





























_PROMPT_MAX_WORDS_FALLBACK = 2
_PROMPT_MAX_CHARS_FALLBACK = 30






_PROMPT_STRIP_WORDS_FALLBACK = {
    "a", "an", "the", "all", "every", "each", "any", "my",
}

_PROMPT_COMMAND_WORDS_FALLBACK = {
    "please", "find", "show", "detect", "segment", "give", "want", "need",
    "can", "could", "would", "select", "identify", "locate", "highlight",
    "where", "how", "what", "which", "extract", "get", "map", "mark",
    "draw", "outline", "make", "generate", "create", "count", "list",
    "i", "me", "you",
}









_PROMPT_ABSTRACT_FALLBACK = {"thing", "things", "stuff"}

_PROMPT_SUBJECTIVE_FALLBACK = {"nice", "beautiful", "ugly"}


_PROMPT_REFERENTIAL_FALLBACK = {"near", "between", "behind"}






_PROMPT_PLURAL_KEEP_FALLBACK = {"species", "series", "lens"}




_MULTI_OBJECT_SEPARATORS = (",", ";", "/", "+", "&", " and ", " or ")



_LEAD_ARTICLES = {
    "the", "a", "an", "le", "la", "les", "l", "un", "une", "des", "du",
    "el", "los", "las", "una", "unos", "unas", "o", "os", "um", "uma",
    "uns", "umas", "il", "lo", "gli", "i", "der", "die", "das", "ein",
    "eine", "de", "d",

    "het", "een",
}







def _build_prompt_tables(policy: dict) -> dict:



    def _as_set(key: str) -> set[str]:
        v = policy.get(key)
        return {str(w).lower() for w in v} if isinstance(v, list) else set()

    def _as_set_with(key: str, shipped: set[str]) -> set[str]:








        return _as_set(key) | shipped

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


        v = policy.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 < float(v) <= 1.0:
            return float(v)
        return fallback

    def _as_steer(key: str) -> dict[str, str]:


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
        "strip": _as_set_with("strip_words", _PROMPT_STRIP_WORDS_FALLBACK),
        "command": _as_set_with("command_words", _PROMPT_COMMAND_WORDS_FALLBACK),
        "abstract": _as_set_with("abstract", _PROMPT_ABSTRACT_FALLBACK),
        "subjective": _as_set_with("subjective", _PROMPT_SUBJECTIVE_FALLBACK),
        "referential": _as_set_with("referential", _PROMPT_REFERENTIAL_FALLBACK),
        "foreign_stopwords": _as_set("foreign_stopwords"),
        "foreign_to_english": _as_map("foreign_to_english"),
        "english_object_words": _as_set("english_object_words"),
        "steer": _as_steer("steer"),





        "plural_keep": _as_set_with("plural_keep", _PROMPT_PLURAL_KEEP_FALLBACK),



        "plural_strip": _as_set("plural_strip"),


        "aliases": _as_map("aliases"),

        "exemplar_boost": _as_set("exemplar_boost"),
        "max_words": _as_int("max_words", _PROMPT_MAX_WORDS_FALLBACK),
        "max_chars": _as_int("max_chars", _PROMPT_MAX_CHARS_FALLBACK),

        "typo_cutoff": _as_ratio("typo_cutoff", 0.8),
        "typo_cutoff_foreign": _as_ratio("typo_cutoff_foreign", 0.84),
        "suggest_cutoff": _as_ratio("suggest_cutoff", 0.72),
    }





_EMPTY_TABLES = _build_prompt_tables({})





_TABLES_CACHE: dict = {"tables": None, "policy": None}


def _prompt_tables() -> dict:




    try:
        from ...core.detection_policy import prompt_policy

        policy = prompt_policy()
    except Exception:  # noqa: BLE001
        return _EMPTY_TABLES
    if not policy:
        return _EMPTY_TABLES
    cached = _TABLES_CACHE["tables"]
    if cached is not None and policy is _TABLES_CACHE["policy"]:
        return cached
    tables = _build_prompt_tables(policy)
    _TABLES_CACHE["tables"] = tables
    _TABLES_CACHE["policy"] = policy
    return tables


def _fold_ascii(text: str) -> str:

    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )





_TOKENS_CACHE_TTL_S = 2.0
_TOKENS_CACHE: dict = {"tokens": None, "time": 0.0}


def _prompt_known_tokens() -> list[str]:
    now = time.monotonic()
    cached = _TOKENS_CACHE["tokens"]
    if cached is not None and (now - _TOKENS_CACHE["time"]) < _TOKENS_CACHE_TTL_S:
        return cached
    try:
        from ...core.presets.segmentation_presets import known_tokens
        tokens = known_tokens()
    except Exception:  # noqa: BLE001
        tokens = []


    if tokens:
        _TOKENS_CACHE["tokens"] = tokens
        _TOKENS_CACHE["time"] = now
    return tokens


def _prompt_suggestion(norm: str, words: list[str]) -> str | None:

    tokens = _prompt_known_tokens()
    if not tokens:
        return None
    word_set = set(words)

    for tok in tokens:
        if " " in tok and tok in norm:
            return tok

    for tok in tokens:
        if " " not in tok and tok in word_set:
            return tok

    cutoff = _prompt_tables()["suggest_cutoff"]
    best, best_ratio = None, 0.0
    for w in words:
        for m in difflib.get_close_matches(w, tokens, n=1, cutoff=cutoff):
            ratio = difflib.SequenceMatcher(None, w, m).ratio()
            if ratio > best_ratio:
                best, best_ratio = m, ratio
    return best


def _english_suggestion(folded: str, words: list[str]) -> str | None:


    foreign = _prompt_tables()["foreign_to_english"]
    phrase = foreign.get(folded)
    if phrase:
        return phrase
    for w in words:
        hit = foreign.get(w)
        if hit:
            return hit
    return None







_LABEL_INDEX_CACHE_TTL_S = 2.0
_LABEL_INDEX_CACHE: dict = {"index": None, "time": 0.0}


def _localized_label_index() -> dict[str, str]:
    now = time.monotonic()
    cached = _LABEL_INDEX_CACHE["index"]
    if cached is not None and (now - _LABEL_INDEX_CACHE["time"]) < _LABEL_INDEX_CACHE_TTL_S:
        return cached
    try:
        from ...core.presets.segmentation_presets import token_by_localized_label
        index = token_by_localized_label()
    except Exception:  # noqa: BLE001
        index = {}
    _LABEL_INDEX_CACHE["index"] = index
    _LABEL_INDEX_CACHE["time"] = now
    return index


def _lookup_variants(phrase: str) -> list[str]:


    words = phrase.split(" ")
    singular = " ".join(
        w[:-1] if len(w) > 3 and w[-1] in "sx" else w for w in words)
    return [phrase] if singular == phrase else [phrase, singular]


def english_token_for(text: str) -> str | None:








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











    raw = (text or "").strip()
    if not raw:
        return ""
    token = english_token_for(raw) or raw
    return _apply_alias(_singular_token(token))


_VOCAB_CACHE: dict = {"vocab": None, "policy_id": None}


def _known_vocabulary() -> set[str]:




    tables = _prompt_tables()
    pid = id(tables)
    cached = _VOCAB_CACHE["vocab"]
    if cached is not None and pid == _VOCAB_CACHE["policy_id"]:
        return cached
    vocab = set(tables["english_object_words"])
    vocab.update(tables["foreign_to_english"].values())



    aliases = tables["aliases"]
    vocab.update(aliases.keys())
    vocab.update(aliases.values())
    tokens = _prompt_known_tokens()
    vocab.update(tokens)
    for phrase in list(vocab):
        vocab.update(phrase.split(" "))
    if tokens:
        _VOCAB_CACHE["vocab"] = vocab
        _VOCAB_CACHE["policy_id"] = pid
    return vocab


def _word_is_known(word: str, vocab: set[str]) -> bool:

    strip = _prompt_tables()["strip"]
    if word in vocab or word in strip:
        return True
    return len(word) > 3 and word[-1] in "sx" and word[:-1] in vocab


def is_known_object(text: str) -> bool:





    strip = _prompt_tables()["strip"]
    norm = re.sub(r"\s+", " ", (text or "")).strip().lower().strip("?.!,;:")
    words = [w for w in _fold_ascii(norm).split(" ") if w]
    core = [w for w in words if w not in strip] or words
    if not core:
        return True
    vocab = _known_vocabulary()
    return all(_word_is_known(w, vocab) for w in core)


def prompt_vocabulary_is_loaded() -> bool:









    return bool(_prompt_tables()["english_object_words"])


def _typo_correction(words: list[str]) -> str | None:






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

    tables = _prompt_tables()
    foreign = tables["foreign_to_english"]
    stopwords = tables["foreign_stopwords"]

    if folded in foreign:
        return True


    if any(c.isalpha() and ord(c) > 0x024F for c in raw_norm):
        return True


    if any(c.isalpha() and ord(c) > 0x7F for c in raw_norm):
        return True


    if any(w in stopwords for w in folded_words):
        return True
    return any(w in foreign for w in folded_words)


def _steer_suggestion(words: list[str]) -> str | None:












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














_SINGULAR_KEEP_ENDINGS = ("ss", "as", "is", "us")



_SINGULAR_MIN_LEN = 3


def _singular_candidates(word: str, forced: bool = False) -> list[str]:












    if not word.endswith("s"):
        return []
    if not forced and (len(word) < 4 or word.endswith(_SINGULAR_KEEP_ENDINGS)):
        return []
    cands: list[str] = []
    stem = word[:-3]
    if word.endswith("ies"):

        if len(word) >= 5:
            cands.append(stem + "y")
        cands.append(stem + "ie")
    elif word.endswith("ves"):



        f_first = word.endswith("lves") or word.endswith(("eaves", "oaves"))
        ve_reading = [stem + "ve"]
        f_reading = [stem + "f", stem + "fe"]
        cands += f_reading + ve_reading if f_first else ve_reading + f_reading
    elif word.endswith("sses"):
        cands.append(word[:-2])
    elif word.endswith(("xes", "ches", "shes")):
        cands.append(word[:-2])
        cands.append(word[:-1])
    elif word.endswith("es"):
        cands.append(word[:-1])
        cands.append(word[:-2])
    else:
        cands.append(word[:-1])
    out: list[str] = []
    for cand in cands:
        if len(cand) >= _SINGULAR_MIN_LEN and cand != word and cand not in out:
            out.append(cand)
    return out


def _bare_plural_of(word: str, vocab: set[str] | None = None) -> str | None:
















    if vocab is None:
        vocab = _known_vocabulary()
    tables = _prompt_tables()
    if word in tables["plural_keep"]:
        return None
    forced = word in tables["plural_strip"]
    if not forced and word in vocab:
        return None
    cands = _singular_candidates(word, forced=forced)
    for cand in cands:
        if cand in vocab:
            return cand
    return cands[0] if cands else None


def _singular_token(token: str) -> str:



    words = token.split(" ")
    if not words:
        return token
    sing = _bare_plural_of(words[-1])
    if sing is None:
        return token
    return " ".join(words[:-1] + [sing])


def _singularize_bare_plural(words: list[str]) -> str | None:




    strip = _prompt_tables()["strip"]
    vocab = _known_vocabulary()
    core = [w for w in words if w not in strip] or words
    if not core:
        return None
    sing = _bare_plural_of(core[-1], vocab)
    if sing is None:
        return None
    return " ".join(core[:-1] + [sing])


def _apply_alias(token: str) -> str:



    aliases = _prompt_tables()["aliases"]
    if not aliases:
        return token
    for probe in _lookup_variants(token.strip().lower()):
        if probe in aliases:
            return aliases[probe]
    return token


def _alias_for(words: list[str]) -> str | None:


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



    aliased = _apply_alias(token)
    if aliased != token:
        return (True, "alias", aliased)
    return (True, base_reason, token)


def is_exemplar_boost_prompt(text: str) -> bool:




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
    except Exception:  # noqa: BLE001
        return False


def validate_prompt(text: str) -> tuple[bool, str | None, str | None]:



























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





    token = english_token_for(raw)
    if token and token != norm:
        return _swap_result(token, "translated")



    if _looks_foreign(norm, folded, folded_words):
        suggestion = _english_suggestion(folded, folded_words)
        if suggestion:

            suggestion = _prompt_suggestion(
                suggestion, suggestion.split(" ")) or suggestion
        return (False, "language", suggestion)

    letters = sum(c.isalpha() for c in norm)
    if letters < max(2, (len(norm) + 1) // 2):
        return (False, "weird", _prompt_suggestion(norm, words))


    if norm not in set(_prompt_known_tokens()) and any(
            len(w) >= 2 and not any(c in "aeiouy" for c in w) for w in words):
        return (False, "weird", _prompt_suggestion(norm, words))






    multi_re = multi_object_pattern(_MULTI_OBJECT_SEPARATORS)
    if multi_re.search(" " + norm + " "):




        first = multi_re.split(" " + norm + " ")[0].strip()
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

    core_words = [w for w in words if w not in strip] or words
    if len(core_words) > tables["max_words"] or len(norm) > tables["max_chars"]:
        return (False, "too_long", _prompt_suggestion(norm, words))




    correction = _typo_correction(folded_words)
    if correction and correction != norm:
        return _swap_result(correction, "translated")




    steer = _steer_suggestion(words)
    plural = _singularize_bare_plural(words)
    if plural is not None and plural != norm and steer is None:
        return _swap_result(plural, "plural")


    alias = _alias_for(words)
    if alias is not None:
        return (True, "alias", alias)



    if steer is not None:
        return (True, "steer", steer)
    return (True, None, None)
