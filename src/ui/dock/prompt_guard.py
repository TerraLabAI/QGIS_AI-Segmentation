
from __future__ import annotations

import difflib
import math
import re
import time
import unicodedata

from ...core.surface_dials import multi_object_pattern





























_PROMPT_MAX_WORDS_FALLBACK = 4
_PROMPT_MAX_CHARS_FALLBACK = 60






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




_PROMPT_MODIFIER_FALLBACK = {
    "red", "green", "blue", "white", "black", "yellow", "orange", "grey",
    "gray", "brown", "pink", "purple", "dark", "light", "large", "small",
    "big", "little", "tall", "long", "short", "wide", "narrow", "new", "old",
}




_PROMPT_VAGUE_FALLBACK = {
    "area", "zone", "region", "part", "section", "surface", "block", "border",
    "boundary", "edge", "center", "centre", "corner", "middle", "square",
    "rectangle", "circle", "row", "line", "patch", "district", "polygon",
    "shape", "place", "location", "spot",
}





_TYPO_MAX_EDITS_FALLBACK = ((5, 1), (9, 2))


_TYPO_FOREIGN_MIN_LEN_FALLBACK = 8




_MULTI_OBJECT_SEPARATORS = (",", ";", "/", "+", "&", " and ", " or ")



_LEAD_ARTICLES = {
    "the", "a", "an", "le", "la", "les", "l", "un", "une", "des", "du",
    "el", "los", "las", "una", "unos", "unas", "o", "os", "as", "um", "uma",
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
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            return int(v)
        return fallback

    def _as_ratio(key: str) -> float | None:


        v = policy.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 < float(v) <= 1.0:
            return float(v)
        return None

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

    def _as_edit_steps(key: str) -> tuple[tuple[int, int], ...]:



        v = policy.get(key)
        if not isinstance(v, list) or not v:
            return _TYPO_MAX_EDITS_FALLBACK
        steps: list[tuple[int, int]] = []
        for pair in v:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or not all(isinstance(x, int) and not isinstance(x, bool) for x in pair)
                    or pair[0] < 4 or not 0 <= pair[1] <= 2):
                return _TYPO_MAX_EDITS_FALLBACK
            steps.append((pair[0], pair[1]))
        return tuple(sorted(steps))

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

        "typo_cutoff": _as_ratio("typo_cutoff"),
        "typo_cutoff_foreign": _as_ratio("typo_cutoff_foreign"),
        "suggest_cutoff": _as_ratio("suggest_cutoff"),

        "modifier_words": _as_set_with("modifier_words", _PROMPT_MODIFIER_FALLBACK),


        "vague_words": _as_set_with("vague_words", _PROMPT_VAGUE_FALLBACK),
        "typo_max_edits": _as_edit_steps("typo_max_edits"),
        "typo_foreign_min_len": max(
            5, _as_int("typo_foreign_min_len", _TYPO_FOREIGN_MIN_LEN_FALLBACK)),
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





_LETTERS_WITHOUT_DECOMPOSITION = str.maketrans(
    {"ł": "l", "Ł": "L", "ß": "ss", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D",
     "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ı": "i"}
)


def _fold_ascii(text: str) -> str:

    return (
        unicodedata.normalize("NFKD", text.translate(_LETTERS_WITHOUT_DECOMPOSITION))
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
    if cutoff is None:
        return None
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


def _collapse_reduplication(word: str) -> str:

    head, sep, tail = word.partition("-")
    return head if sep and head and head == tail else word


def _lookup_variants(phrase: str) -> list[str]:



    words = [_collapse_reduplication(w) for w in phrase.split(" ")]
    out = [phrase]
    for variant in (" ".join(words), " ".join(
            w[:-1] if len(w) > 3 and w[-1] in "sx" else w for w in words)):
        if variant not in out:
            out.append(variant)
    return out


def english_token_for(text: str) -> str | None:








    foreign = _prompt_tables()["foreign_to_english"]
    norm = re.sub(r"\s+", " ", (text or "")).strip().lower().strip("?.!,;:")
    folded = _fold_ascii(norm)
    words = [w for w in folded.split(" ") if w]
    while len(words) > 1 and words[0] in _LEAD_ARTICLES:


        words = words[1:]
    index = _localized_label_index()
    if any(c.isalpha() and ord(c) > 0x024F for c in norm):


        hit = index.get(norm)
        if hit:
            return hit
    if not words:
        return None
    candidate = " ".join(words)
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


def _edit_distance(a: str, b: str, cap: int) -> int:


    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev2: list[int] = []
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
        if min(cur) > cap:
            return cap + 1
    return prev[-1]


def _edit_budget(word: str, steps: tuple[tuple[int, int], ...]) -> int:
    budget = 0
    for min_len, edits in steps:
        if len(word) >= min_len:
            budget = edits
    return budget


def _closest(word: str, pool: list[str], budget: int,
             mapping: dict[str, str] | None = None) -> str | None:



    if budget <= 0:
        return None
    best, cands = budget + 1, []
    for cand in pool:
        if abs(len(cand) - len(word)) > budget:
            continue
        d = _edit_distance(word, cand, budget)
        if d < best:
            best, cands = d, [cand]
        elif d == best:
            cands.append(cand)
    if best > budget or not cands:
        return None
    answers = {mapping[c] for c in cands} if mapping is not None else set(cands)
    return cands[0] if len(answers) == 1 else None


_POOLS_CACHE: dict = {"pools": None, "key": None}


def _corrector_pools() -> dict:








    tables = _prompt_tables()
    tokens = _prompt_known_tokens()
    key = (id(tables), len(tokens))
    cached = _POOLS_CACHE["pools"]
    if cached is not None and key == _POOLS_CACHE["key"]:
        return cached
    skip = tables["vague_words"] | tables["modifier_words"]
    phrases = set(tokens) | tables["english_object_words"] | set(tables["aliases"].values())
    phrases = {p for p in phrases if p not in skip}
    words = {w for p in phrases for w in p.split(" ") if len(w) >= 3 and w not in skip}
    foreign = tables["foreign_to_english"]
    never = (tables["command"] | tables["abstract"] | tables["subjective"]
             | tables["referential"] | set(tables["steer"]) | tables["modifier_words"]
             | tables["vague_words"] | tables["foreign_stopwords"])
    pools = {
        "phrases": phrases,
        "tokens": sorted(t for t in tokens if t not in skip),
        "multi": sorted(p for p in phrases | set(tables["aliases"]) if " " in p),
        "words": sorted(words),



        "spell": sorted(words | {w for p in set(tokens) | tables["english_object_words"]
                                 for w in p.split(" ") if len(w) >= 3}
                        | {w for w in skip if len(w) >= 3 and " " not in w}),
        "foreign": sorted(k for k in foreign if " " not in k and len(k) >= 4),
        "never": never,
    }
    if tokens:
        _POOLS_CACHE["pools"] = pools
        _POOLS_CACHE["key"] = key
    return pools


_COMMON_ENGLISH: dict = {"words": None}


def _common_english() -> frozenset[str]:



    words = _COMMON_ENGLISH["words"]
    if words is None:
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                            "resources", "common_english.txt")
        try:
            with open(path, encoding="utf-8") as fh:
                words = frozenset(w.strip().lower() for w in fh if w.strip())
        except OSError:
            words = frozenset()
        _COMMON_ENGLISH["words"] = words
    return words


_INFLECTIONS = ("ing", "ed", "es", "s", "er", "ers", "est", "ly", "ness", "ment", "ments")


def _is_common_english(word: str) -> bool:


    common = _common_english()
    if word in common:
        return True
    for suf in _INFLECTIONS:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            stem = word[: -len(suf)]
            if stem in common or stem + "e" in common:
                return True
            if len(stem) >= 4 and stem[-1] == stem[-2] and stem[:-1] in common:
                return True
            if stem.endswith("i") and stem[:-1] + "y" in common:
                return True
    return False


def _respell_word(word: str, vocab: set[str], in_phrase: bool = False) -> str | None:










    tables = _prompt_tables()
    pools = _corrector_pools()
    if _word_is_known(word, vocab) or word in pools["never"]:
        return word
    foreign = tables["foreign_to_english"]
    for probe in _lookup_variants(word):
        if probe in foreign:
            return foreign[probe]
    if _is_common_english(word) or any(_is_common_english(c) for c in _singular_candidates(word)):
        return word
    for obj in pools["words"]:
        rest = word[len(obj):]
        if len(obj) >= 4 and word.startswith(obj) and 0 < len(rest) <= 4 and obj.startswith(rest):
            return obj
    steps = tables["typo_max_edits"]



    found: dict[str, int] = {}
    for probe in [word] + _singular_candidates(word):
        budget = _edit_budget(probe, steps)
        if len(probe) == 4 and budget == 0:

            for w in pools["spell"]:
                if (len(w) == 4 and sorted(w) == sorted(probe)
                        and _edit_distance(probe, w, 1) == 1):
                    found[w] = min(found.get(w, 9), 1)
            continue
        if budget <= 0:
            continue
        for w in pools["spell"]:
            if abs(len(w) - len(probe)) <= budget:
                d = _edit_distance(probe, w, budget)
                if d <= budget:
                    found[w] = min(found.get(w, 9), d)
    if found:
        best = min(found.values())
        winners = [w for w, d in found.items() if d == best]
        if len(winners) > 1:


            winners = [w for w in winners if w[0] == word[0]]
        if len(winners) != 1:
            return None
        if in_phrase and len(word) <= 5 and winners[0][0] != word[0]:



            return None



        if best > 1 and any(abs(len(k) - len(word)) < best
                            and _edit_distance(word, k, best - 1) < best
                            for k in pools["foreign"]):
            return None
        return winners[0]
    return None


def _typo_correction(words: list[str]) -> str | None:








    tables = _prompt_tables()
    vocab = _known_vocabulary()
    core = [w for w in words if w not in tables["strip"]] or words
    if len(core) > tables["max_words"] + 2:
        return None
    if all(_word_is_known(w, vocab) for w in core):
        return None
    pools = _corrector_pools()
    candidate = " ".join(core)
    if len(candidate) >= 4:
        prefixed = [t for t in pools["tokens"] if t.startswith(candidate)]
        if len(prefixed) == 1 and len(core) == 1 and candidate not in pools["never"]:
            return prefixed[0]

    if len(candidate) >= 7:
        budget = _edit_budget(candidate.replace(" ", ""), tables["typo_max_edits"])
        hit = _closest(candidate, pools["multi"], budget)
        if hit and hit != candidate:
            return hit
    fixed = []
    for w in core:
        hit = _respell_word(w, vocab, in_phrase=len(core) > 1)
        if not hit:




            return None
        fixed.append(hit)
    if len(core) > 1 and all(f != w for f, w in zip(fixed, core)):



        return None
    out = " ".join(fixed)
    return out if out != candidate else None


def _head_noun(words: list[str]) -> str | None:







    if len(words) < 2:
        return None
    tables = _prompt_tables()
    pools = _corrector_pools()
    phrase = " ".join(words)
    probes = _lookup_variants(phrase) + [_singular_token(phrase)]
    if any(p in pools["phrases"] or p in tables["aliases"] for p in probes):
        return None
    vocab = _known_vocabulary()



    over = len(words) > tables["max_words"]
    for size in (2, 1):
        if len(words) <= size:
            continue
        head = " ".join(words[-size:])
        heads = {head, _singular_token(head)}
        if size == 1:
            ok = any(h in pools["words"] and h in pools["phrases"] for h in heads)
        else:
            ok = any(h in pools["phrases"] for h in heads)
        lead = words[:-size]
        if not ok or not all(w in tables["modifier_words"] for w in lead):
            continue


        for joined in (lead[-1] + words[-size], lead[-1] + "-" + words[-size]):
            if size == 1 and any(j in vocab or _is_common_english(j)
                                 for j in (joined, _singular_token(joined))):
                return " ".join(lead[:-1] + [joined]) if over else joined
        if over:
            return head
        return None
    return None


def _repair_result(phrase: str, norm: str) -> tuple[bool, str, str] | None:


    if not phrase or phrase == norm:
        return None
    ok, reason, sugg = validate_prompt(phrase)
    if ok and reason in ("translated", "plural", "alias") and sugg:
        if sugg == norm:
            return None
        return (True, "alias" if reason == "alias" else "translated", sugg)
    if ok and reason is None:
        return (True, "translated", phrase)
    return None


def _translate_mixed(words: list[str]) -> str | None:








    if len(words) < 2:
        return None
    tables = _prompt_tables()
    foreign = tables["foreign_to_english"]
    vocab = _known_vocabulary()
    out, english = [], 0
    for w in words:
        if _word_is_known(w, vocab):
            out.append(w)
            english += 1
            continue
        hit = next((foreign[p] for p in _lookup_variants(w) if p in foreign), None)
        if not hit or " " in hit or _edit_distance(w, hit.lower(), 2) > 2:
            return None
        out.append(hit)
    return " ".join(out) if english and english < len(words) else None


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


def server_lookup_wanted(text: str, ok: bool, reason: str | None,
                         suggestion: str | None) -> bool:








    if ok and reason is None:
        return not is_known_object(text)
    if not ok and reason == "language":
        return True
    if ok and reason == "plural" and suggestion:
        return not is_known_object(suggestion)
    return False


def vet_server_token(token: str | None) -> str | None:







    if not token:
        return None
    ok, reason, suggestion = validate_prompt(token)
    if not ok:
        return None
    if reason in ("translated", "plural", "alias"):
        return suggestion or None
    if reason == "steer":


        return _singular_token(token)
    if reason is None:
        return token
    return None


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


        mixed = _translate_mixed(folded_words)
        if mixed and mixed != norm:
            repaired = _repair_result(mixed, norm)
            if repaired:
                return repaired
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






    correction = _typo_correction(folded_words)
    work = correction.split(" ") if correction else [
        w for w in folded_words if w not in strip] or folded_words
    head = _head_noun(work)
    repaired = _repair_result(head or correction or "", norm)
    if len(core_words) > tables["max_words"] or len(norm) > tables["max_chars"]:
        if head and repaired:
            return repaired
        return (False, "too_long", _prompt_suggestion(norm, words))
    if repaired:
        return repaired




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
