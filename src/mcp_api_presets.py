"""The object-class catalogue for the public API. Read only.

Part of `SegmentationMCPAPI` (see `mcp_api.py`), split out so one concern sits
in one file.

A zone run takes a word as ``object_class``, and the plugin curates the words
that read well. These two methods hand that list to an outside agent, so it
picks a validated word instead of guessing one. Without them the single most
useful fact this plugin holds, which words work, was only reachable by
importing an internal module by path.

The catalogue is served remotely and cached; this module never touches the
network. A build that has never fetched it answers from the bundled list, and
``source`` says which one answered, because the bundled list is shorter.
"""
from __future__ import annotations

# The catalogue holds a few dozen classes today. The cap exists so a mistyped
# limit cannot ask for a million rows, not to page a list this small.
_CLASS_LIMIT_MAX = 1000
_CLASS_LIMIT_DEFAULT = 100


def _class_limit_ceiling() -> int:
    """The biggest ``limit`` accepted, resolved at call time.

    Floored at the default, so the value a caller gets by passing nothing can
    never be refused.
    """
    try:
        from .core.server_dials import dial_in_range

        return int(dial_in_range(
            "agent.class_limit_max", _CLASS_LIMIT_MAX, _CLASS_LIMIT_DEFAULT, 100_000))
    except Exception:  # noqa: BLE001 - the shipped cap always works
        return _CLASS_LIMIT_MAX


# Said on every successful listing, so a caller that skipped the docs still
# passes the right field. A translated label sent as object_class finds
# nothing, and that mistake is invisible until a run comes back empty.
_CLASS_LIST_HINT = (
    "Pass a 'token' value as object_class to detect_auto(). Tokens are the "
    "words a run understands; a 'label' is for showing a person and finds "
    "nothing if passed."
)


def _catalogue_with_source() -> tuple[list[dict], list[str], str]:
    """The freshest catalogue readable without the network, and where it came from.

    The served catalogue wins whenever one was ever fetched, even past its
    cache TTL: a stale served list still beats the bundled one. The bundled
    list answers only when nothing was ever served. Returns ``(categories,
    top_pick_ids, source)`` with source "server" or "built_in".
    """
    from .core.presets import segmentation_presets
    from .core.presets.segmentation_presets_client import cached_or_offline_catalog

    cats, tops = cached_or_offline_catalog()
    # The fallback path hands back the bundled list object itself, so identity
    # tells the source apart without copying the client's cache keys here.
    source = "built_in" if cats is segmentation_presets.fallback_categories() else "server"
    return cats, [str(t) for t in tops], source


def _object_class_row(preset: dict, category: dict, top_ids: list[str]) -> dict | None:
    """One catalogue preset as a flat row a machine can act on, or None when
    the preset carries no token to pass."""
    from .core.presets.segmentation_presets import pick_label

    token = str((preset or {}).get("prompt") or "").strip()
    if not token:
        return None
    return {
        "token": token,
        "label": pick_label(preset.get("label"), token),
        "category": str(category.get("key") or ""),
        "category_label": pick_label(category.get("label"), ""),
        # weak: a kind of cover rather than a countable object. Outlines come
        # back ragged, and that is the class, not a bad run.
        "weak": bool(preset.get("weak")),
        "popular": bool(preset.get("top_pick")) or str(preset.get("id") or "") in top_ids,
    }


def _normalize_class_token(text) -> str:
    """Lowercase, trimmed, underscores as spaces, so 'Farm_Field' and
    'farm field' name the same class."""
    return str(text or "").strip().lower().replace("_", " ")


class SegmentationPresetsMixin:
    """Serve the validated object-class words a zone run accepts."""

    def list_object_classes(self, query: str = "", limit: int = 100) -> dict:
        """The words a caller can pass as ``object_class`` to detect_auto().

        Read only: lists the catalogue of validated object classes, grouped by
        category so a caller can tell a building from a swimming pool. An
        empty query returns the whole catalogue; a query filters it with the
        same search the panel uses, so any shipped language works and accents
        are optional ("batiment" and "immeuble" both reach "building").

        Parameters
        ----------
        query : str
            Optional filter. Empty returns everything.
        limit : int
            Most classes to return, at least 1; a limit past the accepted
            ceiling is refused, and the refusal names it. Default 100, more
            than the catalogue holds today; ``total`` says when a limit cut
            the list.

        Returns
        -------
        dict with keys:
            "classes" -- list[dict], catalogue order (most asked for first).
                         Each has "token" (str, the word to pass as
                         object_class), "label" (str, what to show a person,
                         in the UI language), "category" (str, stable key),
                         "category_label" (str), "weak" (bool: a kind of
                         cover rather than a countable object, outlines come
                         back ragged) and "popular" (bool).
            "count"    -- int, classes returned.
            "total"    -- int, classes that matched before the limit.
            "source"   -- "server" when the served catalogue answered,
                          "built_in" when this build has never fetched it.
                          The built-in list is shorter, so a caller on a cold
                          cache sees fewer words, not a different contract.
            "hint"     -- str, the next call worth making.
            "_error"   -- str, present only on failure.

        Costs nothing and never touches the network.
        """
        from .core.presets.segmentation_presets import pick_label, preset_matches_query

        try:
            cap = int(limit)
        except (TypeError, ValueError):
            return {"_error": f"limit must be a whole number, got {limit!r}."}
        ceiling = _class_limit_ceiling()
        if not 1 <= cap <= ceiling:
            return {"_error": (
                f"limit must be between 1 and {ceiling}, got {limit!r}.")}
        wanted = str(query or "").strip()

        cats, top_ids, source = _catalogue_with_source()
        rows: list[dict] = []
        for cat in cats:
            if not isinstance(cat, dict):
                continue
            cat_label = pick_label(cat.get("label"), str(cat.get("key") or ""))
            for preset in cat.get("presets") or []:
                if not isinstance(preset, dict):
                    continue
                if wanted and not preset_matches_query(preset, wanted, cat_label):
                    continue
                row = _object_class_row(preset, cat, top_ids)
                if row is not None:
                    rows.append(row)

        return {
            "classes": rows[:cap],
            "count": len(rows[:cap]),
            "total": len(rows),
            "source": source,
            "hint": _CLASS_LIST_HINT,
        }

    def describe_object_class(self, token: str) -> dict:
        """Everything the catalogue knows about one object class.

        Read only. ``token`` is matched case-insensitively, underscores count
        as spaces, and a label in any shipped language resolves to its token,
        so "Piscine" answers for "swimming pool". An unknown word comes back
        as an ``_error`` with the nearest known tokens, so a caller passing
        "buildings" is told to pass "building".

        Parameters
        ----------
        token : str
            The word to look up: a token from list_object_classes(), or a
            label in any shipped language.

        Returns
        -------
        dict with keys:
            "token"          -- str, the word to pass as object_class. When
                                the lookup came in as a label, this is the
                                token it resolved to.
            "label"          -- str, what to show a person, in the UI language.
            "labels"         -- dict, the label in every shipped language.
            "category"       -- str, stable category key.
            "category_label" -- str, the category shown to a person.
            "weak"           -- bool, a kind of cover rather than a countable
                                object: outlines come back ragged by nature.
            "popular"        -- bool, among the most asked-for classes.
            "source"         -- "server" or "built_in", as in
                                list_object_classes().
            "hint"           -- str, the next call worth making.
            "_error"         -- str, present only on failure.

        Costs nothing and never touches the network.
        """
        wanted = _normalize_class_token(token)
        if not wanted:
            return {"_error": "token must be a non-empty string naming an object class."}

        cats, top_ids, source = _catalogue_with_source()
        match: tuple[dict, dict] | None = None
        known: list[str] = []
        for cat in cats:
            if not isinstance(cat, dict):
                continue
            for preset in cat.get("presets") or []:
                if not isinstance(preset, dict):
                    continue
                candidate = str(preset.get("prompt") or "").strip()
                if not candidate:
                    continue
                known.append(candidate)
                if match is None and wanted in (
                    _normalize_class_token(candidate),
                    _normalize_class_token(preset.get("id")),
                ):
                    match = (preset, cat)

        if match is None:
            # A label in any shipped language still deserves an answer: the
            # panel's prompt box resolves those silently, so this does too.
            match = self._class_match_by_label(wanted, cats)
        if match is None:
            from .mcp_api import not_found_error
            return not_found_error(
                "object class", str(token), sorted(set(known)),
                note=(
                    "Call list_object_classes() for every word detect_auto() "
                    "accepts as object_class."
                ),
            )

        preset, cat = match
        row = _object_class_row(preset, cat, top_ids) or {}
        labels = preset.get("label")
        row["labels"] = dict(labels) if isinstance(labels, dict) else {"en": str(labels or row.get("token", ""))}
        row["source"] = source
        hint = f"Pass '{row.get('token', '')}' as object_class to detect_auto()."
        if row.get("weak"):
            hint += (
                " This class names a kind of cover, not a countable object:"
                " expect soft, ragged outlines."
            )
        row["hint"] = hint
        return row

    def _class_match_by_label(self, folded_wanted: str, cats: list[dict]):
        """Resolve a localized label to its preset, or None. ``folded_wanted``
        is already normalized; fold it further for the accent-blind index."""
        from .core.presets.segmentation_presets import fold_search_text, token_by_localized_label

        target = token_by_localized_label().get(fold_search_text(folded_wanted))
        if not target:
            return None
        for cat in cats:
            if not isinstance(cat, dict):
                continue
            for preset in cat.get("presets") or []:
                if isinstance(preset, dict) and str(preset.get("prompt") or "").strip() == target:
                    return (preset, cat)
        return None
