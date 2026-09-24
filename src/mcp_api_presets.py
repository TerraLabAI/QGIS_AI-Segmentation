














from __future__ import annotations



_CLASS_LIMIT_MAX = 1000
_CLASS_LIMIT_DEFAULT = 100


def _class_limit_ceiling() -> int:





    try:
        from .core.server_dials import dial_in_range

        return int(dial_in_range(
            "agent.class_limit_max", _CLASS_LIMIT_MAX, _CLASS_LIMIT_DEFAULT, 100_000))
    except Exception:  # noqa: BLE001
        return _CLASS_LIMIT_MAX





_CLASS_LIST_HINT = (
    "Pass a 'token' value as object_class to detect_auto(). Tokens are the "
    "words a run understands; a 'label' is for showing a person and finds "
    "nothing if passed."
)


def _class_list_hint() -> str:

    from .core.server_dials import dial_text

    return dial_text("tuning.agent.hints", "class_list_hint", 400) or _CLASS_LIST_HINT


def _catalogue_with_source() -> tuple[list[dict], list[str], str]:







    from .core.presets import segmentation_presets
    from .core.presets.segmentation_presets_client import cached_or_offline_catalog

    cats, tops = cached_or_offline_catalog()


    source = "built_in" if cats is segmentation_presets.fallback_categories() else "server"
    return cats, [str(t) for t in tops], source


def _object_class_row(preset: dict, category: dict, top_ids: list[str]) -> dict | None:


    from .core.presets.segmentation_presets import pick_label

    token = str((preset or {}).get("prompt") or "").strip()
    if not token:
        return None
    return {
        "token": token,
        "label": pick_label(preset.get("label"), token),
        "category": str(category.get("key") or ""),
        "category_label": pick_label(category.get("label"), ""),


        "weak": bool(preset.get("weak")),
        "popular": bool(preset.get("top_pick")) or str(preset.get("id") or "") in top_ids,
    }


def _normalize_class_token(text) -> str:


    return str(text or "").strip().lower().replace("_", " ")


class SegmentationPresetsMixin:


    def list_object_classes(self, query: str = "", limit: int = 100) -> dict:







































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
            "hint": _class_list_hint(),
        }

    def describe_object_class(self, token: str) -> dict:


































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
            from .core.server_dials import dial_text


            hint += " " + (dial_text("tuning.agent.hints", "class_weak_suffix", 400) or (
                "This class names a kind of cover, not a countable object:"
                " expect soft, ragged outlines."
            ))
        row["hint"] = hint
        return row

    def _class_match_by_label(self, folded_wanted: str, cats: list[dict]):


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
