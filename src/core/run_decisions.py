






from __future__ import annotations

import math

_MERGE_VALUES = ("separate", "map")


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def parse_run_decisions(plan: object) -> dict | None:




    block = plan.get("decisions") if isinstance(plan, dict) else None
    if not isinstance(block, dict):
        return None
    merge = block.get("merge")
    threshold = _number(block.get("coarse_mask_max_mupp"))
    restore = block.get("restore_partitions")
    if merge not in _MERGE_VALUES or threshold is None or threshold < 0:
        return None
    if not isinstance(restore, bool):
        return None
    from .self_exemplar import parse_self_exemplar_decision

    return {
        "merge_separate": merge == "separate",
        "coarse_mask_max_mupp": threshold,
        "restore_partitions": restore,





        "self_exemplar": parse_self_exemplar_decision(
            plan["self_exemplar"] if "self_exemplar" in plan else block.get("self_exemplar")),
    }


def neutral_run_decisions() -> dict:


    return {
        "merge_separate": True,
        "coarse_mask_max_mupp": 0.0,
        "restore_partitions": False,
        "self_exemplar": None,
    }


def parse_restore_decisions(block: object) -> dict | None:


    if not isinstance(block, dict):
        return None
    merge = block.get("merge")
    restore = block.get("restore_partitions")
    conf = _number(block.get("start_confidence"))
    if merge not in _MERGE_VALUES or not isinstance(restore, bool):
        return None
    if conf is None or not 0.0 <= conf <= 1.0:
        return None
    out = {
        "merge_separate": merge == "separate",
        "restore_partitions": restore,
        "start_confidence": conf,
    }

    floor = _number(block.get("restore_confidence_floor"))
    if floor is not None and 0.0 <= floor <= 1.0:
        out["restore_confidence_floor"] = floor
    return out


def coarse_mask_scale(threshold: float, run_mupp: float) -> int:


    t = _number(threshold)
    m = _number(run_mupp)
    if t is None or m is None or t <= 0 or m <= 0:
        return 1
    return 2 if m <= t else 1





_SESSION_PLANS: dict = {}
_SESSION_PLANS_MAX = 32


def _word_key(word: object) -> str:
    return word.strip().lower() if isinstance(word, str) else ""


def remember_plan(word: str, plan: object) -> None:

    key = _word_key(word)
    if not key or not isinstance(plan, dict) or plan.get("error"):
        return
    if parse_run_decisions(plan) is None:
        return
    _SESSION_PLANS.pop(key, None)
    _SESSION_PLANS[key] = plan
    while len(_SESSION_PLANS) > _SESSION_PLANS_MAX:
        _SESSION_PLANS.pop(next(iter(_SESSION_PLANS)))


def _shape_class_of(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _confidence_of(value: object) -> float | None:
    conf = _number(value)
    return conf if conf is not None and 0.0 <= conf <= 1.0 else None


def _served_entry(key: str) -> dict | None:
    try:
        from .config_cache import get_config

        config = get_config()
    except Exception:  # noqa: BLE001  # nosec B110
        return None
    table = config.get("run_decisions") if isinstance(config, dict) else None
    entry = table.get(key) if isinstance(table, dict) else None
    return entry if isinstance(entry, dict) else None


def fallback_choices(word: str) -> dict | None:






    key = _word_key(word)
    if not key:
        return None
    plan = _SESSION_PLANS.get(key)
    if plan is not None:
        review = plan.get("review")
        return {
            "decisions": parse_run_decisions(plan),
            "confidence_default": _confidence_of(plan.get("confidence_default")),
            "shape_class": _shape_class_of(
                review.get("shape_class") if isinstance(review, dict) else None),
            "review": review if isinstance(review, dict) else None,
            "source": "session",
        }
    entry = _served_entry(key)
    decisions = parse_run_decisions({"decisions": entry}) if entry else None
    if decisions is None:
        return None
    return {
        "decisions": decisions,
        "confidence_default": _confidence_of(entry.get("confidence_default")),
        "shape_class": _shape_class_of(entry.get("shape_class")),
        "review": entry.get("review") if isinstance(entry.get("review"), dict) else None,
        "source": "served",
    }
