






from __future__ import annotations

from .detection_policy_core import (
    _is_finite_policy_value,
    review_policy,
)


def restore_partitions_for(prompt: str, policy: dict | None = None,
                           exemplar_only: bool = False) -> bool:



























    if exemplar_only:
        return bool(merge_policy(policy).get("restore_partition_exemplar_only"))
    norm = (prompt or "").strip().lower().replace("_", " ")
    if not norm:
        return False
    names = merge_policy(policy).get("restore_partition_prompts")
    if not isinstance(names, list):
        return False
    wanted = frozenset(
        str(v).strip().lower().replace("_", " ")
        for v in names if isinstance(v, str) and str(v).strip())
    return norm in wanted


def merge_policy(policy: dict | None = None) -> dict:




    review = review_policy(policy)
    merge = review.get("merge") if isinstance(review, dict) else None
    return merge if isinstance(merge, dict) else {}


def exemplar_only_merge_separate(policy: dict | None = None) -> bool:





    val = merge_policy(policy).get("exemplar_only")
    return not (isinstance(val, str) and val.strip().lower() == "map")


def map_likeness_min_share(policy: dict | None = None) -> float:








    val = merge_policy(policy).get("map_likeness_min_share")
    if _is_finite_policy_value(val):
        return float(val)
    return 0.15








_MERGE_SCALAR_DEFAULTS: dict[str, float] = {
    "merge_ios": 0.15,
    "dedup_ios": 0.5,
    "dup_ios_floor": 0.3,
    "dup_centroid_frac": 0.35,
    "seam_span_ios": 0.03,
    "ios_threshold": 0.5,



    "seam_span_tol": 0.85,



    "jitter_area_frac": 0.02,




    "jitter_erode_px": 1.0,



    "cover_threshold": 0.40,


    "score_floor_frac": 0.5,




    "part_inside": 0.90,


    "part_max_frac": 0.70,


    "part_sibling_ios": 0.20,


    "part_cover_frac": 0.60,

    "part_min_children": 2,
}



_MERGE_INT_SCALARS: frozenset[str] = frozenset({"part_min_children"})


def merge_scalar(key: str, fallback: float | None = None, policy: dict | None = None) -> float:





    if fallback is None:
        fallback = _MERGE_SCALAR_DEFAULTS.get(key, 0.0)
    val = merge_policy(policy).get(key)





    ok = _is_finite_policy_value(val) and float(val) >= 0.0
    resolved = float(val) if ok else float(fallback)
    return int(resolved) if key in _MERGE_INT_SCALARS else resolved


def merge_scalars(policy: dict | None = None) -> dict[str, float]:

    return {k: merge_scalar(k, d, policy) for k, d in _MERGE_SCALAR_DEFAULTS.items()}


def merge_scalar_kwargs(target: object, scalars: dict | None = None,
                        policy: dict | None = None) -> dict[str, float]:









    import inspect

    values = scalars if isinstance(scalars, dict) else merge_scalars(policy)
    try:
        accepted = inspect.signature(target).parameters  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return {}
    return {k: v for k, v in values.items() if k in accepted}
