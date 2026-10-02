






from __future__ import annotations

from .detection_policy_core import (
    policy_scope,
    review_policy,
)
from .served_config import require_served_int, require_served_number


def merge_policy(policy: dict | None = None) -> dict:




    review = review_policy(policy)
    merge = review.get("merge") if isinstance(review, dict) else None
    return merge if isinstance(merge, dict) else {}


def exemplar_only_merge_separate(policy: dict | None = None) -> bool:





    val = merge_policy(policy).get("exemplar_only")
    return not (isinstance(val, str) and val.strip().lower() == "map")


def map_likeness_min_share(policy: dict | None = None) -> float:






    with policy_scope(policy):
        return require_served_number(
            "detection_policy.review.merge.map_likeness_min_share", 0.0, 1.0)




_MERGE_INT_SCALARS: frozenset[str] = frozenset({"part_min_children"})


def merge_scalars(policy: dict | None = None) -> dict[str, float]:




    with policy_scope(policy):
        return {
            "merge_ios": require_served_number("detection_policy.review.merge.merge_ios", 0.0, 1.0),
            "dedup_ios": require_served_number("detection_policy.review.merge.dedup_ios", 0.0, 1.0),
            "dup_ios_floor": require_served_number("detection_policy.review.merge.dup_ios_floor", 0.0, 1.0),
            "dup_centroid_frac": require_served_number(
                "detection_policy.review.merge.dup_centroid_frac", 0.0, 1.0),
            "seam_span_ios": require_served_number("detection_policy.review.merge.seam_span_ios", 0.0, 1.0),
            "ios_threshold": require_served_number("detection_policy.review.merge.ios_threshold", 0.0, 1.0),
            "seam_span_tol": require_served_number("detection_policy.review.merge.seam_span_tol", 0.0, 1.0),
            "jitter_area_frac": require_served_number(
                "detection_policy.review.merge.jitter_area_frac", 0.0, 1.0),
            "jitter_erode_px": require_served_number(
                "detection_policy.review.merge.jitter_erode_px", 0.0, 16.0),
            "cover_threshold": require_served_number(
                "detection_policy.review.merge.cover_threshold", 0.0, 1.0),
            "score_floor_frac": require_served_number(
                "detection_policy.review.merge.score_floor_frac", 0.0, 1.0),
            "part_inside": require_served_number("detection_policy.review.merge.part_inside", 0.0, 1.0),
            "part_max_frac": require_served_number("detection_policy.review.merge.part_max_frac", 0.0, 1.0),
            "part_sibling_ios": require_served_number(
                "detection_policy.review.merge.part_sibling_ios", 0.0, 1.0),
            "part_cover_frac": require_served_number(
                "detection_policy.review.merge.part_cover_frac", 0.0, 1.0),
            "part_min_children": require_served_int(
                "detection_policy.review.merge.part_min_children", 1, 64),
        }


def merge_scalar(key: str, fallback: object = None, policy: dict | None = None) -> float:


    from .served_config import ServedConfigMissing

    values = merge_scalars(policy)
    if key not in values:
        raise ServedConfigMissing("detection_policy.review.merge." + key)
    return values[key]


def merge_scalar_kwargs(target: object, scalars: dict | None = None,
                        policy: dict | None = None) -> dict[str, float]:






    import inspect

    values = scalars if isinstance(scalars, dict) else merge_scalars(policy)
    try:
        accepted = inspect.signature(target).parameters  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return {}
    return {k: v for k, v in values.items() if k in accepted}
