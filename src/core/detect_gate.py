















from __future__ import annotations







MIN_EXAMPLE_POSITIVES = 1





MIN_META_POSITIVES = 1


def _policy_min(getter_name: str, fallback: int) -> int:





    try:
        from . import detection_policy
        return int(getattr(detection_policy, getter_name)(fallback))
    except Exception:  # noqa: BLE001
        return fallback


def min_example_positives() -> int:

    return _policy_min("exemplar_min_example_positives", MIN_EXAMPLE_POSITIVES)


def min_meta_positives() -> int:

    return _policy_min("exemplar_min_meta_positives", MIN_META_POSITIVES)






def meta_satisfied(has_text: bool, positives: int) -> bool:






    return has_text and positives >= min_meta_positives()


def can_detect(has_text: bool, positives: int, excludes: int = 0) -> bool:










    return has_text or positives >= min_example_positives()


def exclude_available(positives: int) -> bool:






    return positives >= min_example_positives()
