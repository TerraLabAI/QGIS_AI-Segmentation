







from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard


def _is_finite_policy_value(value: object) -> TypeGuard[int | float]:

    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def get_detection_policy() -> dict:





    try:
        from .activation_manager import get_server_config

        config = get_server_config()
    except Exception:  # noqa: BLE001  # nosec B110
        return {}
    if not isinstance(config, dict):
        return {}
    policy = config.get("detection_policy")
    return policy if isinstance(policy, dict) else {}


def policy_rev(policy: dict | None = None) -> int | None:





    policy = get_detection_policy() if policy is None else policy
    val = policy.get("version") if isinstance(policy, dict) else None
    if _is_finite_policy_value(val):
        return int(val)
    return None


def seed_policy(policy: dict | None = None) -> dict:

    policy = get_detection_policy() if policy is None else policy
    seed = policy.get("seed") if isinstance(policy, dict) else None
    return seed if isinstance(seed, dict) else {}


def review_policy(policy: dict | None = None) -> dict:

    policy = get_detection_policy() if policy is None else policy
    review = policy.get("review") if isinstance(policy, dict) else None
    return review if isinstance(review, dict) else {}


def network_policy(policy: dict | None = None) -> dict:



    src = get_detection_policy() if policy is None else policy
    val = src.get("network") if isinstance(src, dict) else None
    return val if isinstance(val, dict) else {}


def gate_policy(policy: dict | None = None) -> dict:



    src = get_detection_policy() if policy is None else policy
    val = src.get("gate") if isinstance(src, dict) else None
    return val if isinstance(val, dict) else {}


def saturation_policy(policy: dict | None = None) -> dict:

    sat = seed_policy(policy).get("saturation")
    return sat if isinstance(sat, dict) else {}


def exemplar_policy(policy: dict | None = None) -> dict:

    policy = get_detection_policy() if policy is None else policy
    exemplar = policy.get("exemplar") if isinstance(policy, dict) else None
    return exemplar if isinstance(exemplar, dict) else {}


def auto_regularize_policy(policy: dict | None = None) -> dict:



    policy = get_detection_policy() if policy is None else policy
    val = policy.get("auto_regularize") if isinstance(policy, dict) else None
    return val if isinstance(val, dict) else {}


def prompt_policy(policy: dict | None = None) -> dict:




    policy = get_detection_policy() if policy is None else policy
    prompt = policy.get("prompt") if isinstance(policy, dict) else None
    return prompt if isinstance(prompt, dict) else {}
