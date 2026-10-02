







from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard


def _is_finite_policy_value(value: object) -> TypeGuard[int | float]:

    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False







_run_policy_lock = threading.Lock()
_run_policy: dict | None = None
_scope = threading.local()


def _cached_config_policy() -> dict:
    try:
        from .activation_manager import get_server_config

        config = get_server_config()
    except Exception:  # noqa: BLE001  # nosec B110
        return {}
    if not isinstance(config, dict):
        return {}
    policy = config.get("detection_policy")
    return policy if isinstance(policy, dict) else {}


def capture_run_policy(plan: object = None) -> dict:





    run_policy = plan.get("run_policy") if isinstance(plan, dict) else None
    policy = run_policy if isinstance(run_policy, dict) and run_policy else _cached_config_policy()
    pin_run_policy(policy)
    return policy


def pin_run_policy(policy: object) -> None:


    global _run_policy
    with _run_policy_lock:
        _run_policy = policy if isinstance(policy, dict) and policy else None


def release_run_policy() -> None:

    pin_run_policy(None)


def pinned_run_policy() -> dict | None:

    with _run_policy_lock:
        return _run_policy


@contextmanager
def policy_scope(policy: dict | None):


    if policy is None:
        yield
        return
    previous = getattr(_scope, "policy", None)
    _scope.policy = policy if isinstance(policy, dict) else {}
    try:
        yield
    finally:
        _scope.policy = previous


def get_detection_policy() -> dict:






    scoped = getattr(_scope, "policy", None)
    if isinstance(scoped, dict):
        return scoped
    pinned = pinned_run_policy()
    if pinned is not None:
        return pinned
    return _cached_config_policy()


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
