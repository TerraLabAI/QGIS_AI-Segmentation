







from __future__ import annotations

from .detection_policy_core import policy_scope, seed_policy
from .served_config import require_served_int, require_served_number


def _tile_plan_block(policy: dict | None) -> dict:

    block = seed_policy(policy).get("tile_plan")
    return block if isinstance(block, dict) else {}


def tile_plan_enabled(policy: dict | None = None) -> bool:


    return _tile_plan_block(policy).get("enabled") is True


def tile_plan_half_steps(policy: dict | None = None) -> int:


    with policy_scope(policy):
        return require_served_int("detection_policy.seed.tile_plan.slider_half_steps", 1, 12)


def tile_plan_step_ratio(policy: dict | None = None) -> float:


    with policy_scope(policy):
        return require_served_number("detection_policy.seed.tile_plan.slider_step_ratio", 1.05, 2.0)


def tile_fit_object_frac(policy: dict | None = None) -> float:



    with policy_scope(policy):
        return require_served_number("detection_policy.seed.max_object_tile_frac", 0.01, 1.0)


def density_probe_config(policy: dict | None = None):



    block = seed_policy(policy).get("density_probe")
    if not isinstance(block, dict) or block.get("enabled") is not True:
        return None
    from .density_probe import config_from_block
    return config_from_block(block)


def run_plan_tile_block(plan: object) -> dict:

    if not isinstance(plan, dict):
        return {}
    block = plan.get("tile_plan")
    return block if isinstance(block, dict) else {}
