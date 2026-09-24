







from __future__ import annotations

from .detection_policy_core import _is_finite_policy_value, seed_policy






def _tile_plan_block(policy: dict | None) -> dict:

    block = seed_policy(policy).get("tile_plan")
    return block if isinstance(block, dict) else {}


def tile_plan_enabled(policy: dict | None = None) -> bool:


    return _tile_plan_block(policy).get("enabled") is True


def tile_plan_half_steps(policy: dict | None = None) -> int:

    val = _tile_plan_block(policy).get("slider_half_steps")
    if _is_finite_policy_value(val) and 1 <= val <= 12:
        return int(val)
    from .tile_plan import TILE_STEP_HALF_COUNT
    return TILE_STEP_HALF_COUNT


def tile_plan_step_ratio(policy: dict | None = None) -> float:


    val = _tile_plan_block(policy).get("slider_step_ratio")
    if _is_finite_policy_value(val) and 1.05 <= val <= 2.0:
        return float(val)
    from .tile_plan import TILE_STEP_RATIO
    return TILE_STEP_RATIO


def tile_fit_object_frac(policy: dict | None = None) -> float:


    val = seed_policy(policy).get("max_object_tile_frac")
    if _is_finite_policy_value(val) and 0 < val <= 1:
        return float(val)
    from .tile_plan import TILE_FIT_OBJECT_FRAC
    return TILE_FIT_OBJECT_FRAC


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
