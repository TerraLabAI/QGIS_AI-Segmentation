


























from __future__ import annotations

import math


def credits_per_tile() -> float:














    try:
        from .server_dials import dial_in_range

        return float(dial_in_range("gate.credits_per_tile", 1.0, 0.01, 1.0))
    except Exception:  # noqa: BLE001
        return 1.0


def balance_slack() -> int:







    try:
        from .server_dials import dial_in_range

        return int(dial_in_range("gate.balance_slack", 0, 0, 100_000))
    except Exception:  # noqa: BLE001
        return 0


def run_cost(tiles: int) -> int:






    rate = credits_per_tile()
    if rate == 1.0:


        return math.ceil(float(tiles))
    return math.ceil(float(tiles) * rate)


def low_credit_threshold() -> float:






    try:
        from .server_dials import dial_in_range

        return float(dial_in_range("gate.low_credit_threshold", 0.20, 0.0, 1.0))
    except Exception:  # noqa: BLE001
        return 0.20


def low_credit_ceiling() -> int:












    try:
        from .server_dials import dial_in_range

        return int(dial_in_range("gate.low_credit_ceiling", 30, 0, 100_000))
    except Exception:  # noqa: BLE001
        return 30


def credit_snapshot(usage: dict) -> tuple[int | None, bool]:










    is_free = bool(usage.get("is_free_tier", True))
    if is_free:
        credits = usage.get("free_detections_remaining")
    else:
        used = usage.get("images_used", 0) or 0
        limit = usage.get("images_limit", 0) or 0
        credits = max(0, limit - used)
    return credits, is_free


def free_run_tile_cap(total: int | None, fraction: float) -> int:











    try:
        resolved_total = int(total) if total else _default_monthly_allowance()
    except (TypeError, ValueError):


        resolved_total = _default_monthly_allowance()
    return max(1, int(round(resolved_total * fraction)))


def _default_monthly_allowance() -> int:






    try:
        from .detection_policy import free_monthly_allowance
        return free_monthly_allowance()
    except Exception:  # noqa: BLE001
        return 200


def run_affordable(tiles: int, balance) -> bool:










    if balance is None:
        return True
    return run_cost(tiles) <= int(balance) + balance_slack()


def subdivide_cap(base_tiles: int) -> int:








    from .detection_policy import subdivide_cap_params
    hard_cap, floor, scale = subdivide_cap_params(96, 64, 2)
    return min(hard_cap, max(floor, scale * base_tiles))


def subdivide_budget(credits, base_tiles: int, every: int) -> int:













    cap = subdivide_cap(base_tiles)
    if every <= 0:
        return cap
    if credits is None:
        return cap
    return max(0, min(cap, (int(credits) - base_tiles) * every))
