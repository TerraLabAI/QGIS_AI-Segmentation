"""Pure billing arithmetic for Automatic (Pro) mode.

Automatic runs spend credits (1 tile = 1 credit). Every decision that could
double-charge past a user's balance or block a run that should be allowed goes
through the functions here, so the boundary semantics live in ONE tested place
instead of being re-derived at each call site.

Pure Python: no Qt, no QGIS, and no server-policy import at module load. The
one server-tuned input (the free-run fraction) is passed in by the caller as a
plain float, never imported here, mirroring ``core.detection_policy``'s shape
so this module stays importable in a bare Python process for tests.

Balance boundary (shared by every gate in this module): a run is blocked only
when its tile count STRICTLY exceeds the known balance. A run whose count
equals the balance is allowed, so a user may spend down to exactly zero. An
unknown balance (``None``) never blocks (the caller has no number to gate on).
"""
from __future__ import annotations


def credit_snapshot(usage: dict) -> tuple[int | None, bool]:
    """Return ``(credits_before, is_free_tier)`` from a usage dict.

    ``is_free_tier`` defaults to True (the safe free default) when the field is
    absent, so a not-yet-fetched usage is treated as free. On the free tier the
    credit figure is the raw ``free_detections_remaining`` (``None`` when the
    field is missing, meaning "balance unknown" to the caller). For a subscriber
    the figure is ``max(0, images_limit - images_used)`` with each field
    defaulting to 0 when missing or falsy. Best-effort: never raises on a
    well-formed usage dict.
    """
    is_free = bool(usage.get("is_free_tier", True))
    if is_free:
        credits = usage.get("free_detections_remaining")
    else:
        used = usage.get("images_used", 0) or 0
        limit = usage.get("images_limit", 0) or 0
        credits = max(0, limit - used)
    return credits, is_free


def free_run_tile_cap(total: int | None, fraction: float) -> int:
    """The per-run tile (credit) cap for a free-tier user.

    A single free run may only cost ``fraction`` of the lifetime free allowance
    so one Detect can never drain the whole trial. ``total`` is the lifetime
    allowance (older servers omit it, so a falsy/``None`` ``total`` falls back to
    300, the known lifetime cap). The cap is at least 1 (a free user can always
    launch some run). The subscriber "no cap" decision is NOT here: it is a tier
    read the caller makes before calling this (see ``credit_snapshot``).
    """
    resolved_total = int(total or 300)
    return max(1, int(round(resolved_total * fraction)))


def run_affordable(tiles: int, balance) -> bool:
    """True when a ``tiles``-tile run may launch against ``balance`` credits.

    An unknown balance (``None``) is always affordable (no number to gate on).
    Otherwise the run is affordable while ``tiles <= int(balance)``: it is
    blocked ONLY when ``tiles`` strictly exceeds the balance, so a run that
    spends down to exactly zero is allowed. ``balance`` is coerced with
    ``int()`` exactly as the call site does (a float balance truncates toward
    zero); a non-numeric balance raises, as it would at the call site.
    """
    if balance is None:
        return True
    return tiles <= int(balance)


def insufficient(estimate: int, remaining) -> bool:
    """True when a run ``estimate`` exceeds the known ``remaining`` balance and
    Detect must be blocked as under-funded.

    Exact logical inverse of ``run_affordable``: the pre-submit re-gate in
    ``auto_run`` and the live cost-line gate in ``auto_state`` share ONE
    boundary (block only when the count strictly exceeds the balance), so they
    are defined against a single primitive here and can never drift apart.
    """
    return not run_affordable(estimate, remaining)


def subdivide_cap(base_tiles: int) -> int:
    """Absolute ceiling on extra tiles a run may spend re-splitting saturated
    tiles, BEFORE any credit clamp.

    Large runs scale the budget at 2x the base grid; a floor of 64 lets a small
    dense zone still walk the full re-split ladder, and 96 is the hard cap.
    """
    return min(96, max(64, 2 * base_tiles))


def subdivide_budget(credits, base_tiles: int, every: int) -> int:
    """Extra re-split tiles a run may spend, clamped to the credits left after
    the base grid so a re-split can never be the thing that trips mid-run
    exhaustion on its own.

    The server bills 1 credit per ``every`` re-split quadrants:
      - ``every <= 0``: quadrants ride the parent tile's charge (never billed),
        so no clamp applies and the full ``subdivide_cap`` is available.
      - ``credits`` unknown (``None``): the budget also falls back to the cap
        (best-effort, the caller has no balance to clamp against).
      - otherwise: the credits left after the base grid stretch ``every`` times
        as far, clamped into ``[0, cap]`` as
        ``max(0, min(cap, (int(credits) - base_tiles) * every))``.
    """
    cap = subdivide_cap(base_tiles)
    if every <= 0:
        return cap
    if credits is None:
        return cap
    return max(0, min(cap, (int(credits) - base_tiles) * every))
