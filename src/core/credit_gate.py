"""Pure billing arithmetic for Automatic (Pro) mode.

Automatic runs spend credits (1 tile = 1 credit). Every decision that could
double-charge past a user's balance or block a run that should be allowed goes
through the functions here, so the boundary semantics live in ONE tested place
instead of being re-derived at each call site.

Pure Python: no Qt, no QGIS, and no server-policy import at module load. The
free-run fraction is passed in by the caller as a plain float; the monthly
allowance is read lazily inside the one function that needs it, and only when
the server sent none. Both fail open, so this module stays importable in a
bare Python process for tests.

Balance boundary (shared by every gate in this module): a run is blocked only
when its tile count STRICTLY exceeds the known balance. A run whose count
equals the balance is allowed, so a user may spend down to exactly zero. An
unknown balance (``None``) never blocks (the caller has no number to gate on).

Two of the numbers here are server dials, both defaulting to exactly today's
behaviour. They read the cached configuration only, never the network, because
the cost of a run is shown to the user BEFORE any call: an estimate that came
from one source and a bill that came from another would disagree, which is the
one thing this module exists to prevent.
"""
from __future__ import annotations

import math


def credits_per_tile() -> float:
    """How many credits one tile costs. One, unless the server says otherwise.

    The single place the tile-to-credit rate is decided, so nothing downstream
    re-derives it.

    Capped at one, and that cap is the point. The plugin shows the cost of a
    run BEFORE any call, as one number that reads as both the tile count and
    the credit count. A rate ABOVE one would bill more than that number said
    and could block a run the user was told they could afford, which is the
    disagreement this module exists to prevent. A rate BELOW one only ever
    bills less than was shown, so it is safe from a screen that cannot express
    it: that is the discount lever, and it is the one worth having during an
    outage. Zero is refused too, since a free run still consumes the service.
    """
    try:
        from .server_dials import dial_in_range

        return float(dial_in_range("gate.credits_per_tile", 1.0, 0.01, 1.0))
    except Exception:  # noqa: BLE001 -- billing math must never break on config
        return 1.0


def balance_slack() -> int:
    """Extra credits a run may exceed the balance by. Zero by default.

    The fleet-wide overdraft lever: during an outage a positive value lets
    runs through that the balance would otherwise refuse, without shipping
    anything. It only ever ALLOWS, never blocks: a negative served value is
    refused, so this cannot be used to tighten the gate on anyone.
    """
    try:
        from .server_dials import dial_in_range

        return int(dial_in_range("gate.balance_slack", 0, 0, 100_000))
    except Exception:  # noqa: BLE001 -- billing math must never break on config
        return 0


def run_cost(tiles: int) -> int:
    """The credits a ``tiles``-tile run costs, rounded up.

    Rounded up so the number shown is never lower than what is charged. With
    the default rate of one credit per tile this is the tile count itself,
    which is what every shipped client and every existing run assumes.
    """
    rate = credits_per_tile()
    if rate == 1.0:
        return int(tiles)
    return math.ceil(float(tiles) * rate)


def low_credit_threshold() -> float:
    """Below this fraction of the allowance, the balance reads as low.

    Shared so the footer ring and the Start step's "Running low" note always
    turn amber at the same moment: both read it from here, neither keeps a
    copy.
    """
    try:
        from .server_dials import dial_in_range

        return float(dial_in_range("gate.low_credit_threshold", 0.20, 0.0, 1.0))
    except Exception:  # noqa: BLE001 -- a badge colour must never break a paint
        return 0.20


def low_credit_ceiling() -> int:
    """Above this many credits left, nothing reads as low, whatever the share.

    The share alone opened the warning far too early on a large allowance: a
    fifth of the free trial is sixty credits, which is dozens of objects still
    to save, and a user with that much left was already being asked to pay. The
    two rules are an AND, so the warning needs a small share AND a small
    number.

    Served, because when to start selling is the kind of number that gets
    retuned on what users do, and a plugin release is days away from the
    decision.
    """
    try:
        from .server_dials import dial_in_range

        return int(dial_in_range("gate.low_credit_ceiling", 30, 0, 100_000))
    except Exception:  # noqa: BLE001 -- a badge colour must never break a paint
        return 30


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

    A single free run may only cost ``fraction`` of the monthly free allowance
    so one Detect can never drain the whole month. ``total`` is the allowance
    the server sent; older servers omit it, and a falsy/``None`` ``total`` then
    falls back to ``detection_policy.free_monthly_allowance()`` (a server dial,
    cache-only and fail-open, so this stays offline-safe). The cap is at least
    1 (a free user can always launch some run). The subscriber "no cap"
    decision is NOT here: it is a tier read the caller makes before calling
    this (see ``credit_snapshot``).
    """
    resolved_total = int(total) if total else _default_monthly_allowance()
    return max(1, int(round(resolved_total * fraction)))


def _default_monthly_allowance() -> int:
    """The allowance assumed when the server's usage payload carries none.

    Read through the policy so the number can move without a plugin release;
    any failure falls back to the same value every shipped client assumes, so
    an offline or older client behaves exactly as before.
    """
    try:
        from .detection_policy import free_monthly_allowance
        return free_monthly_allowance()
    except Exception:  # noqa: BLE001 - policy is optional, never break the gate
        return 200


def run_affordable(tiles: int, balance) -> bool:
    """True when a ``tiles``-tile run may launch against ``balance`` credits.

    An unknown balance (``None``) is always affordable (no number to gate on).
    Otherwise the run is affordable while its cost is at most the balance plus
    whatever slack is in force: it is blocked ONLY when the cost strictly
    exceeds that, so a run that spends down to exactly zero is allowed.
    ``balance`` is coerced with ``int()`` exactly as the call site does (a
    float balance truncates toward zero); a non-numeric balance raises, as it
    would at the call site.
    """
    if balance is None:
        return True
    return run_cost(tiles) <= int(balance) + balance_slack()


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

    Large runs scale the budget with the base grid, floored so a small dense
    zone can still walk the full re-split ladder, under a hard cap. All three
    scalars are server dials (seed.saturation.subdivide_cap_*, cache-only,
    fail-open); the constants here are the generic client fallbacks.
    """
    from .detection_policy import subdivide_cap_params
    hard_cap, floor, scale = subdivide_cap_params(96, 64, 2)
    return min(hard_cap, max(floor, scale * base_tiles))


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
