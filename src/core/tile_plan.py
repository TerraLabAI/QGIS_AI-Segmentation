














from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .tile_manager import OVERLAP_FRACTION, TILE_SIZE




TILE_FIT_OBJECT_FRAC = 0.40






TILE_STEP_HALF_COUNT = 8
TILE_STEP_RATIO = 2.0 ** 0.25


TILE_BAND_RATIO = 2.0


_CAP_GROWTH = 1.25
_CAP_GROWTH_STEPS = 64
_CAP_BISECT_STEPS = 12


REASON_PRIOR_PLAN = "prior_plan"
REASON_PRIOR_TIER = "prior_tier"
REASON_PRIOR_EXEMPLAR = "prior_exemplar"
REASON_PRIOR_DEFAULT = "prior_default"
REASON_FIT = "fit"
REASON_DENSITY = "density"
REASON_BAND_MIN = "band_min"
REASON_BAND_MAX = "band_max"
REASON_ROUTE_FLOOR = "route_floor"
REASON_SOURCE_FLOOR = "source_floor"
REASON_IMAGERY_FLOOR = "imagery_floor"
REASON_CAP_TILES = "cap_tiles"
REASON_PADDED = "padded"

_PRIOR_REASONS = frozenset((
    REASON_PRIOR_PLAN, REASON_PRIOR_TIER, REASON_PRIOR_EXEMPLAR,
    REASON_PRIOR_DEFAULT))


def _positive(value: object) -> float:

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    val = float(value)
    return val if math.isfinite(val) and val > 0 else 0.0


@dataclass(frozen=True)
class TilePrior:









    tile_ground_m: float = 0.0
    source: str = REASON_PRIOR_DEFAULT
    band_m: tuple[float, float] | None = None
    size_m: float = 0.0
    fit_frac: float = TILE_FIT_OBJECT_FRAC
    route_floor_m: float = 0.0
    ceiling_m: float = 0.0
    default_tile_ground_m: float = 0.0




    drawn_m: float = 0.0
    drawn_min_px: float = 0.0



    drawn_only: bool = False


@dataclass(frozen=True)
class TileSource:



    native_m: float = 0.0
    allowance: float = 1.0
    imagery_floor_m: float = 0.0


@dataclass(frozen=True)
class TileZone:



    width_m: float = 0.0
    height_m: float = 0.0
    count_tiles: Callable[[float], int] | None = None


@dataclass(frozen=True)
class TileCaps:


    max_tiles: int = 0


@dataclass(frozen=True)
class TileDensity:


    tile_ground_m: float = 0.0


@dataclass(frozen=True)
class TilePlan:


    tile_ground_m: float
    band_m: tuple[float, float]
    reasons: tuple[str, ...]
    prior_m: float
    floor_m: float = 0.0
    tiles: int = -1
    padded: bool = False




    uncapped_m: float = 0.0
    sound_max_m: float = 0.0

    @property
    def reason_text(self) -> str:

        return ",".join(self.reasons)


def tile_count_for_side(width_m: float, height_m: float, side_m: float) -> int:



    width_m = _positive(width_m)
    height_m = _positive(height_m)
    side_m = _positive(side_m)
    if not (width_m and height_m and side_m):
        return 0
    stride = int(TILE_SIZE * (1.0 - OVERLAP_FRACTION))
    mupp = side_m / TILE_SIZE

    def per_axis(span_m: float) -> int:
        span = int(span_m / mupp)
        if span <= TILE_SIZE or stride <= 0:
            return 1
        return (span - TILE_SIZE + stride - 1) // stride + 1

    return per_axis(width_m) * per_axis(height_m)


def _count_at(zone: TileZone, side_m: float) -> int:


    if zone.count_tiles is not None:
        try:
            count = zone.count_tiles(side_m)
        except Exception:  # noqa: BLE001
            count = None
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            return count
    return tile_count_for_side(zone.width_m, zone.height_m, side_m)


def _grow_until_fits(zone: TileZone, side_m: float, cap: int) -> tuple[float, int]:


    longer = max(_positive(zone.width_m), _positive(zone.height_m))
    low = side_m
    high = side_m
    count = _count_at(zone, high)
    for _ in range(_CAP_GROWTH_STEPS):
        if count <= cap or (longer and high >= longer):
            break
        low = high
        high *= _CAP_GROWTH
        if longer and high > longer:

            high = max(longer, low)
        count = _count_at(zone, high)
    if count > cap:
        return high, count
    best, best_count = high, count
    for _ in range(_CAP_BISECT_STEPS):
        mid = math.sqrt(low * best)
        mid_count = _count_at(zone, mid)
        if mid_count <= cap:
            best, best_count = mid, mid_count
        else:
            low = mid
        if best / low < 1.01:
            break
    return best, best_count


def drawn_band_m(plan: TilePrior) -> tuple[float, float]:




    drawn = _positive(plan.drawn_m)
    frac = _positive(plan.fit_frac)
    min_px = _positive(plan.drawn_min_px)
    if not (drawn and min_px and 0 < frac <= 1):
        return 0.0, 0.0
    return drawn / frac, drawn * TILE_SIZE / min_px


def resolve_tile_ground(
    plan: TilePrior,
    source: TileSource,
    zone: TileZone,
    caps: TileCaps,
    probe: TileDensity | None = None,
) -> TilePlan:

    reasons: list[str] = []

    prior = _positive(plan.tile_ground_m)
    prior_reason = plan.source if plan.source in _PRIOR_REASONS else REASON_PRIOR_DEFAULT
    if not prior:
        prior = _positive(plan.default_tile_ground_m) or 143.0
        prior_reason = REASON_PRIOR_DEFAULT
    reasons.append(prior_reason)
    side = prior

    frac = _positive(plan.fit_frac)
    size_m = _positive(plan.size_m)
    if size_m and 0 < frac <= 1 and size_m / frac > side:
        side = size_m / frac
        reasons.append(REASON_FIT)

    route_floor = _positive(plan.route_floor_m)
    ceiling = _positive(plan.ceiling_m)
    band_lo = band_hi = 0.0
    if plan.band_m is not None:
        try:
            band_lo = _positive(plan.band_m[0])
            band_hi = _positive(plan.band_m[1])
        except (TypeError, IndexError, KeyError):
            band_lo = band_hi = 0.0
        if band_lo and band_hi and band_lo > band_hi:
            band_lo = band_hi = 0.0
    if not band_lo:
        band_lo = side / TILE_BAND_RATIO
    if not band_hi:
        band_hi = side * TILE_BAND_RATIO
    if ceiling:
        band_hi = min(band_hi, ceiling)
    if route_floor:
        band_lo = max(band_lo, route_floor)
    drawn_lo, drawn_hi = drawn_band_m(plan)
    if drawn_lo:




        drawn_lo = max(drawn_lo, route_floor)
        drawn_hi = min(drawn_hi, ceiling) if ceiling else drawn_hi
        if plan.drawn_only:
            band_lo, band_hi = drawn_lo, max(drawn_hi, drawn_lo)
        else:
            band_lo = min(band_lo, drawn_lo)
            band_hi = max(band_hi, drawn_hi)


    band_lo = min(band_lo, side)
    band_hi = max(band_hi, side, band_lo)

    if probe is not None:
        want = _positive(probe.tile_ground_m)
        if want and abs(want - side) > 1e-6:



            side = want
            if route_floor:
                side = max(side, route_floor)
            if ceiling:
                side = min(side, ceiling)
            band_lo = min(band_lo, side)
            band_hi = max(band_hi, side)
            reasons.append(REASON_DENSITY)

    sound_max = band_hi
    if side < band_lo:
        side = band_lo
        reasons.append(REASON_BAND_MIN)
    elif side > band_hi:
        side = band_hi
        reasons.append(REASON_BAND_MAX)

    floor_m = 0.0
    if route_floor and side < route_floor:
        side = route_floor
        reasons.append(REASON_ROUTE_FLOOR)
    floor_m = max(floor_m, route_floor)
    native = _positive(source.native_m)
    allowance = _positive(source.allowance) or 1.0
    if native:
        source_floor = TILE_SIZE * native / allowance
        floor_m = max(floor_m, source_floor)
        if side < source_floor:
            side = source_floor
            reasons.append(REASON_SOURCE_FLOOR)
    imagery = _positive(source.imagery_floor_m)
    if imagery:
        imagery_floor = TILE_SIZE * imagery
        floor_m = max(floor_m, imagery_floor)
        if side < imagery_floor:
            side = imagery_floor
            reasons.append(REASON_IMAGERY_FLOOR)

    uncapped = side
    tiles = -1
    width = _positive(zone.width_m)
    height = _positive(zone.height_m)
    if width and height:
        tiles = _count_at(zone, side)
        cap = caps.max_tiles
        if isinstance(cap, bool) or not isinstance(cap, int):
            cap = 0
        if cap > 0 and tiles > cap:
            side, tiles = _grow_until_fits(zone, side, cap)
            reasons.append(REASON_CAP_TILES)
    padded = bool(width and height and side > max(width, height))
    if padded:
        reasons.append(REASON_PADDED)

    band = (min(band_lo, side), max(band_hi, side))
    return TilePlan(
        tile_ground_m=side, band_m=band, reasons=tuple(reasons),
        prior_m=prior, floor_m=floor_m, tiles=tiles, padded=padded,
        uncapped_m=uncapped, sound_max_m=sound_max)



TILE_WARNING_NONE = ""
TILE_WARNING_CAP = "cap"
TILE_WARNING_IMAGERY = "imagery"


def tile_plan_warning(
    tile_plan: TilePlan, rendered_m: float, ratio: float = TILE_STEP_RATIO,
) -> str:








    ratio = ratio if _positive(ratio) > 1.0 else TILE_STEP_RATIO
    side = _positive(rendered_m) or tile_plan.tile_ground_m
    sound = _positive(tile_plan.sound_max_m)
    past_band = bool(sound) and side > sound * 1.0001
    if REASON_CAP_TILES in tile_plan.reasons:
        uncapped = _positive(tile_plan.uncapped_m)
        moved = bool(uncapped) and tile_plan.tile_ground_m >= uncapped * ratio * 0.999
        if moved or past_band:
            return TILE_WARNING_CAP
    if past_band:
        return TILE_WARNING_IMAGERY
    return TILE_WARNING_NONE


def tile_step_count(half_steps: int = TILE_STEP_HALF_COUNT) -> int:

    return 2 * max(1, int(half_steps)) + 1


def tile_step_ground_m(
    tile_plan: TilePlan, step: int,
    half_steps: int = TILE_STEP_HALF_COUNT, ratio: float = TILE_STEP_RATIO,
) -> float:


    half = max(1, int(half_steps))
    ratio = ratio if _positive(ratio) > 1.0 else TILE_STEP_RATIO
    step = min(max(1, int(step)), 2 * half + 1)
    return tile_plan.tile_ground_m * ratio ** (half + 1 - step)


def tile_step_window(
    tile_plan: TilePlan, machine_max_tiles: int,
    count_tiles: Callable[[float], int] | None = None,
    half_steps: int = TILE_STEP_HALF_COUNT, ratio: float = TILE_STEP_RATIO,
) -> tuple[int, int, int]:





    half = max(1, int(half_steps))
    centre = half + 1
    lo_m, hi_m = tile_plan.band_m
    coarsest = centre
    for step in range(centre - 1, 0, -1):
        if tile_step_ground_m(tile_plan, step, half, ratio) > hi_m * 1.0001:
            break
        coarsest = step
    machine = centre
    finest = centre
    for step in range(centre + 1, 2 * half + 2):
        side = tile_step_ground_m(tile_plan, step, half, ratio)
        if tile_plan.floor_m and side < tile_plan.floor_m * 0.9999:
            break
        if machine_max_tiles > 0 and count_tiles is not None:
            try:
                count = count_tiles(side)
            except Exception:  # noqa: BLE001
                break
            if count > machine_max_tiles:
                break
        machine = step
        if side >= lo_m * 0.9999:
            finest = step
    return coarsest, finest, machine
