

















from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass, field

from .review_defaults import (
    AUTO_DEFAULT_CONFIDENCE,
    AUTO_REVIEW_CLEAN_DEFAULT,
    AUTO_REVIEW_EXPAND_DEFAULT,
    AUTO_REVIEW_FILL_HOLES_DEFAULT,
    AUTO_REVIEW_FILL_HOLES_MAX_M2_DEFAULT,
    AUTO_REVIEW_ORTHO_DEFAULT,
    AUTO_REVIEW_POINTS_PCT_DEFAULT,
    AUTO_REVIEW_SIMPLIFY_DEFAULT,
    AUTO_REVIEW_SMOOTH_DEFAULT,
)



RECIPE_SCHEME = "aiseg1"



_MAX_TOKEN_CHARS = 8192
_MAX_ZONE_POINTS = 2000
_MAX_PROMPT_CHARS = 200
_COORD_DECIMALS = 6




_REFINE_DEFAULTS: dict[str, float | bool] = {
    "simplify": AUTO_REVIEW_SIMPLIFY_DEFAULT,
    "clean": AUTO_REVIEW_CLEAN_DEFAULT,
    "smooth": AUTO_REVIEW_SMOOTH_DEFAULT,
    "ortho": AUTO_REVIEW_ORTHO_DEFAULT,
    "expand": AUTO_REVIEW_EXPAND_DEFAULT,
    "fill_holes": AUTO_REVIEW_FILL_HOLES_DEFAULT,


    "fill_holes_max": AUTO_REVIEW_FILL_HOLES_MAX_M2_DEFAULT,
    "min_size_m2": 0.0,
    "max_size_m2": 0.0,
    "points_pct": AUTO_REVIEW_POINTS_PCT_DEFAULT,
    "snap_boundaries": False,
}






_REFINE_BANDS: dict[str, tuple[float, float]] = {
    "simplify": (0.0, 1000.0),
    "clean": (0.0, 50.0),
    "expand": (-1000.0, 1000.0),
    "fill_holes_max": (0.0, 1000000.0),
    "min_size_m2": (0.0, 1e12),
    "max_size_m2": (0.0, 1e12),
    "points_pct": (1.0, 100.0),
}
_REFINE_BOOL_KEYS = ("smooth", "ortho", "fill_holes", "snap_boundaries")
_REFINE_ALIASES = {
    "simplify_px": "simplify",
    "clean_px": "clean", "trim_spikes": "clean", "trim_spikes_px": "clean",
    "round_corners": "smooth", "right_angles": "ortho",
    "expand_px": "expand", "grow_shrink_px": "expand",
    "fill_holes_max_m2": "fill_holes_max",
    "points": "points_pct", "shared_borders": "snap_boundaries",
}


def _normalize_refine(settings: dict) -> dict:
    normalized = {}
    for key, value in settings.items():
        key = _REFINE_ALIASES.get(key, key)
        if key not in _REFINE_DEFAULTS or not isinstance(value, (int, float)):
            continue
        try:
            if not math.isfinite(value):
                continue
        except OverflowError:
            continue
        if key in _REFINE_BOOL_KEYS:
            normalized[key] = bool(value)
        else:
            low, high = _REFINE_BANDS[key]
            normalized[key] = min(high, max(low, float(value)))
    return normalized


class RecipeError(ValueError):
    pass






@dataclass(frozen=True)
class RunRecipe:



    prompt: str
    detail: int
    zone_lonlat: list[tuple[float, float]]
    confidence: float = AUTO_DEFAULT_CONFIDENCE
    refine: dict[str, float | bool] = field(default_factory=dict)

    def normalized_refine(self) -> dict[str, float | bool]:

        merged = dict(_REFINE_DEFAULTS)
        merged.update(_normalize_refine(self.refine))
        return merged


def _round_coord(value: float) -> float:
    return round(float(value), _COORD_DECIMALS)


def _validate_ring(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not isinstance(points, (list, tuple)):
        raise RecipeError("zone is missing or malformed")
    if len(points) < 3:
        raise RecipeError("a zone needs at least 3 points")
    if len(points) > _MAX_ZONE_POINTS:
        raise RecipeError("zone has too many points")
    ring: list[tuple[float, float]] = []
    for pt in points:
        if (not isinstance(pt, (list, tuple)) or len(pt) != 2
                or any(isinstance(value, bool) for value in pt)):
            raise RecipeError("zone point is not a lon/lat pair")
        try:
            lon, lat = float(pt[0]), float(pt[1])
        except (TypeError, ValueError, IndexError, OverflowError) as err:
            raise RecipeError("zone point is not a lon/lat pair") from err
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            raise RecipeError("zone point is outside lon/lat range")
        ring.append((_round_coord(lon), _round_coord(lat)))
    if len(set(ring)) < 3:
        raise RecipeError("a zone needs at least 3 distinct points")
    return ring


def encode(recipe: RunRecipe) -> str:




    ring = _validate_ring(recipe.zone_lonlat)
    if not isinstance(recipe.detail, int) or isinstance(recipe.detail, bool) or recipe.detail < 1:
        raise RecipeError("detail is missing or not a positive integer")
    if (not isinstance(recipe.confidence, (int, float)) or isinstance(recipe.confidence, bool)
            or not 0 <= recipe.confidence <= 1):
        raise RecipeError("confidence is out of range")
    if not isinstance(recipe.prompt, str):
        raise RecipeError("prompt is not a string")
    if not isinstance(recipe.refine, dict):
        raise RecipeError("refine is not an object")
    prompt = recipe.prompt.strip()[:_MAX_PROMPT_CHARS]
    payload: dict[str, object] = {
        "p": prompt,
        "d": int(recipe.detail),

        "z": [c for pt in ring for c in pt],
    }
    if round(float(recipe.confidence), 4) != round(float(AUTO_DEFAULT_CONFIDENCE), 4):
        payload["c"] = round(float(recipe.confidence), 4)
    refined = _normalize_refine(recipe.refine)
    if refined:
        payload["r"] = refined
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    b64 = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    token = f"{RECIPE_SCHEME}:{b64}"
    if len(token) > _MAX_TOKEN_CHARS:
        raise RecipeError("token is too long")
    return token


def decode(token: str) -> RunRecipe:




    if not isinstance(token, str):
        raise RecipeError("token is not a string")
    if len(token) > _MAX_TOKEN_CHARS:
        raise RecipeError("token is too long")
    token = token.strip()
    if len(token) > _MAX_TOKEN_CHARS:
        raise RecipeError("token is too long")
    prefix = RECIPE_SCHEME + ":"
    if not token.startswith(prefix):
        raise RecipeError("unknown recipe scheme or version")
    b64 = token[len(prefix):]
    padding = "=" * (-len(b64) % 4)
    try:
        raw = base64.b64decode(b64 + padding, altchars=b"-_", validate=True)
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, TypeError, RecursionError) as err:
        raise RecipeError("token is not valid base64 JSON") from err
    if not isinstance(payload, dict):
        raise RecipeError("token payload is not an object")

    flat = payload.get("z")
    if (not isinstance(flat, list) or len(flat) % 2 != 0
            or len(flat) > 2 * _MAX_ZONE_POINTS):
        raise RecipeError("zone is missing or malformed")
    pairs = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
    ring = _validate_ring(pairs)

    detail_raw = payload.get("d")
    if not isinstance(detail_raw, int) or isinstance(detail_raw, bool) or detail_raw < 1:
        raise RecipeError("detail is missing or not a positive integer")

    prompt = payload.get("p", "")
    if not isinstance(prompt, str):
        raise RecipeError("prompt is not a string")
    prompt = prompt[:_MAX_PROMPT_CHARS]

    confidence = AUTO_DEFAULT_CONFIDENCE
    if "c" in payload:
        c = payload["c"]
        if not isinstance(c, (int, float)) or isinstance(c, bool) or not (0.0 <= c <= 1.0):
            raise RecipeError("confidence is out of range")
        confidence = float(c)

    refine: dict[str, float | bool] = {}
    if "r" in payload:
        r = payload["r"]
        if not isinstance(r, dict):
            raise RecipeError("refine is not an object")
        refine = _normalize_refine(r)

    return RunRecipe(
        prompt=prompt,
        detail=int(detail_raw),
        zone_lonlat=ring,
        confidence=confidence,
        refine=refine,
    )
