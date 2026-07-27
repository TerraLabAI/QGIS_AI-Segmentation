"""Shareable "run recipe": encode an Automatic run's intent as a short string.

A recipe captures WHAT to segment and WHERE, so another user (or the same user
later) can reproduce a run without retracing the zone or retyping the prompt: the
object prompt, the drawn zone (as WGS84 lon/lat vertices), the detail level, the
review confidence, and the refine settings. It pairs with the run-replay path: a
"it detected badly" report becomes a one-paste reproduction.

Security invariant, enforced by SHAPE: a recipe describes the GESTURE, never the
data or the identity. There is deliberately no field for the raster path, the
activation key, a layer name, a URL, or any coordinate outside the drawn zone.
What the schema cannot hold, it cannot leak.

Pure module: no QGIS import, so the encode/decode core is unit-tested outside a
QGIS runtime. The plugin side reprojects the zone to WGS84 before ``encode`` and
back to the run CRS after ``decode``; this module only ever sees lon/lat.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field

from .review_defaults import (
    AUTO_DEFAULT_CONFIDENCE,
    AUTO_REVIEW_CLEAN_DEFAULT,
    AUTO_REVIEW_EXPAND_DEFAULT,
    AUTO_REVIEW_FILL_HOLES_DEFAULT,
    AUTO_REVIEW_FILL_HOLES_MAX_M2_DEFAULT,
    AUTO_REVIEW_ORTHO_DEFAULT,
    AUTO_REVIEW_SIMPLIFY_DEFAULT,
    AUTO_REVIEW_SMOOTH_DEFAULT,
)

# Token scheme + version. A future breaking change bumps the number; a decoder
# that meets an unknown version says so plainly rather than mis-parsing.
RECIPE_SCHEME = "aiseg1"

# Bounds so a hostile or corrupt token can never allocate unbounded work. These
# guard the decoder; a violation is a typed error the caller can branch on.
_MAX_TOKEN_CHARS = 8192
_MAX_ZONE_POINTS = 2000
_MAX_PROMPT_CHARS = 200
_COORD_DECIMALS = 6

# Refine keys carried in a recipe, each paired with its Automatic-review default.
# A field equal to its default is omitted from the token, so a plain run stays
# short and only what the user changed travels.
_REFINE_DEFAULTS: dict[str, float | bool] = {
    "simplify": AUTO_REVIEW_SIMPLIFY_DEFAULT,
    "clean": AUTO_REVIEW_CLEAN_DEFAULT,
    "smooth": AUTO_REVIEW_SMOOTH_DEFAULT,
    "ortho": AUTO_REVIEW_ORTHO_DEFAULT,
    "expand": AUTO_REVIEW_EXPAND_DEFAULT,
    "fill_holes": AUTO_REVIEW_FILL_HOLES_DEFAULT,
    # Fill holes smaller than this ground area (m2); 0 = every hole. An older
    # decoder drops the key (unknown keys are ignored) and uses its own default.
    # A token only ever carries what the user CHANGED, so both these defaults
    # resolve to whatever the build that reads the token starts runs at, which
    # is the same thing the reader would get from a fresh run.
    "fill_holes_max": AUTO_REVIEW_FILL_HOLES_MAX_M2_DEFAULT,
}


class RecipeError(ValueError):
    """A token is malformed, out of bounds, or of an unknown version.

    Subclasses ValueError so a generic caller treats it as bad input, while a
    caller that cares can catch this specific type and show a precise message.
    """


@dataclass(frozen=True)
class RunRecipe:
    """One Automatic run's reproducible intent. See module docstring for the
    security invariant: no data path, key, layer name, or URL, by construction."""

    prompt: str
    detail: int
    zone_lonlat: list[tuple[float, float]]
    confidence: float = AUTO_DEFAULT_CONFIDENCE
    refine: dict[str, float | bool] = field(default_factory=dict)

    def normalized_refine(self) -> dict[str, float | bool]:
        """Refine with every default filled in, for the plugin to apply."""
        merged = dict(_REFINE_DEFAULTS)
        for k, v in self.refine.items():
            if k in _REFINE_DEFAULTS:
                merged[k] = v
        return merged


def _round_coord(value: float) -> float:
    return round(float(value), _COORD_DECIMALS)


def _validate_ring(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 3:
        raise RecipeError("a zone needs at least 3 points")
    if len(points) > _MAX_ZONE_POINTS:
        raise RecipeError("zone has too many points")
    ring: list[tuple[float, float]] = []
    for pt in points:
        try:
            lon, lat = float(pt[0]), float(pt[1])
        except (TypeError, ValueError, IndexError) as err:
            raise RecipeError("zone point is not a lon/lat pair") from err
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            raise RecipeError("zone point is outside lon/lat range")
        ring.append((_round_coord(lon), _round_coord(lat)))
    return ring


def encode(recipe: RunRecipe) -> str:
    """Serialize a recipe to a ``aiseg1:<base64>`` token.

    Fields equal to their default are omitted so a plain run yields a short
    token. Coordinates are rounded to ~0.1 m (6 decimals)."""
    ring = _validate_ring(recipe.zone_lonlat)
    prompt = (recipe.prompt or "").strip()[:_MAX_PROMPT_CHARS]
    payload: dict[str, object] = {
        "p": prompt,
        "d": int(recipe.detail),
        # Flat [lon, lat, lon, lat, ...] is more compact than nested pairs.
        "z": [c for pt in ring for c in pt],
    }
    if round(float(recipe.confidence), 4) != round(float(AUTO_DEFAULT_CONFIDENCE), 4):
        payload["c"] = round(float(recipe.confidence), 4)
    trimmed = {
        k: v for k, v in recipe.refine.items()
        if k in _REFINE_DEFAULTS and v != _REFINE_DEFAULTS[k]
    }
    if trimmed:
        payload["r"] = trimmed
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    b64 = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"{RECIPE_SCHEME}:{b64}"


def decode(token: str) -> RunRecipe:
    """Parse a token back into a RunRecipe, or raise RecipeError.

    Rejects unknown schemes/versions, oversize tokens, and malformed payloads;
    fills omitted fields with their defaults."""
    if not isinstance(token, str):
        raise RecipeError("token is not a string")
    token = token.strip()
    if len(token) > _MAX_TOKEN_CHARS:
        raise RecipeError("token is too long")
    prefix = RECIPE_SCHEME + ":"
    if not token.startswith(prefix):
        raise RecipeError("unknown recipe scheme or version")
    b64 = token[len(prefix):]
    padding = "=" * (-len(b64) % 4)
    try:
        raw = base64.urlsafe_b64decode(b64 + padding)
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, TypeError) as err:
        raise RecipeError("token is not valid base64 JSON") from err
    if not isinstance(payload, dict):
        raise RecipeError("token payload is not an object")

    flat = payload.get("z")
    if not isinstance(flat, list) or len(flat) % 2 != 0:
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
        for k, v in r.items():
            if k not in _REFINE_DEFAULTS:
                continue  # ignore unknown keys (forward compatibility)
            # bool is a subclass of int, so (int, float) accepts both; the value
            # keeps its own type (True stays bool, 2.0 stays float).
            if isinstance(v, (int, float)):
                refine[k] = v

    return RunRecipe(
        prompt=prompt,
        detail=int(detail_raw),
        zone_lonlat=ring,
        confidence=confidence,
        refine=refine,
    )
