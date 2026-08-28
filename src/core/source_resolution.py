"""How fine a render may go before it is only interpolating the source.

The Automatic grid already refuses to render a local raster much finer than
its own pixels, and an online tiled source much finer than the deepest level
its tile pyramid declares (``ui/plugin/auto_flow._grid_for_detail``). This
module holds the two values that decision reads, so both can be retuned from
the server instead of waiting for a plugin release.

Two separate questions, two separate keys:

- **How far past the source may a render go at all.** One linear factor. A
  little upsampling is not waste: the service reads every tile through a fixed
  window, so a finer grid makes a small object larger inside that window. Far
  past the source it is interpolation, paid for in tiles and answered with
  blur.
- **What the source really holds.** A tile pyramid declares zoom levels, not
  imagery. A satellite basemap commonly serves its deepest levels by enlarging
  a shallower one, so the declared depth can promise several times the
  resolution the pictures carry, and a clamp that trusts it never fires. When
  the server names a source's real ground resolution, that number outranks the
  declared one.

Both fail OPEN, and the shipped fallbacks are exactly what the plugin did
before this module existed: the shipped allowance, and no per-source override
at all. A cold cache, an old server or a malformed value all leave today's
behaviour untouched.
"""
from __future__ import annotations

from .tile_manager import NATIVE_OVERSAMPLE_MAX

# The band a served allowance is held inside. Wider than the shipped constant
# in both directions, because the right allowance depends on the source: what
# the answer needs is a tile covering a given amount of GROUND, and the coarser
# the source, the more the render has to go past it to put that much ground on
# one tile. A single shipped factor cannot be right for a 5 cm drone survey and
# for a half-metre basemap at once.
#
# Bounded all the same. At the low end 1.0 is "never go finer than the source",
# which is a real setting and the coarsest one worth offering. At the high end
# the tile cap and the seed cap already stop a run, but a bad value should be
# refused before it turns into a walk over thousands of levels.
OVERSAMPLE_ALLOWANCE_MIN = 1.0
OVERSAMPLE_ALLOWANCE_MAX = 8.0

# What the plugin uses when the server says nothing: the shipped constant, so
# an absent dial is exactly the behaviour this had before the dial existed.
OVERSAMPLE_ALLOWANCE_DEFAULT = NATIVE_OVERSAMPLE_MAX

# Server key holding the per-source real-resolution table, under `seed`.
_SOURCE_TABLE_KEY = "source_native_mupp"


def oversample_allowance(policy: dict | None = None) -> float:
    """How far past the source's own resolution a render may go (linear).

    Read from the server, then held inside ``[OVERSAMPLE_ALLOWANCE_MIN,
    OVERSAMPLE_ALLOWANCE_MAX]``. An absent, malformed or out-of-band value
    falls back to the shipped constant, so nothing changes until a deploy says
    otherwise. At 1.0 the render never goes finer than the source at all.
    """
    from .detection_policy import native_oversample_max

    try:
        value = float(native_oversample_max(policy))
    except (TypeError, ValueError):
        return OVERSAMPLE_ALLOWANCE_DEFAULT
    if value != value or value in (float("inf"), float("-inf")):
        return OVERSAMPLE_ALLOWANCE_DEFAULT
    if not OVERSAMPLE_ALLOWANCE_MIN <= value <= OVERSAMPLE_ALLOWANCE_MAX:
        return OVERSAMPLE_ALLOWANCE_DEFAULT
    return value


def source_floor_mupp_m(source_uri: str, policy: dict | None = None) -> float:
    """Ground metres per pixel the server says this source really holds, or 0.0.

    ``source_uri`` is the layer's own data-source string. The served table is a
    list of ``{"match": <text>, "mupp": <metres per pixel>}``; the first entry
    whose ``match`` appears in the lower-cased URI wins, so order is priority
    and a narrower rule is listed before a broader one.

    0.0 means no rule matched, and 0.0 is also the whole shipped table: the
    plugin names no source, so nothing here is a claim about anyone's imagery
    until a deploy makes one. Never raises.
    """
    entries = _source_table(policy)
    if not entries:
        return 0.0
    try:
        text = (source_uri or "").lower()
    except (AttributeError, TypeError):
        return 0.0
    if not text:
        return 0.0
    for entry in entries:
        match = entry.get("match")
        if not isinstance(match, str) or not match:
            continue
        if match.lower() not in text:
            continue
        mupp = _positive_float(entry.get("mupp"))
        if mupp > 0:
            return mupp
    return 0.0


def _source_table(policy: dict | None) -> list[dict]:
    """The served per-source table, or an empty list.

    Every non-dict row is dropped rather than failing the whole table, so one
    bad entry in a deploy cannot take the good ones down with it.
    """
    from .detection_policy import seed_policy

    try:
        raw = seed_policy(policy).get(_SOURCE_TABLE_KEY)
    except (AttributeError, TypeError):
        return []
    if not isinstance(raw, list):
        return []
    return [row for row in raw if isinstance(row, dict)]


def _positive_float(value: object) -> float:
    """``value`` as a real positive float, else 0.0.

    Server payloads arrive unvalidated, and ``bool`` is an ``int`` in Python,
    so ``True`` would otherwise read as the number 1.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        return 0.0
    return value if value > 0 else 0.0
