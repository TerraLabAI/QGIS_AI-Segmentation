"""Class-based export symbology: one distinct color per class value.

Auto exports carry a ``class`` field (the detection prompt). A run today
always writes one prompt per layer, so every feature shares one class and one
color (see ``output_store.committed_color_for_prompt``). This module adds the
piece for when a layer holds more than one class, either from a future
multi-class run or from a user merging several committed runs by hand: a
categorized color per distinct value, sampled evenly from the plugin's own
brand ramp, so a land-use/land-cover export reads as a flat-color map with one
hue per class.

Pure color math only, no Qt or QGIS import, so the suite runs anywhere.
Building the actual ``QgsCategorizedSymbolRenderer`` is a separate function
(``make_class_categorized_renderer`` in ``layer_conventions``) that imports
``qgis.core`` lazily.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

# Cap on distinct legend colors: past this, extra class values fold into one
# shared fallback color instead of growing the legend without bound.
MAX_CATEGORIES = 24

# Ramp anchors are the plugin's own brand hues, never a new one: brand blue
# -> lime success -> amber warning -> brand red. Interpolating across them
# gives enough hue spread for a
# readable categorized map without introducing a color outside the taxonomy.
DEFAULT_RAMP_ANCHORS: tuple[str, ...] = ("#1e88e5", "#8bac27", "#f5a623", "#d32f2f")

# Shared color for class values folded past the cap (same neutral grey as
# BRAND_GRAY in styles.py, the plugin's existing "neutral utility" hue).
OTHER_COLOR = "#757575"

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
# A ramp is a handful of anchors. Past this it is a mistake, not a ramp.
_MAX_ANCHORS = 16


def legend_category_cap() -> int:
    """How many distinct legend colors a categorized export may show.

    Two is the floor: below that a categorized renderer has nothing to
    categorize.
    """
    try:
        from .server_dials import dial_in_range

        return int(dial_in_range("taxonomy.legend.max_categories", MAX_CATEGORIES, 2, 128))
    except Exception:  # noqa: BLE001 -- symbology is cosmetic  # nosec B110
        return MAX_CATEGORIES


def legend_ramp_anchors() -> tuple[str, ...]:
    """The hues the categorized ramp interpolates between.

    Replaced whole, not merged: a ramp is an ordered whole, and half a served
    ramp spliced into the shipped one would read as neither. Fewer than two
    usable anchors is not a ramp, so the shipped one stands.
    """
    try:
        from .server_dials import read_value

        value = read_value("taxonomy.legend.ramp_anchors")
        if isinstance(value, (list, tuple)):
            anchors = [
                item.strip().lower() for item in value[:_MAX_ANCHORS]
                if isinstance(item, str) and _HEX_RE.match(item.strip())
            ]
            if len(anchors) >= 2:
                return tuple(anchors)
    except Exception:  # noqa: BLE001 -- symbology is cosmetic  # nosec B110
        pass
    return DEFAULT_RAMP_ANCHORS


def legend_other_color() -> str:
    """The one color every class past the cap shares."""
    try:
        from .server_dials import read_value

        value = read_value("taxonomy.legend.other_color")
        if isinstance(value, str) and _HEX_RE.match(value.strip()):
            return value.strip().lower()
    except Exception:  # noqa: BLE001 -- symbology is cosmetic  # nosec B110
        pass
    return OTHER_COLOR


def normalize_hex(value: str) -> str:
    """Normalize a hex color to its lowercase 6-digit form.

    Accepts the 3-digit shorthand ('#abc' -> '#aabbcc') and an optional
    leading '#'. Raises ValueError on anything else, so a bad color fails
    loudly instead of painting the wrong hue.
    """
    text = (value or "").strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        raise ValueError(f"not a hex color: {value!r}")
    int(text, 16)  # raises ValueError on non-hex digits
    return f"#{text.lower()}"


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = normalize_hex(value)[1:]
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    r, g, b = (max(0, min(255, round(channel))) for channel in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def interpolate_colors(anchors: Iterable[str], n: int) -> list[str]:
    """Sample n evenly-spaced hex colors along the piecewise-linear ramp
    defined by anchors.

    n=1 returns the ramp midpoint. n>=2 always includes both ramp ends.
    Needs at least 2 anchors.
    """
    anchor_list = list(anchors)
    if n <= 0:
        return []
    if len(anchor_list) < 2:
        raise ValueError("interpolate_colors needs at least 2 anchors")
    rgb_anchors = [_hex_to_rgb(a) for a in anchor_list]
    segments = len(rgb_anchors) - 1
    positions = [0.5] if n == 1 else [i / (n - 1) for i in range(n)]
    colors = []
    for pos in positions:
        scaled = pos * segments
        seg_index = min(int(scaled), segments - 1)
        t = scaled - seg_index
        r0, g0, b0 = rgb_anchors[seg_index]
        r1, g1, b1 = rgb_anchors[seg_index + 1]
        colors.append(_rgb_to_hex((
            r0 + (r1 - r0) * t,
            g0 + (g1 - g0) * t,
            b0 + (b1 - b0) * t,
        )))
    return colors


def _unique_labels(values: Iterable[str]) -> list[str]:
    """Distinct, non-empty class labels, sorted so the same class set always
    gets the same colors run to run (not insertion order, which would depend
    on feature write order)."""
    labels = {(v or "").strip() for v in values}
    labels.discard("")
    return sorted(labels)


def needs_categorized_renderer(values: Iterable[str]) -> bool:
    """True when values hold 2+ distinct non-empty class labels.

    A single class (the common one-prompt run) keeps the existing one-color
    committed renderer; this only turns on for an actual multi-class export.
    """
    return len(_unique_labels(values)) > 1


def class_color_mapping(
    values: Iterable[str],
    *,
    max_categories: int | None = None,
    ramp_anchors: Iterable[str] | None = None,
) -> dict[str, str]:
    """Map every distinct class value to a hex color sampled evenly from the
    ramp, every value present even past the cap.

    Up to max_categories distinct values get a distinct ramp color. Beyond
    that, the extra values (in sorted order, so the choice is deterministic)
    all share the fold color: the legend never grows past max_categories
    colors. Both arguments default to the values in force (shipped, or what
    the server sent); pass them to pin a mapping to exact numbers.
    """
    if max_categories is None:
        max_categories = legend_category_cap()
    if ramp_anchors is None:
        ramp_anchors = legend_ramp_anchors()
    other = legend_other_color()
    unique = _unique_labels(values)
    if not unique:
        return {}
    if len(unique) <= max_categories:
        colors = interpolate_colors(ramp_anchors, len(unique))
        return dict(zip(unique, colors))
    kept = unique[: max_categories - 1]
    colors = interpolate_colors(ramp_anchors, len(kept))
    mapping = dict(zip(kept, colors))
    for label in unique[max_categories - 1:]:
        mapping[label] = other
    return mapping


def equal_interval_breaks(values: Iterable[float], n: int) -> list[float]:
    """n+1 equal-width bin edges spanning values, capped to the number of
    distinct values so a sparse field never requests more breaks than the
    data supports."""
    numbers = sorted({float(v) for v in values if v is not None})
    if not numbers:
        return []
    n = max(1, min(n, len(numbers)))
    lo, hi = numbers[0], numbers[-1]
    if lo == hi:
        return [lo, hi]
    step = (hi - lo) / n
    return [lo + step * i for i in range(n + 1)]


def quantile_breaks(values: Iterable[float], n: int) -> list[float]:
    """n+1 bin edges splitting values into equal-count groups, capped to the
    number of distinct values."""
    numbers = sorted(float(v) for v in values if v is not None)
    if not numbers:
        return []
    distinct = len(set(numbers))
    n = max(1, min(n, distinct))
    edges = [numbers[0]]
    for i in range(1, n):
        pos = (len(numbers) - 1) * i / n
        lower = int(pos)
        upper = min(lower + 1, len(numbers) - 1)
        frac = pos - lower
        edges.append(numbers[lower] + (numbers[upper] - numbers[lower]) * frac)
    edges.append(numbers[-1])
    return edges
