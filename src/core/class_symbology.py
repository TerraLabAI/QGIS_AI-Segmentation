















from __future__ import annotations

import re
from collections.abc import Iterable



MAX_CATEGORIES = 24





DEFAULT_RAMP_ANCHORS: tuple[str, ...] = ("#1e88e5", "#8bac27", "#f5a623", "#d32f2f")



OTHER_COLOR = "#757575"

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

_MAX_ANCHORS = 16


def legend_category_cap() -> int:





    try:
        from .server_dials import dial_in_range

        return int(dial_in_range("taxonomy.legend.max_categories", MAX_CATEGORIES, 2, 128))
    except Exception:  # noqa: BLE001  # nosec B110
        return MAX_CATEGORIES


def legend_ramp_anchors() -> tuple[str, ...]:






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
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return DEFAULT_RAMP_ANCHORS


def legend_other_color() -> str:

    try:
        from .server_dials import read_value

        value = read_value("taxonomy.legend.other_color")
        if isinstance(value, str) and _HEX_RE.match(value.strip()):
            return value.strip().lower()
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return OTHER_COLOR


def normalize_hex(value: str) -> str:






    text = (value or "").strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        raise ValueError(f"not a hex color: {value!r}")
    int(text, 16)
    return f"#{text.lower()}"


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = normalize_hex(value)[1:]
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    r, g, b = (max(0, min(255, round(channel))) for channel in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def interpolate_colors(anchors: Iterable[str], n: int) -> list[str]:






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



    labels = {(v or "").strip() for v in values}
    labels.discard("")
    return sorted(labels)


def needs_categorized_renderer(values: Iterable[str]) -> bool:





    return len(_unique_labels(values)) > 1


def class_color_mapping(
    values: Iterable[str],
    *,
    max_categories: int | None = None,
    ramp_anchors: Iterable[str] | None = None,
) -> dict[str, str]:









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
