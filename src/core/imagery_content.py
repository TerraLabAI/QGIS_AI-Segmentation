





























from __future__ import annotations

try:
    from .venv_manager import ensure_venv_packages_available
except ImportError:
    pass
else:
    ensure_venv_packages_available()

import numpy as np  # noqa: E402

DRAWN_MAP = "drawn_map"
TERRAIN = "terrain"



_SAMPLE_SIDE_PX: int = 128



_FLAT_SHARE_MIN: float = 1.0
_FLAT_GROUP_SHARE_MIN: float = 0.5
_FLAT_GROUPS_MIN: int = 10
_FLAT_GROUP_DISTANCE: float = 16.0
_FLAT_TOP_COLOURS: int = 8
_FLAT_TOP_SHARE_MIN: float = 1.0
_FLAT_PAIR_SHARE_MIN: float = 1.0
_FLAT_PAIR_TOP_SHARE_MIN: float = 1.0
_CLIP_LEVEL: int = 4

_NEUTRAL_EPS: int = 2
_TERRAIN_NEUTRAL_MIN: float = 1.0
_TERRAIN_FLAT_SHARE_MAX: float = 0.0
_RELIEF_GRADIENT_MIN: float = 50.0
_SMOOTH_RATIO_MAX: float = 0.0


def _preflight_dial(name: str, fallback, low: float, high: float):
    try:
        from .server_dials import dial_in_range
    except ImportError:
        return fallback
    return dial_in_range("tuning.preflight." + name, fallback, low, high)


def _sample_rgb(rgb: np.ndarray, side: int) -> np.ndarray:

    h, w = rgb.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / float(side))))
    return rgb[::step, ::step, :3]


def imagery_content_stats(rgb: np.ndarray) -> dict | None:




    if rgb is None or getattr(rgb, "ndim", 0) != 3 or rgb.shape[2] < 3:
        return None
    side = int(_preflight_dial("sample_side_px", _SAMPLE_SIDE_PX, 48, 512))
    a = _sample_rgb(rgb, side).astype(np.int32)
    h, w = a.shape[:2]
    if h < 16 or w < 16:
        return None
    packed = (a[:, :, 0] << 16) | (a[:, :, 1] << 8) | a[:, :, 2]
    core = packed[:-1, :-1]
    flat = (core == packed[:-1, 1:]) & (core == packed[1:, :-1])
    n = float(flat.size)
    flat_share = float(flat.sum()) / n




    groups: list = []
    top_share = 0.0
    if flat_share > 0:
        values, counts = np.unique(core[flat], return_counts=True)
        order = np.argsort(counts)[::-1]
        top_n = int(_preflight_dial("flat_top_colours", _FLAT_TOP_COLOURS, 2, 64))
        top_share = float(counts[order[:top_n]].sum()) / float(counts.sum())
        dist = float(_preflight_dial("flat_group_distance", _FLAT_GROUP_DISTANCE, 8.0, 160.0))
        for idx in order[:24]:
            v = int(values[idx])
            col = np.array([(v >> 16) & 255, (v >> 8) & 255, v & 255], dtype=np.float64)
            share = float(counts[idx]) / n
            for g in groups:
                if float(np.linalg.norm(g["rgb"] - col)) < dist:
                    g["share"] += share
                    break
            else:
                groups.append({"rgb": col, "share": share})

    eps = _NEUTRAL_EPS
    neutral = ((np.abs(a[:, :, 0] - a[:, :, 1]) <= eps)
               & (np.abs(a[:, :, 1] - a[:, :, 2]) <= eps))
    lum = a[:, :, 0] * 0.299 + a[:, :, 1] * 0.587 + a[:, :, 2] * 0.114
    grad = float((np.abs(np.diff(lum, axis=1)).mean()
                  + np.abs(np.diff(lum, axis=0)).mean()) / 2.0)
    second = float((np.abs(np.diff(lum, n=2, axis=1)).mean()
                    + np.abs(np.diff(lum, n=2, axis=0)).mean()) / 2.0)
    return {
        "flat_share": flat_share,
        "flat_groups": sorted((g["share"] for g in groups), reverse=True),



        "flat_groups_unclipped": sorted(
            (g["share"] for g in groups
             if g["rgb"].min() > _CLIP_LEVEL and g["rgb"].max() < 255 - _CLIP_LEVEL),
            reverse=True),
        "flat_top_share": top_share,
        "neutral_share": float(neutral.mean()),
        "gradient": grad,
        "smooth_ratio": second / grad if grad > 1e-6 else 0.0,
    }


def imagery_content_verdict(rgb: np.ndarray) -> str | None:




    try:
        return verdict_from_content_stats(imagery_content_stats(rgb))
    except Exception:  # noqa: BLE001
        return None


def verdict_from_content_stats(stats: dict | None) -> str | None:

    if not stats:
        return None
    flat_min = _preflight_dial("flat_share_min", _FLAT_SHARE_MIN, 0.2, 1.0)
    group_min = _preflight_dial("flat_group_share_min", _FLAT_GROUP_SHARE_MIN, 0.005, 0.5)
    groups_min = int(_preflight_dial("flat_groups_min", _FLAT_GROUPS_MIN, 2, 10))
    top_min = _preflight_dial("flat_top_share_min", _FLAT_TOP_SHARE_MIN, 0.3, 1.0)
    big_groups = [s for s in stats["flat_groups"] if s >= group_min]
    if (stats["flat_share"] >= flat_min and len(big_groups) >= groups_min
            and stats["flat_top_share"] >= top_min):
        return DRAWN_MAP
    pair_flat = _preflight_dial("flat_pair_share_min", _FLAT_PAIR_SHARE_MIN, 0.2, 1.0)
    pair_top = _preflight_dial("flat_pair_top_share_min", _FLAT_PAIR_TOP_SHARE_MIN, 0.3, 1.0)
    pair_groups = [s for s in stats.get("flat_groups_unclipped", ()) if s >= group_min]
    if (stats["flat_share"] >= pair_flat and len(pair_groups) >= 2
            and stats["flat_top_share"] >= pair_top):
        return DRAWN_MAP
    neutral_min = _preflight_dial("terrain_neutral_min", _TERRAIN_NEUTRAL_MIN, 0.5, 1.0)
    flat_max = _preflight_dial("terrain_flat_share_max", _TERRAIN_FLAT_SHARE_MAX, 0.0, 1.0)
    relief_min = _preflight_dial("relief_gradient_min", _RELIEF_GRADIENT_MIN, 0.0, 50.0)
    smooth_max = _preflight_dial("smooth_ratio_max", _SMOOTH_RATIO_MAX, 0.0, 2.0)
    if (stats["neutral_share"] >= neutral_min
            and stats["flat_share"] <= flat_max
            and stats["gradient"] >= relief_min
            and stats["smooth_ratio"] <= smooth_max):
        return TERRAIN
    return None


def qimage_to_rgb_array(img) -> np.ndarray | None:


    try:
        from qgis.PyQt.QtGui import QImage

        if img is None or img.isNull():
            return None


        from .qimage_strips import qimage_array_in_strips

        arr = qimage_array_in_strips(img, QImage.Format.Format_RGB888, 3)
        if arr is None:
            return None
        return np.ascontiguousarray(arr)
    except Exception:  # noqa: BLE001
        return None
