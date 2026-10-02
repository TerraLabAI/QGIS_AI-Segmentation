


























from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .server_dials import dial_in_range, feature_switch



_MAX_EXTRA_LEVELS = 3

_DEEPEST_ZOOM = 22


_MIN_AGREEMENT = 0.8
_GRID = 8

_MIN_TEXTURE = 8.0

_PROBE_BUDGET_S = 6.0


_REACH_RADIUS_M = 3000.0

_REACH_MEMORY_PER_SOURCE = 32

_CLICK_PROBE_HALF_SIDE_M = 150.0

_reach_lock = threading.Lock()

_reach_answers: dict[str, list[tuple[float, float, int | None]]] = {}

_reach_in_flight: set[tuple[str, int, int]] = set()


def reach_enabled() -> bool:

    return feature_switch("features.online_zoom_reach", True)


@dataclass(frozen=True)
class ReachRequest:


    template: str
    layer_zmax: int
    candidates: tuple[int, ...]

    points: tuple[tuple[float, float], ...]
    tile_px: int
    source_key: str
    headers: dict[str, str] = field(default_factory=dict)
    proxies: dict[str, str] = field(default_factory=dict)


def reach_request(layer, zone_3857, run_mupp_3857: float) -> ReachRequest | None:







    if not reach_enabled():
        return None
    from .qgis_proxy_reader import qgis_urllib_proxies
    from .xyz_tile_fetch import (
        _parse_layer_source,
        tile_zoom_for_resolution,
    )

    parsed = _parse_layer_source(layer)
    if parsed is None:
        return None
    template, _zmin, zmax, tile_px, headers = parsed
    try:
        if layer.crs().authid() != "EPSG:3857":
            return None
    except (AttributeError, RuntimeError):
        return None
    try:
        xmin, ymin, xmax, ymax = (float(v) for v in zone_3857)
        mupp = float(run_mupp_3857)
    except (TypeError, ValueError):
        return None
    if (not all(math.isfinite(v) for v in (xmin, ymin, xmax, ymax, mupp))
            or mupp <= 0 or xmax <= xmin or ymax <= ymin):
        return None

    deepest = int(dial_in_range("network.xyz.reach_deepest_zoom",
                                _DEEPEST_ZOOM, 18, 24))
    extra = int(dial_in_range("network.xyz.reach_extra_levels",
                              _MAX_EXTRA_LEVELS, 1, 6))
    wanted = tile_zoom_for_resolution(mupp, 0, deepest, tile_px)
    top = min(wanted, zmax + extra, deepest)
    if top <= zmax:
        return None

    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    dx, dy = (xmax - xmin) / 4.0, (ymax - ymin) / 4.0
    points = ((cx, cy), (cx - dx, cy + dy), (cx + dx, cy - dy))
    return ReachRequest(
        template=template, layer_zmax=int(zmax),
        candidates=tuple(range(top, zmax, -1)), points=points,
        tile_px=int(tile_px), source_key=layer.source(), headers=headers,
        proxies=qgis_urllib_proxies())


def probe_reach(request: ReachRequest, cancel_check=None) -> int | None:



    try:
        deadline = time.monotonic() + float(dial_in_range(
            "network.xyz.reach_probe_budget_s", _PROBE_BUDGET_S, 1, 30))
        fetch = _tile_reader(request, deadline, cancel_check)
        base = {pt: _tile_at(fetch, request, request.layer_zmax, pt)
                for pt in request.points}
        if any(tile is None for tile, _box in base.values()):
            return None
        for zoom in request.candidates:
            if time.monotonic() > deadline:
                return None
            if all(_served_at(fetch, request, zoom, pt, base[pt])
                   for pt in request.points):
                return zoom
        return None
    except Exception:  # noqa: BLE001
        return None


def remember_reach(source: str, point, zoom: int | None) -> None:

    with _reach_lock:
        answers = _reach_answers.setdefault(source, [])
        answers.append((float(point[0]), float(point[1]), zoom))
        del answers[:-_REACH_MEMORY_PER_SOURCE]


def reach_known_near(source: str, point):


    radius = dial_in_range("network.xyz.reach_radius_m", _REACH_RADIUS_M,
                           100, 100000)
    best = None
    with _reach_lock:
        for x, y, zoom in _reach_answers.get(source, ()):
            d = math.hypot(x - point[0], y - point[1])
            if d <= radius and (best is None or d < best[0]):
                best = (d, zoom)
    return (False, None) if best is None else (True, best[1])


def reach_zoom_near(layer, point) -> int | None:



    return reach_answer_near(layer, point)[1]


def reach_answer_near(layer, point) -> tuple[bool, int | None]:



    if not reach_enabled():
        return True, None
    try:
        source = layer.source()
    except (AttributeError, RuntimeError):
        return True, None
    known, zoom = reach_known_near(source, point)
    if known:
        return True, zoom
    key = (source, int(point[0] // 1000), int(point[1] // 1000))
    with _reach_lock:
        if key in _reach_in_flight:
            return False, None
    half = _CLICK_PROBE_HALF_SIDE_M
    zone = (point[0] - half, point[1] - half, point[0] + half, point[1] + half)


    request = reach_request(layer, zone, 0.01)
    if request is None:
        return True, None
    with _reach_lock:
        _reach_in_flight.add(key)

    def probe():
        try:
            remember_reach(source, point, probe_reach(request))
        finally:
            with _reach_lock:
                _reach_in_flight.discard(key)

    threading.Thread(target=probe, name="ai-seg-zoom-reach-click",
                     daemon=True).start()
    return False, None


def reach_native_mupp(layer, point, native: float) -> float:



    try:
        return reach_scaled_native(layer, native, reach_zoom_near(layer, point))
    except Exception:  # noqa: BLE001
        return native


def reach_scaled_native(layer, native: float, zoom: int | None,
                        max_extra: int | None = None) -> float:



    from .xyz_tile_fetch import _parse_layer_source

    parsed = _parse_layer_source(layer)
    if parsed is None or native <= 0 or zoom is None:
        return native
    zmax = int(parsed[2])
    if max_extra is not None:
        zoom = min(zoom, zmax + max(0, int(max_extra)))
    if zoom <= zmax:
        return native
    return native / float(1 << (zoom - zmax))


def forget_reach() -> None:

    with _reach_lock:
        _reach_answers.clear()


def reach_clone(layer, zmax: int):


    try:
        from qgis.core import QgsDataSourceUri, QgsRasterLayer

        uri = QgsDataSourceUri()
        uri.setEncodedUri(layer.source())
        uri.removeParam("zmax")
        uri.setParam("zmax", str(int(zmax)))
        encoded = bytes(uri.encodedUri()).decode("utf-8")
        clone = QgsRasterLayer(encoded, layer.name(), layer.providerType())
        if not clone.isValid():
            return None
        renderer = layer.renderer()
        if renderer is not None:
            clone.setRenderer(renderer.clone())
        return clone
    except Exception:  # noqa: BLE001
        return None




def _tile_reader(request: ReachRequest, deadline: float, cancel_check):


    from .xyz_tile_fetch import (
        XyzCropRequest,
        _crop_route,
        _fetch_one_tile,
        _opener_for,
    )

    route = XyzCropRequest(
        template=request.template, zoom=request.layer_zmax,
        tile_range=(0, 0, 0, 0), window=(0.0, 0.0, 1.0, 1.0),
        tile_px=request.tile_px, out_px=1, source_key=request.source_key,
        headers=request.headers, proxies=request.proxies)
    proxies, direct = _crop_route(route)
    opener = _opener_for(proxies, direct)

    def fetch(url):
        outcome, payload = _fetch_one_tile(opener, url, request.headers,
                                           deadline, cancel_check)
        return payload if outcome == "ok" else None

    return fetch


def _tile_at(fetch, request: ReachRequest, zoom: int, point):


    from .xyz_tile_fetch import WEB_MERCATOR_HALF_SPAN, tile_url_for

    n = 1 << zoom
    fx = (point[0] + WEB_MERCATOR_HALF_SPAN) / (2 * WEB_MERCATOR_HALF_SPAN) * n
    fy = (WEB_MERCATOR_HALF_SPAN - point[1]) / (2 * WEB_MERCATOR_HALF_SPAN) * n
    x = min(n - 1, max(0, int(math.floor(fx))))
    y = min(n - 1, max(0, int(math.floor(fy))))
    url = tile_url_for(request.template, zoom, x, y)
    if url is None:
        return None, None
    return _grey(fetch(url)), (x, y)


def _served_at(fetch, request: ReachRequest, zoom: int, point, base) -> bool:







    base_grey, (bx, by) = base
    grey, pos = _tile_at(fetch, request, zoom, point)
    if grey is None or base_grey is None:
        return False
    if float(grey.std()) < _MIN_TEXTURE:
        return False
    steps = zoom - request.layer_zmax
    height, width = base_grey.shape
    span_x, span_y = width >> steps, height >> steps
    if span_x < _GRID or span_y < _GRID:
        return False
    ox = (pos[0] - (bx << steps)) * span_x
    oy = (pos[1] - (by << steps)) * span_y
    corner = base_grey[oy:oy + span_y, ox:ox + span_x]
    if corner.shape != (span_y, span_x):
        return False
    a = _coarse(grey).ravel()
    b = _coarse(corner).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom <= 0:
        return False
    agreement = float((a * b).sum()) / denom
    return agreement >= dial_in_range("network.xyz.reach_min_agreement",
                                      _MIN_AGREEMENT, 0.1, 0.99)


def _grey(payload):

    if not payload:
        return None
    from qgis.PyQt.QtGui import QImage

    image = QImage()
    if not image.loadFromData(payload):
        return None
    if image.width() < _GRID or image.height() < _GRID:
        return None
    from .qimage_strips import qimage_array_in_strips


    grey = qimage_array_in_strips(image, QImage.Format.Format_Grayscale8, 1)
    return None if grey is None else grey[:, :, 0].astype(np.float32)


def _coarse(grey):

    h, w = grey.shape
    trimmed = grey[:h - h % _GRID, :w - w % _GRID]
    return trimmed.reshape(_GRID, trimmed.shape[0] // _GRID,
                           _GRID, trimmed.shape[1] // _GRID).mean(axis=(1, 3))
