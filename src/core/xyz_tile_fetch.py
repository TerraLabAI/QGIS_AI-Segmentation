































from __future__ import annotations

import contextlib
import http.client
import math
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import transport_dials as _td




WEB_MERCATOR_HALF_SPAN = 20037508.342789244
BASE_METERS_PER_PIXEL = 156543.03392804097

_BASE_TILE_PX = 256









_PARALLEL_TILE_REQUESTS = 16
_PARALLEL_TILE_REQUESTS_MAX = 64
_TILE_ATTEMPTS = 3
_TILE_TIMEOUT_S = 6.0





_MAX_TILE_BYTES = 2 * 1024 * 1024






_TILE_DEADLINE_CAP_S = 30.0




_TILE_BACKOFF_S = (0.25, 0.5, 1.0)
_TILE_BACKOFF_SPREAD = 0.25



_THROTTLE_STATUSES = (429, 503)
_MAX_RETRY_AFTER_S = 5.0




_MAX_TILES_PER_CROP = 64




_VERDICT_STATUSES = (400, 404)




_REDIRECT_STATUSES = (301, 302, 303, 307, 308)
_MAX_REDIRECT_HOPS = 2

_USER_AGENT = "Mozilla/5.0 QGIS AI Segmentation"












_failures_by_source: dict[str, tuple[int, float]] = {}
_FAILURES_BEFORE_GIVING_UP = 2
_RETRY_BLOCKED_SOURCE_AFTER_S = 300.0


@dataclass(frozen=True)
class XyzCropRequest:






    template: str
    zoom: int
    tile_range: tuple[int, int, int, int]
    window: tuple[float, float, float, float]
    tile_px: int
    out_px: int
    source_key: str
    headers: dict[str, str] = field(default_factory=dict)
    proxies: dict[str, str] = field(default_factory=dict)

    def tile_count(self) -> int:

        left, top, right, bottom = self.tile_range
        return (right - left + 1) * (bottom - top + 1)


def direct_tile_fetch_available(layer) -> bool:






    try:
        source = layer.source()
    except (AttributeError, RuntimeError):
        return False
    count, last_failure = _failures_by_source.get(source, (0, 0.0))
    if count < _td.xyz_failures_before_giving_up(_FAILURES_BEFORE_GIVING_UP):
        return True
    return (time.monotonic() - last_failure
            >= _td.xyz_retry_blocked_source_after_s(_RETRY_BLOCKED_SOURCE_AFTER_S))




_shared_pool_lock = threading.Lock()
_shared_pool = None
_shared_connections = None
_shared_pool_width = _PARALLEL_TILE_REQUESTS


_shared_pool_width_named = False







_TILE_CACHE_MAX_BYTES = 48 * 1024 * 1024
_TILE_CACHE_MAX_ENTRIES = 4000
_tile_cache_lock = threading.Lock()
_tile_cache: OrderedDict = OrderedDict()
_tile_cache_bytes = 0


def _tile_cache_get(url):

    with _tile_cache_lock:
        hit = _tile_cache.get(url)
        if hit is not None:
            _tile_cache.move_to_end(url)
        return hit


def _tile_cache_put(url, outcome: str, payload) -> None:

    global _tile_cache_bytes
    size = len(payload) if payload else 0
    max_bytes = _td.xyz_cache_max_bytes(_TILE_CACHE_MAX_BYTES)
    max_entries = _td.xyz_cache_max_entries(_TILE_CACHE_MAX_ENTRIES)
    if size > max_bytes // 8:
        return
    with _tile_cache_lock:
        if url in _tile_cache:
            return
        _tile_cache[url] = (outcome, payload)
        _tile_cache_bytes += size
        while _tile_cache and (
                _tile_cache_bytes > max_bytes
                or len(_tile_cache) > max_entries):
            old_payload = _tile_cache.popitem(last=False)[1][1]
            _tile_cache_bytes -= len(old_payload) if old_payload else 0


def clear_tile_cache() -> None:


    global _tile_cache_bytes
    with _tile_cache_lock:
        _tile_cache.clear()
        _tile_cache_bytes = 0


def parallel_tile_requests() -> int:

    return _shared_pool_width


def set_parallel_tile_requests(width) -> None:












    global _shared_pool_width, _shared_pool_width_named
    try:
        wanted = int(width)
    except (TypeError, ValueError):
        return
    if not 1 <= wanted <= _td.xyz_parallel_max(_PARALLEL_TILE_REQUESTS_MAX):
        return
    with _shared_pool_lock:
        _shared_pool_width_named = True
        if wanted == _shared_pool_width:
            return
        _shared_pool_width = wanted
        rebuild = _shared_pool is not None
    if rebuild:
        _close_crop_pool()


def note_direct_tile_fetch_failed(source_key: str) -> None:

    if source_key:
        count = _failures_by_source.get(source_key, (0, 0.0))[0]
        _failures_by_source[source_key] = (count + 1, time.monotonic())


def note_direct_tile_fetch_succeeded(source_key: str) -> None:


    _failures_by_source.pop(source_key, None)


def forget_direct_tile_fetch_failures() -> None:


    _failures_by_source.clear()
    _close_crop_pool()
    clear_tile_cache()


def _crop_pool():








    global _shared_pool, _shared_connections, _shared_pool_width
    with _shared_pool_lock:
        if _shared_pool is None:
            if not _shared_pool_width_named:
                _shared_pool_width = _td.xyz_parallel(_PARALLEL_TILE_REQUESTS)
            _shared_pool = ThreadPoolExecutor(
                max_workers=_shared_pool_width,
                thread_name_prefix="xyztile")
            _shared_connections = _TileConnections()
        return _shared_pool, _shared_connections


def _close_crop_pool() -> None:





    global _shared_pool, _shared_connections
    with _shared_pool_lock:
        pool, connections = _shared_pool, _shared_connections
        _shared_pool = None
        _shared_connections = None
    if connections is not None:
        connections.close_all()
    if pool is not None:
        with contextlib.suppress(Exception):
            pool.shutdown(wait=False)


def xyz_crop_request(layer, extent, out_px: int) -> XyzCropRequest | None:







    parsed = _parse_layer_source(layer)
    if parsed is None:
        return None
    template, zmin, zmax, tile_px, headers = parsed
    try:
        if layer.crs().authid() != "EPSG:3857":
            return None
    except (AttributeError, RuntimeError):
        return None

    bounds = (extent.xMinimum(), extent.yMinimum(),
              extent.xMaximum(), extent.yMaximum())
    span = bounds[2] - bounds[0]
    if (not all(math.isfinite(v) for v in bounds)
            or span <= 0 or bounds[3] <= bounds[1] or out_px <= 0):
        return None
    zoom = tile_zoom_for_resolution(span / out_px, zmin, zmax, tile_px)

    tile_range, window = tile_grid_for_extent(bounds, zoom, tile_px)
    max_tiles = _td.xyz_max_tiles_per_crop(_MAX_TILES_PER_CROP)
    while zoom > zmin and _tiles_in(tile_range) > max_tiles:
        zoom -= 1
        tile_range, window = tile_grid_for_extent(bounds, zoom, tile_px)
    if _tiles_in(tile_range) > max_tiles:
        return None

    return XyzCropRequest(
        template=template, zoom=zoom, tile_range=tile_range, window=window,
        tile_px=tile_px, out_px=out_px, source_key=layer.source(),
        headers=headers, proxies=_qgis_proxy_settings(),
    )


def tile_zoom_for_resolution(map_units_per_pixel: float, zmin: int,
                             zmax: int, tile_px: int = _BASE_TILE_PX) -> int:









    if map_units_per_pixel <= 0:
        return zmax
    side = max(1, int(tile_px or _BASE_TILE_PX))
    base = (2.0 * WEB_MERCATOR_HALF_SPAN) / side
    zoom = int(round(math.log2(base / map_units_per_pixel)))
    return max(zmin, min(zmax, zoom))


def tile_grid_for_extent(bounds, zoom: int, tile_px: int):





    units_per_pixel = (2.0 * WEB_MERCATOR_HALF_SPAN) / (tile_px * (1 << zoom))
    px_min = (bounds[0] + WEB_MERCATOR_HALF_SPAN) / units_per_pixel
    px_max = (bounds[2] + WEB_MERCATOR_HALF_SPAN) / units_per_pixel
    py_min = (WEB_MERCATOR_HALF_SPAN - bounds[3]) / units_per_pixel
    py_max = (WEB_MERCATOR_HALF_SPAN - bounds[1]) / units_per_pixel

    last = (1 << zoom) - 1
    left = max(0, min(last, int(math.floor(px_min / tile_px))))
    right = max(0, min(last, int(math.floor((px_max - 1e-9) / tile_px))))
    top = max(0, min(last, int(math.floor(py_min / tile_px))))
    bottom = max(0, min(last, int(math.floor((py_max - 1e-9) / tile_px))))
    window = (px_min - left * tile_px, py_min - top * tile_px,
              px_max - px_min, py_max - py_min)
    return (left, top, right, bottom), window


def tile_url_for(template: str, zoom: int, x: int, y: int) -> str | None:



    url = template
    if "{-y}" in url:
        url = url.replace("{-y}", str((1 << zoom) - 1 - y))
    url = url.replace("{z}", str(zoom))
    url = url.replace("{x}", str(x))
    url = url.replace("{y}", str(y))
    return None if "{" in url else url


def fetch_xyz_crop(request: XyzCropRequest, cancel_check=None):










    tiles, missing, blank, cancelled, throttled = _download_tiles(
        request, cancel_check)
    if cancelled:
        return None, "crop_error_online_cancelled"
    if missing:




        return None, ("crop_error_online_throttled" if throttled
                      else "crop_error_online_fetch_failed")
    if blank:




        return None, "crop_error_online_blank_tiles"
    mosaic = _paste_tiles(request, tiles)
    if mosaic is None:
        return None, "crop_error_online_fetch_failed"
    return _crop_and_resize(mosaic, request), None




def _tiles_in(tile_range) -> int:
    left, top, right, bottom = tile_range
    return (right - left + 1) * (bottom - top + 1)


def _parse_layer_source(layer):


    try:
        from qgis.core import QgsDataSourceUri

        uri = QgsDataSourceUri()
        uri.setEncodedUri(layer.source())
        if uri.param("type") != "xyz":
            return None
        if uri.param("authcfg"):


            return None
        template = uri.param("url") or ""
        if not template or "{z}" not in template:
            return None
        zmin = int(uri.param("zmin") or 0)
        zmax = int(uri.param("zmax") or 22)



        tile_ratio = int(uri.param("tilePixelRatio") or 0) or 1
        tile_px = 256 * tile_ratio
        if tile_url_for(template, zmin, 0, 0) is None:
            return None
        return template, zmin, zmax, tile_px, _headers_from_uri(uri)
    except Exception:  # noqa: BLE001
        return None


def _headers_from_uri(uri) -> dict[str, str]:


    headers = {"User-Agent": _td.xyz_user_agent(_USER_AGENT)}
    for name, key in (("Referer", "http-header:referer"),
                      ("Referer", "referer")):
        try:
            value = uri.param(key)
        except (AttributeError, RuntimeError, TypeError):
            value = ""
        if value:
            headers[name] = value
    return headers


def _qgis_proxy_settings() -> dict[str, str]:







    try:
        from qgis.core import QgsSettings

        settings = QgsSettings()
        if settings.value("proxy/proxyEnabled", False, type=bool) is not True:
            return {}




        proxies: dict[str, str] = {}
        skipped = _proxy_exclusions(settings)
        if skipped:
            proxies["no"] = skipped
        proxy_type = settings.value("proxy/proxyType", "", type=str) or ""
        if proxy_type not in _URLLIB_PROXY_TYPES:
            return proxies




        if proxy_type == "DefaultProxy":
            return proxies
        host = settings.value("proxy/proxyHost", "", type=str)
        port = settings.value("proxy/proxyPort", "", type=str)
        if not host or not port:
            return proxies
        target = (f"http://{_proxy_credentials_prefix()}"
                  f"{_proxy_authority(host)}:{port}")
        proxies["http"] = target
        proxies["https"] = target
        return proxies
    except Exception:  # noqa: BLE001
        return {}






_URLLIB_PROXY_TYPES = ("", "DefaultProxy", "HttpProxy", "HttpCachingProxy")


def _proxy_credentials_prefix() -> str:







    from .proxy_credentials import qgis_proxy_credentials

    user, password = qgis_proxy_credentials()
    if not user:
        return ""
    return (f"{urllib.parse.quote(user, safe='')}:"
            f"{urllib.parse.quote(password, safe='')}@")


def _proxy_authority(host: str) -> str:





    text = str(host).strip()
    if text.startswith("["):
        return text
    if ":" in text:
        return f"[{text}]"
    return text


def _proxy_exclusions(settings=None) -> str:







    try:
        from qgis.core import QgsSettings

        store = QgsSettings() if settings is None else settings
        raw = store.value("proxy/noProxyUrls", [])
        if isinstance(raw, str):
            raw = [raw]
        hosts: list[str] = []
        for entry in raw or []:
            text = str(entry).strip()
            if not text:
                continue
            if "://" in text:
                host = urllib.parse.urlparse(text).hostname or ""
            else:
                host = text.split("/")[0]
            host = host.strip()
            if host and host not in hosts:
                hosts.append(host)
        return ",".join(hosts)
    except Exception:  # noqa: BLE001
        return ""


def _os_proxies() -> dict[str, str]:






    try:
        published = urllib.request.getproxies() or {}
    except Exception:  # noqa: BLE001
        return {}
    usable: dict[str, str] = {}
    for protocol, address in published.items():
        if str(protocol).lower() not in ("http", "https"):
            continue
        text = str(address or "").strip()
        if not text:
            continue
        scheme = text.split("://", 1)[0].lower() if "://" in text else "http"
        if scheme in ("http", "https"):
            usable[str(protocol).lower()] = text
    return usable


def _download_tiles(request: XyzCropRequest, cancel_check=None):








    left, top, right, bottom = request.tile_range
    coordinates = [(x, y)
                   for y in range(top, bottom + 1)
                   for x in range(left, right + 1)]
    proxies, direct = _crop_route(request)



    os_proxies = {} if direct else _os_proxies()
    opener = _opener_for(proxies, direct, os_proxies)






    proxied = bool(proxies) or bool(os_proxies)
    pool, kept = _crop_pool()
    connections = None if proxied else kept
    waves = max(1, math.ceil(len(coordinates) / _shared_pool_width))
    deadline = time.monotonic() + min(
        _td.xyz_deadline_cap_s(_TILE_DEADLINE_CAP_S),
        _td.xyz_timeout_s(_TILE_TIMEOUT_S) * _td.xyz_attempts(_TILE_ATTEMPTS) * waves)
    header_key = tuple(sorted((str(k).lower(), str(v))
                              for k, v in request.headers.items()))

    def fetch(coordinate):


        if _gave_up(cancel_check):
            return ("cancelled", None)
        x, y = coordinate
        url = tile_url_for(request.template, request.zoom, x, y)
        if url is None:
            return ("missing", None)

        cache_key = (url, header_key)
        cached = _tile_cache_get(cache_key)
        if cached is not None:
            return cached
        if connections is not None and _split_tile_url(url) is not None:
            outcome = _fetch_one_tile_kept(connections, url, request.headers,
                                           deadline, cancel_check)
        else:
            outcome = _fetch_one_tile(opener, url, request.headers, deadline,
                                      cancel_check)
        if outcome[0] in ("ok", "blank"):
            _tile_cache_put(cache_key, outcome[0], outcome[1])
        return outcome



    results = list(pool.map(fetch, coordinates))

    payloads = [payload for _outcome, payload in results]
    missing = sum(1 for outcome, _p in results
                  if outcome in ("missing", "throttled"))
    throttled = sum(1 for outcome, _p in results if outcome == "throttled")
    blank = sum(1 for outcome, _p in results if outcome == "blank")
    cancelled = any(outcome == "cancelled" for outcome, _p in results)
    return payloads, missing, blank, cancelled, throttled


def _gave_up(cancel_check) -> bool:





    if cancel_check is None:
        return False
    try:
        return bool(cancel_check())
    except Exception:  # noqa: BLE001
        return False


def _crop_route(request: XyzCropRequest):






    proxies = dict(request.proxies or {})
    skipped = proxies.pop("no", "")
    if _host_skips_proxy(_template_host(request.template), skipped):
        return {}, True
    return proxies, False


def _template_host(template: str) -> str:

    try:
        return (urllib.parse.urlsplit(template).hostname or "").lower()
    except ValueError:
        return ""


def _host_skips_proxy(host: str, skipped: str) -> bool:





    if not host or not skipped:
        return False
    for entry in skipped.split(","):
        name = entry.strip().lower().lstrip(".")
        if name and (host == name or host.endswith("." + name)):
            return True
    return False


def _opener_for(proxies: dict[str, str], direct: bool,
                os_proxies: dict[str, str] | None = None):







    https = urllib.request.HTTPSHandler(context=_tls_context())
    if proxies:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler(proxies), https)
    if direct:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({}), https)
    return urllib.request.build_opener(urllib.request.ProxyHandler(
        _os_proxies() if os_proxies is None else os_proxies), https)


_tls_lock = threading.Lock()
_shared_tls_context: ssl.SSLContext | None = None


def _tls_context() -> ssl.SSLContext:










    global _shared_tls_context
    with _tls_lock:
        if _shared_tls_context is None:
            _shared_tls_context = ssl.create_default_context()
        return _shared_tls_context


def _fetch_one_tile(opener, url: str, headers: dict[str, str], deadline: float,
                    cancel_check=None):


    throttled = False
    attempts = _td.xyz_attempts(_TILE_ATTEMPTS)
    timeout_s = _td.xyz_timeout_s(_TILE_TIMEOUT_S)
    max_bytes = _td.xyz_max_tile_bytes(_MAX_TILE_BYTES)
    for attempt in range(attempts):
        if _gave_up(cancel_check):
            return ("cancelled", None)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _lost_tile_outcome(cancel_check, throttled)
        retry_after = None
        try:
            appeal = urllib.request.Request(url, headers=headers)
            with opener.open(  # nosec B310
                    appeal, timeout=min(timeout_s, remaining)) as reply:


                payload = reply.read(max_bytes + 1)
            if len(payload) > max_bytes:
                return ("missing", None)
            return ("ok", payload)
        except Exception as err:  # noqa: BLE001
            if isinstance(err, urllib.error.HTTPError):
                if err.code in _VERDICT_STATUSES:


                    return ("blank", None)
                if err.code in _THROTTLE_STATUSES:
                    throttled = True
                    retry_after = _retry_after_seconds(
                        _header_of(err, "Retry-After"))
        if not _pause_before_retry(attempt, retry_after, deadline, cancel_check):
            break
    return _lost_tile_outcome(cancel_check, throttled)


def _lost_tile_outcome(cancel_check, throttled: bool):


    if _gave_up(cancel_check):
        return ("cancelled", None)
    return ("throttled", None) if throttled else ("missing", None)


def _pause_before_retry(attempt: int, retry_after, deadline: float,
                        cancel_check=None) -> bool:







    if attempt + 1 >= _td.xyz_attempts(_TILE_ATTEMPTS):
        return False
    left = deadline - time.monotonic()
    if left <= 0:
        return False
    pause = min(_tile_backoff_pause(attempt, retry_after), left)

    end = time.monotonic() + pause
    while True:
        still_to_wait = end - time.monotonic()
        if still_to_wait <= 0:
            return True
        if _gave_up(cancel_check):
            return False
        time.sleep(min(0.1, still_to_wait))


def _header_of(error, name: str) -> str:

    try:
        return error.headers.get(name, "") or ""
    except Exception:  # noqa: BLE001
        return ""


def _retry_after_seconds(raw: str):






    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _tile_backoff_pause(attempt: int, retry_after=None) -> float:





    import random  # noqa: PLC0415

    ladder = _td.xyz_backoff_s(_TILE_BACKOFF_S)
    base = ladder[min(attempt, len(ladder) - 1)]
    if retry_after is not None:
        base = max(base, min(float(retry_after),
                             _td.xyz_max_retry_after_s(_MAX_RETRY_AFTER_S)))
    return base * (1.0 + random.random() * _td.xyz_backoff_spread(_TILE_BACKOFF_SPREAD))  # nosec B311


class _TileConnections:







    def __init__(self) -> None:
        self._open: dict[tuple[int, str, str], http.client.HTTPConnection] = {}
        self._lock = threading.Lock()

    def acquire(self, scheme: str, host: str, timeout: float):





        key = (threading.get_ident(), scheme, host)
        with self._lock:
            connection = self._open.get(key)
        if connection is not None:
            _hold_to_deadline(connection, timeout)
            return connection, key, True
        if scheme == "https":
            connection = http.client.HTTPSConnection(
                host, timeout=timeout, context=_tls_context())
        else:
            connection = http.client.HTTPConnection(host, timeout=timeout)
        with self._lock:
            self._open[key] = connection
        return connection, key, False

    def drop(self, key) -> None:

        with self._lock:
            connection = self._open.pop(key, None)
        _close_tile_connection(connection)

    def close_all(self) -> None:

        with self._lock:
            connections = list(self._open.values())
            self._open.clear()
        for connection in connections:
            _close_tile_connection(connection)


def _close_tile_connection(connection) -> None:
    if connection is None:
        return
    with contextlib.suppress(Exception):
        connection.close()


def _hold_to_deadline(connection, timeout: float) -> None:

    connection.timeout = timeout
    with contextlib.suppress(Exception):
        if connection.sock is not None:
            connection.sock.settimeout(timeout)


def _split_tile_url(url: str):



    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https"):
        return None
    if not parsed.netloc or "@" in parsed.netloc:
        return None
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return parsed.scheme, parsed.netloc, path


def _fetch_one_tile_kept(connections: _TileConnections, url: str,
                         headers: dict[str, str], deadline: float,
                         cancel_check=None):


    throttled = False
    timeout_s = _td.xyz_timeout_s(_TILE_TIMEOUT_S)
    for attempt in range(_td.xyz_attempts(_TILE_ATTEMPTS)):
        if _gave_up(cancel_check):
            return ("cancelled", None)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _lost_tile_outcome(cancel_check, throttled)
        outcome, retry_after = _tile_over_connection(
            connections, url, headers, min(timeout_s, remaining))
        if outcome is not None:
            return outcome
        if retry_after is not None:
            throttled = True
        if not _pause_before_retry(attempt, retry_after, deadline, cancel_check):
            break
    return _lost_tile_outcome(cancel_check, throttled)


def _tile_over_connection(connections: _TileConnections, url: str,
                          headers: dict[str, str], timeout: float):






    for _hop in range(_MAX_REDIRECT_HOPS + 1):
        reply = _read_over_connection(connections, url, headers, timeout)
        if reply is None:
            return None, None
        status, location, payload, asked_wait = reply
        if 200 <= status < 300:




            if payload is None:
                return ("missing", None), None
            return ("ok", payload), None
        if status in _VERDICT_STATUSES:

            return ("blank", None), None
        if status in _THROTTLE_STATUSES:


            return None, (_retry_after_seconds(asked_wait) or 0.0)
        if status not in _REDIRECT_STATUSES or not location:
            return None, None
        target = urllib.parse.urljoin(url, location)
        if _split_tile_url(target) is None:
            return None, None
        url = target
    return None, None


def _read_over_connection(connections: _TileConnections, url: str,
                          headers: dict[str, str], timeout: float):













    target = _split_tile_url(url)
    if target is None:
        return None
    scheme, host, path = target
    for issue in range(2):
        try:
            connection, key, reused = connections.acquire(scheme, host, timeout)
        except Exception:  # noqa: BLE001
            return None
        try:
            connection.request("GET", path, headers=headers)
            reply = connection.getresponse()
            max_bytes = _td.xyz_max_tile_bytes(_MAX_TILE_BYTES)
            payload = reply.read(max_bytes + 1)
        except Exception:  # noqa: BLE001
            connections.drop(key)
            if reused and issue == 0:
                continue
            return None
        oversized = len(payload) > max_bytes
        if reply.will_close or oversized:
            connections.drop(key)
        return (reply.status, reply.getheader("Location") or "",
                None if oversized else payload,
                reply.getheader("Retry-After") or "")
    return None


def _paste_tiles(request: XyzCropRequest, payloads):









    from qgis.PyQt.QtGui import QImage, QPainter

    left, top, right, bottom = request.tile_range
    tile_px = request.tile_px
    columns = right - left + 1
    rows = bottom - top + 1
    mosaic = QImage(columns * tile_px, rows * tile_px, QImage.Format.Format_RGB888)
    if mosaic.isNull():
        return None
    mosaic.fill(0)
    painter = QPainter(mosaic)
    whole = True
    try:
        for index, payload in enumerate(payloads):
            tile = QImage()
            if not payload or not tile.loadFromData(payload):
                whole = False
                break
            if tile.width() != tile_px or tile.height() != tile_px:
                whole = False
                break
            column = index % columns
            row = index // columns
            painter.drawImage(column * tile_px, row * tile_px, tile)
    finally:
        painter.end()
    return mosaic if whole else None


def _crop_and_resize(mosaic, request: XyzCropRequest) -> np.ndarray:

    from qgis.PyQt.QtCore import Qt

    x, y, width, height = request.window
    cut = mosaic.copy(int(round(x)), int(round(y)),
                      max(1, int(round(width))), max(1, int(round(height))))
    if cut.width() != request.out_px or cut.height() != request.out_px:
        cut = cut.scaled(request.out_px, request.out_px,
                         Qt.AspectRatioMode.IgnoreAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
    return _qimage_to_rgb_array(cut)


def _qimage_to_rgb_array(image) -> np.ndarray:

    from qgis.PyQt.QtGui import QImage

    image = image.convertToFormat(QImage.Format.Format_RGB888)
    width = image.width()
    height = image.height()
    buffer = image.constBits()
    buffer.setsize(image.sizeInBytes())
    flat = np.frombuffer(bytes(buffer), dtype=np.uint8)



    bytes_per_line = image.bytesPerLine()
    return flat.reshape(height, bytes_per_line)[:, :width * 3].reshape(
        height, width, 3).copy()
