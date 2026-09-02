"""Read a tiled web basemap by downloading its tiles, not by rendering it.

The other way to get imagery out of an XYZ layer is to ask its provider for a
block of pixels. That reads through the layer the user is looking at: it turns
the provider's resampling on, and a retry tells the provider to drop the tiles
it holds, which makes the map re-download everything on screen. It also only
returns what the layer has already downloaded, so the read has to be repeated
until the picture stops changing, and that wait is most of the time a click
spends.

A tile is addressed by three numbers, so nothing here needs the layer, the
canvas, or a repaint. The template comes off the layer's own source, the three
numbers come from the extent asked for, and the tiles are downloaded in
parallel and pasted into one image. The layer is never touched.

Two halves, deliberately split by thread:

- `xyz_crop_request` runs where the QGIS objects live (the GUI thread). It
  turns a layer and an extent into plain strings and numbers.
- `fetch_xyz_crop` takes that snapshot and returns pixels. It holds no QGIS
  object, so it is safe to call from a worker.

Anything this module cannot serve returns None, and the caller reads that as
"read the layer the way you always did".

Who reaches this module: the Semi-Auto click crop, through
`core/feature_encoder.py` and `ui/plugin/manual_crops.py`, and nothing else.
An Automatic run does NOT come through here. Its per-tile imagery is a
`QgsMapRendererParallelJob` over the user's layer
(`cloud_detection.start_tile_render_job`), so the pixels come from the QGIS
provider and its own tile cache, not from the pool and the cache below.
"""
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

# Web Mercator, which is the CRS every XYZ basemap is served in. Half the span
# of the projected world in map units, and the map units one pixel covers at
# zoom 0 on a 256 px tile.
WEB_MERCATOR_HALF_SPAN = 20037508.342789244
BASE_METERS_PER_PIXEL = 156543.03392804097
# The tile side those two are quoted for. A layer may serve a larger one.
_BASE_TILE_PX = 256

# Tiles fetched at once, and attempts per tile. The ceiling is what a tile
# host will answer in parallel without throttling the whole crop, and each
# worker holds one connection, so this is also how many sockets a crop opens.
# The width is the client fallback only: the server can widen or narrow it
# per fleet (see set_parallel_tile_requests), because on a cold zone the
# basemap host is what paces a run, and a 1008 px crop is 16 to 25 tiles.
# A third attempt is the last one worth making: a tile the first two lost is
# lost.
_PARALLEL_TILE_REQUESTS = 16
_PARALLEL_TILE_REQUESTS_MAX = 64
_TILE_ATTEMPTS = 3
_TILE_TIMEOUT_S = 6.0

# Ceiling on one tile's body. A tile is a small picture whatever its size, so
# anything past this is not one: a sign-in page, an error document, or a host
# streaming something else at the crop. Reading it to the end would spend the
# whole crop's budget on a body nothing can decode.
_MAX_TILE_BYTES = 2 * 1024 * 1024

# How long a whole crop may take. A pool of workers empties the tiles in
# waves, and each wave can spend a timeout on each of its attempts, so a
# budget written for one wave cuts the later waves off before they are even
# sent. The cap is the point past which falling back to the layer is the
# faster answer anyway.
_TILE_DEADLINE_CAP_S = 30.0

# Waiting between attempts on one tile: doubling, so a host throttling a crop
# is not asked again straight away, and spread by a fraction of the wait so
# the workers that failed together do not all come back together.
_TILE_BACKOFF_S = (0.25, 0.5, 1.0)
_TILE_BACKOFF_SPREAD = 0.25

# Statuses that mean "not now" rather than "not here". The host may name its
# own wait in Retry-After, and that is obeyed up to the crop's own ceiling.
_THROTTLE_STATUSES = (429, 503)
_MAX_RETRY_AFTER_S = 5.0

# Ceiling on the tiles one crop may ask for. A crop that needs more than this
# is asking for a zoom the request has no business fetching, so it drops a
# zoom level rather than fire a hundred requests.
_MAX_TILES_PER_CROP = 64

# A status the host means as a verdict on the URL, not as a hiccup: asking
# again returns it again. 404 says this tile carries no imagery, which is an
# answer; 400 says the numbers are outside the service's range.
_VERDICT_STATUSES = (400, 404)

# Hosts that answer a tile with a redirect. Following it by hand is the price
# of keeping the connection, since urllib is what used to do this. Two hops
# reach every layout seen in the wild: one to the real host, one to a region.
_REDIRECT_STATUSES = (301, 302, 303, 307, 308)
_MAX_REDIRECT_HOPS = 2

_USER_AGENT = "Mozilla/5.0 QGIS AI Segmentation"

# Failures counted per layer source, with the time of the last one. A host that
# will not serve this module but will serve QGIS must stop costing a wasted
# attempt per click; a network that dropped one crop must not disable direct
# fetching for the session. So it takes two failures in a row, and one success
# clears the count.
#
# The time is what makes "for the session" true. Without it the block seals
# itself shut: nothing calls the direct path any more, so nothing can report the
# success that would clear the count, and a couple of dropped crops cost the
# fast path until QGIS restarts. After the cool-down one crop tries again, and
# either clears the count or restarts the wait.
_failures_by_source: dict[str, tuple[int, float]] = {}
_FAILURES_BEFORE_GIVING_UP = 2
_RETRY_BLOCKED_SOURCE_AFTER_S = 300.0


@dataclass(frozen=True)
class XyzCropRequest:
    """Everything needed to fetch one crop, with no QGIS object in it.

    `window` is the crop's pixel rectangle (left, top, width, height) inside
    the tile range, measured from the top-left corner of the first tile.
    """

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
        """How many tiles this request will download."""
        left, top, right, bottom = self.tile_range
        return (right - left + 1) * (bottom - top + 1)


def direct_tile_fetch_available(layer) -> bool:
    """Can this layer's imagery be read by downloading its tiles?

    False once this source has failed twice running, so a host that refuses
    this module costs two attempts, not one per click. True again once the
    cool-down has passed, so the block is a pause and not a life sentence.
    """
    try:
        source = layer.source()
    except (AttributeError, RuntimeError):
        return False
    count, last_failure = _failures_by_source.get(source, (0, 0.0))
    if count < _td.xyz_failures_before_giving_up(_FAILURES_BEFORE_GIVING_UP):
        return True
    return (time.monotonic() - last_failure
            >= _td.xyz_retry_blocked_source_after_s(_RETRY_BLOCKED_SOURCE_AFTER_S))


# The tile transport, shared by every crop of the session. Built on first use
# and dropped at unload; see _crop_pool.
_shared_pool_lock = threading.Lock()
_shared_pool = None
_shared_connections = None
_shared_pool_width = _PARALLEL_TILE_REQUESTS
# Whether a run has named the width; until then the served default applies
# when the pool is first built.
_shared_pool_width_named = False

# Tiles already fetched this session, by URL. Neighbouring crops of one run
# overlap (the grid strides at 0.8 of a tile), so about a third of the tiles a
# run asks for were fetched for the crop next door moments earlier; on a cold
# zone every one of those is a second's wait on the imagery host. Blank
# verdicts are kept too, they are the host's answer for that tile. Bounded by
# bytes and by count, oldest out first, dropped at unload.
_TILE_CACHE_MAX_BYTES = 48 * 1024 * 1024
_TILE_CACHE_MAX_ENTRIES = 4000
_tile_cache_lock = threading.Lock()
_tile_cache: OrderedDict = OrderedDict()
_tile_cache_bytes = 0


def _tile_cache_get(url: str):
    """The cached outcome for this tile URL, or None."""
    with _tile_cache_lock:
        hit = _tile_cache.get(url)
        if hit is not None:
            _tile_cache.move_to_end(url)
        return hit


def _tile_cache_put(url: str, outcome: str, payload) -> None:
    """Keep a served tile (or a blank verdict) for the crops that follow."""
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
            _old_url, (_o, old_payload) = _tile_cache.popitem(last=False)
            _tile_cache_bytes -= len(old_payload) if old_payload else 0


def clear_tile_cache() -> None:
    """Drop every cached tile. Called at unload, and by anything that knows the
    imagery under a source changed."""
    global _tile_cache_bytes
    with _tile_cache_lock:
        _tile_cache.clear()
        _tile_cache_bytes = 0


def parallel_tile_requests() -> int:
    """How many tiles one crop fetches at once, as currently configured."""
    return _shared_pool_width


def set_parallel_tile_requests(width) -> None:
    """Resize the shared tile pool for the crops that follow.

    Called once per run with the served value. A width outside 1..64 or not a
    number leaves the pool as it is. When the width changes, the pool and its
    kept connections are dropped and rebuilt on the next crop; a crop in
    flight on the old pool finishes on it.

    The one caller today is `AutoDetectionWorker.__init__`, and an Automatic
    run never fetches a tile through this module (see the module docstring),
    so what this width really paces is the Semi-Auto clicks that follow in the
    same session. Widening it does not make an Automatic run render faster.
    """
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
    """Count one failed fetch against this source, and restart its cool-down."""
    if source_key:
        count = _failures_by_source.get(source_key, (0, 0.0))[0]
        _failures_by_source[source_key] = (count + 1, time.monotonic())


def note_direct_tile_fetch_succeeded(source_key: str) -> None:
    """Clear this source's failure count, so a passing network gets the direct
    path back after a bad patch."""
    _failures_by_source.pop(source_key, None)


def forget_direct_tile_fetch_failures() -> None:
    """Let every source be tried again, and drop the shared tile transport.
    Called when the plugin unloads."""
    _failures_by_source.clear()
    _close_crop_pool()
    clear_tile_cache()


def _crop_pool():
    """The tile worker pool and its kept connections, built once and reused.

    Both used to be built per crop, and a connection's key carries the thread
    that opened it, so nothing survived the call: every crop on the same host
    paid a fresh handshake per worker, which is exactly the cost the kept
    connections exist to remove. One pool means the same worker threads, so
    their connections are still theirs on the next crop.
    """
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
    """Drop the shared pool and close every connection it kept.

    Never leave a socket or a worker thread behind at unload; the next crop
    builds both again.
    """
    global _shared_pool, _shared_connections
    with _shared_pool_lock:
        pool, connections = _shared_pool, _shared_connections
        _shared_pool = None
        _shared_connections = None
    if connections is not None:
        connections.close_all()
    if pool is not None:
        with contextlib.suppress(Exception):  # a pool already gone needs nothing
            pool.shutdown(wait=False)


def xyz_crop_request(layer, extent, out_px: int) -> XyzCropRequest | None:
    """Snapshot of the tiles covering `extent`, or None to read the layer.

    Call this where the QGIS objects live. None whenever the direct path does
    not apply: a layer that is not a Web Mercator tile service, a template
    with a placeholder this module does not fill, or credentials that only
    QGIS's own network stack holds.
    """
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
    if span <= 0 or out_px <= 0:
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
    """Zoom whose own pixel is closest to the resolution asked for, clamped to
    what the service publishes.

    ``tile_px`` is the layer's own tile side. A high-DPI layer carries twice
    the pixels over the same ground, so its zoom-0 pixel is already twice as
    fine; reading the 256 px constant for it picks a level one too deep, which
    is four times the tiles and four times the bytes for the same ground, all
    thrown away again in the resize.
    """
    if map_units_per_pixel <= 0:
        return zmax
    side = max(1, int(tile_px or _BASE_TILE_PX))
    base = (2.0 * WEB_MERCATOR_HALF_SPAN) / side
    zoom = int(round(math.log2(base / map_units_per_pixel)))
    return max(zmin, min(zmax, zoom))


def tile_grid_for_extent(bounds, zoom: int, tile_px: int):
    """Tiles covering `bounds` at `zoom`, plus where inside them the crop sits.

    Returns ((left, top, right, bottom), (x, y, width, height)), the first in
    tile numbers and the second in pixels of the assembled mosaic.
    """
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
    """One tile's URL, or None when the template holds a placeholder this
    module cannot fill. A URL that still carries braces must never be sent:
    a tile host answers it with an error and the layer looks broken."""
    url = template
    if "{-y}" in url:
        url = url.replace("{-y}", str((1 << zoom) - 1 - y))
    url = url.replace("{z}", str(zoom))
    url = url.replace("{x}", str(x))
    url = url.replace("{y}", str(y))
    return None if "{" in url else url


def fetch_xyz_crop(request: XyzCropRequest, cancel_check=None):
    """Download the request's tiles and return (image, error_code).

    `image` is (out_px, out_px, 3) uint8. `error_code` is None on success, and
    on failure names why so the caller can pick between falling back and
    telling the user. Holds no QGIS object, so a worker may call it.

    ``cancel_check`` is polled between tiles and between attempts on one tile.
    A crop nobody waits on any more stops paying for the rest of its tiles
    instead of running the whole range down to the deadline.
    """
    tiles, missing, blank, cancelled, throttled = _download_tiles(
        request, cancel_check)
    if cancelled:
        return None, "crop_error_online_cancelled"
    if missing:
        # Named apart from a plain loss: "the host is rate-limiting you" and
        # "the tiles never arrived" ask the user for different things, and
        # collapsing both into one code left a throttled user with a slow
        # click and no explanation.
        return None, ("crop_error_online_throttled" if throttled
                      else "crop_error_online_fetch_failed")
    if blank:
        # Even one blank tile leaves an untagged black hole in the mosaic,
        # since a blank tile is skipped rather than painted. The caller falls
        # back to the provider read, which resolves a real coverage gap
        # instead of mosaicing over it.
        return None, "crop_error_online_blank_tiles"
    mosaic = _paste_tiles(request, tiles)
    if mosaic is None:
        return None, "crop_error_online_fetch_failed"
    return _crop_and_resize(mosaic, request), None


# ------------------------------------------------------------------ private

def _tiles_in(tile_range) -> int:
    left, top, right, bottom = tile_range
    return (right - left + 1) * (bottom - top + 1)


def _parse_layer_source(layer):
    """(template, zmin, zmax, tile_px, headers) off the layer's own source, or
    None when this module has no business reading it."""
    try:
        from qgis.core import QgsDataSourceUri

        uri = QgsDataSourceUri()
        uri.setEncodedUri(layer.source())
        if uri.param("type") != "xyz":
            return None
        if uri.param("authcfg"):
            # The credentials live in QGIS's own auth store, which only its
            # network stack can apply.
            return None
        template = uri.param("url") or ""
        if not template or "{z}" not in template:
            return None
        zmin = int(uri.param("zmin") or 0)
        zmax = int(uri.param("zmax") or 22)
        # QGIS stores this as a ratio (0 undefined, 1 standard, 2 high-DPI),
        # not a pixel count, and treats 0 the same as 1. The pixel side of a
        # tile is the base 256 px size times that ratio.
        tile_ratio = int(uri.param("tilePixelRatio") or 0) or 1
        tile_px = 256 * tile_ratio
        if tile_url_for(template, zmin, 0, 0) is None:
            return None
        return template, zmin, zmax, tile_px, _headers_from_uri(uri)
    except Exception:  # noqa: BLE001 -- an unreadable source reads the layer
        return None


def _headers_from_uri(uri) -> dict[str, str]:
    """Request headers the layer carries, plus the agent every tile host wants
    to see. A host that turns away an unnamed client refuses every tile."""
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
    """The proxy QGIS is configured to use, in the form urllib wants. Empty
    when none is set, which is the common case.

    Read through QgsSettings, which is the store the Network page writes and
    the one that honours a profile kept somewhere other than the home
    directory.
    """
    try:
        from qgis.core import QgsSettings

        settings = QgsSettings()
        if settings.value("proxy/proxyEnabled", False, type=bool) is not True:
            return {}
        # The exclusion list travels even when the proxy itself cannot be
        # handed to urllib. A host the user listed must reach the network by
        # itself whichever proxy would otherwise carry it, and the machine
        # publishes one of its own that the fallback opener finds alone.
        proxies: dict[str, str] = {}
        skipped = _proxy_exclusions(settings)
        if skipped:
            proxies["no"] = skipped
        proxy_type = settings.value("proxy/proxyType", "", type=str) or ""
        if proxy_type not in _URLLIB_PROXY_TYPES:
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
    except Exception:  # noqa: BLE001 -- no proxy read means no proxy used
        return {}


# The proxy kinds urllib can carry. QGIS also offers a SOCKS5 proxy and an FTP
# cache, and handing either one to urllib as an http:// address sends every
# tile to a port that speaks another protocol. An empty setting is QGIS's own
# default kind.
_URLLIB_PROXY_TYPES = ("", "DefaultProxy", "HttpProxy", "HttpCachingProxy")


def _proxy_credentials_prefix() -> str:
    """The ``user:password@`` a proxy asks for, or empty when it asks for none.

    The pair can sit in an authentication configuration rather than in the
    clear, so it is read through the one reader that knows both places. Either
    half may hold a character that means something else inside a URL, so both
    are percent-encoded whole. Never logged: a proxy user name names a person.
    """
    from .proxy_credentials import qgis_proxy_credentials

    user, password = qgis_proxy_credentials()
    if not user:
        return ""
    return (f"{urllib.parse.quote(user, safe='')}:"
            f"{urllib.parse.quote(password, safe='')}@")


def _proxy_authority(host: str) -> str:
    """A proxy host as a URL may carry it.

    A numeric address with colons in it is IPv6, and a URL reads the first
    colon as the start of the port unless the address sits in brackets.
    """
    text = str(host).strip()
    if text.startswith("["):
        return text
    if ":" in text:
        return f"[{text}]"
    return text


def _proxy_exclusions(settings=None) -> str:
    """The hosts QGIS is told to reach without the proxy, comma separated.

    Empty when the user listed none. QGIS stores whole URLs and urllib reads
    host names, so only the host part of each entry is kept. ``settings`` is
    the caller's own store when it already has one, so a crop builds one
    QgsSettings rather than two.
    """
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
    except Exception:  # noqa: BLE001 -- an unreadable list is an empty one
        return ""


def _os_proxies() -> dict[str, str]:
    """The machine's own proxy, kept to what urllib can actually carry.

    A machine may publish a proxy that speaks another protocol, and handing
    one to urllib as a web proxy sends every tile to a port that answers
    something else. Only the plain web entries are kept.
    """
    try:
        published = urllib.request.getproxies() or {}
    except Exception:  # noqa: BLE001 -- an unreadable setting is no proxy
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
    """Fetch every tile in the range in parallel.

    Returns (payloads, missing, blank, cancelled, throttled): one entry per
    tile in row-major order, each either the bytes of an image or None; how
    many failed for a reason that may pass; how many the host answered as
    carrying no imagery; whether the caller gave up part way; and how many of
    the failures were the host asking for less traffic.
    """
    left, top, right, bottom = request.tile_range
    coordinates = [(x, y)
                   for y in range(top, bottom + 1)
                   for x in range(left, right + 1)]
    proxies, direct = _crop_route(request)
    # Asked once per crop. On macOS the answer comes from the system
    # configuration store, which is not a cheap read, and both the opener and
    # the connection-reuse decision below want it.
    os_proxies = {} if direct else _os_proxies()
    opener = _opener_for(proxies, direct, os_proxies)
    # Every tile of a crop comes from one host, so one connection per worker
    # thread carries all of them and the handshake is paid once instead of
    # once per tile. A proxy is the exception: tunnelling through one is
    # urllib's job, so a proxied crop keeps the opener. That covers both the
    # proxy QGIS holds and the one the machine publishes to every program,
    # which the opener finds by itself and a raw connection walks past.
    proxied = bool(proxies) or bool(os_proxies)
    pool, kept = _crop_pool()
    connections = None if proxied else kept
    waves = max(1, math.ceil(len(coordinates) / _shared_pool_width))
    deadline = time.monotonic() + min(
        _td.xyz_deadline_cap_s(_TILE_DEADLINE_CAP_S),
        _td.xyz_timeout_s(_TILE_TIMEOUT_S) * _td.xyz_attempts(_TILE_ATTEMPTS) * waves)

    def fetch(coordinate):
        # Polled before each tile as well as inside its attempts, so a crop
        # given up on stops at the next tile instead of the last one.
        if _gave_up(cancel_check):
            return ("cancelled", None)
        x, y = coordinate
        url = tile_url_for(request.template, request.zoom, x, y)
        if url is None:
            return ("missing", None)
        cached = _tile_cache_get(url)
        if cached is not None:
            return cached
        if connections is not None and _split_tile_url(url) is not None:
            outcome = _fetch_one_tile_kept(connections, url, request.headers,
                                           deadline, cancel_check)
        else:
            outcome = _fetch_one_tile(opener, url, request.headers, deadline,
                                      cancel_check)
        if outcome[0] in ("ok", "blank"):
            _tile_cache_put(url, outcome[0], outcome[1])
        return outcome

    # The pool is shared, so the connections opened on its threads are still
    # theirs on the next crop: that is the whole point of keeping them.
    results = list(pool.map(fetch, coordinates))

    payloads = [payload for _outcome, payload in results]
    missing = sum(1 for outcome, _p in results
                  if outcome in ("missing", "throttled"))
    throttled = sum(1 for outcome, _p in results if outcome == "throttled")
    blank = sum(1 for outcome, _p in results if outcome == "blank")
    cancelled = any(outcome == "cancelled" for outcome, _p in results)
    return payloads, missing, blank, cancelled, throttled


def _gave_up(cancel_check) -> bool:
    """Whether the caller has stopped waiting on this crop.

    A predicate that raises is read as still waiting: giving up on a crop
    because its own cancel broke would lose imagery the user asked for.
    """
    if cancel_check is None:
        return False
    try:
        return bool(cancel_check())
    except Exception:  # noqa: BLE001 -- a broken predicate never stops a crop
        return False


def _crop_route(request: XyzCropRequest):
    """How this crop reaches its host: ``(proxies, direct)``.

    ``proxies`` is what urllib should tunnel through, empty when nothing
    should. ``direct`` says the user listed this host among the addresses to
    reach without a proxy, which rules out the machine's own as well.
    """
    proxies = dict(request.proxies or {})
    skipped = proxies.pop("no", "")
    if _host_skips_proxy(_template_host(request.template), skipped):
        return {}, True
    return proxies, False


def _template_host(template: str) -> str:
    """The host a tile template points at, or empty when it names none."""
    try:
        return (urllib.parse.urlsplit(template).hostname or "").lower()
    except ValueError:
        return ""


def _host_skips_proxy(host: str, skipped: str) -> bool:
    """Whether this host is one QGIS is told to reach without the proxy.

    A listed name covers what sits under it, which is how every other program
    reads the same list.
    """
    if not host or not skipped:
        return False
    for entry in skipped.split(","):
        name = entry.strip().lower().lstrip(".")
        if name and (host == name or host.endswith("." + name)):
            return True
    return False


def _opener_for(proxies: dict[str, str], direct: bool,
                os_proxies: dict[str, str] | None = None):
    """The opener a crop's tiles travel through.

    A crop with nothing configured still travels through the proxy the machine
    publishes to every program, which is what it wants and exactly what a host
    on the exclusion list must not get. ``os_proxies`` is that published set
    when the caller has already read it, so one crop asks the machine once.
    """
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
    """The one TLS context every tile connection shares.

    Left to itself, http.client builds a fresh default context for every
    connection it opens, and building one means loading the machine's whole
    trust store. On Windows that store is read certificate by certificate and
    costs a few hundred milliseconds each time, so a crop that opens one
    socket per worker thread spent seconds on trust stores before the first
    tile moved. Built once, on the first crop, and shared from then on: a
    context is safe to share across connections and threads.
    """
    global _shared_tls_context
    with _tls_lock:
        if _shared_tls_context is None:
            _shared_tls_context = ssl.create_default_context()
        return _shared_tls_context


def _fetch_one_tile(opener, url: str, headers: dict[str, str], deadline: float,
                    cancel_check=None):
    """One tile, retried. Returns ("ok", bytes), ("blank", None) when the host
    says it has no imagery there, ("cancelled", None), or ("missing", None)."""
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
            with opener.open(  # nosec B310 -- the template comes from the layer
                    appeal, timeout=min(timeout_s, remaining)) as reply:
                # One byte past the ceiling is enough to know the body is over
                # it, and the rest of it is never pulled down the wire.
                payload = reply.read(max_bytes + 1)
            if len(payload) > max_bytes:
                return ("missing", None)
            return ("ok", payload)
        except Exception as err:  # noqa: BLE001 -- a lost tile is retried below
            if isinstance(err, urllib.error.HTTPError):
                if err.code in _VERDICT_STATUSES:
                    # The host has answered. Asking again returns the same
                    # answer.
                    return ("blank", None)
                if err.code in _THROTTLE_STATUSES:
                    throttled = True
                    retry_after = _retry_after_seconds(
                        _header_of(err, "Retry-After"))
        if not _pause_before_retry(attempt, retry_after, deadline, cancel_check):
            break
    return _lost_tile_outcome(cancel_check, throttled)


def _lost_tile_outcome(cancel_check, throttled: bool):
    """What a tile that never arrived is called: a cancel, a host asking for
    less traffic, or a plain loss."""
    if _gave_up(cancel_check):
        return ("cancelled", None)
    return ("throttled", None) if throttled else ("missing", None)


def _pause_before_retry(attempt: int, retry_after, deadline: float,
                        cancel_check=None) -> bool:
    """Wait out the back-off, and say whether another attempt is worth making.

    Nothing is waited after the last attempt: that wait buys no tile and the
    crop pays for it. The wait is also cut to what is left of the crop's own
    deadline, since a host that asks for a long one must not take the whole
    budget with it.
    """
    if attempt + 1 >= _td.xyz_attempts(_TILE_ATTEMPTS):
        return False
    left = deadline - time.monotonic()
    if left <= 0:
        return False
    pause = min(_tile_backoff_pause(attempt, retry_after), left)
    # Slept in slices so a cancel lands during the wait, not after it.
    end = time.monotonic() + pause
    while True:
        still_to_wait = end - time.monotonic()
        if still_to_wait <= 0:
            return True
        if _gave_up(cancel_check):
            return False
        time.sleep(min(0.1, still_to_wait))


def _header_of(error, name: str) -> str:
    """One header off a failed request, or empty when it carries none."""
    try:
        return error.headers.get(name, "") or ""
    except Exception:  # noqa: BLE001 -- a header nobody sent changes nothing
        return ""


def _retry_after_seconds(raw: str):
    """The wait a host asked for, in seconds, or None when it named none.

    Only the plain number is read. The date form is the other half of the
    header, and a clock that disagrees with the host's would turn it into a
    wait no crop can afford.
    """
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _tile_backoff_pause(attempt: int, retry_after=None) -> float:
    """Seconds to wait before asking for a tile again.

    A host that names its own wait is obeyed, up to the ceiling one crop can
    spend waiting for one tile.
    """
    import random  # noqa: PLC0415 -- only needed once a tile has failed

    ladder = _td.xyz_backoff_s(_TILE_BACKOFF_S)
    base = ladder[min(attempt, len(ladder) - 1)]
    if retry_after is not None:
        base = max(base, min(float(retry_after),
                             _td.xyz_max_retry_after_s(_MAX_RETRY_AFTER_S)))
    return base * (1.0 + random.random() * _td.xyz_backoff_spread(_TILE_BACKOFF_SPREAD))  # nosec B311


class _TileConnections:
    """One live connection per worker thread, kept open for the whole crop.

    A connection is never handed to a second thread: the key carries the
    thread that opened it. Anything the host or the network breaks is dropped
    and reopened, and `close_all` runs once the pool has joined.
    """

    def __init__(self) -> None:
        self._open: dict[tuple[int, str, str], http.client.HTTPConnection] = {}
        self._lock = threading.Lock()

    def acquire(self, scheme: str, host: str, timeout: float):
        """(connection, key, reused) for this thread, opened if it holds none.

        Opening is lazy in the stdlib, so the socket is only built when the
        first request goes out and a refused host surfaces there.
        """
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
        """Close one connection and forget it, so the next tile opens fresh."""
        with self._lock:
            connection = self._open.pop(key, None)
        _close_tile_connection(connection)

    def close_all(self) -> None:
        """Close every connection the crop opened. Never leave a socket behind."""
        with self._lock:
            connections = list(self._open.values())
            self._open.clear()
        for connection in connections:
            _close_tile_connection(connection)


def _close_tile_connection(connection) -> None:
    if connection is None:
        return
    with contextlib.suppress(Exception):  # a socket already gone needs nothing
        connection.close()


def _hold_to_deadline(connection, timeout: float) -> None:
    """Hold a kept connection to what is left of the crop's own deadline."""
    connection.timeout = timeout
    with contextlib.suppress(Exception):  # a socket that refuses this is dropped
        if connection.sock is not None:
            connection.sock.settimeout(timeout)


def _split_tile_url(url: str):
    """(scheme, host, path) for a tile URL, or None when this transport cannot
    carry it: another scheme, or credentials in the URL, which only urllib
    knows how to turn into a header."""
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
    """One tile over this thread's kept connection. Same attempts, same
    verdicts and same back-off as the opener path."""
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
    """One attempt, redirects followed.

    Returns (outcome, retry_after): the tile's outcome, or None when the
    attempt is worth making again, and the wait the host asked for when it
    named one.
    """
    for _hop in range(_MAX_REDIRECT_HOPS + 1):
        reply = _read_over_connection(connections, url, headers, timeout)
        if reply is None:
            return None, None
        status, location, payload, asked_wait = reply
        if 200 <= status < 300:
            # The same band the opener path treats as an answer, so a host
            # that never sends a plain 200 behaves as it always did. A body
            # past the ceiling is not a tile, and asking again brings the same
            # one back.
            if payload is None:
                return ("missing", None), None
            return ("ok", payload), None
        if status in _VERDICT_STATUSES:
            # The host has answered. Asking again returns the same answer.
            return ("blank", None), None
        if status in _THROTTLE_STATUSES:
            # Never None on this branch, so the caller can tell a throttle from
            # a plain retry even when the host named no wait of its own.
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
    """Send one GET on this thread's connection and read the reply.

    Returns (status, location, payload, retry_after), or None when the request
    failed. ``payload`` is None when the body ran past the ceiling a tile is
    allowed, which is not a tile whatever it is.

    A connection kept from an earlier tile can have been dropped by the host
    while it sat idle. That is the transport, not the host, so the socket is
    replaced and the request re-issued without spending one of the tile's
    attempts. The body is read to the end, because a connection carries the
    next tile only once the reply before it is finished, and a body too long
    to read to the end costs the connection instead.
    """
    target = _split_tile_url(url)
    if target is None:
        return None
    scheme, host, path = target
    for issue in range(2):
        try:
            connection, key, reused = connections.acquire(scheme, host, timeout)
        except Exception:  # noqa: BLE001 -- a host that will not open is retried
            return None
        try:
            connection.request("GET", path, headers=headers)
            reply = connection.getresponse()
            max_bytes = _td.xyz_max_tile_bytes(_MAX_TILE_BYTES)
            payload = reply.read(max_bytes + 1)
        except Exception:  # noqa: BLE001 -- see the docstring
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
    """Decode the tiles and paint them into one image, or None.

    Every tile has to decode, and at the size the layer says the service
    serves. A payload that is not a picture leaves a black square in the
    middle of the crop, and one that decodes at another size is painted over
    its neighbours; both reach the model as imagery and neither looks like a
    failure. So a crop that cannot be assembled whole is not assembled at all,
    and the caller reads the layer instead.
    """
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
    """Cut the crop out of the mosaic and scale it to the size asked for."""
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
    """(H, W, 3) uint8 from a QImage, copied out of Qt's own buffer."""
    from qgis.PyQt.QtGui import QImage

    image = image.convertToFormat(QImage.Format.Format_RGB888)
    width = image.width()
    height = image.height()
    buffer = image.constBits()
    buffer.setsize(image.sizeInBytes())
    flat = np.frombuffer(bytes(buffer), dtype=np.uint8)
    # Format_RGB888 pads every row to a 4 byte boundary, so bytesPerLine is
    # not always divisible by 3 and a stride in pixels does not exist. Reshape
    # on the byte count, then cut each row to the pixels it carries.
    bytes_per_line = image.bytesPerLine()
    return flat.reshape(height, bytes_per_line)[:, :width * 3].reshape(
        height, width, 3).copy()
