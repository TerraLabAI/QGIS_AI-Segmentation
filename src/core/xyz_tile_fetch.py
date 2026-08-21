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
"""
from __future__ import annotations

import contextlib
import http.client
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

# Web Mercator, which is the CRS every XYZ basemap is served in. Half the span
# of the projected world in map units, and the map units one pixel covers at
# zoom 0 on a 256 px tile.
WEB_MERCATOR_HALF_SPAN = 20037508.342789244
BASE_METERS_PER_PIXEL = 156543.03392804097

# Tiles fetched at once, and attempts per tile. The ceiling is what a tile
# host will answer in parallel without throttling the whole crop, and each
# worker holds one connection, so this is also how many sockets a crop opens.
# A third attempt is the last one worth making: a tile the first two lost is
# lost.
_PARALLEL_TILE_REQUESTS = 8
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

# Failures counted per layer source. A host that will not serve this module
# but will serve QGIS must stop costing a wasted attempt per click; a network
# that dropped one crop must not disable direct fetching for the session. So
# it takes two failures in a row, and one success clears the count.
_failures_by_source: dict[str, int] = {}
_FAILURES_BEFORE_GIVING_UP = 2


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
    this module costs two attempts, not one per click.
    """
    try:
        source = layer.source()
    except (AttributeError, RuntimeError):
        return False
    return _failures_by_source.get(source, 0) < _FAILURES_BEFORE_GIVING_UP


def note_direct_tile_fetch_failed(source_key: str) -> None:
    """Count one failed fetch against this source."""
    if source_key:
        _failures_by_source[source_key] = _failures_by_source.get(source_key, 0) + 1


def note_direct_tile_fetch_succeeded(source_key: str) -> None:
    """Clear this source's failure count, so a passing network gets the direct
    path back after a bad patch."""
    _failures_by_source.pop(source_key, None)


def forget_direct_tile_fetch_failures() -> None:
    """Let every source be tried again. Called when the plugin unloads."""
    _failures_by_source.clear()


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
    zoom = tile_zoom_for_resolution(span / out_px, zmin, zmax)

    tile_range, window = tile_grid_for_extent(bounds, zoom, tile_px)
    while zoom > zmin and _tiles_in(tile_range) > _MAX_TILES_PER_CROP:
        zoom -= 1
        tile_range, window = tile_grid_for_extent(bounds, zoom, tile_px)
    if _tiles_in(tile_range) > _MAX_TILES_PER_CROP:
        return None

    return XyzCropRequest(
        template=template, zoom=zoom, tile_range=tile_range, window=window,
        tile_px=tile_px, out_px=out_px, source_key=layer.source(),
        headers=headers, proxies=_qgis_proxy_settings(),
    )


def tile_zoom_for_resolution(map_units_per_pixel: float, zmin: int,
                             zmax: int) -> int:
    """Zoom whose own pixel is closest to the resolution asked for, clamped to
    what the service publishes."""
    if map_units_per_pixel <= 0:
        return zmax
    zoom = int(round(math.log2(BASE_METERS_PER_PIXEL / map_units_per_pixel)))
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
    tiles, missing, blank, cancelled = _download_tiles(request, cancel_check)
    if cancelled:
        return None, "crop_error_online_cancelled"
    if missing:
        return None, "crop_error_online_fetch_failed"
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
    headers = {"User-Agent": _USER_AGENT}
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
        skipped = _proxy_exclusions()
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


def _proxy_exclusions() -> str:
    """The hosts QGIS is told to reach without the proxy, comma separated.

    Empty when the user listed none. QGIS stores whole URLs and urllib reads
    host names, so only the host part of each entry is kept.
    """
    try:
        from qgis.core import QgsSettings

        raw = QgsSettings().value("proxy/noProxyUrls", [])
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


def _os_proxy_configured() -> bool:
    """Whether the machine publishes a proxy this crop can travel through.

    urllib's own opener reads that setting and routes through it. A connection
    opened by hand does not, and on a network that only lets the proxy out
    that is every tile lost.
    """
    return bool(_os_proxies())


def _download_tiles(request: XyzCropRequest, cancel_check=None):
    """Fetch every tile in the range in parallel.

    Returns (payloads, missing, blank, cancelled): one entry per tile in
    row-major order, each either the bytes of an image or None; how many
    failed for a reason that may pass; how many the host answered as carrying
    no imagery; and whether the caller gave up part way.
    """
    left, top, right, bottom = request.tile_range
    coordinates = [(x, y)
                   for y in range(top, bottom + 1)
                   for x in range(left, right + 1)]
    proxies, direct = _crop_route(request)
    opener = _opener_for(proxies, direct)
    # Every tile of a crop comes from one host, so one connection per worker
    # thread carries all of them and the handshake is paid once instead of
    # once per tile. A proxy is the exception: tunnelling through one is
    # urllib's job, so a proxied crop keeps the opener. That covers both the
    # proxy QGIS holds and the one the machine publishes to every program,
    # which the opener finds by itself and a raw connection walks past.
    proxied = bool(proxies) or (not direct and _os_proxy_configured())
    connections = None if proxied else _TileConnections()
    waves = max(1, math.ceil(len(coordinates) / _PARALLEL_TILE_REQUESTS))
    deadline = time.monotonic() + min(
        _TILE_DEADLINE_CAP_S, _TILE_TIMEOUT_S * _TILE_ATTEMPTS * waves)

    def fetch(coordinate):
        # Polled before each tile as well as inside its attempts, so a crop
        # given up on stops at the next tile instead of the last one.
        if _gave_up(cancel_check):
            return ("cancelled", None)
        x, y = coordinate
        url = tile_url_for(request.template, request.zoom, x, y)
        if url is None:
            return ("missing", None)
        if connections is not None and _split_tile_url(url) is not None:
            return _fetch_one_tile_kept(connections, url, request.headers,
                                        deadline, cancel_check)
        return _fetch_one_tile(opener, url, request.headers, deadline,
                               cancel_check)

    workers = min(_PARALLEL_TILE_REQUESTS, max(1, len(coordinates)))
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(fetch, coordinates))
    finally:
        # The pool has joined here, so no thread can still hold a socket.
        if connections is not None:
            connections.close_all()

    payloads = [payload for _outcome, payload in results]
    missing = sum(1 for outcome, _p in results if outcome == "missing")
    blank = sum(1 for outcome, _p in results if outcome == "blank")
    cancelled = any(outcome == "cancelled" for outcome, _p in results)
    return payloads, missing, blank, cancelled


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


def _opener_for(proxies: dict[str, str], direct: bool):
    """The opener a crop's tiles travel through.

    A crop with nothing configured still travels through the proxy the machine
    publishes to every program, which is what it wants and exactly what a host
    on the exclusion list must not get.
    """
    if proxies:
        return urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
    if direct:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(
        urllib.request.ProxyHandler(_os_proxies()))


def _fetch_one_tile(opener, url: str, headers: dict[str, str], deadline: float,
                    cancel_check=None):
    """One tile, retried. Returns ("ok", bytes), ("blank", None) when the host
    says it has no imagery there, ("cancelled", None), or ("missing", None)."""
    for attempt in range(_TILE_ATTEMPTS):
        if _gave_up(cancel_check):
            return ("cancelled", None)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return ("missing", None)
        retry_after = None
        try:
            appeal = urllib.request.Request(url, headers=headers)
            with opener.open(  # nosec B310 -- the template comes from the layer
                    appeal, timeout=min(_TILE_TIMEOUT_S, remaining)) as reply:
                # One byte past the ceiling is enough to know the body is over
                # it, and the rest of it is never pulled down the wire.
                payload = reply.read(_MAX_TILE_BYTES + 1)
            if len(payload) > _MAX_TILE_BYTES:
                return ("missing", None)
            return ("ok", payload)
        except Exception as err:  # noqa: BLE001 -- a lost tile is retried below
            if isinstance(err, urllib.error.HTTPError):
                if err.code in _VERDICT_STATUSES:
                    # The host has answered. Asking again returns the same
                    # answer.
                    return ("blank", None)
                if err.code in _THROTTLE_STATUSES:
                    retry_after = _retry_after_seconds(
                        _header_of(err, "Retry-After"))
        if not _pause_before_retry(attempt, retry_after, deadline, cancel_check):
            break
    return ("cancelled", None) if _gave_up(cancel_check) else ("missing", None)


def _pause_before_retry(attempt: int, retry_after, deadline: float,
                        cancel_check=None) -> bool:
    """Wait out the back-off, and say whether another attempt is worth making.

    Nothing is waited after the last attempt: that wait buys no tile and the
    crop pays for it. The wait is also cut to what is left of the crop's own
    deadline, since a host that asks for a long one must not take the whole
    budget with it.
    """
    if attempt + 1 >= _TILE_ATTEMPTS:
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

    base = _TILE_BACKOFF_S[min(attempt, len(_TILE_BACKOFF_S) - 1)]
    if retry_after is not None:
        base = max(base, min(float(retry_after), _MAX_RETRY_AFTER_S))
    return base * (1.0 + random.random() * _TILE_BACKOFF_SPREAD)  # nosec B311


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
        connection_class = (http.client.HTTPSConnection if scheme == "https"
                            else http.client.HTTPConnection)
        connection = connection_class(host, timeout=timeout)
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
    for attempt in range(_TILE_ATTEMPTS):
        if _gave_up(cancel_check):
            return ("cancelled", None)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return ("missing", None)
        outcome, retry_after = _tile_over_connection(
            connections, url, headers, min(_TILE_TIMEOUT_S, remaining))
        if outcome is not None:
            return outcome
        if not _pause_before_retry(attempt, retry_after, deadline, cancel_check):
            break
    return ("cancelled", None) if _gave_up(cancel_check) else ("missing", None)


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
            return None, _retry_after_seconds(asked_wait)
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
            payload = reply.read(_MAX_TILE_BYTES + 1)
        except Exception:  # noqa: BLE001 -- see the docstring
            connections.drop(key)
            if reused and issue == 0:
                continue
            return None
        oversized = len(payload) > _MAX_TILE_BYTES
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
    stride = image.bytesPerLine() // 3
    return flat.reshape(height, stride, 3)[:, :width, :].copy()
