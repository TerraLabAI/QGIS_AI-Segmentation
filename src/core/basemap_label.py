"""Sanitized provider label for the imagery a run is detected on.

The label rides the optional /predict provenance fields, so a "it found
nothing" report on Esri imagery can be told apart from the same report on a
blurry local raster, without asking the user what they had loaded.

What it may never contain: a full data source, a file path, a URL query or an
embedded auth token. A tile URL carries all four. So a known host collapses to
a short provider name ("Esri", "Google", "IGN"), an unknown one keeps the host
and nothing else, and anything that does not parse yields None.

qgis is imported lazily inside the functions, so this module stays headless.
"""
from __future__ import annotations

from .raster_provider_kinds import ONLINE_PROVIDERS

# Known tile hosts -> friendly label. Substring match on the lowercased host,
# first match wins. Mirrors AI Edit's table so both plugins report the same
# name for the same basemap.
_BASEMAP_HOSTS = (
    ("google", "Google"),
    ("gstatic", "Google"),
    ("virtualearth", "Bing"),
    ("bing", "Bing"),
    ("geopf.fr", "IGN"),
    ("ign.fr", "IGN"),
    ("geoportail", "IGN"),
    ("arcgisonline", "Esri"),
    ("esri", "Esri"),
    ("mapbox", "Mapbox"),
    ("openstreetmap", "OSM"),
    ("tile.osm", "OSM"),
    ("cartocdn", "Carto"),
    ("swisstopo", "Swisstopo"),
)

# Server-side cap on the field. A garbled host can never grow the payload.
MAX_BASEMAP_LABEL_CHARS = 64

_LOCAL_RASTER_LABEL = "local raster"


def detect_basemap_label(layer) -> str | None:
    """Short provider label for one raster layer, or None when unknown.

    Never raises: every caller uses this as best-effort provenance and must be
    able to drop the field rather than fail a tile request.
    """
    try:
        provider = (layer.providerType() or "").lower()
    except Exception:  # noqa: BLE001 - best-effort provenance field
        return None
    if provider == "gdal":
        return _LOCAL_RASTER_LABEL
    if provider not in ONLINE_PROVIDERS:
        return provider or None
    return _online_basemap_label(layer)


def _online_basemap_label(layer) -> str | None:
    """Friendly name for a known tile host, else '<kind>:<host>', else None."""
    try:
        from urllib.parse import parse_qs, urlsplit

        params = parse_qs(layer.source() or "")
        url = (params.get("url") or [""])[0]
        kind = "XYZ" if (params.get("type") or [""])[0] == "xyz" else "WMS"
        host = (urlsplit(url).hostname or "").lower()
        label = _host_provider_label(host)
        if label is not None:
            return label
        return (f"{kind}:{host}" if host else kind)[:MAX_BASEMAP_LABEL_CHARS]
    except Exception:  # noqa: BLE001 - best-effort provenance field
        return None


def _host_provider_label(host: str) -> str | None:
    """The shipped label for a known tile host, None for an unknown one."""
    if not host:
        return None
    for needle, label in _BASEMAP_HOSTS:
        if needle in host:
            return label
    return None


def online_basemap_credit(layer) -> str:
    """Provider name to credit in a saved file's metadata, '' when there is none.

    Only an online basemap earns a credit line: a local raster is already
    named in the abstract by its own layer name, and "local raster" as a
    rights statement says nothing to whoever the file is delivered to.
    """
    label = detect_basemap_label(layer)
    if not label or label == _LOCAL_RASTER_LABEL:
        return ""
    return label
