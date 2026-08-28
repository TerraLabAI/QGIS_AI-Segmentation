












from __future__ import annotations

from .raster_provider_kinds import ONLINE_PROVIDERS




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


MAX_BASEMAP_LABEL_CHARS = 64

_LOCAL_RASTER_LABEL = "local raster"


def detect_basemap_label(layer) -> str | None:





    try:
        provider = (layer.providerType() or "").lower()
    except Exception:  # noqa: BLE001
        return None
    if provider == "gdal":
        return _LOCAL_RASTER_LABEL
    if provider not in ONLINE_PROVIDERS:
        return provider or None
    return _online_basemap_label(layer)


def _online_basemap_label(layer) -> str | None:

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
    except Exception:  # noqa: BLE001
        return None


def _host_provider_label(host: str) -> str | None:

    if not host:
        return None
    for needle, label in _BASEMAP_HOSTS:
        if needle in host:
            return label
    return None


def online_basemap_credit(layer) -> str:






    label = detect_basemap_label(layer)
    if not label or label == _LOCAL_RASTER_LABEL:
        return ""
    return label
