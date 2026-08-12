"""A private copy of an online raster layer, to read imagery through.

Reading an online layer is not a passive act. The read turns the provider's
resampling on, forces bilinear, and on a retry tells the provider to drop the
tiles it holds. Done on the layer the user is looking at, that reloads the
basemap under their eyes, which is why nothing speculative was ever allowed to
read one.

A twin removes the reason. It is built from the same provider and the same
source, so it serves the same pixels, and it is never added to the project, so
the layer tree, the canvas and the user's own layer never hear about it. The
tiles it downloads land in the shared network cache, so a later read of the
same ground through the user's layer costs almost nothing.

One twin is kept here, because a raster layer with no parent is collected the
moment nobody holds it, and a cursor moving over the map would otherwise build
one per hover. It is dropped when the source layer changes and when the caller
says the session is over.
"""
from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog, QgsRasterLayer

# The twin being held, and what it was built from: (layer id, source, provider
# name). Any of the three moving makes the twin stale.
_held_twin: QgsRasterLayer | None = None
_held_twin_key: tuple[str, str, str] | None = None


def online_prewarm_enabled() -> bool:
    """Whether imagery may be prepared ahead of a click on an online layer.

    On, a session start and a resting cursor read the ground they are aiming
    at through the twin, so the click that follows finds the tiles already
    downloaded. Off, nothing is read before the click and an online layer
    behaves exactly as it did before the twin existed.

    Fail-open: an absent or damaged configuration prepares imagery as shipped,
    because doing so is free to the user and only ever saves them a wait.
    """
    try:
        from .server_dials import feature_enabled

        return feature_enabled("online_crop_prewarm")
    except Exception:  # noqa: BLE001 -- a kill switch is best-effort  # nosec B110
        return True


def online_layer_twin(source_layer) -> QgsRasterLayer | None:
    """A private copy of ``source_layer`` to read through, or None.

    None whenever the read must not go this way: no layer, a layer QGIS does
    not render (it has a file to read instead), a layer already gone, or a copy
    that did not come out valid. Every caller treats None as "do exactly what
    you did before", so nothing here can cost a user a click.

    The result is cached, so calling this on every mouse move is cheap. The
    caller may drop its reference at any time; this module keeps the twin
    alive until the next source layer or `release_online_layer_twin`.
    """
    global _held_twin, _held_twin_key
    key = _twin_key_for(source_layer)
    if key is None:
        return None
    if _held_twin_key == key and _twin_is_usable(_held_twin):
        return _held_twin
    twin = _build_twin(source_layer, key)
    _held_twin = twin
    _held_twin_key = key if twin is not None else None
    return twin


def release_online_layer_twin() -> None:
    """Drop the twin being held. Called when the session ends and when the
    plugin unloads, so a copy never outlives what it was built for. Idempotent.
    """
    global _held_twin, _held_twin_key
    _held_twin = None
    _held_twin_key = None


def _twin_key_for(source_layer) -> tuple[str, str, str] | None:
    """Identity of the layer a twin would copy, or None when no twin applies."""
    if source_layer is None:
        return None
    try:
        from .feature_encoder import CANVAS_RENDERED_PROVIDERS

        provider = source_layer.dataProvider()
        if provider is None:
            return None
        name = provider.name()
        if name not in CANVAS_RENDERED_PROVIDERS:
            return None
        return (source_layer.id(), provider.dataSourceUri(), name)
    except (RuntimeError, AttributeError, TypeError):
        return None


def _twin_is_usable(twin) -> bool:
    """Is the cached twin still a layer that can be read?"""
    if twin is None:
        return False
    try:
        return bool(twin.isValid())
    except RuntimeError:
        return False


def _build_twin(source_layer, key: tuple[str, str, str]):
    """Build the copy, or return None. Never raises: the caller falls back."""
    _layer_id, uri, provider_name = key
    try:
        twin = QgsRasterLayer(uri, source_layer.name(), provider_name)
        if not twin.isValid():
            return None
    except (RuntimeError, TypeError, ValueError):
        return None
    _copy_renderer(source_layer, twin)
    QgsMessageLog.logMessage(
        "Built a private copy of the rendered layer to read imagery through",
        "AI Segmentation", level=Qgis.MessageLevel.Info)
    return twin


def _copy_renderer(source_layer, twin) -> None:
    """Give the twin the source's own renderer, so a read that falls back to
    drawing the layer draws the picture the user is looking at. Best effort: a
    renderer that will not clone leaves the twin on the provider's default,
    which is what the source started from too."""
    try:
        renderer = source_layer.renderer()
        if renderer is not None:
            twin.setRenderer(renderer.clone())
    except (RuntimeError, AttributeError, TypeError):
        pass  # nosec B110
