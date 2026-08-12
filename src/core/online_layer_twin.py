


















from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog, QgsRasterLayer



_held: dict = {"twin": None, "key": None}


def online_prewarm_enabled() -> bool:










    try:
        from .server_dials import feature_enabled

        return feature_enabled("online_crop_prewarm")
    except Exception:  # noqa: BLE001  # nosec B110
        return True


def online_layer_twin(source_layer) -> QgsRasterLayer | None:











    key = _twin_key_for(source_layer)
    if key is None:
        return None
    held_twin = _held["twin"]
    if _held["key"] == key and _twin_is_usable(held_twin):
        return held_twin
    twin = _build_twin(source_layer, key)
    _held["twin"] = twin
    _held["key"] = key if twin is not None else None
    return twin


def release_online_layer_twin() -> None:



    _held["twin"] = None
    _held["key"] = None


def _twin_key_for(source_layer) -> tuple[str, str, str] | None:

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

    if twin is None:
        return False
    try:
        return bool(twin.isValid())
    except RuntimeError:
        return False


def _build_twin(source_layer, key: tuple[str, str, str]):

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




    try:
        renderer = source_layer.renderer()
        if renderer is not None:
            twin.setRenderer(renderer.clone())
    except (RuntimeError, AttributeError, TypeError):
        pass  # nosec B110
