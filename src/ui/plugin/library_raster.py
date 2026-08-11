













from __future__ import annotations

import os

from qgis.core import QgsCoordinateTransform, QgsGeometry, QgsProject, QgsRasterLayer
from qgis.PyQt.QtCore import QTimer

from ...core.i18n import tr
from ...core.logging_utils import log



_SELECT_RETRY_MS = 150
_SELECT_MAX_TRIES = 10


def _demo_select_retry_ms() -> int:
    from ...core.server_dials import dial_in_range
    return dial_in_range("tuning.ui.demo_select_retry_ms", _SELECT_RETRY_MS, 50, 1000)


def _demo_select_max_tries() -> int:
    from ...core.server_dials import dial_in_range
    return dial_in_range("tuning.ui.demo_select_max_tries", _SELECT_MAX_TRIES, 1, 50)


def _same_source(a: str, b: str) -> bool:
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


def add_library_example_raster(plugin, path: str, label: str,
                               set_zone: bool = False) -> QgsRasterLayer | None:






    project = QgsProject.instance()
    layer = None
    for existing in project.mapLayers().values():
        if isinstance(existing, QgsRasterLayer) and _same_source(existing.source(), path):
            layer = existing
            break
    if layer is None:
        layer = QgsRasterLayer(path, tr("{label} example").format(label=label), "gdal")
        if not layer.isValid():
            log("library raster: the downloaded file is not a valid raster")
            try:
                plugin.iface.messageBar().pushWarning(
                    tr("Segment library"), tr("This image could not be read."))
            except (RuntimeError, AttributeError):
                pass
            return None
        project.addMapLayer(layer, False)
        project.layerTreeRoot().insertLayer(0, layer)
    try:
        plugin.iface.setActiveLayer(layer)
    except (RuntimeError, AttributeError):
        pass
    layer_id = layer.id()
    reenter = _leave_setup_for_library_layer(plugin, layer_id)
    _frame_library_raster(plugin, layer_id)
    _select_library_raster(plugin, layer_id, 0, set_zone, reenter)


    for delay_ms in (80, 250, 600):
        QTimer.singleShot(delay_ms, lambda lid=layer_id: _frame_library_raster(plugin, lid))
    return layer


def _extent_in_canvas_crs(plugin, layer):
    try:
        dst = plugin.iface.mapCanvas().mapSettings().destinationCrs()
    except (RuntimeError, AttributeError):
        return None
    rect = layer.extent()
    if dst.isValid() and layer.crs().isValid() and layer.crs() != dst:
        try:
            xform = QgsCoordinateTransform(layer.crs(), dst, QgsProject.instance())
            rect = xform.transformBoundingBox(rect)
        except Exception:  # nosec B110
            pass
    return rect


def _frame_library_raster(plugin, layer_id: str) -> None:
    layer = QgsProject.instance().mapLayer(layer_id)
    if layer is None:
        return
    rect = _extent_in_canvas_crs(plugin, layer)
    if rect is None:
        return
    try:
        canvas = plugin.iface.mapCanvas()
        canvas.setExtent(rect)
        canvas.refresh()
    except (RuntimeError, AttributeError):
        pass


def _automatic_setup_idle(plugin, dock) -> bool:

    from ..dock.widgets import Mode
    if getattr(dock, "_mode", None) != Mode.AUTOMATIC:
        return False
    if getattr(dock, "_auto_run_active", False) or getattr(dock, "_auto_review_active", False):
        return False
    return (getattr(plugin, "_auto_worker", None) is None
            and getattr(plugin, "_auto_review", None) is None)


def _leave_setup_for_library_layer(plugin, layer_id: str) -> bool:





    dock = getattr(plugin, "dock_widget", None)
    if dock is None:
        return False
    try:
        if not _automatic_setup_idle(plugin, dock):
            return False
        if dock.auto_steps.currentIndex() == 0:
            return False
        current = dock.auto_layer_combo.currentLayer()
        if current is not None and current.id() == layer_id:
            return False
        plugin._on_auto_exit_clicked()
    except (RuntimeError, AttributeError) as exc:
        log(f"library raster: could not leave the form ({type(exc).__name__})")
        return False
    return True


def _reenter_zone_step(plugin) -> None:


    dock = getattr(plugin, "dock_widget", None)
    if dock is None:
        return
    try:
        from ...core import telemetry_run_context
        telemetry_run_context.begin_run_attempt()
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    try:
        dock._auto_started = True
        dock._go_to_auto_step(1)
    except (RuntimeError, AttributeError):
        pass


def _select_library_raster(plugin, layer_id: str, tries: int, set_zone: bool = False,
                           reenter: bool = False) -> None:


    dock = getattr(plugin, "dock_widget", None)
    layer = QgsProject.instance().mapLayer(layer_id)
    if dock is None or layer is None:
        return
    try:
        dock.auto_layer_combo.setLayer(layer)
        resolved = dock.auto_layer_combo.currentLayer()
    except (RuntimeError, AttributeError):
        return
    if resolved is not None and resolved.id() == layer_id:
        if reenter:
            _reenter_zone_step(plugin)
        if set_zone:
            QTimer.singleShot(0, lambda: _set_library_zone(plugin, layer_id))
        return
    if tries < _demo_select_max_tries():
        QTimer.singleShot(
            _demo_select_retry_ms(),
            lambda: _select_library_raster(plugin, layer_id, tries + 1, set_zone, reenter))
        return
    log("library raster: combo did not resolve the example layer")


def _set_library_zone(plugin, layer_id: str) -> None:



    dock = getattr(plugin, "dock_widget", None)
    layer = QgsProject.instance().mapLayer(layer_id)
    if dock is None or layer is None:
        return
    try:
        if not _automatic_setup_idle(plugin, dock):
            return
        current = dock.auto_layer_combo.currentLayer()
        if current is None or current.id() != layer_id:
            return
    except (RuntimeError, AttributeError):
        return
    rect = _extent_in_canvas_crs(plugin, layer)
    if rect is None or rect.isEmpty():
        return
    try:
        if getattr(plugin, "_zone_selection_tool", None) is None:
            plugin._setup_auto_mode()
        plugin._clear_auto_canvas()
        plugin._on_zone_polygon_drawn(QgsGeometry.fromRect(rect))
    except (RuntimeError, AttributeError) as exc:
        log(f"library raster: zone not set ({type(exc).__name__})")
