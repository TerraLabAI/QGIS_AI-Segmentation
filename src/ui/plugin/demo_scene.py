










from __future__ import annotations

from qgis.core import QgsProject, QgsRasterLayer
from qgis.PyQt.QtCore import QTimer

from ...core.i18n import tr
from ...core.logging_utils import log



_ESRI_WORLD_IMAGERY_URI = (
    "type=xyz&url=https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/%7Bz%7D/%7By%7D/%7Bx%7D&zmax=21&zmin=0"
)






_DEMO_VIEW_WGS84 = (-82.0033, 26.5582, -81.9952, 26.5640)

_DEMO_VIEW_SCALE = 1.6




_DEMO_SELECT_RETRY_MS = 150
_DEMO_SELECT_MAX_TRIES = 10


def _demo_view_wgs84() -> tuple:




    try:
        import math

        from ...core.server_dials import read_value

        value = read_value("tuning.ui.demo_view_bbox")
        if isinstance(value, (list, tuple)) and len(value) == 4:
            west, south, east, north = (float(v) for v in value)
            if (all(math.isfinite(v) for v in (west, south, east, north))
                    and west < east and south < north):
                return (west, south, east, north)
    except Exception:  # noqa: BLE001
        return _DEMO_VIEW_WGS84
    return _DEMO_VIEW_WGS84


def _demo_view_scale() -> float:
    from ...core.server_dials import dial_in_range
    return dial_in_range("tuning.ui.demo_view_scale", _DEMO_VIEW_SCALE, 1.0, 3.0)


def _demo_select_retry_ms() -> int:
    from ...core.server_dials import dial_in_range
    return dial_in_range("tuning.ui.demo_select_retry_ms", _DEMO_SELECT_RETRY_MS, 50, 1000)


def _demo_select_max_tries() -> int:
    from ...core.server_dials import dial_in_range
    return dial_in_range("tuning.ui.demo_select_max_tries", _DEMO_SELECT_MAX_TRIES, 1, 50)





_DEMO_PROBE_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/13/3400/2200"
)



_DEMO_REFRAME_WIDTH_FACTOR = 3.0


def _extents_match(a, b) -> bool:






    try:
        tolerance = max(abs(b.width()), abs(b.height())) * 1e-6
        return (abs(a.xMinimum() - b.xMinimum()) <= tolerance
                and abs(a.yMinimum() - b.yMinimum()) <= tolerance
                and abs(a.xMaximum() - b.xMaximum()) <= tolerance
                and abs(a.yMaximum() - b.yMaximum()) <= tolerance)
    except (RuntimeError, AttributeError):
        return False


class DemoSceneMixin:


    def _on_auto_demo_requested(self) -> None:




        project = QgsProject.instance()


        layer = project.mapLayer(getattr(self, "_demo_layer_id", "") or "")
        if layer is None:
            layer = QgsRasterLayer(_ESRI_WORLD_IMAGERY_URI, "Satellite (Esri)", "wms")
            if not layer.isValid():
                log("demo scene: basemap failed to load")
                self._warn_demo_imagery_unavailable()
                return
            project.addMapLayer(layer, False)


            project.layerTreeRoot().insertLayer(-1, layer)
            self._demo_layer_id = layer.id()
            self._probe_demo_imagery()
        self._demo_framed_extent = None
        self._fly_to_demo_scene()



        self._demo_select_tries = 0
        QTimer.singleShot(0, self._select_demo_layer)

    def _warn_demo_imagery_unavailable(self) -> None:

        try:
            self.iface.messageBar().pushWarning(
                "AI Segmentation",
                tr("Couldn't load the demo imagery. Check your internet "
                   "connection, or add your own layer."))
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _probe_demo_imagery(self) -> None:






        try:
            from qgis.core import QgsNetworkAccessManager
            from qgis.PyQt.QtCore import QUrl
            from qgis.PyQt.QtNetwork import QNetworkRequest
        except ImportError:
            return
        try:
            reply = QgsNetworkAccessManager.instance().get(
                QNetworkRequest(QUrl(_DEMO_PROBE_TILE_URL)))
        except (RuntimeError, AttributeError):
            return


        self._demo_probe_reply = reply
        reply.finished.connect(self._on_demo_probe_finished)

    def _on_demo_probe_finished(self) -> None:

        reply = getattr(self, "_demo_probe_reply", None)
        self._demo_probe_reply = None
        if reply is None:
            return
        try:
            from qgis.PyQt.QtNetwork import QNetworkReply

            from ...core.qt_compat import resolve_qt_enum
            no_error = resolve_qt_enum(QNetworkReply, "NetworkError", "NoError")
            failed = reply.error() != no_error
        except (ImportError, RuntimeError, AttributeError):
            failed = False
        try:
            reply.deleteLater()
        except (RuntimeError, AttributeError):
            pass  # nosec B110
        if failed:
            log("demo scene: tile probe failed, imagery is not reachable")
            self._warn_demo_imagery_unavailable()

    def _fly_to_demo_scene(self) -> None:








        from qgis.core import (
            QgsCoordinateReferenceSystem,
            QgsCoordinateTransform,
            QgsRectangle,
        )

        if self.dock_widget is None:
            return
        xmin, ymin, xmax, ymax = _demo_view_wgs84()
        rect = QgsRectangle(xmin, ymin, xmax, ymax)
        rect.scale(_demo_view_scale())
        try:
            canvas = self.iface.mapCanvas()
            src = QgsCoordinateReferenceSystem("EPSG:4326")
            dst = canvas.mapSettings().destinationCrs()
        except (RuntimeError, AttributeError):
            return
        if dst.isValid() and src != dst:
            try:
                xform = QgsCoordinateTransform(src, dst, QgsProject.instance())
                rect = xform.transformBoundingBox(rect)
            except Exception:  # nosec B110
                pass
        last = getattr(self, "_demo_framed_extent", None)
        if last is not None:
            try:
                current = canvas.extent()
            except (RuntimeError, AttributeError):
                return
            if not _extents_match(current, last):




                from ...core.server_dials import dial_in_range
                reframe_factor = dial_in_range(
                    "tuning.ui.demo_reframe_width_factor",
                    _DEMO_REFRAME_WIDTH_FACTOR, 1.5, 10.0)
                if current.width() <= rect.width() * reframe_factor:
                    return
        try:
            canvas.setExtent(rect)
            canvas.refresh()
            self._demo_framed_extent = canvas.extent()
        except (RuntimeError, AttributeError):
            pass

    def _select_demo_layer(self) -> None:




        dock = self.dock_widget
        if dock is None:
            return





        self._fly_to_demo_scene()


        try:
            layer = QgsProject.instance().mapLayer(
                getattr(self, "_demo_layer_id", ""))
            if layer is not None:
                dock.auto_layer_combo.setLayer(layer)
        except (RuntimeError, AttributeError):
            pass


        try:
            resolved = dock.auto_layer_combo.currentLayer()
        except (RuntimeError, AttributeError):
            return
        if resolved is None:

            tries = getattr(self, "_demo_select_tries", 0)
            if tries < _demo_select_max_tries():
                self._demo_select_tries = tries + 1
                QTimer.singleShot(_demo_select_retry_ms(), self._select_demo_layer)
                return
            log("demo scene: combo did not resolve the demo layer")
            return



        for _delay_ms in (80, 250, 600):
            QTimer.singleShot(_delay_ms, self._fly_to_demo_scene)
