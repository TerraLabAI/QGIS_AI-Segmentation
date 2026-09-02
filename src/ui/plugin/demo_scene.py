"""One-click demo imagery for the first run.

A brand-new user usually installs out of curiosity with no imagery loaded;
the empty canvas is where most first sessions die. The dock's first-run hero
(no-rasters state) asks the plugin for a demo: load a world-wide satellite
basemap, fly to a curated place, and select it as the working layer. That is
all it does. The user then runs the real flow themselves (draw a zone, click
Start, pick an object), so nothing is skipped and there is no half-started
state to unwind if they want to start over.
"""

from __future__ import annotations

from qgis.core import QgsProject, QgsRasterLayer
from qgis.PyQt.QtCore import QTimer

from ...core.i18n import tr
from ...core.logging_utils import log

# Esri World Imagery: the key-free, ToS-clean global backdrop QGIS and
# QuickMapServices ship. zmax=21 unlocks sub-metre tiles in metro areas.
_ESRI_WORLD_IMAGERY_URI = (
    "type=xyz&url=https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/%7Bz%7D/%7By%7D/%7Bx%7D&zmax=21&zmin=0"
)

# Cape Coral, Florida: dense villas along canals, with pools, boats, docks,
# roads and trees in the same frame. One scene chosen over a carousel on
# purpose: the place is rich enough that the user can try several prompts on
# the same imagery. These bounds only frame the view; the user draws their own
# zone inside the scene.
_DEMO_VIEW_WGS84 = (-82.0033, 26.5582, -81.9952, 26.5640)
# Fly margin around the framed extent so the scene reads with context.
_DEMO_VIEW_SCALE = 1.6
# The layer combo repopulates from queued layer-tree signals, so the fresh
# basemap can take a few event-loop ticks to become selectable. Retry briefly
# instead of bailing on the first look (one tick is not enough: the demo layer
# would silently stay unselected).
_DEMO_SELECT_RETRY_MS = 150
_DEMO_SELECT_MAX_TRIES = 10
# One tile off the same service, used only to tell "no network" from "tiles on
# the way". An XYZ raster layer reports isValid() with the network unplugged,
# so the layer itself can never answer that question.
_DEMO_PROBE_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/13/3400/2200"
)
# Past this, the re-framing timers stop treating a changed view as ours to
# undo. QGIS's queued zoom-to-full-extent lands on the whole world, many times
# wider than the scene; a user panning or zooming stays near its own scale.
_DEMO_REFRAME_WIDTH_FACTOR = 3.0


def _extents_match(a, b) -> bool:
    """Whether two canvas extents are the same view.

    Compared with a tolerance because the canvas adjusts what it is given to
    its own aspect ratio, so an extent read straight back is never bit-equal
    to the one just set.
    """
    try:
        tolerance = max(abs(b.width()), abs(b.height())) * 1e-6
        return (abs(a.xMinimum() - b.xMinimum()) <= tolerance
                and abs(a.yMinimum() - b.yMinimum()) <= tolerance
                and abs(a.xMaximum() - b.xMaximum()) <= tolerance
                and abs(a.yMaximum() - b.yMaximum()) <= tolerance)
    except (RuntimeError, AttributeError):
        return False


class DemoSceneMixin:
    """Plugin-side handler for the dock's first-run demo hero."""

    def _on_auto_demo_requested(self) -> None:
        """Load the demo basemap, frame the scene, then (deferred one tick so
        the layer combo picks up the new layer) select it as the working layer.
        The flow itself is left untouched: the user draws a zone and clicks
        Start themselves, exactly as with their own imagery."""
        project = QgsProject.instance()
        # A second press must reuse the basemap it already added, or the tree
        # fills up with identical "Satellite (Esri)" layers.
        layer = project.mapLayer(getattr(self, "_demo_layer_id", "") or "")
        if layer is None:
            layer = QgsRasterLayer(_ESRI_WORLD_IMAGERY_URI, "Satellite (Esri)", "wms")
            if not layer.isValid():
                log("demo scene: basemap failed to load")
                self._warn_demo_imagery_unavailable()
                return
            project.addMapLayer(layer, False)
            # Bottom of the tree: any future output layers must stack above the
            # basemap (same convention as AI Edit's onboarding backdrop).
            project.layerTreeRoot().insertLayer(-1, layer)
            self._demo_layer_id = layer.id()
            self._probe_demo_imagery()
        self._demo_framed_extent = None
        self._fly_to_demo_scene()
        # addMapLayer queues QGIS's own zoom-to-first-layer and the combo
        # refresh; deferring past them keeps our framing and lets the combo
        # resolve the new layer before we select it.
        self._demo_select_tries = 0
        QTimer.singleShot(0, self._select_demo_layer)

    def _warn_demo_imagery_unavailable(self) -> None:
        """Say the demo imagery did not arrive, and what the user can do."""
        try:
            self.iface.messageBar().pushWarning(
                "AI Segmentation",
                tr("Couldn't load the demo imagery. Check your internet "
                   "connection, or add your own layer."))
        except (RuntimeError, AttributeError):
            pass  # nosec B110 -- no message bar left to talk to

    def _probe_demo_imagery(self) -> None:
        """Ask for one tile, and warn if it never arrives.

        An XYZ raster layer answers isValid() from its URI alone, so it says
        yes with the network unplugged and the offline message never fired.
        The request is asynchronous, so the canvas keeps drawing while we wait.
        """
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
        # Held on the plugin: a reply parented to the network manager alone is
        # collected the moment this frame ends, and the callback never runs.
        self._demo_probe_reply = reply
        reply.finished.connect(self._on_demo_probe_finished)

    def _on_demo_probe_finished(self) -> None:
        """Read the probe's outcome, then let the reply go."""
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
            pass  # nosec B110 -- already gone
        if failed:
            log("demo scene: tile probe failed, imagery is not reachable")
            self._warn_demo_imagery_unavailable()

    def _fly_to_demo_scene(self) -> None:
        """Frame the demo scene (plus margin) in the canvas CRS.

        Called from four timers that outlive nothing but themselves: a dock
        closed or a plugin unloaded inside the second these chain over used to
        raise out of a bare timer slot, and re-framed a map the user had
        already moved on with. Both are answered by the dock check plus the
        guard around every canvas touch.
        """
        from qgis.core import (
            QgsCoordinateReferenceSystem,
            QgsCoordinateTransform,
            QgsRectangle,
        )

        if self.dock_widget is None:
            return
        xmin, ymin, xmax, ymax = _DEMO_VIEW_WGS84
        rect = QgsRectangle(xmin, ymin, xmax, ymax)
        rect.scale(_DEMO_VIEW_SCALE)
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
            except Exception:  # nosec B110 -- invalid custom CRS: keep WGS84 rect
                pass
        last = getattr(self, "_demo_framed_extent", None)
        if last is not None:
            try:
                current = canvas.extent()
            except (RuntimeError, AttributeError):
                return
            if not _extents_match(current, last):
                # Someone moved the view. Either QGIS's queued zoom to the
                # basemap's full extent, which lands on the whole world, or the
                # user. Only the first is ours to undo, and its width gives it
                # away; anything near the scene's own scale is left alone.
                if current.width() <= rect.width() * _DEMO_REFRAME_WIDTH_FACTOR:
                    return
        try:
            canvas.setExtent(rect)
            canvas.refresh()
            self._demo_framed_extent = canvas.extent()
        except (RuntimeError, AttributeError):
            pass

    def _select_demo_layer(self) -> None:
        """Select the demo basemap in the Automatic layer combo and leave the
        user on the normal Start step. No zone, prompt or Start is auto-set:
        the user runs the real flow (draw a zone, click Start, pick an object)
        so there is nothing to unwind if they want to start over."""
        dock = self.dock_widget
        if dock is None:
            return
        # Hold the framing on every tick: adding the basemap makes QGIS queue a
        # "zoom to the first layer's full extent" (the whole world, for a global
        # XYZ basemap) that fires on a later event-loop turn and would otherwise
        # override our framing while we wait for the layer combo to resolve,
        # leaving the user staring at the globe with the scene an invisible speck.
        self._fly_to_demo_scene()
        # The fresh basemap should be the working raster even if another
        # (invisible) raster was already selected in the combo.
        try:
            layer = QgsProject.instance().mapLayer(
                getattr(self, "_demo_layer_id", ""))
            if layer is not None:
                dock.auto_layer_combo.setLayer(layer)
        except (RuntimeError, AttributeError):
            pass
        # Read inside the guard too: a dock whose C++ half is gone raises here,
        # and this runs from a timer with nothing above it to catch that.
        try:
            resolved = dock.auto_layer_combo.currentLayer()
        except (RuntimeError, AttributeError):
            return
        if resolved is None:
            # Combo model still refreshing: retry a few ticks before giving up.
            tries = getattr(self, "_demo_select_tries", 0)
            if tries < _DEMO_SELECT_MAX_TRIES:
                self._demo_select_tries = tries + 1
                QTimer.singleShot(_DEMO_SELECT_RETRY_MS, self._select_demo_layer)
                return
            log("demo scene: combo did not resolve the demo layer")
            return
        # Final insurance against QGIS's queued zoom-to-world: re-assert the demo
        # framing a few more times after the layer settles. setExtent is instant,
        # so this simply pins the view on the scene however late that zoom lands.
        for _delay_ms in (80, 250, 600):
            QTimer.singleShot(_delay_ms, self._fly_to_demo_scene)
