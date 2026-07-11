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


class DemoSceneMixin:
    """Plugin-side handler for the dock's first-run demo hero."""

    def _on_auto_demo_requested(self) -> None:
        """Load the demo basemap, frame the scene, then (deferred one tick so
        the layer combo picks up the new layer) select it as the working layer.
        The flow itself is left untouched: the user draws a zone and clicks
        Start themselves, exactly as with their own imagery."""
        layer = QgsRasterLayer(_ESRI_WORLD_IMAGERY_URI, "Satellite (Esri)", "wms")
        if not layer.isValid():
            log("demo scene: basemap failed to load")
            try:
                self.iface.messageBar().pushWarning(
                    "AI Segmentation",
                    tr("Couldn't load the demo imagery. Check your internet "
                       "connection, or add your own layer."))
            except (RuntimeError, AttributeError):
                pass
            return
        project = QgsProject.instance()
        project.addMapLayer(layer, False)
        # Bottom of the tree: any future output layers must stack above the
        # basemap (same convention as AI Edit's onboarding backdrop).
        project.layerTreeRoot().insertLayer(-1, layer)
        self._demo_layer_id = layer.id()
        self._fly_to_demo_scene()
        # addMapLayer queues QGIS's own zoom-to-first-layer and the combo
        # refresh; deferring past them keeps our framing and lets the combo
        # resolve the new layer before we select it.
        self._demo_select_tries = 0
        QTimer.singleShot(0, self._select_demo_layer)

    def _fly_to_demo_scene(self) -> None:
        """Frame the demo scene (plus margin) in the canvas CRS."""
        from qgis.core import (
            QgsCoordinateReferenceSystem,
            QgsCoordinateTransform,
            QgsRectangle,
        )

        xmin, ymin, xmax, ymax = _DEMO_VIEW_WGS84
        rect = QgsRectangle(xmin, ymin, xmax, ymax)
        rect.scale(_DEMO_VIEW_SCALE)
        canvas = self.iface.mapCanvas()
        src = QgsCoordinateReferenceSystem("EPSG:4326")
        dst = canvas.mapSettings().destinationCrs()
        if dst.isValid() and src != dst:
            try:
                xform = QgsCoordinateTransform(src, dst, QgsProject.instance())
                rect = xform.transformBoundingBox(rect)
            except Exception:  # nosec B110 -- invalid custom CRS: keep WGS84 rect
                pass
        canvas.setExtent(rect)
        canvas.refresh()

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
        if dock.auto_layer_combo.currentLayer() is None:
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
