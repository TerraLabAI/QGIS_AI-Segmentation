"""One-click demo place for the Automatic first run.

A brand-new user usually installs out of curiosity with no imagery loaded;
the empty canvas is where most first sessions die. The dock's first-run hero
(no-rasters state of the Automatic Start step) asks the plugin for a demo:
load a world-wide satellite basemap, fly to a curated place, and enter the
Automatic flow with the zone already drawn and the prompt already filled, so
the user's next click is Detect (learn by doing: the run itself stays theirs).

The zone commit reuses the exact history-rerun path, so the wrong-layer
guard, free-trial zone cap, canvas badge, tile grid and credit estimate all
apply for free.
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
# purpose: the place is rich enough that after the first "houses" run the
# user can keep iterating with other prompts on the same imagery. The zone is
# ~0.5 km2 (~800 x 650 m), centred on a dense residential block: wide enough
# that the object-aware "houses" seed lays down a real multi-tile grid (~9
# tiles) so the very first run showcases the tiling, still a negligible slice
# of the 300-credit free trial (well under the 5 km2 zone cap).
_DEMO_ZONE_WGS84 = (-82.0033, 26.5582, -81.9952, 26.5640)
_DEMO_PROMPT = "houses"
# Fly margin around the zone so the scene reads with context, not edge-to-edge.
_DEMO_VIEW_SCALE = 1.6
# The layer combo repopulates from queued layer-tree signals, so the fresh
# basemap can take a few event-loop ticks to become selectable. Retry briefly
# instead of bailing on the first look (one tick is not enough: the demo would
# silently stay on the Start step).
_DEMO_FLOW_RETRY_MS = 150
_DEMO_FLOW_MAX_TRIES = 10


class DemoSceneMixin:
    """Plugin-side handler for the dock's first-run demo hero."""

    def _on_auto_demo_requested(self) -> None:
        """Load the demo basemap, frame the scene, then (deferred one tick so
        the layer combo picks up the new layer) enter the Automatic flow."""
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
        # resolve the new layer before Start locks it.
        self._demo_flow_tries = 0
        QTimer.singleShot(0, self._begin_demo_flow)

    def _fly_to_demo_scene(self) -> None:
        """Frame the demo zone (plus margin) in the canvas CRS."""
        from qgis.core import (
            QgsCoordinateReferenceSystem,
            QgsCoordinateTransform,
            QgsRectangle,
        )

        xmin, ymin, xmax, ymax = _DEMO_ZONE_WGS84
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

    def _begin_demo_flow(self) -> None:
        """Enter the Automatic flow on the demo scene: Start (locks the demo
        raster), zone committed through the normal draw path, prompt
        prefilled. The user keeps the Detect click, where the consent
        checkbox now sits (the first Detect seals it)."""
        dock = self.dock_widget
        if dock is None:
            return
        # Hold the framing on every tick: adding the basemap makes QGIS queue a
        # "zoom to the first layer's full extent" (the whole world, for a global
        # XYZ basemap) that fires on a later event-loop turn and would otherwise
        # override our framing while we wait for the layer combo to resolve,
        # leaving the user staring at the globe with the zone an invisible speck.
        self._fly_to_demo_scene()
        # The fresh basemap should be the locked raster even if another
        # (invisible) raster was already selected in the combo.
        try:
            layer = QgsProject.instance().mapLayer(
                getattr(self, "_demo_layer_id", ""))
            if layer is not None:
                dock.auto_layer_combo.setLayer(layer)
        except (RuntimeError, AttributeError):
            pass
        if dock.auto_layer_combo.currentLayer() is None:
            # Combo model still refreshing: retry a few ticks before giving
            # up. On exhaustion, leave the user on the (now raster-populated)
            # Start step rather than half-starting the flow.
            tries = getattr(self, "_demo_flow_tries", 0)
            if tries < _DEMO_FLOW_MAX_TRIES:
                self._demo_flow_tries = tries + 1
                QTimer.singleShot(_DEMO_FLOW_RETRY_MS, self._begin_demo_flow)
                return
            log("demo scene: combo did not resolve the demo layer")
            return
        geom = self._zone_geom_from_extent(list(_DEMO_ZONE_WGS84), "EPSG:4326")
        # Same base as the history re-run: mode switch, clean reset, Start,
        # prompt prefill (prompt must land after Start, which clears it).
        self._enter_auto_flow_for_history(_DEMO_PROMPT)
        if geom is not None:
            self._on_zone_polygon_drawn(geom)
        # Final insurance against QGIS's queued zoom-to-world: re-assert the demo
        # framing a few more times after the flow settles. setExtent is instant,
        # so this simply pins the view on the scene however late that zoom lands.
        for _delay_ms in (80, 250, 600):
            QTimer.singleShot(_delay_ms, self._fly_to_demo_scene)
