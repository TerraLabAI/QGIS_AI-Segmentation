"""Visual exemplars: draw, thumbnails, model-ready stamps, chips.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

import math
import re

from qgis.core import (
    QgsCoordinateTransform,
    QgsGeometry,
    QgsProject,
    QgsRectangle,
)
from qgis.gui import QgsRubberBand
from qgis.PyQt.QtCore import Qt

from ...core.i18n import tr
from ...core.qt_compat import PolygonGeometry
from ..canvas_palette import (
    EXCLUDE_FILL,
    EXCLUDE_STROKE,
    EXEMPLAR_FILL,
    EXEMPLAR_STROKE,
)


class ExemplarsMixin:
    """Visual exemplars: draw, thumbnails, model-ready stamps, chips."""

    # ------------------------------------------------------------------
    # Exemplar nudge in review
    # ------------------------------------------------------------------

    def _maybe_show_exemplar_nudge(self, object_word, scores) -> None:
        """Show the review exemplar tip once for a bottom-heavy, no-example run.

        Fires only when the run's kept scores have a median below 0.35 AND no
        example was drawn (an example IS the lever the tip suggests). Interactive
        only; the caller is the interactive review path."""
        self._auto_nudge_median = None
        if getattr(self, "_auto_headless_run", False):
            return
        try:
            if self._auto_exemplar_store.count() > 0:
                return
        except (RuntimeError, AttributeError):
            return
        vals = sorted(float(s) for s in (scores or []) if s is not None)
        if not vals:
            return
        n = len(vals)
        median = (vals[n // 2] if n % 2
                  else (vals[n // 2 - 1] + vals[n // 2]) / 2.0)
        if median >= 0.35:
            return
        self._auto_nudge_median = median
        obj = object_word or tr("object")
        try:
            self.dock_widget.show_auto_exemplar_nudge(obj)
        except (RuntimeError, AttributeError):
            return
        try:
            from ...core import telemetry, telemetry_events as ev
            telemetry.track(ev.EXEMPLAR_NUDGE_SHOWN, {
                "run_id": getattr(self, "_auto_run_id", "") or "",
                "object_class": obj,
                "median_score": round(median, 3),
            })
        except Exception:
            pass  # nosec B110

    def _on_auto_exemplar_retry_clicked(self) -> None:
        """Review exemplar nudge link: run the standard Adjust & run again path
        (confirm dialog included), then arm the example draw on step 2. Arms only
        when the retry was confirmed, so a cancelled discard leaves nothing armed."""
        median = getattr(self, "_auto_nudge_median", None)
        try:
            from ...core import telemetry, telemetry_events as ev
            telemetry.track(ev.EXEMPLAR_NUDGE_CLICKED, {
                "run_id": getattr(self, "_auto_run_id", "") or "",
                "object_class": (self._auto_run_ctx or {}).get("prompt", "") or "",
                "median_score": round(median, 3) if median is not None else -1.0,
            })
        except Exception:
            pass  # nosec B110
        if self._on_auto_retry_clicked():
            self._on_add_exemplar_requested(1)

    # ------------------------------------------------------------------
    # Visual exemplars ("draw one example, find all")
    # ------------------------------------------------------------------

    def _on_add_exemplar_requested(self, label: int) -> None:
        """Arm the example-POLYGON draw tool (label 1 positive / 0 exclude).

        The user outlines ONE object point-by-point (the same UX as the zone, in
        green for a positive example / red for an exclude). On finish it is stored
        with its polygon + bounding box; the worker masks the stamped crop to the
        polygon so the cloud model's box exemplar isn't polluted by neighbours. No-op without
        a zone, once the store is full, or while a run/review is active.
        """
        # Already armed: the button is a toggle, a second click disarms. This
        # also prevents re-capturing _maptool_before_exemplar from the CURRENT
        # tool (the exemplar tool itself), which corrupted the restore.
        if self._exemplar_maptool is not None:
            self._restore_maptool_after_exemplar()
            return
        # No new examples once a run is in flight or a review is pending: the
        # inputs are locked in for that run.
        if self._auto_worker is not None or self._auto_review is not None:
            return
        if self._auto_zone is None or self._auto_exemplar_store.is_full_for(int(label)):
            return
        from ..polygon_zone_maptool import PolygonZoneMapTool
        canvas = self.iface.mapCanvas()
        # Remember the tool to restore after the one-shot draw (usually the zone tool).
        self._maptool_before_exemplar = canvas.mapTool()
        self._pending_exemplar_label = int(label)
        color = EXEMPLAR_STROKE if label == 1 else EXCLUDE_STROKE
        tool = PolygonZoneMapTool(canvas, color=color)
        tool.zone_selected.connect(self._on_exemplar_polygon_drawn)
        tool.zone_cleared.connect(self._on_exemplar_cancelled)
        tool.back_requested.connect(self._on_exemplar_cancelled)
        self._exemplar_maptool = tool
        # The draw tool owns Escape (cancel) while armed; disable the dock
        # Escape/Enter shortcuts so they cannot exit the flow mid-draw.
        try:
            self.dock_widget.set_auto_shortcuts_enabled(False)
        except (RuntimeError, AttributeError):
            pass
        canvas.setMapTool(tool)
        # Light up the chosen button + show the "now outline on the map"
        # instruction so the click clearly started a draw action.
        try:
            self.dock_widget.set_auto_exemplar_armed(int(label))
        except (RuntimeError, AttributeError):
            pass

    def _on_exemplar_polygon_drawn(self, geom) -> None:
        """Store a drawn example polygon (clipped to the zone), mark it on the
        map, refresh the dock. geom is a polygon in canvas CRS.

        An example drawn ENTIRELY outside the zone is refused with a notification:
        there is no imagery to learn from there and it cannot match anything
        inside. A draw straddling the border is clipped to its in-zone part.
        """
        label = int(getattr(self, "_pending_exemplar_label", 1))
        # Keep the example inside the zone (that is where the imagery is).
        zone_poly = self._auto_zone_polygon
        if (zone_poly is not None and geom is not None and not geom.isEmpty()):
            try:
                clipped = geom.intersection(zone_poly)
            except (RuntimeError, AttributeError):
                clipped = None  # intersection failed: stay lenient, keep original
            if clipped is not None:
                if clipped.isEmpty():
                    # Fully outside the drawn zone: refuse instead of silently
                    # storing an off-zone example the model can never use.
                    self.iface.messageBar().pushWarning(
                        "AI Segmentation",
                        tr("Draw your example inside the selected zone."))
                    self._restore_maptool_after_exemplar()
                    return
                geom = clipped
        if geom is None or geom.isEmpty():
            self._restore_maptool_after_exemplar()
            return
        rect = geom.boundingBox()
        thumb = self._capture_exemplar_thumbnail(rect, polygon=geom)
        eid = self._auto_exemplar_store.add(
            QgsRectangle(rect), label, thumbnail=thumb, polygon=geom)
        if eid is not None:
            band = self._make_exemplar_band_poly(geom, label)
            if band is not None:
                self._exemplar_bands[eid] = band
            self._refresh_exemplar_chips()
            # Prebuild the model stamp now (the canvas already shows these pixels)
            # so Detect never blocks on a per-exemplar render.
            self._prebuild_exemplar_stamp(eid)
            try:
                from ...core import telemetry
                telemetry.track_exemplar_added(count_after=self._auto_exemplar_store.count())
            except Exception:
                pass  # nosec B110
        self._restore_maptool_after_exemplar()

    def _make_exemplar_band_poly(self, geom, label: int):
        """Persistent rubber band tracing one exemplar POLYGON (green/red)."""
        try:
            canvas = self.iface.mapCanvas()
            band = QgsRubberBand(canvas, PolygonGeometry)
            col = EXEMPLAR_STROKE if label == 1 else EXCLUDE_STROKE
            band.setColor(col)
            band.setFillColor(EXEMPLAR_FILL if label == 1 else EXCLUDE_FILL)
            band.setWidth(2)
            band.setToGeometry(geom, None)
            return band
        except (RuntimeError, AttributeError):
            return None

    def _capture_exemplar_thumbnail(self, rect, polygon=None):
        """Grab a QImage crop of the map canvas at the drawn box for the reference
        card, showing the NATURAL crop (real imagery with its surroundings) so the
        card matches what the model receives now. No outline is drawn on top: the
        selection line over the reference read as a bug (it looked like a stray
        stroke on the imagery). Captured large (240 px) so the click-to-enlarge
        detail view stays crisp. Returns an aspect-preserved image or None on
        failure. rect/polygon are canvas CRS."""
        try:
            from qgis.PyQt.QtCore import QRect
            side_px = 240  # capture big so the enlarge view is crisp; card scales it down
            canvas = self.iface.mapCanvas()
            m2p = canvas.getCoordinateTransform()  # QgsMapToPixel
            # y grows up in map coords, down in pixels: yMaximum -> top.
            top_left = m2p.transform(rect.xMinimum(), rect.yMaximum())
            bot_right = m2p.transform(rect.xMaximum(), rect.yMinimum())
            try:
                dpr = float(canvas.devicePixelRatioF())
            except (AttributeError, TypeError):
                dpr = 1.0
            x = int(min(top_left.x(), bot_right.x()) * dpr)
            y = int(min(top_left.y(), bot_right.y()) * dpr)
            w = int(abs(bot_right.x() - top_left.x()) * dpr)
            h = int(abs(bot_right.y() - top_left.y()) * dpr)
            if w < 4 or h < 4:
                return None
            crop = canvas.grab().copy(QRect(x, y, w, h)).toImage()
            if crop.isNull():
                return None
            # Show the NATURAL crop (real imagery with context), matching the
            # reference actually sent to the model. No outline is drawn on top.
            # Keep the FULL aspect ratio (longest side = side_px). A long thin
            # selection (e.g. a solar-panel row) must show its WHOLE length,
            # never be centre-cropped to a square (which showed only the middle
            # and hid the ends).
            return crop.scaled(
                side_px, side_px,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
        except Exception:  # noqa: BLE001 - thumbnail is best-effort, never fatal
            return None

    def _on_exemplar_cancelled(self) -> None:
        """Example draw aborted (Escape / toggle-off): just restore the tool."""
        self._restore_maptool_after_exemplar()

    def _on_exemplar_remove_requested(self, exemplar_id: str) -> None:
        """Remove one exemplar (chip x): drop the box + its rubber band."""
        self._auto_exemplar_store.remove(exemplar_id)
        band = self._exemplar_bands.pop(exemplar_id, None)
        self._remove_rubber_band(band)
        self._refresh_exemplar_chips()
        try:
            from ...core import telemetry
            telemetry.track_exemplar_removed(count_after=self._auto_exemplar_store.count())
        except Exception:
            pass  # nosec B110

    def _clear_exemplars(self) -> None:
        """Drop all exemplars + their rubber bands (zone redraw / exit / reset)."""
        self._auto_exemplar_store.clear()
        for band in self._exemplar_bands.values():
            self._remove_rubber_band(band)
        self._exemplar_bands.clear()
        # Disarm a still-active example draw tool.
        self._restore_maptool_after_exemplar()
        self._refresh_exemplar_chips()

    def _restore_maptool_after_exemplar(self) -> None:
        """Restore the map tool active before a one-shot example draw."""
        tool = self._exemplar_maptool
        # Re-enable the dock Escape/Enter shortcuts the draw had suspended, and
        # clear the armed button + instruction line (covers finish AND cancel).
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_shortcuts_enabled(True)
            except (RuntimeError, AttributeError):
                pass
            try:
                self.dock_widget.set_auto_exemplar_armed(None)
            except (RuntimeError, AttributeError):
                pass
        if tool is None:
            return
        self._exemplar_maptool = None
        try:
            canvas = self.iface.mapCanvas()
            prev = self._maptool_before_exemplar
            if prev is not None:
                canvas.setMapTool(prev)
            elif canvas.mapTool() == tool:
                canvas.unsetMapTool(tool)
        except (RuntimeError, AttributeError):
            pass
        self._maptool_before_exemplar = None

    def _restore_maptool_after_zone(self) -> None:
        """Disarm the zone drawing tool and restore the tool active before it
        (QGIS's pan/hand by default), mirroring Manual's _restore_previous_map_tool.

        unsetMapTool alone leaves the canvas with NO tool (a bare cursor); after
        drawing a zone, or on exit/finish, the user expects the pan tool back so
        they can move around the map. Only acts when the zone tool is the current
        tool, so it never stomps a tool the user has already switched to.
        """
        tool = self._zone_selection_tool
        prev = self._maptool_before_zone
        self._maptool_before_zone = None
        if tool is None:
            return
        try:
            canvas = self.iface.mapCanvas()
            if canvas.mapTool() != tool:
                return  # already disarmed; leave the user's current tool alone
            if prev is not None and prev is not tool:
                canvas.setMapTool(prev)
            else:
                canvas.unsetMapTool(tool)
        except (RuntimeError, AttributeError):
            pass

    def _remove_rubber_band(self, band) -> None:
        """Remove a rubber band from the canvas scene."""
        if band is None:
            return
        try:
            self.iface.mapCanvas().scene().removeItem(band)
        except (RuntimeError, AttributeError):
            pass

    def _refresh_exemplar_chips(self) -> None:
        """Push the current exemplars to the dock chip strip."""
        if not self.dock_widget:
            return
        try:
            items = [(e.id, e.label, e.thumbnail)
                     for e in self._auto_exemplar_store.list()]
            self.dock_widget.set_exemplars(items)
        except (RuntimeError, AttributeError):
            pass

    def _compute_exemplar_pixel_boxes(
        self, layer, geo_bbox: tuple, pixel_w: int, pixel_h: int
    ) -> list[dict]:
        """Convert stored exemplar map boxes to xyxy PIXEL boxes on the rendered
        zone image. geo_bbox is the actual rendered extent in layer CRS."""
        payload: list[dict] = []
        ext_minx, ext_miny, ext_maxx, ext_maxy = geo_bbox
        ext_w = ext_maxx - ext_minx
        ext_h = ext_maxy - ext_miny
        if ext_w <= 0 or ext_h <= 0 or pixel_w <= 0 or pixel_h <= 0:
            return payload

        def _clamp(v, lo, hi):
            return max(lo, min(hi, v))

        for ex in self._auto_exemplar_store.list():
            try:
                rect_layer = self._reproject_zone_to_layer_crs(ex.map_rect, layer)
            except (RuntimeError, AttributeError):
                continue
            x0 = (rect_layer.xMinimum() - ext_minx) / ext_w * pixel_w
            x1 = (rect_layer.xMaximum() - ext_minx) / ext_w * pixel_w
            # y flip: map y grows up, pixel y grows down.
            y0 = (ext_maxy - rect_layer.yMaximum()) / ext_h * pixel_h
            y1 = (ext_maxy - rect_layer.yMinimum()) / ext_h * pixel_h
            box = [
                _clamp(x0, 0, pixel_w), _clamp(y0, 0, pixel_h),
                _clamp(x1, 0, pixel_w), _clamp(y1, 0, pixel_h),
            ]
            if box[2] - box[0] >= 1 and box[3] - box[1] >= 1:
                entry = {"box": [round(v, 1) for v in box], "label": int(ex.label)}
                ring = self._exemplar_polygon_px(
                    ex, layer, geo_bbox, pixel_w, pixel_h)
                if ring:
                    entry["polygon_px"] = ring
                payload.append(entry)
        return payload

    def _exemplar_polygon_px(self, ex, layer, geo_bbox, pixel_w, pixel_h):
        """Exterior ring of an exemplar's polygon in FULL-image pixel coords, so
        the worker can mask the stamped crop to the exact drawn shape. Returns []
        when the exemplar has no polygon (box-only / MCP path) or on any error."""
        poly = getattr(ex, "polygon", None)
        if poly is None or poly.isEmpty():
            return []
        ext_minx, ext_miny, ext_maxx, ext_maxy = geo_bbox
        ext_w = ext_maxx - ext_minx
        ext_h = ext_maxy - ext_miny
        if ext_w <= 0 or ext_h <= 0:
            return []
        g = QgsGeometry(poly)  # canvas CRS copy
        try:
            canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
            layer_crs = layer.crs()
            if (canvas_crs.isValid() and layer_crs.isValid()
                    and canvas_crs != layer_crs):  # noqa: W503
                g.transform(QgsCoordinateTransform(
                    canvas_crs, layer_crs, QgsProject.instance()))
        except Exception:  # noqa: BLE001
            return []
        try:
            rings = g.asPolygon()
            if not rings:
                mp = g.asMultiPolygon()
                rings = mp[0] if mp else []
            ring = rings[0] if rings else []
        except (AttributeError, IndexError):
            return []
        out = []
        for pt in ring:
            px = (pt.x() - ext_minx) / ext_w * pixel_w
            py = (ext_maxy - pt.y()) / ext_h * pixel_h
            out.append([round(px, 1), round(py, 1)])
        return out

    def _mask_stamp_to_polygon(self, img, ex, layer, act):
        """Neutralize everything OUTSIDE the drawn example polygon in the rendered
        crop. NO LONGER called from the stamp path: live user feedback showed the
        natural crop (real surrounding pixels) recalls more true objects than an
        object-on-grey patch, and the tight object box handles neighbour leak.
        Kept for reference / possible reuse. Returns a new QImage (object on
        neutral grey), or the input unchanged when the exemplar has no polygon
        (box-only / MCP path) or on any error."""
        poly = getattr(ex, "polygon", None)
        if poly is None or poly.isEmpty() or act is None:
            return img
        try:
            from qgis.PyQt.QtGui import (
                QImage, QPainter, QColor, QPolygonF, QPainterPath)
            from qgis.PyQt.QtCore import QPointF
            g = QgsGeometry(poly)  # canvas CRS copy
            canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
            layer_crs = layer.crs()
            if (canvas_crs.isValid() and layer_crs.isValid()
                    and canvas_crs != layer_crs):  # noqa: W503
                g.transform(QgsCoordinateTransform(
                    canvas_crs, layer_crs, QgsProject.instance()))
            rings = g.asPolygon()
            if not rings:
                mp = g.asMultiPolygon()
                rings = mp[0] if mp else []
            ring = rings[0] if rings else []
            if len(ring) < 3:
                return img
            aw = act.xMaximum() - act.xMinimum()
            ah = act.yMaximum() - act.yMinimum()
            iw, ih = img.width(), img.height()
            if aw <= 0 or ah <= 0 or iw <= 0 or ih <= 0:
                return img
            # Map the polygon ring to crop pixels (y flip: map y up, pixel y down).
            pts = [
                QPointF((p.x() - act.xMinimum()) / aw * iw,
                        (act.yMaximum() - p.y()) / ah * ih)
                for p in ring
            ]
            masked = QImage(img.size(), QImage.Format.Format_RGB32)
            masked.fill(QColor(127, 127, 127))  # neutral grey = clearly not an object
            painter = QPainter(masked)
            path = QPainterPath()
            path.addPolygon(QPolygonF(pts))
            painter.setClipPath(path)
            painter.drawImage(0, 0, img)
            painter.end()
            return masked
        except Exception:  # noqa: BLE001 -- masking is best-effort; keep the raw crop
            return img

    # Longest-side render bounds for an exemplar stamp, px. The stamp is pasted
    # into a 1008 tile corner, so keep it small enough not to occlude real
    # imagery, but large enough to stay crisp.
    #
    # Min is deliberately LOW: a genuinely small object rendered crisp at the
    # source's true resolution beats an upsampled-blurry crop stretched to a
    # bigger pixel count. Do not manufacture blur just to hit a pixel count
    # (crisp-small beats blurry-big). Max is 512 (crisper large objects) and
    # still fits the 1008 tile corner.
    _STAMP_MIN_SIDE = 96
    _STAMP_MAX_SIDE = 512
    # Absolute floor when the RUN's tile resolution (not the source's finest)
    # sized the stamp: matching the run scale is the point (see
    # _exemplar_render_gsd), so inflating a small object back to 96 px would
    # both blur it (upsample) and reintroduce the scale mismatch that made
    # exemplar matching grab big context structures. Below ~32 px the crop
    # carries no usable texture, so that stays the hard floor.
    _STAMP_ABS_MIN_SIDE = 32
    # Sane default ground resolution when the source's true resolution cannot be
    # determined at all (metres/px). Never fall back to a fixed pixel count.
    _STAMP_FALLBACK_GSD_M = 0.15
    # Web-mercator ground resolution at zoom 0 (earth circumference / 256 px),
    # metres/px at the equator. Divide by 2**zoom, scale by cos(latitude).
    _WEBMERC_GSD0_M = 156543.03392

    def _exemplar_stamp_longest_side(self, layer, padded, run_mupp: float = 0.0) -> int:
        """Longest render side (px) for an exemplar stamp.

        Target resolution = the RUN's tile resolution when known (``run_mupp``,
        layer-CRS units/px), floored at the source's true finest resolution so
        the crop never upsamples past real detail. Matching the run scale is
        what makes exemplar matching work: the model matches by apparent scale,
        so a reference rendered finer than the tiles grabs bigger context
        structures (parcel edges, background patches) instead of the drawn
        object.

        ``padded`` is the render extent in the layer CRS. A local raster uses
        its genuine native resolution; an online tiled source (XYZ/WMS)
        reports an unreliable units/px, so its finest resolution is derived
        from the source max zoom instead (web-mercator GSD,
        latitude-corrected)."""
        gsd_m, longest_m, run_matched = self._exemplar_render_gsd(
            layer, padded, run_mupp)
        if gsd_m <= 0 or longest_m <= 0:
            return self._STAMP_MAX_SIDE
        side = int(round(longest_m / gsd_m))
        # When the run scale sized the stamp, do NOT inflate a small object
        # back to the legacy 96 px floor: that would upsample-blur it AND
        # reintroduce the scale mismatch (crisp-and-run-scaled beats big).
        floor = self._STAMP_ABS_MIN_SIDE if run_matched else self._STAMP_MIN_SIDE
        return max(floor, min(self._STAMP_MAX_SIDE, side))

    def _exemplar_render_gsd(
        self, layer, padded, run_mupp: float = 0.0
    ) -> tuple[float, float, bool]:
        """Return ``(target_gsd, longest_extent, run_matched)``: the ground
        resolution the stamp should be rendered at, the padded extent's longer
        side in the same units, and whether the RUN's resolution (not the
        source's finest) set the target. Dividing longest by gsd yields the
        render pixel count.

        ``run_mupp`` is the run's tile resolution in layer-CRS units/px (0 =
        unknown, size at source-finest as before). The target is
        ``max(source_finest, run_mupp)``: the stamp matches the tiles'
        apparent scale, but never upsamples past the detail the source holds.

        - Local raster with a genuine native resolution: native GSD in the
          layer's own CRS units, extent taken as-is; run_mupp is in the same
          units, so the max() is direct. The caller only divides longest by
          gsd, so the CRS units cancel and the ratio is unit-agnostic.
        - Online tiled source (web-mercator): the finest GSD at the source max
          zoom, latitude-corrected to real metres; run_mupp (projected units)
          is scaled by the same cosine before the max() so both are metres.
        - Neither: run_mupp when known (same units as the extent), else the
          sane fallback GSD.
        """
        if not self._layer_is_online_tiled(layer):
            try:
                rupp_x = float(layer.rasterUnitsPerPixelX())
                rupp_y = float(layer.rasterUnitsPerPixelY())
                if rupp_x > 0 and rupp_y > 0:
                    src_gsd = max(rupp_x, rupp_y)
                    gsd = max(src_gsd, run_mupp) if run_mupp > 0 else src_gsd
                    longest = max(padded.width(), padded.height())
                    return gsd, longest, gsd > src_gsd
            except (AttributeError, RuntimeError, ValueError):
                pass
        else:
            zmax = self._layer_max_zoom(layer)
            if zmax > 0:
                lat = self._extent_centre_latitude(layer, padded)
                cos_lat = max(math.cos(math.radians(lat)), 1e-6)
                # Web-mercator projected units/px are constant across latitude;
                # multiply by cos(lat) to get real ground metres/px (and metres).
                src_gsd = self._WEBMERC_GSD0_M / (2 ** zmax) * cos_lat
                run_m = run_mupp * cos_lat if run_mupp > 0 else 0.0
                gsd = max(src_gsd, run_m)
                longest = max(padded.width(), padded.height()) * cos_lat
                return gsd, longest, gsd > src_gsd
        # Unknown source resolution: the run scale when known (same layer-CRS
        # units as the extent), else the sane metric default.
        if run_mupp > 0:
            return run_mupp, max(padded.width(), padded.height()), True
        return self._STAMP_FALLBACK_GSD_M, max(padded.width(), padded.height()), False

    @staticmethod
    def _layer_is_online_tiled(layer) -> bool:
        """True when the layer is served by an online tiled provider (XYZ/WMS/
        WMTS), where rasterUnitsPerPixel is unreliable. XYZ layers use the 'wms'
        provider with ``type=xyz`` in the URI, so check both the provider name
        and the data-source URI."""
        name = ""
        try:
            prov = layer.dataProvider()
            if prov is not None:
                name = (prov.name() or "").lower()
        except (AttributeError, RuntimeError):
            name = ""
        if not name:
            try:
                name = (layer.providerType() or "").lower()
            except (AttributeError, RuntimeError):
                name = ""
        if name in ("wms", "wmts", "xyz"):
            return True
        try:
            uri = (layer.dataProvider().dataSourceUri() or "").lower()
        except (AttributeError, RuntimeError):
            uri = ""
        return "type=xyz" in uri or "zmax=" in uri or "tiles=" in uri

    def _layer_max_zoom(self, layer) -> int:
        """Parse ``zmax`` from an online tiled layer's data-source URI. Falls back
        to 19 (a common web-tile max) when absent, since this is only asked of a
        layer already known to be online tiled."""
        try:
            uri = layer.dataProvider().dataSourceUri() or ""
        except (AttributeError, RuntimeError):
            return 19
        m = re.search(r"zmax=(\d+)", uri)
        if m:
            try:
                z = int(m.group(1))
                if 0 < z <= 30:
                    return z
            except ValueError:
                pass
        return 19

    def _extent_centre_latitude(self, layer, padded) -> float:
        """Latitude (deg) of the padded extent centre, for the web-mercator GSD
        cosine correction. Transforms the centre from the layer CRS to EPSG:4326;
        on failure inverts spherical web-mercator y, then defaults to 0 (equator,
        no correction)."""
        cx = (padded.xMinimum() + padded.xMaximum()) / 2.0
        cy = (padded.yMinimum() + padded.yMaximum()) / 2.0
        try:
            from qgis.core import QgsCoordinateReferenceSystem, QgsPointXY
            layer_crs = layer.crs()
            wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
            if layer_crs.isValid() and wgs84.isValid() and layer_crs == wgs84:
                return cy
            if layer_crs.isValid() and wgs84.isValid() and layer_crs != wgs84:
                xform = QgsCoordinateTransform(layer_crs, wgs84, QgsProject.instance())
                lat = xform.transform(QgsPointXY(cx, cy)).y()
                if -89.9 <= lat <= 89.9:
                    return lat
        except Exception:  # noqa: BLE001 - fall back to the mercator inverse  # nosec B110
            pass
        try:
            r = 6378137.0  # spherical web-mercator earth radius
            return math.degrees(2.0 * math.atan(math.exp(cy / r)) - math.pi / 2.0)
        except (ValueError, OverflowError):
            return 0.0

    def _exemplar_run_mupp(self, layer) -> float:
        """Layer-CRS ground units per pixel the CURRENT run renders tiles at
        (zone + detail slider), so exemplar crops can be rendered at the same
        apparent scale as the tiles (the exemplar-quality fix, see
        _exemplar_stamp_longest_side). Falls back to the last run's captured
        gsd, then 0.0 (= size at source-finest, the legacy behaviour)."""
        try:
            if self._auto_zone is not None and layer is not None:
                zone_in_layer = self._reproject_zone_to_layer_crs(
                    self._auto_zone, layer)
                sized = self._grid_for_detail(
                    layer, zone_in_layer, self._get_auto_detail_level())
                if sized is not None and sized[2] > 0:
                    return float(sized[2])
        except (RuntimeError, AttributeError, ValueError, TypeError):
            pass
        gsd = getattr(self, "_auto_gsd", 0.0)
        return float(gsd) if gsd and gsd > 0 else 0.0

    @staticmethod
    def _stamp_gsd_matches(cached: float, current: float) -> bool:
        """True when a cached stamp's render resolution still matches the run's
        (5% tolerance, so a hairline detail nudge never forces a re-render).
        An unknown current scale accepts any cache; a legacy source-finest
        cache (0.0) is rebuilt once the run scale is known."""
        if current <= 0:
            return True
        if cached <= 0:
            return False
        return abs(cached - current) / max(cached, current) <= 0.05

    def _render_exemplar_stamp(self, ex, layer, run_mupp: float = 0.0):
        """Render ONE exemplar's natural crop + object box from the LAYER at
        the RUN's tile resolution (floored at the source's true finest, see
        _exemplar_stamp_longest_side). Returns ``(img, obj_box)`` or None.

        ``obj_box`` is ``[x0,y0,x1,y1]`` in crop pixels framing the drawn object,
        so the worker sends the cloud model a box tight to the object. The crop keeps the
        REAL surrounding pixels (natural context): masking the surroundings to
        flat grey produced an out-of-distribution "object on grey" patch that
        under-recalled in live user testing, so the tight box (not grey paint) is
        what mitigates neighbour leak. This is the expensive part (a
        QgsMapRendererParallelJob + nested event loop), pulled out so it can run
        at DRAW time (_prebuild_exemplar_stamp) or lazily at Detect."""
        from ...core.cloud_detection import render_zone_to_image
        from ...core.detection_policy import exemplar_context_pad
        # Small margin so the polygon is not clipped at the crop edge.
        context_pad = exemplar_context_pad()
        try:
            rect = self._reproject_zone_to_layer_crs(ex.map_rect, layer)
        except (RuntimeError, AttributeError):
            return None
        ew, eh = rect.width(), rect.height()
        if ew <= 0 or eh <= 0:
            return None
        # Pad the rendered extent with surrounding context; the object box stays
        # tight (computed below) so the object still dominates the ROI.
        pad_x = ew * context_pad
        pad_y = eh * context_pad
        padded = QgsRectangle(
            rect.xMinimum() - pad_x, rect.yMinimum() - pad_y,
            rect.xMaximum() + pad_x, rect.yMaximum() + pad_y)
        fw, fh = padded.width(), padded.height()
        if fw <= 0 or fh <= 0:
            return None
        # Size the render at the RUN's tile resolution (floored at the source's
        # true finest so it never upsamples): the reference must show the
        # object at the SAME apparent pixel size the tiles will, or the model
        # matches coarser context structures instead of the object.
        target = self._exemplar_stamp_longest_side(layer, padded, run_mupp)
        if fw >= fh:
            pw, ph = target, max(16, int(round(target * fh / fw)))
        else:
            ph, pw = target, max(16, int(round(target * fw / fh)))
        try:
            img, act = render_zone_to_image(layer, padded, pw, ph)
        except Exception:  # noqa: BLE001
            img, act = None, None
        if img is None or img.isNull() or act is None:
            return None
        # Natural context is sent (no grey mask): live user feedback showed the
        # real surrounding pixels recall more true objects than an object-on-grey
        # patch. The tight obj_box below keeps the object dominant in the ROI.
        # Object box (the drawn bbox) within the rendered crop, in px.
        aw = act.xMaximum() - act.xMinimum()
        ah = act.yMaximum() - act.yMinimum()
        iw, ih = img.width(), img.height()
        obj_box = None
        if aw > 0 and ah > 0:
            ox0 = (rect.xMinimum() - act.xMinimum()) / aw * iw
            ox1 = (rect.xMaximum() - act.xMinimum()) / aw * iw
            # y flip: map y grows up, pixel y grows down.
            oy0 = (act.yMaximum() - rect.yMaximum()) / ah * ih
            oy1 = (act.yMaximum() - rect.yMinimum()) / ah * ih
            obj_box = [
                max(0.0, ox0), max(0.0, oy0),
                min(float(iw), ox1), min(float(ih), oy1),
            ]
        return img, obj_box

    def _prebuild_exemplar_stamp(self, eid: str) -> None:
        """Render + cache one exemplar's stamp right after it is drawn, so the
        Detect press does not block on a per-exemplar basemap render (plan 11
        §1.3). The user just finished drawing, so the canvas already shows these
        pixels; one synchronous render is acceptable here (no new threads).
        Best-effort: on any failure the entry stays uncached and
        _build_exemplar_stamps rebuilds it at Detect."""
        ex = self._auto_exemplar_store.get(eid)
        if ex is None:
            return
        layer = self._get_active_raster_layer()
        if layer is None:
            return
        # Render at the run scale the CURRENT zone + detail imply; if the user
        # then moves the detail slider before Detect, the gsd-aware cache check
        # in _build_exemplar_stamps rebuilds the stamp at the final scale.
        run_mupp = self._exemplar_run_mupp(layer)
        built = self._render_exemplar_stamp(ex, layer, run_mupp)
        if built is None:
            return
        ex.stamp_img, ex.stamp_obj_box = built
        ex.stamp_layer_id = layer.id()
        ex.stamp_gsd = run_mupp
        # The card/preview must show the EXACT reference the model receives, so
        # swap the provisional canvas grab for the rendered stamp. The canvas
        # grab embeds live overlays (zone grid lines, draw bands) and gets
        # truncated when the drawn shape spills past the visible canvas; the
        # layer render has neither problem.
        if ex.stamp_img is not None and not ex.stamp_img.isNull():
            ex.thumbnail = ex.stamp_img
            self._refresh_exemplar_chips()

    def _build_exemplar_stamps(self, layer):
        """Return ``[(QImage, label, obj_box)]`` ready to stamp into every tile,
        one per stored exemplar. Reuses the stamp prebuilt when the exemplar was
        drawn when it is present AND still matches this layer;
        otherwise renders it now. So only missing/stale entries pay the render on
        the Detect critical path (usually none). See _render_exemplar_stamp for
        the crop/mask/box rationale."""
        stamps = []
        layer_id = layer.id() if layer is not None else None
        # The run scale at Detect (the zone + committed detail level): a cached
        # stamp built at a different scale (detail moved since the draw) is
        # stale and re-rendered here so the reference always matches the tiles.
        run_mupp = self._exemplar_run_mupp(layer)
        for ex in self._auto_exemplar_store.list():
            if (ex.stamp_img is not None and ex.stamp_layer_id == layer_id
                    and self._stamp_gsd_matches(ex.stamp_gsd, run_mupp)):  # noqa: W503
                stamps.append((ex.stamp_img, int(ex.label), ex.stamp_obj_box))
                continue
            built = self._render_exemplar_stamp(ex, layer, run_mupp)
            if built is None:
                continue
            ex.stamp_img, ex.stamp_obj_box = built
            ex.stamp_layer_id = layer_id
            ex.stamp_gsd = run_mupp
            stamps.append((ex.stamp_img, int(ex.label), ex.stamp_obj_box))
        return stamps
