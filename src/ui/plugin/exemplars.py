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

from ...core.exemplar_size import exemplar_at_max_detail, exemplar_too_small
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
    # Visual exemplars ("draw one example, find all")
    # ------------------------------------------------------------------

    def _on_add_exemplar_requested(self, label: int) -> None:
        """Arm the example-box draw tool (label 1 positive / 0 exclude).

        The user drags a box around ONE object (green for a positive example /
        red for an exclude). On finish it is stored as that rectangle's polygon
        plus its bounding box; the worker sends the cloud model a box tight to
        the object. No-op without a zone, once the store is full, or while a
        run/review is active.
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
        from ..box_draw_maptool import BoxDrawMapTool
        canvas = self.iface.mapCanvas()
        # Remember the tool to restore after the one-shot draw (usually the zone tool).
        self._maptool_before_exemplar = canvas.mapTool()
        self._pending_exemplar_label = int(label)
        color = EXEMPLAR_STROKE if label == 1 else EXCLUDE_STROKE
        tool = BoxDrawMapTool(canvas, color)
        tool.box_selected.connect(self._on_exemplar_box_drawn)
        tool.cancelled.connect(self._on_exemplar_cancelled)
        # The user picking another QGIS tool mid-draw deactivates ours; reset
        # the armed UI without fighting their new tool choice.
        try:
            tool.tool_deactivated.connect(self._on_exemplar_tool_deactivated)
        except (RuntimeError, AttributeError):
            pass
        self._exemplar_maptool = tool
        # The draw tool owns Escape (cancel) while armed; disable the dock
        # Escape/Enter shortcuts so they cannot exit the flow mid-draw.
        try:
            self.dock_widget.set_auto_shortcuts_enabled(False)
        except (RuntimeError, AttributeError):
            pass
        self._connect_exemplar_undo_shortcut()
        # setMapTool deactivates the outgoing tool; if that happens to be a
        # leftover draw tool whose signal is still live, ignore its
        # deactivation so it cannot null the tool we just set.
        self._suspend_exemplar_deactivate = True
        try:
            canvas.setMapTool(tool)
        finally:
            self._suspend_exemplar_deactivate = False
        # Light up the chosen button + show the "now outline on the map"
        # instruction so the click clearly started a draw action. Tracked on
        # the plugin too (not just the dock label): the live size-warning
        # refresh (_refresh_exemplar_size_warning) must no-op while armed, or
        # it would fight the armed instruction for the shared hint label.
        self._auto_exemplar_arming = True
        try:
            self.dock_widget.set_auto_exemplar_armed(int(label))
        except (RuntimeError, AttributeError):
            pass

    def _on_exemplar_box_drawn(self, rect) -> None:
        """Adapt the drag-box tool's rectangle to the stored polygon contract.

        The example/exclude gesture is a drag rectangle now, but everything
        downstream (crop mask, pixel box, stamps, chips, telemetry) still works
        on a polygon geometry. Build the rectangle polygon and hand it to the
        unchanged storage path. rect is a QgsRectangle in canvas CRS."""
        self._on_exemplar_polygon_drawn(QgsGeometry.fromRect(rect))

    def _on_exemplar_polygon_drawn(self, geom) -> None:
        """Store a drawn example polygon (clipped to the zone), mark it on the
        map, refresh the dock. geom is a polygon in canvas CRS.

        An example drawn ENTIRELY outside the zone is refused with a notification:
        there is no imagery to learn from there and it cannot match anything
        inside. A draw straddling the border keeps its WHOLE shape when its
        centroid is inside the zone.
        """
        label = int(getattr(self, "_pending_exemplar_label", 1))
        # The zone only scopes the SEARCH area; the reference must show the
        # WHOLE drawn object (the stamp renders from the LAYER, so imagery
        # exists past the zone edge). A straddling draw whose centroid is
        # inside the zone is kept UNCLIPPED; a mostly-outside straddler is
        # still clipped to its in-zone part; only a draw entirely outside the
        # zone is refused (nothing there can match inside).
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
                if not self._geom_centroid_in(geom, zone_poly):
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
                telemetry.track_exemplar_added(
                    count_after=self._auto_exemplar_store.count(),
                    label="include" if label == 1 else "exclude")
            except Exception:
                pass  # nosec B110
        self._restore_maptool_after_exemplar()
        # After the tool restore (which clears the armed instruction line and
        # flag): advise, never block, when ANY stored example is too small at
        # the current detail level to carry usable texture. Re-checks the
        # WHOLE store (not just this one draw) and the caller re-runs this on
        # every detail change too, so the warning stays live instead of
        # sticking once the object is later drawn big enough.
        self._refresh_exemplar_size_warning()
        # A new example changes the run signature, so the identical-re-run note
        # must clear if it was showing.
        self._refresh_rerun_guard()

    @staticmethod
    def _geom_centroid_in(geom, zone_poly) -> bool:
        """True when geom's centroid falls inside zone_poly. Any failure counts
        as inside (lenient: the caller already proved the draw overlaps the
        zone, so the whole shape is kept on doubt)."""
        try:
            c = geom.centroid()
            if c is None or c.isEmpty():
                return True
            return bool(zone_poly.contains(c))
        except (RuntimeError, AttributeError):
            return True

    def _refresh_exemplar_size_warning(self) -> None:
        """Re-check EVERY stored example against the CURRENT detail level and
        show/clear the shared too-small warning accordingly (below the
        stamp's hard floor the crop carries no usable texture).

        This replaces a former one-shot check computed only at draw time: it
        never re-evaluated, so raising the Detail slider (making the object
        bigger in pixels) left a stale warning on screen even once the object
        was plenty big. This method is the single re-check chokepoint, called
        after every detail change (auto_zone._update_credit_estimate) and
        after every exemplar add/remove, so the warning tracks the CURRENT
        state instead of the state at draw time.

        No-op while a draw is armed (the shared hint label is busy with the
        armed instruction; the plugin-side ``_auto_exemplar_arming`` flag
        avoids fighting it). Advisory only: the run is never blocked. Fully
        guarded: a hiccup here (layer removed, reproject failure, unknown
        resolution) must never raise into a caller that has its own job to
        finish (the credit estimate, an add, a remove).
        """
        if not self.dock_widget or getattr(self, "_auto_exemplar_arming", False):
            return
        try:
            store = self._auto_exemplar_store
        except AttributeError:
            return
        if store.count() == 0:
            try:
                self.dock_widget.clear_auto_exemplar_size_warning()
            except (RuntimeError, AttributeError):
                pass
            return
        layer = self._get_active_raster_layer()
        if layer is None:
            return
        try:
            run_mupp = self._exemplar_run_mupp(layer)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return
        if run_mupp <= 0:
            return
        from ...core.detection_policy import exemplar_render_abs_min_side_px
        abs_min_side = exemplar_render_abs_min_side_px(self._STAMP_ABS_MIN_SIDE)
        any_valid = False
        too_small = False
        for ex in store.list():
            try:
                rect = self._reproject_zone_to_layer_crs(ex.map_rect, layer)
            except (RuntimeError, AttributeError):
                continue
            any_valid = True
            if exemplar_too_small(
                    rect.width(), rect.height(), run_mupp, abs_min_side):
                too_small = True
                break  # one small example is enough to warn
        try:
            if any_valid and too_small:
                # The message must not tell the user to "zoom finer" when the
                # Detail slider is already at its useful maximum for this
                # layer/zone: set_auto_detail_max already clamped the slider's
                # maximum() to that cap (auto_zone._update_credit_estimate),
                # so reading it here is free (no grid recompute).
                at_max_detail = self._auto_detail_slider_at_max()
                self.dock_widget.show_auto_exemplar_size_warning(at_max_detail=at_max_detail)
            else:
                self.dock_widget.clear_auto_exemplar_size_warning()
        except (RuntimeError, AttributeError):
            pass

    def _auto_detail_slider_at_max(self) -> bool:
        """True when the Detail slider is already at its useful maximum, read
        straight off the slider's current maximum() (kept in sync with
        `_max_useful_detail` by `set_auto_detail_max`) rather than recomputing
        the grid. Fails open to False (the conservative "can zoom finer"
        wording) on any widget hiccup."""
        try:
            slider = self.dock_widget.auto_detail_slider
            return exemplar_at_max_detail(int(slider.value()), int(slider.maximum()))
        except (RuntimeError, AttributeError):
            return False

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
        # The removed example may have been the one carrying the too-small
        # warning; re-check the remaining set (clears it once none are small).
        self._refresh_exemplar_size_warning()
        try:
            from ...core import telemetry
            telemetry.track_exemplar_removed(count_after=self._auto_exemplar_store.count())
        except Exception:
            pass  # nosec B110
        # The example count dropped, so re-evaluate the identical-re-run note.
        self._refresh_rerun_guard()

    def _clear_exemplars(self) -> None:
        """Drop all exemplars + their rubber bands (zone redraw / exit / reset)."""
        self._auto_exemplar_store.clear()
        for band in self._exemplar_bands.values():
            self._remove_rubber_band(band)
        self._exemplar_bands.clear()
        # Disarm a still-active example draw tool.
        self._restore_maptool_after_exemplar()
        self._refresh_exemplar_chips()

    def _set_exemplar_bands_visible(self, visible: bool) -> None:
        """Show/hide every exemplar rubber band WITHOUT dropping the exemplars.
        The Refine-in-Manual handoff hides them (the green example outline reads
        as one more editable polygon there); Back to review restores them."""
        for band in self._exemplar_bands.values():
            if band is None:
                continue
            try:
                band.setVisible(visible)
            except (RuntimeError, AttributeError):
                pass

    # -- Ctrl/Cmd+Z while the example box is being drawn --------------------
    # The dock binds the platform Undo at WINDOW level (for the review's
    # Correct step), so it matches the key before the canvas ever sees the
    # press and the draw could never undo from the keyboard. The zone draw was
    # fixed by routing that shortcut to the armed tool; the example and exclude
    # box now do the same, so Undo behaves identically in both draws. Scoped to
    # the armed draw: connected on arm, dropped on disarm, and the handler
    # re-checks that our tool is the live one, so nothing else is shadowed.

    def _connect_exemplar_undo_shortcut(self) -> None:
        """Listen to the dock's Undo shortcut for as long as the draw is armed."""
        shortcut = getattr(self.dock_widget, "auto_correct_undo_shortcut", None)
        if shortcut is None or getattr(self, "_exemplar_undo_connected", False):
            return
        try:
            shortcut.activated.connect(self._on_exemplar_undo_shortcut)
        except (RuntimeError, AttributeError, TypeError):
            return
        self._exemplar_undo_connected = True

    def _disconnect_exemplar_undo_shortcut(self) -> None:
        """Stop listening once the draw is over. Idempotent."""
        if not getattr(self, "_exemplar_undo_connected", False):
            return
        self._exemplar_undo_connected = False
        shortcut = getattr(self.dock_widget, "auto_correct_undo_shortcut", None)
        if shortcut is None:
            return
        try:
            shortcut.activated.disconnect(self._on_exemplar_undo_shortcut)
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _on_exemplar_undo_shortcut(self) -> bool:
        """Platform Undo during an example/exclude draw: take back the
        rectangle being dragged, or (nothing under way) disarm the draw.

        Returns True when the draw acted on the press. The dock's own handler
        for this shortcut only covers the review's Correct step and its gate is
        down while a draw is armed, so exactly one of the two acts on a press.
        """
        tool = self._exemplar_maptool
        if tool is None:
            return False
        try:
            if self.iface.mapCanvas().mapTool() is not tool:
                return False
            if tool.undo_point():
                return True
        except (RuntimeError, AttributeError):
            return False
        self._restore_maptool_after_exemplar()
        return True

    def _restore_maptool_after_exemplar(self) -> None:
        """Restore the map tool active before a one-shot example draw."""
        tool = self._exemplar_maptool
        # Disarm before the dock clears the instruction line, so a
        # size-warning refresh triggered right after (draw-completion path)
        # is no longer blocked by the armed no-op guard.
        self._auto_exemplar_arming = False
        self._disconnect_exemplar_undo_shortcut()
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
        # We are switching the tool ourselves: silence its deactivate signal
        # (it would re-enter this reset) and free the C++ object afterwards.
        try:
            tool.suppress_deactivate_signal = True
        except (RuntimeError, AttributeError):
            pass
        try:
            canvas = self.iface.mapCanvas()
            prev = self._maptool_before_exemplar
            if prev is not None:
                canvas.setMapTool(prev)
            elif canvas.mapTool() == tool:
                canvas.unsetMapTool(tool)
        except (RuntimeError, AttributeError):
            pass
        try:
            tool.deleteLater()
        except (RuntimeError, AttributeError):
            pass
        self._maptool_before_exemplar = None

    def _on_exemplar_tool_deactivated(self) -> None:
        """The armed example-draw tool was dropped because the user picked
        another QGIS map tool. Reset the armed UI (shortcuts, lit button) but
        do NOT restore the previous tool: QGIS already switched to the user's
        new choice."""
        if getattr(self, "_suspend_exemplar_deactivate", False):
            return
        if self._exemplar_maptool is None:
            return
        self._exemplar_maptool = None
        self._maptool_before_exemplar = None
        self._auto_exemplar_arming = False
        self._disconnect_exemplar_undo_shortcut()
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_shortcuts_enabled(True)
                self.dock_widget.set_auto_exemplar_armed(None)
            except (RuntimeError, AttributeError):
                pass

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
            raw = self._exemplar_full_pixel_box(
                ex, layer, geo_bbox, pixel_w, pixel_h)
            if raw is None:
                continue
            box = [
                _clamp(raw[0], 0, pixel_w), _clamp(raw[1], 0, pixel_h),
                _clamp(raw[2], 0, pixel_w), _clamp(raw[3], 0, pixel_h),
            ]
            if box[2] - box[0] >= 1 and box[3] - box[1] >= 1:
                entry = {"box": [round(v, 1) for v in box], "label": int(ex.label)}
                ring = self._exemplar_polygon_px(
                    ex, layer, geo_bbox, pixel_w, pixel_h)
                if ring:
                    entry["polygon_px"] = ring
                payload.append(entry)
        return payload

    def _exemplar_full_pixel_box(
        self, ex, layer, geo_bbox: tuple, pixel_w: int, pixel_h: int
    ) -> list[float] | None:
        """One exemplar's drawn rect as an UNCLAMPED xyxy pixel box on the
        rendered zone grid (geo_bbox = actual rendered extent, layer CRS).
        Unclamped on purpose: a consumer testing whether a tile fully contains
        the box must see the true extent, so a draw overflowing the zone image
        can never pass as contained. None when the grid is degenerate or the
        reprojection fails."""
        ext_minx, ext_miny, ext_maxx, ext_maxy = geo_bbox
        ext_w = ext_maxx - ext_minx
        ext_h = ext_maxy - ext_miny
        if ext_w <= 0 or ext_h <= 0 or pixel_w <= 0 or pixel_h <= 0:
            return None
        try:
            rect_layer = self._reproject_zone_to_layer_crs(ex.map_rect, layer)
        except (RuntimeError, AttributeError):
            return None
        x0 = (rect_layer.xMinimum() - ext_minx) / ext_w * pixel_w
        x1 = (rect_layer.xMaximum() - ext_minx) / ext_w * pixel_w
        # y flip: map y grows up, pixel y grows down.
        y0 = (ext_maxy - rect_layer.yMaximum()) / ext_h * pixel_h
        y1 = (ext_maxy - rect_layer.yMinimum()) / ext_h * pixel_h
        return [x0, y0, x1, y1]

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
            if canvas_crs.isValid() and layer_crs.isValid() and canvas_crs != layer_crs:
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

    # Longest-side render bounds for an exemplar stamp, px. The stamp is pasted
    # into a 1008 tile corner, so keep it small enough not to occlude real
    # imagery, but large enough to stay crisp. These four are the client
    # fallbacks: the server may serve its own under the exemplar block, read at
    # each use site (detection_policy.exemplar_render_*).
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
        from ...core.detection_policy import (
            exemplar_render_abs_min_side_px,
            exemplar_render_side_bounds,
        )
        min_side, max_side = exemplar_render_side_bounds(
            self._STAMP_MIN_SIDE, self._STAMP_MAX_SIDE)
        gsd_m, longest_m, run_matched = self._exemplar_render_gsd(
            layer, padded, run_mupp)
        if gsd_m <= 0 or longest_m <= 0:
            return max_side
        side = int(round(longest_m / gsd_m))
        # When the run scale sized the stamp, do NOT inflate a small object
        # back to the ordinary floor: that would upsample-blur it AND
        # reintroduce the scale mismatch (crisp-and-run-scaled beats big).
        floor = exemplar_render_abs_min_side_px(
            self._STAMP_ABS_MIN_SIDE) if run_matched else min_side
        return max(floor, min(max_side, side))

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
        from ...core.detection_policy import exemplar_render_fallback_gsd_m
        fallback_gsd = exemplar_render_fallback_gsd_m(self._STAMP_FALLBACK_GSD_M)
        return fallback_gsd, max(padded.width(), padded.height()), False

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

    def _render_exemplar_stamp(self, ex, layer, run_mupp: float = 0.0, max_side: int = 0):
        """Render ONE exemplar's natural crop + object box from the LAYER at
        the RUN's tile resolution (floored at the source's true finest, see
        _exemplar_stamp_longest_side). Returns ``(img, obj_box)`` or None.

        ``max_side`` (> 0) clamps the render's longest side to the final per-tile
        paste size, so the crop is filtered ONCE here (the tiles get one
        filtering pass; rendering larger then Qt-downscaling in the worker would
        be a second pass on the same pixels).

        ``obj_box`` is ``[x0,y0,x1,y1]`` in crop pixels framing the drawn object,
        so the worker sends the cloud model a box tight to the object. The crop keeps the
        REAL surrounding pixels (natural context): masking the surroundings to
        flat grey produced an out-of-distribution "object on grey" patch that
        under-recalled in live user testing, so the tight box (not grey paint) is
        what mitigates neighbour leak. This is the expensive part (a
        QgsMapRendererParallelJob + nested event loop), pulled out so it can run
        at DRAW time (_prebuild_exemplar_stamp) or lazily at Detect."""
        from ...core.cloud_detection import render_zone_to_image
        built = self._exemplar_padded_extent(ex, layer, run_mupp)
        if built is None:
            return None
        rect, padded = built
        fw, fh = padded.width(), padded.height()
        if fw <= 0 or fh <= 0:
            return None
        # Size the render at the RUN's tile resolution (floored at the source's
        # true finest so it never upsamples): the reference must show the
        # object at the SAME apparent pixel size the tiles will, or the model
        # matches coarser context structures instead of the object.
        target = self._exemplar_stamp_longest_side(layer, padded, run_mupp)
        # Clamp to the final per-tile paste size so the crop is filtered once at
        # render time; the object box below still derives from the actual render
        # dimensions, so it stays exact under the clamp.
        if max_side > 0:
            target = min(target, int(max_side))
        if fw >= fh:
            pw, ph = target, max(16, int(round(target * fh / fw)))
        else:
            ph, pw = target, max(16, int(round(target * fw / fh)))
        try:
            # resample_local: a fine local raster shrunk to the run resolution
            # must be averaged (as the tiles are), not decimated nearest, or the
            # stamp's texture statistics differ from the tiles it must match.
            img, act = render_zone_to_image(layer, padded, pw, ph, resample_local=True)
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

    def _exemplar_padded_extent(self, ex, layer, run_mupp: float = 0.0):
        """The drawn rect + its context-padded render extent, both in the
        layer CRS, or None. The margin is the policy fraction of the drawn
        bbox, CAPPED in absolute run-scale pixels: a fractional pad on a large
        object drags whole neighbouring structures into the crop (distractors
        the model then matches), so the margin never exceeds a thin ring."""
        from ...core.detection_policy import exemplar_context_pad, exemplar_context_pad_px_cap
        try:
            rect = self._reproject_zone_to_layer_crs(ex.map_rect, layer)
        except (RuntimeError, AttributeError):
            return None
        ew, eh = rect.width(), rect.height()
        if ew <= 0 or eh <= 0:
            return None
        context_pad = exemplar_context_pad()
        pad_x = ew * context_pad
        pad_y = eh * context_pad
        if run_mupp > 0:
            pad_cap = exemplar_context_pad_px_cap() * run_mupp
            pad_x = min(pad_x, pad_cap)
            pad_y = min(pad_y, pad_cap)
        padded = QgsRectangle(
            rect.xMinimum() - pad_x, rect.yMinimum() - pad_y,
            rect.xMaximum() + pad_x, rect.yMaximum() + pad_y)
        return rect, padded

    def _exemplar_runscale_side(self, ex, layer, run_mupp: float) -> int:
        """The crop's long side (px) at the run's TRUE apparent scale (no
        band cap applied): the paste-vs-in-situ decision compares this against
        the band budget."""
        built = self._exemplar_padded_extent(ex, layer, run_mupp)
        if built is None:
            return 0
        _rect, padded = built
        return self._exemplar_stamp_longest_side(layer, padded, run_mupp)

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
        from ...core.cloud_detection import stamp_size_cap
        run_mupp = self._exemplar_run_mupp(layer)
        # Render straight at the final per-tile paste size for this example
        # count; a later count change re-caps and rebuilds in _build_exemplar_stamps.
        cap = stamp_size_cap(self._auto_exemplar_store.count())
        built = self._render_exemplar_stamp(ex, layer, run_mupp, max_side=cap)
        if built is None:
            return
        ex.stamp_img, ex.stamp_obj_box = built
        ex.stamp_layer_id = layer.id()
        ex.stamp_gsd = run_mupp
        ex.stamp_side = max(ex.stamp_img.width(), ex.stamp_img.height())
        # The card/preview must show the EXACT reference the model receives, so
        # swap the provisional canvas grab for the rendered stamp. The canvas
        # grab embeds live overlays (zone grid lines, draw bands) and gets
        # truncated when the drawn shape spills past the visible canvas; the
        # layer render has neither problem.
        if ex.stamp_img is not None and not ex.stamp_img.isNull():
            ex.thumbnail = ex.stamp_img
            self._refresh_exemplar_chips()

    def _build_exemplar_stamps(self, layer, geo_bbox=None, pixel_w=0, pixel_h=0,
                               has_prompt: bool = False):
        """Return ``[(QImage, label, obj_box, full_box)]`` ready to stamp into
        every tile, one per stored exemplar. Reuses the stamp prebuilt when the
        exemplar was drawn when it is present AND still matches this layer;
        otherwise renders it now. So only missing/stale entries pay the render on
        the Detect critical path (usually none). See _render_exemplar_stamp for
        the crop/mask/box rationale.

        ``full_box`` is the exemplar's drawn box in FULL-image pixel coords
        (unclamped; None when the zone grid is not supplied or the box cannot
        be mapped): the worker uses it to send the box at the REAL in-situ
        object on the one tile that fully contains it, instead of pasting the
        crop there.

        Paste-vs-in-situ decision (should_paste_stamp): with a text prompt, an
        example whose run-scale crop would have to shrink below the policy
        fraction of true scale to fit the band is NOT pasted (the model
        matches by apparent scale, so a smaller copy points it at the wrong
        size). Its entry ships with a None image so the worker still sends it
        in-situ on tiles that fully contain the real object; without any
        usable full_box the exemplar is dropped for this run."""
        from ...core.cloud_detection import should_paste_stamp, stamp_size_cap
        from ...core.detection_policy import exemplar_min_paste_scale
        stamps = []
        layer_id = layer.id() if layer is not None else None
        # The run scale at Detect (the zone + committed detail level): a cached
        # stamp built at a different scale (detail moved since the draw) is
        # stale and re-rendered here so the reference always matches the tiles.
        run_mupp = self._exemplar_run_mupp(layer)
        # Final per-tile paste size for this example count. Render directly at it
        # so the worker's downscale is a no-op; a cached stamp rendered wider
        # than the current cap (the count grew since it was built) is rebuilt.
        cap = stamp_size_cap(self._auto_exemplar_store.count())
        min_scale = exemplar_min_paste_scale()
        skipped_paste = 0
        for ex in self._auto_exemplar_store.list():
            full_box = (
                self._exemplar_full_pixel_box(ex, layer, geo_bbox, pixel_w, pixel_h)
                if geo_bbox is not None else None)
            if getattr(ex, "region", False):
                # Region marker (correction box): no crop to render or paste;
                # the worker clips the box to every tile it overlaps. Without a
                # mappable box it has nothing to contribute.
                if full_box is not None:
                    stamps.append((None, int(ex.label), None, full_box, True))
                continue
            true_side = self._exemplar_runscale_side(ex, layer, run_mupp)
            if not should_paste_stamp(true_side, cap, has_prompt, min_scale):
                if full_box is not None:
                    stamps.append((None, int(ex.label), None, full_box))
                skipped_paste += 1
                continue
            stamp_valid = ex.stamp_img is not None and ex.stamp_layer_id == layer_id
            stamp_valid = stamp_valid and self._stamp_gsd_matches(ex.stamp_gsd, run_mupp)
            stamp_valid = stamp_valid and getattr(ex, "stamp_side", 0) <= cap
            if stamp_valid:
                stamps.append(
                    (ex.stamp_img, int(ex.label), ex.stamp_obj_box, full_box))
                continue
            built = self._render_exemplar_stamp(ex, layer, run_mupp, max_side=cap)
            if built is None:
                continue
            ex.stamp_img, ex.stamp_obj_box = built
            ex.stamp_layer_id = layer_id
            ex.stamp_gsd = run_mupp
            ex.stamp_side = max(ex.stamp_img.width(), ex.stamp_img.height())
            stamps.append(
                (ex.stamp_img, int(ex.label), ex.stamp_obj_box, full_box))
        if skipped_paste:
            from qgis.core import Qgis, QgsMessageLog
            QgsMessageLog.logMessage(
                f"Auto detection: {skipped_paste} example(s) larger than the paste band at "
                "run scale; sent in-situ only",
                "AI Segmentation", level=Qgis.MessageLevel.Info,
            )
        return stamps
