"""Per-shape "Shape settings" for the Refine-in-Manual handoff.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out so
agents and humans work on one concern per file. Methods are plain mixin
members: state lives on the plugin instance.

The panel only appears while a detection is OPEN for editing: it shows
that shape's stored settings, and a change re-shapes it from its pristine
geometry (never compounding, so scrubbing a spinbox is safe). Handoff
detections have no source mask, so shaping happens in geometry space
(core.polygon_exporter.shape_polygon_geometry).
"""
from __future__ import annotations


from qgis.core import QgsGeometry


class HandoffShapeMixin:
    """Seeds and applies the open shape's settings in a refine handoff."""

    def _handoff_ground_mupp(self) -> float:
        """Map units per pixel of the handoff raster (its CRS matches the
        stored entry geometries). Falls back to the canvas resolution, then 1."""
        layer = getattr(self, "_handoff_source_layer", None)
        try:
            if layer is not None:
                v = float(layer.rasterUnitsPerPixelX())
                if v > 0:
                    return v
        except (RuntimeError, AttributeError):
            pass
        try:
            v = float(self.iface.mapCanvas().mapUnitsPerPixel())
            if v > 0:
                return v
        except (RuntimeError, AttributeError):
            pass
        return 1.0

    def _current_refine_tuple(self) -> tuple:
        """The live panel settings as a comparable tuple."""
        return (
            int(self._refine_simplify),
            self._refine_smooth > 0,
            int(self._refine_expand),
            bool(self._refine_fill_holes),
            bool(self._refine_ortho),
            float(getattr(self, "_refine_min_size_m2", 0.0) or 0.0),
            float(getattr(self, "_refine_max_size_m2", 0.0) or 0.0),
        )

    @staticmethod
    def _entry_refine_tuple(entry: dict) -> tuple:
        """An entry's stored settings as the same comparable tuple."""
        return (
            int(entry.get("refine_simplify") or 0),
            (entry.get("refine_smooth") or 0) > 0,
            int(entry.get("refine_expand") or 0),
            bool(entry.get("refine_fill_holes")),
            bool(entry.get("refine_ortho")),
            float(entry.get("refine_min_size_m2") or 0.0),
            float(entry.get("refine_max_size_m2") or 0.0),
        )

    def _seed_refine_panel_from_entry(self, entry: dict) -> None:
        """Load an entry's stored shape settings into the panel (no signals)
        and sync the plugin globals, so a later save records exactly what the
        panel shows. Called when an edit session opens."""
        self._refine_simplify = int(entry.get("refine_simplify") or 0)
        self._refine_smooth = int(entry.get("refine_smooth") or 0)
        self._refine_expand = int(entry.get("refine_expand") or 0)
        self._refine_fill_holes = bool(entry.get("refine_fill_holes"))
        self._refine_ortho = bool(entry.get("refine_ortho"))
        self._refine_min_size_m2 = float(entry.get("refine_min_size_m2") or 0.0)
        self._refine_max_size_m2 = float(entry.get("refine_max_size_m2") or 0.0)
        if self.dock_widget:
            try:
                self.dock_widget.set_refine_values(
                    self._refine_simplify, self._refine_smooth,
                    self._refine_expand, self._refine_fill_holes,
                    right_angles=self._refine_ortho)
                self.dock_widget.set_size_filter_values(
                    self._refine_min_size_m2, self._refine_max_size_m2)
            except (RuntimeError, AttributeError):
                pass

    def _apply_handoff_refine_settings(self) -> bool:
        """Route a refine-panel change inside a handoff. Returns True when the
        handoff consumed it (base Manual must not also repaint a mask). The
        panel only shows while an edit is open, so with no open edit there is
        nothing to apply. An edit with editing clicks is a live Manual mask
        session: base Manual's own mask repaint applies the settings there."""
        if not (self._refine_handoff_active or self._is_refining_saved_object):
            return False
        if self.current_mask is not None:
            return False
        if self._refine_edit_session_active():
            self._reshape_open_edit()
        return True

    def _reshape_open_edit(self) -> None:
        """Re-shape the OPEN object from its pristine geometry with the live
        settings. The previous display state goes on the delta history, so
        Ctrl+Z steps a settings change back like any editing click."""
        from ...core.polygon_exporter import shape_polygon_geometry
        base = self._unfrozen_display_polygon
        if base is None or base.isEmpty():
            return
        cur = self._current_refine_tuple()
        if cur == getattr(self, "_refine_edit_last_applied", None):
            return
        pristine = getattr(self, "_refine_edit_pristine", None)
        if pristine is None or pristine.isEmpty():
            pristine = QgsGeometry(base)
            self._refine_edit_pristine = pristine
        info = self._current_crop_info
        if info is not None:
            minx, miny, maxx, maxy = info["bounds"]
            w = max(info["img_shape"][1], 1)
            mupp = (maxx - minx) / w
        else:
            mupp = self._handoff_ground_mupp()
        shaped = shape_polygon_geometry(
            pristine, mupp,
            simplify_px=cur[0], smooth=cur[1], expand_px=cur[2],
            fill_holes=cur[3], ortho=cur[4])
        if shaped is None or shaped.isEmpty():
            return
        # User Min/Max size window (ground m2): drop out-of-window parts, but
        # never blank the whole open object on an over-aggressive filter.
        filtered = self._filter_geometry_parts_by_size(shaped)
        if filtered is not None and not filtered.isEmpty():
            shaped = filtered
        history = getattr(self, "_refine_geom_history", None)
        if history is None:
            history = []
            self._refine_geom_history = history
        history.append(base)
        del history[:-30]
        self._refine_edit_last_applied = cur
        self._unfrozen_display_polygon = shaped
        self._update_mask_visualization()
