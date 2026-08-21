"""One finished outline per (mask, settings), shared by the preview and Export.

A Semi-Auto click leaves a mask in memory. Everything the user does next reads
that same mask: every refine control repaints the preview, Save keeps the
outline, Export writes it to a layer. Each of those used to clean the mask
again, polygonize it again and run the whole geometry tail again, so pressing
Export paid a second time for an outline that was already on screen.

Two memos, both keyed on the IDENTITY of the mask and of the crop, so a new
click or a new crop can never be handed an old shape:

- the mask stage (fill holes, grow/shrink, minimum region, polygonize), which
  only five of the ten refine controls touch;
- the finished outline (trim spikes, simplify, point budget, right angles,
  round corners, then the Min/Max size window).

Nothing here decides geometry. It calls the same methods the preview always
called, in the same order, and remembers what came back. Delete the memos and
the shapes are identical, only slower.
"""
from __future__ import annotations

from qgis.core import QgsGeometry


class ManualShapeCacheMixin:
    """The Semi-Auto session's computed outline, kept until its inputs move."""

    def _manual_shape_cache_reset(self) -> None:
        """Forget both memos. Cheap insurance, not correctness: every read
        checks its own key, so a stale entry is never served."""
        self._mask_preview_memo = None
        self._manual_outline_memo = None

    def _manual_mask_polygons(self, fill_holes, max_hole_px, simplify_tol,
                              mask=None, info=None):
        """``(cleaned_mask, geometries)`` for one mask, the active one when
        ``mask`` and ``info`` are left out.

        The heavy half of a repaint: scipy hole filling and region labelling,
        then a rasterio polygonize of the whole crop. Six of the ten refine
        controls change none of its inputs, so on those it is served from the
        memo and costs nothing. Only the click session's own mask is memoized:
        the hover ghost hands its answer in explicitly and must never evict
        what the session's repaints read.
        """
        from ...core.polygon_exporter import apply_mask_refinement, mask_to_polygons

        if mask is None:
            mask = self.current_mask
        if info is None:
            info = self.current_transform_info
        is_active = (mask is self.current_mask
                     and info is self.current_transform_info)
        key = (self._refine_expand, fill_holes, self._refine_min_area,
               max_hole_px, simplify_tol)
        if is_active:
            memo = getattr(self, "_mask_preview_memo", None)
            if (memo is not None and memo[0] is mask and memo[1] is info
                    and memo[2] == key):
                return memo[3], memo[4]

        cleaned = mask
        if fill_holes or self._refine_expand != 0 or self._refine_min_area > 0:
            cleaned = apply_mask_refinement(
                mask,
                expand_value=self._refine_expand,
                fill_holes=fill_holes,
                min_area=self._refine_min_area,
                max_hole_px=max_hole_px,
            )
        geometries = mask_to_polygons(cleaned, info, simplify_tol)
        if is_active:
            # Holding the mask and the crop keeps the identity checks above
            # honest: a live reference cannot have its id reused later.
            self._mask_preview_memo = (mask, info, key, cleaned, geometries)
        return cleaned, geometries

    def _manual_outline_for(self, mask, info):
        """The finished outline for ONE mask over ONE crop window, or None.

        The compute half of _manual_active_outline, with the mask and window
        as arguments so the hover ghost can shape the answer it holds with the
        exact chain a click would run: mask refinement, polygonize, the shared
        geometry tail, then the Min/Max size window. Reads the live refine
        settings and the served dials at call time; writes nothing back: no
        layer, no memo, no session state, no history.
        """
        if mask is None or info is None:
            return None
        from ...core.detection_policy import manual_simplify_multiple_of_px

        multiple = manual_simplify_multiple_of_px()
        tolerance = (multiple * self._crop_pixel_size_units(info)
                     if multiple > 0 else 0.0)
        fill_holes, max_hole_px = self._fill_holes_arguments(info)
        _cleaned, geometries = self._manual_mask_polygons(
            fill_holes, max_hole_px, tolerance, mask=mask, info=info)
        if not geometries:
            return None
        combined = QgsGeometry.unaryUnion(geometries)
        if combined is None or combined.isEmpty():
            return None
        combined = self._shape_active_geometry(combined, info)
        if combined is not None and not combined.isEmpty():
            combined = self._filter_geometry_parts_by_size(combined)
        if combined is None or combined.isEmpty():
            return None
        return combined

    def _manual_outline_key(self, mask_stage):
        """Every setting the finished outline depends on.

        The mask-stage settings are in here too, through ``mask_stage``. The
        memo one level down catches them for its own answer, but this one is
        read FIRST: without them, turning Fill holes off would hit here and
        hand back the outline that still has them filled.
        """
        return mask_stage + (
            bool(getattr(self, "_refine_ortho", False)),
            float(getattr(self, "_refine_clean", 0.0) or 0.0),
            float(getattr(self, "_refine_simplify", 0.0) or 0.0),
            int(getattr(self, "_refine_points_pct", 100) or 100),
            int(getattr(self, "_refine_smooth", 0) or 0),
            float(getattr(self, "_refine_min_size_m2", 0.0) or 0.0),
            float(getattr(self, "_refine_max_size_m2", 0.0) or 0.0),
            # Both gate the size window (see _filter_geometry_parts_by_size),
            # so a shape computed under one must never be served under the
            # other.
            bool(getattr(self, "_refine_handoff_active", False)),
            bool(getattr(self, "_active_refine_origin_entry", None)),
        )

    def _manual_active_outline(self):
        """The active mask as ONE finished geometry, or None.

        The single answer the preview draws, Save keeps and Export writes, so
        the polygon a user gets is always exactly the one they were shown.
        Returns a copy, so a caller may reshape it without touching the memo.
        """
        mask = self.current_mask
        info = self.current_transform_info
        if mask is None or info is None:
            return None

        # The geometry tail reads server dials (the point budget, the
        # regularizer, the rounding pass). A publish swaps the whole snapshot
        # object rather than editing it, so holding the one this shape was
        # computed under is all it takes to drop the shape when new values land
        # mid-session.
        from ...core.config_cache import get_config
        from ...core.detection_policy import manual_simplify_multiple_of_px

        # Mask-level simplify tolerance: a multiple of the crop pixel, OFF
        # unless the server tunes it.
        multiple = manual_simplify_multiple_of_px()
        tolerance = (multiple * self._crop_pixel_size_units(info)
                     if multiple > 0 else 0.0)
        fill_holes, max_hole_px = self._fill_holes_arguments()
        mask_stage = (self._refine_expand, fill_holes, self._refine_min_area,
                      max_hole_px, tolerance)

        config = get_config()
        key = self._manual_outline_key(mask_stage)
        memo = getattr(self, "_manual_outline_memo", None)
        if (memo is not None and memo[0] is mask and memo[1] is info
                and memo[2] is config and memo[3] == key):
            return QgsGeometry(memo[4]) if memo[4] is not None else None

        outline = self._manual_outline_for(mask, info)

        self._manual_outline_memo = (mask, info, config, key, outline)
        return QgsGeometry(outline) if outline is not None else None
