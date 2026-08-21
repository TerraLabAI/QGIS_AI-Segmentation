"""Save-time footprint alignment for Semi-Auto (internal mode "interactive").

When the user saves an object with the Right angles refine toggle ON, the
committed shape runs the same candidate competition as the Automatic finalize
pass (core.footprint_alignment.align_saved_footprint), with the session's
already saved shapes standing in as the neighbourhood. So the 2nd, 3rd, Nth
building of a street snaps to the grid the earlier saves agree on, instead of
each one squaring to its own angle a few degrees apart.

Gating, in order:
- The user's Right angles toggle is the intent signal (Semi-Auto has no shape
  class), so the pass only ever runs with the toggle ON.
- The served ``auto_regularize.enabled`` flag, fail-closed, the SAME switch
  that gates the Automatic pass. The served ``classes`` list does not gate
  here, because no class exists in this mode. A cold cache (offline Manual
  included) keeps the pass off: the guards travel with the flag.
- The give-up guards inside the alignment revert any snap that strays from
  the shape on screen, so the saved shape never differs visibly from the one
  the user approved.

Runs entirely client-side at save; no network call is made.
"""
from __future__ import annotations

from qgis.core import Qgis, QgsCoordinateReferenceSystem, QgsMessageLog


def align_manual_saved_shape(controller, combined):
    """The save-time aligned form of ``combined``, or ``combined`` unchanged
    when the pass is gated off, cannot run, or reverted. Never raises."""
    try:
        if combined is None or combined.isEmpty():
            return combined
        if not getattr(controller, "_refine_ortho", False):
            return combined
        if getattr(controller, "_is_non_georeferenced_mode", False):
            # A pixel grid has no ground metres to measure the dials in.
            return combined
        from ...core.detection_policy import manual_save_alignment_settings
        settings = manual_save_alignment_settings()
        if settings is None:
            return combined
        # A multi-part save keeps every part the user traced: the alignment
        # rebuilds ONE outline, so it only runs on a single-polygon shape.
        if combined.isMultipart() and len(combined.asMultiPolygon() or []) != 1:
            return combined
        crs = _manual_session_crs(controller)
        if crs is None:
            return combined
        from ...core.footprint_alignment import align_saved_footprint
        from ...core.layer_conventions import make_area_measurer
        aligned = align_saved_footprint(
            combined,
            _saved_neighbour_geoms(controller),
            settings,
            _ground_pixel_m(controller, combined),
            crs,
            make_area_measurer(crs))
        if aligned is None or aligned.isEmpty():
            return combined
        if not getattr(controller, "_save_alignment_logged", False):
            controller._save_alignment_logged = True
            QgsMessageLog.logMessage(
                "Semi-Auto save: footprints align to the session grid",
                "AI Segmentation", level=Qgis.MessageLevel.Info)
        return aligned
    except Exception:  # noqa: BLE001 -- alignment must never break a save
        return combined


def _manual_session_crs(controller) -> QgsCoordinateReferenceSystem | None:
    """The CRS the shape being saved lives in, or None when nothing usable
    answers, which fails the alignment closed.

    The LIVE transform info answers first: the saved-polygons definition is
    the FIRST save's CRS, and a session that switched rasters mid-way would
    otherwise measure this save's metres in the old grid."""
    try:
        for definition in (_live_session_crs_definition(controller),
                           controller._manual_saved_crs_definition()):
            if definition:
                crs = QgsCoordinateReferenceSystem(definition)
                if crs.isValid():
                    return crs
        layer = getattr(controller, "_current_layer", None)
        if layer is not None and layer.crs().isValid():
            return layer.crs()
    except Exception:  # noqa: BLE001 -- no CRS means no alignment
        return None
    return None


def _live_session_crs_definition(controller) -> str:
    """The CRS string of the crop under the shape being saved, or empty."""
    info = getattr(controller, "current_transform_info", None) or {}
    value = info.get("crs")
    return value.strip() if isinstance(value, str) else ""


def _ground_pixel_m(controller, combined) -> float:
    """Ground size of one crop pixel in metres, or 0.0 (the alignment then
    falls back to its served minimum simplify tolerance)."""
    try:
        info = getattr(controller, "current_transform_info", None)
        if not info:
            return 0.0
        px_units = controller._crop_pixel_size_units(info)
        if px_units <= 0:
            return 0.0
        centre = combined.boundingBox().center()
        factor = controller._manual_metres_per_unit(centre.x(), centre.y())
        if factor is None or factor <= 0:
            return 0.0
        return px_units * factor
    except Exception:  # noqa: BLE001 -- an unknown pixel is not a blocker
        return 0.0


def _saved_neighbour_geoms(controller) -> list:
    """The session's already saved shapes (handoff seeds included), as shared
    QgsGeometry references. The alignment reads coordinates only and never
    mutates them; the radius filter and the nearest-neighbour cap are its
    job, not this one's. An entry recorded in another CRS is skipped: mixing
    frames inside one distance computation gives nonsense metres."""
    session_crs = _live_session_crs_definition(controller)
    out = []
    for entry in getattr(controller, "saved_polygons", None) or []:
        geom = None
        try:
            entry_crs = (entry.get("transform_info") or {}).get("crs")
            if (session_crs and isinstance(entry_crs, str)
                    and entry_crs.strip() and entry_crs.strip() != session_crs):
                continue
            geom = controller._entry_geom(entry)
        except Exception:  # noqa: BLE001 -- one bad entry costs only itself
            geom = None
        if geom is not None and not geom.isEmpty():
            out.append(geom)
    return out
