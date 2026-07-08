




from __future__ import annotations

from qgis.PyQt.QtGui import QColor

from ..canvas_palette import (
    KEPT_FILL,
    KEPT_STROKE,
    OUTLINE_MODE_STROKE_STR,
    PENDING_FILL,
    PENDING_STROKE,
)
from .auto_review_display import _random_mode_style


class ManualDisplayMixin:


    def _manual_distinct_colors(self, entry=None) -> tuple[QColor, QColor]:






        palette = getattr(self, "_manual_distinct_palette", None)
        if palette is None:
            buckets, hue_step, saturation, lightness = _random_mode_style()
            palette = []
            for bucket in range(buckets):
                color = QColor.fromHslF(
                    int(bucket * hue_step) / 360.0, saturation, lightness)

                color.setAlpha(154)
                palette.append(color)
            self._manual_distinct_palette = palette
        origin = (entry if entry is not None else
                  getattr(self, "_active_refine_origin_entry", None)) or {}
        identity = origin.get("det_id")
        if identity is None:
            identity = getattr(self, "_handoff_det_id_seq", None) or 100000
        try:
            bucket = (abs(int(identity)) * 67) % len(palette)
        except (TypeError, ValueError, OverflowError):
            bucket = 0
        return palette[bucket], QColor(20, 20, 20, 200)

    def _apply_manual_band_display_style(self, band, entry=None) -> None:

        if band is None or getattr(self, "_refine_handoff_active", False):
            return
        mode = getattr(self, "_manual_display_mode", "normal")
        if mode == "outline":
            fill = QColor(PENDING_FILL)
            fill.setAlpha(0)
            stroke = QColor(*map(int, OUTLINE_MODE_STROKE_STR.split(",")))
        elif mode == "random":
            fill, stroke = self._manual_distinct_colors(entry)
        elif entry is not None and entry.get("validated", False):
            fill, stroke = KEPT_FILL, KEPT_STROKE
        else:
            fill, stroke = PENDING_FILL, PENDING_STROKE
        try:
            band.setFillColor(fill)
            band.setStrokeColor(stroke)


            band.update()
        except (RuntimeError, AttributeError):
            pass

    def _on_manual_display_mode_changed(self, mode: str) -> None:

        self._manual_display_mode = (
            mode if mode in ("normal", "outline", "random") else "normal")
        for entry, band in zip(self.saved_polygons, self.saved_rubber_bands):
            self._apply_manual_band_display_style(band, entry)
        self._apply_mask_band_style()
        try:
            from ...core.telemetry_run_events import track_review_display_mode

            track_review_display_mode(mode=self._manual_display_mode)
        except Exception:
            pass  # nosec B110
