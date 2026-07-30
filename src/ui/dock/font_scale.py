












from __future__ import annotations

import re




_MAX_FONT_SCALE = 2.0


_FONT_PX_PATTERN = re.compile(r"font-size:\s*(\d+)px")



_ALREADY_SCALED_MARK = "/* fs */"

_cached_font_scale: float | None = None


def ui_font_scale() -> float:







    global _cached_font_scale
    if _cached_font_scale is not None:
        return _cached_font_scale
    _cached_font_scale = _measure_font_scale()
    return _cached_font_scale


def _measure_font_scale() -> float:

    try:
        from qgis.PyQt.QtGui import QFontDatabase

        from ...core.qt_compat import resolve_qt_enum

        general = resolve_qt_enum(QFontDatabase, "SystemFont", "GeneralFont")
        os_points = float(QFontDatabase.systemFont(general).pointSizeF())
        chosen_points = _qgis_chosen_font_points()
        if os_points <= 0.0 or chosen_points <= 0.0:
            return 1.0
        return max(1.0, min(_MAX_FONT_SCALE, chosen_points / os_points))
    except Exception:  # noqa: BLE001
        return 1.0


def _qgis_chosen_font_points() -> float:









    try:
        from qgis.core import QgsSettings

        points = QgsSettings().value("qgis/stylesheet/fontPointSize", 0.0, type=float)
        if points and float(points) > 0.0:
            return float(points)
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    try:
        from qgis.PyQt.QtWidgets import QApplication

        return float(QApplication.font().pointSizeF())
    except Exception:  # noqa: BLE001
        return 0.0


def widget_pixel_ratio(widget) -> float:








    try:
        screen = widget.screen()
        if screen is not None:
            ratio = float(screen.devicePixelRatio())
            if ratio > 0.0:
                return ratio
    except (AttributeError, RuntimeError):
        pass
    try:
        ratio = float(widget.devicePixelRatioF())
    except (AttributeError, RuntimeError):
        return 1.0
    return ratio if ratio > 0.0 else 1.0


def scale_point_size(points: int) -> int:





    return _rounded_up_from_half(points * ui_font_scale())


def scale_qss_font_px(qss: str) -> str:






    if not qss or _ALREADY_SCALED_MARK in qss:
        return qss
    scale = ui_font_scale()
    if scale <= 1.0:
        return qss

    def _grow(match: re.Match) -> str:
        return f"font-size: {_rounded_up_from_half(int(match.group(1)) * scale)}px"

    grown, count = _FONT_PX_PATTERN.subn(_grow, qss)
    return grown + _ALREADY_SCALED_MARK if count else qss


def _rounded_up_from_half(value: float) -> int:








    return max(1, int(value + 0.5))


def scale_px_length(pixels: int) -> int:






    return _rounded_up_from_half(pixels * ui_font_scale())


def apply_font_scale_to_tree(root) -> None:







    if ui_font_scale() <= 1.0:
        return
    try:
        from qgis.PyQt.QtWidgets import QWidget

        widgets = [root] + root.findChildren(QWidget)
    except Exception:  # noqa: BLE001
        return
    for widget in widgets:
        try:
            sheet = widget.styleSheet()
            if not sheet or "font-size" not in sheet:
                continue
            grown = scale_qss_font_px(sheet)
            if grown != sheet:
                widget.setStyleSheet(grown)
        except RuntimeError:
            continue


def fit_spin_width(spin, floor_px: int, cap_px: int) -> None:








    try:
        needed = int(spin.minimumSizeHint().width())
    except (AttributeError, RuntimeError, TypeError):
        needed = 0
    spin.setMinimumWidth(max(scale_px_length(floor_px), needed))
    spin.setMaximumWidth(max(scale_px_length(cap_px), needed))
