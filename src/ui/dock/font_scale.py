"""Make the panel match the screen and the text size the user set in QGIS.

Every font size in this plugin is written in the stylesheets as a pixel value,
which Qt applies literally. A user who raises QGIS's application font because
they cannot read 11px text sees the whole of QGIS grow and this panel stay
small, with no setting of its own to fix it.

The factor here is the ratio between the font QGIS is running with and the
font the operating system hands out by default, so a user who never touched
the setting gets exactly 1.0 and nothing changes anywhere. It only ever grows:
shrinking would push text under the fixed heights the panel is built from, and
nobody raises a font size to make it smaller.
"""
from __future__ import annotations

import re

# Widest sensible growth. Past twice the base size the panel is no longer laid
# out for the text, and a huge accessibility font is better served by QGIS's
# own zoom than by a panel that overflows its own columns.
_MAX_FONT_SCALE = 2.0

# Tolerates a missing space after the colon, which several stylesheets omit.
_FONT_PX_PATTERN = re.compile(r"font-size:\s*(\d+)px")

# Marks a stylesheet this module already grew. A sheet can reach the tree walk
# after a style helper scaled it, and a second pass would compound the factor.
_ALREADY_SCALED_MARK = "/* fs */"

_cached_font_scale: float | None = None


def ui_font_scale() -> float:
    """How much larger than the OS default the QGIS application font is.

    1.0 on a default install of any operating system, so the panel looks
    exactly as it does today unless the user asked for bigger text. Computed
    once: QGIS applies its font at startup and does not change it while
    running, and this is read for every styled widget in the panel.
    """
    global _cached_font_scale
    if _cached_font_scale is not None:
        return _cached_font_scale
    _cached_font_scale = _measure_font_scale()
    return _cached_font_scale


def _measure_font_scale() -> float:
    """Work the factor out once. 1.0 whenever anything is unreadable."""
    try:
        from qgis.PyQt.QtGui import QFontDatabase

        from ...core.qt_compat import resolve_qt_enum

        general = resolve_qt_enum(QFontDatabase, "SystemFont", "GeneralFont")
        os_points = float(QFontDatabase.systemFont(general).pointSizeF())
        chosen_points = _qgis_chosen_font_points()
        if os_points <= 0.0 or chosen_points <= 0.0:
            return 1.0
        return max(1.0, min(_MAX_FONT_SCALE, chosen_points / os_points))
    except Exception:  # noqa: BLE001 - an unreadable font must not break the panel
        return 1.0


def _qgis_chosen_font_points() -> float:
    """The text size the user picked in QGIS, 0.0 when it cannot be read.

    Read from the setting first, not from the application font. QGIS 3.22
    through 3.32, three of the versions this plugin supports, apply the size
    through an application style sheet, and a style sheet never changes the
    application font: reading the font there answers "default" for every user
    who raised it, and the whole of this module would quietly do nothing on a
    third of the range. Newer versions do set the font, which is the fallback.
    """
    try:
        from qgis.core import QgsSettings

        points = QgsSettings().value("qgis/stylesheet/fontPointSize", 0.0, type=float)
        if points and float(points) > 0.0:
            return float(points)
    except Exception:  # noqa: BLE001 - fall through to the font  # nosec B110
        pass
    try:
        from qgis.PyQt.QtWidgets import QApplication

        return float(QApplication.font().pointSizeF())
    except Exception:  # noqa: BLE001 - no application, no size
        return 0.0


def widget_pixel_ratio(widget) -> float:
    """Pixels the widget's own screen puts behind one drawing unit.

    Asked of the screen, not of the widget: a widget Qt has not placed yet
    answers with the highest ratio anywhere on the machine, so a laptop lid
    left open at 150 percent would decide how everything is drawn on the
    external monitor the user actually works on, and the picture would come
    out softer than before it was scaled at all. Never zero.
    """
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
    """A painted text size that follows the same factor as the panel's own.

    Text drawn by hand takes no style sheet, so it is the one place the
    scaling has to be asked for rather than rewritten.
    """
    return _rounded_up_from_half(points * ui_font_scale())


def scale_qss_font_px(qss: str) -> str:
    """Grow every pixel font size in a stylesheet by the factor above.

    Returns the stylesheet unchanged when the user is on the default font, so
    the common path costs one substring test. Safe to call twice on the same
    string.
    """
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
    """Round to the nearest whole size, halves going up.

    Python's own round() sends halves to the even neighbour, so at one and a
    half times the base 10px would land on 15 and 11px on 16: the two closest
    sizes in the panel would come out one apart instead of two, and 11px is
    the size most of the text uses. Up is also the right way to miss for
    somebody who asked for bigger text.
    """
    return max(1, int(value + 0.5))


def scale_px_length(pixels: int) -> int:
    """Grow a fixed box that has to keep holding its own text.

    A round step dial or a sign badge is a glyph inside a circle of a set
    size: growing the glyph alone would push it out of its own circle, so the
    two travel together.
    """
    return _rounded_up_from_half(pixels * ui_font_scale())


def apply_font_scale_to_tree(root) -> None:
    """Grow the pixel font sizes of a built widget and everything under it.

    Called once a panel or dialog has finished building, which is what makes
    the stylesheets written inline at each widget follow the setting without
    every one of them having to ask. A widget restyled later goes through the
    style helpers, which scale on their own.
    """
    if ui_font_scale() <= 1.0:
        return
    try:
        from qgis.PyQt.QtWidgets import QWidget

        widgets = [root] + root.findChildren(QWidget)
    except Exception:  # noqa: BLE001 - a torn-down tree is not worth a traceback
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
