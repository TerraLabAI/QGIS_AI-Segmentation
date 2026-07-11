"""Small reusable widgets for the AI Segmentation dock: wheel guard,
Mode enum, footer icon button, spinner, zone-gesture glyph, mode switch,
inline keyboard-shortcut hint."""
from __future__ import annotations

import enum
import html

from qgis.PyQt.QtCore import QEvent, QObject, Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


from ...core.i18n import tr
from .styles import (
    _BTN_BLUE_PRIMARY,
    BRAND_BLUE,
    BRAND_BLUE_HOVER,
    BRAND_GREEN,
    BTN_GREEN,
)


class _WheelGuard(QObject):
    """Stop mouse-wheel scrolling from changing combo/spin values in the panel.

    Inside a scroll area, hovering a QComboBox/QSpinBox while scrolling the panel
    silently changes its value (e.g. the "what to detect" text flips). This guard
    only lets the wheel change a value when the widget is actually focused (the
    user clicked into it). Otherwise it redirects the wheel to the scroll
    viewport so the panel scrolls and the value is left alone.
    """

    def __init__(self, viewport, parent=None):
        super().__init__(parent)
        self._viewport = viewport

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            if self._viewport is not None:
                QApplication.sendEvent(self._viewport, event)
            return True
        return False


class Mode(enum.Enum):
    INTERACTIVE = "interactive"
    AUTOMATIC = "automatic"


def _hero_rule():
    """A hairline 1px rule for the 'or' divider (a plain tinted QFrame, not the
    dated etched HLine)."""
    line = QFrame()
    line.setFixedHeight(1)
    line.setStyleSheet("background-color: rgba(128, 128, 128, 0.28); border: none;")
    return line


def build_no_imagery_hero(on_demo, *, glyph: str = "🗺️"):
    """Build the shared 'no imagery loaded' first-run hero (Manual + Automatic
    render the identical card). Returns ``(wrapper, demo_btn)``.

    Pared to five one-job elements: a glyph anchor, a title that names the
    user's job (the imagery is THEIRS to bring), one quiet line listing what
    counts as imagery, an 'or' divider that splits the two real paths, and the
    demo button as the fallback. The divider does the "alternatively" work
    structurally instead of in words.

    Layout: a transparent, vertically-EXPANDING wrapper holds the compact
    blue-tinted card at the TOP with a single stretch below it, so the card pins
    to the top and the surplus falls below - the plugin reads top-to-bottom, so
    the empty state starts at the top too, never centered or bottom-drifting.
    The wrapper's Expanding policy does the filling on its own; do NOT add a
    layout stretch factor in a flat (non-stacked) parent layout, or the hidden
    hero leaks vertical-expand into siblings. Caller wires visibility +
    placement.
    """
    wrapper = QWidget()
    wrapper.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
    outer = QVBoxLayout(wrapper)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)

    card = QWidget()
    card.setObjectName("firstRunHero")
    card.setStyleSheet(
        "QWidget#firstRunHero { background-color: rgba(30, 136, 229, 0.08);"
        " border: 1px solid rgba(30, 136, 229, 0.28); border-radius: 6px; }"
        "QLabel { background: transparent; border: none; color: palette(text); }"
    )
    col = QVBoxLayout(card)
    col.setContentsMargins(16, 16, 16, 16)
    col.setSpacing(7)

    _glyph = QLabel(glyph)
    _glyph.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    _glyph.setStyleSheet("font-size: 26px;")
    col.addWidget(_glyph)

    _title = QLabel(tr("Load your own imagery"))
    _title.setWordWrap(True)
    _title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    _title.setStyleSheet("font-weight: 700; font-size: 15px;")
    col.addWidget(_title)

    # One quiet line, one job: name what counts as imagery. No workflow prose.
    _formats = QLabel(tr("Any GeoTIFF, WMS or XYZ basemap."))
    _formats.setWordWrap(True)
    _formats.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    _formats.setStyleSheet("font-size: 11px; color: rgba(128, 128, 128, 0.95);")
    col.addWidget(_formats)

    # 'or' divider: the structural device that separates the two real paths
    # (bring your own vs. try a sample) so the example reads as the fallback
    # without a sentence spelling it out.
    _div = QHBoxLayout()
    _div.setContentsMargins(0, 0, 0, 0)
    _div.setSpacing(8)
    _or = QLabel(tr("or"))
    _or.setStyleSheet("font-size: 10px; color: rgba(128, 128, 128, 0.8);")
    _div.addWidget(_hero_rule(), 1)
    _div.addWidget(_or, 0)
    _div.addWidget(_hero_rule(), 1)
    col.addSpacing(2)
    col.addLayout(_div)
    col.addSpacing(2)

    demo_btn = QPushButton(tr("Load example imagery"))
    demo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    demo_btn.setMinimumHeight(30)
    demo_btn.setStyleSheet(_BTN_BLUE_PRIMARY)
    demo_btn.clicked.connect(on_demo)
    col.addWidget(demo_btn)

    outer.addWidget(card)
    outer.addStretch(1)
    return wrapper, demo_btn


class _FooterIconButton(QToolButton):
    """QToolButton whose hover tint is driven by an explicit ``hover``
    dynamic property rather than Qt's :hover pseudo-state.

    With InstantPopup menus, Qt fails to fire the synthetic Leave event
    after the menu closes, so the button stays visually pressed/hovered
    until the next real mouse move. Tracking hover ourselves lets us
    force-reset it on ``menu.aboutToHide``.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("hover", False)
        self.setProperty("active", False)

    def _repolish(self) -> None:
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def set_hovered(self, hovered: bool) -> None:
        if bool(self.property("hover")) == hovered:
            return
        self.setProperty("hover", hovered)
        self._repolish()

    def set_active(self, active: bool) -> None:
        """Leaf-green tint while the attached menu is open (mirrors AI Edit)."""
        if bool(self.property("active")) == active:
            return
        self.setProperty("active", active)
        self._repolish()

    def enterEvent(self, event):  # noqa: N802
        self.set_hovered(True)
        super().enterEvent(event)

    def leaveEvent(self, event):  # noqa: N802
        self.set_hovered(False)
        super().leaveEvent(event)


class _Spinner(QWidget):
    """A small rotating arc, the conventional 'busy' indicator. Driven by an
    external QTimer calling ``advance()`` so one timer can be paused with the
    section it belongs to. Mirrors AI Edit's pairing spinner."""

    def __init__(self, diameter: int = 16, parent=None):
        super().__init__(parent)
        self._angle = 0
        self._d = diameter
        self.setFixedSize(diameter, diameter)

    def advance(self):
        self._angle = (self._angle + 30) % 360
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt signature
        from qgis.PyQt.QtCore import QRectF
        from qgis.PyQt.QtGui import QColor, QPainter, QPen
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        margin = 2.0
        rect = QRectF(margin, margin, self._d - 2 * margin, self._d - 2 * margin)
        pen = QPen(QColor(BRAND_GREEN))
        pen.setWidthF(2.2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(rect, int(-self._angle * 16), 270 * 16)
        painter.end()


class _ZoneGestureGlyph(QWidget):
    """Vector 'click points to outline an area' glyph: dots joined by edges with
    a dashed edge running to a cursor (the next point being placed). Painted live
    in paintEvent so it stays crisp at any DPI. Blue, to echo the zone outline
    drawn on the canvas.
    """

    def __init__(self, color, size: int = 56, parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(size, size)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def paintEvent(self, event):  # noqa: N802 - Qt signature
        from qgis.PyQt.QtCore import QPointF
        from qgis.PyQt.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
        s = float(self.width())
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # The placed points of a polygon being drawn.
        pts = [(0.22, 0.34), (0.55, 0.20), (0.82, 0.46), (0.60, 0.74)]
        scr = [QPointF(x * s, y * s) for (x, y) in pts]
        # Solid edges between the placed points.
        line = QPen(self._color)
        line.setWidthF(s * 0.045)
        line.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        line.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(line)
        for i in range(len(scr) - 1):
            p.drawLine(scr[i], scr[i + 1])
        # Dashed edge from the last point to the cursor (the point being placed).
        cursor_tip = QPointF(0.30 * s, 0.72 * s)
        dashed = QPen(self._color)
        dashed.setWidthF(s * 0.045)
        dashed.setStyle(Qt.PenStyle.DashLine)
        p.setPen(dashed)
        p.drawLine(scr[-1], cursor_tip)
        # A clear white-filled, blue-ringed dot at each placed point.
        ring = QPen(self._color)
        ring.setWidthF(s * 0.03)
        p.setPen(ring)
        p.setBrush(QBrush(QColor(255, 255, 255)))
        r = s * 0.055
        for pt in scr:
            p.drawEllipse(pt, r, r)
        # Mouse cursor (arrow) at the tip, blue fill with a white edge.
        f = s * 0.020
        shape = [(0, 0), (0, 15), (3.5, 11.5), (6, 17), (8, 16), (5.5, 10.5), (10, 10)]
        cursor = QPolygonF([QPointF(cursor_tip.x() + x * f, cursor_tip.y() + y * f)
                            for (x, y) in shape])
        edge = QPen(QColor(255, 255, 255, 235))
        edge.setWidthF(s * 0.022)
        edge.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(edge)
        p.setBrush(QBrush(self._color))
        p.drawPolygon(cursor)
        p.end()


# Shared key-badge convention across TerraLab plugins: monospace span on a
# subtle grey pill. Same style as the About page's shortcuts dialog.
_KEY_BADGE_STYLE = (
    "font-family: monospace;"
    " background-color: rgba(128,128,128,0.18);"
    " border: 1px solid rgba(128,128,128,0.35);"
    " border-radius: 3px;"
    " padding: 1px 5px;"
)


def native_key(key) -> str:
    """Platform-native display text for a Qt key, key combination or
    QKeySequence.StandardKey (macOS: ⌘Z / ⌫ / ⎋; Windows/Linux: Ctrl+Z /
    Backspace / Esc)."""
    from qgis.PyQt.QtGui import QKeySequence
    text = QKeySequence(key).toString(QKeySequence.SequenceFormat.NativeText)
    # Qt names the main enter key "Return" on Windows/Linux; users read "Enter".
    return "Enter" if text == "Return" else text


def make_shortcut_hint(pairs: list[tuple[str, str]]) -> QLabel:
    """One quiet line of inline keyboard hints: each ``(key, action)`` pair
    renders the key as a small monospace badge (design-system key-badge token)
    followed by the muted action text, pairs separated by a middle dot.

    ``key`` must already be a platform-native string (use :func:`native_key`);
    ``action`` is plain text (already translated). 11px, palette(text), so the
    line stays discreet on light and dark themes.
    """
    parts = []
    for key, action in pairs:
        parts.append(
            '<span style="{style}">{key}</span>&nbsp;{action}'.format(
                style=_KEY_BADGE_STYLE,
                key=html.escape(key),
                action=html.escape(action)))
    label = QLabel("&nbsp;&nbsp;·&nbsp;&nbsp;".join(parts))
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setWordWrap(True)
    label.setStyleSheet("font-size: 11px; color: palette(text);")
    return label


# QSS for the segmented mode switch: a rounded container holding two flat
# buttons. The active segment gets a SOLID mode-colored fill (green Manual,
# blue Automatic) + white bold text; the inactive one stays quiet but clearly
# clickable (palette text + hover tint). The :checked[mode=...] rules outrank
# the plain :hover rule (pseudo-class + attribute), so hovering the active
# segment never washes out its fill.
_MODE_SWITCH_QSS = (
    "QFrame#modeSwitchFrame {"
    "  background: rgba(128,128,128,0.14);"
    "  border: 1px solid rgba(128,128,128,0.22);"
    "  border-radius: 8px;"
    "}"
    "QPushButton {"  # ui-ok: segment halves of the mode switch, a self-contained component
    "  background: transparent;"
    "  border: none;"
    "  border-radius: 6px;"
    "  padding: 5px 12px;"
    "  font-size: 12px;"
    "  color: palette(text);"
    "}"
    "QPushButton:hover {"
    "  background: rgba(128,128,128,0.18);"
    "}"
    'QPushButton:checked[mode="interactive"] {'
    f"  background: {BTN_GREEN};"
    "  color: #ffffff;"
    "  font-weight: 600;"
    "}"
    'QPushButton:checked[mode="automatic"] {'
    f"  background: {BRAND_BLUE};"
    "  color: #ffffff;"
    "  font-weight: 600;"
    "}"
)


class _ModeSwitch(QFrame):
    """Segmented control with Interactive / Automatic buttons and a PRO badge."""

    mode_selected = pyqtSignal(object)  # emits Mode value

    def __init__(self, current_mode: Mode, parent=None):
        super().__init__(parent)
        self.setObjectName("modeSwitchFrame")
        self.setFixedHeight(36)
        self.setMinimumWidth(260)
        self.setAccessibleName(tr("Mode selection"))
        self.setAccessibleDescription(
            tr("Choose between Manual (local) and Automatic (cloud) segmentation"))

        outer = QHBoxLayout(self)
        outer.setContentsMargins(3, 3, 3, 3)
        outer.setSpacing(3)

        self._interactive_btn = QPushButton(tr("Manual"))
        self._interactive_btn.setCheckable(True)
        self._interactive_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._interactive_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._interactive_btn.setProperty("mode", "interactive")

        self._automatic_btn = QPushButton(tr("Automatic"))
        self._automatic_btn.setCheckable(True)
        self._automatic_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._automatic_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._automatic_btn.setProperty("mode", "automatic")

        # PRO badge label overlaid on the Automatic button. White pill with
        # blue text so it stays legible on BOTH the solid blue active fill
        # and the grey inactive container.
        self._pro_badge = QLabel("PRO", self)
        self._pro_badge.setStyleSheet(
            "background-color: rgba(255,255,255,0.92); color: {blue};"
            " border-radius: 3px; padding: 0px 4px;"
            " font-size: 9px; font-weight: bold;".format(blue=BRAND_BLUE)
        )
        self._pro_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pro_badge.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._pro_badge.adjustSize()

        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)
        self._btn_group.addButton(self._interactive_btn, 0)
        self._btn_group.addButton(self._automatic_btn, 1)

        # Equal stretch so the two segments split the control's width evenly.
        outer.addWidget(self._interactive_btn, 1)
        outer.addWidget(self._automatic_btn, 1)

        self.setStyleSheet(_MODE_SWITCH_QSS)

        # Set initial state without emitting
        self._interactive_btn.blockSignals(True)
        self._automatic_btn.blockSignals(True)
        if current_mode == Mode.INTERACTIVE:
            self._interactive_btn.setChecked(True)
        else:
            self._automatic_btn.setChecked(True)
        self._repolish(self._interactive_btn)
        self._repolish(self._automatic_btn)
        self._interactive_btn.blockSignals(False)
        self._automatic_btn.blockSignals(False)

        self._btn_group.idToggled.connect(self._on_id_toggled)

    def _repolish(self, btn: QPushButton) -> None:
        btn.style().unpolish(btn)
        btn.style().polish(btn)
        btn.update()

    def _on_id_toggled(self, btn_id: int, checked: bool) -> None:
        if not checked:
            return
        mode = Mode.INTERACTIVE if btn_id == 0 else Mode.AUTOMATIC
        self._repolish(self._interactive_btn)
        self._repolish(self._automatic_btn)
        self.mode_selected.emit(mode)

    def set_mode(self, mode: Mode) -> None:
        """Set the active button without emitting mode_selected."""
        self._btn_group.blockSignals(True)
        if mode == Mode.INTERACTIVE:
            self._interactive_btn.setChecked(True)
        else:
            self._automatic_btn.setChecked(True)
        self._repolish(self._interactive_btn)
        self._repolish(self._automatic_btn)
        self._btn_group.blockSignals(False)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._position_pro_badge()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._position_pro_badge()

    def _position_pro_badge(self) -> None:
        """Place the PRO badge in the upper-right area of the Automatic button."""
        self._pro_badge.adjustSize()
        btn = self._automatic_btn
        bx = btn.x()
        bw = btn.width()
        bh = btn.height()
        pw = self._pro_badge.width()
        ph = self._pro_badge.height()
        x = bx + bw - pw - 4
        y = (bh - ph) // 2 - 2
        self._pro_badge.move(x, max(2, y))


def checkbox_indicator_qss(dock) -> str:
    """QSS fragment that draws a VISIBLE checkbox indicator in both states.

    Qt's native indicator can render invisible when unchecked (dark themes,
    macOS), so an OFF checkbox reads as an empty row - the user cannot even
    tell there is something to click. This paints two theme-agnostic pixmaps
    at runtime (off: grey rounded outline box; on: brand-blue filled box with
    a white check) into a per-dock temp dir and returns the stylesheet block
    referencing them. The dir is stored as dock._checkbox_icon_dir (reused on
    repeat calls, deleted by the dock's unload cleanup)."""
    import os
    import tempfile

    from qgis.PyQt.QtGui import QColor, QPainter, QPen, QPixmap

    sz = 18
    icon_dir = getattr(dock, "_checkbox_icon_dir", None)
    if not icon_dir:
        icon_dir = tempfile.mkdtemp(prefix="qgis_ai_seg_")
        dock._checkbox_icon_dir = icon_dir
    path_off = os.path.join(icon_dir, "cb_off.png").replace("\\", "/")
    path_on = os.path.join(icon_dir, "cb_on.png").replace("\\", "/")
    if not (os.path.exists(path_off) and os.path.exists(path_on)):
        box = (1, 1, sz - 3, sz - 3)
        # Unchecked: transparent fill + mid-grey outline (legible on both
        # light and dark backgrounds).
        pm_off = QPixmap(sz, sz)
        pm_off.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm_off)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(140, 140, 140, 230))
        pen.setWidthF(1.5)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(*box, 4, 4)
        p.end()
        # Checked: brand-blue filled box + white check (the darker hover
        # shade reads better than the base blue behind a white checkmark).
        pm_on = QPixmap(sz, sz)
        pm_on.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm_on)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        blue = QColor(BRAND_BLUE_HOVER)
        p.setPen(QPen(blue, 1.5))
        p.setBrush(blue)
        p.drawRoundedRect(*box, 4, 4)
        check = QPen(QColor(255, 255, 255))
        check.setWidthF(2.2)
        check.setCapStyle(Qt.PenCapStyle.RoundCap)
        check.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(check)
        p.drawLine(5, 9, 8, 12)
        p.drawLine(8, 12, 13, 5)
        p.end()
        pm_off.save(path_off, "PNG")
        pm_on.save(path_on, "PNG")
    return (
        "QCheckBox {{ background: transparent; }}"
        "QCheckBox::indicator {{ width: {sz}px; height: {sz}px; border: none;"
        " image: url({off}); }}"
        "QCheckBox::indicator:checked {{ image: url({on}); }}"
    ).format(sz=sz, off=path_off, on=path_on)
