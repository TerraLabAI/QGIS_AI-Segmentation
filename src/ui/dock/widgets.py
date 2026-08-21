"""Small reusable widgets for the AI Segmentation dock: wheel guard,
Mode enum, footer icon button, spinner, zone-gesture glyph, mode switch,
inline keyboard-shortcut hint, control label with its target hint."""
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
from ...core.server_dials import dial_copy
from .font_scale import scale_px_length
from .styles import (
    _BTN_BLUE_PRIMARY,
    _CLOUD_EMOJI,
    _ENGINE_CARD_GLOSS_ON_QSS,
    _ENGINE_CARD_GLOSS_QSS,
    _ENGINE_CARD_QSS,
    _ENGINE_CARD_TITLE_ON_QSS,
    _ENGINE_CARD_TITLE_QSS,
    _LAPTOP_EMOJI,
    _METHOD_SWITCH_QSS,
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


class _ShortcutArmingFilter(QObject):
    """Re-arm the dock's window shortcuts just before Qt matches a key.

    A window shortcut takes its key the moment it matches, whatever its
    handler decides afterwards, so a gate inside the handler cannot give the
    key back: Escape stays eaten across QGIS, Delete never reaches the vertex
    tool, and a second enabled Ctrl+Z in the window makes Qt drop the press
    for both. Only the enabled flag is read before the key is taken, and the
    states that decide it are written all over the dock.

    Qt offers the key to the focus widget as a ShortcutOverride first, and an
    ignored one walks up to the window, so this sees every press that the
    shortcut map is about to be asked about. Nothing is consumed here.
    """

    # G is here for the Semi-Auto Start key, which is scoped to the dock
    # rather than the window: it still matches before the focused drop-down is
    # offered the letter, so it needs the same re-arm on the press itself.
    _KEYS = frozenset({
        Qt.Key.Key_Escape, Qt.Key.Key_Return, Qt.Key.Key_Enter,
        Qt.Key.Key_Delete, Qt.Key.Key_Backspace, Qt.Key.Key_Z,
        Qt.Key.Key_G,
    })

    def __init__(self, dock, parent=None):
        super().__init__(parent)
        self._dock = dock

    def eventFilter(self, _obj, event):
        if event.type() == QEvent.Type.ShortcutOverride:
            try:
                if event.key() in self._KEYS:
                    self._dock.refresh_auto_shortcut_arming()
            except (RuntimeError, AttributeError):
                pass  # nosec B110 - a dock being torn down arms nothing
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


class _ClickableFooterArea(QWidget):
    """A footer strip that answers a click, for the credit gauge.

    The gauge is two widgets, a painted ring and its label, and Qt gives a
    plain container no clicked signal. Neither child accepts mouse buttons, so
    a press on either lands here and the whole strip reads as one link to the
    dashboard.
    """

    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt override
        # Release inside the strip, the same contract as a button: a press that
        # drags away and lets go elsewhere is a cancelled click.
        from ...core.qt_compat import event_pos
        if (event.button() == Qt.MouseButton.LeftButton
                and self.rect().contains(event_pos(event))):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


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
        # Guarded like every other paintEvent in the plugin: an exception
        # raised here escapes into Qt's own paint dispatch and takes QGIS down
        # at startup. The imports sit inside the method, so a reload that has
        # purged sys.modules is enough to raise.
        try:
            from qgis.PyQt.QtCore import QRectF
            from qgis.PyQt.QtGui import QColor, QPainter, QPen
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            margin = 2.0
            rect = QRectF(margin, margin,
                          self._d - 2 * margin, self._d - 2 * margin)
            pen = QPen(QColor(BRAND_GREEN))
            pen.setWidthF(2.2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawArc(rect, int(-self._angle * 16), 270 * 16)
            painter.end()
        except Exception:  # noqa: BLE001 -- a paintEvent must never raise
            return


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
        # Guarded like every other paintEvent in the plugin: an exception
        # raised here escapes into Qt's own paint dispatch and takes QGIS down
        # at startup. The imports sit inside the method, so a reload that has
        # purged sys.modules is enough to raise.
        try:
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
        except Exception:  # noqa: BLE001 -- a paintEvent must never raise
            return


# Shared key-badge convention across TerraLab plugins: monospace span on a
# subtle grey pill. Same style as the About page's shortcuts dialog.
# Named faces before the generic: Qt only resolves the CSS generic "monospace"
# where the platform font database aliases it, which in practice means
# fontconfig. On Windows the badge fell back to proportional Segoe UI and lost
# the key-cap look.
_KEY_BADGE_STYLE = (
    "font-family: Consolas, 'DejaVu Sans Mono', Menlo, monospace;"
    " background-color: rgba(128,128,128,0.18);"
    " border: 1px solid rgba(128,128,128,0.35);"
    " border-radius: 3px;"
    " padding: 1px 5px;"
)


def label_with_target_hint(label: str, hint: str) -> str:
    """One control label followed by the objects it is made for, in a dimmed
    parenthesis: "Right angles (buildings)".

    The toggles that shape an outline were skipped because nothing on the row
    said what they were for; the tooltip only pays off once the user hovers.
    Rich text, so the hint reads as secondary next to the label. Both parts
    arrive translated.
    """
    return (f'{html.escape(label)} <span style="color: rgba(128, 128, 128, '
            f'0.85);">({html.escape(hint)})</span>')


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
            f'<span style="{_KEY_BADGE_STYLE}">{html.escape(key)}</span>&nbsp;{html.escape(action)}')
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
        # Grows with its own labels: a fixed 36 clips "Automatic" and the PRO
        # badge as soon as the user raises the QGIS text size.
        self.setFixedHeight(scale_px_length(36))
        self.setMinimumWidth(scale_px_length(260))
        self.setAccessibleName(tr("Mode selection"))
        self.setAccessibleDescription(
            tr("Choose between Semi-Auto and Automatic segmentation"))

        outer = QHBoxLayout(self)
        outer.setContentsMargins(3, 3, 3, 3)
        outer.setSpacing(3)

        # The label the user reads. The internal value stays "interactive"
        # below, because the MCP API is built on it.
        self._interactive_btn = QPushButton(tr("Semi-Auto"))
        self._interactive_btn.setCheckable(True)
        self._interactive_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._interactive_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._interactive_btn.setProperty("mode", "interactive")
        # The two halves carried no description at all, on or off screen. The
        # segment is where the user picks a way of working, so each half says
        # what it does and what it costs before they commit to it.
        # Never says where this mode runs: Semi-Auto carries its own engine
        # picker underneath, so the tooltip claiming "runs on your computer"
        # contradicted the Cloud AI card the user had just selected.
        self._interactive_btn.setToolTip(tr(
            "One object at a time: click it, the AI outlines it. You choose "
            "where it runs, on our servers or on your own computer."))

        self._automatic_btn = QPushButton(tr("Automatic"))
        self._automatic_btn.setCheckable(True)
        self._automatic_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._automatic_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._automatic_btn.setProperty("mode", "automatic")
        self._automatic_btn.setToolTip(tr(
            "Draw a zone, name one kind of object, get all of them in one run. "
            "Runs on our servers and uses your cloud detections."))

        # No badge on the Automatic half. A "PRO" pill on the tab read as a
        # locked, complicated mode before the user had seen what it does; the
        # paid part is explained where it costs something (the credit gauge,
        # the upsell band, Account Settings), not on the way in.
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


class _MethodSwitch(QFrame):
    """Segmented AI | Manual switch for the Correct step.

    Two equal halves; the active half carries the armed-blue tint. It swaps
    ONLY the fix method (on-device AI points vs QGIS vertices), so it mirrors
    the mode switch above but simpler: no PRO badge, and it emits the plain
    method string ("ai" | "manual") on a user toggle only.
    """

    method_selected = pyqtSignal(str)  # "ai" | "manual"

    def __init__(self, current: str = "manual", parent=None):
        super().__init__(parent)
        self.setObjectName("methodSwitchFrame")
        self.setFixedHeight(scale_px_length(32))
        self.setAccessibleName(tr("Fix method"))
        self.setAccessibleDescription(
            tr("Choose how to fix the polygon: AI points or QGIS vertices"))
        self.setToolTip(tr(
            "AI: point at what to keep or trim, one cloud detection per "
            "polygon. Manual: move the corners yourself, free."))

        outer = QHBoxLayout(self)
        outer.setContentsMargins(3, 3, 3, 3)
        outer.setSpacing(3)

        self._ai_btn = QPushButton(tr("AI"))
        self._manual_btn = QPushButton(tr("Manual"))
        for btn, key in ((self._ai_btn, "ai"), (self._manual_btn, "manual")):
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setProperty("method", key)

        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)
        self._btn_group.addButton(self._ai_btn, 0)
        self._btn_group.addButton(self._manual_btn, 1)

        # AI reads first (left half): it is the method the step opens on, and
        # the default and the reading order must agree. The QButtonGroup ids
        # above (ai=0, manual=1) are unaffected by layout order, so
        # _on_id_toggled's id-to-method mapping stays correct as-is, and so
        # does the tab order, which follows the order widgets are added.
        outer.addWidget(self._ai_btn, 1)
        outer.addWidget(self._manual_btn, 1)
        self.setStyleSheet(_METHOD_SWITCH_QSS)

        self._ai_btn.blockSignals(True)
        self._manual_btn.blockSignals(True)
        (self._manual_btn if current == "manual" else self._ai_btn).setChecked(True)
        self._repolish(self._ai_btn)
        self._repolish(self._manual_btn)
        self._ai_btn.blockSignals(False)
        self._manual_btn.blockSignals(False)

        self._btn_group.idToggled.connect(self._on_id_toggled)

    def _repolish(self, btn: QPushButton) -> None:
        btn.style().unpolish(btn)
        btn.style().polish(btn)
        btn.update()

    def _on_id_toggled(self, btn_id: int, checked: bool) -> None:
        if not checked:
            return
        self._repolish(self._ai_btn)
        self._repolish(self._manual_btn)
        self.method_selected.emit("manual" if btn_id == 1 else "ai")

    def method(self) -> str:
        return "manual" if self._manual_btn.isChecked() else "ai"

    def set_method(self, method: str) -> None:
        """Set the active half without emitting method_selected."""
        self._btn_group.blockSignals(True)
        (self._manual_btn if method == "manual" else self._ai_btn).setChecked(True)
        self._repolish(self._ai_btn)
        self._repolish(self._manual_btn)
        self._btn_group.blockSignals(False)


class _EngineSwitch(QWidget):
    """Where a Semi-Auto click is answered: two option cards, pick one.

    Not the segmented bar the mode and method switches use. That bar switches
    a view and a transparent resting half is right for it. This picks where
    the work runs and what it costs, so both sides carry a fill and a border
    even unpicked: the one you did not take has to look like something you
    could take.

    Two builds got this wrong before. A full chooser screen stopped the user
    before the Start button. A single "Cloud AI" checkbox then hid the
    alternative, so the choice read as a guess.

    Each card carries the name and the one thing that separates it from the
    other. The detail goes on the line the caller writes underneath.
    """

    engine_selected = pyqtSignal(bool)  # True = cloud, False = this computer

    def __init__(self, cloud: bool = True, parent=None):
        super().__init__(parent)
        self.setAccessibleName(tr("AI engine"))
        self.setAccessibleDescription(
            tr("Choose where the AI runs: on TerraLab servers, or on your "
               "own computer"))

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        # Filled by _build_card: card button -> (title label, gloss label).
        self._card_text: dict[QPushButton, tuple[QLabel, QLabel]] = {}

        # The gloss carries the REASON to pick a side, not a feature list.
        # "Sharper, nothing to install" led on the install, which is the small
        # half of the difference, and "sharper" is not what the user is buying.
        # Speed and accuracy are, so they lead; the line under the cards says
        # what pays for them.
        #
        # Both glosses are served, because these two half-sentences are what
        # the whole choice turns on and a shipped one cannot be retuned until
        # the user updates. An id the server does not carry falls through to
        # the shipped English below.
        # No model name on either card. A version number tells a GIS user
        # nothing they can act on, and it dates the plugin the day we swap the
        # model. Size and where it runs are the two facts that decide the pick.
        #
        # ONE LINE, hard budget: the gloss wraps but the card does not grow, so
        # a second line is clipped mid-word. About 31 characters fit per card at
        # the dock's 260px minimum. Keep every locale under that, here and in
        # the served copy.
        self._cloud_btn = self._build_card(
            "cloud",
            f"{_CLOUD_EMOJI}  " + tr("Cloud AI"),
            dial_copy("engine.cloud_gloss",
                      tr("Bigger model, more accurate")))
        self._local_btn = self._build_card(
            "local", f"{_LAPTOP_EMOJI}  " + tr("My computer"),
            dial_copy("engine.local_gloss",
                      tr("Smaller model, works offline")))

        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)
        self._btn_group.addButton(self._cloud_btn, 0)
        self._btn_group.addButton(self._local_btn, 1)

        # Cloud reads first: it is what the mode opens on, and the default and
        # the reading order must agree.
        outer.addWidget(self._cloud_btn, 1)
        outer.addWidget(self._local_btn, 1)

        self._cloud_btn.blockSignals(True)
        self._local_btn.blockSignals(True)
        (self._cloud_btn if cloud else self._local_btn).setChecked(True)
        self._repolish()
        self._cloud_btn.blockSignals(False)
        self._local_btn.blockSignals(False)

        self._btn_group.idToggled.connect(self._on_engine_id_toggled)

    def _build_card(self, key: str, title: str, gloss: str) -> QPushButton:
        """One option card: a checkable button wearing two lines of text.

        Labels rather than a button caption with a newline in it, because the
        two lines are not the same thing and must not read the same weight.
        They are transparent to the mouse, so a click anywhere on the card
        reaches the button under them.

        Both labels are kept in ``_card_text`` so _repolish can recolour them.
        A label's own stylesheet beats anything the button's QSS says about its
        children, so the black-on-blue of the picked state has to be written
        here rather than selected for.
        """
        btn = QPushButton()
        btn.setCheckable(True)
        btn.setMinimumHeight(scale_px_length(50))
        btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setProperty("engine", key)
        btn.setStyleSheet(_ENGINE_CARD_QSS)

        inner = QVBoxLayout(btn)
        inner.setContentsMargins(9, 7, 9, 7)
        inner.setSpacing(2)
        name = QLabel(title)
        # 11px on the panel's own text colour for the second line, not the 10px
        # grey it started at. That line is the whole difference between the two
        # cards, and grey made it the faintest thing on the page: the user was
        # being asked to choose from two names and a whisper.
        note = QLabel(gloss)
        note.setWordWrap(True)
        for label in (name, note):
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            inner.addWidget(label)
        self._card_text[btn] = (name, note)
        return btn

    def _repolish(self) -> None:
        """Redraw both cards, and recolour their text for the new pick."""
        for btn in (self._cloud_btn, self._local_btn):
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            on = btn.isChecked()
            name, note = self._card_text[btn]
            name.setStyleSheet(_ENGINE_CARD_TITLE_ON_QSS if on
                               else _ENGINE_CARD_TITLE_QSS)
            note.setStyleSheet(_ENGINE_CARD_GLOSS_ON_QSS if on
                               else _ENGINE_CARD_GLOSS_QSS)
            btn.update()

    def _on_engine_id_toggled(self, btn_id: int, checked: bool) -> None:
        if not checked:
            return
        self._repolish()
        self.engine_selected.emit(btn_id == 0)

    def is_cloud(self) -> bool:
        return self._cloud_btn.isChecked()

    def set_cloud_gloss(self, gloss: str) -> None:
        """Rewrite the cloud card's second line (same one-line budget).

        The build-time gloss is written before any server has answered, so the
        dock rewrites it on refresh when a served feature changes what the
        cloud side gives (see manual_engine._manual_engine_cloud_gloss)."""
        _name, note = self._card_text[self._cloud_btn]
        if note.text() != gloss:
            note.setText(gloss)

    def set_cloud(self, cloud: bool) -> None:
        """Set the picked card without emitting engine_selected."""
        self._btn_group.blockSignals(True)
        (self._cloud_btn if cloud else self._local_btn).setChecked(True)
        self._repolish()
        self._btn_group.blockSignals(False)


def checkbox_indicator_qss(dock) -> str:
    """QSS fragment that draws a VISIBLE checkbox indicator in both states.

    Qt's native indicator can render invisible when unchecked (dark themes,
    macOS), so an OFF checkbox reads as an empty row - the user cannot even
    tell there is something to click. This writes two theme-agnostic SVG files
    at runtime (off: grey rounded outline box; on: brand-blue filled box with
    a white check) into a per-dock temp dir and returns the stylesheet block
    referencing them. The dir is stored as dock._checkbox_icon_dir (reused on
    repeat calls, deleted by the dock's unload cleanup).

    SVG, not PNG, because Qt rasterises it at the pixel count the screen
    really has, whatever that is and whenever it changes. A bitmap has to be
    painted for one display ratio, and the ratio a dock reports before it is
    on screen is 1.0, so on a 125 percent display Qt stretched an 18 pixel
    box over 22.5 and the box came out chewed. The SVG icon engine is already
    a dependency: every QGIS theme icon the plugin loads is one.

    A temp directory that is full, read-only or redirected under a quota
    answers with the native indicator instead. This runs inside the dock
    build, which nothing above it guards, so a raised OSError here is the
    whole plugin failing to load."""
    import os
    import tempfile

    sz = 18
    native_only = "QCheckBox { background: transparent; }"
    try:
        icon_dir = getattr(dock, "_checkbox_icon_dir", None)
        if not icon_dir:
            icon_dir = tempfile.mkdtemp(prefix="qgis_ai_seg_")
            dock._checkbox_icon_dir = icon_dir
        path_off = os.path.join(icon_dir, "cb_off.svg").replace("\\", "/")
        path_on = os.path.join(icon_dir, "cb_on.svg").replace("\\", "/")
        if not (os.path.exists(path_off) and os.path.exists(path_on)):
            head = (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{sz}" height="{sz}"'
                f' viewBox="0 0 {sz} {sz}">'
            )
            box = f'<rect x="1" y="1" width="{sz - 3}" height="{sz - 3}" rx="4" ry="4"'
            # Unchecked: transparent fill + mid-grey outline (legible on both
            # light and dark backgrounds).
            svg_off = f'{head}{box} fill="none" stroke="#8c8c8c" stroke-opacity="0.9" stroke-width="1.5"/></svg>'
            # Checked: brand-blue filled box + white check (the darker hover
            # shade reads better than the base blue behind a white checkmark).
            svg_on = (
                f'{head}{box} fill="{BRAND_BLUE_HOVER}" stroke="{BRAND_BLUE_HOVER}" stroke-width="1.5"/>'
                '<path d="M5 9 L8 12 L13 5" fill="none" stroke="#ffffff" stroke-width="2.2"'
                ' stroke-linecap="round" stroke-linejoin="round"/></svg>'
            )
            for path, body in ((path_off, svg_off), (path_on, svg_on)):
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(body)
    except OSError:
        return native_only
    # Quoted: Qt's CSS scanner only accepts a narrow character set inside an
    # unquoted url(), so a temp path containing a space (a Windows account
    # name with a space in it) would silently drop the whole declaration.
    #
    # And escaped inside the quotes, because a double quote is legal in a path
    # on macOS and Linux (the temp root comes from the environment). One in the
    # path would close the string early and drop the declaration the quoting
    # exists to save, leaving a checkbox with no indicator at all. Qt reads the
    # backslash escape (QCss::Symbol::lexem drops it and keeps the character).
    css_off = path_off.replace('"', '\\"')
    css_on = path_on.replace('"', '\\"')
    return (
        native_only
        + f"QCheckBox::indicator {{ width: {sz}px; height: {sz}px; border: none;"
        f' image: url("{css_off}"); }}'
        f'QCheckBox::indicator:checked {{ image: url("{css_on}"); }}'
    )
