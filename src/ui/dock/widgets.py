



from __future__ import annotations

import enum
import html

from qgis.PyQt.QtCore import QEvent, QObject, Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
from ...core.server_dials import dial_copy
from .engine_card_button import EngineCardButton
from .font_scale import scale_px_length, scale_qss_font_px
from .styles import (
    _BTN_GHOST,
    _ENGINE_CARD_GLOSS_PICKED_QSS,
    _ENGINE_CARD_GLOSS_QSS,
    _ENGINE_CARD_PICK_QSS,
    _ENGINE_CARD_TITLE_PICKED_QSS,
    _ENGINE_CARD_TITLE_QSS,
    _HERO_TITLE_QSS,
    _HINT_LINE_QSS,
    _METHOD_SWITCH_QSS,
    BRAND_BLUE,
    BRAND_BLUE_HOVER,
    ENGINE_CARD_HUES,
    FIELD,
    FONT_BODY,
    HAIRLINE_STRONG,
    INK_2,
    INK_3,
    LINE,
    MODE_HUES,
    ON_ACCENT,
    RADIUS_CHIP,
    SPACE_CARD,
    SPACE_STAGE,
    _mode_tabs_qss,
    category_ink,
)


class _WheelGuard(QObject):









    def __init__(self, viewport, parent=None):
        super().__init__(parent)
        self._viewport = viewport

    def eventFilter(self, obj, event):







        try:
            if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
                if self._viewport is not None:
                    QApplication.sendEvent(self._viewport, event)
                return True
        except RuntimeError:
            return False
        return False


class _ShortcutArmingFilter(QObject):

















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
                pass  # nosec B110
        return False


class Mode(enum.Enum):
    INTERACTIVE = "interactive"
    AUTOMATIC = "automatic"



_HERO_TILE_PX = 44
_HERO_GLYPH_PX = 22


def _paint_hero_glyph(label: QLabel, name: str) -> None:

    try:
        from ..icons import pixmap_for

        label.setPixmap(pixmap_for(label, name, _HERO_GLYPH_PX, QColor(INK_2)))
    except Exception:  # noqa: BLE001
        label.clear()


def build_no_imagery_hero(on_demo, *, glyph: str = "map"):















    wrapper = QWidget()
    wrapper.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
    outer = QVBoxLayout(wrapper)
    outer.setContentsMargins(8, 0, 8, 0)
    outer.setSpacing(0)
    outer.addStretch(1)

    card = QWidget()
    card.setObjectName("firstRunHero")
    card.setStyleSheet(
        "QWidget#firstRunHero { background: transparent; border: none; }"
        "QLabel { background: transparent; border: none; }"
    )
    col = QVBoxLayout(card)
    col.setContentsMargins(0, 0, 0, 0)
    col.setSpacing(6)

    tile_px = scale_px_length(_HERO_TILE_PX)
    _glyph = QLabel()
    _glyph.setFixedSize(tile_px, tile_px)
    _glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)
    _glyph.setStyleSheet(
        f"QLabel {{ background: {FIELD}; border: 1px solid {LINE};"
        f" border-radius: {tile_px // 2}px; }}")
    _paint_hero_glyph(_glyph, glyph)
    col.addWidget(_glyph, 0, Qt.AlignmentFlag.AlignHCenter)
    col.addSpacing(6)

    _title = QLabel(tr("Load your own imagery"))
    _title.setWordWrap(True)
    _title.setAlignment(Qt.AlignmentFlag.AlignHCenter)

    _title.setStyleSheet(_HERO_TITLE_QSS)
    col.addWidget(_title)


    _formats = QLabel(tr("Any GeoTIFF, WMS or XYZ basemap."))
    _formats.setWordWrap(True)
    _formats.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    _formats.setStyleSheet(scale_qss_font_px(
        f"font-size: {FONT_BODY}px; color: {INK_2};"))
    col.addWidget(_formats)





    _div = QHBoxLayout()
    _div.setContentsMargins(0, 0, 0, 0)
    col.addLayout(_div)
    col.addSpacing(SPACE_STAGE)



    demo_btn = QPushButton(tr("Load example imagery"))
    demo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    demo_btn.setStyleSheet(_BTN_GHOST)
    demo_btn.clicked.connect(on_demo)
    col.addWidget(demo_btn, 0, Qt.AlignmentFlag.AlignHCenter)




    show_btn = QPushButton(tr("Show it on the map"))
    show_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    show_btn.setStyleSheet(_BTN_GHOST)
    show_btn.setVisible(False)
    col.addWidget(show_btn, 0, Qt.AlignmentFlag.AlignHCenter)

    outer.addWidget(card)
    outer.addStretch(2)
    wrapper.hero_glyph = _glyph
    wrapper.hero_title = _title
    wrapper.hero_line = _formats
    wrapper.hero_divider = _div
    wrapper.hero_demo_btn = demo_btn
    wrapper.hero_show_btn = show_btn
    wrapper.hero_variant = "empty"
    wrapper.hero_glyph_text = glyph
    return wrapper, demo_btn


def set_hero_variant(wrapper, variant: str) -> None:





    if getattr(wrapper, "hero_variant", None) == variant:
        return
    wrapper.hero_variant = variant
    hidden = variant == "hidden"
    _paint_hero_glyph(wrapper.hero_glyph, "eye" if hidden else wrapper.hero_glyph_text)
    wrapper.hero_title.setText(
        tr("Your imagery is hidden") if hidden else tr("Load your own imagery"))
    wrapper.hero_line.setText(
        tr("It is unchecked in the Layers panel.") if hidden
        else tr("Any GeoTIFF, WMS or XYZ basemap."))
    for i in range(wrapper.hero_divider.count()):
        item = wrapper.hero_divider.itemAt(i)
        if item is not None and item.widget() is not None:
            item.widget().setVisible(not hidden)
    wrapper.hero_demo_btn.setVisible(not hidden)
    wrapper.hero_show_btn.setVisible(hidden)


class _Spinner(QWidget):








    def __init__(self, diameter: int = 16, parent=None):
        super().__init__(parent)
        self._angle = 0
        diameter = scale_px_length(diameter)
        self._d = diameter
        self.setFixedSize(diameter, diameter)

    def advance(self):
        self._angle = (self._angle + 30) % 360
        self.update()

    def paintEvent(self, event):  # noqa: N802




        try:
            from qgis.PyQt.QtCore import QRectF
            from qgis.PyQt.QtGui import QColor, QPainter, QPen
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            margin = 2.0
            rect = QRectF(margin, margin,
                          self._d - 2 * margin, self._d - 2 * margin)
            pen = QPen(QColor(BRAND_BLUE))
            pen.setWidthF(2.2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawArc(rect, int(-self._angle * 16), 270 * 16)
            painter.end()
        except Exception:  # noqa: BLE001
            return


class _ZoneGestureGlyph(QWidget):






    def __init__(self, color, size: int = 56, parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(size, size)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def paintEvent(self, event):  # noqa: N802




        try:
            from qgis.PyQt.QtCore import QPointF
            from qgis.PyQt.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
            s = float(self.width())
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            pts = [(0.22, 0.34), (0.55, 0.20), (0.82, 0.46), (0.60, 0.74)]
            scr = [QPointF(x * s, y * s) for (x, y) in pts]

            line = QPen(self._color)
            line.setWidthF(s * 0.045)
            line.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            line.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(line)
            for i in range(len(scr) - 1):
                p.drawLine(scr[i], scr[i + 1])

            cursor_tip = QPointF(0.30 * s, 0.72 * s)
            dashed = QPen(self._color)
            dashed.setWidthF(s * 0.045)
            dashed.setStyle(Qt.PenStyle.DashLine)
            p.setPen(dashed)
            p.drawLine(scr[-1], cursor_tip)

            ring = QPen(self._color)
            ring.setWidthF(s * 0.03)
            p.setPen(ring)
            p.setBrush(QBrush(QColor(255, 255, 255)))
            r = s * 0.055
            for pt in scr:
                p.drawEllipse(pt, r, r)

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
        except Exception:  # noqa: BLE001
            return






_KEY_BADGE_STYLE = (
    f"background-color: {FIELD};"
    f" border: 1px solid {HAIRLINE_STRONG};"
    f" border-radius: {RADIUS_CHIP}px;"
    " padding: 1px 5px;"
)


def label_with_target_hint(label: str, hint: str) -> str:








    return (f'{html.escape(label)} <span style="color: {INK_2};">'
            f'({html.escape(hint)})</span>')


def native_key(key) -> str:



    from qgis.PyQt.QtGui import QKeySequence
    text = QKeySequence(key).toString(QKeySequence.SequenceFormat.NativeText)

    return "Enter" if text == "Return" else text


def make_shortcut_hint(pairs: list[tuple[str, str]]) -> QLabel:








    parts = []
    for key, action in pairs:
        parts.append(
            f'<span style="{_KEY_BADGE_STYLE}">{html.escape(key)}</span>&nbsp;{html.escape(action)}')
    label = QLabel("&nbsp;&nbsp;·&nbsp;&nbsp;".join(parts))
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setWordWrap(True)
    label.setStyleSheet(_HINT_LINE_QSS)
    return label











_MODE_SWITCH_QSS = _mode_tabs_qss("modeSwitchFrame") + "".join(
    f'QPushButton[mode="{mode}"]:checked {{'
    f"  color: {category_ink(hue)};"
    f"  border-bottom: 2px solid {category_ink(hue)};"
    "}"
    for mode, hue in MODE_HUES.items()
)


class _ModeSwitch(QFrame):



    mode_selected = pyqtSignal(object)

    def __init__(self, current_mode: Mode, parent=None):
        super().__init__(parent)
        self.setObjectName("modeSwitchFrame")


        self.setFixedHeight(scale_px_length(34))



        self.setMinimumWidth(scale_px_length(220))
        self.setAccessibleName(tr("Mode selection"))
        self.setAccessibleDescription(
            tr("Choose between Semi-Auto and Automatic segmentation"))

        outer = QHBoxLayout(self)


        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)



        self._interactive_btn = QPushButton(tr("Semi-Auto"))
        self._interactive_btn.setCheckable(True)
        self._interactive_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._interactive_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._interactive_btn.setProperty("mode", "interactive")






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
            "Runs on our servers and uses your cloud objects."))





        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)
        self._btn_group.addButton(self._interactive_btn, 0)
        self._btn_group.addButton(self._automatic_btn, 1)




        for tab in (self._interactive_btn, self._automatic_btn):
            tab.setSizePolicy(QSizePolicy.Policy.Preferred,
                              QSizePolicy.Policy.Expanding)

        outer.addWidget(self._interactive_btn, 1)
        outer.addWidget(self._automatic_btn, 1)

        self.setStyleSheet(scale_qss_font_px(_MODE_SWITCH_QSS))


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


        try:
            from qgis.PyQt.QtGui import QIcon

            btn.setIcon(QIcon())
        except Exception:  # noqa: BLE001
            return
        btn.update()

    def _on_id_toggled(self, btn_id: int, checked: bool) -> None:
        if not checked:
            return
        mode = Mode.INTERACTIVE if btn_id == 0 else Mode.AUTOMATIC
        self._repolish(self._interactive_btn)
        self._repolish(self._automatic_btn)
        self.mode_selected.emit(mode)

    def set_mode(self, mode: Mode) -> None:

        self._btn_group.blockSignals(True)
        if mode == Mode.INTERACTIVE:
            self._interactive_btn.setChecked(True)
        else:
            self._automatic_btn.setChecked(True)
        self._repolish(self._interactive_btn)
        self._repolish(self._automatic_btn)
        self._btn_group.blockSignals(False)


class _MethodSwitch(QFrame):








    method_selected = pyqtSignal(str)

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

        self._btn_group.blockSignals(True)
        (self._manual_btn if method == "manual" else self._ai_btn).setChecked(True)
        self._repolish(self._ai_btn)
        self._repolish(self._manual_btn)
        self._btn_group.blockSignals(False)


_ENGINE_GLYPH_PX = 16


class _EngineSwitch(QWidget):
















    engine_selected = pyqtSignal(bool)

    def __init__(self, cloud: bool = True, parent=None):
        super().__init__(parent)
        self.setAccessibleName(tr("AI engine"))
        self.setAccessibleDescription(
            tr("Choose where the AI runs: on TerraLab servers, or on your "
               "own computer"))

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACE_CARD)



        self._card_text: dict[QPushButton, tuple[QLabel, QLabel]] = {}
        self._card_glyph: dict[QPushButton, tuple[QLabel, str]] = {}
        self._card_tick: dict[QPushButton, QLabel] = {}


















        self._cloud_btn = self._build_card(
            "cloud",
            tr("Cloud AI"),
            dial_copy("engine.cloud_gloss",
                      tr("Bigger model, more accurate")))
        self._local_btn = self._build_card(
            "local", tr("My computer"),
            dial_copy("engine.local_gloss",
                      tr("Smaller model, works offline")))

        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)
        self._btn_group.addButton(self._cloud_btn, 0)
        self._btn_group.addButton(self._local_btn, 1)



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












        btn = EngineCardButton(scale_px_length(54))
        btn.setCheckable(True)
        btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setProperty("engine", key)
        btn.setStyleSheet(_ENGINE_CARD_PICK_QSS)

        btn.setAccessibleName(title)
        btn.setAccessibleDescription(gloss)

        inner = QVBoxLayout(btn)
        inner.setContentsMargins(10, 8, 10, 8)
        inner.setSpacing(2)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(6)
        glyph = QLabel()
        glyph_px = scale_px_length(_ENGINE_GLYPH_PX)
        glyph.setFixedSize(glyph_px, glyph_px)
        glyph.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        glyph.setStyleSheet("background: transparent; border: none;")
        head.addWidget(glyph, 0, Qt.AlignmentFlag.AlignVCenter)
        self._card_glyph[btn] = (glyph, "cloud" if key == "cloud" else "laptop")
        name = QLabel(title)




        note = QLabel(gloss)
        note.setWordWrap(True)


        name.setWordWrap(True)
        for label in (name, note):
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        head.addWidget(name, 1)


        tick = QLabel()
        tick_px = scale_px_length(14)
        tick.setFixedSize(tick_px, tick_px)
        tick.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        tick.setStyleSheet("background: transparent; border: none;")
        head.addWidget(tick, 0, Qt.AlignmentFlag.AlignTop)
        self._card_tick[btn] = tick
        inner.addLayout(head)
        inner.addWidget(note)


        inner.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._card_text[btn] = (name, note)
        return btn

    def _repolish(self) -> None:

        for btn in (self._cloud_btn, self._local_btn):
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            on = btn.isChecked()
            name, note = self._card_text[btn]

            name.setStyleSheet(_ENGINE_CARD_TITLE_PICKED_QSS if on
                               else _ENGINE_CARD_TITLE_QSS)
            note.setStyleSheet(_ENGINE_CARD_GLOSS_PICKED_QSS if on
                               else _ENGINE_CARD_GLOSS_QSS)
            glyph, glyph_name = self._card_glyph[btn]
            try:
                from ..icons import pixmap_for



                hue = ENGINE_CARD_HUES["cloud" if glyph_name == "cloud" else "local"]
                colour = QColor(category_ink(hue))
                glyph.setPixmap(pixmap_for(glyph, glyph_name, _ENGINE_GLYPH_PX, colour))
                tick = self._card_tick.get(btn)
                if tick is not None:
                    if on:
                        tick.setPixmap(pixmap_for(tick, "check", 14, colour))
                    else:
                        tick.clear()
            except Exception:  # noqa: BLE001
                glyph.clear()
            btn.update()

    def _on_engine_id_toggled(self, btn_id: int, checked: bool) -> None:
        if not checked:
            return
        self._repolish()
        self.engine_selected.emit(btn_id == 0)

    def set_engine_cloud(self, cloud: bool) -> None:






        btn = self._cloud_btn if cloud else self._local_btn
        if btn.isChecked():
            return
        self._cloud_btn.blockSignals(True)
        self._local_btn.blockSignals(True)
        self._btn_group.blockSignals(True)
        try:
            btn.setChecked(True)
        finally:
            self._btn_group.blockSignals(False)
            self._cloud_btn.blockSignals(False)
            self._local_btn.blockSignals(False)
        self._repolish()

    def set_cloud_gloss(self, gloss: str) -> None:





        _name, note = self._card_text[self._cloud_btn]
        if note.text() != gloss:
            note.setText(gloss)

    def set_cloud(self, cloud: bool) -> None:

        self._btn_group.blockSignals(True)
        (self._cloud_btn if cloud else self._local_btn).setChecked(True)
        self._repolish()
        self._btn_group.blockSignals(False)


def checkbox_indicator_qss(dock) -> str:





















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


            svg_off = f'{head}{box} fill="none" stroke="#8c8c8c" stroke-opacity="0.9" stroke-width="1.5"/></svg>'


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









    css_off = path_off.replace('"', '\\"')
    css_on = path_on.replace('"', '\\"')
    return (
        native_only
        + f"QCheckBox::indicator {{ width: {sz}px; height: {sz}px; border: none;"
        f' image: url("{css_off}"); }}'
        f'QCheckBox::indicator:checked {{ image: url("{css_on}"); }}'
    )


def _key_chip_pixmap(key: str, on_fill: bool, ratio: float):

    from qgis.PyQt.QtCore import QRectF
    from qgis.PyQt.QtGui import QFont, QFontMetricsF, QPainter, QPixmap

    font = QFont(QApplication.font())


    font.setPixelSize(scale_px_length(11))
    font.setWeight(QFont.Weight.DemiBold)
    metrics = QFontMetricsF(font)

    height = scale_px_length(16)
    width = max(height + 2, int(metrics.horizontalAdvance(key) + scale_px_length(10)))
    pixmap = QPixmap(int(width * ratio), int(height * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if on_fill:


            fill, ink = QColor(0, 0, 0, 34), QColor(ON_ACCENT)
            ink.setAlpha(170)
        else:

            fill, ink = QColor(FIELD), QColor(INK_3)
        rect = QRectF(0.0, 0.0, float(width), float(height))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        painter.drawRoundedRect(rect, 5.0, 5.0)
        painter.setPen(ink)
        painter.setFont(font)
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), key)
    finally:
        painter.end()
    return pixmap, width, height


def set_key_chip(button: QPushButton, key: str, on_fill: bool = False) -> None:







    try:
        from qgis.PyQt.QtCore import QSize
        from qgis.PyQt.QtGui import QIcon

        from ..icons import widget_pixel_ratio

        ratio = max(2.0, float(widget_pixel_ratio(button)))
        pixmap, width, height = _key_chip_pixmap(key, on_fill, ratio)
        button.setIcon(QIcon(pixmap))
        button.setIconSize(QSize(width, height))
        button.setLayoutDirection(Qt.LayoutDirection.RightToLeft)

        from .styles import repaint_on_theme_change

        repaint_on_theme_change(
            button, lambda w, k=key, f=on_fill: set_key_chip(w, k, f))
    except Exception:  # noqa: BLE001
        return


def strip_key_suffix(label: str, key: str) -> str:


    import re

    return re.sub(r"\s*\(\s*" + re.escape(key) + r"\s*\)\s*$", "", label or "") or label
