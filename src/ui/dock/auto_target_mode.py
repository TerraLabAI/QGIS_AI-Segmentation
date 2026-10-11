










from __future__ import annotations

from qgis.PyQt.QtCore import QRectF, QSize, Qt
from qgis.PyQt.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from qgis.PyQt.QtWidgets import (
    QAbstractButton,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
from .font_scale import scale_px_length, scale_qss_font_px
from .styles import (
    BRAND_BLUE_HOVER,
    DARK_UI,
    FIELD,
    FONT_BODY,
    FONT_HINT,
    HOVER,
    INK,
    INK_2,
    LINE,
)

__all__ = ["DockAutoTargetModeMixin", "LAND_COVER_WORD", "LAND_COVER_WORDS",
           "land_cover_edge_color"]


LAND_COVER_WORD = "land cover"
LAND_COVER_WORDS = ("land cover", "landcover")
_MODES = ("objects", "land_cover")


def land_cover_edge_color() -> QColor:


    return QColor(255, 255, 255, 120) if DARK_UI else QColor(0, 0, 0, 90)




_LAST_COLORS: list = []
_LAST_LEGEND: list = []


def last_land_cover_legend() -> list:

    return list(_LAST_LEGEND)


def is_land_cover_word(text) -> bool:
    return str(text or "").strip().lower() in LAND_COVER_WORDS




LAND_USE_WORDS = ("land use", "landuse", "land-use", "occupation du sol",
                  "usage du sol", "utilisation du sol", "occupation des sols")


def land_cover_request(text) -> bool:

    word = " ".join(str(text or "").replace("_", " ").split()).lower()
    return word in LAND_COVER_WORDS or word in LAND_USE_WORDS


def _legend_colors(legend: list) -> list[str]:
    colors = [e.get("color") for e in legend or () if isinstance(e, dict) and e.get("color")]
    if len(colors) >= 2:
        return colors
    from ...core.class_symbology import interpolate_colors, legend_ramp_anchors
    return interpolate_colors(legend_ramp_anchors(), 6)


def paint_land_cover_glyph(painter: QPainter, rect: QRectF, colors: list,
                           edge: QColor | None = None) -> None:

    colors = colors or _LAST_COLORS or _legend_colors([])
    cells = [(0, 0, 2, 1), (2, 0, 1, 1), (0, 1, 1, 1), (1, 1, 2, 1),
             (0, 2, 1, 1), (1, 2, 2, 1)]
    cw, ch = rect.width() / 3.0, rect.height() / 3.0
    painter.setPen(Qt.PenStyle.NoPen)
    for i, (cx, cy, w, h) in enumerate(cells):
        painter.setBrush(QColor(colors[i % len(colors)]))
        painter.drawRect(QRectF(rect.left() + cx * cw, rect.top() + cy * ch, w * cw, h * ch))
    painter.setPen(QPen(edge or land_cover_edge_color(), 1))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(rect, 2.5, 2.5)


class LandCoverGlyph(QWidget):


    def __init__(self, side: int = 16, parent=None):
        super().__init__(parent)
        px = scale_px_length(side)
        self.setFixedSize(px, px)

    def paintEvent(self, _event):  # noqa: N802
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            paint_land_cover_glyph(painter, QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), [])
            painter.end()
        except Exception:  # noqa: BLE001  # nosec B110
            pass


class _TargetSegment(QAbstractButton):



    def __init__(self, kind: str, text: str, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.setText(text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.colors: list[str] = []
        font = QFont(self.font())
        font.setPixelSize(scale_px_length(FONT_BODY + 1))
        font.setWeight(QFont.Weight.DemiBold)
        self.setFont(font)
        self.setAccessibleName(text)

    def sizeHint(self):  # noqa: N802
        metrics = QFontMetrics(self.font())
        width = (scale_px_length(16) + scale_px_length(6)
                 + metrics.horizontalAdvance(self.text()) + scale_px_length(20))
        return QSize(width, scale_px_length(30))

    def minimumSizeHint(self):  # noqa: N802
        return self.sizeHint()

    def _glyph(self, painter: QPainter, rect: QRectF, ink: QColor, on: bool) -> None:
        if self.kind == "land_cover":
            paint_land_cover_glyph(painter, rect, self.colors,
                                   QColor(255, 255, 255, 200) if on else None)
            return

        pen = QPen(ink, 1.4)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        s = rect.width()
        painter.drawRoundedRect(QRectF(rect.left(), rect.top(), s * 0.46, s * 0.46), 1.5, 1.5)
        painter.drawEllipse(QRectF(rect.left() + s * 0.56, rect.top() + s * 0.06, s * 0.40, s * 0.40))
        painter.drawRoundedRect(QRectF(rect.left() + s * 0.22, rect.top() + s * 0.58,
                                       s * 0.56, s * 0.38), 1.5, 1.5)

    def paintEvent(self, _event):  # noqa: N802
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            outer = QRectF(self.rect()).adjusted(1, 1, -1, -1)
            radius = outer.height() / 2.0
            on = self.isChecked()
            if on:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(BRAND_BLUE_HOVER))
                painter.drawRoundedRect(outer, radius, radius)
                ink = QColor("#ffffff")
            else:
                if self.underMouse():
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(HOVER))
                    painter.drawRoundedRect(outer, radius, radius)
                ink = QColor(INK if self.underMouse() else INK_2)
            if self.hasFocus():
                painter.setPen(QPen(QColor("#1e88e5") if not on else QColor("#ffffff"), 1.5))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(outer.adjusted(1, 1, -1, -1), radius - 1, radius - 1)
            metrics = QFontMetrics(self.font())
            glyph = float(scale_px_length(14))
            gap = float(scale_px_length(6))
            text_w = metrics.horizontalAdvance(self.text())
            room = outer.width() - scale_px_length(12)
            text = self.text()
            if glyph + gap + text_w > room:
                text = metrics.elidedText(text, Qt.TextElideMode.ElideRight,
                                          int(max(0.0, room - glyph - gap)))
                text_w = metrics.horizontalAdvance(text)
            total = glyph + gap + text_w
            left = outer.left() + (outer.width() - total) / 2.0
            glyph_rect = QRectF(left, outer.center().y() - glyph / 2.0, glyph, glyph)
            self._glyph(painter, glyph_rect, ink, on)
            painter.setPen(ink)
            painter.setFont(self.font())
            painter.drawText(
                QRectF(left + glyph + gap, outer.top(), text_w + 2, outer.height()),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), text)
            painter.end()
        except Exception:  # noqa: BLE001  # nosec B110
            pass


class LandCoverClassChips(QWidget):


    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[tuple[str, str]] = []
        font = QFont(self.font())
        font.setPixelSize(scale_px_length(FONT_HINT + 1))
        self.setFont(font)
        sp = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)

    def set_items(self, items: list[tuple[str, str]]) -> None:
        self._items = list(items)
        self.updateGeometry()
        self.update()

    def _layout(self, width: int) -> list[tuple[QRectF, str, str]]:
        metrics = QFontMetrics(self.font())
        h = float(scale_px_length(24))
        gap = float(scale_px_length(6))
        dot = float(scale_px_length(10))
        pad = float(scale_px_length(9))
        x = y = 0.0
        out = []
        for name, color in self._items:
            w = pad + dot + scale_px_length(6) + metrics.horizontalAdvance(name) + pad + 4
            w = min(w, float(max(1, width)))
            if x > 0 and x + w > width:
                x, y = 0.0, y + h + gap
            out.append((QRectF(x, y, w, h), name, color))
            x += w + gap
        return out

    def hasHeightForWidth(self):  # noqa: N802
        return True

    def heightForWidth(self, width):  # noqa: N802
        rects = self._layout(width)
        return int(rects[-1][0].bottom()) + 1 if rects else 0

    def sizeHint(self):  # noqa: N802
        w = max(self.width(), scale_px_length(240))
        return QSize(w, self.heightForWidth(w))

    def minimumSizeHint(self):  # noqa: N802
        return QSize(scale_px_length(80), self.heightForWidth(self.width() or 240))

    def paintEvent(self, _event):  # noqa: N802
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setFont(self.font())
            metrics = QFontMetrics(self.font())
            dot = float(scale_px_length(10))
            pad = float(scale_px_length(9))
            for rect, name, color in self._layout(self.width()):
                r = rect.adjusted(0.5, 0.5, -0.5, -0.5)
                painter.setPen(QPen(QColor(LINE), 1))
                painter.setBrush(QColor(FIELD))
                painter.drawRoundedRect(r, r.height() / 2.0, r.height() / 2.0)
                d = QRectF(r.left() + pad, r.center().y() - dot / 2.0, dot, dot)
                painter.setPen(QPen(land_cover_edge_color(), 1))
                painter.setBrush(QColor(color))
                painter.drawEllipse(d)
                painter.setPen(QColor(INK))
                tx = d.right() + scale_px_length(6)
                text = metrics.elidedText(name, Qt.TextElideMode.ElideRight,
                                          int(max(0.0, r.right() - pad - tx)))
                painter.drawText(QRectF(tx, r.top(), r.right() - tx, r.height()),
                                 int(Qt.AlignmentFlag.AlignVCenter), text)
            painter.end()
        except Exception:  # noqa: BLE001  # nosec B110
            pass


class DockAutoTargetModeMixin:


    def _build_auto_target_selector(self, card_layout) -> None:

        bar = QWidget()
        bar.setObjectName("autoTargetSeg")
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bar.setStyleSheet(
            f"QWidget#autoTargetSeg {{ background-color: {FIELD};"
            f" border: 1px solid {LINE}; border-radius: {scale_px_length(16)}px; }}")
        row = QHBoxLayout(bar)
        row.setContentsMargins(2, 2, 2, 2)
        row.setSpacing(2)
        self._auto_target_segments = {
            "objects": _TargetSegment("objects", tr("Objects")),
            "land_cover": _TargetSegment("land_cover", tr("Land cover")),
        }
        self._auto_target_segments["objects"].setToolTip(
            tr("Find separate objects you name, one by one."))
        self._auto_target_segments["land_cover"].setToolTip(
            tr("Map the whole zone into land cover classes, one layer."))
        for mode in _MODES:
            seg = self._auto_target_segments[mode]
            seg.clicked.connect(lambda _c=False, m=mode: self.set_auto_target_mode(m, "click"))
            row.addWidget(seg, 1)
        card_layout.addWidget(bar)
        card_layout.addSpacing(2)
        self._auto_target_bar = bar

    def _build_auto_land_cover_pane(self, card_layout) -> None:

        pane = QWidget()
        pane.setObjectName("autoLandCoverPane")
        col = QVBoxLayout(pane)
        col.setContentsMargins(0, 0, 0, 2)
        col.setSpacing(8)
        line = QLabel(tr("One layer, 6 classes, no gaps."))
        line.setWordWrap(True)
        line.setStyleSheet(scale_qss_font_px(
            f"QLabel {{ color: {INK_2}; font-size: {FONT_BODY}px; background: transparent; }}"))
        col.addWidget(line)
        self._auto_lc_line = line
        self._auto_lc_chips = LandCoverClassChips()
        col.addWidget(self._auto_lc_chips)

        self._build_my_classes_section(col)
        pane.setVisible(False)
        card_layout.addWidget(pane)
        self._auto_land_cover_pane = pane
        self.set_auto_land_cover_legend([])

    def _lc_setup_active(self) -> bool:

        return (self.__dict__.get("_auto_target_mode") == "land_cover"
                or bool(self.__dict__.get("_auto_land_cover_ready")))

    def auto_target_mode(self) -> str:
        return self.__dict__.get("_auto_target_mode") or "objects"

    def restore_auto_target_mode(self) -> None:



        if "_auto_target_mode" not in self.__dict__:
            self.set_auto_target_mode("objects", "restore")

    def set_auto_land_cover_legend(self, legend: list) -> None:

        legend = [e for e in legend or () if isinstance(e, dict)]
        if legend:
            items = [(e.get("name") or tr("Class {n}").format(n=e.get("id")),
                      e.get("color") or "#808080") for e in legend]
        else:
            items = []
        chips = self.__dict__.get("_auto_lc_chips")
        if chips is not None:
            chips.set_items(items)
            chips.setVisible(bool(items) and not self.my_classes_active())
        if legend:
            _LAST_COLORS[:] = _legend_colors(legend)
            _LAST_LEGEND[:] = legend
        segs = self.__dict__.get("_auto_target_segments")
        if segs:
            segs["land_cover"].colors = _legend_colors(legend)
            segs["land_cover"].update()

    def set_auto_target_mode(self, mode: str, source: str = "click") -> None:


        if mode not in _MODES:
            return
        previous = self.auto_target_mode()
        segs = self.__dict__.get("_auto_target_segments")
        if segs:
            for key, seg in segs.items():
                seg.setChecked(key == mode)
        if mode == previous and source != "restore":


            if mode == "land_cover":
                box = self.auto_prompt_input
                if box.text().strip().lower() not in LAND_COVER_WORDS:
                    box.blockSignals(True)
                    box.setText(LAND_COVER_WORD)
                    box.blockSignals(False)
                self._apply_auto_target_look()
                self._emit_auto_prompt_committed(force=True)
            return
        self._auto_target_mode = mode
        box = self.auto_prompt_input
        if mode == "land_cover":
            typed = box.text().strip()


            if typed.lower() not in LAND_COVER_WORDS:
                self._auto_objects_text = typed
            elif source == "typed":
                self._auto_objects_text = ""
            box.blockSignals(True)
            box.setText(LAND_COVER_WORD)
            box.blockSignals(False)
        elif previous == "land_cover":
            current = box.text().strip()
            keep = current if current.lower() not in LAND_COVER_WORDS else ""
            box.blockSignals(True)
            box.setText(self.__dict__.get("_auto_objects_text") or keep)
            box.blockSignals(False)
        self._apply_auto_target_look()
        if source != "restore" or mode == "land_cover":
            self._emit_auto_prompt_committed(force=True)


        if source not in ("restore", "agent"):
            try:
                from ...core import telemetry_run_events
                telemetry_run_events.track_auto_target_switched(to=mode, source=source)
            except Exception:  # noqa: BLE001  # nosec B110
                pass

    def _apply_auto_target_look(self) -> None:

        land = self._lc_setup_active()
        try:
            self._auto_prompt_header.setText(
                tr("Map the whole zone") if land else tr("Describe what to detect"))
            self.auto_prompt_composer.setVisible(not land)
            pane = self.__dict__.get("_auto_land_cover_pane")
            if pane is not None:
                pane.setVisible(land)
                if land:
                    self._refresh_my_classes_section()
            if land:
                self.auto_prompt_info.setVisible(False)
                self.auto_prompt_tip.setVisible(False)
            in_setup = not (self._auto_run_active or self._auto_review_active)
            self.auto_exemplar_panel.setVisible(
                in_setup and not land and self._EXEMPLARS_ENABLED
                and self.auto_steps.currentIndex() == 2)
            self.auto_detail_row.setVisible(
                in_setup and not land and bool(self._auto_zone_is_set))
        except (RuntimeError, AttributeError):
            pass
        self._refresh_auto_cost_label()
