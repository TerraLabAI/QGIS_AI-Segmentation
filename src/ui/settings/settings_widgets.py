






from __future__ import annotations

from qgis.PyQt.QtCore import QRectF, QSize, Qt
from qgis.PyQt.QtGui import QColor, QPainter, QPalette, QPen
from qgis.PyQt.QtWidgets import (
    QAbstractButton,
    QBoxLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..dock.font_scale import scale_px_length, scale_qss_font_px
from ..dock.styles import (
    _SCROLL_AREA_QSS,
    BRAND_BLUE,
    FIELD,
    FONT_BASE,
    FONT_BODY,
    FONT_HINT,
    HOVER,
    HOVER_ON,
    INK,
    INK_2,
    LINE,
    RADIUS_CARD,
    RADIUS_ROW,
    SURFACE,
    category_fill,
    category_ink,
    category_line,
    category_tint,
    gauge_category,
)
from .a11y import DANGER_INK, FOCUS_RING, SWITCH_OFF_ALPHA, make_accessible




TITLE_PX = FONT_BASE + 3
PAGE_TITLE_QSS = scale_qss_font_px(
    f"font-size: {TITLE_PX}px; font-weight: 600; color: palette(text); background: transparent;")
PAGE_SUBTITLE_QSS = scale_qss_font_px(
    f"font-size: {FONT_BODY}px; color: {INK_2}; background: transparent;")
ROW_TITLE_QSS = scale_qss_font_px(
    f"font-size: {FONT_BASE}px; color: palette(text); background: transparent;")
ROW_TITLE_STRONG_QSS = scale_qss_font_px(
    f"font-size: {FONT_BASE}px; font-weight: 600; color: palette(text); background: transparent;")
ROW_NOTE_QSS = scale_qss_font_px(
    f"font-size: {FONT_HINT}px; color: {INK_2}; background: transparent;")
ROW_ERROR_QSS = scale_qss_font_px(
    f"font-size: {FONT_HINT}px; color: {DANGER_INK}; background: transparent;")
GROUP_TITLE_QSS = scale_qss_font_px(
    f"font-size: {FONT_BODY}px; font-weight: 600; color: palette(text); background: transparent;")
SAVED_HINT_QSS = scale_qss_font_px(
    f"font-size: {FONT_HINT}px; color: {INK_2}; background: transparent;")



_GROUP_QSS = (
    f"QFrame#settingsGroup {{ background: {SURFACE}; border: 1px solid {LINE};"
    f" border-radius: {RADIUS_CARD}px; }}"
    "QFrame#settingsGroup QFrame#settingsRow { border: none; background: transparent; }"
    "QFrame#settingsGroup QLabel { background: transparent; border: none; }"
)
_ROW_DIVIDER_QSS = f"QFrame {{ background: {LINE}; border: none; }}"




SCROLL_QSS = _SCROLL_AREA_QSS


SIDEBAR_QSS = (
    f"QFrame#settingsSidebar {{ background: {FIELD}; border: none;"
    f" border-right: 1px solid {LINE}; }}"
    "QFrame#settingsSidebar QLabel { background: transparent; border: none; }"
)


def nav_qss(category: str | None = None) -> str:








    del category
    return scale_qss_font_px(
        "QListWidget#settingsNav { background: transparent; border: none; outline: none;"
        f" padding: 4px 6px; font-size: {FONT_BASE}px; }}"


        f"QListWidget#settingsNav::item {{ padding: 5px 6px; border-radius: {RADIUS_ROW}px;"
        " border: 2px solid transparent; border-left: 3px solid transparent;"
        " color: palette(text); margin: 1px 0; }"

        f"QListWidget#settingsNav::item:focus {{ border-color: {FOCUS_RING}; }}"
        f"QListWidget#settingsNav::item:hover {{ background: {HOVER}; }}"
        f"QListWidget#settingsNav::item:selected {{ background: {HOVER_ON};"
        f" border-left: 3px solid {BRAND_BLUE}; color: palette(text); }}"
    )


NAV_QSS = nav_qss()


WINDOW_QSS = "QDialog#AISegmentationSettings { background: palette(window); }"

_PROGRESS_QSS = (
    "QProgressBar { background: rgba(128,128,128,0.18); border: none; border-radius: 3px; }"
    "QProgressBar::chunk { background: %s; border-radius: 3px; }"
)

_BILLING_CARD_QSS = scale_qss_font_px(
    f"QFrame#billingCard {{ background: {SURFACE}; border: 1px solid {LINE};"
    f" border-radius: {RADIUS_CARD}px; }}"
    "QFrame#billingCard QLabel { background: transparent; border: none; }"


    f"QLabel#billingName {{ font-size: {FONT_BASE}px; font-weight: 600; color: {INK}; }}"
    f"QLabel#billingStat {{ font-size: 26px; font-weight: 600; color: {INK}; }}"
    f"QLabel#billingStatus {{ font-size: {FONT_BODY}px; color: {INK_2}; }}"
    f"QLabel#billingNote {{ font-size: {FONT_HINT}px; color: {INK_2}; }}"
)

_BILLING_STACK_W = 720
_BILLING_CARD_MIN_W = 240
_SWITCH_W, _SWITCH_H = 38, 22


def _faint_tint(category: str) -> str:

    tint = category_tint(category)
    head, _, alpha = tint.rstrip(")").rpartition(",")
    try:
        return f"{head}, {float(alpha) / 2:.3f})"
    except ValueError:
        return tint


def balance_bar(parent: QWidget, left: int, total: int, accessible: str = "") -> QProgressBar:



    bar = QProgressBar(parent)
    total = max(int(total), 1)
    left = max(0, min(int(left), total))
    bar.setRange(0, total)
    bar.setValue(left)
    bar.setTextVisible(False)
    bar.setFixedHeight(6)
    qss = _PROGRESS_QSS % category_fill(gauge_category(left / total))
    if left <= 0:

        qss += f"QProgressBar {{ background: {category_tint('coral', strong=True)}; }}"
    bar.setStyleSheet(qss)
    if accessible:
        bar.setAccessibleName(accessible)
    return bar


def settings_button(text: str, qss: str, parent: QWidget | None = None) -> QPushButton:


    button = QPushButton(text, parent)
    make_accessible(button, qss)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setAutoDefault(False)
    button.setDefault(False)
    button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return button


def muted_label(text: str, parent: QWidget | None = None) -> QLabel:
    label = QLabel(text, parent)
    label.setStyleSheet(ROW_NOTE_QSS)
    label.setWordWrap(True)
    label.setContentsMargins(4, 4, 4, 4)
    return label


class ElidedLabel(QLabel):



    def __init__(self, text: str, parent=None,
                 mode: Qt.TextElideMode = Qt.TextElideMode.ElideMiddle):
        super().__init__(parent)
        self._mode = mode
        self._full = str(text or "")
        self.setToolTip(self._full)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(self._full)

    def set_full_text(self, text: str) -> None:
        self._full = str(text or "")
        self.setToolTip(self._full)
        self.setText(self._full)
        self._elide()

    def minimumSizeHint(self):  # noqa: N802
        hint = super().minimumSizeHint()
        hint.setWidth(0)
        return hint

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        try:
            if self.width() <= 1:
                return
            self.ensurePolished()
            elided = self.fontMetrics().elidedText(self._full, self._mode, self.width())
            if elided != self.text():
                self.setText(elided)
        except RuntimeError:
            pass  # nosec B110


def clear_layout(layout) -> None:






    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.hide()
            widget.setParent(None)
            widget.deleteLater()
        elif item.layout() is not None:
            clear_layout(item.layout())


class SettingSwitch(QAbstractButton):


    def __init__(self, parent=None, checked: bool = False):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(bool(checked))
        self.setCursor(Qt.CursorShape.PointingHandCursor)


        self.setFixedSize(scale_px_length(_SWITCH_W), scale_px_length(_SWITCH_H))
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            on = self.isChecked()

            ink = QColor(self.palette().color(QPalette.ColorRole.WindowText))
            track = QColor(BRAND_BLUE) if on else QColor(ink)
            if not on:
                track.setAlphaF(SWITCH_OFF_ALPHA)
            if not self.isEnabled():
                track.setAlphaF(0.25)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(track)
            painter.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)
            knob_d = rect.height() - 6
            x = rect.right() - 3 - knob_d if on else rect.left() + 3
            knob = QColor("#ffffff") if on else QColor(ink)
            if not on:
                knob.setAlphaF(0.9)
            painter.setBrush(knob)
            painter.drawEllipse(QRectF(x, rect.top() + 3, knob_d, knob_d))
            if self.hasFocus():

                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(FOCUS_RING), 2))
                ring = QRectF(self.rect()).adjusted(1, 1, -1, -1)
                painter.drawRoundedRect(ring, ring.height() / 2, ring.height() / 2)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        finally:
            painter.end()



_WORDS_FLOOR_W = 180


class ButtonFlow(QWidget):







    def __init__(self, parent=None, spacing: int = 8):
        super().__init__(parent)
        self._box = QBoxLayout(QBoxLayout.Direction.LeftToRight, self)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._box.setSpacing(spacing)
        self._box.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

    def add(self, button: QWidget) -> QWidget:
        button.setParent(self)
        self._box.addWidget(button)
        return button

    def _buttons(self) -> list:
        items = (self._box.itemAt(i).widget() for i in range(self._box.count()))
        return [w for w in items if w is not None and not w.isHidden()]

    def row_width(self) -> int:
        buttons = self._buttons()
        return (sum(b.sizeHint().width() for b in buttons)
                + self._box.spacing() * max(0, len(buttons) - 1))

    def minimumSizeHint(self):  # noqa: N802





        try:
            hint = super().minimumSizeHint()
            buttons = self._buttons()
            if buttons:
                hint.setWidth(max(b.minimumSizeHint().width() for b in buttons))
            return hint
        except RuntimeError:
            return QSize(0, 0)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        direction = (QBoxLayout.Direction.LeftToRight
                     if event.size().width() >= self.row_width()
                     else QBoxLayout.Direction.TopToBottom)
        if self._box.direction() != direction:
            self._box.setDirection(direction)
            self.updateGeometry()


class SettingRow(QFrame):








    def __init__(self, title: str, note: str = "", control: QWidget | None = None,
                 parent=None, lead: QWidget | None = None, title_label: QLabel | None = None):


        super().__init__(parent)
        self.setObjectName("settingsRow")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        outer = QBoxLayout(QBoxLayout.Direction.LeftToRight, self)
        outer.setContentsMargins(14, 10, 14, 10)
        outer.setSpacing(16)
        self._outer = outer
        self._control = control
        words = QVBoxLayout()
        words.setContentsMargins(0, 0, 0, 0)
        words.setSpacing(2)
        if title_label is None:
            title_label = QLabel(title, self)
            title_label.setStyleSheet(ROW_TITLE_QSS)
            title_label.setWordWrap(True)
        title_label.setParent(self)
        self.title_label = title_label
        words.addWidget(self.title_label)
        self.note_label = QLabel(note, self)
        self.note_label.setStyleSheet(ROW_NOTE_QSS)
        self.note_label.setWordWrap(True)
        self.note_label.setVisible(bool(note))
        words.addWidget(self.note_label)
        self.words = words
        if lead is not None:
            lead.setParent(self)
            head = QHBoxLayout()
            head.setContentsMargins(0, 0, 0, 0)
            head.setSpacing(12)
            head.addWidget(lead, 0, Qt.AlignmentFlag.AlignVCenter)
            head.addLayout(words, 1)
            outer.addLayout(head, 1)
        else:
            outer.addLayout(words, 1)
        if control is not None:
            control.setParent(self)
            outer.addWidget(control, 0, Qt.AlignmentFlag.AlignVCenter)
            if not control.accessibleName():
                control.setAccessibleName(title)
            if note and not control.accessibleDescription():
                control.setAccessibleDescription(note)

    def set_note(self, note: str) -> None:
        self.note_label.setText(note)
        self.note_label.setVisible(bool(note))

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        control = self._control
        if control is None or isinstance(control, SettingSwitch):
            return
        wide = control.row_width() if isinstance(control, ButtonFlow) else control.sizeHint().width()
        margins = self._outer.contentsMargins()
        need = (margins.left() + margins.right() + self._outer.spacing() + wide
                + scale_px_length(_WORDS_FLOOR_W))
        stacked = event.size().width() < need
        direction = (QBoxLayout.Direction.TopToBottom if stacked
                     else QBoxLayout.Direction.LeftToRight)
        if self._outer.direction() != direction:
            self._outer.setDirection(direction)
            self._outer.setSpacing(8 if stacked else 16)


            below = (Qt.AlignmentFlag(0) if isinstance(control, ButtonFlow)
                     else Qt.AlignmentFlag.AlignLeft)
            self._outer.setAlignment(control, below if stacked
                                     else Qt.AlignmentFlag.AlignVCenter)
            self.updateGeometry()


class SettingGroup(QFrame):





    def __init__(self, parent=None, category: str | None = None):
        super().__init__(parent)
        self.setObjectName("settingsGroup")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        qss = _GROUP_QSS
        if category:
            qss += (f"QFrame#settingsGroup {{ background: {_faint_tint(category)};"
                    f" border-color: {category_line(category)}; }}")
        self._divider_qss = (f"QFrame {{ background: {category_line(category)}; border: none; }}"
                             if category else _ROW_DIVIDER_QSS)
        self.setStyleSheet(qss)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self._col = QVBoxLayout(self)
        self._col.setContentsMargins(0, 0, 0, 0)
        self._col.setSpacing(0)

    def add_row(self, row: QWidget) -> QWidget:
        if self._col.count():
            line = QFrame(self)
            line.setStyleSheet(self._divider_qss)
            line.setFixedHeight(1)
            self._col.addWidget(line)
        row.setParent(self)
        self._col.addWidget(row)
        return row


class SettingsPage(QWidget):





    def __init__(self, title: str, subtitle: str, parent=None,
                 glyph: str = "", category: str | None = None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        head = QVBoxLayout()
        head.setContentsMargins(24, 22, 24, 12)
        head.setSpacing(3)
        self.title_label = QLabel(title, self)
        self.title_label.setStyleSheet(PAGE_TITLE_QSS)
        self.title_label.setWordWrap(True)
        if glyph:
            from .category_tile import category_icon_tile, tile_beside

            self.title_tile = category_icon_tile(glyph, category, self)
            head.addLayout(tile_beside(self.title_tile, self.title_label))
            head.addSpacing(4)
        else:
            head.addWidget(self.title_label)
        self.subtitle_label = QLabel(subtitle, self)
        self.subtitle_label.setStyleSheet(PAGE_SUBTITLE_QSS)
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setVisible(bool(subtitle))
        head.addWidget(self.subtitle_label)
        outer.addLayout(head)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet(SCROLL_QSS)
        self.body_widget = QWidget(self.scroll)
        self.body = QVBoxLayout(self.body_widget)
        self.body.setContentsMargins(24, 4, 24, 20)
        self.body.setSpacing(14)
        self.body.addStretch(1)
        self.scroll.setWidget(self.body_widget)
        outer.addWidget(self.scroll, 1)

    def add(self, widget: QWidget) -> QWidget:

        widget.setParent(self.body_widget)
        self.body.insertWidget(self.body.count() - 1, widget)
        return widget

    def add_group_title(self, text: str) -> QLabel:
        label = QLabel(text, self.body_widget)
        label.setStyleSheet(GROUP_TITLE_QSS)
        label.setWordWrap(True)
        label.setContentsMargins(2, 4, 0, 0)
        return self.add(label)

    def add_box(self) -> QVBoxLayout:

        box = QWidget(self.body_widget)
        col = QVBoxLayout(box)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(14)
        self.add(box)
        return col


class BillingCard(QFrame):




    def __init__(self, title: str, parent=None, category: str | None = None):
        super().__init__(parent)
        self.setObjectName("billingCard")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        qss = _BILLING_CARD_QSS
        if category:
            ink = category_ink(category)
            qss += f"QLabel#billingStat {{ color: {ink}; }}"
        self.setStyleSheet(qss)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(scale_px_length(_BILLING_CARD_MIN_W))
        self.column = QVBoxLayout(self)
        self.column.setContentsMargins(16, 14, 16, 14)
        self.column.setSpacing(6)
        name = QLabel(title, self)
        name.setObjectName("billingName")
        name.setWordWrap(True)
        self.column.addWidget(name)


        self.column.addStretch(1)

    def add(self, widget: QWidget) -> QWidget:
        widget.setParent(self)
        self.column.insertWidget(self.column.count() - 1, widget)
        return widget

    def _add_label(self, text: str, name: str) -> QLabel:
        label = QLabel(text, self)
        label.setObjectName(name)
        label.setWordWrap(True)
        return self.add(label)

    def add_stat(self, value: str, caption: str = "") -> QLabel:

        label = self._add_label(value, "billingStat")
        if caption:
            self.add_status(caption)
        return label

    def add_status(self, text: str) -> QLabel:
        return self._add_label(text, "billingStatus")

    def add_note(self, text: str) -> QLabel:
        return self._add_label(text, "billingNote")

    def add_button(self, button: QPushButton) -> QPushButton:

        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        holder = QWidget(self)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 6, 0, 0)
        row.addWidget(button, 1)
        self.add(holder)
        return button


class BillingCardRow(QWidget):


    def __init__(self, parent=None):
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(12)
        self._cards: list = []
        self._stacked = False

    def add_card(self, card: QWidget) -> QWidget:
        self._cards.append(card)
        card.setParent(self)
        self._place_cards()
        return card

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        window = self.window()
        width = window.width() if window is not None else self.width()


        widest = max((card.minimumSizeHint().width() for card in self._cards), default=0)
        side_by_side = 2 * widest + self._grid.horizontalSpacing()
        stacked = (int(width) < scale_px_length(_BILLING_STACK_W)
                   or self.width() < side_by_side)
        if stacked != self._stacked:
            self._stacked = stacked
            self._place_cards()

    def _place_cards(self) -> None:
        for index, card in enumerate(self._cards):
            self._grid.removeWidget(card)
            if self._stacked:
                self._grid.addWidget(card, index, 0)
            else:



                self._grid.addWidget(card, 0, index, Qt.AlignmentFlag.AlignTop)
        columns = 1 if self._stacked else max(len(self._cards), 1)
        for column in range(max(len(self._cards), 1)):
            self._grid.setColumnStretch(column, 1 if column < columns else 0)
