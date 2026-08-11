














from __future__ import annotations

from qgis.PyQt.QtCore import QSize
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ....core import qt_compat as QtC
from ....core.i18n import tr
from ...dock.font_scale import scale_px_length
from ...dock.styles import INK, INK_2, LINE
from .common import (
    _ARROW_BTN_QSS,
    _PAGE_LINE_QSS,
    _PAGE_TITLE_QSS,
    _SEARCH_PILL_QSS,
    _SHELF_TITLE_QSS,
    _TEXT_LINK_QSS,
    _category_glyph,
)

_SEARCH_W = 260
_TILE_GAP = 16
_PER_PAGE = 3
_SHELF_GLYPH_PX = 18
_ARROW_PX = 32


def _glyph_label(parent: QWidget, name: str, size: int, colour: str) -> QLabel:

    label = QLabel(parent)
    label.setStyleSheet("background: transparent; border: none;")
    label.setFixedSize(size, size)
    label.setAttribute(QtC.WA_TransparentForMouseEvents)
    try:
        from ...icons import pixmap_for
        label.setPixmap(pixmap_for(label, name, size, QColor(colour)))
    except (ImportError, RuntimeError, TypeError, AttributeError):
        pass  # nosec B110
    return label


def style_search_pill(field) -> None:

    field.setStyleSheet(_SEARCH_PILL_QSS)
    field.setFixedWidth(scale_px_length(_SEARCH_W))
    glyph = _glyph_label(field, "search", 16, INK_2)
    glyph.move(12, max(0, (scale_px_length(34) - 16) // 2))


class LibraryPageHeader(QWidget):


    def __init__(self, right: QWidget | None = None, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(16)
        words = QVBoxLayout()
        words.setContentsMargins(0, 0, 0, 0)
        words.setSpacing(4)
        self._title_row = QHBoxLayout()
        self._title_row.setContentsMargins(0, 0, 0, 0)
        self._title_row.setSpacing(10)
        self._glyph: QLabel | None = None
        self.title = QLabel(self)
        self.title.setTextFormat(QtC.PlainText)
        self.title.setStyleSheet(_PAGE_TITLE_QSS)
        self._title_row.addWidget(self.title, 1, QtC.AlignVCenter)
        words.addLayout(self._title_row)
        self.subtitle = QLabel(self)
        self.subtitle.setTextFormat(QtC.PlainText)
        self.subtitle.setWordWrap(True)
        self.subtitle.setStyleSheet(_PAGE_LINE_QSS)
        words.addWidget(self.subtitle)
        row.addLayout(words, 1)
        if right is not None:
            row.addWidget(right, 0, QtC.AlignTop)

    def set_text(self, title: str, subtitle: str = "", glyph: str = "") -> None:


        self.title.setText(str(title or ""))
        self.subtitle.setText(str(subtitle or ""))
        self.subtitle.setVisible(bool(subtitle))
        if self._glyph is not None:
            self._glyph.hide()
            self._glyph.setParent(None)
            self._glyph.deleteLater()
            self._glyph = None
        if glyph:
            self._glyph = _glyph_label(self, glyph, scale_px_length(22), INK)
            self._title_row.insertWidget(0, self._glyph, 0, QtC.AlignVCenter)

    def set_subtitle(self, subtitle: str) -> None:
        self.subtitle.setText(str(subtitle or ""))
        self.subtitle.setVisible(bool(subtitle))


class _ArrowButton(QPushButton):


    def __init__(self, glyph: str, tooltip: str, parent=None):
        super().__init__(parent)
        side = scale_px_length(_ARROW_PX)
        self.setFixedSize(side, side)
        self.setCursor(QtC.PointingHandCursor)
        self.setAutoDefault(False)
        self.setFocusPolicy(QtC.NoFocus)
        self.setToolTip(tooltip)
        self.setAccessibleName(tooltip)
        try:
            from ...icons import icon_for
            self.setIcon(icon_for(self, glyph, 16, QColor(INK),
                                  disabled_color=QColor(LINE)))
            self.setIconSize(QSize(16, 16))
        except (ImportError, RuntimeError, TypeError, AttributeError):
            self.setText("<" if "left" in glyph else ">")
        self.setStyleSheet(_ARROW_BTN_QSS)


class _ShelfScroll(QScrollArea):


    def wheelEvent(self, event):  # noqa: N802
        try:
            delta = event.angleDelta()
            if abs(delta.x()) > abs(delta.y()):
                super().wheelEvent(event)
                return
            event.ignore()
        except RuntimeError:
            pass  # nosec B110


class LibraryShelf(QWidget):







    def __init__(self, key: str, title: str, tiles: list, see_all, on_scrolled,
                 parent=None):
        super().__init__(parent)
        self.key = key
        self.tiles = list(tiles)
        self._width = 0
        self._step = 0
        self._on_scrolled = on_scrolled
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 8)
        col.setSpacing(12)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        head.addWidget(_glyph_label(self, _category_glyph(key),
                                    scale_px_length(_SHELF_GLYPH_PX), INK),
                       0, QtC.AlignVCenter)
        name = QLabel(str(title or ""), self)
        name.setTextFormat(QtC.PlainText)
        name.setStyleSheet(_SHELF_TITLE_QSS)
        head.addWidget(name, 0, QtC.AlignVCenter)
        head.addStretch(1)
        more = QPushButton(tr("See all"), self)
        more.setCursor(QtC.PointingHandCursor)
        more.setAutoDefault(False)
        more.setFocusPolicy(QtC.NoFocus)
        more.setStyleSheet(_TEXT_LINK_QSS)
        more.clicked.connect(lambda _c=False: see_all())
        head.addWidget(more, 0, QtC.AlignVCenter)
        self._prev = _ArrowButton("chevron_left", tr("Previous"), self)
        self._prev.clicked.connect(self.previous_page)
        self._next = _ArrowButton("chevron_right", tr("Next"), self)
        self._next.clicked.connect(self.next_page)
        head.addWidget(self._prev, 0, QtC.AlignVCenter)
        head.addWidget(self._next, 0, QtC.AlignVCenter)
        col.addLayout(head)

        self._scroll = _ShelfScroll(self)
        self._scroll.setFrameShape(QtC.FrameNoFrame)
        self._scroll.setWidgetResizable(False)
        self._scroll.setHorizontalScrollBarPolicy(QtC.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(QtC.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }")
        self._strip = QWidget()
        self._strip.setStyleSheet("background: transparent;")
        row = QHBoxLayout(self._strip)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(scale_px_length(_TILE_GAP))
        for tile in self.tiles:
            tile.setParent(self._strip)
            row.addWidget(tile)
        row.addStretch(1)
        self._scroll.setWidget(self._strip)
        self._scroll.viewport().setStyleSheet("background: transparent;")
        col.addWidget(self._scroll)
        bar = self._scroll.horizontalScrollBar()
        bar.valueChanged.connect(self._scrolled)
        bar.rangeChanged.connect(self._arrows)
        self._arrows()

    def viewport(self):
        return self._scroll.viewport()

    def set_width(self, width: int) -> None:

        if width <= 0 or width == self._width:
            return
        self._width = width
        gap = scale_px_length(_TILE_GAP)
        tile_w = max(1, (width - (_PER_PAGE - 1) * gap) // _PER_PAGE)
        height = 0
        for tile in self.tiles:
            tile.set_tile_width(tile_w)
            height = max(height, tile.sizeHint().height())
        count = len(self.tiles)
        strip_w = count * tile_w + max(0, count - 1) * gap
        for tile in self.tiles:
            tile.setFixedHeight(height)
        self._strip.setFixedSize(max(strip_w, width), height)
        self._scroll.setFixedHeight(height)
        self._step = tile_w + gap
        self._arrows()

    def resizeEvent(self, ev):  # noqa: N802
        super().resizeEvent(ev)
        try:
            self.set_width(self.width())
        except RuntimeError:
            pass  # nosec B110

    def _scrolled(self, *_args) -> None:
        self._arrows()
        try:
            self._on_scrolled()
        except RuntimeError:
            pass  # nosec B110

    def _arrows(self, *_args) -> None:
        bar = self._scroll.horizontalScrollBar()
        more = len(self.tiles) > _PER_PAGE
        self._prev.setVisible(more)
        self._next.setVisible(more)
        self._prev.setEnabled(bar.value() > bar.minimum())
        self._next.setEnabled(bar.value() < bar.maximum())

    def next_page(self) -> None:
        bar = self._scroll.horizontalScrollBar()
        bar.setValue(self._snap(bar.value() + _PER_PAGE * self._step))

    def previous_page(self) -> None:
        bar = self._scroll.horizontalScrollBar()
        bar.setValue(self._snap(bar.value() - _PER_PAGE * self._step))

    def _snap(self, value: int) -> int:

        step = self._step or 1
        bar = self._scroll.horizontalScrollBar()
        return max(0, min(bar.maximum(), round(value / step) * step))


__all__ = ["LibraryPageHeader", "LibraryShelf", "style_search_pill"]
