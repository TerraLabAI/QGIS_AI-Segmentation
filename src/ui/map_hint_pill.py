








from __future__ import annotations

from qgis.PyQt.QtCore import QEvent, QObject, QTimer
from qgis.PyQt.QtWidgets import QLabel

from ..core.qt_compat import AlignCenter, WA_TransparentForMouseEvents
from .dock.styles import FONT_BODY, INK, LINE_STRONG, SURFACE


_TOP_GAP = 12

FLASH_MS = 2200


class MapHintPill(QObject):


    def __init__(self, canvas):
        super().__init__(canvas)
        self._canvas = canvas
        self._label: QLabel | None = None
        self._state_text = ""
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._end_flash)
        canvas.installEventFilter(self)


        self._message_bar = self._find_message_bar()
        if self._message_bar is not None:
            self._message_bar.installEventFilter(self)



    def show_state(self, text: str, end_flash: bool = True) -> None:




        if end_flash and text != self._state_text:
            self._flash_timer.stop()
        self._state_text = text
        if not self._flash_timer.isActive():
            self._render(text)

    def flash(self, text: str) -> None:
        self._render(text)
        self._flash_timer.start(FLASH_MS)

    def text(self) -> str:

        if self._label is None or not self._label.isVisible():
            return ""
        return self._label.text()

    def dispose(self) -> None:
        self._flash_timer.stop()
        for watched in (self._canvas, self._message_bar):
            if watched is None:
                continue
            try:
                watched.removeEventFilter(self)
            except RuntimeError:
                pass
        self._message_bar = None
        if self._label is not None:
            try:
                self._label.hide()
                self._label.deleteLater()
            except RuntimeError:
                pass
            self._label = None


        try:
            self.deleteLater()
        except RuntimeError:
            pass



    def _end_flash(self) -> None:
        self._render(self._state_text)

    def _render(self, text: str) -> None:
        if not text:
            if self._label is not None:
                self._label.hide()
            return
        label = self._ensure_label()
        label.setText(text)
        label.adjustSize()
        self._place()
        label.show()
        label.raise_()

    def _ensure_label(self) -> QLabel:
        if self._label is None:
            label = QLabel(self._canvas)
            label.setAttribute(WA_TransparentForMouseEvents, True)
            label.setAlignment(AlignCenter)


            label.setStyleSheet(
                f"QLabel {{ background: {SURFACE}; color: {INK};"
                f" border: 1px solid {LINE_STRONG}; border-radius: 14px;"
                f" font-size: {FONT_BODY}px; padding: 5px 14px; }}"
            )
            self._label = label
        return self._label

    def _place(self) -> None:
        if self._label is None:
            return
        viewport = self._canvas.viewport()
        geo = viewport.geometry() if viewport is not None else self._canvas.rect()
        x = geo.x() + max(0, (geo.width() - self._label.width()) // 2)
        top = geo.y()
        bar = self._message_bar
        try:
            if bar is not None and bar.isVisible():
                bottom = bar.mapTo(self._canvas.window(), bar.rect().bottomLeft())
                bottom = self._canvas.mapFrom(self._canvas.window(), bottom)
                if 0 <= bottom.y() < geo.height() // 2:
                    top = max(top, bottom.y())
        except (RuntimeError, TypeError):
            pass
        self._label.move(x, top + _TOP_GAP)

    def _find_message_bar(self):
        try:
            from qgis.utils import iface
            return iface.messageBar() if iface is not None else None
        except (ImportError, AttributeError, RuntimeError):
            return None

    def eventFilter(self, obj, event):  # noqa: N802
        kind = event.type()
        on_canvas = obj is self._canvas and kind == QEvent.Type.Resize
        on_bar = obj is self._message_bar and kind in (
            QEvent.Type.Show, QEvent.Type.Hide, QEvent.Type.Resize
        )
        if on_canvas or on_bar:
            self._place()
        return False
