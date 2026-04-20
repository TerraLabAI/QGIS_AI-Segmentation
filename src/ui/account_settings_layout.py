





from __future__ import annotations

from qgis.PyQt.QtCore import QRect



_SCREEN_MARGIN_PX = 48


_SETTINGS_W, _SETTINGS_H = 960, 700
_SETTINGS_MIN_W, _SETTINGS_MIN_H = 640, 460


class AccountLayoutMixin:


    def _fit_to_screen(self) -> None:






        from .dock.font_scale import scale_px_length

        width, height = scale_px_length(_SETTINGS_W), scale_px_length(_SETTINGS_H)
        min_w, min_h = scale_px_length(_SETTINGS_MIN_W), scale_px_length(_SETTINGS_MIN_H)
        available = self._available_screen_rect()
        if available is not None:
            frame_extra = max(0, self.frameGeometry().height() - self.height())
            cap_w = max(320, available.width() - _SCREEN_MARGIN_PX)
            cap_h = max(320, available.height() - frame_extra - _SCREEN_MARGIN_PX)
            width, height = min(width, cap_w), min(height, cap_h)
            min_w, min_h = min(min_w, cap_w), min(min_h, cap_h)
        self.setMinimumSize(min_w, min_h)
        self.resize(width, height)
        if available is None:
            return


        frame = self.frameGeometry()
        frame.moveCenter(self._centre_anchor(available))
        left = min(max(available.left(), frame.left()),
                   max(available.left(), available.right() - frame.width()))
        top = min(max(available.top(), frame.top()),
                  max(available.top(), available.bottom() - frame.height()))
        self.move(left, top)

    def _centre_anchor(self, available: QRect):

        try:
            parent = self.parentWidget()
            if parent is not None and parent.isVisible():
                centre = parent.frameGeometry().center()
                if available.contains(centre):
                    return centre
        except (AttributeError, RuntimeError):
            pass  # nosec B110
        return available.center()

    def _available_screen_rect(self) -> QRect | None:

        try:
            screen = self.screen()
        except (AttributeError, RuntimeError):
            screen = None
        if screen is None:
            from qgis.PyQt.QtGui import QGuiApplication

            screen = QGuiApplication.primaryScreen()
        if screen is None:
            return None
        available = screen.availableGeometry()
        if available.isEmpty():
            return None
        return available
