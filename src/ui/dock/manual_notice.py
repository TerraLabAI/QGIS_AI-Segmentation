












from __future__ import annotations

from qgis.PyQt.QtCore import QTimer

from ...core.server_dials import dial_in_range
from .styles import _msg_label_qss, msg_rich



_NOTICE_MS = 12000

_NOTICE_BAR_SECONDS = 10


class DockManualNoticeMixin:


    def show_manual_notice(self, text: str) -> None:






        if not text:
            return
        label = getattr(self, "instructions_label", None)
        try:
            usable = label is not None and label.isVisibleTo(self)
        except RuntimeError:
            usable = False
        if not usable:
            self._push_notice_to_message_bar(text)
            return
        self._manual_notice_text = text
        self._paint_manual_notice(label, text)
        timer = self._manual_notice_timer()
        timer.start(dial_in_range(
            "tuning.manual.notice_ms", _NOTICE_MS, 3000, 60000))

    def clear_manual_notice(self) -> None:





        if not getattr(self, "_manual_notice_text", ""):
            return
        self._manual_notice_text = ""
        timer = getattr(self, "_manual_notice_qtimer", None)
        if timer is not None:
            try:
                timer.stop()
            except RuntimeError:  # nosec B110
                pass
        try:
            self._update_instructions()
        except (RuntimeError, AttributeError):  # nosec B110
            pass

    def manual_notice_is_live(self) -> bool:

        return bool(getattr(self, "_manual_notice_text", ""))

    def _paint_manual_notice(self, label, text: str) -> None:

        self._instructions_style = "notice"
        label.setStyleSheet(_msg_label_qss("error_transient"))
        label.setMinimumHeight(0)


        from qgis.PyQt.QtCore import Qt
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setText(msg_rich("error", text))
        label.setVisible(True)

    def _manual_notice_timer(self) -> QTimer:






        timer = getattr(self, "_manual_notice_qtimer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self.clear_manual_notice)
            self._manual_notice_qtimer = timer
        return timer

    def _push_notice_to_message_bar(self, text: str) -> None:

        try:
            from qgis.core import Qgis
            from qgis.utils import iface as _iface

            _iface.messageBar().pushMessage(
                "AI Segmentation", text,
                level=Qgis.MessageLevel.Warning,
                duration=dial_in_range(
                    "tuning.manual.notice_bar_seconds",
                    _NOTICE_BAR_SECONDS, 3, 60))
        except Exception:  # noqa: BLE001
            pass  # nosec B110
