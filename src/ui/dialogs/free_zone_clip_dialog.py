













from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ...core.free_clip_choice import DETECT_CLIPPED, DISMISSED, DRAW_SMALLER, UPGRADE
from ...core.i18n import tr
from ..dock.font_scale import scale_qss_font_px
from ..dock.styles import (
    _BTN_GHOST,
    _BTN_LINK,
    _BTN_PRIMARY,
    FONT_BASE,
    FONT_BODY,
    INK,
    INK_2,
    SURFACE,
)

_DIALOG_QSS = f"QDialog#freeZoneClipDialog {{ background: {SURFACE}; }}"
_TITLE_QSS = (f"font-size: {FONT_BASE + 2}px; font-weight: 600; color: {INK};"
              " background: transparent;")
_BODY_QSS = f"font-size: {FONT_BODY}px; color: {INK_2}; background: transparent;"


class FreeZoneClipDialog(QDialog):


    def __init__(self, zone_km2: str, left_km2: str, done_km2: str,
                 on_answer=None, parent=None):
        super().__init__(parent)
        self._on_answer = on_answer
        self.setObjectName("freeZoneClipDialog")



        self.setWindowTitle("AI Segmentation")
        self.setStyleSheet(_DIALOG_QSS)
        self.setModal(True)
        self.chosen = DISMISSED

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 18)
        layout.setSpacing(8)




        title = QLabel(tr(
            "Your {zone} km² zone is larger than this month's free surface."
        ).format(zone=zone_km2))
        title.setWordWrap(True)
        title.setStyleSheet(scale_qss_font_px(_TITLE_QSS))
        layout.addWidget(title)

        body = QLabel(tr(
            "We can detect on the {done} km² outlined on the map "
            "(center of your zone)."
        ).format(done=done_km2))
        body.setWordWrap(True)
        body.setStyleSheet(scale_qss_font_px(_BODY_QSS))
        layout.addWidget(body)
        layout.addSpacing(8)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.detect_btn = QPushButton(
            tr("Detect on {done} km²").format(done=done_km2))
        self.detect_btn.setObjectName("freeClipDetectBtn")
        self.detect_btn.setStyleSheet(scale_qss_font_px(_BTN_PRIMARY))
        self.detect_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.detect_btn.setDefault(True)
        self.detect_btn.clicked.connect(lambda: self._answer(DETECT_CLIPPED))
        self.draw_btn = QPushButton(tr("Draw a smaller zone"))
        self.draw_btn.setObjectName("freeClipDrawBtn")
        self.draw_btn.setStyleSheet(scale_qss_font_px(_BTN_GHOST))
        self.draw_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.draw_btn.setAutoDefault(False)
        self.draw_btn.clicked.connect(lambda: self._answer(DRAW_SMALLER))
        row.addWidget(self.detect_btn, 1)
        row.addWidget(self.draw_btn, 1)
        layout.addLayout(row)

        self.upgrade_btn = QPushButton(
            tr("Run the whole zone with Pro"))
        self.upgrade_btn.setObjectName("freeClipUpgradeLink")
        self.upgrade_btn.setStyleSheet(scale_qss_font_px(_BTN_LINK))
        self.upgrade_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.upgrade_btn.setAutoDefault(False)
        self.upgrade_btn.clicked.connect(lambda: self._answer(UPGRADE))

        layout.addWidget(self.upgrade_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        self.setMinimumWidth(420)

    def done(self, result: int) -> None:
        super().done(result)
        callback, self._on_answer = self._on_answer, None
        if callback is not None:
            callback(self.chosen)

    def _answer(self, choice: str) -> None:
        self.chosen = choice
        if choice == DETECT_CLIPPED:
            self.accept()
        else:
            self.reject()
