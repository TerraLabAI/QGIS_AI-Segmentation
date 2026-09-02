





from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ...core.i18n import tr
from .font_scale import scale_qss_font_px
from .styles import _CARD_MARGINS, _CARD_QSS, FONT_BASE, FONT_HINT, INK, INK_2



RUN_REFERENCE_PX = 88


class AutoRunSummaryCard(QWidget):








    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("autoRunSummaryCard")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            _CARD_QSS.format(name="autoRunSummaryCard")
            + scale_qss_font_px(
                f"QLabel#autoRunPromptCaption {{ color: {INK_2};"
                f" font-size: {FONT_HINT}px; background: transparent; border: none; }}"
                f"QLabel#autoRunPromptWord {{ color: {INK}; font-weight: 600;"
                f" font-size: {FONT_BASE + 3}px; background: transparent; border: none; }}"))
        self.setAccessibleName(tr("What you asked for"))
        _col = QVBoxLayout(self)
        _col.setContentsMargins(*_CARD_MARGINS)
        _col.setSpacing(6)
        _caption = QLabel(tr("What you asked for"))
        _caption.setObjectName("autoRunPromptCaption")
        _col.addWidget(_caption)
        self._summary_word = QLabel("")
        self._summary_word.setObjectName("autoRunPromptWord")
        self._summary_word.setTextFormat(Qt.TextFormat.PlainText)
        self._summary_word.setWordWrap(True)
        self._summary_word.setVisible(False)
        _col.addWidget(self._summary_word)



        self._summary_thumbs = QWidget()
        self._summary_thumbs_layout = QHBoxLayout(self._summary_thumbs)
        self._summary_thumbs_layout.setContentsMargins(0, 2, 0, 0)
        self._summary_thumbs_layout.setSpacing(8)
        self._summary_thumbs_layout.addStretch()
        self._summary_thumbs.setVisible(False)
        _col.addWidget(self._summary_thumbs)
        self.setVisible(False)

    def set_run_recipe(self, word: str, chips: list) -> None:





        while self._summary_thumbs_layout.count() > 1:
            item = self._summary_thumbs_layout.takeAt(0)
            old = item.widget()
            if old is not None:
                old.setParent(None)
                old.deleteLater()
        word = (word or "").strip()
        self._summary_word.setText(word)
        self._summary_word.setVisible(bool(word))
        for i, chip in enumerate(chips):
            self._summary_thumbs_layout.insertWidget(i, chip)
        self._summary_thumbs.setVisible(bool(chips))
