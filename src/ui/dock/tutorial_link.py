








from __future__ import annotations

from qgis.PyQt.QtCore import QSize, Qt
from qgis.PyQt.QtWidgets import QHBoxLayout, QPushButton, QWidget

from ...core.i18n import tr
from .font_scale import scale_qss_font_px
from .styles import (
    ACCENT_BORDER,
    ACCENT_TINT,
    ACCENT_TINT_ON,
    FONT_BODY,
    INK,
    MUTED,
    RADIUS_CONTROL,
)

_TUTORIAL_GLYPH_PX = 16
_TUTORIAL_ROW_PX = 28





_TUTORIAL_LINK_QSS = scale_qss_font_px(
    "QPushButton#segTutorialLink { background: transparent;"
    f" color: {MUTED}; border: 2px solid transparent;"
    f" border-radius: {RADIUS_CONTROL}px;"
    f" padding: 2px 10px; font-size: {FONT_BODY}px; }}"
    f"QPushButton#segTutorialLink:hover {{ background: {ACCENT_TINT}; color: {INK}; }}"
    f"QPushButton#segTutorialLink:pressed {{ background: {ACCENT_TINT_ON}; }}"
    f"QPushButton#segTutorialLink:focus {{ border-color: {ACCENT_BORDER}; }}"
)


def build_home_tutorial_link(on_click, name: str) -> QWidget:


    row = QWidget()
    row.setObjectName(name)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    button = QPushButton(tr("Tutorial"), row)
    button.setObjectName("segTutorialLink")
    button.setStyleSheet(_TUTORIAL_LINK_QSS)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setAutoDefault(False)
    button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
    button.setToolTip(tr("Open the step-by-step tutorial"))
    button.setAccessibleName(tr("Tutorial"))
    button.setMinimumHeight(_TUTORIAL_ROW_PX)
    try:
        from qgis.PyQt.QtGui import QColor

        from ..icons import icon_for



        from .styles import HUE_TUTORIAL, category_ink

        button.setIcon(icon_for(button, "play", _TUTORIAL_GLYPH_PX,
                                QColor(category_ink(HUE_TUTORIAL))))
        button.setIconSize(QSize(_TUTORIAL_GLYPH_PX, _TUTORIAL_GLYPH_PX))
    except (RuntimeError, AttributeError, TypeError):
        pass  # nosec B110
    button.clicked.connect(on_click)
    layout.addStretch(1)
    layout.addWidget(button)
    layout.addStretch(1)
    return row


def add_home_quiet_link(row: QWidget, text: str, glyph: str, on_click) -> QPushButton:



    button = QPushButton(text, row)
    button.setObjectName("segTutorialLink")
    button.setStyleSheet(_TUTORIAL_LINK_QSS)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setAutoDefault(False)
    button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
    button.setAccessibleName(text)
    button.setMinimumHeight(_TUTORIAL_ROW_PX)
    try:
        from qgis.PyQt.QtGui import QColor

        from ..icons import icon_for

        button.setIcon(icon_for(button, glyph, _TUTORIAL_GLYPH_PX, QColor(MUTED)))
        button.setIconSize(QSize(_TUTORIAL_GLYPH_PX, _TUTORIAL_GLYPH_PX))
    except (RuntimeError, AttributeError, TypeError):
        pass  # nosec B110
    button.clicked.connect(on_click)
    layout = row.layout()
    layout.insertWidget(max(0, layout.count() - 1), button)
    return button
