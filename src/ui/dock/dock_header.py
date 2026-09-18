





















from __future__ import annotations

from qgis.PyQt.QtCore import QEvent, QSize, Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
from ...core.qt_compat import event_pos
from ..icons import icon_for, logo_pixmap, logo_size
from .font_scale import scale_px_length, scale_qss_font_px
from .styles import (
    ACCENT_BORDER,
    ACCENT_TINT,
    BRAND_BLUE,
    FONT_BODY,
    FONT_MICRO,
    HOVER,
    HOVER_ON,
    INK,
    INK_2,
    LINE,
    LINE_STRONG,
    RADIUS_CONTROL,
    SPACE_CARD,
    category_ink,
)

HEADER_HEIGHT = 36


_MARK_PX = 20
_MARK_GAP = 6
_GLYPH_PX = 18

_BUTTON_PX = 28





_PRO_PILL_PX = 24
_PRO_GLYPH_PX = 14
_WIDGET_MAX = 16777215


_PRODUCT_PAGE_URL = (
    "https://terra-lab.ai/ai-segmentation?utm_source=qgis&utm_medium=plugin"
    "&utm_campaign=ai-segmentation&utm_content=title_link")





_BYLINE_PX = FONT_MICRO



_HEADER_V_MARGIN = 2


_DOCK_HEADER_QSS = scale_qss_font_px(
    "QWidget#segHeader { background: transparent;"
    f" border-bottom: 1px solid {LINE}; }}"
    f"QLabel#segHeaderTitle {{ font-size: {FONT_BODY}px; font-weight: 600;"
    f" color: {INK}; background: transparent; border: none; }}"
    f"QLabel#segHeaderByline {{ font-size: {_BYLINE_PX}px;"
    f" color: {INK_2}; background: transparent; border: none; }}"
    "QLabel { background: transparent; border: none; }"
)




_HEADER_ICON_BTN_QSS = (
    "QToolButton { background: transparent; border: none; padding: 3px;"
    f" border-radius: {RADIUS_CONTROL}px; }}"
    f"QToolButton:hover {{ background: {HOVER}; }}"
    f'QToolButton[hover="true"] {{ background: {HOVER}; }}'
    f"QToolButton:pressed {{ background: {HOVER_ON}; }}"
    f'QToolButton[active="true"] {{ background: {ACCENT_TINT}; }}'
    "QToolButton:disabled { background: transparent; }"
    "QToolButton::menu-indicator { image: none; width: 0; }"
    f"QToolButton:focus {{ border: 2px solid {ACCENT_BORDER}; }}"
)


def _pro_pill_qss(height: int) -> str:

    return scale_qss_font_px(
        "QPushButton#segProPill { background: transparent;"
        f" color: {INK}; border: 1px solid {LINE_STRONG};"
        f" border-radius: {height // 2}px; padding: 0 10px 0 8px;"
        f" font-size: {FONT_BODY}px; font-weight: 600; }}"
        f"QPushButton#segProPill:hover {{ background: {HOVER}; }}"
        f"QPushButton#segProPill:pressed {{ background: {HOVER_ON}; }}"
        f"QPushButton#segProPill:focus {{ border: 2px solid {ACCENT_BORDER};"
        " padding: 0 9px 0 7px; }"
        'QPushButton#segProPill[compact="true"] { padding: 0; }'
        'QPushButton#segProPill[compact="true"]:focus { padding: 0; }'
    )


class _ElidedTitle(QLabel):



    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self._full_text = text
        self.setMinimumWidth(0)

    def minimumSizeHint(self):  # noqa: N802
        hint = super().minimumSizeHint()
        try:
            return QSize(self.fontMetrics().horizontalAdvance("AI\u2026"), hint.height())
        except (RuntimeError, AttributeError):
            return hint

    def sizeHint(self):  # noqa: N802
        hint = super().sizeHint()
        try:
            width = self.fontMetrics().horizontalAdvance(self._full_text) + 2
            return QSize(width, hint.height())
        except (RuntimeError, AttributeError):
            return hint

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        try:
            shown = self.fontMetrics().elidedText(
                self._full_text, Qt.TextElideMode.ElideRight, self.width())
            if shown != self.text():
                super().setText(shown)
            self.setToolTip("" if shown == self._full_text else self._full_text)
        except (RuntimeError, AttributeError):
            pass  # nosec B110


class _HeaderBrandTile(QWidget):


    clicked = pyqtSignal()

    def mouseReleaseEvent(self, event):  # noqa: N802
        try:
            inside = self.rect().contains(event_pos(event))
        except (TypeError, AttributeError):
            inside = True
        if event.button() == Qt.MouseButton.LeftButton and inside:
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class HeaderIconButton(QToolButton):






    def __init__(self, parent, glyph: str, tooltip: str, accessible_name: str = ""):
        super().__init__(parent)
        self._glyph = glyph
        self._hovering = False
        self.setProperty("hover", False)
        self.setProperty("active", False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setAutoRaise(True)
        self.setStyleSheet(_HEADER_ICON_BTN_QSS)
        self.setFixedSize(_BUTTON_PX, _BUTTON_PX)
        self.setIconSize(QSize(_GLYPH_PX, _GLYPH_PX))
        self.setToolTip(tooltip)
        self.setAccessibleName(accessible_name or tooltip)
        self._repaint_header_glyph()

    def _repaint_header_glyph(self) -> None:
        try:
            color = QColor(BRAND_BLUE) if self._hovering else None
            self.setIcon(icon_for(self, self._glyph, _GLYPH_PX, color))
        except (RuntimeError, AttributeError, TypeError):
            pass  # nosec B110

    def enterEvent(self, event):  # noqa: N802
        super().enterEvent(event)
        self._set_glyph_hover(True)

    def leaveEvent(self, event):  # noqa: N802
        super().leaveEvent(event)
        self._set_glyph_hover(False)
        self.set_hovered(False)

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        try:
            if event.type() == QEvent.Type.PaletteChange:
                self._repaint_header_glyph()
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _set_glyph_hover(self, hovering: bool) -> None:
        if self._hovering == hovering or not self.isEnabled():
            return
        self._hovering = hovering
        self._repaint_header_glyph()

    def _repolish_header_button(self) -> None:
        style = self.style()
        style.unpolish(self)
        style.polish(self)
        self.update()

    def set_hovered(self, hovered: bool) -> None:
        if bool(self.property("hover")) == hovered:
            return
        self.setProperty("hover", hovered)
        self._repolish_header_button()

    def set_active(self, active: bool) -> None:

        if bool(self.property("active")) == active:
            return
        self.setProperty("active", active)
        self._repolish_header_button()

    def attach_menu(self, menu: QMenu) -> None:

        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setMenu(menu)
        menu.aboutToShow.connect(lambda: self.set_active(True))
        menu.aboutToHide.connect(self._on_header_menu_closed)

    def _on_header_menu_closed(self) -> None:
        self.setDown(False)
        self.set_hovered(False)
        self.set_active(False)
        self._set_glyph_hover(self.underMouse())


class DockHeader(QWidget):


    pro_pill_clicked = pyqtSignal()
    settings_clicked = pyqtSignal()
    float_clicked = pyqtSignal()
    close_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("segHeader")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(_DOCK_HEADER_QSS)


        self.setFixedHeight(scale_px_length(HEADER_HEIGHT))
        row = QHBoxLayout(self)
        row.setContentsMargins(12, _HEADER_V_MARGIN, 8, _HEADER_V_MARGIN)

        row.setSpacing(SPACE_CARD)

        brand_host = _HeaderBrandTile(self)
        brand_host.setCursor(Qt.CursorShape.PointingHandCursor)
        brand_host.setToolTip(tr("Open the AI Segmentation page"))
        brand_host.clicked.connect(self._open_product_page)
        brand = QHBoxLayout(brand_host)
        brand.setContentsMargins(0, 0, 0, 0)
        brand.setSpacing(_MARK_GAP)
        self._logo = QLabel(brand_host)
        self._logo.setFixedSize(logo_size(_MARK_PX))
        self._logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._logo.setPixmap(logo_pixmap(self, _MARK_PX))
        brand.addWidget(self._logo, 0, Qt.AlignmentFlag.AlignVCenter)
        names = QVBoxLayout()
        names.setContentsMargins(0, 0, 0, 0)
        names.setSpacing(0)
        self._title = _ElidedTitle(tr("AI Segmentation"), brand_host)
        self._title.setObjectName("segHeaderTitle")
        names.addWidget(self._title)
        self._byline = _ElidedTitle(tr("by TerraLab"), brand_host)
        self._byline.setObjectName("segHeaderByline")
        names.addWidget(self._byline)
        brand.addLayout(names)
        brand.setAlignment(names, Qt.AlignmentFlag.AlignVCenter)
        self.brand_tile = brand_host


        row.addWidget(brand_host, 0)
        row.addStretch(1)


        self.pro_pill = self._build_pro_pill()
        row.addWidget(self.pro_pill, 0, Qt.AlignmentFlag.AlignVCenter)

        self.settings_btn = HeaderIconButton(self, "gear", tr("Settings"))
        self.settings_btn.clicked.connect(self.settings_clicked.emit)
        row.addWidget(self.settings_btn, 0, Qt.AlignmentFlag.AlignVCenter)



        self.float_btn = HeaderIconButton(
            self, "float_window", tr("Dock or undock this panel"))
        self.float_btn.clicked.connect(self.float_clicked.emit)
        row.addWidget(self.float_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        self.close_btn = HeaderIconButton(self, "close", tr("Close"))
        self.close_btn.clicked.connect(self.close_clicked.emit)
        row.addWidget(self.close_btn, 0, Qt.AlignmentFlag.AlignVCenter)



    def _build_pro_pill(self) -> QPushButton:
        pill = QPushButton(tr("Get Pro"), self)
        pill.setObjectName("segProPill")
        height = scale_px_length(_PRO_PILL_PX)
        pill.setStyleSheet(_pro_pill_qss(height))
        pill.setFixedHeight(height)
        pill.setIcon(icon_for(pill, "gem", _PRO_GLYPH_PX,
                              QColor(category_ink("amber"))))
        pill.setIconSize(QSize(_PRO_GLYPH_PX, _PRO_GLYPH_PX))
        pill.setCursor(Qt.CursorShape.PointingHandCursor)

        pill.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        pill.setAutoDefault(False)
        pill.setToolTip(tr("See what Pro unlocks"))
        pill.setAccessibleName(tr("See what Pro unlocks"))
        pill.setProperty("compact", False)
        pill.clicked.connect(self.pro_pill_clicked.emit)
        pill.setVisible(False)
        self._pro_pill_wanted = False

        self._pro_pill_full_width = pill.sizeHint().width()
        return pill

    def set_pro_pill_visible(self, visible: bool) -> None:

        visible = bool(visible)
        if visible == self._pro_pill_wanted and self.pro_pill.isHidden() != visible:
            return
        self._pro_pill_wanted = visible
        self.pro_pill.setVisible(visible)
        self._fit_pro_pill()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._fit_pro_pill()

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        self._fit_pro_pill()

    def _fit_pro_pill(self) -> None:







        try:
            if not self._pro_pill_wanted:
                return
            row = self.layout()
            margins = row.contentsMargins()
            others = [w for w in (self.brand_tile, self.settings_btn,
                                  self.float_btn, self.close_btn)
                      if not w.isHidden()]
            used = margins.left() + margins.right()
            used += sum((w.minimumSizeHint() if w is self.brand_tile
                         else w.sizeHint()).width() for w in others)
            used += row.spacing() * len(others)
            compact = used + self._pro_pill_full_width > self.width()
            pill = self.pro_pill
            if bool(pill.property("compact")) == compact:
                return
            pill.setProperty("compact", compact)
            if compact:
                pill.setText("")
                pill.setFixedWidth(pill.height())
            else:
                pill.setText(tr("Get Pro"))
                pill.setMinimumWidth(0)
                pill.setMaximumWidth(_WIDGET_MAX)
            style = pill.style()
            style.unpolish(pill)
            style.polish(pill)
            pill.update()
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _open_product_page(self) -> None:

        from ..external_links import open_external_url

        open_external_url(_PRODUCT_PAGE_URL, parent=self)
