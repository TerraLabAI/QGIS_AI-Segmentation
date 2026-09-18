






from __future__ import annotations

from qgis.PyQt.QtCore import QEvent, QObject, QSize, Qt, QTimer, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...core.activation_manager import get_privacy_url, get_terms_url
from ...core.i18n import tr
from ...core.qt_compat import safe_single_shot
from ..dock.font_scale import scale_px_length, scale_qss_font_px
from ..dock.styles import (
    _BTN_SETTINGS_RAIL_CTA,
    FONT_BASE,
    FONT_HINT,
    INK_2,
    INK_3,
    category_ink,
)
from ..external_links import open_external_url
from ..icons import icon_for, logo_pixmap, logo_size
from .category_tile import NAV_GLYPH_CATEGORIES
from .settings_widgets import SAVED_HINT_QSS, SIDEBAR_QSS, nav_qss, settings_button


SETTINGS_PRODUCT_URL = (
    "https://terra-lab.ai/ai-segmentation"
    "?utm_source=qgis&utm_medium=plugin&utm_campaign=ai-segmentation"
    "&utm_content=settings_wordmark"
)
_NAV_W = 186

_NAV_MAX_W = 260



_NAV_ROW_CHROME = 76
_RAIL_CTA_MARGINS = 26
_WORDMARK_PX = 22
_SAVED_HINT_MS = 1600

_NAV_CATEGORY_ROLE = int(Qt.ItemDataRole.UserRole) + 1
_WORDMARK_QSS = scale_qss_font_px(
    f"font-size: {FONT_BASE}px; font-weight: 700; color: palette(text); background: transparent;")
_WORDMARK_NOTE_QSS = scale_qss_font_px(
    f"font-size: {FONT_HINT}px; color: {INK_2}; background: transparent;")

_FOOT_LINK_QSS = scale_qss_font_px(f"font-size: {FONT_HINT}px;")
_FOOT_LINK_H = 28
_ROW_H = 32


def _keyboard_link(label: QLabel) -> QLabel:

    label.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse
                                  | Qt.TextInteractionFlag.LinksAccessibleByKeyboard)
    label.setFocusPolicy(Qt.FocusPolicy.TabFocus)
    label.setMinimumHeight(scale_px_length(_FOOT_LINK_H))
    return label


class _RailActivationKeys(QObject):


    _KEYS = (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space)

    def __init__(self, nav: QListWidget, handler):
        super().__init__(nav)
        self._nav = nav
        self._handler = handler

    def eventFilter(self, watched, event):  # noqa: N802
        if (event.type() == QEvent.Type.KeyPress and event.key() in self._KEYS
                and not event.isAutoRepeat()):
            try:
                return bool(self._handler(self._nav.currentItem()))
            except RuntimeError:
                return False
        return False


class _WordmarkTile(QWidget):


    clicked = pyqtSignal()

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class SettingsSidebarMixin:


    def _build_sidebar(self) -> QFrame:
        side = QFrame(self)
        side.setObjectName("settingsSidebar")
        side.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        side.setStyleSheet(SIDEBAR_QSS)
        side.setFixedWidth(scale_px_length(_NAV_W))
        self._rail = side
        col = QVBoxLayout(side)
        col.setContentsMargins(0, 14, 0, 12)
        col.setSpacing(6)
        self._wordmark = self._build_wordmark(side)
        col.addWidget(self._wordmark)

        self._nav = QListWidget(side)
        self._nav.setObjectName("settingsNav")
        self._nav.setStyleSheet(nav_qss())
        self._nav.setIconSize(QSize(16, 16))
        self._nav.setFrameShape(QFrame.Shape.NoFrame)
        self._nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._nav.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._nav.setCursor(Qt.CursorShape.PointingHandCursor)
        self._nav.setUniformItemSizes(False)
        self._nav.setWordWrap(True)
        self._nav.setTextElideMode(Qt.TextElideMode.ElideNone)


        self._nav.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._nav.setAccessibleName(tr("Settings pages"))
        self._nav.currentRowChanged.connect(self._on_nav_row_changed)

        self._nav_keys = _RailActivationKeys(self._nav, self._on_nav_item_pressed_by_key)
        self._nav.installEventFilter(self._nav_keys)
        self._nav.itemClicked.connect(self._on_nav_item_clicked_again)
        col.addWidget(self._nav, 1)



        self._upgrade_pill = settings_button(tr("Do more with Pro"), _BTN_SETTINGS_RAIL_CTA, side)
        self._upgrade_pill.setSizePolicy(self._nav.sizePolicy().horizontalPolicy(),
                                         self._upgrade_pill.sizePolicy().verticalPolicy())
        self._upgrade_pill.setToolTip(tr("What Pro unlocks, on the TerraLab website."))
        self._upgrade_pill.clicked.connect(self._on_upgrade_clicked)
        self._upgrade_pill.hide()
        col.addWidget(self._upgrade_pill)

        muted = f"color: {INK_2}; text-decoration: none;"
        maker = QLabel(f'<a href="#site" style="{muted}">{tr("TerraLab")}</a>', side)
        maker.setStyleSheet(_FOOT_LINK_QSS)
        maker.setContentsMargins(16, 0, 12, 0)
        maker.setToolTip(tr("Open terra-lab.ai"))
        maker.linkActivated.connect(self._open_terralab_site)
        foot = QVBoxLayout()
        foot.setContentsMargins(0, 0, 0, 0)
        foot.setSpacing(0)
        foot.addWidget(_keyboard_link(maker))

        legal = QLabel(
            f'<a href="{get_terms_url()}" style="{muted}">{tr("Terms")}</a>'
            f' <span style="color: {INK_3};">·</span> '
            f'<a href="{get_privacy_url()}" style="{muted}">{tr("Privacy")}</a>', side)

        legal.setWordWrap(True)
        legal.setStyleSheet(_FOOT_LINK_QSS)
        legal.setContentsMargins(16, 0, 0, 0)
        legal.setOpenExternalLinks(False)
        legal.linkActivated.connect(lambda url: open_external_url(url, parent=self))
        foot.addWidget(_keyboard_link(legal))
        col.addLayout(foot)
        return side

    def _build_wordmark(self, side: QFrame) -> QWidget:
        host = _WordmarkTile(side)
        host.setCursor(Qt.CursorShape.PointingHandCursor)
        host.setToolTip(tr("Open the AI Segmentation page"))
        host.clicked.connect(lambda: open_external_url(SETTINGS_PRODUCT_URL, parent=self))
        row = QHBoxLayout(host)
        row.setContentsMargins(14, 0, 12, 6)
        row.setSpacing(8)
        height = scale_px_length(_WORDMARK_PX)
        mark = QLabel(host)
        pixmap = logo_pixmap(mark, height)
        mark.setFixedSize(logo_size(height))
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setPixmap(pixmap)
        row.addWidget(mark, 0, Qt.AlignmentFlag.AlignVCenter)
        names = QVBoxLayout()
        names.setContentsMargins(0, 0, 0, 0)
        names.setSpacing(0)
        title = QLabel(tr("AI Segmentation"), host)
        title.setStyleSheet(_WORDMARK_QSS)
        names.addWidget(title)
        maker = QLabel(tr("by TerraLab"), host)
        maker.setStyleSheet(_WORDMARK_NOTE_QSS)
        names.addWidget(maker)
        row.addLayout(names, 1)
        return host

    def _fit_rail_to_text(self) -> None:






        rail, nav, pill = self._rail, self._nav, self._upgrade_pill
        nav.ensurePolished()
        metrics = nav.fontMetrics()
        chrome = scale_px_length(_NAV_ROW_CHROME)
        texts = [nav.item(i).text() for i in range(nav.count())]
        need = max((metrics.horizontalAdvance(t) for t in texts), default=0) + chrome
        pill.ensurePolished()
        cta_margins = scale_px_length(_RAIL_CTA_MARGINS)
        need = max(need, pill.sizeHint().width() + cta_margins,
                   self._wordmark.sizeHint().width())
        width = min(max(scale_px_length(_NAV_W), need), scale_px_length(_NAV_MAX_W))
        rail.setFixedWidth(width)
        text_w = max(40, width - chrome)
        row_h = scale_px_length(_ROW_H)
        line_pad = row_h - metrics.height()
        for i in range(nav.count()):
            rect = metrics.boundingRect(0, 0, text_w, 10000,
                                        int(Qt.TextFlag.TextWordWrap), texts[i])
            nav.item(i).setSizeHint(QSize(0, max(row_h, rect.height() + line_pad)))
        if pill.sizeHint().width() + cta_margins > width:


            full = pill.text()
            room = width - cta_margins - 2 * scale_px_length(10)
            pill.setText(pill.fontMetrics().elidedText(full, Qt.TextElideMode.ElideRight, room))
            pill.setToolTip(full)

    def _open_terralab_site(self, _link: str = "") -> None:
        from ..terralab_menu import terralab_url

        open_external_url(terralab_url(), parent=self)

    def _nav_icon(self, glyph: str):

        category = NAV_GLYPH_CATEGORIES.get(glyph)
        if category:
            return icon_for(self, glyph, 16, QColor(category_ink(category)))
        return icon_for(self, glyph, 16)

    def _tint_nav_selection(self, row: int) -> None:



        for index in range(self._nav.count()):
            item = self._nav.item(index)
            if item is None:
                continue
            font = item.font()
            if font.bold() != (index == row):
                font.setBold(index == row)
                item.setFont(font)

    def _add_settings_page(self, key: str, glyph: str, label: str, page: QWidget) -> None:
        item = QListWidgetItem(self._nav_icon(glyph), label)
        item.setData(Qt.ItemDataRole.AccessibleTextRole, label)
        item.setSizeHint(QSize(0, scale_px_length(_ROW_H)))
        item.setData(Qt.ItemDataRole.UserRole, f"page:{key}")
        item.setData(_NAV_CATEGORY_ROLE, NAV_GLYPH_CATEGORIES.get(glyph, ""))
        self._nav.addItem(item)
        self._pages.addWidget(page)
        self._page_rows[key] = self._nav.count() - 1

    def _add_settings_action(self, glyph: str, label: str, kind: str) -> None:


        item = QListWidgetItem(self._nav_icon(glyph), label)
        item.setSizeHint(QSize(0, scale_px_length(_ROW_H)))
        item.setData(Qt.ItemDataRole.UserRole, f"action:{kind}")
        item.setData(Qt.ItemDataRole.AccessibleTextRole, label)
        item.setData(Qt.ItemDataRole.AccessibleDescriptionRole, tr("Opens a dialog"))


        item.setToolTip(tr("Opens a dialog"))
        self._nav.addItem(item)

    def _on_nav_row_changed(self, row: int) -> None:
        item = self._nav.item(row) if row >= 0 else None
        data = str(item.data(Qt.ItemDataRole.UserRole) or "") if item is not None else ""
        if data.startswith("action:"):
            if QApplication.mouseButtons() == Qt.MouseButton.NoButton:


                return


            action = data.split(":", 1)[1]

            self._nav_press_opened = item
            safe_single_shot(0, self, self._restore_page_row)
            safe_single_shot(0, self, lambda: self._run_rail_action(action))
            return
        if 0 <= row < self._pages.count():
            self._tint_nav_selection(row)
            self._current_page_row = row
            self._pages.setCurrentIndex(row)
            if data == "page:plugins":

                self._refresh_plugins_page()

    def _nav_action_of(self, item) -> str:
        data = str(item.data(Qt.ItemDataRole.UserRole) or "") if item is not None else ""
        return data.split(":", 1)[1] if data.startswith("action:") else ""

    def _on_nav_item_pressed_by_key(self, item) -> bool:


        action = self._nav_action_of(item)
        if action:
            safe_single_shot(0, self, self._restore_page_row)
            safe_single_shot(0, self, lambda: self._run_rail_action(action))
            return True
        return False

    def _on_nav_item_clicked_again(self, item) -> None:




        opened_by_press = getattr(self, "_nav_press_opened", None)
        self._nav_press_opened = None
        if item is opened_by_press:
            return
        if item is not None and item is self._nav.currentItem():
            action = self._nav_action_of(item)
            if action:
                safe_single_shot(0, self, self._restore_page_row)
                safe_single_shot(0, self, lambda: self._run_rail_action(action))

    def _restore_page_row(self) -> None:
        try:
            if self._nav.currentRow() != self._current_page_row:
                self._nav.blockSignals(True)
                self._nav.setCurrentRow(self._current_page_row)
                self._nav.blockSignals(False)
        except RuntimeError:
            pass  # nosec B110

    def _run_rail_action(self, kind: str) -> None:

        if kind == "contact":
            from .contact_dialog import show_contact_dialog

            show_contact_dialog(self)
        elif kind == "report":
            from ..error_report_dialog import show_error_report

            show_error_report(self, tr("Report a problem"), "", error_code="user_reported")

    def show_settings_page(self, key: str) -> None:


        row = self._page_rows.get(str(key))
        if row is not None:
            self._nav.setCurrentRow(row)

    def _show_saved(self, text: str = "") -> None:

        hint = self._saved_hint
        try:
            hint.setText(text or tr("Saved"))
            hint.adjustSize()

            hint.move(self._right.width() - hint.width() - scale_px_length(24),
                      scale_px_length(26))
            hint.raise_()
            hint.show()
            self._saved_timer.start(_SAVED_HINT_MS)
        except RuntimeError:
            pass  # nosec B110

    def _build_saved_hint(self, right: QWidget) -> None:
        self._saved_hint = QLabel(tr("Saved"), right)
        self._saved_hint.setStyleSheet(SAVED_HINT_QSS)
        self._saved_hint.hide()
        self._saved_timer = QTimer(self)
        self._saved_timer.setSingleShot(True)
        self._saved_timer.timeout.connect(self._saved_hint.hide)

    def _sync_upgrade_pill(self, free_account: bool) -> None:
        pill = getattr(self, "_upgrade_pill", None)
        if pill is not None:
            pill.setVisible(bool(free_account))


__all__ = ["SETTINGS_PRODUCT_URL", "SettingsSidebarMixin"]
