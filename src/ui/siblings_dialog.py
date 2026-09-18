






























from __future__ import annotations

import os

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QPixmap
from qgis.PyQt.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..core.i18n import tr
from ..core.server_dials import dial_copy
from ..core.surface_dials import cross_sell_ai_edit_enabled
from .cross_plugin_discovery import (
    SIBLINGS,
    is_sibling_installed,
    open_sibling,
    open_sibling_tutorial,
    sibling_field_url,
)
from .dock.font_scale import apply_font_scale_to_tree, scale_px_length, scale_qss_font_px
from .dock.styles import (
    _BTN_SETTINGS_GHOST,
    _BTN_SETTINGS_STEP,
    _BTN_SETTINGS_TEXT,
    FONT_BASE,
    FONT_BODY,
    FONT_HINT,
    INK,
    INK_2,
    LINE,
    RADIUS_CARD,
    SURFACE,
    apply_quiet_scrollbar,
)
from .external_links import open_external_url
from .settings.a11y import DANGER_INK, WARN_INK, accessible_qss, make_accessible
from .settings.settings_widgets import TITLE_PX, ButtonFlow, ElidedLabel
from .sibling_thumbnails import (
    _ALIGN_LEFT,
    _ALIGN_TOP,
    _ALIGN_VCENTER,
    _CARD_MIN_W,
    _POINTING_HAND,
    _SIZE_EXPANDING,
    _SIZE_PREFERRED,
    _SMOOTH,
    SiblingShot,
    _enum,
    load_shot_image,
)

_ICON_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "resources", "icons")


def _logo_path(file_name: str) -> str:
    return os.path.join(_ICON_DIR, file_name) if file_name else ""


def _open_more_url() -> None:


    from .terralab_menu import terralab_url

    open_external_url(terralab_url())


def _card_copy():








    cards = [
        ("ai-agent", dial_copy("siblings.ai_agent_body", tr(
            "Run QGIS from a sentence."))),
    ]
    if cross_sell_ai_edit_enabled():
        cards.append(("ai-edit", dial_copy("siblings.ai_edit_body", tr(
            "Repaint your imagery from a sentence."))))
    return tuple(cards)


_LOGO_PX = 24




_CARD_MAX_W = 460
_CARD_PADDING = 14


_CARD_PADDING_TOP = 10
_DIALOG_MARGIN = 16
_GRID_SPACING = 12


_CARD_RADIUS = RADIUS_CARD

_CARD_QSS = scale_qss_font_px(
    f"QFrame#siblingCard {{ background: {SURFACE}; border: 1px solid {LINE};"
    f" border-radius: {_CARD_RADIUS}px; }}"
    "QFrame#siblingCard QLabel { background: transparent; border: none; }"
    f"QLabel#siblingName {{ font-size: {FONT_BASE}px; font-weight: 600; color: palette(text); }}"
    f"QLabel#siblingNote {{ font-size: {FONT_BODY}px; color: {INK_2}; }}"
    f"QLabel#siblingState {{ font-size: {FONT_HINT}px; color: {INK_2}; }}"
    f"QLabel#siblingProblem {{ font-size: {FONT_HINT}px; color: {DANGER_INK}; }}"
)
_DIALOG_QSS = scale_qss_font_px(
    "QDialog#siblingsDialog { background: palette(window); }"
    f"QLabel#siblingsTitle {{ font-size: {TITLE_PX}px; font-weight: 600; color: {INK}; }}"
)






STATE_OPEN, STATE_UPDATE, STATE_ENABLE, STATE_RESTART, STATE_INSTALL = (
    "open", "update", "enable", "restart", "install")


def current_sibling_state(product_id: str) -> str:
    from . import cross_plugin_discovery as discovery

    reader = getattr(discovery, "sibling_state", None)
    if reader is not None:
        try:
            return str(reader(product_id))
        except Exception:  # noqa: BLE001
            pass  # nosec B110
    return STATE_OPEN if is_sibling_installed(product_id) else STATE_INSTALL


def take_sibling_action(product_id: str) -> str:
    from . import cross_plugin_discovery as discovery

    runner = getattr(discovery, "run_sibling_action", None)
    if runner is not None:
        return str(runner(product_id) or "")
    return str(open_sibling(product_id) or "")


def _action_copy(state: str):

    if state == STATE_OPEN:
        return (tr("Open in QGIS"), tr("Show the plugin's panel."), _BTN_SETTINGS_GHOST, True)
    if state == STATE_UPDATE:
        return (tr("Update"),
                tr("A newer version is ready. Opens the QGIS plugin manager on it."),
                _BTN_SETTINGS_STEP, True)
    if state == STATE_ENABLE:
        return (tr("Turn on"),
                tr("It is installed but switched off. Turns it on and opens it."),
                _BTN_SETTINGS_STEP, True)
    if state == STATE_RESTART:
        return (tr("Restart QGIS"),
                tr("It is installed but did not start. Restart QGIS to use it."),
                _BTN_SETTINGS_GHOST, False)
    return (tr("Install in QGIS"), tr("Opens the QGIS plugin manager on this plugin."),
            _BTN_SETTINGS_STEP, True)


def _state_copy(state: str) -> str:
    return {
        STATE_OPEN: tr("Installed"),
        STATE_UPDATE: tr("Update available"),
        STATE_ENABLE: tr("Turned off"),
        STATE_RESTART: tr("Needs a restart"),
    }.get(state, "")


class SiblingCard(QFrame):



    install_requested = pyqtSignal(str)
    tutorial_requested = pyqtSignal(str)

    def __init__(self, product_id: str, note: str, parent=None):
        super().__init__(parent)
        self.product_id = str(product_id)
        self.state = STATE_INSTALL
        sibling = SIBLINGS.get(self.product_id, {})
        self.setObjectName("siblingCard")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(_CARD_QSS)
        self.setSizePolicy(_SIZE_EXPANDING, _SIZE_PREFERRED)
        self.setMaximumWidth(scale_px_length(_CARD_MAX_W))

        outer = QVBoxLayout(self)

        outer.setContentsMargins(1, 1, 1, 0)
        outer.setSpacing(0)
        logo_path = _logo_path(str(sibling.get("icon") or ""))
        self._shot = SiblingShot(logo_path, self)
        outer.addWidget(self._shot)

        column = QVBoxLayout()
        column.setContentsMargins(_CARD_PADDING, _CARD_PADDING_TOP,
                                  _CARD_PADDING, _CARD_PADDING)
        column.setSpacing(8)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)

        if logo_path and os.path.isfile(logo_path):
            chip = QPixmap(logo_path)
            if not chip.isNull():
                mark = QLabel(self)
                mark.setPixmap(chip.scaled(_LOGO_PX, _LOGO_PX,
                                           _enum(Qt, "AspectRatioMode", "KeepAspectRatio"),
                                           _SMOOTH))
                head.addWidget(mark, 0, _ALIGN_VCENTER)
        name = QLabel(str(sibling.get("label") or ""), self)
        name.setObjectName("siblingName")
        head.addWidget(name, 0, _ALIGN_VCENTER)


        self._state = ElidedLabel("", self, Qt.TextElideMode.ElideRight)
        self._state.setObjectName("siblingState")
        head.addWidget(self._state, 1, _ALIGN_VCENTER)
        column.addLayout(head)

        body = QLabel(note, self)
        body.setObjectName("siblingNote")
        body.setWordWrap(True)
        body.setAlignment(_ALIGN_TOP | _ALIGN_LEFT)
        column.addWidget(body, 1)

        self._problem = QLabel(
            tr("Turn it on in Plugins > Manage and Install Plugins."), self)
        self._problem.setObjectName("siblingProblem")
        self._problem.setWordWrap(True)
        self._problem.hide()
        column.addWidget(self._problem)


        actions = ButtonFlow(self, spacing=6)
        actions.setContentsMargins(0, 4, 0, 0)
        self._action = QPushButton(self)
        self._action.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._action.setCursor(_POINTING_HAND)
        self._action.setAutoDefault(False)
        self._action.clicked.connect(
            lambda: self.install_requested.emit(self.product_id))
        actions.add(self._action)

        guide = QPushButton(tr("Read the guide"), self)
        make_accessible(guide, _BTN_SETTINGS_TEXT)
        guide.setCursor(_POINTING_HAND)
        guide.setAutoDefault(False)
        guide.setToolTip(tr("The written tutorial, on the TerraLab blog."))
        guide.clicked.connect(lambda: self.tutorial_requested.emit(self.product_id))
        actions.add(guide)
        column.addWidget(actions, 0, _ALIGN_LEFT)
        outer.addLayout(column, 1)

        self._loader = load_shot_image(
            self._shot, self, sibling_field_url(self.product_id, "thumbnail_url"))
        self.refresh()

    def refresh(self) -> None:

        self.state = current_sibling_state(self.product_id)
        label, tooltip, style, enabled = _action_copy(self.state)
        self._action.setText(label)
        self._action.setToolTip(tooltip)
        self._action.setStyleSheet(accessible_qss(style))


        self._action.setEnabled(enabled)
        self._action.setVisible(enabled)
        self._state.set_full_text(_state_copy(self.state))
        if not enabled:
            self._state.setToolTip(tooltip)
        warn = self.state in (STATE_UPDATE, STATE_RESTART)
        self._state.setStyleSheet(f"color: {WARN_INK};" if warn else "")
        if self.state != STATE_ENABLE:
            self._problem.hide()

    def show_problem(self) -> None:
        self._problem.show()


class SiblingCardGrid(QWidget):






    opened = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(_GRID_SPACING)
        self._cards = [SiblingCard(product_id, note, self)
                       for product_id, note in _card_copy()]
        for card in self._cards:
            card.install_requested.connect(self._on_install)
            card.tutorial_requested.connect(self._on_tutorial)
        self._columns = 0
        self.reflow_cards(2)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        room = event.size().width()

        card_w = max([scale_px_length(_CARD_MIN_W)]
                     + [card.minimumSizeHint().width() for card in self._cards])
        self.reflow_cards(2 if room >= 2 * card_w + _GRID_SPACING else 1)

    def reflow_cards(self, columns: int) -> None:
        if columns == self._columns:
            return
        self._columns = columns
        for index, card in enumerate(self._cards):


            self._grid.addWidget(card, index // columns, index % columns)
        for index in range(2):
            self._grid.setColumnStretch(index, 1 if index < columns else 0)

    def refresh(self) -> None:
        for card in self._cards:
            card.refresh()

    def _on_install(self, product_id: str) -> None:



        outcome = take_sibling_action(product_id)


        self.refresh()
        if outcome == "enable_failed":
            for card in self._cards:
                if card.product_id == product_id:
                    card.show_problem()
        self.opened.emit(product_id, outcome)

    def _on_tutorial(self, product_id: str) -> None:
        open_sibling_tutorial(product_id)
        self.opened.emit(product_id, "guide")


class SiblingsDialog(QDialog):


    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("siblingsDialog")
        self.setStyleSheet(_DIALOG_QSS)



        self.setWindowTitle(tr("More plugins"))
        self.setModal(False)
        self.setMinimumWidth(scale_px_length(_CARD_MIN_W) + 2 * _DIALOG_MARGIN)

        column = QVBoxLayout(self)
        column.setContentsMargins(20, 18, 20, 14)
        column.setSpacing(12)
        title = QLabel(tr("More plugins"), self)
        title.setObjectName("siblingsTitle")
        column.addWidget(title)



        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        apply_quiet_scrollbar(scroll, " QScrollArea, QScrollArea > QWidget > QWidget"
                                      " { background: transparent; border: none; }")
        holder = QWidget(scroll)
        holder_col = QVBoxLayout(holder)
        holder_col.setContentsMargins(0, 0, 0, 0)
        self._cards = SiblingCardGrid(holder)
        holder_col.addWidget(self._cards)
        holder_col.addStretch(1)
        scroll.setWidget(holder)
        column.addWidget(scroll, 1)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(6)
        more = QPushButton(tr("Everything we make"), self)
        make_accessible(more, _BTN_SETTINGS_TEXT)
        more.setCursor(_POINTING_HAND)
        more.setAutoDefault(False)
        more.clicked.connect(_open_more_url)
        footer.addWidget(more)
        footer.addStretch(1)
        close = QPushButton(tr("Close"), self)
        make_accessible(close, _BTN_SETTINGS_GHOST)
        close.setCursor(_POINTING_HAND)
        close.setAutoDefault(False)
        close.clicked.connect(self.close)
        footer.addWidget(close)
        column.addLayout(footer)

        apply_font_scale_to_tree(self)
        self.resize(2 * scale_px_length(_CARD_MAX_W) + _GRID_SPACING + 40,
                    scale_px_length(560))

    def refresh(self) -> None:
        self._cards.refresh()


_OPEN_DIALOG = None


def show_siblings_dialog(parent=None) -> SiblingsDialog:





    global _OPEN_DIALOG  # noqa: PLW0603
    try:
        if _OPEN_DIALOG is not None and _OPEN_DIALOG.isVisible():
            _OPEN_DIALOG.refresh()
            _OPEN_DIALOG.raise_()
            _OPEN_DIALOG.activateWindow()
            return _OPEN_DIALOG
    except RuntimeError:
        _OPEN_DIALOG = None
    _OPEN_DIALOG = SiblingsDialog(parent)
    _OPEN_DIALOG.show()
    _OPEN_DIALOG.raise_()
    _OPEN_DIALOG.activateWindow()
    return _OPEN_DIALOG
