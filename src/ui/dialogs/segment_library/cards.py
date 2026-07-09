








from __future__ import annotations

from qgis.PyQt.QtCore import QPoint, Qt, pyqtSignal
from qgis.PyQt.QtGui import QPixmap
from qgis.PyQt.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ....core import qt_compat as QtC
from ....core.i18n import tr
from ....core.presets.segmentation_presets import pick_label
from ...before_after_slider import BeforeAfterSlider
from ...dock.font_scale import scale_px_length, scale_qss_font_px
from ...dock.styles import (
    BTN_SMALL_PX,
    FONT_BASE,
    FONT_BODY,
    HOVER,
    HOVER_ON,
    INK,
    INK_3,
    RADIUS_CARD,
    RADIUS_CONTROL,
)
from ...template_demo_loader import TemplateDemoLoader
from .common import (
    _CARD_HOVER,
    _CARD_NORMAL,
    _META_QSS,
    _STAR_BTN_QSS,
    _build_use_hint,
    _demo_url,
    _set_star_glyph,
    _set_use_hint,
)





_RECENT_ACTION_QSS = scale_qss_font_px(
    f"QPushButton {{ background: transparent; color: {INK};"
    f" border: none; border-radius: {RADIUS_CONTROL}px;"
    f" padding: 5px 8px; font-weight: 500; font-size: {FONT_BODY}px;"
    " text-align: left; }"
    f"QPushButton:hover {{ background: {HOVER}; }}"
    f"QPushButton:pressed {{ background: {HOVER_ON}; }}"
    f"QPushButton:disabled {{ color: {INK_3}; }}"
)

_TITLE_QSS = scale_qss_font_px(
    f"font-size: {FONT_BASE}px; font-weight: 600; color: {INK};"
    " background: transparent; border: none;"
)

_PREVIEW_H = 175

_CARD_MIN_W = 200


def _card_min_width() -> int:

    return scale_px_length(_CARD_MIN_W)


def _released_on_card(card: QWidget, ev) -> bool:





    try:
        if ev.button() != Qt.MouseButton.LeftButton:
            return False
        pt = QtC.event_pos(ev)
        return card.rect().contains(QPoint(int(pt.x()), int(pt.y())))
    except (RuntimeError, AttributeError, TypeError):
        return False


def _preview_height() -> int:





    return scale_px_length(_PREVIEW_H)


class _ElidedTitle(QLabel):






    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self._full = str(text or "")
        self.setTextFormat(QtC.PlainText)
        self.setStyleSheet(_TITLE_QSS)
        self.setToolTip(self._full)
        self.setMinimumWidth(24)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(self._full)

    def resizeEvent(self, ev):  # noqa: N802
        super().resizeEvent(ev)
        try:
            elided = self.fontMetrics().elidedText(
                self._full, Qt.TextElideMode.ElideRight, max(24, self.width()))
            if elided != self.text():
                self.setText(elided)
        except (RuntimeError, AttributeError, TypeError):
            pass  # nosec B110


class _PresetCard(QFrame):






    activated = pyqtSignal(dict)
    star_toggled = pyqtSignal(dict, bool)

    def __init__(self, preset: dict, parent=None):
        super().__init__(parent)
        self._preset = preset
        self._missing: set[str] = set()
        self._requested = False
        self.setObjectName("card")
        self.setStyleSheet(_CARD_NORMAL)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumWidth(_card_min_width())
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.slider = BeforeAfterSlider(
            self, auto_loop=False, show_badges=False, handle_grab_only=True)
        self.slider.setFixedHeight(_preview_height())


        self.slider.set_card_corners(RADIUS_CARD, top_only=True)
        self.slider.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.slider.set_placeholder_text(tr("Loading..."))



        self.slider.clicked.connect(self._fire)
        lay.addWidget(self.slider)

        footer_wrap = QWidget(self)
        footer = QHBoxLayout(footer_wrap)
        footer.setContentsMargins(12, 8, 8, 8)
        footer.setSpacing(2)
        title = _ElidedTitle(
            pick_label(preset.get("label"), preset.get("prompt", "")), footer_wrap)
        footer.addWidget(title, 1)
        self.star_btn = QToolButton()
        self.star_btn.setCheckable(True)
        self.star_btn.setStyleSheet(_STAR_BTN_QSS)
        self.star_btn.setFixedSize(scale_px_length(BTN_SMALL_PX),
                                   scale_px_length(BTN_SMALL_PX))
        self.star_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.star_btn.setToolTip(tr("Keep this object in Favorites"))
        self.set_favorite(False)
        self.star_btn.clicked.connect(self._on_star_clicked)
        footer.addWidget(self.star_btn)
        self._hint = _build_use_hint(self)
        footer.addWidget(self._hint)
        lay.addWidget(footer_wrap)

    def _on_star_clicked(self, checked: bool) -> None:
        self.set_favorite(checked)
        self.star_toggled.emit(self._preset, checked)

    def set_favorite(self, fav: bool) -> None:
        self.star_btn.blockSignals(True)
        self.star_btn.setChecked(fav)
        _set_star_glyph(self.star_btn, fav)
        self.star_btn.setAccessibleName(
            tr("Remove from favorites") if fav else tr("Add to favorites"))
        self.star_btn.blockSignals(False)

    def request_demos(self, loader: TemplateDemoLoader, base: str) -> None:
        if self._requested:
            return
        self._requested = True
        loader.request(self._preset["id"], "before", _demo_url(base, self._preset, "before"))
        loader.request(self._preset["id"], "after", _demo_url(base, self._preset, "after"))

    def set_image(self, which: str, pixmap) -> None:
        if which == "before":
            self.slider.set_before(pixmap)
        elif which == "after":
            self.slider.set_after(pixmap)

    def mark_missing(self, which: str) -> None:
        if which not in ("before", "after"):
            return
        self._missing.add(which)



        self.slider.mark_unavailable(which)
        if {"before", "after"} <= self._missing and not self.slider.has_images():
            self.slider.set_placeholder_text(tr("No preview"))

    def _fire(self) -> None:
        self.activated.emit(self._preset)

    def enterEvent(self, ev):  # noqa: N802
        self.setStyleSheet(_CARD_HOVER)
        _set_use_hint(self._hint, True)
        super().enterEvent(ev)

    def leaveEvent(self, ev):  # noqa: N802
        self.setStyleSheet(_CARD_NORMAL)
        _set_use_hint(self._hint, False)
        super().leaveEvent(ev)

    def mouseReleaseEvent(self, ev):  # noqa: N802





        pt = QtC.event_pos(ev)
        if (_released_on_card(self, ev) and not self.slider.geometry().contains(
                QPoint(int(pt.x()), int(pt.y())))):
            self._fire()
        super().mouseReleaseEvent(ev)


class _RecentCard(QFrame):
















    activated = pyqtSignal(dict)
    rerun_requested = pyqtSignal(dict)
    reuse_prompt_requested = pyqtSignal(dict)

    def __init__(self, entry: dict, parent=None, *, view_only: bool = False):
        super().__init__(parent)
        self._entry = entry
        self._view_only = bool(view_only)
        self.setObjectName("card")
        self.setStyleSheet(_CARD_NORMAL)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumWidth(_card_min_width())
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)







        self._thumb = BeforeAfterSlider(
            self, auto_loop=False, show_badges=False, handle_grab_only=True)
        self._thumb.setFixedHeight(_preview_height())
        self._thumb.set_card_corners(RADIUS_CARD, top_only=True)
        self._thumb.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._thumb.set_placeholder_text(tr("No preview"))
        self._thumb.mark_unavailable("before")
        self._thumb.clicked.connect(self._fire)
        thumb_path = entry.get("_thumb") or ""
        if thumb_path:
            pixmap = QPixmap(thumb_path)
            if not pixmap.isNull():
                self._thumb.set_after(pixmap)
        lay.addWidget(self._thumb)

        footer_wrap = QWidget(self)
        footer = QVBoxLayout(footer_wrap)
        footer.setContentsMargins(12, 10, 12, 10)
        footer.setSpacing(3)
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title = _ElidedTitle(entry.get("label") or entry.get("prompt", ""), footer_wrap)
        title_row.addWidget(title, 1)
        self._hint = _build_use_hint(self)
        title_row.addWidget(self._hint)
        footer.addLayout(title_row)

        meta = entry.get("_meta") or ""
        if meta:
            meta_lbl = QLabel(meta)
            meta_lbl.setTextFormat(QtC.PlainText)
            meta_lbl.setStyleSheet(_META_QSS)
            footer.addWidget(meta_lbl)





        if self._view_only:
            lay.addWidget(footer_wrap)
            return
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 2, 0, 0)
        actions.setSpacing(2)
        has_zone = bool(entry.get("extent") and entry.get("crs"))
        if has_zone:
            again_btn = QPushButton(tr("Run again here"))
            again_btn.setStyleSheet(_RECENT_ACTION_QSS)
            again_btn.setAutoDefault(False)
            again_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            again_btn.setToolTip(
                tr("Reload this zone and object, ready to detect."))
            again_btn.clicked.connect(
                lambda: self.rerun_requested.emit(self._entry))
            actions.addWidget(again_btn)
        new_zone_btn = QPushButton(tr("Same object, new zone"))
        new_zone_btn.setStyleSheet(_RECENT_ACTION_QSS)
        new_zone_btn.setAutoDefault(False)
        new_zone_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        new_zone_btn.setToolTip(
            tr("Keep this object and draw a new zone on the map."))
        new_zone_btn.clicked.connect(
            lambda: self.reuse_prompt_requested.emit(self._entry))
        actions.addWidget(new_zone_btn)
        actions.addStretch()
        footer.addLayout(actions)
        lay.addWidget(footer_wrap)

    def enterEvent(self, ev):  # noqa: N802
        self.setStyleSheet(_CARD_HOVER)
        _set_use_hint(self._hint, True)
        super().enterEvent(ev)

    def leaveEvent(self, ev):  # noqa: N802
        self.setStyleSheet(_CARD_NORMAL)
        _set_use_hint(self._hint, False)
        super().leaveEvent(ev)

    def _fire(self) -> None:
        self.activated.emit(self._entry)

    def mouseReleaseEvent(self, ev):  # noqa: N802


        pt = QtC.event_pos(ev)
        if (_released_on_card(self, ev) and not self._thumb.geometry().contains(
                QPoint(int(pt.x()), int(pt.y())))):
            self._fire()
        super().mouseReleaseEvent(ev)
