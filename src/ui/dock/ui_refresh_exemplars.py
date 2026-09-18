







from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import QFrame, QLabel, QToolButton, QWidget

from ...core.i18n import tr
from ...core.qt_compat import safe_single_shot
from ..icons import icon_for
from .font_scale import scale_px_length, scale_qss_font_px, widget_pixel_ratio
from .guidance import HINT_EXEMPLAR_DRAW_BOX, HINT_EXEMPLAR_EXCLUDE_BOX
from .styles import (
    BRAND_GREEN,
    FONT_BASE,
    FONT_HINT,
    FONT_MICRO,
    INK_2,
    RADIUS_CARD,
    RADIUS_CHIP,
    _msg_label_qss,
    msg_rich,
)


class DockExemplarsMixin:





    def _set_btn_armed(self, btn, armed: bool) -> None:


        btn.setProperty("armed", "true" if armed else "false")
        btn.style().unpolish(btn)
        btn.style().polish(btn)

    def _refresh_exemplar_button_labels(self) -> None:














        armed_label = getattr(self, "_auto_exemplar_armed_label", None)
        drawing = tr("Drawing (click to stop)")
        drawn = getattr(self, "_auto_positive_exemplars", 0)
        try:
            self.auto_ex_inc_btn.setText(
                drawing if armed_label == 1
                else (tr("Draw another example") if drawn
                      else tr("Draw on the map")))
        except (RuntimeError, AttributeError):

            pass
        exc = getattr(self, "auto_ex_exc_btn", None)
        if exc is None:
            return
        try:
            exc.setText(drawing if armed_label == 0
                        else tr("Exclude a look-alike"))
        except (RuntimeError, AttributeError):

            pass

    def _auto_exemplar_line_busy(self) -> bool:






        for name in ("auto_exemplar_size_warning", "auto_exemplar_armed_tip"):
            try:
                if getattr(self, name).isVisible():
                    return True
            except (RuntimeError, AttributeError):
                continue
        return False

    def set_auto_exemplar_armed(self, label) -> None:





        self._auto_exemplar_armed_label = label
        try:
            self._set_btn_armed(self.auto_ex_inc_btn, label == 1)
            exc = getattr(self, "auto_ex_exc_btn", None)
            if exc is not None:
                self._set_btn_armed(exc, label == 0)
            self._refresh_exemplar_button_labels()
        except (RuntimeError, AttributeError):
            return
        if label is None:
            self.auto_exemplar_size_warning.setVisible(False)
            self.auto_exemplar_armed_tip.setVisible(False)
            self._refresh_auto_exemplar_explainer()
            self._auto_exemplar_hint_kind = None
            self._set_exemplar_quality()
            return


        try:
            self._set_auto_exemplar_expanded(True)
        except (RuntimeError, AttributeError):

            pass




        self.auto_exemplar_size_warning.setVisible(False)
        shown = self.auto_exemplar_armed_tip.set_hint(
            HINT_EXEMPLAR_EXCLUDE_BOX if label == 0 else HINT_EXEMPLAR_DRAW_BOX,

            (
                tr("Click points around one look-alike, then double-click "
                   "to close.")
                if label == 0 else
                tr("Click points around one object, then double-click "
                   "to close.")
            ),
            dense=True)
        self.auto_exemplar_armed_tip.setVisible(shown)
        self._refresh_auto_exemplar_explainer(slot_taken=shown)
        self._auto_exemplar_hint_kind = "armed"
        self._set_exemplar_quality()

    def show_auto_exemplar_size_warning(self, at_max_detail: bool = False) -> None:


















        try:
            self.auto_exemplar_armed_tip.setVisible(False)
            self.auto_exemplar_size_warning.setStyleSheet(
                _msg_label_qss("warning"))
            self.auto_exemplar_size_warning.setTextFormat(Qt.TextFormat.RichText)
            self.auto_exemplar_size_warning.setText(msg_rich("warning", (
                tr("This example is very small even at full precision. "
                   "Draw a larger object, or it may be too small to detect.")
                if at_max_detail else
                tr("This example is very small at this precision. "
                   "Raise the precision or draw a larger object."))))
            self.auto_exemplar_size_warning.setVisible(True)
            self._refresh_auto_exemplar_explainer(slot_taken=True)
            self._auto_exemplar_hint_kind = "warning"


            self._apply_prompt_hint_on_edit()
            self._set_exemplar_quality()
        except (RuntimeError, AttributeError):

            pass

    def clear_auto_exemplar_size_warning(self) -> None:







        if getattr(self, "_auto_exemplar_hint_kind", None) != "warning":
            return
        try:
            self.auto_exemplar_size_warning.setVisible(False)
            self._refresh_auto_exemplar_explainer()
            self._auto_exemplar_hint_kind = None


            self._apply_prompt_hint_on_edit()
            self._set_exemplar_quality()
        except (RuntimeError, AttributeError):

            pass

    def set_exemplars(self, items: list) -> None:




        layout = self._auto_exemplar_chips_layout


        self._auto_exemplar_items = list(items)

        while layout.count() > 1:
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        from ...core.detect_gate import exclude_available
        from ...core.exemplar_store import max_exclude, max_positive, max_total
        self._auto_positive_exemplars = sum(1 for it in items if it[1] == 1)
        exclude_count = sum(1 for it in items if it[1] == 0)


        self._refresh_exemplar_button_labels()
        for idx, it in enumerate(items):
            eid, label = it[0], it[1]
            thumb = it[2] if len(it) > 2 else None
            card = self._make_exemplar_chip(eid, label, idx + 1, thumb)
            layout.insertWidget(layout.count() - 1, card)











        total_left = (self._auto_positive_exemplars + exclude_count) < max_total()
        try:
            self.auto_ex_inc_btn.setEnabled(
                total_left and self._auto_positive_exemplars < max_positive())
            exc = getattr(self, "auto_ex_exc_btn", None)
            if exc is not None:
                exc_available = exclude_available(self._auto_positive_exemplars)
                exc.setVisible(exc_available)
                exc.setEnabled(
                    exc_available and total_left and exclude_count < max_exclude())
        except (RuntimeError, AttributeError):

            pass


        try:
            self.hide_auto_exemplar_upsell()
        except (RuntimeError, AttributeError):
            pass


        self._auto_exemplar_count = len(items)
        self._refresh_auto_exemplar_explainer(
            slot_taken=self._auto_exemplar_line_busy())
        self._set_exemplar_quality()


        if items:
            try:
                self._set_auto_exemplar_expanded(True)
            except (RuntimeError, AttributeError):

                pass



        self._apply_prompt_hint_on_edit()
        self._update_auto_detect_enabled()

    def _set_exemplar_quality(self) -> None:









        positives = (getattr(self, "_auto_positive_exemplars", 0)
                     if getattr(self, "_EXEMPLARS_ENABLED", True) else 0)
        dots = getattr(self, "auto_exemplar_quality_dots", None)
        line = getattr(self, "auto_exemplar_quality_line", None)
        if dots is None or line is None:
            return



        known_free = (getattr(self, "_auto_credits", None) is not None
                      and not getattr(self, "_auto_is_subscriber", False))
        if known_free:
            try:
                dots.setVisible(False)
                line.setVisible(False)
            except (RuntimeError, AttributeError):

                pass
            return


        filled = min(positives, 2)
        empty = "rgba(128, 128, 128, 0.45)"
        marks = []

        from .font_scale import scale_point_size
        dot_px = scale_point_size(FONT_BASE)
        for i in range(2):
            color = BRAND_GREEN if i < filled else empty
            marks.append(
                f'<span style="color: {color}; font-size: {dot_px}px;">&#9679;</span>')
        try:
            dots.setText("&nbsp;".join(marks))
            dots.setVisible(positives > 0)
        except (RuntimeError, AttributeError):

            pass


        armed_showing = self._auto_exemplar_line_busy()
        try:
            if positives <= 0 or armed_showing:
                line.setVisible(False)
            elif positives == 1:



                line.setObjectName("autoHint")
                line.setStyleSheet("")
                line.setText(tr("Add one more example for the best results."))
                line.setVisible(True)
            else:



                line.setVisible(False)
        except (RuntimeError, AttributeError):

            pass

    def _make_exemplar_chip(self, exemplar_id: str, label: int,
                            index: int = 1, thumbnail=None,
                            removable: bool = True,
                            side_px: int = 52) -> QWidget:






        from qgis.PyQt.QtGui import QPixmap
        is_pos = label == 1
        rgba = "67,160,71" if is_pos else "229,57,53"

        side = scale_px_length(side_px)
        card = QFrame()
        card.setObjectName("refCard")
        card.setFixedSize(side, side)


        card.setStyleSheet(
            f"QFrame#refCard {{ border: 1px solid rgba({rgba},0.85);"
            f" border-radius: {RADIUS_CARD}px; background: rgba({rgba},0.10); }}")
        card.setToolTip(tr("Example"))

        thumb_lbl = QLabel(card)
        thumb_lbl.setGeometry(1, 1, side - 2, side - 2)


        thumb_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb_lbl.setStyleSheet("border: none; background: transparent;")
        if thumbnail is not None:
            try:
                pm = QPixmap.fromImage(thumbnail)
                if not pm.isNull():



                    ratio = widget_pixel_ratio(thumb_lbl)
                    edge = int(round((side - 2) * ratio))
                    pm = pm.scaled(
                        edge, edge,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)
                    pm.setDevicePixelRatio(ratio)
                    thumb_lbl.setPixmap(pm)



                card.setCursor(Qt.CursorShape.PointingHandCursor)
                card.setToolTip(tr("Click to enlarge"))



                card.mousePressEvent = (
                    lambda _ev, im=thumbnail, n=index, lb=label: (
                        safe_single_shot(
                            0, self,
                            lambda: self._show_exemplar_detail(im, n, lb))))
            except (RuntimeError, TypeError):

                pass

        badge = QLabel(str(index), card)
        badge.setStyleSheet(scale_qss_font_px(
            "QLabel { background: rgba(0,0,0,0.6); color: rgba(255,255,255,0.92);"
            f" font-size: {FONT_MICRO}px; font-weight: 600; border: none;"
            f" border-radius: {RADIUS_CHIP}px; padding: 0 4px; }}"))
        badge.adjustSize()
        badge.move(3, 3)

        if not removable:
            return card
        remove = QToolButton(card)
        remove.setIcon(icon_for(remove, "close", 10, QColor("#ffffff")))
        remove.setCursor(Qt.CursorShape.PointingHandCursor)
        remove.setToolTip(tr("Remove"))
        remove.setAccessibleName(tr("Remove"))
        remove_side = scale_px_length(16)
        remove.setFixedSize(remove_side, remove_side)
        remove.setStyleSheet(scale_qss_font_px(
            "QToolButton { background: rgba(0,0,0,0.62);"
            " border: 1px solid rgba(255,255,255,0.75);"
            f" border-radius: {remove_side // 2}px; padding: 0; }}"
            "QToolButton:hover { background: rgba(0,0,0,0.88); }"
            "QToolButton:pressed { background: rgba(0,0,0,0.95); }"))
        remove.move(side - remove_side - 3, 3)
        remove.clicked.connect(
            lambda _checked=False, eid=exemplar_id: self.auto_exemplar_remove_requested.emit(eid))
        return card

    def _show_exemplar_detail(self, image, index: int = 1,
                              label: int = 1) -> None:







        if image is None:
            return
        try:
            from qgis.PyQt.QtGui import QPixmap
            from qgis.PyQt.QtWidgets import QDialog, QLabel, QVBoxLayout
            pm = QPixmap.fromImage(image)
            if pm.isNull():
                return



            ratio = widget_pixel_ratio(self)
            edge = int(round(320 * ratio))
            pm = pm.scaled(
                edge, edge,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            pm.setDevicePixelRatio(ratio)
            dlg = QDialog(self)
            dlg.setWindowTitle(
                tr("Exclude {n}").format(n=index) if label == 0
                else tr("Reference {n}").format(n=index))
            lay = QVBoxLayout(dlg)
            lay.setContentsMargins(12, 12, 12, 12)
            lay.setSpacing(8)
            img_lbl = QLabel(dlg)
            img_lbl.setPixmap(pm)
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(img_lbl)
            hint = QLabel(
                tr("The AI drops objects that look like this.") if label == 0
                else tr("The AI looks for more objects like this."))
            hint.setWordWrap(True)
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hint.setStyleSheet(scale_qss_font_px(
                f"color: {INK_2}; font-size: {FONT_HINT}px;"))
            lay.addWidget(hint)
            from .font_scale import apply_font_scale_to_tree

            apply_font_scale_to_tree(dlg)
            dlg.exec()
        except Exception:  # noqa: BLE001  # nosec B110
            pass
