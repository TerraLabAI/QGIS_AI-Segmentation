











from __future__ import annotations

from ...core.i18n import tr
from ...core.server_dials import dial_copy, dial_in_range
from ...core.tile_manager import MAX_DETAIL_LEVEL
from ..icons import pixmap_for
from .auto_flow_look import token_qcolor
from .font_scale import scale_qss_font_px
from .styles import (
    FONT_HINT,
    INK_2,
    INK_3,
    _msg_label_qss,
    msg_rich,
)




_DETAIL_EMIT_DEBOUNCE_MS = 150


def _detail_hint_copy(state: str, fallback: str, obj: str = "") -> str:













    text = dial_copy(f"detail_hint.{state}", fallback)
    return text.replace("{obj}", obj) if obj else text


class DockAutoDetailLevelMixin:



    def _refresh_auto_advanced_header(self) -> None:





        try:
            self.auto_advanced_toggle_title.setText(tr("Detail level"))
            chevron = self.auto_advanced_toggle_chevron
            name = "chevron_down" if self._auto_advanced_open else "chevron_right"



            chevron.setPixmap(pixmap_for(chevron, name, 14, token_qcolor(INK_3)))
        except (RuntimeError, AttributeError):

            pass

    def _on_auto_advanced_toggle_clicked(self) -> None:







        self.set_auto_advanced_open(not self._auto_advanced_open)

    def set_auto_advanced_open(self, open_: bool) -> None:






        open_ = bool(open_)
        if open_ == getattr(self, "_auto_advanced_open", False):
            return
        try:
            self.auto_advanced_body.setVisible(
                open_ and self._auto_detail_object_known())
        except (RuntimeError, AttributeError):
            return
        self._auto_advanced_open = open_
        self._refresh_auto_advanced_header()
        self.auto_advanced_toggled.emit(open_)

    def _auto_detail_object_known(self) -> bool:





        return bool(getattr(self, "_auto_detail_has_object", False))

    def _apply_auto_detail_gate(self, has_object: bool) -> None:










        has_object = bool(has_object)
        if getattr(self, "_auto_detail_has_object", None) is has_object:
            return
        self._auto_detail_has_object = has_object
        try:


            card = self.auto_advanced_body
            card.setEnabled(has_object)
            card.setVisible(
                has_object and bool(getattr(self, "_auto_advanced_open", False)))



            self._auto_advanced_fold.setVisible(has_object)
        except (RuntimeError, AttributeError):

            pass
        self._refresh_auto_advanced_header()
        if has_object:
            try:


                self._refresh_auto_detail_hint()
            except (RuntimeError, AttributeError):

                pass

    def _on_auto_detail_changed(self, value: int) -> None:




        self._refresh_auto_detail_hint()
        self._auto_detail_pending_value = value
        timer = getattr(self, "_auto_detail_emit_timer", None)
        if timer is None:
            from qgis.PyQt.QtCore import QTimer
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._flush_auto_detail_changed)
            self._auto_detail_emit_timer = timer
        timer.start(dial_in_range(
            "tuning.auto.detail_emit_debounce_ms",
            _DETAIL_EMIT_DEBOUNCE_MS, 50, 1000))

    def _on_auto_zone_fit_clicked(self) -> None:












        if getattr(self, "_auto_zone_fit_action", "fit") != "fit":

            handler = getattr(self, "_on_auto_run_block_redraw", None)
            if handler is not None:
                handler()
            return
        slider = getattr(self, "auto_detail_slider", None)
        if slider is None:
            return
        try:
            if slider.value() > slider.minimum():
                slider.setValue(slider.minimum())
        except RuntimeError:
            pass  # nosec B110

    def set_auto_zone_fit_available(self, can_fit: bool | None) -> None:







        self._auto_zone_can_fit = bool(can_fit)

    def set_auto_zone_fit_visible(self, visible: bool) -> None:











        btn = getattr(self, "auto_zone_fit_btn", None)
        if btn is None:
            return
        slider = getattr(self, "auto_detail_slider", None)
        room = bool(slider is not None and slider.value() > slider.minimum())
        can_fit = bool(getattr(self, "_auto_zone_can_fit", False)) and room
        try:
            if not visible:
                btn.setVisible(False)
                return
            if can_fit:
                btn.setText(dial_copy("zone.fit_precision_cta",
                                      tr("Lower precision to fit")))
                btn.setToolTip(tr("Sweeps the same zone in a coarser grid, so "
                                  "it fits in one run."))
            else:



                btn.setText(dial_copy("run_block.redraw_cta",
                                      tr("Draw a smaller zone")))
                btn.setToolTip("")
            self._auto_zone_fit_action = "fit" if can_fit else "redraw"
            btn.setVisible(True)
        except RuntimeError:
            pass  # nosec B110

    def _flush_auto_detail_changed(self) -> None:

        value = getattr(self, "_auto_detail_pending_value", None)
        if value is not None:
            self.auto_detail_changed.emit(value)

    def _cancel_pending_auto_detail(self) -> None:







        timer = getattr(self, "_auto_detail_emit_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except (RuntimeError, AttributeError):
                pass  # nosec B110
        self._auto_detail_pending_value = None

    def set_auto_detail_value(self, n: int) -> None:






        self._cancel_pending_auto_detail()
        s = self.auto_detail_slider
        s.blockSignals(True)
        if s.maximum() < n:
            s.setMaximum(min(MAX_DETAIL_LEVEL, int(n)))
        s.setValue(max(s.minimum(), min(s.maximum(), int(n))))
        s.blockSignals(False)
        self._refresh_auto_detail_hint()

    def set_auto_detail_visible(self, visible: bool) -> None:

        self.auto_detail_row.setVisible(visible)

    def set_auto_detail_gsd_warning(
        self, coarse: bool, can_improve: bool | None = None
    ) -> None:
















        s = self.auto_detail_slider




        raise_helps = (bool(can_improve) if can_improve is not None
                       else s.value() < s.maximum())
        coarse = bool(coarse) and raise_helps
        if coarse:
            self.auto_detail_warning_label.setText(



                tr("Each tile covers a lot of ground at this precision. Raise"
                   " the precision for sharper detections."))
        self._auto_gsd_warning_on = coarse
        self.auto_detail_warning.setVisible(coarse)
        if coarse:
            self.auto_detail_hint.setVisible(False)
        else:






            self._refresh_auto_detail_hint()







    def set_auto_detail_range(
        self, lo: int, hi: int, object_bound: bool = False
    ) -> None:













        hi = max(1, min(MAX_DETAIL_LEVEL, int(hi)))
        lo = max(1, min(hi, int(lo)))
        self._auto_detail_object_bound = bool(object_bound)
        self._cancel_pending_auto_detail()
        slider = self.auto_detail_slider
        slider.blockSignals(True)



        slider.setMaximum(max(hi, slider.maximum()))
        slider.setMinimum(lo)
        slider.setMaximum(hi)
        value = min(max(slider.value(), lo), hi)
        if slider.value() != value:
            slider.setValue(value)
        slider.blockSignals(False)



        self._auto_detail_single_level = lo >= hi
        try:
            self.auto_detail_slider_row.setVisible(lo < hi)
            self.auto_detail_sub.setVisible(lo < hi)
        except (RuntimeError, AttributeError):

            pass
        self._refresh_auto_detail_hint()

    def set_auto_detail_feedback(self, state: str | None, object_word: str) -> None:





        word = (object_word or "").strip()
        if len(word) > 24:
            word = word[:24] + "\u2026"
        self._auto_detail_feedback = (state, word) if state else None
        self._refresh_auto_detail_hint()

    def _set_detail_hint_style(self, qss: str) -> None:



        if getattr(self, "_auto_detail_hint_qss", None) == qss:
            return
        self._auto_detail_hint_qss = qss
        try:
            self.auto_detail_hint.setStyleSheet(qss)
        except (RuntimeError, AttributeError):
            pass

    def _refresh_auto_detail_hint(self) -> None:







        self._write_auto_detail_hint()
        try:
            if not getattr(self, "_auto_gsd_warning_on", False):
                self.auto_detail_hint.setVisible(bool(self.auto_detail_hint.text()))
        except (RuntimeError, AttributeError):
            pass

    def _write_auto_detail_hint(self) -> None:







        s = self.auto_detail_slider
        capped = s.maximum() < MAX_DETAIL_LEVEL and s.value() >= s.maximum()
        feedback = getattr(self, "_auto_detail_feedback", None)
        _plain_hint = scale_qss_font_px(
            f"font-size: {FONT_HINT}px; color: {INK_2};"
            " background: transparent; border: none;")
        self.auto_detail_hint.setToolTip("")
        if getattr(self, "_auto_detail_single_level", False):
            word = feedback[1] if feedback else ""
            obj = f'"{word}"' if word else tr("your object")
            self._set_detail_hint_style(_plain_hint)
            self.auto_detail_hint.setText(_detail_hint_copy(
                "single", tr(
                    "One precision level fits {obj} in a zone this size - draw"
                    " a larger zone for a choice."), obj))
            return
        if capped and getattr(self, "_auto_detail_object_bound", False):





            word = feedback[1] if feedback else ""
            obj = f'"{word}"' if word else tr("your object")
            self._set_detail_hint_style(_plain_hint)
            self.auto_detail_hint.setText(_detail_hint_copy(
                "objcap",
                tr("As fine as {obj} benefits from - finer splits them into"
                   " pieces."), obj))
            return
        if feedback and not (capped and feedback[0] in ("coarse", "below")):


            state, word = feedback
            obj = f'"{word}"' if word else tr("your object")
            if state == "coarse":
                self._set_detail_hint_style(_msg_label_qss("warning"))
                self.auto_detail_hint.setText(msg_rich("warning", _detail_hint_copy(
                    "coarse", tr(
                        "At this precision {obj} is too small to spot - raise the"
                        " precision."), obj)))
            elif state == "over":



                self._set_detail_hint_style(_msg_label_qss("warning"))
                self.auto_detail_hint.setText(msg_rich("warning", _detail_hint_copy(
                    "over", tr(
                        "Very fine for {obj} - large ones may come back split"
                        " in parts."), obj)))
            elif state == "above":
                self._set_detail_hint_style(_plain_hint)
                self.auto_detail_hint.setText(_detail_hint_copy(
                    "above", tr(
                        "Sharper than {obj} usually needs - catches the smallest"
                        " ones."), obj))
            elif state == "helps":
                self._set_detail_hint_style(_plain_hint)
                self.auto_detail_hint.setText(_detail_hint_copy(
                    "helps",
                    tr("More precision keeps helping {obj} in this zone."), obj))
            elif state == "below":
                self._set_detail_hint_style(_plain_hint)
                self.auto_detail_hint.setText(_detail_hint_copy(
                    "below",
                    tr("Small {obj} may be missed at this level."), obj))
            else:






                self._set_detail_hint_style(_msg_label_qss("success"))
                self.auto_detail_hint.setText(msg_rich("success", _detail_hint_copy(
                    "recommended",
                    tr("Right level for {obj} in this zone."), obj)))
            return
        if capped:
            self._set_detail_hint_style(_plain_hint)
            self.auto_detail_hint.setText(_detail_hint_copy("capped", tr(
                "Max precision for this zone - draw a larger zone to go finer.")))
        else:





            self._set_detail_hint_style(_plain_hint)
            self.auto_detail_hint.setText(_detail_hint_copy("default", ""))
