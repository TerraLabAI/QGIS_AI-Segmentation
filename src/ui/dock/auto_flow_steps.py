






from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLineEdit,
)

from ...core.i18n import tr
from .styles import combo_theme_qss, locked_combo_qss
from .widgets import (
    Mode,
)


class DockAutoFlowStepsMixin:



    def _on_auto_layer_changed(self, layer) -> None:


        if self._mode == Mode.AUTOMATIC:
            self._update_ui_state_automatic()
        self._update_auto_detect_enabled()
        self._refresh_auto_layer_lock()

    def _on_auto_start_clicked(self) -> None:






        layer = self.auto_layer_combo.currentLayer()
        if layer is None:
            return






        self.set_auto_status("idle")

        self.clear_auto_export_success()
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_auto_start_clicked(
                layer_kind=self._auto_layer_kind(layer),
                has_credits_known=self._auto_credits is not None,
            )
        except Exception:
            pass  # nosec B110
        self._auto_started = True
        self._go_to_auto_step(1)

    @staticmethod
    def _auto_layer_kind(layer) -> str:

        try:
            provider = (layer.dataProvider().name() or "").lower()
            if provider == "gdal":
                return "local"
            source = (layer.source() or "").lower()
            if "type=xyz" in source:
                return "xyz"
            if provider in ("wms", "wmts"):
                return "wms"
            return provider or "other"
        except Exception:
            return "other"

    def reset_auto_to_start(self) -> None:






        self._auto_started = False


        self._auto_finalizing = False


        self.clear_auto_export_success()

        self.set_auto_zone_rejected(None)
        self.set_auto_zone_refusal(None)
        self.set_auto_exhausted_subscribe_visible(False)

        self.set_auto_zone_surface(None)


        self.set_auto_status("idle")


        if not self._refine_handoff:
            self.mode_switch.setEnabled(True)
            self.mode_switch.setToolTip("")
        self.auto_prompt_input.blockSignals(True)
        self.auto_prompt_input.clear()
        self.auto_prompt_input.blockSignals(False)




        self._last_committed_prompt = None
        self._boost_nudge_tracked = None


        self._abandon_prompt_lookup()
        self._set_prompt_info()

        self._auto_detail_feedback = None



        self._update_auto_detect_enabled()




        self.auto_layer_combo.setVisible(True)
        self.auto_review_view_row.setVisible(False)
        self._go_to_auto_step(0)

    def _go_to_auto_step(self, index: int) -> None:







        if self._auto_run_active:
            index = 2






        if not self._auto_run_active and not getattr(self, "_auto_finalizing", False):
            try:
                self.auto_run_summary_card.setVisible(False)
            except (RuntimeError, AttributeError):

                pass



        try:
            self.forget_prompt_suggest_recent()
        except (RuntimeError, AttributeError):
            pass








        self.auto_steps.setVisible(
            index != 0 or self.auto_layer_combo.count_layers() > 0)
        self.auto_steps.setCurrentIndex(index)



        if index == 1:
            self.refresh_auto_zone_ready_card()
        self._refresh_auto_layer_lock()


        self._refresh_mode_switch_visibility()
        self._update_auto_detect_enabled()




        self.auto_exemplar_panel.setVisible(
            self._EXEMPLARS_ENABLED and index == 2 and not self._auto_run_active and not self._auto_review_active
        )

        self.refresh_auto_shortcut_arming()


        self.auto_step_changed.emit(index)

    def _refresh_auto_layer_lock(self) -> None:












        if self.auto_steps.currentIndex() == 0 and self.auto_layer_combo.count_layers() == 0:
            self.auto_layer_label.setVisible(False)
            self.auto_layer_combo.setVisible(False)
            return
        on_start = self.auto_steps.currentIndex() == 0
        self.auto_layer_label.setVisible(on_start)
        self.auto_layer_combo.setEnabled(on_start)



        self.auto_layer_combo.set_frozen(not on_start)



        qss = (combo_theme_qss() if on_start else
               locked_combo_qss("automatic"))


        if getattr(self, "_auto_layer_combo_qss", None) != qss:
            self._auto_layer_combo_qss = qss
            self.auto_layer_combo.setStyleSheet(qss)

    def on_zone_deleted_from_canvas(self) -> None:

        self.set_auto_zone_state("idle")
        self._go_to_auto_step(1)

    def set_auto_zone_state(self, state: str) -> None:

        self._auto_zone_is_set = state == "zone_set"



        if state in ("idle", "zone_set"):
            self.set_auto_zone_rejected(None)
            self.set_auto_zone_refusal(None)
        if state in ("idle", "drawing"):
            self._auto_zone_too_large = False




            try:
                self.set_auto_advanced_open(False)
            except (RuntimeError, AttributeError):

                pass






            self._auto_est_credits = None
            self.set_auto_zone_surface(None)
            self.refresh_auto_run_estimate()
        elif state == "zone_set":


            self._auto_started = True

            self._go_to_auto_step(2)






            self._begin_auto_prompt_focus()
        self._update_auto_detect_enabled()

    def set_auto_zone_refusal(self, message: str | None) -> None:





        label = getattr(self, "_auto_zone_refusal_label", None)
        if not message:
            if label is not None:
                try:
                    label.setVisible(False)
                except RuntimeError:
                    self._auto_zone_refusal_label = None
            return
        try:
            if label is None:
                from qgis.PyQt.QtWidgets import QLabel

                from .styles import _msg_label_qss
                label = QLabel("")
                label.setWordWrap(True)
                label.setTextFormat(Qt.TextFormat.RichText)
                label.setStyleSheet(_msg_label_qss("error"))
                self.auto_zone_hero.layout().addWidget(label)
                self._auto_zone_refusal_label = label
            from .styles import msg_rich
            label.setText(msg_rich("error", message))
            label.setVisible(True)
        except (RuntimeError, AttributeError):
            self._auto_zone_refusal_label = None

    def set_zone_draw_progress(self, count: int) -> None:


        if count <= 0:
            txt = tr("Click on the map to outline your zone.")
        elif count < 3:
            txt = tr("Keep clicking around the zone, at least 3 points.")
        else:



            txt = tr("Double-click, or click the first point, to close the zone.")
        try:
            self._auto_zone_hint.setText(txt)
        except (RuntimeError, AttributeError):
            pass

    def _begin_auto_prompt_focus(self) -> None:








        from qgis.PyQt.QtCore import QTimer
        timer = getattr(self, "_auto_prompt_focus_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setInterval(120)
            timer.timeout.connect(self._tick_auto_prompt_focus)
            self._auto_prompt_focus_timer = timer
        self._auto_prompt_focus_ticks = 0
        self._tick_auto_prompt_focus()
        timer.start()

    def _tick_auto_prompt_focus(self) -> None:

        timer = getattr(self, "_auto_prompt_focus_timer", None)
        try:
            self._auto_prompt_focus_ticks = getattr(
                self, "_auto_prompt_focus_ticks", 0) + 1
            prompt = self.auto_prompt_input
            if (self.auto_steps.currentIndex() != 2 or not prompt.isVisible() or not prompt.isEnabled()):
                if timer is not None:
                    timer.stop()
                return
            focused = QApplication.focusWidget()




            if focused is not prompt and (
                    focused is None or not self.isAncestorOf(focused)):
                prompt.setFocus(Qt.FocusReason.OtherFocusReason)
            if self._auto_prompt_focus_ticks >= 8 and timer is not None:
                timer.stop()
        except (RuntimeError, AttributeError):
            if timer is not None:
                timer.stop()

    def _is_auto_for_us(self) -> bool:



        return self._mode == Mode.AUTOMATIC and self._auto_started and not self._auto_run_active

    def auto_flow_owns_keys(self) -> bool:






        return self._mode == Mode.AUTOMATIC and self._auto_started

    def _on_auto_escape_shortcut(self) -> None:







        if self._prompt_suggest_popup_open():
            self._prompt_suggest_completer.popup().hide()
            return
        if self._mode == Mode.AUTOMATIC and self._auto_started:
            self.auto_escape_pressed.emit()

    def _prompt_suggest_popup_open(self) -> bool:





        try:
            completer = getattr(self, "_prompt_suggest_completer", None)
            return bool(completer is not None and completer.popup().isVisible())
        except (RuntimeError, AttributeError):
            return False

    def _dock_combo_has_focus(self) -> bool:








        try:
            return isinstance(QApplication.focusWidget(),
                              (QComboBox, QAbstractItemView))
        except (RuntimeError, AttributeError):
            return False

    def _on_auto_enter_shortcut(self) -> None:















        if not self._is_auto_for_us():
            return



        if self._prompt_suggest_popup_open() and self._on_prompt_suggest_enter():
            return
        fw = QApplication.focusWidget()
        if isinstance(fw, QLineEdit):
            fw.returnPressed.emit()
            return
        if isinstance(fw, QAbstractSpinBox):
            fw.interpretText()
            fw.editingFinished.emit()
            return
        if self._dock_combo_has_focus():
            return
        self.auto_enter_pressed.emit()

    def _is_auto_correct_shortcut_active(self) -> bool:

        active = self._mode == Mode.AUTOMATIC and bool(getattr(self, "_auto_review_active", False))
        active = active and getattr(self, "_auto_review_step", 0) == 1
        active = active and not bool(getattr(self, "_refine_handoff", False))
        return active and not bool(getattr(self, "_qgis_bridge_active_ui", False))

    def _on_auto_correct_remove_shortcut(self) -> None:

        if not self._is_auto_correct_shortcut_active():
            return
        try:
            if self.auto_correct_select_card.isVisible():
                self.auto_remove_requested.emit()
        except (RuntimeError, AttributeError):
            pass

    def _ai_fix_session_owns_undo(self) -> bool:







        if not bool(getattr(self, "_refine_handoff", False)):
            return False
        return not bool(getattr(self, "_qgis_bridge_active_ui", False))

    def _on_auto_correct_undo_shortcut(self) -> None:







        if self._is_auto_correct_shortcut_active() or self._ai_fix_session_owns_undo():
            self.auto_correction_undo_requested.emit()

    def set_auto_shortcuts_enabled(self, enabled: bool) -> None:













        self._auto_shortcuts_master = bool(enabled)
        self.refresh_auto_shortcut_arming()

    def _auto_undo_shortcut_armed(self) -> bool:
















        if bool(getattr(self, "_qgis_bridge_active_ui", False)):
            return False
        if self._ai_fix_session_owns_undo():
            return True
        if not self.auto_flow_owns_keys():
            return False





        return not bool(getattr(self, "_auto_correct_merge_armed", False))

    def refresh_auto_shortcut_arming(self) -> None:



















        master = bool(getattr(self, "_auto_shortcuts_master", True))
        try:
            typing_in_list = self._dock_combo_has_focus()
            flow_keys = master and self.auto_flow_owns_keys()
            remove_keys = master and self._is_auto_correct_shortcut_active()
            undo_key = self._auto_undo_shortcut_armed()
        except (RuntimeError, AttributeError):
            return
        armed = (
            (("auto_escape_shortcut",), flow_keys),
            (("auto_enter_shortcut", "auto_enter_shortcut_kp"),
             flow_keys and not typing_in_list),
            (("auto_correct_remove_delete_shortcut",
              "auto_correct_remove_backspace_shortcut"), remove_keys),
            (("auto_correct_undo_shortcut",), undo_key),
            (("start_shortcut",), not typing_in_list),
        )
        for names, enabled in armed:
            for name in names:
                sc = getattr(self, name, None)
                if sc is None:
                    continue
                try:
                    sc.setEnabled(enabled)
                except (RuntimeError, AttributeError):
                    pass  # nosec B110
