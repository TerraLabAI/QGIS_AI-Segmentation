






from __future__ import annotations

from ...core.i18n import tr
from .auto_run_status import _auto_progress_bar_qss
from .auto_run_summary import RUN_REFERENCE_PX
from .styles import (
    msg_rich,
)


class DockAutoRunLifecycleMixin:



    def arm_auto_cancel_confirm(self) -> None:










        self._auto_cancel_gesture = True

    def take_auto_cancel_gesture(self) -> bool:






        armed = bool(getattr(self, "_auto_cancel_gesture", False))
        self._auto_cancel_gesture = False
        return armed

    def _confirm_auto_cancel(self) -> bool:






        if getattr(self, "_auto_cancelling", False):
            return True
        from ..dialogs.confirm_dialog import question
        found = max(0, int(getattr(self, "_auto_found_count", 0) or 0))
        if found == 1:

            body = tr(
                "The object already found is kept and opens in the review. "
                "The surface already scanned still counts.")
        elif found > 1:


            body = tr(
                "The {n} objects already found are kept and open in the "
                "review. The surface already scanned still counts."
            ).format(n=found)
        else:
            body = tr(
                "Nothing has been found yet. The surface already scanned "
                "still counts."
            )


        return question(
            self, tr("Cancel this detection?"), body,
            default_yes=False, destructive=True,
            yes_label=tr("Cancel detection"), no_label=tr("Keep running"))

    def set_auto_run_active(self, active: bool) -> None:



        self._auto_cancel_gesture = False
        self._auto_run_active = active
        if active:



            self._auto_finalizing = False





        hold = (not active) and getattr(self, "_auto_finalizing", False)
        self.auto_cancel_btn.setVisible(active)




        if active:
            self.set_auto_exhausted_subscribe_visible(False)





            self.hide_auto_zero_assist()
            self._auto_cancelling = False
            self.auto_cancel_btn.setEnabled(True)
            self.auto_cancel_btn.setText(tr("Cancel detection"))


            self._auto_wait_phase = "imagery"
            self._auto_link_slow = False
            self._auto_link_local = False










        self.auto_detect_row.setVisible(not (active or hold))

        self.auto_settings_box.setVisible(False)
        self.auto_detail_row.setVisible(
            self._auto_zone_is_set if not (active or hold) else False)
        if active or hold:
            self.auto_credit_cost_label.setVisible(False)
        else:
            self._refresh_auto_cost_label()







        in_run = active or hold
        self.auto_prompt_card.setVisible(not in_run)


        self._refresh_auto_layer_lock()
        if in_run:
            self._refresh_auto_run_summary()
        else:
            self.auto_run_summary_card.setVisible(False)
        if active:
            self._go_to_auto_step(2)
        elif not hold:
            self._refresh_auto_layer_lock()

        if in_run:
            self.auto_exemplar_panel.setVisible(False)
        else:
            self.auto_exemplar_panel.setVisible(
                self._EXEMPLARS_ENABLED and self.auto_steps.currentIndex() == 2 and not self._auto_review_active)
        self._update_auto_detect_enabled()
        if active:

            self._auto_found_count = 0
            self._auto_progress_pair = (0, 0)
            self._auto_progress_ratio = 0.0


            self._auto_progress_target = 0
            self._auto_progress_shown = 0
            self._auto_progress_dirty = False
            self._auto_progress_phase = "grid"


            self._auto_queue_position = 0
            self._auto_queue_eta = 0
            self._stop_auto_warming_anim()
            self._stop_auto_progress_ease()
            self.auto_progress_count_label.setText("")
            self.auto_progress_pct_label.setText("")
            self.auto_progress_label.setVisible(False)


            try:
                self.auto_tile_progress.setStyleSheet(_auto_progress_bar_qss(None))
            except (RuntimeError, AttributeError):
                pass

            clock = getattr(self, "auto_progress_clock", None)
            if clock is not None:
                clock.restart_clock()
        else:

            self._stop_auto_warming_anim()
            self._stop_auto_progress_ease()

    def set_auto_finalizing(self, finalizing: bool) -> None:










        finalizing = bool(finalizing)
        if finalizing == getattr(self, "_auto_finalizing", False):
            return
        self._auto_finalizing = finalizing
        self._refresh_auto_layer_lock()
        if finalizing:




            self._auto_finalize_tiles = (0, 0)
            self._auto_finalize_phase_text = ""



            if not self.auto_status_banner.isVisible():
                self._paint_auto_finalize_card()
            self.set_auto_run_active(False)
        elif not self._auto_review_active:
            self.set_auto_run_active(False)

    def _refresh_auto_run_summary(self) -> None:




        try:
            word = self.auto_prompt_input.text().strip()
            items = (list(getattr(self, "_auto_exemplar_items", []))
                     if self._EXEMPLARS_ENABLED else [])
            chips = []
            for idx, it in enumerate(items):
                thumb = it[2] if len(it) > 2 else None
                chips.append(self._make_exemplar_chip(
                    it[0], it[1], idx + 1, thumb, removable=False,
                    side_px=RUN_REFERENCE_PX))
            self.auto_run_summary_card.set_run_recipe(word, chips)
            self.auto_run_summary_card.setVisible(bool(word or chips))
        except (RuntimeError, AttributeError):
            pass



    def _refresh_auto_exemplar_explainer(self, slot_taken: bool = False) -> None:








        from .guidance import HINT_EXEMPLAR_TIP, is_hint_dismissed
        try:
            show = not slot_taken and not getattr(self, "_auto_exemplar_count", 0)
            show = show and not is_hint_dismissed(HINT_EXEMPLAR_TIP)


            if getattr(self, "_auto_prompt_canopy", False):
                self.auto_exemplar_explainer.set_body_text(
                    tr("Shadows getting detected instead of trees? Use "
                       "'Exclude a look-alike' on one shadow - the AI "
                       "drops similar false positives."))
            elif self._auto_credits is not None and not self._auto_is_subscriber:


                self.auto_exemplar_explainer.set_body_text(
                    tr("The AI finds every object that looks like your "
                       "example. Free includes one example per run."))
            else:
                from ...core.exemplar_store import max_total
                self.auto_exemplar_explainer.set_body_text(
                    tr("The AI finds every object that looks like your "
                       "examples - you can draw up to {max}.")
                    .replace("{max}", str(max_total())))
            self.auto_exemplar_explainer.setVisible(show)
        except (RuntimeError, AttributeError):
            pass

    def _set_auto_exemplar_expanded(self, expanded: bool) -> None:



        self._auto_exemplar_expanded = True
        self.auto_exemplar_content.setVisible(True)

    def _on_auto_cancel_clicked(self) -> None:










        self.arm_auto_cancel_confirm()

    def set_auto_cancelling(self) -> None:









        self._auto_cancelling = True
        try:
            self.auto_cancel_btn.setEnabled(False)
            self.auto_cancel_btn.setText(tr("Stopping..."))
        except (RuntimeError, AttributeError):
            pass



        try:
            if self.auto_progress_card.isVisible():
                self.auto_progress_label.setText(
                    tr("Stopping - keeping everything already found..."))
                self.auto_progress_label.setVisible(True)
        except (RuntimeError, AttributeError):
            pass







        for _w in (self.auto_cancel_btn, self.auto_progress_label):
            try:
                _w.repaint()
            except (RuntimeError, AttributeError):
                pass







    def set_auto_export_success(self, count: int, layer_name: str,
                                object_word=None, layer_id=None) -> None:



        self.show_export_success_line(
            "auto", count, layer_name, object_word=object_word,
            layer_id=layer_id)

    def show_export_success_line(self, mode: str, count: int, layer_name: str,
                                 object_word=None, layer_id=None) -> None:





        manual = str(mode) == "manual"
        try:
            lbl = getattr(
                self, "manual_export_success" if manual
                else "auto_export_success", None)
            if lbl is None:
                return
            from .auto_recap import auto_export_success_html
            self._auto_recap_layer_id = layer_id or ""
            lbl.setText(msg_rich("success", auto_export_success_html(
                count, object_word or "", layer_name or "",
                linked=bool(layer_id)), is_html=True))
            lbl.setToolTip(
                tr("Click the layer name to see it on the map")
                if layer_id else "")
            lbl.setVisible(True)
        except Exception:  # nosec B110
            pass
        if count > 0:

            self.show_pro_after_success("manual" if manual else "auto")

    def clear_manual_export_success(self) -> None:


        try:
            lbl = getattr(self, "manual_export_success", None)
            if lbl is not None:
                lbl.setVisible(False)
        except (RuntimeError, AttributeError):
            pass

    def clear_auto_export_success(self) -> None:


        try:
            lbl = getattr(self, "auto_export_success", None)
            if lbl is not None:
                lbl.setVisible(False)
        except (RuntimeError, AttributeError):
            pass
        self.hide_pro_after_success()

    def _on_auto_recap_link(self, _href: str) -> None:



        try:
            from qgis.core import QgsProject
            from qgis.utils import iface
            layer_id = getattr(self, "_auto_recap_layer_id", "")
            layer = QgsProject.instance().mapLayer(layer_id) if layer_id else None
            if layer is None or iface is None:
                return
            iface.setActiveLayer(layer)
            _show_layer_in_layers_panel(iface, layer)
            iface.zoomToActiveLayer()
        except Exception:  # nosec B110
            pass


def _show_layer_in_layers_panel(iface, layer) -> None:







    try:
        from qgis.PyQt.QtWidgets import QDockWidget
        window = iface.mainWindow()
        dock = window.findChild(QDockWidget, "Layers") if window is not None else None
        if dock is not None:
            dock.setVisible(True)
            dock.raise_()
        view = iface.layerTreeView()
        if view is not None:
            view.setCurrentLayer(layer)
    except (RuntimeError, AttributeError, TypeError):

        pass
