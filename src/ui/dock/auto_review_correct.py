






from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QSpinBox

from ...core.i18n import tr
from ...core.review_defaults import (
    AUTO_REVIEW_POINTS_PCT_DEFAULT as _AUTO_REVIEW_POINTS_PCT_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_SIMPLIFY_DEFAULT as _AUTO_REVIEW_SIMPLIFY_DEFAULT,
)
from ..icons import icon_for
from .auto_correct_build import (
    CORRECT_METHOD_GLYPH,
    EDIT_BRANCH_GLYPH,
    set_branch_glyph,
)
from .correct_summary_row import clear_all_rest_label
from .styles import (
    LINK_INK,
    _msg_label_qss,
    msg_rich,
)
from .widgets import (
    Mode,
)


def add_lane_action_row_visible(keep_available: bool, undo_available: bool) -> bool:












    return bool(keep_available) or bool(undo_available)


class DockAutoReviewCorrectMixin:



    def set_correct_selection(self, count: int) -> None:




        try:
            selected = int(count) > 0
            self._auto_correct_has_selection = selected
            self.auto_correct_select_card.setVisible(selected)
            if not selected:





                self.set_correct_armed_line("")
                self.set_correct_session_active(False)
            self._refresh_correct_panels()
        except (RuntimeError, AttributeError):

            pass





    def get_correct_method(self) -> str:


        return getattr(self, "_correct_method", "ai")

    def set_correct_method(self, method: str) -> None:


        method = "manual" if str(method) == "manual" else "ai"
        self._correct_method = method
        try:
            self.auto_correct_method_switch.set_method(method)
        except (RuntimeError, AttributeError):

            pass
        self._refresh_correct_method_ui()

    def _on_correct_method_toggled(self, method: str) -> None:


        method = "manual" if str(method) == "manual" else "ai"
        self._correct_method = method
        self._refresh_correct_method_ui()


        if bool(getattr(self, "_auto_correct_session_active", False)):
            self.set_correct_armed_line(self._default_correct_armed_line())
        self.auto_correct_method_changed.emit(method)

    def _refresh_correct_method_ui(self) -> None:




        method = getattr(self, "_correct_method", "ai")
        if not getattr(self, "_auto_add_lane_armed", False):
            try:



                if method == "manual":
                    self.auto_add_lane_btn.setText(tr("Draw its corners"))
                else:
                    self.auto_add_lane_btn.setText(tr("Point at it on the map"))
                self.auto_add_lane_btn.setIcon(icon_for(
                    self.auto_add_lane_btn, CORRECT_METHOD_GLYPH[method], 16))
            except (RuntimeError, AttributeError):

                pass
        try:


            manual = method == "manual"
            self.auto_add_lane_method_line.setText(
                tr("You place the corners.") if manual
                else tr("The AI outlines it."))
            self.auto_add_lane_card.setToolTip(
                tr("You place the corners, the same as on any QGIS layer. Free.")
                if manual else
                tr("The AI outlines it. One cloud detection per object."))
        except (RuntimeError, AttributeError):
            pass
        self._apply_shape_only_mode(method)
        self._apply_correct_hero_mode()

    def _default_correct_armed_line(self) -> str:


        if getattr(self, "_correct_method", "ai") == "manual":
            return tr(
                "Drag a corner to move it. Click an edge to add one, "
                "right-click removes.")
        return tr(
            "Left-click adds a keep point, right-click a trim point. The "
            "outline follows.")

    def set_correct_armed_line(self, text: str, kind: str = "armed") -> None:


        try:
            lbl = self.auto_correct_armed_line
        except AttributeError:
            return
        try:
            if not text:
                lbl.setText("")
                lbl.setVisible(False)
                return
            lbl.setStyleSheet(_msg_label_qss(kind))
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setText(msg_rich(kind, text))
            lbl.setVisible(True)
        except (RuntimeError, AttributeError):
            pass

    def set_correct_session_active(self, active: bool) -> None:














        self._auto_correct_session_active = bool(active)
        try:
            self.auto_correct_session_row.setVisible(bool(active))
        except (RuntimeError, AttributeError):
            pass
        self._apply_merge_tile()


        self._apply_step_next_visibility()
        if not active:
            self.set_correct_armed_line("")
        self._refresh_correct_summary_row()

    def set_correct_session_undo_available(self, available: bool) -> None:





        try:
            self.auto_correct_session_undo_btn.setVisible(bool(available))
        except (RuntimeError, AttributeError):

            pass

    def set_correct_class_label(self, text: str) -> None:




        word = str(text or "").strip()
        if word:
            word = word[:1].upper() + word[1:]
        try:
            self.auto_correct_selected_label.setText(
                word or tr("This polygon"))
        except (RuntimeError, AttributeError):

            pass





    def set_add_lane_visible(self, visible: bool) -> None:

        try:
            self.auto_add_lane_card.setVisible(bool(visible))
        except (RuntimeError, AttributeError):
            pass

    def set_add_lane_armed(self, armed: bool, method: str) -> None:







        method = "manual" if str(method) == "manual" else "ai"
        armed = bool(armed)
        if armed and not getattr(self, "_auto_add_lane_armed", False):
            self._auto_add_lane_kept = 0
        self._auto_add_lane_armed = armed
        self._auto_add_lane_method = method
        try:
            from .styles import _BTN_GHOST
            if armed:
                self.auto_add_lane_btn.setStyleSheet(_BTN_GHOST)
                self.auto_add_lane_btn.setText(tr("Stop adding"))
                self.auto_add_lane_btn.setIcon(
                    icon_for(self.auto_add_lane_btn, "close", 14))
                self.auto_add_lane_btn.setToolTip(tr(
                    "Go back to picking polygons. Everything you kept stays, "
                    "and so does the outline on screen."))
            else:
                self.auto_add_lane_btn.setStyleSheet(_BTN_GHOST)
                self.set_add_lane_keep_available(False)
                self.set_add_lane_undo_available(False)






                self.auto_add_lane_btn.setToolTip(tr(
                    "Add an object the AI missed. In AI, point at it and the "
                    "model outlines it for one cloud detection; in Manual, draw "
                    "its corners for free."))
                self._refresh_correct_method_ui()
            self._refresh_add_lane_line()
        except (RuntimeError, AttributeError):
            pass


        self._refresh_correct_selection_hint()

    def set_add_lane_keep_available(self, available: bool) -> None:



        self._auto_add_lane_has_outline = bool(available)
        try:
            self.auto_add_lane_keep_btn.setVisible(bool(available))
        except (RuntimeError, AttributeError):
            pass
        self._refresh_add_lane_action_row()
        self._refresh_add_lane_line()
        self._refresh_correct_selection_hint()

    def set_add_lane_undo_available(self, available: bool) -> None:


        self._auto_add_lane_undo_wanted = bool(available)
        try:
            self.auto_add_lane_undo_btn.setVisible(bool(available))
        except (RuntimeError, AttributeError):
            pass
        self._refresh_add_lane_action_row()

    def _refresh_add_lane_action_row(self) -> None:






        try:
            self.auto_add_lane_action_row.setVisible(add_lane_action_row_visible(
                getattr(self, "_auto_add_lane_has_outline", False),
                getattr(self, "_auto_add_lane_undo_wanted", False)))
        except (RuntimeError, AttributeError):

            pass

    def set_add_lane_progress(self, count: int) -> None:



        try:
            self._auto_add_lane_kept = max(0, int(count))
        except (TypeError, ValueError):
            self._auto_add_lane_kept = 0
        self._refresh_add_lane_line()

    def _refresh_add_lane_line(self) -> None:








        try:
            lbl = self.auto_add_lane_line
        except AttributeError:
            return
        if not getattr(self, "_auto_add_lane_armed", False):
            try:
                lbl.setVisible(False)
            except (RuntimeError, AttributeError):
                pass
            return
        manual = getattr(self, "_auto_add_lane_method", "ai") == "manual"
        kept = int(getattr(self, "_auto_add_lane_kept", 0) or 0)
        if getattr(self, "_auto_add_lane_has_outline", False):
            kind = "armed"
            txt = (tr("Keep this one, or keep placing corners.") if manual
                   else tr("Keep this one, or click again to correct the "
                           "outline."))
        elif manual:
            kind = "armed"
            txt = tr("Click each corner on the map, then Finish the line.")
        else:
            kind = "armed"
            txt = tr("Click an object on the map and the AI outlines it.")
        if kept:
            kind = "success"
            txt += " " + (tr("1 polygon added so far.") if kept == 1
                          else tr("{count} polygons added so far.").format(
                              count=kept))
        try:
            lbl.setStyleSheet(_msg_label_qss(kind))
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setText(msg_rich(kind, txt))
            lbl.setVisible(True)
        except (RuntimeError, AttributeError):

            pass

    def _on_add_lane_clicked(self) -> None:


        if getattr(self, "_correct_method", "ai") == "manual":
            self.auto_add_polygon_requested.emit()
        else:
            self.auto_ai_add_requested.emit()

    def set_merge_available(self, available: bool) -> None:







        self._auto_merge_available = bool(available)
        self._apply_merge_tile()

    def _apply_merge_tile(self) -> None:




        try:
            self.auto_shape_merge_btn.setVisible(
                bool(getattr(self, "_auto_merge_available", False))
                and not bool(getattr(self, "_auto_correct_session_active", False))
                and not bool(getattr(self, "_qgis_bridge_active_ui", False)))
        except (RuntimeError, AttributeError):
            pass





    def set_correct_selection_info(self, text: str) -> None:


        try:
            self.auto_correct_selected_info.setText(str(text or ""))
        except (RuntimeError, AttributeError):

            pass

    def _shape_only_widgets(self) -> dict:



        return {
            "points_pct": self.auto_shape_only_points,
            "simplify_px": self.auto_shape_only_simplify,
            "open_px": self.auto_shape_only_clean,
            "expand_px": self.auto_shape_only_expand,
            "smooth": self.auto_shape_only_smooth,
            "fill_holes": self.auto_shape_only_fill,
            "ortho": self.auto_shape_only_ortho,
        }

    def set_shape_only_values(self, values: dict, overridden: bool) -> None:





        try:
            widgets = self._shape_only_widgets()
        except (RuntimeError, AttributeError):
            return
        for w in widgets.values():
            w.blockSignals(True)
        try:
            for key, widget in widgets.items():
                if key not in values:
                    continue
                raw = values[key]
                if hasattr(widget, "setChecked"):
                    widget.setChecked(bool(raw))
                elif isinstance(widget, QSpinBox):
                    widget.setValue(int(raw))
                else:
                    widget.setValue(float(raw))
            self.auto_shape_only_reset.setVisible(bool(overridden))
            self._auto_shape_only_overridden = bool(overridden)
        except (RuntimeError, AttributeError, TypeError, ValueError):

            pass
        finally:
            for w in widgets.values():
                w.blockSignals(False)
        self._sync_shape_only_right_angles()

    def _sync_shape_only_right_angles(self) -> None:




        ortho = getattr(self, "auto_shape_only_ortho", None)
        if ortho is None or not ortho.isChecked():
            return
        from .right_angles_support import gate_right_angles

        gate_right_angles(
            ortho, getattr(self, "auto_shape_only_ortho_label", None))

    def get_shape_only_values(self) -> dict:

        try:
            widgets = self._shape_only_widgets()
        except (RuntimeError, AttributeError):
            return {"points_pct": _AUTO_REVIEW_POINTS_PCT_DEFAULT,
                    "simplify_px": _AUTO_REVIEW_SIMPLIFY_DEFAULT}
        out: dict = {}
        for key, widget in widgets.items():
            try:
                if hasattr(widget, "isChecked"):
                    out[key] = bool(widget.isChecked())
                elif isinstance(widget, QSpinBox):
                    out[key] = int(widget.value())
                else:
                    out[key] = float(widget.value())
            except (RuntimeError, AttributeError):
                continue
        return out

    def _emit_shape_only_changed(self, control: str, param: str) -> None:







        values = self.get_shape_only_values()
        self.auto_shape_only_changed.emit(values)
        try:
            tracked = getattr(self, "_review_shape_tracked", None)
            if tracked is None:
                tracked = set()
                self._review_shape_tracked = tracked
            if control not in tracked:
                tracked.add(control)
                from ...core import telemetry, telemetry_run_events
                telemetry_run_events.track_review_shape_adjusted(
                    control=control,
                    value=values.get(param),
                    run_id=telemetry.get_last_run_id() or "")
        except Exception:
            pass  # nosec B110

    def _apply_shape_only_mode(self, method: str) -> None:









        manual = str(method) == "manual"
        if manual is not getattr(self, "_auto_shape_only_manual", None):
            self._auto_shape_only_manual = manual
            self._auto_shape_only_expanded = manual
        try:
            self.auto_shape_only_toggle.setVisible(True)
            self.auto_shape_only_scope_line.setText(
                tr("This polygon only. Fewer points means fewer corners to drag.")
                if manual else
                tr("This polygon only. Every other one follows the Shapes step."))
        except (RuntimeError, AttributeError):

            pass
        self._apply_shape_only_toggle()

    def _on_shape_only_toggle_clicked(self) -> None:

        self._auto_shape_only_expanded = not bool(
            getattr(self, "_auto_shape_only_expanded", False))
        self._apply_shape_only_toggle()

    def _apply_shape_only_toggle(self) -> None:





        expanded = bool(getattr(self, "_auto_shape_only_expanded", False))
        manual = bool(getattr(self, "_auto_shape_only_manual", False))
        try:

            self.auto_shape_only_toggle.set_fold_title(
                tr("Simplify this outline first") if manual
                else tr("Clean up this outline"))
            self.auto_shape_only_toggle.set_fold_open(expanded)
            self.auto_shape_only_toggle.setToolTip(
                tr("Thin this outline before you drag its corners, without "
                   "moving the dials that drive the whole layer.") if manual
                else tr("Give this one polygon its own shape settings, without "
                        "moving the dials that drive the whole layer."))




            self.auto_shape_only_box.setVisible(expanded)
        except (RuntimeError, AttributeError):

            pass

    def _should_show_correct_selection_hint(self) -> bool:








        resting = self._mode == Mode.AUTOMATIC and bool(getattr(self, "_auto_review_active", False))
        resting = resting and getattr(self, "_auto_review_step", 0) == 1
        resting = resting and not bool(getattr(self, "_auto_correct_has_selection", False))
        resting = resting and not bool(getattr(self, "_auto_correct_merge_armed", False))
        resting = resting and not bool(getattr(self, "_auto_zero_entry", False))
        return resting and not bool(getattr(self, "_auto_add_lane_armed", False))

    def _refresh_correct_selection_hint(self) -> None:




        try:
            show = self._should_show_correct_selection_hint()
            self.auto_correct_pick_hero.setVisible(show)
            self._apply_correct_hero_mode()
        except (RuntimeError, AttributeError):
            pass

    def _apply_correct_hero_mode(self) -> None:







        method = getattr(self, "_correct_method", "ai")
        try:
            self.auto_correct_gesture_art.set_method(method)
        except (RuntimeError, AttributeError):
            pass
        try:
            if method == "manual":
                line = tr("Click a polygon, then drag any corner.")
            else:
                line = tr("Click a polygon, then click the spot the AI missed.")



            set_branch_glyph(self.auto_correct_pick_glyph, EDIT_BRANCH_GLYPH)
            self.auto_correct_pick_title.setText(
                tr("Edit an existing polygon"))
            self.auto_correct_pick_hint.setText(line)
        except (RuntimeError, AttributeError):

            pass

    def _correct_info_line_gate(self) -> bool:




        resting = self._mode == Mode.AUTOMATIC and bool(getattr(self, "_auto_review_active", False))
        resting = resting and getattr(self, "_auto_review_step", 0) == 1
        resting = resting and not bool(getattr(self, "_auto_correct_has_selection", False))
        return resting and not bool(getattr(self, "_auto_zero_entry", False))

    def _refresh_correct_info_line(self) -> None:


        try:
            hint = self.auto_correct_delete_tip
        except AttributeError:
            return
        try:
            from .guidance import HINT_REVIEW_RIGHT_CLICK_DELETE, is_hint_dismissed
            if (self._correct_info_line_gate()
                    and not is_hint_dismissed(HINT_REVIEW_RIGHT_CLICK_DELETE)):
                hint.show()
            else:
                hint.hide()
        except (RuntimeError, AttributeError):
            pass

    def _refresh_correct_panels(self) -> None:


        selected = bool(getattr(self, "_auto_correct_has_selection", False))
        try:



            self.set_add_lane_visible(not selected)
        except (RuntimeError, AttributeError):

            pass
        self._refresh_correct_selection_hint()
        self._refresh_correct_info_line()

    def enter_ai_reshape_state(self) -> None:















        self._refine_handoff = True
        try:
            self.set_correct_session_active(True)


            self.set_correct_armed_line(self._default_correct_armed_line())


            self.set_correct_session_undo_available(False)
            self._set_review_dials_locked(True, 1)
        except (RuntimeError, AttributeError):
            pass

    def leave_ai_reshape_state(self) -> None:


        self._refine_handoff = False
        try:
            self.set_correct_session_active(False)
            self.set_auto_review_installing(False)
            self.set_auto_review_step(1)
        except (RuntimeError, AttributeError):
            pass

    def set_correct_armed(self, which: str | None) -> None:




        try:
            btn = self.auto_shape_merge_btn
        except AttributeError:
            return
        try:
            from .styles import _BTN_BLUE_OUTLINE, _BTN_GHOST
            armed = (which == "merge")
            self._auto_correct_merge_armed = armed





            btn.setStyleSheet(_BTN_BLUE_OUTLINE if armed else _BTN_GHOST)


            btn.setText(tr("Merge with neighbours"))
            self._refresh_correct_selection_hint()
        except (RuntimeError, AttributeError):
            pass

    def set_correct_status(self, kind: str, text: str,
                           action_text: str = "") -> None:







        try:
            lbl = self.auto_correct_status
        except AttributeError:
            return
        try:
            if not text:
                lbl.setText("")
                lbl.setVisible(False)
                return
            lbl.setStyleSheet(_msg_label_qss(kind))
            if action_text:
                import html
                body = msg_rich(kind, text)
                act = (f'<a href="action" style="color: {LINK_INK};'
                       f' text-decoration: none;">{html.escape(action_text)}</a>')
                rows = [f'<tr><td colspan="2">{body}</td></tr>',
                        f'<tr><td colspan="2">{act}</td></tr>']
                lbl.setTextFormat(Qt.TextFormat.RichText)
                lbl.setText('<table width="100%">' + "".join(rows) + "</table>")
            else:
                lbl.setTextFormat(Qt.TextFormat.RichText)
                lbl.setText(msg_rich(kind, text))
            lbl.setVisible(True)
        except (RuntimeError, AttributeError):

            pass

    def _on_correct_status_link(self, href: str) -> None:

        if href == "action":
            self.auto_correct_status_action_requested.emit()

    def set_correction_summary(self, count: int) -> None:




        self._correction_summary_count = max(0, int(count))
        try:
            if count == 1:
                text = tr("1 correction this round")
            elif count > 1:
                text = tr("{n} corrections this round").format(n=count)
            else:
                text = ""
            if text:
                self.auto_correct_summary_label.setText(text)




            guard = getattr(self, "_correct_clear_confirm", None)
            if guard is not None:
                guard.set_count(count)
            else:
                self.auto_correct_clear_btn.setText(
                    clear_all_rest_label(count))
        except (RuntimeError, AttributeError):

            pass
        self._refresh_correct_summary_row()

    def _refresh_correct_summary_row(self) -> None:

        try:
            self.auto_correct_summary_row.setVisible(
                int(getattr(self, "_correction_summary_count", 0)) > 0 and not bool(getattr(
                    self, "_auto_correct_session_active", False)))
        except (RuntimeError, AttributeError):
            pass

    def set_zero_detection_entry(self, active: bool) -> None:




        self._auto_zero_entry = bool(active)
        try:
            if active:
                self.set_auto_review_step(1)
                self.auto_correct_zero_line.setVisible(True)
                self.auto_step_next_btn.setVisible(False)
            else:
                self.auto_correct_zero_line.setVisible(False)

            self.auto_review_view_row.setVisible(
                bool(getattr(self, "_auto_review_active", False)) and not active)
            self._refresh_correct_panels()
        except (RuntimeError, AttributeError):

            pass
