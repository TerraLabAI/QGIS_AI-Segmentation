"""A detection run from launch to teardown: the in-run read-only panel, the
cancel feedback, and the value recap the Start page keeps afterwards.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.
"""
from __future__ import annotations

from ...core.i18n import tr
from .styles import (
    _msg_text,
)


class DockAutoRunLifecycleMixin:
    """A detection run from launch to teardown: the in-run read-only panel, the
    cancel feedback, and the value recap the Start page keeps afterwards."""

    def set_auto_run_active(self, active: bool) -> None:
        self._auto_run_active = active
        if active:
            # A fresh run cannot inherit a stale hand-over hold (see
            # set_auto_finalizing): that would keep the prompt step away for
            # good if a previous finalize died without releasing it.
            self._auto_finalizing = False
        # The run is over but the results are still being turned into the
        # review: keep the run screen exactly as the last tile left it. Without
        # this the pre-run controls came back for the whole finalize (which
        # yields to the event loop), so the user watched the prompt step flash
        # by before the review opened.
        hold = (not active) and getattr(self, "_auto_finalizing", False)
        self.auto_cancel_btn.setVisible(active)
        # A fresh run clears any leftover exhausted-credits subscribe link and
        # restores the cancel button from a previous run's "Stopping…" state
        # (set_auto_cancelling disables + relabels it; the button is hidden but
        # not reset when the run winds down).
        if active:
            self.set_auto_exhausted_subscribe_visible(False)
            self._auto_cancelling = False
            self.auto_cancel_btn.setEnabled(True)
            self.auto_cancel_btn.setText(tr("Cancel detection"))
        # Lock the gear (Account Settings) and help controls while tiles are in
        # flight: opening either fires its own network work on top of an already
        # busy GUI thread, which compounds the freeze. They unlock the moment the
        # run ends (review or back to Start).
        self.set_footer_controls_locked(active or hold)
        # Mirror AI Edit: while tiles are in flight, clear away the non-essential
        # params (detail, confidence, cost) and the Detect/Exit row so only the
        # "Detecting X" label + progress + Cancel remain. They reappear when the
        # run ends; if the run then enters review, set_auto_review_active
        # re-hides them. The detail row honors the zone state on restore.
        self.auto_detect_row.setVisible(not (active or hold))
        # The meta intercept belongs to the pre-run setup; a run in flight (or
        # winding down) never shows it.
        if active:
            self._reset_meta_intercept()
        # The confidence box stays hidden in the prompt step (post-run only).
        self.auto_settings_box.setVisible(False)
        self.auto_detail_row.setVisible(
            self._auto_zone_is_set if not (active or hold) else False)
        self.auto_credit_cost_label.setVisible(
            self.auto_credit_cost_label.text() != ""
            if not (active or hold) else False)
        # Keep the prompt card VISIBLE during a run, read-only (AI Edit pattern):
        # the chosen object stays framed above the progress so the user always
        # knows what is being detected, and the Library button stays clickable
        # (view-only). It returns to editable when the run ends; if the run
        # enters review, set_auto_review_active hides the whole card.
        self.auto_prompt_card.setVisible(True)
        self._set_auto_prompt_readonly(active or hold)
        if active:
            self._go_to_auto_step(2)
        elif not hold:
            self._refresh_auto_layer_lock()
        # Keep the drawn reference visible during a run, read-only: hide the
        # add/exclude/remove affordances, keep the thumbnails browsable (click
        # to enlarge). No reference drawn = the whole panel stays hidden. Done
        # AFTER _go_to_auto_step, which force-hides the panel during a run - this
        # is the deliberate in-run exception that keeps the reference on screen.
        has_ref = self._EXEMPLARS_ENABLED and self._auto_positive_exemplars > 0
        if active or hold:
            self._set_exemplar_readonly(True)
            self.auto_exemplar_panel.setVisible(has_ref)
        else:
            self._set_exemplar_readonly(False)
            self.auto_exemplar_panel.setVisible(
                self._EXEMPLARS_ENABLED and self.auto_steps.currentIndex() == 2 and not self._auto_review_active)
        self._update_auto_detect_enabled()
        if active:
            # Reset the live readout for the fresh run.
            self._auto_found_count = 0
            self._auto_progress_pair = (0, 0)
            self._auto_progress_ratio = 0.0
            # The fill's high-water mark is per RUN: carrying the last run's over
            # would open this one on a full bar (see set_auto_tile_progress).
            self._auto_progress_target = 0
            self._auto_progress_shown = 0
            self._auto_progress_dirty = False
            self._auto_progress_phase = "grid"
            # Fresh warming counter + no known queue place yet, so the elapsed
            # readout starts at zero for this run.
            self._auto_queue_position = 0
            self._auto_queue_eta = 0
            self._stop_auto_warming_anim()
            self._stop_auto_progress_ease()
            self.auto_progress_count_label.setText("")
            self.auto_progress_pct_label.setText("")
            self.auto_progress_label.setVisible(False)
        else:
            # Run ended (review / Exit / error): stop both heartbeats.
            self._stop_auto_warming_anim()
            self._stop_auto_progress_ease()

    def set_auto_finalizing(self, finalizing: bool) -> None:
        """Hold the run screen between the last tile and the review.

        Turning a finished run into the review is cooperative work that yields
        to the event loop, so the dock repaints while it runs. Without this
        hold the pre-run controls (prompt box, examples, Detect row) came back
        for that whole stretch and the user saw the setup step flash by before
        the review opened. While the hold is set they stay away and the run
        card keeps the screen; releasing it hands the screen back to the prompt
        step, unless the review already took it.
        """
        finalizing = bool(finalizing)
        if finalizing == getattr(self, "_auto_finalizing", False):
            return
        self._auto_finalizing = finalizing
        if finalizing:
            # A terminal that has something to say (out of credits, a failed
            # run) already put its banner up: one surface at a time, so leave
            # it alone and only hold the pre-run controls away.
            if not self.auto_status_banner.isVisible():
                self._paint_auto_finalize_card()
            self.set_auto_run_active(False)
        elif not self._auto_review_active:
            self.set_auto_run_active(False)

    def _set_auto_prompt_readonly(self, readonly: bool) -> None:
        """Lock the prompt card for the in-run read-only view: the text stays
        crisp and readable (setReadOnly, not disable, so it never greys out),
        the clear button is dropped, and the Library button stays clickable so
        the user can browse the library view-only while tiles are in flight."""
        try:
            self.auto_prompt_input.setReadOnly(readonly)
            self.auto_prompt_input.setClearButtonEnabled(not readonly)
            self.auto_library_btn.setEnabled(True)
            self.auto_library_btn.setToolTip(
                tr("Browse the library (view only while detecting).") if readonly
                else tr("Browse ready-to-use objects with before / after previews."))
        except (RuntimeError, AttributeError):
            pass

    # -- Optional-example section collapse ---------------------------------

    def _refresh_auto_exemplar_explainer(self, armed: bool = False) -> None:
        """The one-line example tip shows only while the section is fresh: an
        armed draw (the instruction line) or an existing reference (the
        thumbnails) replaces it, so the card never stacks guidance. A tip the
        user closed with its x stays closed (DismissibleHint persistence)."""
        from .guidance import HINT_EXEMPLAR_TIP, is_hint_dismissed
        try:
            show = not armed and not getattr(self, "_auto_exemplar_count", 0)
            show = show and not is_hint_dismissed(HINT_EXEMPLAR_TIP)
            # Same widget, one state: a canopy prompt gets the specific,
            # actionable variant (what to exclude) instead of the generic line.
            if getattr(self, "_auto_prompt_canopy", False):
                self.auto_exemplar_explainer.set_body_text(
                    tr("Shadows getting detected instead of trees? Draw an "
                       "'Exclude a look-alike' box on one shadow - the AI "
                       "drops similar false positives."))
            else:
                self.auto_exemplar_explainer.set_body_text(
                    tr("The AI finds every object that looks like your "
                       "examples - you can draw up to 3."))
            self.auto_exemplar_explainer.setVisible(show)
        except (RuntimeError, AttributeError):
            pass

    def _set_auto_exemplar_expanded(self, expanded: bool) -> None:
        """Compat no-op: the example card is always visible now (the collapsed
        dropdown read as noise, not as an option). Callers that auto-opened it
        (armed draw, existing reference, flow reset) need nothing anymore."""
        self._auto_exemplar_expanded = True
        self.auto_exemplar_content.setVisible(True)

    def _set_exemplar_readonly(self, readonly: bool) -> None:
        """Swap the reference panel between its editable form and the in-run
        read-only form: hide the header/hint/buttons, show a quiet caption, and
        hide the per-thumbnail remove x while keeping the thumbnail click-to-
        enlarge alive. Best-effort (the panel may not be built yet)."""
        try:
            self._auto_exemplar_header.setVisible(not readonly)
            self.auto_exemplar_readonly_caption.setVisible(readonly)
            self.auto_exemplar_edit_controls.setVisible(not readonly)
            # In-run the collapse header is gone, so the content card must
            # show on its own (it holds the caption + thumbnails); back in
            # edit mode the user's collapse state resumes.
            self.auto_exemplar_content.setVisible(
                readonly or self._auto_exemplar_expanded)
            layout = self._auto_exemplar_chips_layout
            for i in range(layout.count()):
                w = layout.itemAt(i).widget()
                if w is None:
                    continue
                rb = getattr(w, "_remove_btn", None)
                if rb is not None:
                    rb.setVisible(not readonly)
        except (RuntimeError, AttributeError):
            pass

    def _on_auto_cancel_clicked(self) -> None:
        # Dock-side no-op: the plugin connects this same button to its real
        # cancel handler (request_stop + teardown). Kept so the button has a
        # dock-side slot and future dock-only feedback has a home.
        pass

    def set_auto_cancelling(self) -> None:
        """Instant feedback the moment Cancel is pressed, BEFORE the worker
        thread winds down. The stop is cooperative (the worker checks its flag
        between network events and drains the tiles already in flight), so the
        page cannot flip to the review on the same click; without this the
        button stays 'Cancel detection' and the bar keeps moving, so the click
        reads as ignored. Disable + relabel the button (a second click is a
        no-op anyway) and swap the progress line to a reassuring 'keeping what's
        found' note. The run's terminal handler (_on_auto_cancelled) then flips
        into the review of the salvaged tiles."""
        self._auto_cancelling = True
        try:
            self.auto_cancel_btn.setEnabled(False)
            self.auto_cancel_btn.setText(tr("Stopping..."))
        except (RuntimeError, AttributeError):
            pass
        # Keep the progress card up and say the paid-for tiles are being kept.
        # The _auto_cancelling flag makes set_auto_tile_progress hold this note
        # even as the salvaged tiles tick the count up during the drain.
        try:
            if self.auto_progress_card.isVisible():
                self.auto_progress_label.setText(
                    tr("Stopping - keeping the tiles already found..."))
                self.auto_progress_label.setVisible(True)
        except (RuntimeError, AttributeError):
            pass
        # Paint this feedback on THIS click. setText/setEnabled only schedule a
        # deferred repaint, and the GUI thread is about to churn the in-flight
        # tile-render backlog (each render spins a nested event loop) plus the
        # salvage drain for a couple of seconds, which would starve that paint so
        # the click reads as ignored. A synchronous repaint of just these two
        # widgets shows "Stopping…" now; it paints only them and pumps no input
        # events, so it cannot re-enter the cancel slot or the render handlers.
        for _w in (self.auto_cancel_btn, self.auto_progress_label):
            try:
                _w.repaint()
            except (RuntimeError, AttributeError):
                pass

    def set_footer_controls_locked(self, locked: bool) -> None:
        """Disable the footer gear (Account Settings) and help controls during a
        run. Each opens a dialog/menu that does its own network work, so clicking
        them mid-detection piles onto a busy GUI thread and worsens the freeze;
        they grey out until the run finishes. Best-effort: the footer buttons may
        not exist yet on an early call."""
        busy_tip = tr("Available when detection finishes")
        for btn, ready_tip in (
            (getattr(self, "_settings_btn", None), tr("Settings")),
            (getattr(self, "_help_btn", None), tr("Help / Report a problem")),
        ):
            if btn is None:
                continue
            try:
                btn.setEnabled(not locked)
                btn.setToolTip(busy_tip if locked else ready_tip)
            except (RuntimeError, AttributeError):
                pass

    @staticmethod
    def _format_recap_area(area_km2) -> str:
        """Compact km2 for the recap line: integer at scale, else 1-2 decimals
        so a small town-block zone never reads as a bare '0.0'."""
        try:
            a = float(area_km2 or 0.0)
        except (TypeError, ValueError):
            return "0.00"
        if a >= 10:
            return str(int(round(a)))
        if a >= 1:
            return f"{a:.1f}"
        return f"{a:.2f}"

    def set_last_run_recap(self, count: int, object_word: str, area_km2,
                           credits_used, credits_left=None) -> None:
        """Store the session-only value recap for the Automatic Start page: one
        quiet line summarizing what the last run produced.

        One message per state: right after a Finish the success line already
        tells the whole story, so while it is visible the recap only STORES its
        text and stays hidden; dismissing the success line (next Start click or
        mode switch) reveals it as the session memory. Best-effort by contract
        (the export already committed): never raises, so a recap problem can
        never surface as a failed Finish. Balance is dropped when unknown (the
        post-run usage refresh has not returned yet)."""
        try:
            recap = getattr(self, "auto_last_run_recap", None)
            if recap is None:
                return
            obj = (object_word or tr("object")).strip() or tr("object")
            area = self._format_recap_area(area_km2)
            if credits_left is not None:
                text = tr(
                    "Last run: {count} {object} exported · {area} km² "
                    "· {used} credits used, {left} left"
                ).format(count=count, object=obj, area=area,
                         used=credits_used, left=credits_left)
            else:
                text = tr(
                    "Last run: {count} {object} exported · {area} km² "
                    "· {used} credits used"
                ).format(count=count, object=obj, area=area, used=credits_used)
            recap.setText(text)
            # isHidden (the widget's OWN flag), not isVisible: the latter is
            # False whenever an ancestor is hidden, which would wrongly show
            # both messages when this runs while the dock is not on screen.
            success = getattr(self, "auto_export_success", None)
            recap.setVisible(success is None or success.isHidden())
        except Exception:  # nosec B110 -- recap is best-effort, never break Finish
            pass

    def clear_last_run_recap(self) -> None:
        """Retire the last-run recap card (text included, so a later success
        dismissal cannot resurface a stale run's numbers). Called when a new
        run starts. Safe to call when the card was never built."""
        try:
            recap = getattr(self, "auto_last_run_recap", None)
            if recap is not None:
                recap.setText("")
                recap.setVisible(False)
        except (RuntimeError, AttributeError):
            pass

    def set_auto_export_success(self, count: int, layer_name: str,
                                object_word=None, area_km2=None,
                                credits_used=None) -> None:
        """Show the post-export success line on the Start page: what was saved,
        where, and the run's measured value (km2 / credits) when known, so it
        is the ONE message right after a Finish (the recap card stays hidden
        until this line is dismissed). Set AFTER reset_auto_to_start (which
        clears it), so it survives the return to Start; dismissed on the next
        Start click or mode switch. Best-effort; never raises into a committed
        export."""
        try:
            lbl = getattr(self, "auto_export_success", None)
            if lbl is None:
                return
            obj = (object_word or "").strip() or tr("polygons")
            text = tr('{n} {object} saved to layer "{name}"').format(
                n=count, object=obj, name=layer_name or "")
            if area_km2 is not None:
                text += " · " + tr("{area} km²").format(
                    area=self._format_recap_area(area_km2))
            if credits_used is not None:
                text += " · " + tr("{used} credits used").format(
                    used=credits_used)
            lbl.setText(_msg_text("success", text))
            lbl.setVisible(True)
            # One message per state: the recap card repeats this story, so it
            # waits behind the success line (see clear_auto_export_success).
            recap = getattr(self, "auto_last_run_recap", None)
            if recap is not None:
                recap.setVisible(False)
        except Exception:  # nosec B110 -- success line is best-effort
            pass

    def clear_auto_export_success(self) -> None:
        """Hide the post-export success line (a new Start, a mode switch, any
        reset), and let the stored last-run recap take over as the quiet
        session memory. Safe to call when the labels were never built."""
        try:
            lbl = getattr(self, "auto_export_success", None)
            if lbl is not None:
                lbl.setVisible(False)
            recap = getattr(self, "auto_last_run_recap", None)
            if recap is not None and bool(recap.text()):
                recap.setVisible(True)
        except (RuntimeError, AttributeError):
            pass
