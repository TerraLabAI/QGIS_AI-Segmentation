"""A detection run from launch to teardown: the in-run receipt card, the
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
    """A detection run from launch to teardown: the in-run receipt card, the
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
            # A fresh run opens on imagery and on a link assumed healthy: the
            # previous run's phase and slow-link note must not carry over.
            self._auto_wait_phase = "imagery"
            self._auto_link_slow = False
            self._auto_link_local = False
        # The gear (Account Settings) and the help menu stay clickable during a
        # run. Neither blocks the GUI thread: the account dialog fetches on a
        # task thread, the help entries are local, and the one destructive
        # action it offers (removing the downloaded AI data) already refuses
        # while a run is live (is_local_ai_busy covers _auto_worker).
        # Mirror AI Edit: while tiles are in flight, clear away the non-essential
        # params (detail, confidence, cost) and the Detect/Exit row so only the
        # receipt + progress + Cancel remain. They reappear when the run ends;
        # if the run then enters review, set_auto_review_active re-hides them.
        # The detail row honors the zone state on restore.
        self.auto_detect_row.setVisible(not (active or hold))
        # The confidence box stays hidden in the prompt step (post-run only).
        self.auto_settings_box.setVisible(False)
        self.auto_detail_row.setVisible(
            self._auto_zone_is_set if not (active or hold) else False)
        self.auto_credit_cost_label.setVisible(
            self.auto_credit_cost_label.text() != ""
            if not (active or hold) else False)
        # A run takes no input, so the two setup cards go and the receipt takes
        # their place: the word, the references, or both, in one card that
        # cannot be typed into. The "and / or" separator goes with them, since
        # it joins two choices and the choice is already made (the separator
        # follows the example card on its own, see ExampleCardWithSeparator).
        # The cards come back when the run ends; if the run enters review,
        # set_auto_review_active keeps them away.
        in_run = active or hold
        self.auto_prompt_card.setVisible(not in_run)
        if in_run:
            self._refresh_auto_run_summary()
        else:
            self.auto_run_summary_card.setVisible(False)
        if active:
            self._go_to_auto_step(2)
        elif not hold:
            self._refresh_auto_layer_lock()
        # Done AFTER _go_to_auto_step, which drives the same panel.
        if in_run:
            self.auto_exemplar_panel.setVisible(False)
        else:
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
            # A fresh hand-over knows nothing about its fold yet: a count left
            # over from the last run must not show on this one's card.
            self._auto_finalize_tiles = (0, 0)
            # A terminal that has something to say (out of credits, a failed
            # run) already put its banner up: one surface at a time, so leave
            # it alone and only hold the pre-run controls away.
            if not self.auto_status_banner.isVisible():
                self._paint_auto_finalize_card()
            self.set_auto_run_active(False)
        elif not self._auto_review_active:
            self.set_auto_run_active(False)

    def _refresh_auto_run_summary(self) -> None:
        """Fill the in-run receipt from what the setup cards hold right now and
        show it. Nothing given (no word, no reference) leaves the card away
        rather than showing an empty header. Best-effort: a run must never die
        on its own recap."""
        try:
            word = self.auto_prompt_input.text().strip()
            items = (list(getattr(self, "_auto_exemplar_items", []))
                     if self._EXEMPLARS_ENABLED else [])
            chips = []
            for idx, it in enumerate(items):
                thumb = it[2] if len(it) > 2 else None
                chips.append(self._make_exemplar_chip(
                    it[0], it[1], idx + 1, thumb, removable=False))
            self.auto_run_summary_card.set_run_recipe(word, chips)
            self.auto_run_summary_card.setVisible(bool(word or chips))
        except (RuntimeError, AttributeError):
            pass

    # -- Optional-example section collapse ---------------------------------

    def _refresh_auto_exemplar_explainer(self, slot_taken: bool = False) -> None:
        """The one-line example tip shows only while the section is fresh: an
        armed draw (the instruction line) or an existing reference (the
        thumbnails) replaces it, so the card never stacks guidance. A tip the
        user closed with its x stays closed (DismissibleHint persistence).

        ``slot_taken`` says something else already holds the card's one line;
        it is not an armed state of its own.
        """
        from .guidance import HINT_EXEMPLAR_TIP, is_hint_dismissed
        try:
            show = not slot_taken and not getattr(self, "_auto_exemplar_count", 0)
            show = show and not is_hint_dismissed(HINT_EXEMPLAR_TIP)
            # Same widget, one state: a canopy prompt gets the specific,
            # actionable variant (what to exclude) instead of the generic line.
            if getattr(self, "_auto_prompt_canopy", False):
                self.auto_exemplar_explainer.set_body_text(
                    tr("Shadows getting detected instead of trees? Use "
                       "'Exclude a look-alike' on one shadow - the AI "
                       "drops similar false positives."))
            elif self._auto_credits is not None and not self._auto_is_subscriber:
                # Known free plan: say the ceiling up front, so the offer that
                # meets the second example is never a surprise.
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
        """Compat no-op: the example card is always visible now (the collapsed
        dropdown read as noise, not as an option). Callers that auto-opened it
        (armed draw, existing reference, flow reset) need nothing anymore."""
        self._auto_exemplar_expanded = True
        self.auto_exemplar_content.setVisible(True)

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
                    tr("Stopping - keeping everything already found..."))
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

    # set_last_run_recap and clear_last_run_recap were removed with the card
    # they drove. What a finished run produced is in the legend and on the
    # footer credit ring; the Start page is about the next run. The
    # success line below is what survives, because it answers "where did it go"
    # at the one moment the user asks it.

    def set_auto_export_success(self, count: int, layer_name: str,
                                object_word=None, layer_id=None) -> None:
        """Show the post-export success line on the Start page: how many objects
        were saved and the layer they went to, as a link that frames it on the
        map. The one message on the page right after a Finish. Set AFTER
        reset_auto_to_start (which clears it), so it survives the return to
        Start; dismissed on the next Start click or mode switch. Best-effort;
        never raises into a committed export."""
        try:
            lbl = getattr(self, "auto_export_success", None)
            if lbl is None:
                return
            from .auto_recap import auto_export_success_html
            self._auto_recap_layer_id = layer_id or ""
            lbl.setText(_msg_text("success", auto_export_success_html(
                count, object_word or "", layer_name or "",
                linked=bool(layer_id))))
            lbl.setToolTip(
                tr("Click the layer name to see it on the map")
                if layer_id else "")
            lbl.setVisible(True)
        except Exception:  # nosec B110 -- success line is best-effort
            pass

    def clear_auto_export_success(self) -> None:
        """Hide the post-export success line (a new Start, a mode switch, any
        reset). Safe to call when the label was never built."""
        try:
            lbl = getattr(self, "auto_export_success", None)
            if lbl is not None:
                lbl.setVisible(False)
        except (RuntimeError, AttributeError):
            pass

    def _on_auto_recap_link(self, _href: str) -> None:
        """Reveal the layer the last run exported to: make it the active layer,
        show it selected in the Layers panel, and frame it. A layer removed
        since the export resolves to nothing, so the click is simply ignored."""
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
        except Exception:  # nosec B110 -- a recap click must never raise
            pass


def _show_layer_in_layers_panel(iface, layer) -> None:
    """Bring the Layers panel on screen with ``layer`` as its current row.

    Framing the layer alone tells the user nothing when the panel is closed
    or sits behind another tab: the canvas moves and the new layer stays out
    of sight. Reopen a closed panel, raise a tabified one, then select the row
    so the eye lands on the layer the recap names.
    """
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
