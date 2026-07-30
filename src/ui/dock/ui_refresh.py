"""Install progress, session state, instructions, exemplar chips, cleanup.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.
"""
from __future__ import annotations

from qgis.core import QgsProject, QgsRasterLayer
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QFrame,
    QLabel,
    QToolButton,
    QWidget,
)

from ...core.activation_manager import (
    has_tos_accepted,
    has_tos_locked,
)
from ...core.i18n import tr
from .font_scale import scale_qss_font_px, widget_pixel_ratio
from .guidance import (
    HINT_EXEMPLAR_DRAW_BOX,
    HINT_EXEMPLAR_EXCLUDE_BOX,
    HINT_TRY_AUTOMATIC,
    HINT_TUTORIAL_FIRST_STEPS,
    is_hint_dismissed,
)
from .styles import (
    _BTN_EXPORT_DISABLED,
    _BTN_EXPORT_READY,
    _INSTRUCTIONS_CARD_QSS,
    _INSTRUCTIONS_HINT_QSS,
    BRAND_GREEN,
    ERROR_TEXT,
    SUCCESS_TEXT,
    _msg_label_qss,
    _msg_text,
)
from .widgets import (
    Mode,
)


class DockStateMixin:
    """Install progress, session state, instructions, exemplar chips, cleanup."""

    # ---- End Automatic mode helpers -------------------------------------------

    def set_dependency_status(self, ok: bool, message: str):
        self._dependencies_ok = ok

        if ok:
            self.setup_status_label.setText(message)
            self.setup_status_label.setVisible(True)
            self.setup_status_label.setStyleSheet("font-weight: bold; color: palette(text);")
            self.install_button.setVisible(False)
            self.cancel_toggle.setVisible(False)
            self.cancel_button.setVisible(False)
            self.setup_progress.setVisible(False)
            self.setup_progress_label.setVisible(False)
        else:
            is_update = "updating" in message.lower() or "upgrading" in message.lower()
            is_dll_error = "dll" in message.lower() and "failed" in message.lower()
            if is_dll_error:
                short_msg = tr(
                    "Missing Visual C++ Redistributable. "
                    "Install it, restart your computer, then click Retry.")
                self.setup_status_label.setText(short_msg)
                self.setup_status_label.setStyleSheet(
                    f"font-weight: bold; color: {ERROR_TEXT};")
                self.setup_status_label.setVisible(True)
                self.install_button.setText(tr("Retry"))
            else:
                self.setup_status_label.setVisible(False)
                if is_update:
                    self.install_button.setText(tr("Update"))
                else:
                    self.install_button.setText(tr("Install"))
            self.install_button.setVisible(True)
            self.install_button.setEnabled(True)
            self.setup_group.setVisible(True)
            # The setup section is now the interactive content: remember it so a
            # switch to Automatic and back restores it (see _setup_section_wanted).
            self._setup_section_wanted = True

        self._update_full_ui()

    def set_install_progress(self, percent: int, message: str):
        """Unified progress for deps install + model download."""
        import time

        # Mirror progress into the review's inline install banner while a
        # background install runs from the Automatic review (D1). Harmless no-op
        # otherwise. The Manual setup group below stays hidden in Automatic mode.
        if getattr(self, "_auto_review_installing", False):
            try:
                self.auto_review_install_progress.setValue(
                    max(0, min(100, int(percent))))
                if message:
                    self.auto_review_install_label.setText(message)
            except (RuntimeError, AttributeError):
                pass

        self._target_progress = percent

        time_info = ""
        now = time.time()
        if percent > 10 and percent < 100 and self._install_start_time:
            elapsed = now - self._install_start_time
            if elapsed > 5:
                overall_speed = percent / elapsed
                remaining_pct = 100 - percent

                has_prev = self._last_percent_time is not None
                pct_increased = percent > self._last_percent
                time_increased = now > self._last_percent_time if has_prev else False
                if has_prev and pct_increased and time_increased:
                    dt = now - self._last_percent_time
                    dp = percent - self._last_percent
                    recent_speed = dp / dt
                    blended_speed = 0.7 * recent_speed + 0.3 * overall_speed
                else:
                    blended_speed = overall_speed

                if blended_speed > 0:
                    remaining = remaining_pct / blended_speed
                    max_remaining = 480
                    remaining = min(remaining, max_remaining)
                    if remaining > 60:
                        time_info = f" (~{int(remaining / 60)} min left)"
                    elif remaining > 10:
                        time_info = f" (~{int(remaining)} sec left)"

        if percent > self._last_percent:
            self._last_percent_time = now
            self._last_percent = percent

        self.setup_progress_label.setText(f"{message}{time_info}")

        is_update = self.install_button.text() in (
            tr("Update"), tr("Updating..."))

        if percent == 0:
            # An install just started: the setup section owns the interactive
            # content until it completes, so a mode round trip must restore it.
            self._setup_section_wanted = True
            self._install_start_time = time.time()
            self._current_progress = 0
            self._last_percent = 0
            self._last_percent_time = None
            self._creep_counter = 0
            self.setup_progress.setValue(0)
            self.setup_progress.setVisible(True)
            self.setup_progress_label.setVisible(True)
            self.cancel_toggle.setVisible(True)
            self.cancel_toggle.setArrowType(Qt.ArrowType.RightArrow)
            self.cancel_button.setVisible(False)
            self.install_button.setVisible(False)
            self.setup_status_label.setVisible(False)
            self.welcome_title.setText(tr("Installing AI Segmentation..."))
            self._progress_timer.start(500)
        elif percent >= 100 or "cancel" in message.lower() or "failed" in message.lower():
            self._progress_timer.stop()
            self._install_start_time = None
            self.setup_progress.setValue(percent)
            self.setup_progress.setVisible(False)
            self.setup_progress_label.setVisible(False)
            self.cancel_toggle.setVisible(False)
            self.cancel_button.setVisible(False)
            self.install_button.setVisible(True)
            self.install_button.setEnabled(True)
            if is_update:
                self.install_button.setText(tr("Update"))
            else:
                self.install_button.setText(tr("Install"))
            if "cancel" in message.lower():
                self.setup_status_label.setVisible(True)
                self.setup_status_label.setText(tr("Installation cancelled"))
                self.welcome_title.setText(tr("Click Install to set up AI Segmentation"))
            elif "failed" in message.lower():
                self.setup_status_label.setVisible(True)
                self.setup_status_label.setText(tr("Installation failed"))
                self.welcome_title.setText(tr("Click Install to set up AI Segmentation"))
            else:
                self.welcome_title.setText(tr("Click Install to set up AI Segmentation"))
        else:
            # Intermediate tick (1..99). The deps phase hides the bar via
            # set_dependency_status once deps are validated; the very next
            # callback then enters the model-download/load phase but never
            # re-shows the progress UI, so users sat on "Dependencies ready"
            # for several minutes with no visible activity. Re-asserting
            # visibility here keeps the hand-off seamless.
            self.setup_progress.setVisible(True)
            self.setup_progress_label.setVisible(True)
            if self._current_progress < percent:
                self._current_progress = percent
                self.setup_progress.setValue(percent)
            msg_lower = message.lower() if message else ""
            if "loading" in msg_lower and "model" in msg_lower:
                self.welcome_title.setText(tr("Loading AI model..."))
            elif "downloading" in msg_lower and "model" in msg_lower:
                self.welcome_title.setText(tr("Downloading AI model..."))
            elif "verifying" in msg_lower:
                self.welcome_title.setText(tr("Verifying installation..."))

    def _on_progress_tick(self):
        """Animate progress bar smoothly between updates."""
        if self._current_progress < self._target_progress:
            step = max(1, (self._target_progress - self._current_progress) // 3)
            self._current_progress = min(
                self._current_progress + step, self._target_progress)
            self._creep_counter = 0
        elif self._current_progress < 99 and self._target_progress > 0:
            self._creep_counter += 1
            if self._creep_counter >= 4:
                self._creep_counter = 0
                if self._current_progress < self._target_progress + 5:
                    self._current_progress += 1

        self.setup_progress.setValue(self._current_progress)

    def set_checkpoint_status(self, ok: bool, message: str):
        self._checkpoint_ok = ok
        if ok:
            self.setup_status_label.setText(message)
            self.setup_status_label.setStyleSheet("font-weight: bold; color: palette(text);")
            self.setup_group.setVisible(False)
            # Model ready = setup done: the setup section is no longer the
            # interactive content, so a later mode round trip must not resurrect it.
            self._setup_section_wanted = False
        self._update_full_ui()

    def set_segmentation_active(self, active: bool, layer=None):
        self._segmentation_active = active

        # Track which layer we're segmenting. The caller may pass the layer
        # explicitly (the Refine-in-Manual handoff starts on the resolved run
        # raster, which the combo may not yet reflect); fall back to the combo
        # for standalone Manual. Keeps MCP/other callers backward-compatible.
        if active:
            layer = layer or self.layer_combo.currentLayer()
            self._segmentation_layer_id = layer.id() if layer else None
        else:
            self._segmentation_layer_id = None

        self._update_button_visibility()
        self._update_ui_state()
        # Starting/stopping a Manual session crosses the Start screen boundary,
        # so re-evaluate whether the mode switch belongs on screen.
        self._refresh_mode_switch_visibility()
        if active:
            self._update_instructions()

    def _update_button_visibility(self):
        if self._segmentation_active:
            # Hide label, lock combo (grayed out, no dropdown arrow)
            self.layer_label.setVisible(False)
            self.layer_combo.setEnabled(False)
            self.layer_combo.setStyleSheet(
                "QComboBox { color: palette(text); }"
                "QComboBox::drop-down { width: 0px; border: none; }"
            )

            self.start_container.setVisible(False)
            # In a refine handoff the state card IS the guidance (and carries
            # the action buttons); the instructions label belongs to base Manual.
            self.instructions_label.setVisible(not self._refine_handoff)
            self._update_instructions()

            # Refine panel visibility
            self._update_refine_panel_visibility()

            # Save/Undo/Stop are base-Manual controls. In a refine handoff the
            # state card owns every action (Keep / Edit shape / Undo click /
            # Remove), so the legacy rows stay hidden for the whole handoff.
            save_visible = not self._refine_handoff
            self.save_mask_button.setVisible(save_visible)
            self.save_mask_button.setEnabled(self._has_mask)

            # Export button: visible during segmentation, EXCEPT in a refine
            # handoff, where committing goes through Back to review -> Finish (a
            # direct manual export would orphan the held Automatic review).
            self.export_button.setVisible(not self._refine_handoff)
            self._update_export_button_style()

            secondary_visible = not self._refine_handoff
            self.secondary_buttons_widget.setVisible(secondary_visible)
            self.undo_button.setVisible(secondary_visible)
            has_points = self._positive_count > 0 or self._negative_count > 0
            can_undo_saved = self._saved_polygon_count > 0
            self.undo_button.setEnabled(has_points or can_undo_saved)
            # Stop discards the manual session; in a refine handoff that would
            # bypass the return to review, so hide it (Back to review is the exit).
            self.stop_button.setVisible(not self._refine_handoff)
            self.stop_button.setEnabled(True)

            # Info box ("one element at a time / save before next"). In a refine
            # handoff the blue refine hint already says this, so hide it to avoid
            # two stacked hints saying the same thing.
            self.batch_info_widget.setVisible(not self._refine_handoff)
        else:
            # Not segmenting - show label, unlock combo, restore dropdown arrow
            self.layer_label.setVisible(True)
            self.layer_combo.setEnabled(True)
            self.layer_combo.setStyleSheet("QComboBox { color: palette(text); }")

            self.start_container.setVisible(True)
            self.instructions_label.setVisible(False)
            self.refine_group.setVisible(False)
            self.save_mask_button.setVisible(False)
            self.export_button.setVisible(False)
            self.undo_button.setVisible(False)
            self.stop_button.setVisible(False)
            self.secondary_buttons_widget.setVisible(False)
            self.batch_info_widget.setVisible(False)

        # The bottom-pinned nudges follow the Start view: they live outside
        # start_container (in main_layout, above the footer) so their visibility
        # must be driven explicitly whenever a session starts/stops.
        self._update_try_automatic_hint_visibility()

    def _should_show_try_automatic(self) -> bool:
        """True only on the Manual Start view: signed in, Manual (Interactive)
        mode, setup complete, no active session, hint not dismissed.

        The Try-Automatic nudge belongs to Manual ONLY - it must NEVER appear in
        the Automatic section, which the mode check enforces.
        """
        setup_complete = self._dependencies_ok and self._checkpoint_ok
        is_manual_mode = self._mode == Mode.INTERACTIVE
        base_ready = self._plugin_activated and is_manual_mode and setup_complete and not self._segmentation_active
        # Empty state shows ONLY the hero (one info per state): the
        # cross-sell band waits until imagery exists, like the Automatic
        # tutorial gate below.
        has_rasters = self.layer_combo.count_layers() > 0
        return bool(base_ready and has_rasters and not is_hint_dismissed(HINT_TRY_AUTOMATIC))

    def _update_try_automatic_hint_visibility(self):
        """Drive the bottom-pinned Try-Automatic nudge from its Start-view gate.

        The band is pinned to the dock bottom (main_layout, above the footer),
        so unlike start_container it is not naturally hidden by mode/session
        state and must be gated explicitly.
        """
        hint = getattr(self, "try_automatic_hint", None)
        if hint is not None:
            hint.setVisible(self._should_show_try_automatic())

    def _should_show_auto_tutorial(self) -> bool:
        """True only on the Automatic Start step: signed in, Automatic mode, on
        step 0, hint not dismissed. Hidden mid-flow, during a run/review, and in
        Manual mode."""
        if not (self._plugin_activated and self._mode == Mode.AUTOMATIC):
            return False
        if is_hint_dismissed(HINT_TUTORIAL_FIRST_STEPS):
            return False
        try:
            # The empty state shows ONLY the hero card (one info per state):
            # the guide banner waits until imagery exists.
            if self.auto_layer_combo.count_layers() == 0:
                return False
        except (RuntimeError, AttributeError):
            pass
        try:
            return self.auto_steps.currentIndex() == 0
        except (RuntimeError, AttributeError):
            return not getattr(self, "_auto_started", False)

    def _update_auto_tutorial_banner_visibility(self):
        """Drive the bottom-pinned first-steps guide banner from its gate."""
        banner = getattr(self, "auto_tutorial_banner", None)
        if banner is not None:
            banner.setVisible(self._should_show_auto_tutorial())

    def _update_refine_panel_visibility(self):
        """Update refine panel visibility based on mask state."""
        if not self._segmentation_active:
            self.refine_group.setVisible(False)
            return

        if self._refine_handoff:
            # "Shape settings" belong to the OPEN edit only (the geometry-delta
            # edit has no SAM points, so _has_mask alone would keep the panel
            # hidden forever). A mere selection shows just Edit shape / Remove.
            self.refine_group.setVisible(self._has_mask or getattr(self, "_handoff_editing", False))
            return

        self.refine_group.setVisible(self._has_mask)

    def _update_export_button_style(self):
        count = self._saved_polygon_count
        if count > 1:
            self.export_button.setText(
                tr("Export {count} polygons to a layer").format(count=count)
            )
        else:
            self.export_button.setText(tr("Export polygon to a layer"))

        if count > 0:
            self.export_button.setEnabled(True)
            self.export_button.setStyleSheet(_BTN_EXPORT_READY)
            self.export_button.setToolTip(
                tr("Writes a GeoPackage layer with your {n} kept polygons.").format(
                    n=count))
        else:
            self.export_button.setEnabled(False)
            self.export_button.setStyleSheet(_BTN_EXPORT_DISABLED)
            self.export_button.setToolTip("")

    def set_point_count(self, positive: int, negative: int):
        self._positive_count = positive
        self._negative_count = negative
        total = positive + negative
        has_points = total > 0
        old_has_mask = self._has_mask
        self._has_mask = has_points

        # Undo enabled if: has points OR has saved masks
        can_undo_saved = self._saved_polygon_count > 0
        self.undo_button.setEnabled((has_points or can_undo_saved) and self._segmentation_active)
        self.save_mask_button.setEnabled(has_points)

        if self._segmentation_active:
            self._update_instructions()
            self._update_export_button_style()
            if old_has_mask != self._has_mask:
                self._update_refine_panel_visibility()
                # Opening/closing an object flips whether Save/Undo apply; in a
                # handoff that also flips their visibility, so re-assert it.
                if self._refine_handoff:
                    self._update_button_visibility()

    def _set_instructions_style(self, style: str) -> None:
        """Restyle the instructions label. Three looks, one label:

        - "card": the framed Manual guidance, the resting look;
        - "compact": the muted hint used for refine-handoff status lines;
        - "waiting": the info-tinted line the plugin shows while it reads
          imagery. Same taxonomy kind as the review's Correct step uses for
          the same wait, so a user who learned one reads the other.

        No-op when the label already wears the requested look."""
        if getattr(self, "_instructions_style", "card") == style:
            return
        self._instructions_style = style
        if style == "waiting":
            self.instructions_label.setStyleSheet(_msg_label_qss("info"))
        else:
            self.instructions_label.setStyleSheet(
                _INSTRUCTIONS_HINT_QSS if style == "compact"
                else _INSTRUCTIONS_CARD_QSS)
        self.instructions_label.setMinimumHeight(0 if style == "compact" else 70)

    def set_manual_encoding(self, reading: bool) -> None:
        """Base Manual is reading the imagery around the click, or has stopped.

        That read takes seconds and used to show nothing but an hourglass, so
        the instruction card says what the wait is for. One line, same card, no
        second place to look."""
        if getattr(self, "_manual_encoding", False) == bool(reading):
            return
        self._manual_encoding = bool(reading)
        self._update_instructions()

    def _update_instructions(self):
        """Update instruction text based on current segmentation state."""
        total = self._positive_count + self._negative_count

        if self._refine_handoff:
            # In-place AI reshape: the Correct card carries the guidance and the
            # Done/Undo buttons (auto_correct_build.py); the Manual instruction
            # label stays out of the way. No separate handoff card any more.
            self.instructions_label.setVisible(False)
            return

        if getattr(self, "_manual_encoding", False):
            # The only state where the panel says what the plugin is doing
            # rather than what to do next: the click has been taken and the
            # imagery it needs is being read.
            self._set_instructions_style("waiting")
            self.instructions_label.setText(
                _msg_text("info", tr("Reading the imagery around your click...")))
            return
        self._set_instructions_style("card")
        if total == 0 and self._saved_polygon_count > 0:
            # Already saved polygon(s), encourage next or export
            text = (
                tr("Polygon saved ({n} total). Click another element, or export "
                   "when done.").format(n=self._saved_polygon_count) + "\n\n"
                "\U0001F7E2 " + tr("Left-click to select")
            )
        elif total == 0:
            text = (
                tr("Click on the element you want to segment:") + "\n\n"
                "\U0001F7E2 " + tr("Left-click to select")
            )
        else:
            text = (
                "\U0001F7E2 " + tr("Left-click to add more") + "\n"
                "\u274C " + tr("Right-click to exclude from selection")
            )

        self.instructions_label.setText(text)

    def closeEvent(self, event):
        # T12 (decision locked): a paid Automatic run CONTINUES in the
        # background when the dock is closed; a message-bar notice says how
        # to get back to it. Completion while hidden lands in the review
        # (the flags pin the page for the next reopen).
        if self._auto_run_active:
            try:
                from qgis.utils import iface as _iface
                _iface.messageBar().pushInfo(
                    "AI Segmentation",
                    tr("Detection continues in the background. "
                       "Reopen AI Segmentation to follow it."))
            except Exception:
                pass  # nosec B110
        elif self._refine_handoff:
            # T13: closing mid-reshape folds the AI edits into the held review,
            # then lets the dock hide (the review is there on reopen). The fold
            # also ends the manual session, so the stop-dialog branch below
            # no-ops.
            try:
                self.auto_reshape_done_requested.emit()
            except (TypeError, RuntimeError):
                pass
        # Route the close-button (X) through the existing Stop flow when a
        # session is active, so the user gets the discard-warning dialog
        # instead of silently leaving the map tool armed without a panel.
        if self._segmentation_active:
            self.stop_segmentation_requested.emit()
            if self._segmentation_active:
                event.ignore()
                return
        # Ship any queued telemetry before the dock (and possibly QGIS) goes away.
        try:
            from ...core.telemetry import flush as _telemetry_flush
            _telemetry_flush()
        except Exception:
            pass  # nosec B110
        super().closeEvent(event)

    def reset_session(self):
        self._has_mask = False
        self._segmentation_active = False
        self._segmentation_layer_id = None
        self._saved_polygon_count = 0
        self._positive_count = 0
        self._negative_count = 0
        self._handoff_editing = False
        self._handoff_selected = 0
        self._manual_encoding = False
        self.reset_refine_sliders()
        self._update_button_visibility()
        self._update_ui_state()

    def set_saved_polygon_count(self, count: int):
        self._saved_polygon_count = count
        self._update_refine_panel_visibility()
        self._update_export_button_style()
        # Update undo button state
        if self._segmentation_active:
            has_points = self._positive_count > 0 or self._negative_count > 0
            can_undo_saved = count > 0
            self.undo_button.setEnabled(has_points or can_undo_saved)
        # Keep the instruction copy in sync (the refine-handoff hint depends on
        # the saved/candidate count, e.g. right after the detections import).
        self._update_instructions()

    def set_manual_last_run_recap(self, count: int, area_m2,
                                  layer_name: str | None = None,
                                  layer_id: str | None = None) -> None:
        """Show the session-only value recap on the Manual Start view after a
        successful export: what the session produced and where it went.

        The layer name is a link: clicking it selects the layer and frames it on
        the map, so the user reaches the result without hunting through the
        legend. Best-effort by contract (the export already committed): never
        raises, so a recap problem can never surface as a failed export."""
        try:
            recap = getattr(self, "manual_last_run_recap", None)
            if recap is None:
                return
            from .manual_recap import manual_recap_html
            self._manual_recap_layer_id = layer_id or ""
            recap.setText(manual_recap_html(count, area_m2, layer_name))
            recap.setToolTip(
                tr("Click the layer name to see it on the map")
                if layer_id else "")
            recap.setVisible(True)
        except Exception:  # nosec B110 -- recap is best-effort, never break export
            pass

    def _on_manual_recap_link(self, _href: str) -> None:
        """Reveal the layer the last Manual session exported to: make it the
        active layer and frame it. A layer removed since the export resolves to
        nothing, so the click is simply ignored."""
        try:
            from qgis.core import QgsProject
            from qgis.utils import iface
            layer_id = getattr(self, "_manual_recap_layer_id", "")
            layer = QgsProject.instance().mapLayer(layer_id) if layer_id else None
            if layer is None or iface is None:
                return
            iface.setActiveLayer(layer)
            iface.zoomToActiveLayer()
        except Exception:  # nosec B110 -- a recap click must never raise
            pass

    @staticmethod
    def _is_online_layer(layer) -> bool:
        """Check if a raster layer is an online/remote service."""
        if layer is None or not isinstance(layer, QgsRasterLayer):
            return False
        provider = layer.dataProvider()
        if provider is None:
            return False
        return provider.name() in ("wms", "wmts", "xyz", "arcgismapserver", "wcs")

    def _is_layer_georeferenced(self, layer) -> bool:
        """Check if a raster layer is properly georeferenced. Delegates to the
        shared helper (this copy used to lack the deleted-layer RuntimeError
        guards the plugin's had; one owner ends that drift)."""
        from ..plugin.shared import is_layer_georeferenced
        return is_layer_georeferenced(layer)

    def _update_ui_state(self):
        if self._mode == Mode.INTERACTIVE:
            self._update_ui_state_interactive()
        else:
            self._update_ui_state_automatic()

    def _update_ui_state_interactive(self):
        """Interactive mode UI state - verbatim original _update_ui_state logic."""
        layer = self.layer_combo.currentLayer()
        has_layer = layer is not None

        has_rasters_available = self.layer_combo.count_layers() > 0
        empty = not has_rasters_available and not self._segmentation_active
        # One info per state: with no raster the Manual page
        # shows ONLY the hero (identical to Automatic), never the header or the
        # Start card behind it. Everything returns once imagery exists. This is
        # the Manual half of the fix that keeps fresh + delete-last-layer paths
        # landing on the same clean empty screen.
        self.no_rasters_widget.setVisible(empty)
        self.layer_combo.setVisible(has_rasters_available)
        self.layer_label.setVisible(not self._segmentation_active and has_rasters_available)
        # start_container (Terms + Start button + caption) is a Start-view thing;
        # hide it in the empty state so only the hero shows. Session visibility is
        # owned elsewhere, so only touch it while no segmentation is active.
        if not self._segmentation_active:
            self.start_container.setVisible(has_rasters_available)

        deps_ok = self._dependencies_ok
        checkpoint_ok = self._checkpoint_ok
        activated = self._plugin_activated
        # Once the ToS lock is set, consent is permanent - skip the accepted check.
        tos_ok = has_tos_locked() or has_tos_accepted()
        can_start = deps_ok and checkpoint_ok and has_layer and activated and tos_ok
        self.start_button.setEnabled(can_start and not self._segmentation_active)

    def _update_ui_state_automatic(self):
        """Automatic mode UI state - update dynamic elements of the auto page."""
        if not self._plugin_activated:
            return
        # Raster-availability gating (no-rasters warning, combo visibility, Start
        # button) is only meaningful BEFORE the flow starts. Once started, the
        # source raster is locked and pinned by id in the run context, independent
        # of layer-tree visibility: toggling a layer's checkbox (e.g. hiding the
        # basemap to peek underneath) must never collapse the header or bounce the
        # user out of a drawn zone / review. So gate only on the Start step.
        # Consent checkbox, shared with Manual (one global state). It lives on
        # the LAST step, right above Detect (friction at the latest possible
        # moment); keep it mirrored (e.g. accepted from
        # Manual) and hide it once consent is sealed.
        try:
            self.auto_tos_checkbox.blockSignals(True)
            self.auto_tos_checkbox.setChecked(has_tos_accepted())
            self.auto_tos_checkbox.blockSignals(False)
            self.auto_tos_container.setVisible(not has_tos_locked())
        except (RuntimeError, AttributeError):
            pass
        if not self._auto_started:
            has_auto_rasters = self.auto_layer_combo.count_layers() > 0
            # One info per state: with no raster the page
            # shows ONLY the hero card. The label and the whole steps stack
            # (Start button + caption) come back once imagery exists.
            self.auto_no_rasters_widget.setVisible(not has_auto_rasters)
            self.auto_layer_combo.setVisible(has_auto_rasters)
            self.auto_layer_label.setVisible(has_auto_rasters)
            self.auto_steps.setVisible(has_auto_rasters)
            # Start only needs a raster to lock; consent is asked at Detect.
            self.auto_start_btn.setEnabled(
                has_auto_rasters and self.auto_layer_combo.currentLayer() is not None)
        self._update_auto_detect_enabled()

    def _update_auto_page_state(self):
        """Update which sub-sections of auto_page are visible based on account state."""
        if not self._plugin_activated:
            self.auto_upsell_card.setVisible(False)
            self.auto_controls_section.setVisible(False)
            return
        # The upsell card replaces the controls only once the lifetime free
        # detections hit zero: at that point the user needs the pricing info
        # to continue, so it earns the space. Before that, the footer ring +
        # Subscribe pill are the only upsell surface.
        exhausted = self._is_free_exhausted()
        self.auto_upsell_card.setVisible(exhausted)
        self.auto_controls_section.setVisible(not exhausted)
        if exhausted:
            try:
                from ...core import telemetry_session_events
                telemetry_session_events.track_pro_upsell_viewed(trigger="free_exhausted")
            except Exception:
                pass  # nosec B110
        # Land on the first incomplete step when (re)entering the page; never
        # yank the user backwards while a run is in flight.
        # Start (not started) -> Draw zone (started, no zone) -> Prompt (zone set).
        if not exhausted and not self._auto_run_active:
            if not self._auto_started:
                self._go_to_auto_step(0)
            elif not self._auto_zone_is_set:
                self._go_to_auto_step(1)
            else:
                self._go_to_auto_step(2)
        self._refresh_auto_credits_display()
        self._update_auto_detect_enabled()

    def _is_free_exhausted(self) -> bool:
        """True when a non-subscriber has confirmed zero free detections left."""
        return not self._auto_is_subscriber and self._auto_free_left is not None and self._auto_free_left <= 0

    def _set_btn_armed(self, btn, armed: bool) -> None:
        """Toggle the [armed] dynamic property + re-polish so the filled 'armed'
        look in the button's stylesheet applies/clears without a rebuild."""
        btn.setProperty("armed", "true" if armed else "false")
        btn.style().unpolish(btn)
        btn.style().polish(btn)

    def _refresh_exemplar_button_labels(self) -> None:
        """Write both example buttons' text for the state they are in.

        Three states have to be told apart at a glance, and the fill alone
        cannot carry the last one:
          - ready, nothing drawn yet: the action ("Draw on the map");
          - drawing now: the way OUT, because a second click disarms and the
            armed instruction line below already carries the gesture;
          - one example already drawn: "Draw another example", so the button
            stops reading as a step that is still waiting to be done.

        Reads the armed state off ``_auto_exemplar_armed_label`` (1 example,
        0 exclude, None not armed), so a chip refresh landing mid-draw cannot
        overwrite the armed wording.
        """
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
        """True while the example card's message slot actually shows something:
        the amber too-small warning or the armed draw instruction.

        Asks the widgets, not the state flag, on purpose. A user who closed the
        armed instruction leaves the slot free even while a draw is armed, and
        the explainer takes it back instead of the card showing a gap."""
        for name in ("auto_exemplar_size_warning", "auto_exemplar_armed_tip"):
            try:
                if getattr(self, name).isVisible():
                    return True
            except (RuntimeError, AttributeError):
                continue
        return False

    def set_auto_exemplar_armed(self, label) -> None:
        """Reflect the example draw-tool armed state on the draw-example /
        exclude buttons + the instruction line. label 1 = positive example
        armed, 0 = exclude armed, None = not armed. The plugin calls this when
        it arms the draw tool and again when the draw finishes or is cancelled,
        so the 'now draw on the map' feedback always clears."""
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
        # An armed draw opens the collapsed example section so the armed
        # instruction is actually visible (e.g. armed from a rescue chip).
        try:
            self._set_auto_exemplar_expanded(True)
        except (RuntimeError, AttributeError):
            pass
        # One message at a time: the armed instruction takes the line from the
        # too-small warning. Each wording is its own closable tip, and a user
        # who closed it gets the explainer back rather than an empty gap (the
        # armed button already says the tool is live).
        self.auto_exemplar_size_warning.setVisible(False)
        shown = self.auto_exemplar_armed_tip.set_hint(
            HINT_EXEMPLAR_EXCLUDE_BOX if label == 0 else HINT_EXEMPLAR_DRAW_BOX,
            _msg_text("armed", (
                tr("Click points around one look-alike, then double-click "
                   "to close.")
                if label == 0 else
                tr("Click points around one object, then double-click "
                   "to close."))),
            dense=True)
        self.auto_exemplar_armed_tip.setVisible(shown)
        self._refresh_auto_exemplar_explainer(armed=shown)
        self._auto_exemplar_hint_kind = "armed"
        self._set_exemplar_quality()

    def show_auto_exemplar_size_warning(self, at_max_detail: bool = False) -> None:
        """One calm line while at least one stored example renders below a
        usable pixel size AT THE CURRENT DETAIL LEVEL. Advisory only (the run
        stays available), and never closable: it says why the run will miss
        things. One message at a time on the example card, so it replaces the
        explainer and the next armed draw (or its finish) replaces or clears
        it.

        Dynamic, not one-shot: the caller (the plugin's
        ``_refresh_exemplar_size_warning``) re-evaluates every stored example
        against the CURRENT detail level on every detail change and on every
        exemplar add/remove, so raising the Detail slider clears this once
        every example is large enough (see ``clear_auto_exemplar_size_warning``).

        ``at_max_detail`` (default False, the conservative "can zoom finer"
        wording) drops the "zoom the detail slider finer" suggestion once the
        slider has no finer level left, so the message never tells the user
        to do something the UI already forbids.
        """
        try:
            self.auto_exemplar_armed_tip.setVisible(False)
            self.auto_exemplar_size_warning.setStyleSheet(
                _msg_label_qss("warning"))
            self.auto_exemplar_size_warning.setText(_msg_text("warning", (
                tr("This example is very small even at full precision. "
                   "Draw a larger object, or it may be too small to detect.")
                if at_max_detail else
                tr("This example is very small at this precision. "
                   "Raise the precision or draw a larger object."))))
            self.auto_exemplar_size_warning.setVisible(True)
            self._refresh_auto_exemplar_explainer(armed=True)
            self._auto_exemplar_hint_kind = "warning"
            # The size warning is the more urgent message: make the
            # second-example nudge (a DIFFERENT label) yield to it now.
            self._apply_prompt_hint_on_edit()
            self._set_exemplar_quality()
        except (RuntimeError, AttributeError):
            pass

    def clear_auto_exemplar_size_warning(self) -> None:
        """Hide the too-small warning, but ONLY when it is the message
        currently showing on the example card. The slot is shared with the
        armed-draw instruction (one message at a time), so a blind clear here
        could wipe "drag a box around one object..." out from under an
        in-progress draw; the ``_auto_exemplar_hint_kind`` flag (set by the
        three writers: armed / warning / None) is what makes this safe.
        Mirrors the ``label is None`` branch of ``set_auto_exemplar_armed``."""
        if getattr(self, "_auto_exemplar_hint_kind", None) != "warning":
            return
        try:
            self.auto_exemplar_size_warning.setVisible(False)
            self._refresh_auto_exemplar_explainer()
            self._auto_exemplar_hint_kind = None
            # The warning is gone: let the second-example nudge return if the
            # single-positive state still calls for it.
            self._apply_prompt_hint_on_edit()
            self._set_exemplar_quality()
        except (RuntimeError, AttributeError):
            pass

    def set_exemplars(self, items: list) -> None:
        """Rebuild the reference thumbnail strip. Each item is ``(id, label)`` or
        ``(id, label, thumbnail)`` where thumbnail is a QImage of the drawn crop.
        label 1 = positive (find similar), 0 = exclude. Also refreshes the Detect
        gate + the add-buttons' cap state."""
        layout = self._auto_exemplar_chips_layout
        # Clear existing cards (keep the trailing stretch at the end).
        while layout.count() > 1:
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        from ...core.detect_gate import exclude_available
        from ...core.exemplar_store import max_exclude, max_positive
        self._auto_positive_exemplars = sum(1 for it in items if it[1] == 1)
        exclude_count = sum(1 for it in items if it[1] == 0)
        # The first example turns "Draw on the map" into "Draw another
        # example": the step is done, the button is now an optional extra.
        self._refresh_exemplar_button_labels()
        for idx, it in enumerate(items):
            eid, label = it[0], it[1]
            thumb = it[2] if len(it) > 2 else None
            card = self._make_exemplar_chip(eid, label, idx + 1, thumb)
            layout.insertWidget(layout.count() - 1, card)
        # The positive add button disables at its own cap. The exclude add
        # button is a bonus refinement, offered ONLY once the positive set is
        # strong enough; below that it stays HIDDEN so the primary flow is a
        # single green "Draw an example" button. When shown it still disables at
        # its own cap. Hiding, not disabling, keeps one clear affordance at a
        # time. Both ceilings come from the store's resolvers, the same ones the
        # store enforces on insert, so the row can never offer a slot the store
        # would refuse (or hide one the server just opened).
        try:
            self.auto_ex_inc_btn.setEnabled(
                self._auto_positive_exemplars < max_positive())
            exc = getattr(self, "auto_ex_exc_btn", None)
            if exc is not None:
                exc_available = exclude_available(self._auto_positive_exemplars)
                exc.setVisible(exc_available)
                exc.setEnabled(
                    exc_available and exclude_count < max_exclude())
        except (RuntimeError, AttributeError):
            pass
        # A drawn reference speaks for itself: the how/why explainer yields to
        # the thumbnails, and returns if every reference is removed.
        self._auto_exemplar_count = len(items)
        self._refresh_auto_exemplar_explainer(
            armed=self._auto_exemplar_line_busy())
        self._set_exemplar_quality()
        # An existing reference keeps the example section open (it holds the
        # thumbnails); an empty list leaves the collapse state to the user.
        if items:
            try:
                self._set_auto_exemplar_expanded(True)
            except (RuntimeError, AttributeError):
                pass
        # Drawing the first positive example while the prompt is empty should
        # surface the "your examples drive the search" hint (and clear it when
        # every example is removed).
        self._apply_prompt_hint_on_edit()
        self._update_auto_detect_enabled()

    def _set_exemplar_quality(self) -> None:
        """Reflect how close the example set is to the recommended pair (two
        positive references, the model's strongest mode) on the example card:
        the header dots fill lime as they are drawn, and one subtle line
        nudges toward the second, then confirms best quality at two.

        Both hide with no example. The line also yields while a draw is armed
        or the too-small size warning shows (that shared hint is the louder,
        actionable message), so the card never stacks guidance; the dots stay
        (they are a glanceable meter, not a sentence)."""
        positives = (getattr(self, "_auto_positive_exemplars", 0)
                     if getattr(self, "_EXEMPLARS_ENABLED", True) else 0)
        dots = getattr(self, "auto_exemplar_quality_dots", None)
        line = getattr(self, "auto_exemplar_quality_line", None)
        if dots is None or line is None:
            return
        # Header dots: filled lime up to the drawn count, hollow grey for the
        # rest of the recommended two.
        filled = min(positives, 2)
        empty = "rgba(128, 128, 128, 0.45)"
        marks = []
        for i in range(2):
            color = BRAND_GREEN if i < filled else empty
            marks.append(
                f'<span style="color: {color}; font-size: 13px;">&#9679;</span>')
        try:
            dots.setText("&nbsp;".join(marks))
            dots.setVisible(positives > 0)
        except (RuntimeError, AttributeError):
            pass
        # The nudge/confirm line yields to an armed instruction or a size
        # warning in the card's message slot.
        armed_showing = self._auto_exemplar_line_busy()
        try:
            if positives <= 0 or armed_showing:
                line.setVisible(False)
            elif positives == 1:
                line.setStyleSheet(scale_qss_font_px(
                    "font-size: 11px; color: rgba(128, 128, 128, 0.95);"
                    " background: transparent; border: none;"))
                line.setText(tr("Add one more example for the best results."))
                line.setVisible(True)
            else:
                line.setStyleSheet(scale_qss_font_px(
                    f"font-size: 11px; font-weight: 600; color: {SUCCESS_TEXT};"
                    " background: transparent; border: none;"))
                line.setText(
                    "✓  " + tr("Best quality. Two references locked in."))
                line.setVisible(True)
        except (RuntimeError, AttributeError):
            pass

    def _make_exemplar_chip(self, exemplar_id: str, label: int,
                            index: int = 1, thumbnail=None) -> QWidget:
        """A 52px reference card (AI-Edit _ThumbWidget look): the drawn crop +
        a numbered badge + a remove x. Border green for a positive example, red
        for an exclude box. Falls back to a flat tinted tile when no thumbnail
        was captured."""
        from qgis.PyQt.QtGui import QPixmap
        is_pos = label == 1
        rgba = "67,160,71" if is_pos else "229,57,53"
        side = 52
        card = QFrame()
        card.setObjectName("refCard")
        card.setFixedSize(side, side)
        card.setStyleSheet(
            f"QFrame#refCard {{ border: 1px solid rgba({rgba},0.85);"
            f" border-radius: 4px; background: rgba({rgba},0.12); }}")
        card.setToolTip(tr("Example"))
        # The drawn crop, if captured.
        thumb_lbl = QLabel(card)
        thumb_lbl.setGeometry(1, 1, side - 2, side - 2)
        # Preserve aspect (no stretch) so a long thin reference shows as a thin
        # bar, not distorted into a square. The enlarge popup shows it full size.
        thumb_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb_lbl.setStyleSheet("border: none; background: transparent;")
        if thumbnail is not None:
            try:
                pm = QPixmap.fromImage(thumbnail)
                if not pm.isNull():
                    # The capture is taken at the screen's own pixel count on
                    # purpose; scaling it back down to 50 flat threw that away
                    # and left a soft chip on every scaled display.
                    ratio = widget_pixel_ratio(thumb_lbl)
                    edge = int(round((side - 2) * ratio))
                    pm = pm.scaled(
                        edge, edge,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)
                    pm.setDevicePixelRatio(ratio)
                    thumb_lbl.setPixmap(pm)
                # Click the card (anywhere but the remove x) to see the reference
                # enlarged: it shows what the AI uses (the object with a little
                # of its surroundings).
                card.setCursor(Qt.CursorShape.PointingHandCursor)
                card.setToolTip(tr("Click to enlarge"))
                card.mousePressEvent = (
                    lambda _ev, im=thumbnail: self._show_exemplar_detail(im))
            except (RuntimeError, TypeError):
                pass
        # Numbered badge, top-left.
        badge = QLabel(str(index), card)
        badge.setStyleSheet(
            "QLabel { background: rgba(0,0,0,0.6); color: rgba(255,255,255,0.92);"
            " font-size: 9px; font-weight: bold; border: none;"
            " border-top-left-radius: 3px; border-bottom-right-radius: 3px;"
            " padding: 0 3px; }")
        badge.adjustSize()
        badge.move(1, 1)
        # Remove x, top-right.
        remove = QToolButton(card)
        remove.setText("✕")
        remove.setCursor(Qt.CursorShape.PointingHandCursor)
        remove.setToolTip(tr("Remove"))
        remove.setFixedSize(16, 16)
        remove.setStyleSheet(
            "QToolButton { background: rgba(0,0,0,0.6); color: #fff; border: none;"
            " border-radius: 8px; font-size: 9px; font-weight: bold; }"
            "QToolButton:hover { background: rgba(211,47,47,0.9); }")
        remove.move(side - 17, 1)
        remove.clicked.connect(
            lambda _checked=False, eid=exemplar_id: self.auto_exemplar_remove_requested.emit(eid))
        # Kept so the in-run read-only view can hide the remove x while leaving
        # the thumbnail (and its click-to-enlarge) in place (see
        # _set_exemplar_readonly).
        card._remove_btn = remove
        return card

    def _show_exemplar_detail(self, image) -> None:
        """Popup showing the reference example enlarged, so the user can inspect
        what the AI uses (the object with a little of its natural surroundings).
        Best-effort: never fatal."""
        if image is None:
            return
        try:
            from qgis.PyQt.QtGui import QPixmap
            from qgis.PyQt.QtWidgets import QDialog, QLabel, QVBoxLayout
            pm = QPixmap.fromImage(image)
            if pm.isNull():
                return
            # This window exists to be looked at closely, so it gets the
            # screen's real pixel count rather than a 320-unit copy stretched
            # back up.
            ratio = widget_pixel_ratio(self)
            edge = int(round(320 * ratio))
            pm = pm.scaled(
                edge, edge,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            pm.setDevicePixelRatio(ratio)
            dlg = QDialog(self)
            dlg.setWindowTitle(tr("Example"))
            lay = QVBoxLayout(dlg)
            lay.setContentsMargins(12, 12, 12, 12)
            lay.setSpacing(8)
            img_lbl = QLabel(dlg)
            img_lbl.setPixmap(pm)
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(img_lbl)
            hint = QLabel(tr(
                "This is exactly what the AI uses: your object with a little "
                "of its surroundings."))
            hint.setWordWrap(True)
            hint.setStyleSheet("color: rgba(128,128,128,0.9); font-size: 11px;")
            lay.addWidget(hint)
            from .font_scale import apply_font_scale_to_tree

            apply_font_scale_to_tree(dlg)
            dlg.exec()
        except Exception:  # noqa: BLE001 -- preview popup is best-effort  # nosec B110
            pass

    def _update_auto_detect_enabled(self):
        """Enable the Detect button based on current Automatic mode state."""
        if not self._plugin_activated or self._auto_review_active:
            self.auto_detect_btn.setEnabled(False)
            return
        from ...core.detect_gate import can_detect
        has_layer = self.auto_layer_combo.currentLayer() is not None
        # The big green Detect enables on the floor: a typed prompt, or one
        # drawn example, or both. Each of the three is a query the model can
        # run, and asking for the full prompt-plus-example combination before
        # lighting the button hid the two single-input paths behind a link
        # nobody read. The combination is still the most accurate mode, so the
        # cards keep nudging toward it; it is no longer a gate.
        has_text = bool(self.auto_prompt_input.text().strip())
        positives = self._auto_positive_exemplars if self._EXEMPLARS_ENABLED else 0
        # An object CONCEPT exists once a prompt is typed or one example is
        # drawn; that is what the Detail slider gate needs (see below).
        has_object = has_text or positives > 0
        can_run = can_detect(has_text, positives)
        # The Detail slider gates on the object CONCEPT, not on can_run: its
        # default is OBJECT-AWARE (committing a prompt re-seeds it), so a value
        # tuned BEFORE the object was named got thrown away by the re-seed and
        # the user lost their adjustment. Greying it until the object exists
        # makes the order explicit: name it, then fine-tune the grid. It stays
        # available with a single example (the concept is there) even while
        # Detect waits for the second one.
        self._apply_auto_detail_gate(has_object)
        not_too_large = not self._auto_zone_too_large
        if self._auto_is_subscriber:
            # Block a subscriber whose balance is known to be 0 (None = not yet
            # fetched, fail open). Mirrors the free-tier credit gate below.
            credits_ok = self._auto_credits is None or self._auto_credits > 0
        else:
            free_left = self._auto_free_left
            credits_ok = (free_left is not None and free_left > 0) or free_left is None
        # Hard credit gate: block when the drawn zone would cost more tiles than
        # the balance can cover, so a run never launches under-funded and stops
        # mid-zone at 0 (set in set_auto_credit_estimate).
        credits_enough = not self._auto_insufficient_credits
        # Free-plan per-run cap: the detail slider deliberately keeps its full
        # (Pro) travel, so a gated level greys Detect here instead (the red
        # cost line + detail hint name the fix: lower detail, or upgrade).
        premium_ok = not getattr(self, "_auto_premium_gated", False)
        # Consent gates DETECT (the moment credits are spent), not Start: the
        # checkbox sits right above this button (see auto_build).
        tos_ok = has_tos_locked() or has_tos_accepted()
        hard_ok = has_layer and not_too_large and credits_ok
        hard_ok = hard_ok and credits_enough and premium_ok and tos_ok
        run_allowed = hard_ok and can_run
        self.auto_detect_btn.setEnabled(run_allowed and not self._auto_run_active)
        # A disabled button with no reason reads as broken: consent and the
        # empty setup are the two gates the user can fix right here, so say so.
        if not tos_ok:
            tip = tr("Accept the Terms and Privacy Policy first.")
        elif hard_ok and not has_object:
            tip = tr("Type a word for the object, or draw an example.")
        else:
            tip = ""
        self.auto_detect_btn.setToolTip(tip)
        # detect_blocked telemetry: once per episode, only when the run is set up
        # (layer + object) but a hard gate (credits / zone too large) blocks it.
        reason = None
        if has_layer and has_object and not self._auto_run_active and not run_allowed:
            if not not_too_large:
                reason = "zone_too_large"
            elif not credits_ok or not credits_enough:
                reason = "credits"
            elif not premium_ok:
                reason = "detail_premium_cap"
        if reason and reason != getattr(self, "_detect_blocked_last", None):
            try:
                from ...core import telemetry_session_events
                telemetry_session_events.track_detect_blocked(reason=reason)
            except Exception:
                pass  # nosec B110
        self._detect_blocked_last = reason

    def _refresh_auto_credits_display(self):
        """Drive the footer credit gauge (ring + count + Subscribe pill)."""
        in_auto = self._mode == Mode.AUTOMATIC and self._plugin_activated
        remaining = self._auto_credits
        total = self._auto_credits_total

        show_gauge = in_auto and remaining is not None
        if show_gauge:
            if total is not None and total > 0:
                self._footer_credits_label.setText(f"{remaining} / {total}")
                self._credit_ring.set_credits(
                    max(0, total - remaining), total,
                    free_tier=not self._auto_is_subscriber)
                self._credit_ring.setVisible(True)
            else:
                self._footer_credits_label.setText(str(remaining))
                self._credit_ring.setVisible(False)
            if self._auto_is_subscriber:
                tooltip = tr("{n} credits remaining").format(n=remaining)
            elif total is not None and total > 0:
                tooltip = tr("{n} of {total} free detections left").format(
                    n=remaining, total=total)
            else:
                tooltip = tr("{n} free detection(s) remaining").format(n=remaining)
            # A balance with no return date cannot be acted on: the quota
            # renews on the sign-up anniversary, which nobody can guess.
            # Pre-formatted by set_auto_credits, empty on servers that send no
            # period_end, and the line simply drops out then.
            reset_day = getattr(self, "_auto_reset_display", "")
            if reset_day:
                tooltip += "\n" + tr("Credits come back on {date}").format(
                    date=reset_day)
            # The gauge is a link, so the tooltip says where it goes.
            tooltip += "\n" + tr("Click to open your dashboard")
            self._footer_credits_label.setToolTip(tooltip)
            self._credit_ring.setToolTip(tooltip)
            gauge = getattr(self, "_credit_gauge", None)
            if gauge is not None:
                gauge.setToolTip(tooltip)
            self._footer_credits_label.setVisible(True)
        else:
            self._credit_ring.setVisible(False)
            self._footer_credits_label.setVisible(False)

        pill_shown = in_auto and not self._auto_is_subscriber
        self._subscribe_pill.setVisible(pill_shown)
        # The pill carries most of the upsell clicks in the product and used to
        # report none of the matching impressions, so its click-through rate
        # could not be computed at all. track_pro_upsell_viewed dedupes per
        # trigger, so calling it on every refresh reports once per session.
        if pill_shown:
            try:
                from ...core import telemetry_session_events
                telemetry_session_events.track_pro_upsell_viewed(trigger="subscribe_pill")
            except Exception:
                pass  # nosec B110
        # Free-tier low-credit nudge on the Start page (driven by the same
        # remaining/total the ring uses).
        self._update_auto_low_credit_note()
        # Keep the exhausted-upsell card title's count in sync with that total.
        self._refresh_auto_upsell_title()

    def _refresh_auto_upsell_title(self):
        """Set the exhausted-upsell card title from the fetched free-detection
        total (the same source the credit ring uses), falling back to a
        number-free line until the total is known."""
        title = getattr(self, "_auto_upsell_title", None)
        if title is None:
            return
        from ...core.server_dials import dial_copy

        # Name the free way out beside the paid one, whenever the server told
        # us when the quota renews.
        reset_line = getattr(self, "_auto_upsell_reset", None)
        if reset_line is not None:
            reset_day = getattr(self, "_auto_reset_display", "")
            if reset_day:
                reset_line.setText(tr(
                    "Your free detections come back on {date}.").format(
                        date=reset_day))
            reset_line.setVisible(bool(reset_day))

        total = self._auto_credits_total
        if total and total > 0:
            # A served sentence names the count with {n}. Plain replace, never
            # format(), so a stray brace in served text cannot raise here.
            served = dial_copy(
                "trial.exhausted",
                tr("Your {n} free detections are used up"))
            title.setText(served.replace("{n}", str(int(total))))
        else:
            title.setText(dial_copy(
                "trial.exhausted_no_count",
                tr("Your free detections are used up")))

    def cleanup_signals(self):
        """Disconnect project signals and clean up shortcuts/timers on plugin reload."""
        try:
            self.layer_combo.cleanup()
        except (TypeError, RuntimeError, AttributeError):
            pass
        # The automatic-mode raster picker registers its own QgsProject signals;
        # clean it up too or project edits fire into a destroyed widget on reload.
        try:
            self.auto_layer_combo.cleanup()
        except (TypeError, RuntimeError, AttributeError):
            pass
        try:
            QgsProject.instance().layersAdded.disconnect(self._on_layers_added)
        except (TypeError, RuntimeError):
            pass
        try:
            QgsProject.instance().layersRemoved.disconnect(self._on_layers_removed)
        except (TypeError, RuntimeError):
            pass
        try:
            QgsProject.instance().layerTreeRoot().visibilityChanged.disconnect(
                self._on_layer_visibility_changed)
        except (TypeError, RuntimeError, AttributeError):
            pass
        # Clean up every QShortcut to prevent stale callbacks. They are all
        # window-level, so one left connected keeps firing into a dock that is
        # gone. getattr keeps this working if a shortcut was never built.
        for name in (
                "start_shortcut",
                "auto_escape_shortcut",
                "auto_enter_shortcut",
                "auto_enter_shortcut_kp",
                "auto_correct_remove_delete_shortcut",
                "auto_correct_remove_backspace_shortcut",
                "auto_correct_undo_shortcut"):
            shortcut = getattr(self, name, None)
            if shortcut is None:
                continue
            try:
                shortcut.activated.disconnect()
                shortcut.deleteLater()
            except (TypeError, RuntimeError, AttributeError):
                pass
        # Stop timers first, then disconnect to avoid race conditions
        try:
            self._progress_timer.blockSignals(True)
            self._progress_timer.stop()
            self._progress_timer.timeout.disconnect()
        except (TypeError, RuntimeError, AttributeError):
            pass
        try:
            self._refine_debounce_timer.blockSignals(True)
            self._refine_debounce_timer.stop()
            self._refine_debounce_timer.timeout.disconnect()
        except (TypeError, RuntimeError, AttributeError):
            pass
        try:
            self._auto_review_debounce_timer.blockSignals(True)
            self._auto_review_debounce_timer.stop()
            self._auto_review_debounce_timer.timeout.disconnect()
        except (TypeError, RuntimeError, AttributeError):
            pass
        try:
            self._auto_conf_debounce_timer.blockSignals(True)
            self._auto_conf_debounce_timer.stop()
            self._auto_conf_debounce_timer.timeout.disconnect()
        except (TypeError, RuntimeError, AttributeError):
            pass
        try:
            self._visibility_debounce_timer.blockSignals(True)
            self._visibility_debounce_timer.stop()
            self._visibility_debounce_timer.timeout.disconnect()
        except (TypeError, RuntimeError, AttributeError):
            pass
        try:
            self._pairing_anim_timer.blockSignals(True)
            self._pairing_anim_timer.stop()
            self._pairing_anim_timer.timeout.disconnect()
        except (TypeError, RuntimeError, AttributeError):
            pass
        # Remove the temp dir holding the generated checkbox icons
        if getattr(self, "_checkbox_icon_dir", None):
            import shutil
            shutil.rmtree(self._checkbox_icon_dir, ignore_errors=True)
            self._checkbox_icon_dir = None

    def is_activated(self) -> bool:
        """Check if the plugin is activated."""
        return self._plugin_activated

    def showEvent(self, event):
        """Emit plugin_opened once per session when the dock becomes visible.

        The event is queued; if the plugin is not yet activated it parks in the
        telemetry pre-auth queue and ships on the first authenticated flush, so
        a first-run user who activates later still gets one plugin_opened for
        this session.
        """
        super().showEvent(event)
        # Re-sync the raster combos with the layer tree whenever the dock
        # becomes visible: a project opened or layers changed while the dock
        # was hidden can leave the list stale, and a stale EMPTY list shows
        # the "Load your own imagery" hero over a project that has imagery
        # (observed live; the debounced refresh then repaints the state).
        for combo_name in ("layer_combo", "auto_layer_combo"):
            try:
                getattr(self, combo_name)._schedule_refresh()
            except (RuntimeError, AttributeError):
                pass
        if getattr(self, "_plugin_opened_emitted", False):
            return
        try:
            from ...core.telemetry_session_events import track_plugin_first_open, track_plugin_opened
            # First-ever open on this machine (self-guarded by a persistent
            # QSettings flag) precedes the per-session plugin_opened, so the
            # install -> first-open -> activation funnel has a clean entry.
            track_plugin_first_open()
            track_plugin_opened()
            self._plugin_opened_emitted = True
        except Exception:
            pass  # nosec B110
