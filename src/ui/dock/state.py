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
from .guidance import (
    HINT_TRY_AUTOMATIC,
    HINT_TUTORIAL_FIRST_STEPS,
    is_hint_dismissed,
)
from .styles import (
    ERROR_TEXT,
    _BTN_EXPORT_DISABLED,
    _BTN_EXPORT_READY,
    _INSTRUCTIONS_CARD_QSS,
    _INSTRUCTIONS_HINT_QSS,
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
            self.instructions_label.setVisible(True)
            self._update_instructions()

            # Refine panel visibility
            self._update_refine_panel_visibility()

            # Save + Undo act on the object currently OPEN for editing. In a
            # refine handoff no object is open until the user clicks a blue
            # detection, so keep them out of the way until then (dead grey
            # buttons at the start read as broken and drown the "pick one"
            # guidance). Base Manual keeps Save always visible as before.
            editing_open = self._has_mask
            save_visible = editing_open if self._refine_handoff else True
            self.save_mask_button.setVisible(save_visible)
            self.save_mask_button.setEnabled(self._has_mask)

            # Export button: visible during segmentation, EXCEPT in a refine
            # handoff, where committing goes through Back to review -> Finish (a
            # direct manual export would orphan the held Automatic review).
            self.export_button.setVisible(not self._refine_handoff)
            self._update_export_button_style()

            # Secondary buttons: visible during segmentation. In a handoff the
            # only one here is Undo (Stop is hidden), so gate the whole row on an
            # object being open, matching the Save button above.
            secondary_visible = editing_open if self._refine_handoff else True
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

    def _set_instructions_compact(self, compact: bool) -> None:
        """Swap the instructions label between the framed card (normal Manual
        guidance) and the compact muted hint used for the refine-handoff status
        lines. No-op when already in the requested style."""
        if getattr(self, "_instructions_compact", None) == compact:
            return
        self._instructions_compact = compact
        self.instructions_label.setStyleSheet(
            _INSTRUCTIONS_HINT_QSS if compact else _INSTRUCTIONS_CARD_QSS)
        self.instructions_label.setMinimumHeight(0 if compact else 70)

    def _update_instructions(self):
        """Update instruction text based on current segmentation state."""
        total = self._positive_count + self._negative_count

        if self._refine_handoff:
            # A single framed guidance card in both handoff sub-states, so the
            # flow is always legible (the old thin one-liner under the banner
            # was too easy to miss). blue = still editable, green = validated;
            # right-click carves a part, it does NOT delete the object. The
            # Save/Undo buttons only appear once an object is open for editing
            # (see _update_button_visibility), so this card carries the "pick
            # one first" step on its own.
            self._set_instructions_compact(False)
            if total > 0:
                # Rich-text controls card: a keys -> action table reads at a
                # glance where the old three prose lines did not. Keys carry
                # the brand blue so they pop as the interactive vocabulary.
                def _kbd(k: str) -> str:
                    return "<b style='color:#1e88e5; white-space:nowrap;'>" + k + "</b>"

                rows = (
                    (tr("Left-click"), tr("adds area")),
                    (tr("Right-click"), tr("removes area")),
                    ("S", tr("keeps it (turns green)")),
                    (tr("Delete"), tr("removes the object")),
                )
                header_html = "<b>✎ " + tr("Editing this detection") + "</b>"
                table_open_html = "<table cellspacing='0' cellpadding='1'>"
                rows_html = "".join(
                    "<tr><td style='padding-right: 12px;'>" + _kbd(key) + "</td><td>" + action + "</td></tr>"
                    for key, action in rows)
                text = header_html + table_open_html + rows_html + "</table>"
            elif self._handoff_seed_total > self._handoff_kept:
                # The controls table above appears once an object is open, so
                # this state carries only the one action + live progress.
                header = "<b>" + tr("Click a blue detection to open it for editing.") + "</b>"
                kept_msg = tr("{kept} of {total} kept - 'Back to review' to export.").format(
                    kept=self._handoff_kept, total=self._handoff_seed_total)
                status = "<br/><span style='color: rgba(128,128,128,0.95);'>" + kept_msg + "</span>"
                text = header + status
            else:
                text = tr("All detections kept. Go 'Back to review' to export.")
            self.instructions_label.setText(text)
            return

        self._set_instructions_compact(False)
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
            # T13: closing mid-handoff behaves like "Back to review": harvest
            # the manual edits into the held review, then let the dock hide
            # (the review is there on reopen). The harvest also ends the
            # manual session, so the stop-dialog branch below no-ops.
            try:
                self.back_to_review_requested.emit()
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

    def set_manual_last_run_recap(self, count: int, area_km2) -> None:
        """Show the session-only value recap on the Manual Start view after a
        successful export: one quiet line under the Start button/caption
        (mirrors the Automatic set_last_run_recap, without credits: Manual is
        local and free). Best-effort by contract (the export already
        committed): never raises, so a recap problem can never surface as a
        failed export."""
        try:
            recap = getattr(self, "manual_last_run_recap", None)
            if recap is None:
                return
            area = self._format_recap_area(area_km2)
            recap.setText(tr(
                "Last session: {count} polygon(s) exported · {area} km2"
            ).format(count=count, area=area))
            recap.setVisible(True)
        except Exception:  # nosec B110 -- recap is best-effort, never break export
            pass

    def clear_manual_last_run_recap(self) -> None:
        """Hide the Manual last-session recap (session only). Safe to call when
        the card was never built."""
        try:
            recap = getattr(self, "manual_last_run_recap", None)
            if recap is not None:
                recap.setVisible(False)
        except (RuntimeError, AttributeError):
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
        """Check if a raster layer is properly georeferenced."""
        if layer is None or not isinstance(layer, QgsRasterLayer):
            return False

        # Check file extension for compatible formats
        source = layer.source().lower()

        # PNG, JPG, BMP etc. without world files are not georeferenced
        non_georef_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

        has_non_georef_ext = any(source.endswith(ext) for ext in non_georef_extensions)

        # If it's a known non-georeferenced format, check if it has a valid CRS
        if has_non_georef_ext:
            # Check if the layer has a valid CRS (not just default)
            if not layer.crs().isValid():
                return False
            # Check if extent looks like pixel coordinates (0,0 to width,height)
            extent = layer.extent()
            if extent.xMinimum() == 0 and extent.yMinimum() == 0:
                # Likely not georeferenced - just pixel dimensions
                return False

        return True

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
                from ...core import telemetry
                telemetry.track_pro_upsell_viewed(trigger="free_exhausted")
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

    def set_auto_exemplar_armed(self, label) -> None:
        """Reflect the example draw-tool armed state on the draw-example /
        exclude buttons + the instruction line. label 1 = positive example
        armed, 0 = exclude armed, None = not armed. The plugin calls this when
        it arms the draw tool and again when the draw finishes or is cancelled,
        so the 'now draw on the map' feedback always clears."""
        try:
            self._set_btn_armed(self.auto_ex_inc_btn, label == 1)
            exc = getattr(self, "auto_ex_exc_btn", None)
            if exc is not None:
                self._set_btn_armed(exc, label == 0)
        except (RuntimeError, AttributeError):
            return
        if label is None:
            self.auto_exemplar_armed_hint.setVisible(False)
            return
        self.auto_exemplar_armed_hint.setText(
            tr("Now outline a look-alike to exclude, then click the first "
               "point to close.")
            if label == 0 else
            tr("Now outline one object, then click the first point to close."))
        self.auto_exemplar_armed_hint.setVisible(True)

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
        self._auto_positive_exemplars = sum(1 for it in items if it[1] == 1)
        for idx, it in enumerate(items):
            eid, label = it[0], it[1]
            thumb = it[2] if len(it) > 2 else None
            card = self._make_exemplar_chip(eid, label, idx + 1, thumb)
            layout.insertWidget(layout.count() - 1, card)
        # At the cap (EXEMPLAR_MAX = 4, positives + excludes) both add buttons
        # disable. The Exclude button appears only once a positive example
        # exists, so the primary flow stays one green button.
        full = len(items) >= 4
        try:
            self.auto_ex_inc_btn.setEnabled(not full)
            exc = getattr(self, "auto_ex_exc_btn", None)
            if exc is not None:
                exc.setVisible(self._auto_positive_exemplars > 0)
                exc.setEnabled(not full)
        except (RuntimeError, AttributeError):
            pass
        # Drawing the first positive example while the prompt is empty should
        # surface the "your examples drive the search" hint (and clear it when
        # every example is removed).
        self._apply_prompt_hint_on_edit()
        self._update_auto_detect_enabled()

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
                    pm = pm.scaled(
                        side - 2, side - 2,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)
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
            from qgis.PyQt.QtWidgets import QDialog, QVBoxLayout, QLabel
            from qgis.PyQt.QtGui import QPixmap
            pm = QPixmap.fromImage(image)
            if pm.isNull():
                return
            pm = pm.scaled(
                320, 320,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
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
            dlg.exec()
        except Exception:  # noqa: BLE001 -- preview popup is best-effort  # nosec B110
            pass

    def _update_auto_detect_enabled(self):
        """Enable the Detect button based on current Automatic mode state."""
        if not self._plugin_activated or self._auto_review_active:
            self.auto_detect_btn.setEnabled(False)
            return
        has_layer = self.auto_layer_combo.currentLayer() is not None
        # The object is defined by EITHER a drawn reference (the primary input)
        # OR a typed/gallery prompt (the optional secondary). Any non-empty text
        # counts here: the guard rail runs when the user COMMITS (Detect/Enter,
        # via confirm_prompt_for_detect), so a mid-typing prompt never greys the
        # button, and an off-rails one gets an explanation instead of a dead
        # click on a disabled control.
        has_object = bool(self.auto_prompt_input.text().strip()) or (
            self._EXEMPLARS_ENABLED and self._auto_positive_exemplars > 0)
        # The Detail slider gates on the same condition: its default is
        # OBJECT-AWARE (committing a prompt re-seeds it), so a value tuned
        # BEFORE the object was named got thrown away by the re-seed and the
        # user lost their adjustment. Greying it until the object
        # exists makes the order explicit: name it, then fine-tune the grid.
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
        # Consent gates DETECT (the moment credits are spent), not Start: the
        # checkbox sits right above this button (see auto_build).
        tos_ok = has_tos_locked() or has_tos_accepted()
        can_detect = has_layer and has_object and not_too_large and credits_ok and credits_enough and tos_ok
        self.auto_detect_btn.setEnabled(can_detect and not self._auto_run_active)
        # A disabled button with no reason reads as broken; consent is the one
        # gate the user can fix right here, so say so.
        self.auto_detect_btn.setToolTip(
            "" if tos_ok else tr("Accept the Terms and Privacy Policy first."))
        # detect_blocked telemetry: once per episode, only when the run is set up
        # (layer + object) but a hard gate (credits / zone too large) blocks it.
        reason = None
        if has_layer and has_object and not self._auto_run_active and not can_detect:
            if not not_too_large:
                reason = "zone_too_large"
            elif not credits_ok or not credits_enough:
                reason = "credits"
        if reason and reason != getattr(self, "_detect_blocked_last", None):
            try:
                from ...core import telemetry
                telemetry.track_detect_blocked(reason=reason)
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
            self._footer_credits_label.setToolTip(tooltip)
            self._credit_ring.setToolTip(tooltip)
            self._footer_credits_label.setVisible(True)
        else:
            self._credit_ring.setVisible(False)
            self._footer_credits_label.setVisible(False)

        self._subscribe_pill.setVisible(in_auto and not self._auto_is_subscriber)
        # Free-tier low-credit nudge on the Start page (driven by the same
        # remaining/total the ring uses).
        self._update_auto_low_credit_note()

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
        # Clean up QShortcut to prevent stale callbacks
        try:
            self.start_shortcut.activated.disconnect()
            self.start_shortcut.deleteLater()
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
        if getattr(self, "_plugin_opened_emitted", False):
            return
        try:
            from ...core.telemetry import (
                track_plugin_first_open,
                track_plugin_opened,
            )
            # First-ever open on this machine (self-guarded by a persistent
            # QSettings flag) precedes the per-session plugin_opened, so the
            # install -> first-open -> activation funnel has a clean entry.
            track_plugin_first_open()
            track_plugin_opened()
            self._plugin_opened_emitted = True
        except Exception:
            pass  # nosec B110
