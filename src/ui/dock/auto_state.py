"""Automatic mode state: steps, zone, credits, run/review display updates.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.
"""
from __future__ import annotations

import time

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QGraphicsOpacityEffect,
    QLineEdit,
)


from ...core.credit_gate import insufficient as _credit_insufficient
from ...core.i18n import tr
from ...core.review_defaults import (
    AUTO_REVIEW_CLEAN_DEFAULT as _AUTO_REVIEW_CLEAN_DEFAULT,
    AUTO_REVIEW_EXPAND_DEFAULT as _AUTO_REVIEW_EXPAND_DEFAULT,
    AUTO_REVIEW_FILL_HOLES_DEFAULT as _AUTO_REVIEW_FILL_HOLES_DEFAULT,
    AUTO_REVIEW_ORTHO_DEFAULT as _AUTO_REVIEW_ORTHO_DEFAULT,
    AUTO_REVIEW_SIMPLIFY_DEFAULT as _AUTO_REVIEW_SIMPLIFY_DEFAULT,
    AUTO_REVIEW_SMOOTH_DEFAULT as _AUTO_REVIEW_SMOOTH_DEFAULT,
)
from ...core.tile_manager import MAX_DETAIL_LEVEL
from .auto_review_build import _export_btn_label
from .prompt_guard import english_token_for, is_known_object, validate_prompt
from .styles import (
    BRAND_BLUE,
    BRAND_GREEN,
    ERROR_TEXT,
    _PREMIUM_STAR,
    _REVIEW_CONF_MIN,
    _REVIEW_CONF_SPIN_MIN,
    _msg_label_qss,
    _msg_text,
    _snap_review_conf,
)
from .widgets import (
    Mode,
)


class DockAutoStateMixin:
    """Automatic mode state: steps, zone, credits, run/review display updates."""

    # ---- Automatic mode helpers ------------------------------------------------

    def _on_upgrade_clicked(self) -> None:
        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices
        try:
            from ...core import telemetry
            sender = self.sender()
            if sender is getattr(self, "_subscribe_pill", None):
                source = "subscribe_pill"
            elif sender is getattr(self, "auto_exhausted_subscribe_link", None):
                source = "exhausted_status"
            else:
                source = "upsell_card"
            telemetry.track_pro_upsell_clicked(source=source)
        except Exception:
            pass  # nosec B110
        url = self._build_upgrade_url()
        QDesktopServices.openUrl(QUrl(url))

    def _build_upgrade_url(self) -> str:
        from ...core.activation_manager import get_dashboard_url
        base = get_dashboard_url()
        return (
            f"{base}?utm_source=qgis&utm_medium=plugin"
            "&utm_campaign=ai-segmentation-pro&utm_content=upsell_card"
        )

    def _on_shape_control_changed(self, control: str, value) -> None:
        """A review shape-refine control changed: re-derive the visible set (via
        the debounced auto_refine_changed) and track the adjustment once per
        control per review."""
        self.auto_refine_changed.emit()
        try:
            tracked = getattr(self, "_review_shape_tracked", None)
            if tracked is None:
                tracked = set()
                self._review_shape_tracked = tracked
            if control not in tracked:
                tracked.add(control)
                from ...core import telemetry
                telemetry.track_review_shape_adjusted(control=control, value=value)
        except Exception:
            pass  # nosec B110

    def _on_auto_layer_changed(self, layer) -> None:
        # A raster appearing or vanishing must show/hide the combo and the
        # no-rasters warning live, not only on the next mode switch.
        if self._mode == Mode.AUTOMATIC:
            self._update_ui_state_automatic()
        self._update_auto_detect_enabled()
        self._refresh_auto_layer_lock()

    def _on_auto_start_clicked(self) -> None:
        """Step 0 "Start": lock the chosen raster and open the draw-zone step.

        Mirrors the Interactive Start: from here on the layer combo is greyed
        and read-only (the header still names the locked raster) until the
        user clicks Exit.
        """
        layer = self.auto_layer_combo.currentLayer()
        if layer is None:
            return
        # Consent is NOT asked here: the checkbox sits on the last step, right
        # above Detect, and the first Detect seals it (see seal_tos_consent).
        # Clear any leftover "Saved N polygon(s)" banner from a previous run so
        # it never lingers into this fresh run's steps (it is shown on the Start
        # step right after Finish, and belongs only to that just-finished run).
        self.set_auto_status("idle")
        # Leaving step 0 for a fresh run retires the previous run's value recap
        # and the post-export success line.
        self.clear_last_run_recap()
        self.clear_auto_export_success()
        try:
            from ...core import telemetry
            telemetry.track_auto_start_clicked(
                layer_kind=self._auto_layer_kind(layer),
                has_credits_known=self._auto_credits is not None,
            )
        except Exception:
            pass  # nosec B110
        self._auto_started = True
        self._go_to_auto_step(1)

    @staticmethod
    def _auto_layer_kind(layer) -> str:
        """Coarse raster-source class for telemetry: local / xyz / wms / other."""
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
        """Exit the Automatic flow back to the Start step, layer editable again.

        Called by the dock Exit button (via the plugin, which first clears the
        zone + any review) and on any hard teardown that should unlock the
        layer. Idempotent.
        """
        self._auto_started = False
        # Any hard teardown / mode switch retires the post-export success line
        # (the export path re-shows it AFTER calling this).
        self.clear_auto_export_success()
        # Leaving the flow retires the free-trial zone-cap message.
        self.set_auto_zone_rejected(None)
        self.hide_auto_exemplar_nudge()
        self.set_auto_exhausted_subscribe_visible(False)
        # Clear any leftover run status ("Saved N polygon(s) to ...") so it never
        # lingers on the Start / prompt page of the next run.
        self.set_auto_status("idle")
        # Back at Start the footer is always usable again (covers any hard
        # teardown that bypasses set_auto_run_active(False), e.g. a mode switch
        # mid-run): never leave the gear/help locked once the flow is reset.
        self.set_footer_controls_locked(False)
        # Likewise re-enable the mode toggle, disabled during review: at Start
        # there is no review to protect (skip while a handoff owns the toggle).
        if not self._refine_handoff:
            self.mode_switch.setEnabled(True)
            self.mode_switch.setToolTip("")
        self.auto_prompt_input.blockSignals(True)
        self.auto_prompt_input.clear()
        self.auto_prompt_input.blockSignals(False)
        # The blocked clear() fired no textChanged, so the commit dedupe would
        # still hold the LAST run's prompt: typing the same object again next
        # run would silently skip its commit (no re-seed, no run plan). Reset
        # it with the box.
        self._last_committed_prompt = None
        self._auto_prompt_valid = False
        self._set_prompt_info()
        # The object is gone with the prompt: drop its slider verdict too.
        self._auto_detail_feedback = None
        # The prompt was cleared with signals blocked (no textChanged), so
        # refresh the Detect + Detail gates explicitly: the next run must
        # start with the Detail slider gated again until an object exists.
        self._update_auto_detect_enabled()
        # The review hides the layer header (see set_auto_review_active); a
        # hard teardown that skips the review-off call must restore it.
        self.auto_layer_combo.setVisible(True)
        self._go_to_auto_step(0)

    def _apply_auto_detail_gate(self, has_object: bool) -> None:
        """Grey the whole Detail card until the object is defined (typed prompt
        or drawn example). The slider's default is object-aware, so an
        adjustment made BEFORE the object was named was thrown away by the
        prompt-commit re-seed: gating the control makes the order explicit.
        Disabling the container blocks every child, and the opacity dim makes
        the gate unmistakable (a same-color disabled slider read as broken);
        the slider QSS adds a grey :disabled track on top so no brand blue
        survives the dim. The programmatic seed still lands (setValue works
        while disabled). The one-line hint explains the greyed state instead
        of leaving a dead control unexplained."""
        try:
            card = self.auto_detail_row
            if card.isEnabled() == has_object:
                return
            card.setEnabled(has_object)
            if has_object:
                card.setGraphicsEffect(None)
                # Route through the shared refresher so a capped slider keeps
                # its capped wording (free-plan upsell or zone advice).
                self._refresh_auto_detail_hint()
            else:
                dim = QGraphicsOpacityEffect(card)
                dim.setOpacity(0.45)
                card.setGraphicsEffect(dim)
                self.auto_detail_hint.setStyleSheet(
                    "font-size: 10px; color: palette(text);")
                self.auto_detail_hint.setText(tr(
                    "Name the object (or draw an example) first - Detail "
                    "then tunes itself to it."))
        except (RuntimeError, AttributeError):
            pass

    def set_prompt_text(self, text: str) -> None:
        """Set the prompt box (used by the Library 'Use this prompt' flow)."""
        self._prompt_from_library = True
        self.auto_prompt_input.setText(text or "")
        self.auto_prompt_input.setFocus()

    def _on_auto_search_text_changed(self, _text: str = "") -> None:
        # Track validity quietly while typing; the guard-rail message itself
        # only appears when the user commits (Detect / Enter), never on the
        # first keystrokes - see confirm_prompt_for_detect. Any edit clears a
        # previously shown message (the user is already acting on it).
        text = self.auto_prompt_input.text()
        ok, _reason, _suggestion = validate_prompt(text)
        self._auto_prompt_valid = ok
        # Editing the prompt makes the zero-result chips stale (their labels
        # quote the old word); the synonym prefill lands here too and cleans
        # itself up the same way.
        self.hide_auto_zero_assist()
        self._apply_prompt_hint_on_edit()
        self._update_auto_detect_enabled()
        # Re-seed the object-aware detail default once the typed object settles.
        self._auto_prompt_debounce_timer.start(500)

    def confirm_prompt_for_detect(self) -> bool:
        """Commit-time guard rail: called by the plugin when a detection is
        requested (Detect click or Enter). A clean prompt (or an example-only
        run with an empty prompt) passes; anything off the rails blocks the
        run and shows the guidance right under the prompt box, with focus back
        in it so the fix is one keystroke away."""
        text = self.auto_prompt_input.text()
        if not text.strip():
            # Example-only run: Detect is only enabled with a drawn example, so
            # an empty prompt commits straight away (the examples drive it).
            self._apply_prompt_hint_on_edit()
            return True
        ok, reason, suggestion = validate_prompt(text)
        if ok and reason is None and not is_known_object(text):
            # Valid-LOOKING but unrecognized word: could be a language the
            # offline lexicon does not cover (Polish, Turkish, ...) or a rare
            # English word. One cached server lookup decides; None means "run
            # it as typed".
            token = self._resolve_prompt_via_server(text)
            if token:
                reason, suggestion = "translated", token
        elif not ok and reason == "language":
            # Foreign word beyond the offline lexicon: try the same server
            # lookup before giving up and lecturing about English.
            token = self._resolve_prompt_via_server(text)
            if token:
                ok, reason, suggestion = True, "translated", token
        self._auto_prompt_valid = ok
        if ok:
            if reason == "translated" and suggestion:
                # The user typed the object in their own language: run the
                # English token for them (setText first - its textChanged
                # clears the info line, which we then set) and say so in a
                # quiet note, not an error. The run proceeds untouched.
                typed = text.strip()
                self.auto_prompt_input.setText(suggestion)
                self._set_prompt_info(
                    tr('"{word}" will run as "{token}".').format(
                        word=typed, token=suggestion), info=True)
            elif reason == "steer":
                # Valid word, weak from above ('wall'): a light non-blocking
                # nudge toward the term that detects best. The run proceeds as
                # typed - we do not overrule the user, only advise.
                self._set_prompt_info(
                    self._prompt_steer_message(suggestion), tip=True)
                self._track_prompt_steered(text, suggestion)
            elif reason == "multi_first" and suggestion:
                # Several objects in one box ("buildings and roads"): the
                # model grounds ONE concept per run, so run the FIRST object
                # now instead of refusing, with a quiet hint to run the rest
                # separately. setText first - its textChanged clears the info
                # line, which we then set (same order as the translated case).
                typed = text.strip()
                if suggestion != typed:
                    self.auto_prompt_input.setText(suggestion)
                self._set_prompt_info(
                    tr('One object per run - detecting "{first}" now. '
                       'Run the other objects as separate detections.').format(
                        first=suggestion), tip=True)
                try:
                    from ...core import telemetry
                    # prompt = the 1-2 word object that actually runs;
                    # "multi_first" marks the guided-multi case for analytics.
                    telemetry.track_auto_prompt_steered(
                        prompt=suggestion, suggestion="multi_first")
                except Exception:
                    pass  # nosec B110
            else:
                self._set_prompt_info()
            return True
        # Off-rails text, but a drawn example is already a full query: for the
        # generic/subjective cases (where no single word fits) point the user at
        # clearing the box to run from the example alone. Clearing it makes the
        # prompt empty, which passes the guard, so the run is one keystroke away.
        exemplar_guard = self._EXEMPLARS_ENABLED
        exemplar_guard = exemplar_guard and getattr(self, "_auto_positive_exemplars", 0) > 0
        exemplar_guard = exemplar_guard and reason in ("abstract", "subjective")
        if exemplar_guard:
            guidance = tr(
                "Too generic to name. Clear the box to search from your "
                "example alone, or type a concrete object.")
        else:
            guidance = self._prompt_guidance_message(reason, suggestion)
        self._set_prompt_info(guidance, error=True)
        self.auto_prompt_input.setFocus()
        self.auto_prompt_input.selectAll()
        try:
            from ...core import telemetry
            telemetry.track_detect_blocked(
                reason="prompt_{}".format(reason or "invalid"))
        except Exception:
            pass  # nosec B110
        return False

    def _resolve_prompt_via_server(self, text: str) -> str | None:
        """Commit-time-only server translation for words the offline lexicon
        cannot resolve. Blocking but cached (one round-trip per distinct word
        per machine, failures negative-cached for the session); a wait cursor
        covers the one slow first lookup. The returned token is re-vetted by
        the guard so a bad server answer can never reach the model."""
        try:
            from ...api.prompt_translation import resolve_english_prompt
        except Exception:  # noqa: BLE001
            return None
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            token = resolve_english_prompt(text)
        except Exception:  # noqa: BLE001
            token = None
        finally:
            QApplication.restoreOverrideCursor()
        if not token:
            return None
        ok, reason, _suggestion = validate_prompt(token)
        return token if ok and reason is None else None

    def _prompt_plausibly_complete(self, text: str) -> bool:
        """While the user is still typing in the box, only a prompt the
        vocabulary recognizes (English object word, catalogue token, or a
        known word in another language) commits on the debounce. Unknown
        fragments ('buil') wait for Enter / focus-out / Detect, where intent
        is explicit, so partial words never hit analytics or the server."""
        try:
            if not self.auto_prompt_input.hasFocus():
                return True
            return is_known_object(text) or english_token_for(text) is not None
        except (RuntimeError, AttributeError):
            return True

    def _on_auto_prompt_editing_finished(self) -> None:
        """Enter or focus-out: the prompt is explicitly settled. Flush any
        pending debounce and commit now, fragments included (blur is intent)."""
        try:
            self._auto_prompt_debounce_timer.stop()
        except (RuntimeError, AttributeError):
            pass
        self._emit_auto_prompt_committed(force=True)

    def _emit_auto_prompt_committed(self, force: bool = False) -> None:
        """Fire after the prompt settles so the plugin re-seeds the detail
        default for the current object (no-op when the box is empty; the plugin
        also respects a manual slider override and requires a drawn zone).

        Commit side-effects are one-shot per settled prompt: a mid-word
        fragment does not commit on the debounce (see
        _prompt_plausibly_complete), and the same text never commits twice in
        a row, so the downstream telemetry + server run-plan fetch fire about
        once per real prompt instead of once per typing pause."""
        text = self.auto_prompt_input.text().strip()
        if text and not force and not self._prompt_plausibly_complete(text):
            return
        if text == getattr(self, "_last_committed_prompt", None):
            return
        self._last_committed_prompt = text
        if text:
            # Light steer nudge the moment the prompt settles, BEFORE Detect is
            # clicked, so a weak-from-above word is redirected before any credit
            # is spent. The amber guard block stays commit-only; this is a quiet
            # advisory that never blocks.
            try:
                ok, reason, suggestion = validate_prompt(text)
                if ok and reason == "steer":
                    self._set_prompt_info(
                        self._prompt_steer_message(suggestion), tip=True)
                    self._track_prompt_steered(text, suggestion)
                elif ok and reason == "multi_first" and suggestion:
                    # Early heads-up for a several-objects prompt: the box is
                    # left as typed (the swap happens at Detect), the hint
                    # just says what will run so no credit surprises.
                    self._set_prompt_info(
                        tr('One object per run - Detect will run "{first}" '
                           'first.').format(first=suggestion), tip=True)
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:
                from ...core import telemetry
                telemetry.track_auto_prompt_committed(
                    prompt=text,
                    from_library=getattr(self, "_prompt_from_library", False),
                )
            except Exception:
                pass  # nosec B110
        # A library pick's from_library marker is consumed once here; later manual
        # edits report from_library=False.
        self._prompt_from_library = False
        self.auto_prompt_committed.emit(text)

    def _on_auto_search_return_pressed(self) -> None:
        # Enter in the prompt box routes through the plugin's single Enter
        # dispatcher (_route_enter), which launches the detection once
        # everything required (layer locked, zone drawn, object) is in place.
        self.auto_enter_pressed.emit()

    def _set_prompt_info(self, text: str | None = None, error: bool = False,
                         info: bool = False, tip: bool = False) -> None:
        """Guard-rail message under the prompt. Hidden when the prompt is empty
        or valid; an amber callout when the committed prompt is off the rails;
        a quiet neutral note (``info=True``) for the silent-translation case
        ('"piscine" will run as "swimming pool"'); a blue information callout
        with a leading info bubble (``tip=True``) for the steer nudge that
        suggests a better prompt. ``error`` is kept for call-site
        compatibility; plain non-info/non-tip text shows as the amber callout."""
        if not text:
            self.auto_prompt_info.setText("")
            self.auto_prompt_info.setVisible(False)
            return
        if tip:
            # Blue information callout: a leading info bubble (rich text so only
            # the glyph is blue) marks it clearly as a helpful tip, not an error.
            self.auto_prompt_info.setTextFormat(Qt.TextFormat.RichText)
            self.auto_prompt_info.setText(
                '<span style="color:{c}; font-weight:bold;">&#9432;</span>'
                '&nbsp;&nbsp;{t}'.format(c=BRAND_BLUE, t=text))
            self.auto_prompt_info.setStyleSheet(_msg_label_qss("info"))
            self.auto_prompt_info.setVisible(True)
            return
        self.auto_prompt_info.setTextFormat(Qt.TextFormat.PlainText)
        self.auto_prompt_info.setText(text)
        if info:
            self.auto_prompt_info.setStyleSheet(_msg_label_qss("neutral"))
        else:
            self.auto_prompt_info.setStyleSheet(_msg_label_qss("warning"))
        self.auto_prompt_info.setVisible(True)

    def _apply_prompt_hint_on_edit(self) -> None:
        """Keep the note under the prompt box in sync while the user types or
        draws examples. A non-empty prompt clears any stale guard message (the
        guard only fires on commit). With no prompt and examples drawn the note
        is count-aware: one positive nudges toward a second (Detect stays gated
        until two exist, since reference-image detection is far better with a
        pair); two or more say the examples now drive the run.

        This writes the prompt-info line (auto_prompt_info), a DIFFERENT label
        from the example card's shared armed/size-warning hint
        (auto_exemplar_armed_hint), so the two never fight one label. When the
        card is showing the too-small size warning it is the more urgent,
        actionable message, so the second-example nudge yields to it (one calm
        info per state); show/clear of that warning re-run this method so the
        nudge returns once the warning clears."""
        try:
            has_text = bool(self.auto_prompt_input.text().strip())
        except (RuntimeError, AttributeError):
            return
        positives = (getattr(self, "_auto_positive_exemplars", 0)
                     if self._EXEMPLARS_ENABLED else 0)
        if has_text or positives <= 0:
            # A prompt clears any stale note; no example means nothing to say.
            self._set_prompt_info()
            return
        if positives == 1:
            if getattr(self, "_auto_exemplar_hint_kind", None) == "warning":
                # The too-small size warning owns the guidance for this state.
                self._set_prompt_info()
                return
            self._set_prompt_info(
                tr("Add a second example - two references detect far better "
                   "than one."),
                info=True)
            return
        # Two or more positives: the example-only run is ready.
        self._set_prompt_info(tr("Your examples drive the search."), info=True)

    def _prompt_guidance_message(self, reason: str | None, suggestion: str | None) -> str:
        msgs = {
            "too_long": tr("Use just 1-2 words for the object."),
            "sentence": tr("Type the object itself, not a sentence or question."),
            "abstract": tr("Too generic. Draw an example instead, or use a "
                           "concrete word like building."),
            "subjective": tr("Name a concrete object, not how it looks."),
            "referential": tr("Segment one object - drop words like 'near' or 'with'."),
            "weird": tr("Use a real object word."),
            "language": tr("That word isn't recognized - try a common object like building or tree."),
            "multi": tr("One object per run - start with the first one, then run again."),
        }
        base = msgs.get(reason or "", tr("Use a 1-2 word object name."))
        if suggestion:
            return base + " " + tr("Did you mean '{term}'?").format(term=suggestion)
        return base + " " + tr("The Library has ready-to-use objects.")

    def _prompt_steer_message(self, suggestion: str | None) -> str:
        """Light nudge for a valid but weak prompt: name the single object term
        that works better, or point at the Library when there is no clean term."""
        if suggestion:
            return tr("Try '{term}' - it's a better prompt.").format(
                term=suggestion)
        return tr("Try an object from the Library - it's a better prompt.")

    def _track_prompt_steered(self, prompt: str, suggestion: str | None) -> None:
        """Fire-and-forget telemetry when the steer nudge is shown, so the
        weak-prompt demand signal (which words users try, what we steer to) is
        measurable. Deduped per distinct prompt per session so a re-commit of
        the same word does not double-count."""
        try:
            seen = self._steered_prompts_seen
        except AttributeError:
            seen = self._steered_prompts_seen = set()
        key = (prompt or "").strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        try:
            from ...core import telemetry
            telemetry.track_auto_prompt_steered(
                prompt=key, suggestion=(suggestion or ""))
        except Exception:  # noqa: BLE001
            pass  # nosec B110

    def _is_auto_for_us(self) -> bool:
        """True when the Automatic flow should own Enter: in Automatic mode,
        the flow is started, and no run is in flight (Enter has no job during
        a run; Escape has its own gate so it can soft-cancel one)."""
        return self._mode == Mode.AUTOMATIC and self._auto_started and not self._auto_run_active

    def _on_auto_escape_shortcut(self) -> None:
        """Escape: delegate to the plugin's single dispatcher (_route_escape).

        Unlike Enter this also fires DURING a run (Escape = soft Cancel
        there), so it gates only on the Automatic flow being started."""
        if self._mode == Mode.AUTOMATIC and self._auto_started:
            self.auto_escape_pressed.emit()

    def _on_auto_enter_shortcut(self) -> None:
        """Enter: delegate to the plugin's single dispatcher (_route_enter),
        unless a text editor or spinbox has focus (its own returnPressed /
        value-commit handles Enter, avoiding a double trigger)."""
        if not self._is_auto_for_us():
            return
        fw = QApplication.focusWidget()
        if isinstance(fw, (QLineEdit, QComboBox, QAbstractSpinBox)):
            return
        self.auto_enter_pressed.emit()

    def set_auto_shortcuts_enabled(self, enabled: bool) -> None:
        """Toggle the Escape/Enter shortcuts. Disabled while an example box is
        being drawn so the draw tool owns Escape (cancel) with no race."""
        for sc in (self.auto_escape_shortcut, self.auto_enter_shortcut,
                   self.auto_enter_shortcut_kp):
            try:
                sc.setEnabled(enabled)
            except (RuntimeError, AttributeError):
                pass

    def _on_auto_detail_changed(self, value: int) -> None:
        # The slider now shows plain Coarse/Fine ends; the only numeric feedback
        # is the credit cost, which the plugin recomputes from the real grid.
        self._refresh_auto_detail_hint()
        self.auto_detail_changed.emit(value)

    # -- Review confidence: slider <-> spinbox sync + debounced re-filter -----

    def _on_conf_slider_moved(self, value: int) -> None:
        """Slider dragged: snap to the nearest 5% stop, mirror into the spinbox
        only (cheap, no re-filter). QSlider singleStep only constrains keyboard
        input, so the mouse drag is snapped here to stop the handle (and the
        debounced re-filter on release) from landing on every tiny percentage.
        The spinbox keeps 1% precision; this snapped mirror just rounds it when
        the user drives from the slider."""
        snapped = _snap_review_conf(value)
        if snapped != value:
            self.auto_review_confidence_slider.blockSignals(True)
            self.auto_review_confidence_slider.setValue(snapped)
            self.auto_review_confidence_slider.blockSignals(False)
        if self.auto_review_confidence_spin.value() != snapped:
            self.auto_review_confidence_spin.blockSignals(True)
            self.auto_review_confidence_spin.setValue(snapped)
            self.auto_review_confidence_spin.blockSignals(False)
        # A user-initiated move retires the auto-lowered note and moves the
        # histogram's dimmed/kept boundary.
        self.auto_conf_lowered_note.setVisible(False)
        if getattr(self, "auto_conf_histogram", None) is not None:
            self.auto_conf_histogram.set_cutoff(snapped / 100.0)
        # Live feedback as the handle moves: a fast, cheap preview re-filter on a
        # short debounce, plus the accurate rebuild on a longer one so it also
        # runs after the user settles via the keyboard (no sliderReleased then).
        # A mouse release additionally triggers the accurate path immediately.
        self._auto_conf_preview_timer.start(40)
        self._auto_conf_debounce_timer.start(250)

    def _emit_auto_confidence_preview(self) -> None:
        self.auto_review_confidence_preview.emit(self.auto_review_confidence_spin.value())

    def _on_conf_spin_changed(self, value: int) -> None:
        """Spinbox edited: mirror into the slider, then schedule the re-filter."""
        if self.auto_review_confidence_slider.value() != value:
            self.auto_review_confidence_slider.blockSignals(True)
            self.auto_review_confidence_slider.setValue(value)
            self.auto_review_confidence_slider.blockSignals(False)
        # The slider mirror above is signal-blocked, so move the histogram's
        # kept/dimmed boundary here too (the slider path does it on its own move).
        if getattr(self, "auto_conf_histogram", None) is not None:
            self.auto_conf_histogram.set_cutoff(value / 100.0)
        self._schedule_conf_refilter()

    def seed_review_confidence(self, pct: int) -> None:
        """Signal-free mirror of the starting cutoff into the review slider and
        spinbox. The review page seeds its widgets from the pre-run dial when it
        opens, but the async finalize computes the real starting cutoff (class
        default / adaptive) AFTER that, so it pushes the final value here; the
        handle, the spin, the histogram boundary and the count line then all
        read the same number."""
        try:
            value = int(pct)
            self.auto_review_confidence_slider.blockSignals(True)
            self.auto_review_confidence_slider.setValue(value)
            self.auto_review_confidence_slider.blockSignals(False)
            self.auto_review_confidence_spin.blockSignals(True)
            self.auto_review_confidence_spin.setValue(value)
            self.auto_review_confidence_spin.blockSignals(False)
        except (RuntimeError, AttributeError):
            pass

    def set_review_conf_floor(self, floor_pct: int) -> None:
        """Clamp the review confidence controls so neither the slider nor the
        spinbox can dial below the run's noise floor: sub-floor detections are
        excluded from the review, so a cutoff under it would filter nothing.
        Keeps the design constants as the minimum; only raises them."""
        slider_min = max(_REVIEW_CONF_MIN, int(floor_pct))
        spin_min = max(_REVIEW_CONF_SPIN_MIN, int(floor_pct))
        try:
            self.auto_review_confidence_slider.setMinimum(slider_min)
            self.auto_review_confidence_spin.setMinimum(spin_min)
        except (RuntimeError, AttributeError):
            pass

    def _schedule_conf_refilter(self) -> None:
        """Coalesce rapid confidence changes so the heavy re-merge runs once."""
        self._auto_conf_debounce_timer.start(200)

    def _emit_auto_confidence_changed(self) -> None:
        # Emit the spinbox value: it is the exact-cutoff source of truth (free 1%
        # precision). A slider drag has already snapped itself to a 5% step and
        # mirrored that into the spinbox, so this stays correct for both paths.
        self.auto_review_confidence_changed.emit(self.auto_review_confidence_spin.value())

    def set_auto_detail_value(self, n: int) -> None:
        """Seed the detail slider with a good default for a freshly drawn zone.

        Signal-free: the plugin recomputes the credit estimate (and the proper
        slider max) right after. Raises the max first if needed so the seeded
        value is not clamped by a previous zone's smaller cap.
        """
        s = self.auto_detail_slider
        s.blockSignals(True)
        if s.maximum() < n:
            s.setMaximum(min(MAX_DETAIL_LEVEL, int(n)))
        s.setValue(max(s.minimum(), min(s.maximum(), int(n))))
        s.blockSignals(False)
        self._refresh_auto_detail_hint()

    def set_auto_detail_visible(self, visible: bool) -> None:
        """Show the detail slider whenever a zone is drawn; hidden while no zone is set."""
        self.auto_detail_row.setVisible(visible)

    def set_auto_detail_gsd_warning(self, coarse: bool) -> None:
        """Show the boxed amber warning when the chosen detail leaves the imagery
        too coarse for the cloud model (effective ground resolution >= ~0.5 m/px, where
        detection quality drops sharply). The detail seed now auto-raises past
        the soft tile budget, so this fires only when the USER dragged detail
        down (fix: raise it back) or the zone is so large even the slider max
        stays coarse (fix: a smaller zone; "raise detail" would be a dead end).
        The neutral hint hides while the warning shows so the two never stack."""
        if coarse:
            s = self.auto_detail_slider
            self.auto_detail_warning_label.setText(
                tr("This area is large for this detail level. Raise detail or zoom"
                   " in for sharper detections.")
                if s.value() < s.maximum() else
                tr("This zone is too large for sharp detections, even at maximum"
                   " detail. Draw a smaller zone for the best results."))
        self.auto_detail_warning.setVisible(coarse)
        self.auto_detail_hint.setVisible(not coarse)

    def set_auto_detail_max(self, n: int) -> None:
        """Cap the detail slider at ``n`` useful levels (1-MAX_DETAIL_LEVEL).

        Clamps the current value down if it now exceeds the cap. Signal-free
        on purpose: the plugin calls this from _update_credit_estimate right
        before recomputing the grid, so the clamped value is picked up
        immediately without a re-entrant signal.
        """
        n = max(1, min(MAX_DETAIL_LEVEL, int(n)))
        slider = self.auto_detail_slider
        slider.blockSignals(True)
        slider.setMaximum(n)
        if slider.value() > n:
            slider.setValue(n)
        slider.blockSignals(False)
        self._refresh_auto_detail_hint()

    def set_auto_free_run_cap(self, cap: int | None) -> None:
        """Per-run credit cap for the free plan (None = subscriber, uncapped).

        Set by the plugin from the credit-estimate chokepoint, right before
        the estimate itself lands. The slider keeps its full (Pro) travel; the
        cap gates DETECT instead: set_auto_credit_estimate compares the live
        estimate against it and flips the premium gate."""
        self._auto_free_run_cap = int(cap) if cap is not None else None

    def _set_auto_premium_gated(self, gated: bool) -> None:
        """Flip the free-plan premium gate (estimate above the per-run cap).

        Greys Detect (via _update_auto_detect_enabled, run by the caller) and
        swaps the detail hint to the upgrade link. The upsell view is tracked
        once per gate episode (rising edge)."""
        gated = bool(gated)
        if gated == self._auto_premium_gated:
            return
        self._auto_premium_gated = gated
        if gated and not self._detail_cap_upsell_tracked:
            self._detail_cap_upsell_tracked = True
            try:
                from ...core import telemetry
                telemetry.track_pro_upsell_viewed(trigger="detail_cap")
            except Exception:
                pass  # nosec B110
        elif not gated:
            # Next gate episode counts as a fresh upsell view.
            self._detail_cap_upsell_tracked = False
        self._refresh_auto_detail_hint()

    def _on_detail_cap_upgrade_link(self, _href: str = "") -> None:
        """Upgrade link inside the detail hint: same dashboard URL as every
        other upsell surface, its own telemetry source."""
        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices
        try:
            from ...core import telemetry
            telemetry.track_pro_upsell_clicked(source="detail_cap")
        except Exception:
            pass  # nosec B110
        QDesktopServices.openUrl(QUrl(self._build_upgrade_url()))

    def set_auto_detail_feedback(self, state: str | None, object_word: str) -> None:
        """Live verdict for the CURRENT slider level against the named object
        and the drawn zone, computed by the plugin at the credit-estimate
        chokepoint. States: coarse / below / recommended / helps / above /
        over (None clears). Stored here and rendered by
        _refresh_auto_detail_hint, which owns the priority order."""
        word = (object_word or "").strip()
        if len(word) > 24:
            word = word[:24] + "..."
        self._auto_detail_feedback = (state, word) if state else None
        self._refresh_auto_detail_hint()

    def _refresh_auto_detail_hint(self) -> None:
        """Swap the muted line under the detail slider by state. Premium-gated
        (the free plan, not the zone, is what blocks this level) shows the
        upgrade link; then the object-aware verdict when one is known, so the
        guidance moves live with the slider, the prompt and the zone; the
        handle sitting at a zone/native-capped maximum keeps the
        draw-a-larger-zone advice when raising detail is the (impossible)
        fix, so a slider that stops early never reads as broken. Same label,
        text swap only (no layout jump)."""
        s = self.auto_detail_slider
        capped = s.maximum() < MAX_DETAIL_LEVEL and s.value() >= s.maximum()
        feedback = getattr(self, "_auto_detail_feedback", None)
        _plain_hint = "font-size: 10px; color: palette(text);"
        if self._auto_premium_gated:
            # Premium taxonomy: a dedicated blue-family line with the star
            # prefix and an underlined upgrade link (never inline in guidance).
            self.auto_detail_hint.setStyleSheet(_msg_label_qss("premium"))
            _hint = _PREMIUM_STAR + " "
            _hint += tr("This detail level is a Pro feature. Lower the detail, or")
            _hint += f' <a href="upgrade" style="color: {BRAND_BLUE};'
            _hint += ' text-decoration: underline;">'
            _hint += tr("upgrade to unlock it")
            _hint += "</a>."
            self.auto_detail_hint.setText(_hint)
            return
        if feedback and not (capped and feedback[0] in ("coarse", "below")):
            # "Raise the detail" advice is a dead end at a capped maximum;
            # the capped branch below gives the actionable fix instead.
            state, word = feedback
            obj = f'"{word}"' if word else tr("your object")
            if state == "coarse":
                self.auto_detail_hint.setStyleSheet(_msg_label_qss("warning"))
                self.auto_detail_hint.setText(_msg_text("warning", tr(
                    "At this detail {obj} is too small to spot - raise the"
                    " detail.").format(obj=obj)))
            elif state == "over":
                # Quality fact only (large objects can fragment past this
                # point); never a nudge about credits - the cost line above
                # already says the price, guidance stays informational.
                self.auto_detail_hint.setStyleSheet(_msg_label_qss("warning"))
                self.auto_detail_hint.setText(_msg_text("warning", tr(
                    "Very fine for {obj} - large ones may come back split"
                    " in parts.").format(obj=obj)))
            elif state == "above":
                self.auto_detail_hint.setStyleSheet(_plain_hint)
                self.auto_detail_hint.setText(tr(
                    "Sharper than {obj} usually needs - catches the smallest"
                    " ones.").format(obj=obj))
            elif state == "helps":
                self.auto_detail_hint.setStyleSheet(_plain_hint)
                self.auto_detail_hint.setText(tr(
                    "Extra detail keeps helping {obj} in this zone.").format(
                        obj=obj))
            elif state == "below":
                self.auto_detail_hint.setStyleSheet(_plain_hint)
                self.auto_detail_hint.setText(tr(
                    "Small {obj} may be missed at this level.").format(obj=obj))
            else:  # recommended
                self.auto_detail_hint.setStyleSheet(_plain_hint)
                self.auto_detail_hint.setText("✓ " + tr(
                    "Right level for {obj} in this zone.").format(obj=obj))
            return
        if capped:
            self.auto_detail_hint.setStyleSheet(_plain_hint)
            self.auto_detail_hint.setText(tr(
                "Max detail for this zone - draw a larger zone for finer detail."))
        else:
            self.auto_detail_hint.setStyleSheet(_plain_hint)
            self.auto_detail_hint.setText(tr("Finer detail finds smaller objects."))

    def on_zone_deleted_from_canvas(self) -> None:
        """Called by the plugin when the user clicks the zone's x badge."""
        self.set_auto_zone_state("idle")
        self._go_to_auto_step(1)

    def _go_to_auto_step(self, index: int) -> None:
        """Switch the Automatic flow to the given step.

        step 0 Start (layer editable) | step 1 Draw zone | step 2 Prompt +
        settings. From step 1 on the layer header is locked; the canvas x badge
        re-draws the zone and the Exit button returns to step 0.
        """
        # A running detection pins the flow to the launch step.
        if self._auto_run_active:
            index = 2
        # The stack is hidden in the empty state (hero only); any explicit step
        # change means the flow is live, so it must be visible again.
        self.auto_steps.setVisible(True)
        self.auto_steps.setCurrentIndex(index)
        self._refresh_auto_layer_lock()
        # Leaving step 0 launches the Automatic flow (returning to it exits): the
        # mode switch is only shown on the Start screen, so re-evaluate it here.
        self._refresh_mode_switch_visibility()
        self._update_auto_detect_enabled()
        # The exemplar panel belongs to the idle prompt step (2): an example is
        # drawn inside the zone, which exists by the time step 2 opens. Hidden on
        # the Start/Draw steps and while a run or review is active. Also gated
        # behind _EXEMPLARS_ENABLED (feature hidden for now).
        self.auto_exemplar_panel.setVisible(
            self._EXEMPLARS_ENABLED and index == 2 and not self._auto_run_active and not self._auto_review_active
        )
        # The bottom-pinned first-steps guide banner shows on the Start step only.
        self._update_auto_tutorial_banner_visibility()
        # The plugin reacts to step changes (e.g. arms the zone drawing tool
        # whenever the zone step opens without a zone set).
        self.auto_step_changed.emit(index)

    def _refresh_auto_layer_lock(self) -> None:
        """Greyed/locked layer header from step 1 on; editable on the Start step.

        Reuses the Interactive locked-combo look: disabled combo with the
        dropdown arrow hidden, and the label hidden so only the raster name
        shows. Step 0 restores the editable combo + its label.
        """
        # Empty-project guard: on the Start step with NO raster loaded the page
        # is the no-imagery hero, so the header must stay hidden. Without this,
        # this method (which runs LAST on a combo change) would re-show the
        # "Select a Raster Layer" label over the hero - the orphaned-label bug
        # seen after deleting the last layer. The fresh path never calls this,
        # which is why only the delete path broke.
        if self.auto_steps.currentIndex() == 0 and self.auto_layer_combo.count_layers() == 0:
            self.auto_layer_label.setVisible(False)
            self.auto_layer_combo.setVisible(False)
            return
        on_start = self.auto_steps.currentIndex() == 0
        self.auto_layer_label.setVisible(on_start)
        self.auto_layer_combo.setEnabled(on_start)
        # Freeze the combo whenever the header is locked (steps 1/2/run/review):
        # once a source is locked, hiding layers in the tree must not drop it from
        # the list or re-pick another. Unfreezes + resyncs on returning to Start.
        self.auto_layer_combo.set_frozen(not on_start)
        if on_start:
            self.auto_layer_combo.setStyleSheet(
                "QComboBox { color: palette(text); }")
        else:
            self.auto_layer_combo.setStyleSheet(
                "QComboBox { color: palette(text); }"
                "QComboBox::drop-down { width: 0px; border: none; }")

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

    def set_auto_zone_rejected(self, area_km2: float | None) -> None:
        """Show (or hide with None) the free-trial zone-cap message in the
        step-1 draw hero: the drawn zone was refused because it exceeds
        FREE_TRIAL_MAX_ZONE_KM2. Contextual upsell touchpoint: the subscribe
        link opens the same dashboard upgrade URL as the footer pill. The
        message clears as soon as a valid zone lands or the flow is exited
        (see set_auto_zone_state / reset_auto_to_start)."""
        label = getattr(self, "_auto_zone_cap_label", None)
        if area_km2 is None:
            if label is not None:
                try:
                    label.setVisible(False)
                except (RuntimeError, AttributeError):
                    pass
            return
        if label is None:
            from qgis.PyQt.QtWidgets import QLabel
            label = QLabel()
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextBrowserInteraction)
            # Quiet warning card: translucent amber tint from the message
            # taxonomy, readable text on both themes (palette(text)).
            label.setStyleSheet(_msg_label_qss("warning"))
            label.linkActivated.connect(self._on_zone_cap_link_activated)
            try:
                self.auto_zone_hero.layout().addWidget(label)
            except (RuntimeError, AttributeError):
                return
            self._auto_zone_cap_label = label
        from ..plugin.shared import free_zone_cap_km2
        line1 = tr(
            "This zone is {area} km² - free trial zones go up to {max} km²."
        ).format(area="{:.1f}".format(area_km2),
                 max="{:g}".format(free_zone_cap_km2()))
        line2 = tr(
            'Draw a smaller zone, or <a href="{url}">subscribe</a> to '
            "segment areas of any size."
        ).format(url=self._build_upgrade_url())
        label.setText("{}<br/>{}".format(_msg_text("warning", line1), line2))
        label.setVisible(True)

    def _on_zone_cap_link_activated(self, url: str) -> None:
        """Subscribe link inside the zone-cap message: same destination as the
        footer pill, tracked with its own upsell source."""
        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices
        try:
            from ...core import telemetry
            telemetry.track_pro_upsell_clicked(source="zone_too_large")
        except Exception:
            pass  # nosec B110
        QDesktopServices.openUrl(QUrl(url))

    def _update_auto_low_credit_note(self) -> None:
        """Free-tier low-credit nudge on the Automatic Start step (step 0).

        Shows a discreet one-line "Running low" note with a Subscribe link once
        a free user drops below 20% of their free detections (and still has some
        left; a fully exhausted balance shows the upsell card instead).
        Subscribers never see it: the footer credit ring owns their balance.
        The line lives on the step-0 page, so it only shows on Start.
        """
        remaining = self._auto_credits
        total = self._auto_credits_total
        show = self._mode == Mode.AUTOMATIC and self._plugin_activated
        show = show and not self._auto_is_subscriber
        show = show and remaining is not None and total and total > 0
        show = show and 0 < remaining <= total * 0.20
        line = getattr(self, "_auto_low_credit_line", None)
        if not show:
            if line is not None:
                try:
                    line.setVisible(False)
                except (RuntimeError, AttributeError):
                    pass
            return
        if line is None:
            line = self._build_auto_low_credit_line()
            if line is None:
                return
        line.setText(tr(
            'Running low: {n} free detections left. '
            '<a href="{url}">Subscribe</a> to keep going.'
        ).format(n=remaining, url=self._build_upgrade_url()))
        line.setVisible(True)
        # Track the banner view once per session (the click was already
        # tracked, the view was not).
        if not getattr(self, "_low_credit_note_seen", False):
            self._low_credit_note_seen = True
            try:
                from ...core import telemetry
                telemetry.track_low_credit_banner_viewed(int(remaining), int(total))
            except Exception:  # nosec B110
                pass

    def _build_auto_low_credit_line(self):
        """Lazily create the step-0 low-credit note (amber card) and slot it
        under the free-trial line. Returns the label, or None if the step-0
        page is not built yet."""
        from qgis.PyQt.QtWidgets import QLabel
        try:
            page = self.auto_steps.widget(0)
            layout = page.layout()
        except (RuntimeError, AttributeError):
            return None
        if layout is None:
            return None
        label = QLabel()
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        label.setOpenExternalLinks(False)
        label.setStyleSheet(_msg_label_qss("warning"))
        label.linkActivated.connect(self._on_low_credit_link_activated)
        anchor = getattr(self, "auto_start_caption", None)
        idx = layout.indexOf(anchor) if anchor is not None else -1
        if idx >= 0:
            layout.insertWidget(idx + 1, label)
        else:
            layout.addWidget(label)
        self._auto_low_credit_line = label
        return label

    def _on_low_credit_link_activated(self, url: str) -> None:
        """Subscribe link inside the low-credit note: same destination as the
        footer pill, tracked with its own upsell source."""
        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices
        try:
            from ...core import telemetry
            telemetry.track_pro_upsell_clicked(source="low_credit")
        except Exception:
            pass  # nosec B110
        QDesktopServices.openUrl(QUrl(url))

    def set_auto_zone_state(self, state: str) -> None:
        """Reflect the canvas zone state. State: 'idle', 'drawing', 'zone_set'."""
        self._auto_zone_is_set = state == "zone_set"
        # A valid zone landing, or leaving the flow, retires the free-trial
        # zone-cap message; while state stays 'drawing' (a rejected zone puts
        # the user back there) the message remains as guidance.
        if state in ("idle", "zone_set"):
            self.set_auto_zone_rejected(None)
        if state in ("idle", "drawing"):
            self._auto_zone_too_large = False
            # No zone = no per-zone estimate; drop the stale cost label AND the
            # cached estimate/gate so neither lingers from the previous zone.
            self._auto_est_credits = None
            self._auto_insufficient_credits = False
            self.auto_credit_cost_label.setVisible(False)
        elif state == "zone_set":
            # A drawn zone completes step 2.
            self._go_to_auto_step(2)
            # Land AND HOLD the caret in the prompt box so the user types the
            # object without a click. A single setFocus is not enough: the zone
            # finishes on a canvas event, and for the first ~second the map
            # canvas can reclaim keyboard focus on a deferred repaint (the slow
            # basemap redraw), which yanked the caret back out after ~0.5 s. So
            # this holds focus across that window (see _begin_auto_prompt_focus).
            self._begin_auto_prompt_focus()
        self._update_auto_detect_enabled()

    def _begin_auto_prompt_focus(self) -> None:
        """Start holding the keyboard caret in the prompt box for ~1 s.

        The caret is re-asserted on a short repeating timer because a single
        setFocus loses to the canvas reclaiming focus a few hundred ms later
        (deferred basemap repaint after the zone is drawn). Each tick reclaims
        focus ONLY when it has drifted OUTSIDE the dock (i.e. to the map
        canvas) or nowhere, so a deliberate click on another dock control is
        respected and typing is never interrupted. Stops after the window."""
        from qgis.PyQt.QtCore import QTimer
        timer = getattr(self, "_auto_prompt_focus_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setInterval(120)  # ~8 ticks fill the ~1 s steal window
            timer.timeout.connect(self._tick_auto_prompt_focus)
            self._auto_prompt_focus_timer = timer
        self._auto_prompt_focus_ticks = 0
        self._tick_auto_prompt_focus()  # claim immediately, then hold
        timer.start()

    def _tick_auto_prompt_focus(self) -> None:
        """One focus-hold tick (see _begin_auto_prompt_focus)."""
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
            # Reclaim only when focus sits outside the dock (the map canvas) or
            # nowhere; a real click on a sibling dock control (focus inside the
            # dock) is left alone, and if the prompt already holds it this is a
            # no-op.
            if focused is not prompt and (
                    focused is None or not self.isAncestorOf(focused)):
                prompt.setFocus(Qt.FocusReason.OtherFocusReason)
            if self._auto_prompt_focus_ticks >= 8 and timer is not None:
                timer.stop()
        except (RuntimeError, AttributeError):
            if timer is not None:
                timer.stop()

    def set_zone_draw_progress(self, count: int) -> None:
        """Live guidance under the 'Draw your zone' title while the user clicks
        points, so it is always clear what to do next and how to finish."""
        if count <= 0:
            txt = tr("Click on the map to outline the area to scan.")
        elif count < 3:
            txt = tr("Keep clicking around the area, at least 3 points.")
        else:
            txt = tr("Click the first point to close the zone.")
        try:
            self._auto_zone_hint.setText(txt)
        except (RuntimeError, AttributeError):
            pass

    def set_auto_credits(self, credits: int, reset_date: str,
                         is_subscriber: bool,
                         total: int | None = None) -> None:
        """Called by plugin after loading usage data. Main thread only.

        ``total`` feeds the footer credit ring (remaining / total). Optional
        so older callers keep working; without it the ring stays hidden and
        only the count label shows.
        """
        self._auto_credits = credits
        self._auto_credits_total = total
        self._auto_is_subscriber = is_subscriber
        self._auto_reset_date = reset_date or ""
        if not is_subscriber:
            self._auto_free_left = credits
        self._refresh_auto_credits_display()
        # A balance change (typically the post-run refresh that debits the tiles
        # just spent) must re-run the credit gate against the LAST estimate, so a
        # now-underfunded zone blocks Detect immediately instead of waiting for
        # the next slider move. Only while a zone is set and no run/review owns
        # the cost label, so this never force-shows it on the Start step.
        _cost_label_free = not self._auto_run_active and not self._auto_review_active
        if self._auto_est_credits is not None and self._auto_zone_is_set and _cost_label_free:
            self.set_auto_credit_estimate(self._auto_est_credits)
        self._update_full_ui()

    def set_auto_run_active(self, active: bool) -> None:
        self._auto_run_active = active
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
        self.set_footer_controls_locked(active)
        # Mirror AI Edit: while tiles are in flight, clear away the non-essential
        # params (detail, confidence, cost) and the Detect/Exit row so only the
        # "Detecting X" label + progress + Cancel remain. They reappear when the
        # run ends; if the run then enters review, set_auto_review_active
        # re-hides them. The detail row honors the zone state on restore.
        self.auto_detect_row.setVisible(not active)
        # The confidence box stays hidden in the prompt step (post-run only).
        self.auto_settings_box.setVisible(False)
        self.auto_detail_row.setVisible(
            self._auto_zone_is_set if not active else False)
        self.auto_credit_cost_label.setVisible(
            self.auto_credit_cost_label.text() != "" if not active else False)
        # Keep the prompt card VISIBLE during a run, read-only (AI Edit pattern):
        # the chosen object stays framed above the progress so the user always
        # knows what is being detected, and the Library button stays clickable
        # (view-only). It returns to editable when the run ends; if the run
        # enters review, set_auto_review_active hides the whole card.
        self.auto_prompt_card.setVisible(True)
        self._set_auto_prompt_readonly(active)
        if active:
            self._go_to_auto_step(2)
        else:
            self._refresh_auto_layer_lock()
        # Keep the drawn reference visible during a run, read-only: hide the
        # add/exclude/remove affordances, keep the thumbnails browsable (click
        # to enlarge). No reference drawn = the whole panel stays hidden. Done
        # AFTER _go_to_auto_step, which force-hides the panel during a run - this
        # is the deliberate in-run exception that keeps the reference on screen.
        has_ref = self._EXEMPLARS_ENABLED and self._auto_positive_exemplars > 0
        if active:
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
            # Fresh warming counter + no known queue place yet, so the elapsed
            # readout starts at zero for this run.
            self._auto_queue_position = 0
            self._auto_queue_eta = 0
            self._stop_auto_warming_anim()
            self.auto_progress_count_label.setText("")
            self.auto_progress_pct_label.setText("")
            self.auto_progress_label.setVisible(False)
        else:
            # Run ended (review / Exit / error): stop the heartbeat.
            self._stop_auto_warming_anim()

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

    def set_auto_run_found_count(self, obj: str, count: int) -> None:
        """Live in-run feedback: the running found-object count, shown in the
        progress card's Row 1 next to the tile count so a slow zone never feels
        dead. ``obj`` is kept for call-site compatibility (the card no longer
        restates the object; the prompt card above already names it)."""
        self._auto_found_count = count if count > 0 else 0
        if self.auto_progress_card.isVisible():
            self._refresh_auto_progress_readout()

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

    def _format_auto_review_count(self, visible: int, total: int, pct: int,
                                  size_bound: bool = False) -> str:
        """ONE compact review readout line, always honest: green check + bold
        shown-count, then a muted tail with what the confidence cutoff hides.
        Sits at the top of the review card (it is the live readout of the
        filters below it). A run that found something NEVER reads as '0
        detected'. ``size_bound`` (only when visible == 0) swaps the reveal hint
        to the Min size filter when that, not Confidence, is hiding everything.
        The check is the lime success accent (the CTA green never announces
        success)."""
        check = f'<span style="color:{BRAND_GREEN};">&#10003;</span> '
        muted = 'style="color: rgba(128,128,128,0.95);"'
        if total <= 0:
            # Empty runs use the guidance box instead of this label; safe fallback.
            return '<b>{title}</b>'.format(title=tr("No objects found"))
        if visible >= total:
            bold = (tr("1 object found") if total == 1
                    else tr("{n} objects found").format(n=total))
            tail = tr("all shown")
        elif visible > 0:
            bold = tr("{visible} of {n} shown").format(visible=visible, n=total)
            tail = tr("{hidden} below {pct}%").format(
                hidden=total - visible, pct=pct)
        else:
            # No green check at 0 visible: nothing is shown, but the count is
            # honest and the tail tells the user how to reveal them - naming the
            # binding filter (Min size vs Confidence) so they pull the right one.
            bold = (tr("1 object found") if total == 1
                    else tr("{n} objects found").format(n=total))
            if size_bound:
                tail = tr("0 shown - lower the Min size filter to reveal them")
            else:
                tail = tr(
                    "0 shown at {pct}% - lower Confidence to reveal them").format(pct=pct)
            return '<b>{bold}</b> <span {muted}>· {tail}</span>'.format(
                bold=bold, muted=muted, tail=tail)
        return '{check}<b>{bold}</b> <span {muted}>· {tail}</span>'.format(
            check=check, bold=bold, muted=muted, tail=tail)

    def set_auto_review_installing(self, active: bool) -> None:
        """Reflect a background local-AI install kicked off from the review (D1).

        Clicking Refine in Manual mode without the local AI installed starts the
        setup in the BACKGROUND while this review stays fully usable. The inline
        banner shows progress (fed by set_install_progress) and the Refine button
        is disabled until the AI is ready, at which point the plugin re-enables it
        and opens the handoff automatically.
        """
        self._auto_review_installing = bool(active)
        try:
            self.auto_review_install_banner.setVisible(active)
            if active:
                self.auto_review_install_progress.setValue(0)
                self.auto_review_install_label.setText(tr(
                    "Setting up Manual mode in the background. You can keep "
                    "reviewing; refining opens automatically when it is ready."))
            self.auto_refine_in_manual_btn.setEnabled(not active)
            self.auto_refine_in_manual_btn.setText(
                ("✎  " + tr("Preparing Manual mode...")) if active
                else "✎  " + tr("Refine in Manual mode"))
        except (RuntimeError, AttributeError):
            pass

    def set_auto_review_active(self, active: bool, count: int = 0,
                               reset_controls: bool = True,
                               preset: dict | None = None) -> None:
        """Post-run review state: refine + Finish replace the Detect/Exit row.

        reset_controls=False keeps the current filter values: used when returning
        from a Manual refine handoff, where a reset would wipe the locked
        confidence and any size filter the user set before handing off.

        preset: the run's smart review defaults (core.review_presets dict,
        keyed like the controls). When given with reset_controls, the shape
        refine + Min size seed from it instead of the static review_defaults
        constants, so the first result is already tuned to the prompt and the
        run resolution. Every NEW review reseeds (no cross-run memory).
        """
        self._auto_review_active = active
        # A fresh review or a review teardown ends any prior background-install
        # banner (D1); a refine-handoff return (reset_controls=False) is left as
        # is (its env is already ready, no install running).
        if not active or reset_controls:
            self.set_auto_review_installing(False)
        # The exemplar nudge is per-run: clear it on every review transition; the
        # plugin re-shows it after a fresh review only when the run was
        # bottom-heavy and used no example.
        self.hide_auto_exemplar_nudge()
        if active and reset_controls:
            # Fresh review: reset the per-control shape-adjust telemetry dedup set.
            self._review_shape_tracked = set()
            # The collapsible "Shape and size settings" section starts closed
            # on every NEW review (setVisible only, no control signals) so
            # Confidence and Export stay above the fold. A refine handoff
            # return (reset_controls=False) keeps whatever the user had open.
            self.set_auto_shape_expanded(False)
        # The top mode toggle is disabled during review: switching modes here is
        # the destructive path that discards (and red-autosaves) the review. The
        # only intended door to Manual is the "Refine in Manual mode" button,
        # which preserves the detections. (The handoff itself manages the toggle
        # separately, so don't fight it while a handoff is live.)
        if not self._refine_handoff:
            self.mode_switch.setEnabled(not active)
            self.mode_switch.setToolTip(
                tr("Finish or exit the review to switch modes.") if active else "")
        self.auto_review_panel.setVisible(active)
        # The locked, greyed layer header is dead weight during review (the
        # raster cannot change until the review ends anyway): hide it so the
        # result + filters own the panel. Restored on review end and, for hard
        # teardowns that skip this call, by reset_auto_to_start.
        self.auto_layer_combo.setVisible(not active)
        # Hide the prompt card (Describe what to find + Library) during review:
        # the search is done, so the result + filters should own the panel.
        self.auto_prompt_card.setVisible(not active)
        self.auto_detect_row.setVisible(not active)
        self.auto_exemplar_panel.setVisible(not active)
        self.auto_detail_row.setVisible(not active and self.auto_detail_row.isVisible())
        # The pre-run confidence box is always hidden now (confidence lives in
        # the review panel below); keep it hidden here too.
        self.auto_settings_box.setVisible(False)
        # The run is paid for: the cost estimate is pre-run info that only
        # confuses the review. (The locked layer header keeps naming the raster.)
        self.auto_credit_cost_label.setVisible(
            not active and self.auto_credit_cost_label.isVisible())
        # Going back mid-review would silently desync the review from the
        # inputs; the exits (Finish, zone x / Escape, mode switch) all discard
        # or commit the review explicitly.
        if active:
            # Clear any leftover run status (tile progress / info banner) as
            # the review opens so it never lingers next to the result count.
            self.set_auto_status("idle")
            if reset_controls:
                # A fresh review is never a refine handoff: confidence is editable.
                self.set_protected_note(False)
                # Fresh review: size filters seed from the run's smart preset
                # (prompt-aware Min size floor; no preset = neutral 0). block-
                # Signals avoids debounced preview refreshes; the plugin renders
                # the preview right after.
                p = preset or {}
                for w in (self.auto_min_size_spin, self.auto_max_size_spin):
                    w.blockSignals(True)
                self.auto_min_size_spin.setValue(float(p.get("min_size_m2", 0.0)))
                self.auto_max_size_spin.setValue(0)
                for w in (self.auto_min_size_spin, self.auto_max_size_spin):
                    w.blockSignals(False)
                # Shape refine seeds from the same preset (Right angles + Fill
                # holes for buildings, Round corners for vegetation, ...), with
                # the static faithful defaults as fallback. blockSignals so this
                # seed never fires the debounced re-derive.
                for w in (self.auto_simplify_spin, self.auto_round_corners_check,
                          self.auto_expand_spin, self.auto_fill_holes_check,
                          self.auto_clean_spin, self.auto_ortho_check):
                    w.blockSignals(True)
                self.auto_simplify_spin.setValue(
                    float(p.get("simplify_px", _AUTO_REVIEW_SIMPLIFY_DEFAULT)))
                self.auto_round_corners_check.setChecked(
                    bool(p.get("smooth", _AUTO_REVIEW_SMOOTH_DEFAULT)))
                self.auto_expand_spin.setValue(
                    int(p.get("expand_px", _AUTO_REVIEW_EXPAND_DEFAULT)))
                self.auto_fill_holes_check.setChecked(
                    bool(p.get("fill_holes", _AUTO_REVIEW_FILL_HOLES_DEFAULT)))
                self.auto_clean_spin.setValue(
                    float(p.get("clean_px", _AUTO_REVIEW_CLEAN_DEFAULT)))
                self.auto_ortho_check.setChecked(
                    bool(p.get("ortho", _AUTO_REVIEW_ORTHO_DEFAULT)))
                for w in (self.auto_simplify_spin, self.auto_round_corners_check,
                          self.auto_expand_spin, self.auto_fill_holes_check,
                          self.auto_clean_spin, self.auto_ortho_check):
                    w.blockSignals(False)
                # Debug tile overlay is off by default each new review (signal-free:
                # the plugin clears the grid when the run started).
                self.auto_show_tiles_check.blockSignals(True)
                self.auto_show_tiles_check.setChecked(False)
                self.auto_show_tiles_check.blockSignals(False)
                # Seed the review confidence slider from the pre-run dial so the
                # starting filter matches what the run used (no signal: the plugin
                # renders the preview right after).
                pct = _snap_review_conf(int(round(self.auto_confidence_spin.value() * 100)))
                self.auto_review_confidence_slider.blockSignals(True)
                self.auto_review_confidence_slider.setValue(pct)
                self.auto_review_confidence_slider.blockSignals(False)
                self.auto_review_confidence_spin.blockSignals(True)
                self.auto_review_confidence_spin.setValue(pct)
                self.auto_review_confidence_spin.blockSignals(False)
            # The header + Export label are set by the plugin's first
            # update_auto_review_count call (right after this, via the review
            # preview push), so both counts and the pct are always consistent.
        self._update_auto_detect_enabled()

    def update_auto_review_count(self, visible: int, total: int, pct: int,
                                 size_bound: bool = False) -> None:
        """Update the two-line review header + the Export button label after a
        live confidence re-filter. ``visible`` = objects shown now, ``total`` =
        objects the run found, ``pct`` = current confidence cutoff. ``size_bound``
        (only meaningful when visible == 0) means the Min size filter, not
        Confidence, is what hides the objects, so the guidance names it."""
        try:
            self._auto_review_count_label.setText(
                self._format_auto_review_count(visible, total, pct, size_bound))
            self.auto_export_btn.setText(_export_btn_label(visible))
            self.auto_export_btn.setEnabled(visible > 0)
            if visible == 0:
                tip = (tr("Lower the Min size filter to show objects first.")
                       if size_bound else
                       tr("Lower Confidence to show objects first."))
            else:
                tip = ""
            self.auto_export_btn.setToolTip(tip)
        except (RuntimeError, AttributeError):
            pass

    def set_review_conf_lowered_note(self, lowered: bool, pct: int,
                                     adaptive: bool = False,
                                     tuned: bool = False) -> None:
        """Show/hide the starting-cutoff explainer note under the slider.

        Three distinct reasons, three texts: ``tuned`` means the start IS the
        default, just an object-specific one (transparency, not a lowering);
        ``adaptive`` means the run's score distribution lowered the start to
        show more of one coherent population (objects DO score above);
        otherwise nothing scored above the default."""
        if lowered:
            if tuned:
                self.auto_conf_lowered_note.setText(
                    tr("Started at {pct}% - the usual sweet spot for this "
                       "object type.").format(pct=pct))
            elif adaptive:
                self.auto_conf_lowered_note.setText(
                    tr("Started at {pct}% to fit this run's scores - raise "
                       "to tighten.").format(pct=pct))
            else:
                self.auto_conf_lowered_note.setText(
                    tr("Started at {pct}% - nothing scored above.").format(pct=pct))
        self.auto_conf_lowered_note.setVisible(bool(lowered))

    def set_auto_display_mode(self, mode: str) -> None:
        """Programmatically select a review display colour mode ('normal' /
        'outline' / 'confidence' / 'random') without emitting
        auto_display_mode_changed: the plugin stores the mode and applies the
        renderer itself, so the combo must follow silently (never desync)."""
        combo = getattr(self, "auto_display_combo", None)
        if combo is None:
            return
        idx = combo.findData(mode)
        if idx < 0:
            return
        combo.blockSignals(True)
        combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def set_display_legend(self, text: str) -> None:
        """Set the muted legend line under the View-detections-as combo (what the
        colours mean for the selected mode). No-op if the label is absent."""
        legend = getattr(self, "auto_display_legend", None)
        if legend is not None:
            legend.setText(text)

    def show_auto_exemplar_nudge(self, object_word: str) -> None:
        """Show the review exemplar nudge. Called by the plugin
        only for bottom-heavy, no-exemplar runs. The label must stay short:
        QPushButton text does not wrap, so a long tip silently elides."""
        obj = object_word or tr("object")
        self.auto_exemplar_nudge_link.setText(
            "✎  " + tr("Draw an example of one {object} to find more").format(
                object=obj))
        self.auto_exemplar_nudge_link.setToolTip(tr(
            "Runs with a drawn example return far fewer empty results. "
            "This re-runs the zone with the example draw armed (new credits)."))
        self.auto_exemplar_nudge_link.setVisible(True)

    def hide_auto_exemplar_nudge(self) -> None:
        try:
            self.auto_exemplar_nudge_link.setVisible(False)
        except (RuntimeError, AttributeError):
            pass

    def show_auto_zero_assist(self, object_word: str,
                              has_examples: bool = False) -> None:
        """Show the zero-result rescue under the status banner, exemplar first.

        Called by the plugin right after the zero-detection status is posted
        (never on the network-failure variant, where the levers do not apply).
        The example call is the hero (the proven rescue for a zero-result);
        the synonym chip only shows when the server steer table knows a
        stronger word for this prompt, so the suggestion stays server-tunable
        with no plugin release. Labels must stay short: QPushButton text does
        not wrap. ``has_examples`` switches the label to "another" when the
        run already carried drawn examples."""
        obj = (object_word or "").strip()
        if has_examples:
            label = tr("Add another example - more references detect more")
        else:
            label = tr("Draw one example - the AI finds the rest")
        self.auto_zero_example_chip.setText("✎  " + label)
        self.auto_zero_example_chip.setToolTip(tr(
            "Outline ONE example of the object on the map, then run again. "
            "Runs with a drawn example return far fewer empty results."))
        suggestion = ""
        if obj:
            try:
                ok, reason, extra = validate_prompt(obj)
                if ok and reason == "steer" and extra:
                    suggestion = str(extra)
            except Exception:  # nosec B110 -- the chip is best-effort rescue UI
                suggestion = ""
        self._auto_zero_synonym = suggestion
        if suggestion:
            self.auto_zero_synonym_chip.setText(
                "→  " + tr('Try "{word}" instead').format(word=suggestion))
        self.auto_zero_synonym_chip.setVisible(bool(suggestion))
        self.auto_zero_assist_row.setVisible(True)
        # The rescue sits under the Detect row, which is below the fold on a
        # small dock: bring it into view once the layout has settled, or the
        # whole state reads as "nothing happened".
        try:
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(0, self._scroll_auto_zero_assist_into_view)
        except (RuntimeError, AttributeError):
            pass

    def _scroll_auto_zero_assist_into_view(self) -> None:
        try:
            if self.auto_zero_assist_row.isVisible():
                self._dock_scroll_area.ensureWidgetVisible(
                    self.auto_zero_assist_row, 0, 24)
        except (RuntimeError, AttributeError):
            pass

    def hide_auto_zero_assist(self) -> None:
        try:
            self.auto_zero_assist_row.setVisible(False)
        except (RuntimeError, AttributeError):
            pass

    def set_auto_exhausted_subscribe_visible(self, visible: bool) -> None:
        """Show/hide the free-user 'Subscribe to finish this zone' link shown
        under the status when a run stops on exhausted credits (Moment C)."""
        try:
            self.auto_exhausted_subscribe_link.setVisible(bool(visible))
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
            return "{:.1f}".format(a)
        return "{:.2f}".format(a)

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

    def set_auto_credit_estimate(self, credits: int) -> None:
        # Remember the estimate so a later balance change (e.g. after a run
        # consumes credits) can re-run the gate against it, see set_auto_credits.
        self._auto_est_credits = credits
        if credits < 0:
            self.auto_credit_cost_label.setText(
                tr("Zone too large - reduce the selection area"))
            self.auto_credit_cost_label.setStyleSheet(
                f"color: {ERROR_TEXT}; font-size: 11px;")
            self.auto_credit_cost_label.setToolTip("")
            self._auto_zone_too_large = True
            self._auto_insufficient_credits = False
            self._set_auto_premium_gated(False)
        else:
            # Make the per-tile billing explicit right before Detect: the run
            # scans the zone tile by tile and spends 1 credit per tile, so the
            # count reads as the equation "N tiles = N credits" (same N on
            # purpose - that IS the lesson). The footer credit ring owns the
            # remaining balance, so no "M left" suffix here.
            if credits == 1:
                text = tr("≈ 1 tile = 1 credit")
            else:
                text = tr("≈ {n} tiles = {n} credits").format(n=credits)
            remaining = (self._auto_credits if self._auto_is_subscriber
                         else self._auto_free_left)
            # Hard credit gate: a run may never launch
            # under-funded. When the estimate exceeds the known balance, block
            # Detect and turn the cost line red with a fix-it instruction, the
            # same in-context pattern as the "Zone too large" block. This
            # replaces the old amber "will stop after N" partial-run allowance,
            # which let a run burn straight down to 0 and stop mid-zone.
            # credit_gate.insufficient owns the boundary: block only when the
            # estimate STRICTLY exceeds the balance (== is allowed), the same
            # rule as the auto_run pre-submit re-gate.
            insufficient = _credit_insufficient(credits, remaining)
            self._auto_insufficient_credits = insufficient
            # Free-plan per-run cap: the slider deliberately keeps its full
            # (Pro) travel, so past the cap the run is blocked HERE, with the
            # upgrade as the named fix. The balance gate wins when both apply
            # (an underfunded run can never launch regardless of plan).
            cap = self._auto_free_run_cap
            self._set_auto_premium_gated(
                not insufficient and not self._auto_is_subscriber and cap is not None and credits > cap)
            if insufficient:
                # A subscriber is already paying, so point them at the levers they
                # can pull now (detail/zone); only free users get the subscribe CTA.
                if self._auto_is_subscriber:
                    text = tr(
                        "Not enough credits: {n} tiles, only {left} left. "
                        "Reduce the detail or zone.").format(
                            n=credits, left=int(remaining))
                else:
                    text = tr(
                        "Not enough credits: {n} tiles, only {left} left. "
                        "Reduce the detail or zone, or subscribe.").format(
                            n=credits, left=int(remaining))
                self.auto_credit_cost_label.setStyleSheet(
                    f"color: {ERROR_TEXT}; font-weight: bold; font-size: 11px;")
            elif self._auto_premium_gated:
                # Premium taxonomy (blue + star), never the error red: this is
                # a paid-capability gate, not a failure. The cost line stays
                # the SHORT equation (it sits on the Detail header row, a long
                # sentence would widen the dock); the premium hint box under
                # the slider carries the explanation and the upgrade link.
                text = _PREMIUM_STAR + " " + (
                    tr("≈ 1 tile = 1 credit") if credits == 1
                    else tr("≈ {n} tiles = {n} credits").format(n=credits))
                self.auto_credit_cost_label.setStyleSheet(
                    f"color: {BRAND_BLUE}; font-size: 11px; font-weight: bold;")
            else:
                self.auto_credit_cost_label.setStyleSheet(
                    "color: palette(text); font-size: 11px;")
            self.auto_credit_cost_label.setText(text)
            _base_tip = tr(
                "Automatic mode scans your zone tile by tile. 1 tile = 1 credit, "
                "so this run costs about {n} credits. More detail splits the zone "
                "into more tiles, which costs more credits.").format(n=credits)
            _extra_tip = tr("1 credit ~ 0.17 km² at default detail.")
            self.auto_credit_cost_label.setToolTip(_base_tip + " " + _extra_tip)
            self._auto_zone_too_large = False
        self.auto_credit_cost_label.setVisible(True)
        self._update_auto_detect_enabled()

    def _set_auto_progress_visible(self, visible: bool) -> None:
        """Show/hide the run progress card as one unit (count row + bar + the
        conditional status line)."""
        if not visible:
            # Card gone (review / idle / error): kill the warming heartbeat so
            # no stray tick repaints a torn-down card.
            self._stop_auto_warming_anim()
        self.auto_progress_card.setVisible(visible)

    def _refresh_auto_progress_readout(self) -> None:
        """Rebuild the progress card's Row 1 (tile count + live found count) and
        the right-aligned percent from the remembered tile pair + found count."""
        current, total = getattr(self, "_auto_progress_pair", (0, 0))
        found = getattr(self, "_auto_found_count", 0)
        count_txt = tr("Tile {current}/{total}").format(
            current=max(current, 0), total=total)
        if found > 0:
            count_txt += (
                ' <span style="color: rgba(128,128,128,0.95);">· ' + tr("{n} found so far").format(n=found) + "</span>")
        self.auto_progress_count_label.setText(count_txt)
        pct = int(round(current / total * 100)) if total and current > 0 else 0
        self.auto_progress_pct_label.setText("{}%".format(max(0, min(100, pct))))

    def set_auto_tile_progress(self, current: int, total: int) -> None:
        self.auto_status_banner.setVisible(False)
        self.hide_auto_zero_assist()
        # Remembered so a cleared queue state can restore the live tile count.
        self._auto_progress_pair = (current, total)
        self._refresh_auto_progress_readout()
        self._set_auto_progress_visible(True)
        if current > 0:
            # Real progress: an honest determinate bar. The warming animation
            # (if it was running) has done its job.
            self._stop_auto_warming_anim()
            self.auto_tile_progress.setRange(0, max(1, total))
            self.auto_tile_progress.setValue(current)
        else:
            # No tile has landed yet (a cold GPU can take ~a minute to answer):
            # keep the bar ALIVE instead of a frozen 0%. _ensure_auto_warming_anim
            # switches the bar to indeterminate (Qt animates it) and runs a 1s
            # timer that evolves the label. Row-3 copy is chosen by the shared
            # renderer below.
            self._ensure_auto_warming_anim()
        self._render_auto_wait_label()
        self._auto_progress_ratio = (current / total) if total else 0.0

    def set_auto_queue_state(self, position: int, depth: int, eta_s: int) -> None:
        """Honest launch-spike feedback on the progress bar's label. The server
        answers a saturated moment with a real place in its fair queue; showing
        that place (and watching it move) is what keeps a user from reading the
        wait as a hang. position >= 1 = known place in line; -1 = busy but no
        position known (older server / platform rejection / cold start);
        (0, 0, 0) = flowing again, restore the tile count. The bar itself is
        never animated on a timer: only real state changes repaint it."""
        if not self.auto_progress_card.isVisible():
            return
        self._auto_queue_position = position
        self._auto_queue_eta = eta_s if eta_s and eta_s > 0 else 0
        if position == 0 and depth == 0:
            # Flowing again: restore the live tile count (which re-arms the
            # warming animation if we are still at zero tiles).
            current, total = getattr(self, "_auto_progress_pair", (0, 0))
            self.set_auto_tile_progress(current, total)
            return
        # A busy/queued answer means we are still pre-first-tile: keep the bar
        # animated and let the shared renderer pick the right copy (real place
        # in line vs generic "waking up").
        self._ensure_auto_warming_anim()
        self._render_auto_wait_label()

    # ------------------------------------------------------------------
    # Pre-first-tile "waking up" animation (single timer-driven renderer)
    # ------------------------------------------------------------------
    def _ensure_auto_warming_anim(self) -> None:
        """Start (or keep) the pre-first-tile feedback: an indeterminate
        (Qt-animated) bar plus a 1s timer that evolves the label. Idempotent -
        safe to call from every progress/queue update."""
        if self._auto_warming_since is None:
            self._auto_warming_since = time.monotonic()
        # An indeterminate range is Qt's built-in animated busy bar: it always
        # moves, so the wait can never read as frozen. (This deliberately
        # overrides the old "never animate on a timer" rule, per the request
        # for constant motion during cold starts.)
        self.auto_tile_progress.setRange(0, 0)
        if self._auto_warmup_timer is None:
            from qgis.PyQt.QtCore import QTimer
            self._auto_warmup_timer = QTimer(self)
            self._auto_warmup_timer.setInterval(1000)
            self._auto_warmup_timer.timeout.connect(self._on_auto_warming_tick)
        if not self._auto_warmup_timer.isActive():
            self._auto_warmup_timer.start()

    def _stop_auto_warming_anim(self) -> None:
        """End the pre-first-tile animation once tiles flow or the run ends."""
        self._auto_warming_since = None
        if self._auto_warmup_timer is not None and self._auto_warmup_timer.isActive():
            self._auto_warmup_timer.stop()

    def _on_auto_warming_tick(self) -> None:
        """1s heartbeat while no tile has landed: re-assert the animated bar and
        recount the label so the wait is visibly progressing."""
        if not getattr(self, "_auto_run_active", False):
            self._stop_auto_warming_anim()
            return
        current, _total = getattr(self, "_auto_progress_pair", (0, 0))
        if current > 0:
            self._stop_auto_warming_anim()
            return
        if self.auto_tile_progress.maximum() != 0:
            self.auto_tile_progress.setRange(0, 0)
        self._render_auto_wait_label()

    def _render_auto_wait_label(self) -> None:
        """Single source of truth for the Row-3 status line. Priority: the
        cancel note, then a real place in the server queue, then the generic
        'waking up' copy. Hidden once tiles flow (unless cancelling)."""
        current, _total = getattr(self, "_auto_progress_pair", (0, 0))
        if current > 0 and not self._auto_cancelling:
            self.auto_progress_label.setVisible(False)
            return
        if self._auto_cancelling:
            text = tr("Stopping - keeping the tiles already found...")
        else:
            pos = getattr(self, "_auto_queue_position", 0)
            eta_s = getattr(self, "_auto_queue_eta", 0)
            if pos == 1:
                text = tr("You're next · starting now...")
            elif pos > 1:
                if 0 < eta_s < 10:
                    text = tr("Spot reserved · starting in a few seconds...")
                else:
                    eta = self._friendly_eta(eta_s)
                    text = (tr("Spot reserved · starting in ~{eta}").format(eta=eta)
                            if eta else tr("Spot reserved · starting soon..."))
            else:
                text = self._warming_message()
        self.auto_progress_label.setText(text)
        self.auto_progress_label.setVisible(True)

    def _warming_message(self) -> str:
        """Evolving pre-first-tile copy with a live elapsed count, so the wait
        is visibly moving even before the first tile answers."""
        since = self._auto_warming_since
        elapsed = int(time.monotonic() - since) if since is not None else 0
        if elapsed < 6:
            return tr("Sending to the AI...")
        if elapsed < 22:
            return tr("Waking up the AI... {n}s").format(n=elapsed)
        return tr("The AI is starting up, almost there... {n}s").format(n=elapsed)

    @staticmethod
    def _friendly_eta(eta_s: int) -> str:
        """Rounded, human wait estimate ('' when unknown or tiny). Never
        false-precise: seconds snap to 5s steps, past a minute whole minutes."""
        if eta_s is None or eta_s < 10:
            return ""
        if eta_s < 60:
            return tr("{s} seconds").format(s=int(round(eta_s / 5.0) * 5))
        return tr("{m} min").format(m=int((eta_s + 59) // 60))

    def set_auto_status(self, kind: str, message: str = "") -> None:
        """Single surface for run feedback. kind: 'idle', 'progress', 'info',
        'error'. Exactly one of progress bar / status banner is visible at a
        time; 'idle' hides both and clears any stale text."""
        # Any new status replaces the context the zero-result chips belonged
        # to; the plugin re-shows them explicitly after the zero status.
        self.hide_auto_zero_assist()
        if kind == "progress":
            self.auto_status_banner.setVisible(False)
            return  # set_auto_tile_progress drives the bar itself
        self._set_auto_progress_visible(False)
        if kind == "idle" or not message:
            self.auto_status_banner.setVisible(False)
            self.auto_status_banner.setText("")
            return
        if kind == "error":
            self.auto_status_banner.setStyleSheet(_msg_label_qss("error"))
        else:
            self.auto_status_banner.setStyleSheet(_msg_label_qss("info"))
        self.auto_status_banner.setText(
            _msg_text("error" if kind == "error" else "info", message))
        self.auto_status_banner.setVisible(True)
