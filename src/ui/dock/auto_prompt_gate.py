"""Commit-time gates in front of Detect: the prompt guard rail, the
prompt-plus-example intercept, and the off-thread server name lookup.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QTimer

from ...core.i18n import tr
from ...core.telemetry import slot_guard
from .prompt_guard import is_known_object, validate_prompt

# Reasons that mean "run a cleaner phrase instead of the typed text" (swap the
# box, show a quiet note). The commit-time guard returns "translated" / "plural"
# / "alias" with the token as its suggestion; "server_rewrite" arrives from the
# async run plan (a server-side language-model rewrite) and rides the SAME
# swap-and-tell channel (apply_prompt_swap). The dock treats them identically.
_SILENT_SWAP_REASONS = frozenset(
    {"translated", "plural", "alias", "server_rewrite"})

# Last resort for the commit-time prompt lookup: if the task never reports back
# (no scheduler slot, a torn-down task manager), answer "no token" after this
# and run the word as typed. Comfortably above the request's own timeout, so a
# normal slow reply always wins the race.
_PROMPT_LOOKUP_TIMEOUT_MS = 12_000


class DockAutoPromptGateMixin:
    """Commit-time gates in front of Detect: the prompt guard rail, the
    prompt-plus-example intercept, and the off-thread server name lookup."""

    def confirm_prompt_for_detect(self) -> bool:
        """Commit-time guard rail: called by the plugin when a detection is
        requested (Detect click or Enter). A clean prompt (or an example-only
        run with an empty prompt) passes; anything off the rails blocks the
        run and shows the guidance right under the prompt box, with focus back
        in it so the fix is one keystroke away.

        False also means "not yet": a word that needs the server lookup defers
        this commit while the lookup runs off the GUI thread, and the answer
        re-fires the detection request (see _resolve_prompt_via_server). The
        multi-object branch defers the same way, for a different reason: it
        DROPS part of what was typed, so the user confirms the reduced prompt
        with a second click instead of paying for it on the first."""
        text = self.auto_prompt_input.text()
        pending = getattr(self, "_prompt_lookup_key_pending", None)
        if pending is not None:
            if pending == self._prompt_lookup_key(text):
                # A lookup for this exact word is already running. Repeat the
                # note instead of nesting a second one: the answer re-fires the
                # detection on its own, so the click is not lost.
                self._set_prompt_info(self._prompt_lookup_note(), tip=True)
                return False
            # The word moved on since the lookup started: drop it and judge the
            # text that is in the box now.
            self._abandon_prompt_lookup()
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
            token, waiting = self._resolve_prompt_via_server(text)
            if waiting:
                return False
            if token:
                reason, suggestion = "translated", token
        elif not ok and reason == "language":
            # Foreign word beyond the offline lexicon: try the same server
            # lookup before giving up and lecturing about English.
            token, waiting = self._resolve_prompt_via_server(text)
            if waiting:
                return False
            if token:
                ok, reason, suggestion = True, "translated", token
        self._auto_prompt_valid = ok
        if ok:
            if reason in _SILENT_SWAP_REASONS and suggestion:
                # The user typed the object in their own language (translated),
                # as a plural, or under a catalogue alias: run the English token
                # for them and say so in a quiet note, not an error. The run
                # proceeds untouched (shared swap-and-tell channel).
                self.apply_prompt_swap(suggestion, reason)
            elif reason == "multi_first" and suggestion:
                # Several objects in one box ("buildings and roads"): the
                # model grounds ONE concept per run, so the run narrows to the
                # FIRST object instead of being refused, with a quiet hint to
                # run the rest separately. setText first - its textChanged
                # clears the info line, which we then set (same order as the
                # translated case).
                typed = text.strip()
                narrowed = suggestion != typed
                if narrowed:
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
                if narrowed:
                    # Unlike the translated / plural / alias swaps, which name
                    # the SAME object, this one drops part of the request. The
                    # connector split is only a guess at where a name ends
                    # ("bed and breakfast" narrows to "bed"), so the reduced
                    # prompt is shown and the click withheld. The box now holds
                    # a single object, so the next click validates clean and
                    # launches. Nothing the user did not read gets billed.
                    return False
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

    def _refresh_meta_escape(self, show: bool, has_text: bool,
                             positives: int = 0) -> None:
        """Show/hide the small "Detect with text/examples only" link under the
        (grey) Detect button. Called from _update_auto_detect_enabled with the
        already-computed gate outcome: the link is visible exactly when only
        the missing half of the prompt-plus-example setup blocks the button,
        so the escape stays one visible click away without a dead click.
        The blue hint line above it belongs to the Enter-path intercept only
        (confirm_meta_for_detect); this never shows it."""
        try:
            if show and not getattr(self, "_meta_escape_seen", False):
                # Once per episode: how often users SEE the escape state is
                # the launch question for the prompt-plus-example default.
                # Same event family as the commit intercept, -1.0 sentinel.
                self._meta_escape_seen = True
                try:
                    from ...core import telemetry
                    from ...core import telemetry_events as ev
                    telemetry.track(ev.EXEMPLAR_NUDGE_SHOWN, {
                        "run_id": "",
                        "object_class": (self.auto_prompt_input.text().strip()
                                         if has_text else ""),
                        "median_score": -1.0,
                    })
                except Exception:
                    pass  # nosec B110
            if show:
                if has_text:
                    label = tr("Detect with text only")
                elif positives == 1:
                    label = tr("Detect with example only")
                else:
                    label = tr("Detect with examples only")
                self.auto_detect_anyway_btn.setText(label)
            else:
                self.auto_meta_hint.setVisible(False)
            self.auto_detect_anyway_btn.setVisible(show)
            self.auto_meta_intercept.setVisible(
                show or self.auto_meta_hint.isVisible())
        except (RuntimeError, AttributeError):
            pass

    def confirm_meta_for_detect(self) -> bool:
        """Commit-time guard for the prompt-plus-example default. The green
        Detect only enables on the full setup, so this mostly covers the
        OTHER entry points (Enter in the prompt box, the escape link): a
        half-setup commit without the override shows a blue line naming the
        missing half (the link below it is already visible) and blocks; the
        escape link sets the override and re-emits, which passes here."""
        if not self._EXEMPLARS_ENABLED:
            return True
        # One-shot pending override from the escape link. Consumed HERE, not
        # at click time: the prompt guard runs first and its translated /
        # multi-object branches call setText, whose textChanged fires
        # _reset_meta_intercept synchronously - a plain flag set at click
        # time would be wiped before this guard ever saw it, silently
        # swallowing the escape click for exactly the localized prompts.
        if getattr(self, "_auto_meta_override_pending", False):
            self._auto_meta_override_pending = False
            self._auto_meta_override = True
        if getattr(self, "_auto_meta_override", False):
            return True
        from ...core.detect_gate import meta_satisfied
        has_text = bool(self.auto_prompt_input.text().strip())
        positives = getattr(self, "_auto_positive_exemplars", 0)
        if meta_satisfied(has_text, positives):
            return True
        if has_text:
            # Name the typed object so the line reads as advice about THIS
            # run, and point at the exact gesture (step 2's map draw) that
            # completes it.
            word = self.auto_prompt_input.text().strip()
            hint = tr('Almost there: in step 2, outline one "{word}" on the '
                      'map so the AI sees what yours look like.').format(
                word=word)
        else:
            hint = tr("Almost there: in step 1, name the object your "
                      "examples show - words plus examples detect best.")
        try:
            self.auto_meta_hint.setText(hint)
            self.auto_meta_hint.setVisible(True)
            self.auto_meta_intercept.setVisible(True)
        except (RuntimeError, AttributeError):
            return True
        try:
            from ...core import telemetry
            from ...core import telemetry_events as ev
            # Pre-run intercept, reusing the exemplar-nudge event; the -1.0
            # median is the sentinel separating it from the in-review nudge
            # (whose median is always a real 0..0.35 score).
            telemetry.track(ev.EXEMPLAR_NUDGE_SHOWN, {
                "run_id": "",
                "object_class": self.auto_prompt_input.text().strip(),
                "median_score": -1.0,
            })
        except Exception:
            pass  # nosec B110
        return False

    def _on_auto_detect_anyway_clicked(self) -> None:
        """The explicit escape from the prompt-plus-example default: run once
        with the current half-setup. Sets the PENDING one-shot flag (consumed
        by confirm_meta_for_detect) rather than the live override: the prompt
        guard's translated/multi-object branches edit the prompt box on the
        way, and that edit resets the live override synchronously."""
        self._auto_meta_override_pending = True
        try:
            self.auto_meta_intercept.setVisible(False)
        except (RuntimeError, AttributeError):
            pass
        try:
            from ...core import telemetry
            from ...core import telemetry_events as ev
            telemetry.track(ev.EXEMPLAR_NUDGE_CLICKED, {
                "run_id": "",
                "object_class": self.auto_prompt_input.text().strip(),
                "median_score": -1.0,
            })
        except Exception:
            pass  # nosec B110
        self.auto_detect_requested.emit()

    def _reset_meta_intercept(self) -> None:
        """Retire the intercept message + escape link and drop the override.
        Runs whenever the setup changes (prompt edit, example added/removed) or
        the flow resets; _update_auto_detect_enabled then re-shows the link
        for the new state when it still applies."""
        self._auto_meta_override = False
        self._meta_escape_seen = False
        try:
            self.auto_meta_hint.setVisible(False)
            self.auto_detect_anyway_btn.setVisible(False)
            self.auto_meta_intercept.setVisible(False)
        except (RuntimeError, AttributeError):
            pass

    @staticmethod
    def _prompt_lookup_key(text: str) -> str:
        """Identity of a prompt for the answer cache. Only ever compared with
        another key from this same function."""
        return " ".join((text or "").split()).lower()

    @staticmethod
    def _prompt_lookup_note() -> str:
        return tr("Checking the object name...")

    def _prompt_lookup_answers(self) -> dict:
        """Per-session word -> server answer (None = no usable token). Keeps a
        second Detect on the same word instant, answered or not."""
        cache = getattr(self, "_prompt_lookup_cache", None)
        if cache is None:
            cache = {}
            self._prompt_lookup_cache = cache
        return cache

    def _resolve_prompt_via_server(self, text: str) -> tuple[str | None, bool]:
        """Commit-time-only server translation for words the offline lexicon
        cannot resolve. Returns ``(token, waiting)``.

        The lookup runs OFF the GUI thread: Detect is the most-clicked button
        in the product and a slow link must never freeze QGIS on it. The first
        Detect on a given word returns ``(None, True)``: the caller stops this
        commit, Detect greys and a note says the name is being checked. When
        the answer lands (or the request fails, or the watchdog gives up) the
        detection request is re-emitted, this returns the cached answer, and
        the run starts without a second click. Answers are cached per word, so
        every later Detect on it is decided synchronously.

        The token is re-vetted by the guard on the way out, so a bad server
        answer can never reach the model."""
        key = self._prompt_lookup_key(text)
        if not key:
            return None, False
        cache = self._prompt_lookup_answers()
        if key not in cache:
            return None, self._start_prompt_lookup(text, key)
        token = cache.get(key)
        if not token:
            return None, False
        ok, reason, _suggestion = validate_prompt(token)
        return (token if ok and reason is None else None), False

    def _start_prompt_lookup(self, text: str, key: str) -> bool:
        """Queue the off-thread lookup for ``key``. True when the caller must
        wait for the answer, False when the lookup could not be started at all
        (cached as "no token", so the run goes ahead with the word as typed and
        the click never dies on a greyed button)."""
        if getattr(self, "_prompt_lookup_key_pending", None) == key:
            return True
        self._abandon_prompt_lookup()
        try:
            from qgis.core import QgsApplication

            from ...api.prompt_translation import resolve_english_prompt
            from ...workers.generic_request_task import GenericRequestTask
        except Exception:  # noqa: BLE001 -- translation is best-effort
            self._prompt_lookup_answers()[key] = None
            return False
        generation = int(getattr(self, "_prompt_lookup_generation", 0)) + 1
        self._prompt_lookup_generation = generation
        self._prompt_lookup_key_pending = key
        try:
            task = GenericRequestTask(
                tr("Checking the object name"),
                lambda word=text: {"token": resolve_english_prompt(word)},
                hidden=True,
            )
            task.succeeded.connect(
                lambda answer, g=generation: self._on_prompt_lookup_done(g, answer))
            task.failed.connect(
                lambda *_a, g=generation: self._on_prompt_lookup_done(g, None))
            self._prompt_lookup_task = task
            QgsApplication.taskManager().addTask(task)
        except Exception:  # noqa: BLE001 -- translation is best-effort
            self._prompt_lookup_task = None
            self._prompt_lookup_key_pending = None
            self._prompt_lookup_answers()[key] = None
            return False
        try:
            QTimer.singleShot(
                _PROMPT_LOOKUP_TIMEOUT_MS,
                lambda g=generation: self._on_prompt_lookup_done(g, None))
        except (RuntimeError, AttributeError):
            pass
        self._set_prompt_info(self._prompt_lookup_note(), tip=True)
        self._set_prompt_lookup_busy(True)
        return True

    @slot_guard(stage="segment", user_message=tr(
        "Something went wrong starting the detection. Please try again."))
    def _on_prompt_lookup_done(self, generation: int, answer: object) -> None:
        """Main thread: file the answer and resume the Detect the user already
        clicked. A late answer from a superseded lookup is dropped on the
        generation check, so an abandoned lookup can never start a run.

        Qt direct connections run a slot inline, so a failure anywhere down
        the whole Detect chain (auto_flow._on_auto_detect_requested and
        everything it calls) would otherwise unwind to the emit below.
        slot_guard is the one boundary that reports it instead of a silent
        dead click; the dock being torn down mid-lookup (flow left, plugin
        unloaded) is the one case that is expected and stays silent, via the
        sip check right before the emit."""
        if generation != int(getattr(self, "_prompt_lookup_generation", 0)):
            return
        key = getattr(self, "_prompt_lookup_key_pending", None)
        if key is None:
            return
        self._prompt_lookup_key_pending = None
        self._prompt_lookup_task = None
        token = answer.get("token") if isinstance(answer, dict) else None
        self._prompt_lookup_answers()[key] = token if isinstance(token, str) else None
        self._set_prompt_lookup_busy(False)
        self._set_prompt_info()
        if self._prompt_lookup_key(self.auto_prompt_input.text()) != key:
            return  # the box moved on: the answer stays cached, no run
        if self._auto_run_active or self._auto_review_active:
            return
        try:
            from qgis.PyQt import sip
            # "is True": only a real, unambiguous sip answer short-circuits
            # the emit. Anything else (not sip-wrapped, a test double, a
            # mocked sip) falls through and runs normally.
            if sip.isdeleted(self) is True:
                return  # torn down between the checks above and here
        except (ImportError, TypeError):
            pass  # not sip-wrapped (e.g. a test double): nothing to check
        self.auto_detect_requested.emit()

    def _abandon_prompt_lookup(self) -> None:
        """Drop any in-flight commit-time lookup: bump the generation so a late
        answer is ignored, cancel the task, and hand Detect back. Called when
        the prompt changes under the lookup and on any flow reset, so cancelling
        or leaving the flow never leaves a greyed button behind."""
        self._prompt_lookup_generation = int(
            getattr(self, "_prompt_lookup_generation", 0)) + 1
        task = getattr(self, "_prompt_lookup_task", None)
        self._prompt_lookup_task = None
        was_pending = getattr(self, "_prompt_lookup_key_pending", None) is not None
        self._prompt_lookup_key_pending = None
        if task is not None:
            try:
                task.cancel()
            except (RuntimeError, AttributeError):
                pass
        if was_pending:
            self._set_prompt_lookup_busy(False)

    def _set_prompt_lookup_busy(self, busy: bool) -> None:
        """Grey Detect while the lookup runs, then hand it back through the
        shared gate so it ends in the state the rest of the setup asks for
        (never force-enabled)."""
        try:
            if busy:
                self.auto_detect_btn.setEnabled(False)
            else:
                self._update_auto_detect_enabled()
        except (RuntimeError, AttributeError):
            pass
