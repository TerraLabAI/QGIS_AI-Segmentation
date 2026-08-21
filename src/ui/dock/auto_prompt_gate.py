"""Commit-time gates in front of Detect: the prompt guard rail and the
off-thread server name lookup.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QTimer

from ...core.i18n import tr
from ...core.telemetry_errors import slot_guard
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
# normal slow reply always wins the race. Shipped value and the fallback the
# getter below returns.
_PROMPT_LOOKUP_TIMEOUT_MS = 12_000

# How far the served timeout may travel. Under a second no reply could land;
# past half a minute the click has been held long enough to read as a freeze.
_MIN_LOOKUP_TIMEOUT_MS = 1_000
_MAX_LOOKUP_TIMEOUT_MS = 30_000


def prompt_server_lookup_enabled() -> bool:
    """Whether an unknown word may be sent to the server on the first Detect.

    Off, the lookup never runs: the typed word goes to the model as it stands,
    Detect stays synchronous, and the click costs no network. Fail-open, so an
    absent or damaged configuration looks words up exactly as shipped.
    """
    try:
        from ...core.server_dials import feature_enabled

        return feature_enabled("prompt_server_lookup")
    except Exception:  # noqa: BLE001 -- a kill switch is best-effort
        return True


def prompt_lookup_timeout_ms() -> int:
    """How long the lookup may hold the Detect click, in milliseconds."""
    try:
        from ...core.server_dials import dial_in_range

        return int(dial_in_range("prompt.lookup_timeout_ms",
                                 _PROMPT_LOOKUP_TIMEOUT_MS,
                                 _MIN_LOOKUP_TIMEOUT_MS, _MAX_LOOKUP_TIMEOUT_MS))
    except Exception:  # noqa: BLE001 -- a timeout is best-effort
        return _PROMPT_LOOKUP_TIMEOUT_MS


class DockAutoPromptGateMixin:
    """Commit-time gates in front of Detect: the prompt guard rail and the
    off-thread server name lookup."""

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
                    from ...core import telemetry_run_events
                    # prompt = the 1-2 word object that actually runs;
                    # "multi_first" marks the guided-multi case for analytics.
                    telemetry_run_events.track_auto_prompt_steered(
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
            elif reason == "steer":
                # A real word that names the object badly from above: a part of
                # it ("wall" -> building), a cover the model grounds one crown
                # at a time ("forest" -> tree), or something an aerial image
                # never shows (an indoor room, a person, a shadow: empty
                # suggestion). The nudge names the better word and the run goes
                # ahead on what was typed, because the user may mean exactly
                # that word.
                self._show_prompt_steer_nudge(text, suggestion)
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
            from ...core import telemetry_session_events
            telemetry_session_events.track_detect_blocked(
                reason="prompt_{}".format(reason or "invalid"))
        except Exception:
            pass  # nosec B110
        return False

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
        if not prompt_server_lookup_enabled():
            # Switched off fleet-wide: no network on the click, and the answer
            # is the one that runs the word as typed.
            return None, False
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
                prompt_lookup_timeout_ms(),
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
