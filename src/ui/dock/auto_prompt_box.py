"""Automatic mode prompt box: what the user types, what commits, and the
one message line under it (guidance, swap note, steer and boost nudges).

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import Qt

from ...core.i18n import tr
from .guidance import (
    BLUE_TINT,
    HINT_PROMPT_EXAMPLES_DRIVE,
    HINT_PROMPT_EXEMPLAR_BOOST,
    HINT_PROMPT_ONE_OBJECT_PER_RUN,
    HINT_PROMPT_RUN_PLAN,
    HINT_PROMPT_SILENT_SWAP,
    HINT_PROMPT_TREE_OR_FOREST,
    HINT_PROMPT_UNKNOWN_OBJECT,
    NEUTRAL_TINT,
)
from .prompt_guard import english_token_for, is_known_object, validate_prompt
from .styles import (
    BRAND_BLUE,
    _msg_label_qss,
)


class DockAutoPromptBoxMixin:
    """Automatic mode prompt box: what the user types, what commits, and the
    one message line under it (guidance, swap note, steer and boost nudges)."""

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
        # A commit-time lookup belongs to the word that started it. Editing the
        # box retires it (and gives Detect back) instead of leaving the button
        # greyed on an answer that no longer applies.
        if getattr(self, "_prompt_lookup_key_pending", None) is not None:
            self._abandon_prompt_lookup()
        # Editing the prompt makes the zero-result chips stale (their labels
        # quote the old word); the synonym prefill lands here too and cleans
        # itself up the same way.
        self.hide_auto_zero_assist()
        # A prompt edit changes the setup: the meta intercept (and its one-run
        # override) must be re-earned against the new state.
        self._reset_meta_intercept()
        self._apply_prompt_hint_on_edit()
        self._update_auto_detect_enabled()
        # Re-seed the object-aware detail default once the typed object settles.
        self._auto_prompt_debounce_timer.start(500)

    def _on_auto_search_return_pressed(self) -> None:
        # Enter in the prompt box routes through the plugin's single Enter
        # dispatcher (_route_enter), which launches the detection once
        # everything required (layer locked, zone drawn, object) is in place.
        self.auto_enter_pressed.emit()

    def _on_auto_prompt_editing_finished(self) -> None:
        """Enter or focus-out: the prompt is explicitly settled. Flush any
        pending debounce and commit now, fragments included (blur is intent)."""
        try:
            self._auto_prompt_debounce_timer.stop()
        except (RuntimeError, AttributeError):
            pass
        self._emit_auto_prompt_committed(force=True)

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
        # Canopy prompts swap the example tip for its shadow-exclude variant
        # (tall-tree shadows are the classic false positive on tree cover).
        try:
            from ...core.detection_policy import prompt_suggests_canopy
            self._auto_prompt_canopy = bool(text) and prompt_suggests_canopy(text)
            self._refresh_auto_exemplar_explainer(
                armed=self._auto_exemplar_line_busy())
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        if text:
            # Quiet advisories the moment the prompt settles, BEFORE Detect is
            # clicked. The amber guard block stays commit-only; nothing here
            # blocks. NO word-is-better nudge: which word wins depends on what
            # the user is after ("forest" for one continuous block, "tree" for
            # individual crowns), so the line SAYS what each does and lets the
            # user pick, instead of ranking one above the other.
            try:
                ok, reason, suggestion = validate_prompt(text)
                if ok and text.strip().lower() in ("tree", "trees"):
                    # Count-vs-map heads-up, minimal on purpose: dense forest
                    # reads better as one continuous block.
                    self._set_prompt_info(
                        tr('Dense forest? "Forest" takes it as one block; '
                           '"Tree" picks individual trees.'), tip=True,
                        hint_id=HINT_PROMPT_TREE_OR_FOREST)
                elif ok and reason == "multi_first" and suggestion:
                    # Early heads-up for a several-objects prompt: the box is
                    # left as typed (the swap happens at Detect), the hint
                    # just says what will run so no credit surprises.
                    self._set_prompt_info(
                        tr('One object per run - Detect will run "{first}" '
                           'first.').format(first=suggestion), tip=True,
                        hint_id=HINT_PROMPT_ONE_OBJECT_PER_RUN)
                elif ok and reason is None:
                    # Clean prompt: if it is a curated object that text alone
                    # rarely finds and no example is drawn, nudge (once) toward
                    # drawing one. Non-blocking, and only when nothing more
                    # specific claimed the line above.
                    self._maybe_show_exemplar_boost_nudge(text)
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

    def _set_prompt_info(self, text: str | None = None, error: bool = False,
                         info: bool = False, tip: bool = False,
                         kind: str | None = None,
                         hint_id: str | None = None) -> bool:
        """Guard-rail message under the prompt. Hidden when the prompt is empty
        or valid; an amber callout when the committed prompt is off the rails;
        a quiet neutral note (``info=True``) for the silent-translation case
        ('"piscine" will run as "swimming pool"'); a blue information callout
        with a leading info bubble (``tip=True``) for the steer nudge that
        suggests a better prompt. ``error`` is kept for call-site
        compatibility; plain non-info/non-tip text shows as the amber callout.

        ``hint_id`` makes the message closable: it renders in the dismissible
        card (the same small x as every other tip, remembered in QSettings,
        back with the Account Settings guidance reset) instead of the plain
        line, and a tip the user already closed is not shown at all. Leave it
        None for what has to be read: the amber guard, the commit-time notes
        that explain a click the flow withheld, and the lookup status.

        ``kind`` records which message owns the line so the async run-plan hint
        can respect precedence (see show_auto_prompt_hint): "error" (blocking
        guard) > "swap" (will-run-as note) > "boost" (exemplar nudge) > "hint"
        (plan hint) > "tip" (generic advisory). When not given it is derived
        from the styling (tip/info -> "tip", else "error"); an empty message,
        and a tip the user closed, clear it to None.

        Returns True when the line now shows something."""
        # One line, two widgets: whichever is written hides the other, so a tip
        # and the guard can never stack.
        self.auto_prompt_tip.setVisible(False)
        if not text:
            self._prompt_info_kind = None
            self.auto_prompt_info.setText("")
            self.auto_prompt_info.setVisible(False)
            return False
        if hint_id:
            self.auto_prompt_info.setVisible(False)
            # A quiet fact keeps the neutral card and no lightbulb; guidance
            # wears the blue the taxonomy gives tips.
            quiet = info and not tip
            if not self.auto_prompt_tip.set_hint(
                    hint_id, text,
                    tint=NEUTRAL_TINT if quiet else BLUE_TINT,
                    show_glyph=not quiet):
                self._prompt_info_kind = None
                return False
            self._prompt_info_kind = kind or "tip"
            self.auto_prompt_tip.setVisible(True)
            return True
        self._prompt_info_kind = kind or ("tip" if (tip or info) else "error")
        if tip:
            # Blue information callout: a leading info bubble (rich text so only
            # the glyph is blue) marks it clearly as a helpful tip, not an error.
            self.auto_prompt_info.setTextFormat(Qt.TextFormat.RichText)
            self.auto_prompt_info.setText(
                f'<span style="color:{BRAND_BLUE}; font-weight:bold;">&#9432;</span>'
                f'&nbsp;&nbsp;{text}')
            self.auto_prompt_info.setStyleSheet(_msg_label_qss("info"))
            self.auto_prompt_info.setVisible(True)
            return True
        self.auto_prompt_info.setTextFormat(Qt.TextFormat.PlainText)
        self.auto_prompt_info.setText(text)
        if info:
            self.auto_prompt_info.setStyleSheet(_msg_label_qss("neutral"))
        else:
            self.auto_prompt_info.setStyleSheet(_msg_label_qss("warning"))
        self.auto_prompt_info.setVisible(True)
        return True

    def _maybe_show_exemplar_boost_nudge(self, token: str) -> bool:
        """While no positive example is drawn, show ONE quiet heads-up that a
        drawn example finds far more, for two kinds of clean prompt: a curated
        object that text alone rarely finds (server exemplar_boost list), or an
        object name outside the known-object library entirely (the cloud model
        probably cannot ground the word, an example shows it what to find).
        Returns True when it was shown. Non-blocking and yields to any other
        message (the caller only reaches here for a clean prompt).

        Both checks are exception-safe and fail open to no nudge. No-op when
        the example feature is off, an example already exists, or a run/review
        is in flight."""
        token = (token or "").strip()
        if not token or not self._EXEMPLARS_ENABLED:
            return False
        if self._auto_run_active or self._auto_review_active:
            return False
        if getattr(self, "_auto_positive_exemplars", 0) > 0:
            return False
        try:
            from .prompt_guard import is_exemplar_boost_prompt, is_known_object
            if is_exemplar_boost_prompt(token):
                kind, hint_id = "exemplar_boost", HINT_PROMPT_EXEMPLAR_BOOST
                message = tr(
                    '"{obj}" is often missed from text alone. Draw one '
                    'example on the map to find far more.').format(obj=token)
            elif not is_known_object(token):
                # Unknown to the whole library (catalogue + object words +
                # lexicon): the commit path may still translate it server-side,
                # and until then an example is the reliable way to guide the
                # model toward an uncommon object.
                kind, hint_id = "unknown_object", HINT_PROMPT_UNKNOWN_OBJECT
                message = tr(
                    '"{obj}" is not an object the AI knows well. Drawing one '
                    'example on the map shows it what to find.').format(
                        obj=token)
            else:
                return False
        except Exception:  # noqa: BLE001 -- policy is best-effort; fail open
            return False
        # The card renders plain text, so the typed word needs no escaping.
        if not self._set_prompt_info(
                message, tip=True, kind="boost", hint_id=hint_id):
            return False
        # Fire the shown-telemetry once per distinct committed prompt.
        key = (kind, token.lower())
        if getattr(self, "_boost_nudge_tracked", None) != key:
            self._boost_nudge_tracked = key
            try:
                from ...core import telemetry
                telemetry.track_auto_prompt_hint_shown(
                    kind=kind, prompt=token)
            except Exception:
                pass  # nosec B110
        return True

    def show_auto_prompt_hint(self, hint: str) -> bool:
        """Display a server-supplied run-plan hint under the prompt box, but only
        when no higher-priority message owns the line (precedence: guard error >
        swap > exemplar-boost nudge > plan hint). Returns True when it was shown.

        The hint arrives from the server in English and is shown verbatim (never
        translated), in the plain-text tip card, so server copy can never inject
        markup. Empty/blank hints are ignored, and so is a hint the user has
        already closed (False, so the caller logs no shown-event)."""
        text = (hint or "").strip()
        if not text:
            return False
        if getattr(self, "_prompt_info_kind", None) not in (None, "tip"):
            return False
        return self._set_prompt_info(
            text, tip=True, kind="hint", hint_id=HINT_PROMPT_RUN_PLAN)

    def apply_prompt_swap(self, token: str, reason: str) -> bool:
        """Swap the prompt box to a cleaner English run ``token`` and show the
        quiet '"typed" will run as "token"' note, then fire the rewrite
        telemetry with ``reason`` as its kind. Returns True when a swap was
        made (a no-op when the token is empty or already what the box shows).

        The single swap-and-tell channel, shared by the commit-time guard
        (``translated`` / ``plural`` / ``alias``) and the async server rewrite
        (``server_rewrite``); ``reason`` is one of _SILENT_SWAP_REASONS. The
        phrase is applied verbatim (the server preserves any attributes). Order
        matters: setText fires textChanged FIRST, which clears the info line and
        restarts the prompt-commit debounce that re-seeds the detail for the new
        token; the note is (re)set right after, so the swap and its re-seed use
        the one existing path."""
        token = (token or "").strip()
        typed = self.auto_prompt_input.text().strip()
        if not token or not typed or token.lower() == typed.lower():
            return False
        self.auto_prompt_input.setText(token)
        self._set_prompt_info(
            tr('"{word}" will run as "{token}".').format(
                word=typed, token=token), info=True, kind="swap",
            hint_id=HINT_PROMPT_SILENT_SWAP)
        try:
            from ...core import telemetry
            telemetry.track_auto_prompt_rewritten(kind=reason, prompt=token)
        except Exception:  # noqa: BLE001 -- telemetry is best-effort
            pass  # nosec B110
        return True

    def show_auto_prompt_decline(self, reason: str) -> bool:
        """Show the server's decline reason as a non-blocking guard nudge under
        the prompt box, under the SAME precedence as a plan hint (only when no
        higher-priority message owns the line). Never blocks Detect. Returns
        True when it was shown.

        The reason is server-authored English, shown verbatim and HTML-escaped
        so it can never inject markup into the rich-text tip label. Empty/blank
        reasons are ignored."""
        text = (reason or "").strip()
        if not text:
            return False
        if getattr(self, "_prompt_info_kind", None) not in (None, "tip"):
            return False
        import html
        self._set_prompt_info(html.escape(text), tip=True, kind="hint")
        return True

    def _apply_prompt_hint_on_edit(self) -> None:
        """Keep the note under the prompt box in sync while the user types or
        draws examples. A non-empty prompt clears any stale guard message (the
        guard only fires on commit). With no prompt and examples drawn the note
        is count-aware: one positive nudges toward a second (reference-image
        detection is far better with a pair; the single-example run stays
        available through the escape link); two or more say the examples now
        drive the run.

        This writes the prompt-info line, a DIFFERENT widget from the example
        card's armed instruction and size warning, so the two never fight one
        label. When the card is showing the too-small size warning it is the
        more urgent, actionable message, so the second-example nudge yields to
        it (one calm info per state); show/clear of that warning re-run this
        method so the nudge returns once the warning clears."""
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
            # The example card owns the "add a second example" nudge now (its
            # quality dots + line under the draw button), so the prompt note
            # stays quiet here rather than saying the same thing twice.
            self._set_prompt_info()
            return
        # Two or more positives: the example-only run is possible, but the
        # accurate default still pairs the examples with a word (informative
        # note; committing without one triggers the meta intercept).
        self._set_prompt_info(
            tr("Your examples drive the search - naming the object makes it "
               "even more accurate."), info=True,
            hint_id=HINT_PROMPT_EXAMPLES_DRIVE)

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
