






from __future__ import annotations

from qgis.PyQt.QtCore import Qt

from ...core.i18n import tr
from ...core.server_dials import ServerDialSet, dial_copy, dial_in_range
from .guidance import (
    BLUE_TINT,
    HINT_PROMPT_EXAMPLES_DRIVE,
    HINT_PROMPT_EXEMPLAR_BOOST,
    HINT_PROMPT_ONE_OBJECT_PER_RUN,
    HINT_PROMPT_RUN_PLAN,
    HINT_PROMPT_STEER_OBJECT,
    HINT_PROMPT_TREE_OR_FOREST,
    HINT_PROMPT_UNKNOWN_OBJECT,
    NEUTRAL_TINT,
)
from .prompt_guard import (
    english_token_for,
    is_known_object,
    prompt_vocabulary_is_loaded,
    validate_prompt,
)
from .styles import (
    _msg_label_qss,
    msg_rich,
)





COUNT_VS_MAP_WORDS = ServerDialSet(
    "prompt.count_vs_map_words", ("tree", "trees"), normalize=str.lower)




_PROMPT_ADVICE_WAIT_MS = 900


class DockAutoPromptBoxMixin:



    def set_prompt_text(self, text: str) -> None:

        self._prompt_from_library = True
        self.auto_prompt_input.setText(text or "")
        self.auto_prompt_input.setFocus()

    def _on_auto_search_text_changed(self, _text: str = "") -> None:





        text = self.auto_prompt_input.text()



        if getattr(self, "_prompt_lookup_key_pending", None) is not None:
            self._abandon_prompt_lookup()



        self.hide_auto_zero_assist()
        self._apply_prompt_hint_on_edit()



        self.refresh_prompt_suggestions(text)
        self._update_auto_detect_enabled()

        self.refresh_auto_run_estimate()

        self._auto_prompt_debounce_timer.start(500)

    def _on_auto_search_return_pressed(self) -> None:



        self.auto_enter_pressed.emit()

    def _on_auto_prompt_editing_finished(self) -> None:


        try:
            self._auto_prompt_debounce_timer.stop()
        except (RuntimeError, AttributeError):

            pass
        self._emit_auto_prompt_committed(force=True)

    def _prompt_plausibly_complete(self, text: str) -> bool:





        try:
            if not self.auto_prompt_input.hasFocus():
                return True
            return is_known_object(text) or english_token_for(text) is not None
        except (RuntimeError, AttributeError):
            return True

    def _emit_auto_prompt_committed(self, force: bool = False) -> None:









        text = self.auto_prompt_input.text().strip()
        if text and not force and not self._prompt_plausibly_complete(text):
            self._advise_on_uncommitted_prompt(text)
            return
        self._prompt_advice_waiting = None
        if text == getattr(self, "_last_committed_prompt", None):
            return
        self._last_committed_prompt = text


        try:
            from ...core.detection_policy import prompt_suggests_canopy
            self._auto_prompt_canopy = bool(text) and prompt_suggests_canopy(text)
            self._refresh_auto_exemplar_explainer(
                slot_taken=self._auto_exemplar_line_busy())
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        if text:






            try:
                ok, reason, suggestion = validate_prompt(text)
                if ok and text.strip().lower() in COUNT_VS_MAP_WORDS:


                    self._set_prompt_info(
                        dial_copy(
                            "prompt_guard.count_vs_map",
                            tr('Dense forest? "Forest" takes it as one block; '
                               '"Tree" picks individual trees.')),
                        tip=True, hint_id=HINT_PROMPT_TREE_OR_FOREST)
                elif ok and reason == "multi_first" and suggestion:



                    self._set_prompt_info(
                        tr('One object per run - Detect will run "{first}" '
                           'first.').format(first=suggestion), tip=True,
                        hint_id=HINT_PROMPT_ONE_OBJECT_PER_RUN)
                elif ok and reason == "steer":



                    self._show_prompt_steer_nudge(text, suggestion)
                elif ok and reason is None:





                    from ...core.detection_policy import prompt_hint_for
                    served = prompt_hint_for(text)
                    if served is None or not self._set_prompt_info(
                            served[1], tip=True, hint_id=served[0]):




                        self._maybe_show_exemplar_boost_nudge(text)
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:
                from ...core import telemetry_run_events
                telemetry_run_events.track_auto_prompt_committed(
                    prompt=text,
                    from_library=getattr(self, "_prompt_from_library", False),
                )
            except Exception:
                pass  # nosec B110


        self._prompt_from_library = False
        self.auto_prompt_committed.emit(text)

    def _advise_on_uncommitted_prompt(self, text: str) -> None:








        if getattr(self, "_prompt_advice_waiting", None) != text:
            self._prompt_advice_waiting = text
            try:
                self._auto_prompt_debounce_timer.start(dial_in_range(
                    "tuning.auto.prompt_advice_wait_ms",
                    _PROMPT_ADVICE_WAIT_MS, 600, 5000))
            except (RuntimeError, AttributeError):

                pass
            return
        self._prompt_advice_waiting = None
        try:
            ok, reason, _suggestion = validate_prompt(text)
            if ok and reason is None:
                self._maybe_show_exemplar_boost_nudge(text)
        except Exception:  # noqa: BLE001
            pass  # nosec B110

    def _set_prompt_info(self, text: str | None = None, error: bool = False,
                         info: bool = False, tip: bool = False,
                         kind: str | None = None,
                         hint_id: str | None = None) -> bool:


























        self.auto_prompt_tip.setVisible(False)
        if not text:
            self._prompt_info_kind = None
            self.auto_prompt_info.setText("")
            self.auto_prompt_info.setVisible(False)
            return False
        if hint_id:
            self.auto_prompt_info.setVisible(False)


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






            self.auto_prompt_info.setTextFormat(Qt.TextFormat.RichText)
            self.auto_prompt_info.setText(msg_rich("info", text, is_html=True))
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

    def _prompt_is_unknown_word(self, token: str) -> bool:



















        try:


            if not prompt_vocabulary_is_loaded():
                return False
            if is_known_object(token) or english_token_for(token) is not None:
                return False
            answers = self._prompt_lookup_answers()
            answer = answers.get(self._prompt_lookup_key(token))
            if not isinstance(answer, str) or not answer.strip():
                return True
            ok, _reason, _suggestion = validate_prompt(answer)
            return not ok
        except Exception:  # noqa: BLE001
            return False

    def _show_prompt_steer_nudge(self, typed: str, suggestion: str | None) -> bool:










        word = (typed or "").strip()
        if not word:
            return False
        term = (suggestion or "").strip()
        if term:

            shown = self._set_prompt_info(
                tr('"{word}" is hard to spot from above - "{term}" detects '
                   'better. Your word still runs.').format(word=word, term=term),
                tip=True, kind="steer", hint_id=HINT_PROMPT_STEER_OBJECT)
        else:

            import html
            shown = self._set_prompt_info(
                tr('"{word}" cannot be seen from above. Pick an object on the '
                   'ground - the Library has ready-to-use ones.').format(
                       word=html.escape(word)),
                tip=True, kind="steer")
        if not shown:
            return False

        key = word.lower()
        if getattr(self, "_steer_nudge_tracked", None) != key:
            self._steer_nudge_tracked = key
            try:
                from ...core import telemetry_run_events
                telemetry_run_events.track_auto_prompt_steered(
                    prompt=word, suggestion=term)
            except Exception:
                pass  # nosec B110
        return True

    def _maybe_show_exemplar_boost_nudge(self, token: str) -> bool:

















        token = (token or "").strip()
        if not token:
            return False
        if self._auto_run_active or self._auto_review_active:
            return False
        can_draw_example = (
            self._EXEMPLARS_ENABLED
            and getattr(self, "_auto_positive_exemplars", 0) <= 0)
        try:
            from .prompt_guard import is_exemplar_boost_prompt
            if can_draw_example and is_exemplar_boost_prompt(token):
                kind, hint_id = "exemplar_boost", HINT_PROMPT_EXEMPLAR_BOOST
                message = tr(
                    '"{obj}" is often missed from text alone. Draw one '
                    'example on the map to find far more.').format(obj=token)
            elif self._prompt_is_unknown_word(token):
                kind, hint_id = "unknown_object", HINT_PROMPT_UNKNOWN_OBJECT
                if can_draw_example:
                    message = tr(
                        '"{obj}" is not an object the AI knows well. Drawing '
                        'one example on the map shows it what to detect.').format(
                            obj=token)
                else:
                    message = tr(
                        '"{obj}" is not an object the AI knows well. The run '
                        'may come back empty - a more common word finds '
                        'more.').format(obj=token)
            else:
                return False
        except Exception:  # noqa: BLE001
            return False

        if not self._set_prompt_info(
                message, tip=True, kind="boost", hint_id=hint_id):
            return False

        key = (kind, token.lower())
        if getattr(self, "_boost_nudge_tracked", None) != key:
            self._boost_nudge_tracked = key
            try:
                from ...core import telemetry_run_events
                telemetry_run_events.track_auto_prompt_hint_shown(
                    kind=kind, prompt=token)
            except Exception:
                pass  # nosec B110
        return True

    def show_auto_prompt_hint(self, hint: str) -> bool:








        text = (hint or "").strip()
        if not text:
            return False
        if getattr(self, "_prompt_info_kind", None) not in (None, "tip"):
            return False
        return self._set_prompt_info(
            text, tip=True, kind="hint", hint_id=HINT_PROMPT_RUN_PLAN)

    def apply_prompt_swap(self, token: str, reason: str) -> bool:



















        token = (token or "").strip()
        typed = self.auto_prompt_input.text().strip()
        if not token or not typed or token.lower() == typed.lower():
            return False
        self.auto_prompt_input.setText(token)
        shown = self._set_prompt_info(
            tr('"{word}" will run as "{token}".').format(
                word=typed, token=token), info=True, kind="swap")
        if shown:
            try:
                from ...core import telemetry_run_events
                telemetry_run_events.track_auto_prompt_rewritten(
                    kind=reason, prompt=token)
            except Exception:  # noqa: BLE001
                pass  # nosec B110
        return True

    def show_auto_prompt_decline(self, reason: str) -> bool:








        text = (reason or "").strip()
        if not text:
            return False
        if getattr(self, "_prompt_info_kind", None) not in (None, "tip"):
            return False
        import html
        self._set_prompt_info(html.escape(text), tip=True, kind="hint")
        return True

    def _apply_prompt_hint_on_edit(self) -> None:













        try:
            has_text = bool(self.auto_prompt_input.text().strip())
        except (RuntimeError, AttributeError):
            return
        positives = (getattr(self, "_auto_positive_exemplars", 0)
                     if self._EXEMPLARS_ENABLED else 0)
        if has_text or positives <= 0:

            self._set_prompt_info()
            return
        if positives == 1:



            self._set_prompt_info()
            return



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
