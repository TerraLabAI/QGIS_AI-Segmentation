






from __future__ import annotations

from ...core.i18n import tr
from ...core.qt_compat import safe_single_shot
from ...core.telemetry_errors import slot_guard
from .prompt_guard import is_known_object, validate_prompt






_SILENT_SWAP_REASONS = frozenset(
    {"translated", "plural", "alias", "server_rewrite"})






_PROMPT_LOOKUP_TIMEOUT_MS = 12_000



_MIN_LOOKUP_TIMEOUT_MS = 1_000
_MAX_LOOKUP_TIMEOUT_MS = 30_000


def prompt_server_lookup_enabled() -> bool:






    try:
        from ...core.server_dials import feature_enabled

        return feature_enabled("prompt_server_lookup")
    except Exception:  # noqa: BLE001
        return True


def prompt_lookup_timeout_ms() -> int:

    try:
        from ...core.server_dials import dial_in_range

        return int(dial_in_range("prompt.lookup_timeout_ms",
                                 _PROMPT_LOOKUP_TIMEOUT_MS,
                                 _MIN_LOOKUP_TIMEOUT_MS, _MAX_LOOKUP_TIMEOUT_MS))
    except Exception:  # noqa: BLE001
        return _PROMPT_LOOKUP_TIMEOUT_MS


class DockAutoPromptGateMixin:



    def confirm_prompt_for_detect(self) -> bool:












        text = self.auto_prompt_input.text()
        pending = getattr(self, "_prompt_lookup_key_pending", None)
        if pending is not None:
            if pending == self._prompt_lookup_key(text):



                self._set_prompt_info(self._prompt_lookup_note(), tip=True)
                return False


            self._abandon_prompt_lookup()
        if not text.strip():


            self._apply_prompt_hint_on_edit()
            return True
        ok, reason, suggestion = validate_prompt(text)
        if ok and reason is None and not is_known_object(text):




            token, waiting = self._resolve_prompt_via_server(text)
            if waiting:
                return False
            if token:
                reason, suggestion = "translated", token
        elif not ok and reason == "language":


            token, waiting = self._resolve_prompt_via_server(text)
            if waiting:
                return False
            if token:
                ok, reason, suggestion = True, "translated", token
        if ok:
            if reason in _SILENT_SWAP_REASONS and suggestion:




                self.apply_prompt_swap(suggestion, reason)
            elif reason == "multi_first" and suggestion:






                typed = text.strip()
                narrowed = suggestion != typed
                if narrowed:
                    self.auto_prompt_input.setText(suggestion)


                import html
                self._set_prompt_info(
                    tr('One object per run - detecting "{first}" now. '
                       'Run the other objects as separate detections.').format(
                        first=html.escape(suggestion)), tip=True)
                try:
                    from ...core import telemetry_run_events


                    telemetry_run_events.track_auto_prompt_steered(
                        prompt=suggestion, suggestion="multi_first")
                except Exception:
                    pass  # nosec B110
                if narrowed:







                    return False
            elif reason == "steer":







                self._show_prompt_steer_nudge(text, suggestion)
            else:
                self._set_prompt_info()
            return True




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


        return " ".join((text or "").split()).lower()

    @staticmethod
    def _prompt_lookup_note() -> str:
        return tr("Checking the object name...")

    def _prompt_lookup_answers(self) -> dict:


        cache = getattr(self, "_prompt_lookup_cache", None)
        if cache is None:
            cache = {}
            self._prompt_lookup_cache = cache
        return cache

    def _resolve_prompt_via_server(self, text: str) -> tuple[str | None, bool]:














        if not prompt_server_lookup_enabled():


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




        if getattr(self, "_prompt_lookup_key_pending", None) == key:
            return True
        self._abandon_prompt_lookup()
        try:
            from qgis.core import QgsApplication

            from ...api.prompt_translation import resolve_english_prompt
            from ...workers.generic_request_task import GenericRequestTask
        except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
            self._prompt_lookup_task = None
            self._prompt_lookup_key_pending = None
            self._prompt_lookup_answers()[key] = None
            return False
        try:


            safe_single_shot(
                prompt_lookup_timeout_ms(), self,
                lambda g=generation: self._on_prompt_lookup_done(g, None))
        except (RuntimeError, AttributeError):
            pass
        self._set_prompt_info(self._prompt_lookup_note(), tip=True)
        self._set_prompt_lookup_busy(True)
        return True

    @slot_guard(stage="segment", user_message=tr(
        "Something went wrong starting the detection. Please try again."))
    def _on_prompt_lookup_done(self, generation: int, answer: object) -> None:











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
            return
        if self._auto_run_active or self._auto_review_active:
            return
        try:
            from qgis.PyQt import sip



            if sip.isdeleted(self) is True:
                return
        except (ImportError, TypeError):
            pass
        self.auto_detect_requested.emit()

    def _abandon_prompt_lookup(self) -> None:




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



        try:
            if busy:
                self.auto_detect_btn.setEnabled(False)
            else:
                self._update_auto_detect_enabled()
        except (RuntimeError, AttributeError):
            pass
