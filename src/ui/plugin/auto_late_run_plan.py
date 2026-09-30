














from __future__ import annotations

import time as _t

from qgis.core import Qgis, QgsMessageLog



_LATE_PLAN_FINALIZE_CAP_S = 3.0


class AutoLateRunPlanMixin:


    def _run_plan_fetch_in_flight(self, token: str) -> bool:

        return (getattr(self, "_auto_run_plan_task", None) is not None
                and (getattr(self, "_auto_run_plan_task_prompt", "") or "").lower()
                == (token or "").lower())

    def _late_plan_begin(self, prompt: str) -> None:


        token = self._resolve_object_token(prompt) if prompt else ""
        if not self._run_plan_fetch_in_flight(token):
            try:
                self._fetch_auto_run_plan(token)
            except Exception:  # noqa: BLE001  # nosec B110
                pass
        from ...core.run_decisions import fallback_choices

        fallback = fallback_choices(token) if token else None
        self._auto_late_plan = {
            "token": token, "prompt": prompt, "state": "pending",
            "t0": _t.monotonic(), "fallback": fallback,
        }
        if fallback is not None:
            self._auto_run_decisions = dict(fallback["decisions"])
            self._auto_late_plan["start_decisions"] = dict(fallback["decisions"])
        source = fallback["source"] if fallback else "neutral"
        QgsMessageLog.logMessage(
            f"Auto detection: run started before its plan; {source} choices "
            "until it answers", "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _late_plan_pending(self) -> bool:
        lp = getattr(self, "_auto_late_plan", None)
        return isinstance(lp, dict) and lp.get("state") == "pending"

    def _late_plan_fallback(self, prompt: str | None = None) -> dict | None:


        lp = getattr(self, "_auto_late_plan", None)
        fb = lp.get("fallback") if isinstance(lp, dict) else None
        if fb is not None and prompt is not None:
            words = {(lp.get("prompt") or "").strip().lower(),
                     (lp.get("token") or "").strip().lower()}
            if (prompt or "").strip().lower() not in words:
                return None
        return fb if isinstance(fb, dict) else None

    def _late_plan_clear(self) -> None:
        self._auto_late_plan = None
        self._auto_late_remerge = None

    def _late_plan_on_ready(self, prompt: str, plan: object, exemplar_size_m=None) -> bool:


        lp = getattr(self, "_auto_late_plan", None)
        if not self._late_plan_pending():
            return False
        if (prompt or "").strip().lower() != (lp.get("token") or "").lower():
            return False
        if not isinstance(plan, dict) or plan.get("error"):
            self._late_plan_settle(None, "failed")
            return True
        from ...core.run_decisions import parse_run_decisions

        decisions = parse_run_decisions(plan)


        self._auto_run_plan = {
            "prompt": lp.get("prompt") or "", "plan": plan,
            "exemplar_size_m": exemplar_size_m}
        self._auto_review_preset_memo = None
        conf = plan.get("confidence_default")
        if isinstance(conf, (int, float)) and not isinstance(conf, bool):
            self._auto_start_confidence_default = float(conf)
        self._late_plan_settle(decisions, "arrived" if decisions else "no_decisions")
        return True

    def _late_plan_on_failed(self, prompt: str) -> bool:

        lp = getattr(self, "_auto_late_plan", None)
        if not self._late_plan_pending():
            return False
        if (prompt or "").strip().lower() != (lp.get("token") or "").lower():
            return False
        self._late_plan_settle(None, "failed")
        return True

    def _late_plan_settle(self, decisions: dict | None, outcome: str) -> None:


        from ...core.run_decisions import neutral_run_decisions

        lp = getattr(self, "_auto_late_plan", None)
        if not isinstance(lp, dict) or lp.get("state") != "pending":
            return
        lp["state"] = outcome
        self._auto_run_decisions = decisions or neutral_run_decisions()
        after_s = _t.monotonic() - float(lp.get("t0") or _t.monotonic())
        QgsMessageLog.logMessage(
            f"Auto detection: late run plan {outcome} {after_s:.1f} s after "
            f"the start; {'plan' if decisions else 'neutral'} choices",
            "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _late_plan_hold_finalize(self, state: dict) -> bool:



        lp = getattr(self, "_auto_late_plan", None)
        if not isinstance(lp, dict):
            return False
        if lp.get("state") == "pending":
            fallback = self._late_plan_fallback()
            if fallback is not None:
                self._late_plan_settle(
                    fallback["decisions"], f"fallback_{fallback['source']}")
            elif self._run_plan_fetch_in_flight(lp.get("token") or ""):
                state["phase"] = "plan"
                state["plan_until"] = _t.monotonic() + _LATE_PLAN_FINALIZE_CAP_S
                from qgis.PyQt.QtCore import QTimer
                QTimer.singleShot(self._auto_offgui_poll_ms(),
                                  self._step_auto_finalize_refine)
                return True
            else:
                self._late_plan_settle(None, "not_fetched")
        self._late_plan_apply_decisions()
        return False

    def _late_plan_still_waiting(self, state: dict) -> bool:


        if not self._late_plan_pending():
            return False
        if _t.monotonic() < float(state.get("plan_until") or 0.0):
            return True
        self._late_plan_settle(None, "timeout")
        return False

    def _late_plan_settle_now(self) -> None:

        if self._late_plan_pending():
            fallback = self._late_plan_fallback()
            self._late_plan_settle(
                fallback["decisions"] if fallback else None,
                f"fallback_{fallback['source']}" if fallback else "not_arrived")
        if isinstance(getattr(self, "_auto_late_plan", None), dict):
            self._late_plan_apply_decisions()

    def _late_plan_apply_decisions(self) -> None:



        lp = getattr(self, "_auto_late_plan", None)
        if not isinstance(lp, dict) or lp.get("applied"):
            return
        lp["applied"] = True
        self._auto_late_remerge = None
        if getattr(self, "_auto_is_exemplar_only", False):
            self._auto_restore_partitions = bool(
                (self._auto_run_decisions or {}).get("restore_partitions", False))
            return
        decisions = self._auto_run_decisions or {}
        if decisions == lp.get("start_decisions"):
            return
        want_separate = bool(decisions.get("merge_separate", True))
        restore = bool(decisions.get("restore_partitions", False))
        if want_separate == bool(self._auto_merge_separate) and not (want_separate and restore):
            return
        if not getattr(self, "_auto_raw_fragments", None):
            QgsMessageLog.logMessage(
                "Auto detection: the late plan's grouping could not be applied "
                "(no retained fragments); keeping the run's own",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return
        self._auto_restore_partitions = restore
        self._auto_late_remerge = want_separate
        self._auto_finalize_fragments = None
