






from __future__ import annotations


class AutoFlowErrorsMixin:


    @staticmethod
    def _classify_auto_error(msg: str) -> str:








        from ...core.error_policy import classify_run_error

        return classify_run_error(msg)

    def _open_auto_error_report(
        self, title: str, message: str, error_code: str, *, track: bool,
    ) -> None:





        if getattr(self, "_auto_headless_run", False):
            return
        if getattr(self, "_auto_error_dialog_shown", False):
            return
        self._auto_error_dialog_shown = True
        try:
            from ..error_report_dialog import show_error_report
            show_error_report(
                self.iface.mainWindow(), title, message, error_code, track=track)
        except Exception:  # nosec B110
            pass

    def _track_manual_run_failed(self) -> None:

        try:
            import time as _time

            from ...core import telemetry_session_events
            start_ts = getattr(self, "_segmentation_start_ts", None)
            duration_ms = int((_time.time() - start_ts) * 1000) if start_ts else None
            telemetry_session_events.track_segmentation_run(success=False, duration_ms=duration_ms)
        except Exception:
            pass  # nosec B110
