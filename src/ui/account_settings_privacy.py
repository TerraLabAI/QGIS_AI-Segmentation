





from __future__ import annotations

from ..core.i18n import tr


class AccountPrivacyMixin:





    def _on_telemetry_toggled(self, enabled: bool):





        from ..core.telemetry import set_telemetry_enabled
        from ..core.telemetry_session_events import track_telemetry_opt_changed

        if not enabled:
            try:
                track_telemetry_opt_changed(False)
            except Exception:
                pass  # nosec B110
        set_telemetry_enabled(enabled)
        self._show_saved()
        if enabled:
            try:
                track_telemetry_opt_changed(True)
            except Exception:
                pass  # nosec B110

    def _on_reset_hints(self):
        from ..core.qt_compat import safe_single_shot
        from .dock.guidance import reset_hints

        reset_hints()
        if hasattr(self, "_reset_hints_btn"):




            self._reset_hints_btn.setText(tr("Restored"))
            safe_single_shot(2000, self, self._restore_reset_hints_label)

    def _restore_reset_hints_label(self) -> None:
        try:
            self._reset_hints_btn.setText(tr("Show again"))
        except (RuntimeError, AttributeError):
            pass  # nosec B110
