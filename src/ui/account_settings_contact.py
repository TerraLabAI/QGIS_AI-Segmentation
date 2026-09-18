





from __future__ import annotations

from .dock.contact_copy import copy_with_feedback


class AccountContactMixin:





    def _track_contact_click(self, action: str) -> None:



        try:
            from ..core import telemetry_session_events
            telemetry_session_events.track_pro_upsell_clicked(
                source="account_contact_" + action)
        except Exception:  # noqa: BLE001
            pass  # nosec B110

    def _on_contact_copy(self, button, address: str) -> None:
        self._track_contact_click("copy")
        copy_with_feedback(button, address)

    def _on_contact_call(self, url: str) -> None:
        self._track_contact_click("call")
        from .external_links import open_external_url
        open_external_url(url, parent=self)
