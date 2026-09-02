









from __future__ import annotations

from ...core.i18n import tr
from ...core.server_dials import dial_copy
from .contact_copy import copy_cta_text, copy_with_feedback


class DockProCeilingContactMixin:


    def _pro_ceiling_copy(self) -> tuple[str, str]:

        return (
            dial_copy("pro_ceiling.body",
                      tr("Need more this month? Write to us and we set up a "
                         "custom quota.")),
            copy_cta_text(),
        )

    def _pro_ceiling_detail(self) -> str:

        from ...core.pro_ceiling import pro_ceiling_contact_email
        return pro_ceiling_contact_email()

    def _on_pro_contact_clicked(self, source: str, button=None) -> None:





        from ...core.pro_ceiling import pro_ceiling_contact_email
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pro_upsell_clicked(
                source="pro_ceiling_" + source)
        except Exception:
            pass  # nosec B110
        email = pro_ceiling_contact_email()
        if button is None:
            from .contact_copy import copy_to_clipboard
            copy_to_clipboard(email)
            return
        copy_with_feedback(button, email)

    def _on_pro_contact_low_credit(self) -> None:
        line = getattr(self, "_auto_low_credit_line", None)
        self._on_pro_contact_clicked(
            "low_credit", getattr(line, "button", None))

    def _on_pro_contact_km2_block(self) -> None:
        self._on_pro_contact_clicked(
            "km2_block", getattr(self, "auto_km2_block_upgrade", None))

    def _on_pro_contact_run_block(self) -> None:


        card = getattr(self, "auto_run_block_card", None)
        self._on_pro_contact_clicked(
            "km2_block", getattr(card, "button", None))

    def _on_pro_contact_objects_wall(self) -> None:
        lane = getattr(self, "manual_credit_contact_lane", None)
        self._on_pro_contact_clicked(
            "objects_wall", getattr(lane, "button", None))
