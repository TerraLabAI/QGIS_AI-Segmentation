"""The contact button a subscriber meets at the end of the month.

Three cards route here (the km² wall, the running-low note, the Semi-Auto
objects wall). Each shows our address as its own line and a button that
copies it; the button says "Copied!" for a moment and goes back. The copy
of the card is shared by the three so a deploy rewords them together.

The mixin owns nothing but these handlers. The cards are built and shown
by auto_credits.py and manual_credit_gate.py.
"""
from __future__ import annotations

from ...core.i18n import tr
from ...core.server_dials import dial_copy
from .contact_copy import copy_cta_text, copy_with_feedback


class DockProCeilingContactMixin:
    """Handlers of the Pro ceiling contact button, per card."""

    def _pro_ceiling_copy(self) -> tuple[str, str]:
        """(body, cta) of every contact card, served under one id each."""
        return (
            dial_copy("pro_ceiling.body",
                      tr("Need more this month? Write to us and we set up a "
                         "plan that fits your volume.")),
            copy_cta_text(),
        )

    def _pro_ceiling_detail(self) -> str:
        """The address line between the body and the button."""
        from ...core.pro_ceiling import pro_ceiling_contact_email
        return pro_ceiling_contact_email()

    def _on_pro_contact_clicked(self, source: str, button=None) -> None:
        """Copy our address and let ``button`` say so for two seconds.

        ``source`` names the card for telemetry, the same three names the
        old mail path reported, so the funnel keeps its history.
        """
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
        """The takeover card's contact button. Same source as the km² wall it
        stands in for: one refusal, one line in the funnel."""
        card = getattr(self, "auto_run_block_card", None)
        self._on_pro_contact_clicked(
            "km2_block", getattr(card, "button", None))

    def _on_pro_contact_objects_wall(self) -> None:
        lane = getattr(self, "manual_credit_contact_lane", None)
        self._on_pro_contact_clicked(
            "objects_wall", getattr(lane, "button", None))
