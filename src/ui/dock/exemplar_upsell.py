"""The Pro offer on the example card: shown when a free account asks for a
second example.

The free plan includes one positive example per run and no exclude. The
draw buttons stay live past that point on purpose (a hidden or greyed button
sells nothing), and the click that would have armed the draw tool shows this
card instead. It is the shared premium ``UpsellCard`` in its blocking form:
the fact (free takes one example, this run has it), one line on what Pro
takes, the blue button, and the free way out (run it with this example).

Built once on first refusal, under the example card's editing controls, and
hidden again as soon as the example set changes (an example removed, the
zone redrawn, a run started), so it never outlives the state that raised it.
Copy goes through ``dial_copy`` so a served sentence can retune it without a
plugin release; the example count it quotes is the served Pro ceiling.
"""
from __future__ import annotations

from ...core.i18n import tr
from ...core.server_dials import dial_copy


class DockExemplarUpsellMixin:
    """Example-cap offer card on the dock (Automatic mode)."""

    def show_auto_exemplar_upsell(self) -> None:
        """Show the Pro offer where the second example would have gone.

        Called by the plugin when the account is on the free plan and the
        example store refuses a label a paid plan would still take. Builds
        the card on first use; a stale widget (dock rebuilt) is rebuilt too.
        """
        card = getattr(self, "_auto_exemplar_upsell_card", None)
        if card is None:
            card = self._build_exemplar_upsell_card()
            if card is None:
                return
        from ...core.exemplar_store import max_total
        total = str(max_total())
        title = dial_copy("exemplar_cap.title", tr(
            "Free includes one example per run, and this run has it."))
        body = dial_copy("exemplar_cap.body", tr(
            "Pro takes up to {max} examples per run, look-alikes to exclude "
            "included, so the AI finds exactly what you mean."))
        escape = dial_copy("exemplar_cap.escape", tr(
            "Or run it with this one example."))
        fill = lambda text: text.replace("{max}", total)  # noqa: E731
        try:
            card.set_text(fill(title), fill(body),
                          dial_copy("upsell.cta", tr("Upgrade to Pro")),
                          escape=fill(escape))
            card.setVisible(True)
        except (RuntimeError, AttributeError):
            return
        # Impression tracked so the click has a denominator. Deduped per
        # trigger, so once per session.
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pro_upsell_viewed(trigger="exemplar_cap")
        except Exception:
            pass  # nosec B110

    def hide_auto_exemplar_upsell(self) -> None:
        """Hide the offer. Safe to call when it was never built."""
        card = getattr(self, "_auto_exemplar_upsell_card", None)
        if card is None:
            return
        try:
            card.setVisible(False)
        except (RuntimeError, AttributeError):
            self._auto_exemplar_upsell_card = None

    def _build_exemplar_upsell_card(self):
        """Build the card once, under the example card's editing controls.
        Returns None when that layout is not there to hold it."""
        from .upsell_card import UpsellCard
        layout = getattr(self, "_auto_exemplar_edit_layout", None)
        if layout is None:
            return None
        card = UpsellCard(
            "autoExemplarCapCard", "full",
            on_cta=self._on_exemplar_upsell_cta)
        try:
            layout.addWidget(card)
        except (RuntimeError, AttributeError):
            return None
        self._auto_exemplar_upsell_card = card
        return card

    def _on_exemplar_upsell_cta(self) -> None:
        """The blue button: same destination as every other Pro button in the
        dock, tracked with its own source."""
        from ..external_links import open_external_url
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pro_upsell_clicked(source="exemplar_cap")
        except Exception:
            pass  # nosec B110
        open_external_url(self._build_upgrade_url("plugin_exemplar_cap"), parent=self)
