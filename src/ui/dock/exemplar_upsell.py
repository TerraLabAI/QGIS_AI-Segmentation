

















from __future__ import annotations

from ...core.i18n import tr
from ...core.server_dials import dial_copy
from .pro_nudges import PRO_CARD_EXEMPLAR_CAP, pro_card_dismissed


class DockExemplarUpsellMixin:


    def show_auto_exemplar_upsell(self) -> None:








        if pro_card_dismissed(PRO_CARD_EXEMPLAR_CAP):
            self.hide_auto_exemplar_upsell()
            return
        card = getattr(self, "_auto_exemplar_upsell_card", None)
        if card is None:
            card = self._build_exemplar_upsell_card()
            if card is None:
                return
        from ...core.exemplar_store import max_total
        total = str(max_total())
        title = dial_copy("exemplar_cap.title", tr(
            "Free takes one example per run"))
        body = dial_copy("exemplar_cap.body", tr(
            "Pro takes up to {max}, look-alikes to exclude included."))
        escape = dial_copy("exemplar_cap.escape", tr(
            "Or run with this one."))
        fill = lambda text: text.replace("{max}", total)  # noqa: E731
        try:
            card.set_text(fill(title), fill(body),
                          dial_copy("exemplar_cap.cta", tr("Get Pro")),
                          escape=fill(escape))
            card.set_pro_offer("plugin_exemplar_cap")
            card.setVisible(True)
        except (RuntimeError, AttributeError):
            return


        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pro_upsell_viewed(trigger="exemplar_cap")
        except Exception:
            pass  # nosec B110

    def hide_auto_exemplar_upsell(self) -> None:

        card = getattr(self, "_auto_exemplar_upsell_card", None)
        if card is None:
            return
        try:
            card.setVisible(False)
        except (RuntimeError, AttributeError):
            self._auto_exemplar_upsell_card = None

    def _build_exemplar_upsell_card(self):


        from .upsell_card import UpsellCard
        layout = getattr(self, "_auto_exemplar_edit_layout", None)
        if layout is None:
            return None
        card = UpsellCard(
            "autoExemplarCapCard", "full",
            on_cta=self._on_exemplar_upsell_cta)
        card.enable_dismiss()
        card.dismissed.connect(
            lambda: self._dismiss_pro_card(PRO_CARD_EXEMPLAR_CAP, card))
        try:
            layout.addWidget(card)
        except (RuntimeError, AttributeError):
            return None
        self._auto_exemplar_upsell_card = card
        return card

    def _on_exemplar_upsell_cta(self) -> None:


        from ...core.pro_page_link import open_pro_page
        open_pro_page("plugin_exemplar_cap", "exemplar_cap", parent=self)
