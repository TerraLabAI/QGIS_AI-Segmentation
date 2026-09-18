










from __future__ import annotations

import html

from qgis.PyQt.QtWidgets import QWidget

from ...core.activation_manager import book_a_call_url
from ...core.i18n import tr
from ...core.server_dials import dial_copy
from ..account_settings_plan import account_balance_lines, resolve_plan_credits
from ..dock.contact_copy import copy_cta_text
from ..dock.font_scale import apply_font_scale_to_tree
from ..dock.styles import _BTN_SETTINGS_GHOST
from .category_tile import TILE_SMALL_GLYPH_PX, TILE_SMALL_PX, category_icon_tile
from .settings_widgets import (
    BillingCard,
    BillingCardRow,
    ButtonFlow,
    SettingGroup,
    SettingRow,
    SettingsPage,
    balance_bar,
    clear_layout,
    muted_label,
    settings_button,
)


class BillingPageMixin:


    def _build_billing_page(self) -> SettingsPage:
        page = SettingsPage(tr("Billing"), "", self, glyph="gem", category="amber")
        self._billing_col = page.add_box()
        return page

    def _paint_billing(self, state: dict) -> None:
        col = getattr(self, "_billing_col", None)
        if col is None:
            return
        clear_layout(col)
        kind = (state or {}).get("kind")
        if kind == "loading":
            col.addWidget(muted_label(tr("Loading your plan...")))
            return
        cards = BillingCardRow()
        if kind == "loaded":
            account = dict(state.get("account") or {})
            usage = dict(state.get("usage") or {})
            sub = self._find_subscription(account) or {}
            plan = resolve_plan_credits(usage, sub)
            cards.add_card(self._billing_plan_card(plan, account_balance_lines(usage, sub)))
            cards.add_card(self._billing_manage_card(plan.is_subscriber, plan.is_subscriber))
            col.addWidget(cards)
            col.addWidget(self._billing_contact_group(plan.is_subscriber))
        else:


            col.addWidget(muted_label(tr("Shown once your account loads.")))
            cards.add_card(self._billing_manage_card(True, False))
            col.addWidget(cards)
        apply_font_scale_to_tree(col.parentWidget())

    def _billing_plan_card(self, plan, lines: list) -> BillingCard:
        card = BillingCard(tr("Pro plan") if plan.is_subscriber else tr("Free plan"),
                           category="amber")
        for index, line in enumerate(lines):
            if index:
                spacer = QWidget(card)
                spacer.setFixedHeight(10)
                card.add(spacer)
            card.add_stat(line.figure, line.caption)
            if line.total_units > 0:
                card.add(balance_bar(card, line.left_units, line.total_units, line.title))
        if not lines:
            card.add_status(tr("No usage counted yet this month."))
        from ...core.quota_reset_date import format_quota_reset_date

        reset_str = format_quota_reset_date(plan.reset_date)
        if reset_str:
            card.add_note(tr("Resets {date}").format(date=reset_str))
        if not plan.is_subscriber:
            card.add_note(tr("Personal, non-commercial use only."))
        return card

    def _billing_manage_card(self, is_subscriber: bool, dashboard_filled: bool) -> BillingCard:



        del dashboard_filled
        card = BillingCard(tr("Manage"))
        card.add_status(tr("Plan, payment and invoices."))
        if not is_subscriber:
            offer = settings_button(tr("See what Pro unlocks"), _BTN_SETTINGS_GHOST)
            offer.clicked.connect(self._on_upgrade_clicked)
            card.add_button(offer)
        dashboard = settings_button(tr("Open dashboard"), _BTN_SETTINGS_GHOST)
        dashboard.setToolTip(tr("Opens your terra-lab.ai dashboard in the browser."))
        dashboard.clicked.connect(lambda: self._open_dashboard("billing_page"))
        card.add_button(dashboard)
        card.add_note(tr("Payments happen on terra-lab.ai."))
        return card

    def _billing_contact_group(self, is_subscriber: bool) -> SettingGroup:

        title = (dial_copy("account.contact_pro_title", tr("Need more than Pro?"))
                 if is_subscriber else
                 dial_copy("account.contact_free_title", tr("Working in a team?")))
        body = dial_copy("account.contact_body",
                         tr("Custom quota, team seats, invoices, or a custom AI solution."))


        address = html.unescape(dial_copy(
            "account.contact_email", "yvann.barbot@terra-lab.ai", max_chars=120, escape=True))
        buttons = ButtonFlow()
        copy_btn = settings_button(copy_cta_text().replace("&", "&&"), _BTN_SETTINGS_GHOST, buttons)
        copy_btn.setToolTip(address)
        copy_btn.clicked.connect(
            lambda _=False, btn=copy_btn: self._on_contact_copy(btn, address))
        buttons.add(copy_btn)


        call_url = book_a_call_url()
        if call_url:
            call_btn = settings_button(
                dial_copy("account.contact_call_cta", tr("Book a call")), _BTN_SETTINGS_GHOST, buttons)
            call_btn.clicked.connect(lambda _=False, url=call_url: self._on_contact_call(url))
            buttons.add(call_btn)
        group = SettingGroup()
        tile = category_icon_tile("chat_bubble", "sky", None, TILE_SMALL_PX, TILE_SMALL_GLYPH_PX)
        group.add_row(SettingRow(title, body, buttons, lead=tile))
        return group
