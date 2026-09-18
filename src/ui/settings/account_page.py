











from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QLabel

from ...core.i18n import tr
from ...core.server_dials import dial_copy
from ..account_settings_plan import _STATUS_DISPLAY, resolve_plan_credits
from ..dock.font_scale import apply_font_scale_to_tree, scale_px_length, scale_qss_font_px
from ..dock.styles import (
    _BTN_SETTINGS_ACCENT,
    _BTN_SETTINGS_DANGER,
    _BTN_SETTINGS_GHOST,
    FONT_BASE,
    ON_ACCENT,
    category_fill,
)
from .category_tile import TILE_SMALL_GLYPH_PX, TILE_SMALL_PX, avatar_category, category_icon_tile
from .settings_widgets import (
    ROW_ERROR_QSS,
    ROW_NOTE_QSS,
    ROW_TITLE_STRONG_QSS,
    ButtonFlow,
    ElidedLabel,
    SettingGroup,
    SettingRow,
    SettingsPage,
    SettingSwitch,
    clear_layout,
    settings_button,
)

_AVATAR_D = 36


def _row_tile(glyph: str, category: str):

    return category_icon_tile(glyph, category, None, TILE_SMALL_PX, TILE_SMALL_GLYPH_PX)


class AccountPageMixin:


    def _build_account_page(self) -> SettingsPage:
        page = SettingsPage(tr("Account"), "", self, glyph="person", category="green")


        self._account_col = page.add_box()
        self._add_advanced_group(page)
        self._add_danger_zone(page)
        return page



    def _paint_account_state(self, state: dict) -> None:

        self._account_state = dict(state or {})
        kind = self._account_state.get("kind")
        self._cancel_avatar_load()
        self._avatar_label = None
        clear_layout(self._account_col)
        free_account = False
        if kind == "loading":
            group = SettingGroup()
            group.add_row(SettingRow(tr("Loading account info..."), "", None))
            self._account_col.addWidget(group)
        elif kind == "error":
            self._show_account_error(str(self._account_state.get("code") or ""),
                                     dict(self._account_state.get("payload") or {}))
        else:
            account = dict(self._account_state.get("account") or {})
            usage = dict(self._account_state.get("usage") or {})
            sub = self._find_subscription(account) or {}
            plan = resolve_plan_credits(usage, sub)
            free_account = not plan.is_subscriber
            self._show_account(account, usage, sub, plan)
        self._sync_upgrade_pill(free_account)
        self._sync_delete_row()
        try:
            self._paint_billing(self._account_state)
        except RuntimeError:
            pass  # nosec B110
        apply_font_scale_to_tree(self._account_col.parentWidget())

    def _show_account_error(self, code: str, payload: dict) -> None:

        from ..account_settings_session import _ACCOUNT_OFFLINE_CODES

        code = (code or "").strip().upper()
        retry, manage = False, False
        if code == "SUBSCRIPTION_INACTIVE":
            title = tr("Your last payment may have failed")
            note = ""
            manage = True
        elif code == "INVALID_KEY":
            title = tr("This computer is no longer signed in")
            note = tr("Sign in again.")
        elif code == "DEVICE_LIMIT_EXCEEDED":
            used, cap = payload.get("active_devices"), payload.get("device_limit")
            title = tr("Your plan is on its maximum number of computers")
            if isinstance(used, int) and isinstance(cap, int) and cap > 0:
                title = tr("{used} of {cap} computers in use.").format(used=used, cap=cap)
            note = tr("Close it on another computer.")
        elif code in _ACCOUNT_OFFLINE_CODES:
            title = tr("Could not reach TerraLab")
            note = tr("Check your connection.")
            retry = True
        else:
            title = tr("Could not load your account")
            note = tr("Try again in a moment.")
            retry = True
        buttons = ButtonFlow()
        if manage:
            fix = settings_button(tr("Update payment method"), _BTN_SETTINGS_ACCENT, buttons)
            fix.setToolTip(tr("Opens your terra-lab.ai account in the browser."))
            fix.clicked.connect(lambda: self._open_dashboard("error_card"))
            buttons.add(fix)
        if retry:
            again = settings_button(tr("Retry"), _BTN_SETTINGS_GHOST, buttons)
            again.clicked.connect(self._fetch_account)
            buttons.add(again)
        out = settings_button(tr("Sign out"), _BTN_SETTINGS_GHOST, buttons)
        out.clicked.connect(lambda: self._on_sign_out("error_card"))
        buttons.add(out)
        group = SettingGroup()
        error_row = SettingRow(title, note, buttons, lead=_row_tile("warning", "amber"))
        error_row.title_label.setStyleSheet(ROW_TITLE_STRONG_QSS)
        group.add_row(error_row)
        self._account_col.addWidget(group)



    def _show_account(self, account: dict, usage: dict, sub: dict, plan) -> None:
        email = str(account.get("email") or "-")
        group = SettingGroup()
        group.add_row(self._identity_row(account, email, sub, plan))

        group.add_row(self._usage_link_row())
        if not plan.is_subscriber:
            group.add_row(self._pro_row())
        self._account_col.addWidget(group)

    def _identity_row(self, account: dict, email: str, sub: dict, plan) -> SettingRow:
        diameter = scale_px_length(_AVATAR_D)
        avatar = QLabel(email[:1].upper() if email and email != "-" else "?")
        avatar.setFixedSize(diameter, diameter)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        avatar.setAccessibleName(tr("Account picture"))


        avatar.setStyleSheet(scale_qss_font_px(
            f"background: {category_fill(avatar_category(email))}; color: {ON_ACCENT};"
            f" border-radius: {diameter // 2}px; font-size: {FONT_BASE + 2}px; font-weight: 700;"))
        self._avatar_label = avatar
        self._show_account_picture(account.get("avatar_url"), diameter)



        address = ElidedLabel(email)
        address.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        address.setStyleSheet(ROW_TITLE_STRONG_QSS)
        if plan.is_subscriber:
            status = str(sub.get("status") or "active")


            status_text, _colour = _STATUS_DISPLAY.get(
                status, (status.replace("_", " ").capitalize(), ""))
            line = tr("{plan} · {status}").format(plan=tr("Pro plan"), status=status_text)
        else:
            line = tr("{plan} · Personal, non-commercial").format(plan=tr("Free plan"))

        buttons = ButtonFlow()
        manage = settings_button(tr("Manage"), _BTN_SETTINGS_GHOST)
        manage.setToolTip(tr("Opens your terra-lab.ai dashboard in the browser."))
        manage.setAccessibleName(tr("Manage account in browser"))
        manage.clicked.connect(lambda: self._open_dashboard("account_card"))
        buttons.add(manage)
        out = settings_button(tr("Sign out"), _BTN_SETTINGS_GHOST)
        out.clicked.connect(lambda: self._on_sign_out("account_card"))
        buttons.add(out)

        row = SettingRow(email, line.replace("-", "\u2011"), buttons,
                         lead=avatar, title_label=address)
        row.layout().setContentsMargins(14, 12, 14, 12)
        cancelled = plan.is_subscriber and str(sub.get("status") or "") not in ("", "active", "trialing")
        row.note_label.setStyleSheet(ROW_ERROR_QSS if cancelled else ROW_NOTE_QSS)
        return row

    def _usage_link_row(self) -> SettingRow:

        see = settings_button(tr("See usage"), _BTN_SETTINGS_GHOST)
        see.setToolTip(tr("What is left of your plan this month."))
        see.clicked.connect(lambda: self.show_settings_page("billing"))
        return SettingRow(tr("Usage"), "", see, lead=_row_tile("chart", "amber"))

    def _pro_row(self) -> SettingRow:



        upgrade = settings_button(tr("Get Pro"), _BTN_SETTINGS_GHOST)
        upgrade.setAccessibleName(tr("Get Pro"))
        upgrade.clicked.connect(self._on_upgrade_clicked)

        detail = " · ".join((
            dial_copy("upsell.bullet_quota_manual",
                      tr("500 cloud objects every month in Semi-Auto")),
            dial_copy("account.upgrade_title",
                      tr("200 km² of Automatic a month, on zones of any size.")),
        ))
        upgrade.setToolTip(detail + "\n" + dial_copy(
            "account.upgrade_tooltip", tr("Opens terra-lab.ai in your browser.")))
        row = SettingRow(tr("Pro plan"), tr("Commercial use, higher limits"), upgrade,
                         lead=_row_tile("gem", "amber"))
        row.setToolTip(detail)

        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pro_upsell_viewed(
                trigger="account_dialog", cta_source="account_dialog")
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        return row



    def _add_advanced_group(self, page: SettingsPage) -> None:
        from ...core.telemetry import is_telemetry_enabled

        page.add_group_title(tr("Advanced"))
        group = SettingGroup()
        self._telemetry_switch = SettingSwitch(None, is_telemetry_enabled())
        self._telemetry_switch.toggled.connect(self._on_telemetry_toggled)
        detail = (tr("Errors, versions and the words you type, linked to your "
                     "account. Never your imagery, layers or coordinates.")
                  + " " + tr("After an Automatic run, its technical log lines are sent too.")
                  + "\n" + tr("On Pro, lifecycle and counts only, no content."))
        row = SettingRow(tr("Usage statistics"), tr("Never your imagery or coordinates"),
                         self._telemetry_switch, lead=_row_tile("shield", "green"))
        row.setToolTip(detail)
        self._telemetry_switch.setToolTip(detail)
        group.add_row(row)
        self._reset_hints_btn = settings_button(tr("Show again"), _BTN_SETTINGS_GHOST)
        self._reset_hints_btn.clicked.connect(self._on_reset_hints)
        self._reset_hints_btn.setToolTip(tr("The tips you closed in the panel come back."))
        group.add_row(SettingRow(tr("Guidance tips"), "", self._reset_hints_btn,
                                 lead=_row_tile("lightbulb", "green")))
        page.add(group)

    def _add_danger_zone(self, page: SettingsPage) -> None:
        page.add_group_title(tr("Danger zone"))
        group = SettingGroup(category="coral")
        self._delete_btn = settings_button(tr("Delete"), _BTN_SETTINGS_DANGER)
        self._delete_btn.clicked.connect(self._on_delete_account_clicked)
        self._delete_row = SettingRow(tr("Delete account"), "", self._delete_btn,
                                      lead=_row_tile("trash", "coral"))


        self._delete_status = QLabel("")
        self._delete_status.setWordWrap(True)
        self._delete_status.setStyleSheet(ROW_NOTE_QSS)
        self._delete_status.setVisible(False)
        self._delete_row.words.addWidget(self._delete_status)
        group.add_row(self._delete_row)
        page.add(group)
        self._sync_delete_row()

    def _sync_delete_row(self) -> None:

        button = getattr(self, "_delete_btn", None)
        row = getattr(self, "_delete_row", None)
        if button is None or row is None:
            return
        try:
            if self._delete_running:
                button.setEnabled(False)
                return
            button.setEnabled(bool(self._account_email))
            if self._account_email:
                row.set_note(tr("Permanent. Stops every TerraLab plugin."))
                row.setToolTip(tr("Erases your account and its data. Every TerraLab plugin "
                                  "stops, and a paid plan stops renewing.")
                               + "\n" + tr("To confirm, you type your email address again."))
            else:
                row.set_note(tr("After your account loads"))
                row.setToolTip("")
        except RuntimeError:
            pass  # nosec B110
