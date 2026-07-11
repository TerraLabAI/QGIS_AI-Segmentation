"""Account Settings dialog for AI Segmentation plugin.

Mirrors AI Edit's Settings: an account chip (avatar + email + status) with a
quiet inline Sign out link, a product card, a prominent Manage button, and a
discreet legal footer. The activation key lives only in the web dashboard, so
it is intentionally not shown here.
"""
from __future__ import annotations

import os
from datetime import datetime

from qgis.PyQt.QtCore import Qt, QUrl, pyqtSignal
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.activation_manager import (
    get_dashboard_url,
    get_privacy_url,
    get_terms_url,
    get_upgrade_url,
)
from ..core.i18n import tr
from ..workers.generic_request_task import GenericRequestTask
from .ai_segmentation_dockwidget import (
    BRAND_BLUE,
    BRAND_BLUE_HOVER,
    BRAND_GREEN,
    BRAND_GREEN_TEXT,
    BRAND_RED,
)

PRODUCT_ID = "ai-segmentation"
PRODUCT_NAME = "AI Segmentation"

_STATUS_DISPLAY = {
    "active": (tr("Active"), BRAND_GREEN_TEXT),
    "trialing": (tr("Free Trial"), "#f57c00"),
    "canceled": (tr("Canceled"), BRAND_RED),
}

_LINK_BTN = (
    f"QPushButton {{ border: none; color: {BRAND_BLUE}; font-size: 11px;"
    f" text-decoration: underline; padding: 2px 4px; background: transparent; }}"
    f"QPushButton:hover {{ color: {BRAND_BLUE_HOVER}; }}"
)

# The single primary action of the dialog (Upgrade to Pro). Everything else
# is a quiet link or outlined button so the eye lands here first.
_PRIMARY_BTN = (
    f"QPushButton {{ background-color: {BRAND_BLUE}; color: #ffffff;"
    f" border: none; border-radius: 8px; padding: 9px 16px;"
    f" font-size: 12px; font-weight: 600; }}"
    f"QPushButton:hover {{ background-color: {BRAND_BLUE_HOVER}; }}"
)

# Manage account: demoted from a second full-width blue button (it competed
# with Upgrade and the two read as equals) to a quiet blue link in the plan
# card header, symmetric with the Sign out link in the account chip.
_MANAGE_LINK = (
    f"QPushButton {{ border: none; background: transparent; color: {BRAND_BLUE};"
    f" font-size: 11px; font-weight: 600; padding: 2px 4px; }}"
    f"QPushButton:hover {{ color: {BRAND_BLUE_HOVER};"
    f" text-decoration: underline; }}"
)

# Compact sign-out link (sits inside the account chip, not a full-width button).
_SIGNOUT_LINK = (
    "QPushButton { border: none; background: transparent; color: palette(text);"
    " font-size: 11px; text-decoration: underline; padding: 2px 4px; }"
    f"QPushButton:hover {{ color: {BRAND_RED}; }}"
)

# Small outlined secondary button for minor actions inside a card (e.g. "Open
# folder"), distinct from the prominent blue Manage button. Mirrors AI Edit's
# Browse button (_PREF_BTN): buttons inside a card need an explicit background,
# else the card's "QPushButton { background: transparent }" rule breaks native
# rendering. Text uses palette(text), NOT palette(button-text): the QGIS dark
# theme sets ButtonText to black on a dark Button (unreadable black-on-dark).
_SECONDARY_BTN = (
    "QPushButton { background: palette(button); color: palette(text);"
    " border: 1px solid rgba(128,128,128,0.45); border-radius: 5px;"
    " padding: 3px 12px; }"
    "QPushButton:hover { background: rgba(128,128,128,0.18); }"
)

_CARD_STYLE = (
    "QFrame { background: rgba(128,128,128,0.08);"
    " border: 1px solid rgba(128,128,128,0.2);"
    " border-radius: 6px; }"
    "QLabel { background: transparent; border: none; }"
    "QPushButton { background: transparent; }"
)


def _load_account_and_usage(client, auth) -> dict:
    """Fetch account + usage off the main thread for GenericRequestTask.

    Returns the account error dict (with its connectivity code) on failure so
    the task routes it to ``failed`` and the dialog shows the friendly message.
    Usage is optional: a usage error never blocks the dialog.

    Account and usage are fetched CONCURRENTLY (one round-trip, not two
    sequential blocking GETs) so the dialog paints as soon as the slower of the
    two returns instead of their sum.
    """
    try:
        try:
            account, usage = client.get_account_and_usage(auth=auth)
        except Exception:  # nosec B110 -- fall back to sequential if the batch path is unavailable
            account = client.get_account(auth=auth)
            usage = {} if "error" in account else client.get_usage(auth=auth)
    finally:
        # The concurrent path lazily created a private network manager with
        # THIS pooled task thread's affinity; release it here, on its own
        # thread, so a later Retry on a different pool thread never touches
        # it cross-thread and main-thread GC never destroys it from afar.
        try:
            client.release_thread_nam()
        except Exception:  # noqa: BLE001
            pass  # nosec B110
    if "error" in account:
        return account
    if not isinstance(usage, dict) or "error" in usage:
        usage = {}
    return {"account": account, "usage": usage}


class AccountSettingsDialog(QDialog):

    sign_out_requested = pyqtSignal()

    def __init__(self, client, auth, activation_key, parent=None,
                 on_remove_ai_data=None, is_busy_check=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Account Settings"))
        self.setModal(True)
        # Optional plugin callbacks. on_remove_ai_data() -> (ok, message)
        # deletes the local venv + weights + key + settings; is_busy_check()
        # -> bool tells us not to offer removal while an install/run is live.
        # Both absent (MCP/headless construction) => the removal action is
        # simply not shown.
        self._on_remove_ai_data = on_remove_ai_data
        self._is_busy_check = is_busy_check
        # Wide enough for the Dependencies + Privacy cards to sit side by side;
        # the old 400-500 window forced everything into one tall column.
        self.setMinimumWidth(600)
        self.setMaximumWidth(720)

        # ``activation_key`` is accepted for signature compatibility but no
        # longer displayed; key management happens on the web dashboard.
        del activation_key
        self._worker = None

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(16, 16, 16, 12)
        self._layout.setSpacing(12)

        self._loading_label = QLabel(tr("Loading account info..."))
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading_label.setStyleSheet("color: palette(text); padding: 16px;")
        self._layout.addWidget(self._loading_label)

        self._error_widget = QWidget()
        error_layout = QVBoxLayout(self._error_widget)
        error_layout.setContentsMargins(0, 0, 0, 0)
        error_layout.setSpacing(8)
        self._error_label = QLabel()
        self._error_label.setWordWrap(True)
        self._error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._error_label.setStyleSheet(f"color: {BRAND_RED}; padding: 12px;")
        error_layout.addWidget(self._error_label)
        self._retry_btn = QPushButton(tr("Retry"))
        self._retry_btn.setMaximumWidth(100)
        self._retry_btn.clicked.connect(self._fetch_account)
        # Billing-problem CTA: shown only when the server reports the
        # subscription is not active (payment failed / lapsed). Opens the
        # account page where the user can update their payment method or
        # review their plan. Hidden for ordinary (network) errors.
        self._error_manage_btn = QPushButton(tr("Update payment method"))
        self._error_manage_btn.setStyleSheet(_PRIMARY_BTN)
        self._error_manage_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._error_manage_btn.setMinimumHeight(36)
        self._error_manage_btn.setToolTip(
            tr("Opens your terra-lab.ai account in the browser."))
        self._error_manage_btn.clicked.connect(self._open_dashboard)
        self._error_manage_btn.setVisible(False)
        self._error_sign_out_btn = QPushButton(tr("Sign out"))
        self._error_sign_out_btn.setStyleSheet(_LINK_BTN)
        self._error_sign_out_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._error_sign_out_btn.clicked.connect(self._on_sign_out)
        retry_row = QHBoxLayout()
        retry_row.addStretch()
        retry_row.addWidget(self._retry_btn)
        retry_row.addStretch()
        error_layout.addLayout(retry_row)
        error_layout.addWidget(self._error_manage_btn)
        signout_row = QHBoxLayout()
        signout_row.addStretch()
        signout_row.addWidget(self._error_sign_out_btn)
        signout_row.addStretch()
        error_layout.addLayout(signout_row)
        self._error_widget.setVisible(False)
        self._layout.addWidget(self._error_widget)

        self._content_widget = QWidget()
        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(10)
        self._content_widget.setVisible(False)
        self._layout.addWidget(self._content_widget)

        self._layout.addStretch()

        self._client = client
        self._auth = auth
        self._fetch_account()

    def _fetch_account(self):
        self._loading_label.setVisible(True)
        self._error_widget.setVisible(False)
        self._content_widget.setVisible(False)

        from qgis.core import QgsApplication

        # Drop any previous in-flight load (Retry) so its result can't land late.
        self._cancel_worker()
        client, auth = self._client, self._auth
        self._worker = GenericRequestTask(
            tr("Loading account info..."),
            lambda: _load_account_and_usage(client, auth),
            hidden=True,
        )
        self._worker.succeeded.connect(self._on_loaded)
        self._worker.failed.connect(self._on_failed)
        QgsApplication.taskManager().addTask(self._worker)

    def _cancel_worker(self):
        """Disconnect then cancel the loader task so a late result never fires
        into a closed dialog. We never force-kill a thread mid network-call,
        which can corrupt Qt's socket state and crash QGIS; cancellation is
        cooperative and the task manager drains run() on its own."""
        if self._worker is None:
            return
        try:
            self._worker.succeeded.disconnect()
            self._worker.failed.disconnect()
        except (RuntimeError, TypeError):  # nosec B110
            pass
        try:
            self._worker.cancel()
        except Exception:  # nosec B110
            pass
        self._worker = None

    def _on_loaded(self, data: dict):
        self._loading_label.setVisible(False)
        self._error_widget.setVisible(False)

        # Support combined {"account": ..., "usage": ...} from the updated worker
        # as well as the legacy flat account dict format.
        account_data = data.get("account", data)
        usage_data = data.get("usage", {})

        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._content_layout.addWidget(self._build_account_card(account_data))

        sub = self._find_subscription(account_data)
        if sub:
            self._content_layout.addWidget(self._build_subscription_card(sub, usage_data))

        # Dependencies and Privacy are secondary, same-weight info: side by
        # side they read as one quiet row instead of extending the tower of
        # full-width cards. Wrapped in a QWidget (not a bare layout) so the
        # clear loop above, which deletes item.widget(), removes it on reload.
        two_up_row = QWidget()
        two_up = QHBoxLayout(two_up_row)
        two_up.setContentsMargins(0, 0, 0, 0)
        two_up.setSpacing(10)
        two_up.addWidget(self._build_dependencies_card(), 1)
        two_up.addWidget(self._build_privacy_card(), 1)
        self._content_layout.addWidget(two_up_row)

        # Discreet footer: thin top separator, small muted Terms / Privacy links.
        footer = QFrame()
        footer.setObjectName("legalFooter")
        footer.setStyleSheet(
            "QFrame#legalFooter { border: none;"
            " border-top: 1px solid rgba(127,127,127,0.18); }"
        )
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 8, 0, 0)
        footer_layout.setSpacing(0)
        footer_layout.addStretch()
        legal_label = QLabel(
            f'<a href="{get_terms_url()}" style="color: rgba(128,128,128,0.85);'
            f' text-decoration: none;">{tr("Terms")}</a>'
            f' <span style="color: rgba(128,128,128,0.5);">·</span> '
            f'<a href="{get_privacy_url()}" style="color: rgba(128,128,128,0.85);'
            f' text-decoration: none;">{tr("Privacy")}</a>'
        )
        legal_label.setOpenExternalLinks(True)
        legal_label.setStyleSheet("font-size: 10px;")
        footer_layout.addWidget(legal_label)
        footer_layout.addStretch()
        self._content_layout.addWidget(footer)

        self._content_widget.setVisible(True)
        self.adjustSize()

    @staticmethod
    def _format_reset_date(iso: str | None) -> str:
        """Parse an ISO date string to a full human-friendly form
        ("July 23, 2026", mirrors AI Edit's reset line).

        Falls back to the raw string if parsing fails.
        """
        if not iso:
            return ""
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        except ValueError:
            return iso
        return dt.strftime("%B %d, %Y")

    def _on_failed(self, message: str, code: str = ""):
        self._loading_label.setVisible(False)
        # A payment failure / lapsed subscription authenticates locally (the key
        # is still stored) but every server call is rejected. The raw
        # "Subscription expired" + Retry read as a mysterious outage; show a
        # clear billing message and route the user to their account page to
        # update their payment method instead. The server currently collapses
        # past_due / unpaid / canceled into one code; the finer status is an
        # action item on the server side.
        if code == "SUBSCRIPTION_INACTIVE":
            self._error_label.setText(tr(
                "There's a problem with your subscription. Your last payment "
                "may have failed. Open your account to update your payment "
                "method or review your plan."))
            self._retry_btn.setVisible(False)
            self._error_manage_btn.setVisible(True)
        else:
            self._error_label.setText(message)
            self._retry_btn.setVisible(True)
            self._error_manage_btn.setVisible(False)
        self._error_widget.setVisible(True)
        self._content_widget.setVisible(False)

    @staticmethod
    def _find_subscription(data: dict) -> dict | None:
        # Prefer the paid plan when the account holds it, so the card shows Pro
        # status/credits rather than the free row that may also exist.
        subs = data.get("subscriptions", [])
        for pid in (f"{PRODUCT_ID}-pro", PRODUCT_ID):
            for s in subs:
                if s.get("product_id") == pid:
                    return s
        return None

    def _build_account_card(self, data: dict) -> QFrame:
        card = QFrame()
        card.setStyleSheet(_CARD_STYLE)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        email = data.get("email", "-")

        # A real account "chip": round avatar (first letter) + email + status.
        chip = QFrame()
        chip.setStyleSheet(
            "QFrame { background: palette(base);"
            " border: 1px solid rgba(128,128,128,0.25); border-radius: 8px; }"
            "QLabel { background: transparent; border: none; }"
        )
        chip_row = QHBoxLayout(chip)
        chip_row.setContentsMargins(12, 10, 12, 10)
        chip_row.setSpacing(11)

        avatar = QLabel(email[:1].upper() if email and email != "-" else "?")
        avatar.setFixedSize(38, 38)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        avatar.setStyleSheet(
            f"background: {BRAND_GREEN}; color: #14210A; border-radius: 19px;"
            " font-size: 17px; font-weight: 700;"
        )
        chip_row.addWidget(avatar)

        id_col = QVBoxLayout()
        id_col.setSpacing(2)
        email_val = QLabel(email)
        email_val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        email_val.setStyleSheet(
            "font-size: 13px; font-weight: 600; color: palette(text);"
        )
        id_col.addWidget(email_val)
        status_lbl = QLabel("✓ " + tr("Connected"))
        status_lbl.setStyleSheet(
            f"font-size: 11px; font-weight: 600; color: {BRAND_GREEN_TEXT};"
        )
        id_col.addWidget(status_lbl)
        chip_row.addLayout(id_col, 1)

        # Sign out is a quiet inline link on the right of the chip, not a heavy
        # full-width button below the email.
        sign_out_btn = QPushButton(tr("Sign out"))
        sign_out_btn.setStyleSheet(_SIGNOUT_LINK)
        sign_out_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        sign_out_btn.clicked.connect(self._on_sign_out)
        chip_row.addWidget(sign_out_btn, 0, Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(chip)
        return card

    def _open_dashboard(self):
        QDesktopServices.openUrl(QUrl(get_dashboard_url()))

    def _build_subscription_card(self, sub: dict, usage: dict | None = None) -> QFrame:
        """Plan status, credits (Pro) or lifetime free allowance (Free), with a
        single primary Upgrade CTA. Manage account lives in the card header as
        a quiet link so it never competes with Upgrade for attention."""
        if usage is None:
            usage = {}

        card = QFrame()
        card.setStyleSheet(_CARD_STYLE)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 10, 12, 12)
        card_layout.setSpacing(6)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        title = QLabel(f"<b>{PRODUCT_NAME}</b>")
        title.setStyleSheet("font-size: 13px; color: palette(text);")
        header.addWidget(title, 1)
        manage_btn = QPushButton(tr("Manage account") + " ↗")
        manage_btn.setStyleSheet(_MANAGE_LINK)
        manage_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        manage_btn.setToolTip(
            tr("Opens your terra-lab.ai dashboard in the browser."))
        manage_btn.clicked.connect(self._open_dashboard)
        header.addWidget(manage_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        card_layout.addLayout(header)

        status = sub.get("status", "active")
        status_text, status_color = _STATUS_DISPLAY.get(
            status, (status.title(), BRAND_RED)
        )

        # Determine subscriber status from usage data; fall back to the
        # subscription plan field from the account endpoint.
        is_subscriber = usage.get("is_subscriber", False)
        if not is_subscriber:
            # The account endpoint uses plan="pro" / "free"; the usage endpoint
            # uses is_free_tier. Use both sources for maximum compatibility.
            sub_plan = sub.get("plan", "free")
            is_subscriber = sub_plan == "pro" and sub.get("free_detections_remaining") is not True
            is_free_tier = usage.get("is_free_tier", True)
            if not is_free_tier:
                is_subscriber = True

        remaining = usage.get("remaining_credits", None)
        total = usage.get("total_credits", 10000 if is_subscriber else 300)
        reset_date = usage.get("reset_date") or usage.get("period_end")
        free_left = usage.get("free_detections_remaining") or sub.get("free_detections_remaining")

        # One quiet status line under the header (the old two-row labeled grid
        # read as a form; this is a summary, not a form).
        plan_name = tr("Pro plan") if is_subscriber else tr("Free plan")
        plan_status = QLabel(
            f"{plan_name} · <span style='color:{status_color};'>{status_text}</span>"
        )
        plan_status.setStyleSheet("font-size: 12px; color: palette(text);")
        card_layout.addWidget(plan_status)

        bar: QProgressBar | None = None
        reset_str = ""
        if is_subscriber and remaining is not None:
            # Credits line for Pro subscribers. Lime fill for the bar; the
            # darker green tone for the number so it stays AA-readable on the
            # light dialog (mirrors AI Edit); red across the board once
            # exhausted.
            reset_str = self._format_reset_date(reset_date)
            fill = BRAND_GREEN if remaining > 0 else BRAND_RED
            text_color = BRAND_GREEN_TEXT if remaining > 0 else BRAND_RED
            credits_text = tr("{remaining} / {total} credits").format(
                remaining=f"{remaining:,}", total=f"{total:,}"
            )
            credits_lbl = QLabel(credits_text)
            credits_lbl.setStyleSheet(f"font-size: 12px; color: {text_color};")
            card_layout.addWidget(credits_lbl)
            bar = self._credits_bar(remaining, total, fill)
        elif not is_subscriber and free_left is not None:
            # Free-taste line (one-time lifetime allowance)
            fill = BRAND_GREEN if free_left > 0 else BRAND_RED
            text_color = BRAND_GREEN_TEXT if free_left > 0 else BRAND_RED
            # The lifetime total is a newer usage field; only draw the gauge
            # and the "of N" phrasing when the server reports it (older
            # responses omit it).
            free_total = usage.get("free_detections_total")
            if free_total:
                free_text = tr("{n} of {total} free detections left").format(
                    n=free_left, total=int(free_total))
                bar = self._credits_bar(free_left, int(free_total), fill)
            else:
                free_text = tr("{n} free detection(s) remaining (lifetime)").format(
                    n=free_left)
            free_lbl = QLabel(free_text)
            free_lbl.setStyleSheet(f"font-size: 12px; color: {text_color};")
            card_layout.addWidget(free_lbl)

        if bar is not None:
            card_layout.addWidget(bar)

        # A small reset note (paid renewal date) above the action buttons,
        # replacing the old inline "(resets Jul 3)" suffix on the credits row.
        if is_subscriber and reset_str:
            reset_row = QHBoxLayout()
            reset_row.setContentsMargins(0, 2, 0, 0)
            reset_label = QLabel(tr("Resets {date}").format(date=reset_str))
            reset_label.setStyleSheet("font-size: 10px; color: palette(text);")
            reset_row.addWidget(reset_label)
            reset_row.addStretch()
            card_layout.addLayout(reset_row)

        # Upgrade CTA (Free accounts only): the one primary button of the whole
        # dialog, with a small benefit caption under it. Pro accounts have no
        # big button here; the header's Manage link covers plan management.
        if not is_subscriber:
            card_layout.addSpacing(6)
            upgrade_btn = QPushButton(tr("Upgrade to Pro"))
            upgrade_btn.setStyleSheet(_PRIMARY_BTN)
            upgrade_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            upgrade_btn.setMinimumHeight(36)
            upgrade_btn.setToolTip(
                tr("Opens terra-lab.ai in your browser."))
            upgrade_btn.clicked.connect(
                lambda: QDesktopServices.openUrl(QUrl(get_upgrade_url()))
            )
            card_layout.addWidget(upgrade_btn)
            benefit = QLabel(tr("10,000 credits every month. Cancel anytime."))
            benefit.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            benefit.setStyleSheet(
                "font-size: 10px; color: rgba(128,128,128,0.9);")
            card_layout.addWidget(benefit)

        contact = QLabel(
            tr("Team or organization?") + " "
            + tr("Write to us:") + " <b>yvann.barbot@terra-lab.ai</b>"
        )
        contact.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        contact.setWordWrap(True)
        contact.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        contact.setStyleSheet("font-size: 10px; color: rgba(128,128,128,0.9);")
        card_layout.addWidget(contact)

        return card

    def _build_dependencies_card(self) -> QFrame:
        """Local AI dependencies: where they live, how big, and an Open button.

        The isolated venv and model weights (~1 GB) live outside QGIS under the
        user home. Surfacing the path + size + an Open button answers the common
        support needs (disk space, antivirus exclusions, manual cleanup) without
        the user having to hunt for a hidden folder.
        """
        from ..core.venv_manager import CACHE_DIR, VENV_DIR

        card = QFrame()
        card.setStyleSheet(_CARD_STYLE)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        title = QLabel(f"<b>{tr('Dependencies')}</b>")
        title.setStyleSheet("font-size: 13px; color: palette(text);")
        layout.addWidget(title)

        caption = QLabel(tr("Local AI model files stored on this computer."))
        caption.setWordWrap(True)
        caption.setStyleSheet("font-size: 11px; color: palette(text);")
        layout.addWidget(caption)

        installed = os.path.isdir(VENV_DIR)
        size_text = self._format_dir_size(CACHE_DIR) if installed else tr("Not installed")
        info = QLabel(f"{tr('On disk')}: {size_text}")
        info.setStyleSheet("font-size: 11px; color: rgba(128,128,128,0.9);")
        layout.addWidget(info)

        path_lbl = QLabel(CACHE_DIR)
        path_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        path_lbl.setWordWrap(True)
        path_lbl.setStyleSheet("font-size: 10px; color: rgba(128,128,128,0.7);")
        layout.addWidget(path_lbl)

        open_btn = QPushButton(tr("Open folder"))
        open_btn.setStyleSheet(_SECONDARY_BTN)
        open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_btn.clicked.connect(lambda: self._open_install_folder(CACHE_DIR))
        row = QHBoxLayout()
        row.setContentsMargins(0, 2, 0, 0)
        row.addStretch()
        row.addWidget(open_btn)
        layout.addLayout(row)

        # Remove-downloaded-data action: frees the multi-GB venv + weights and
        # clears the stored key + settings (so a paying key never lingers on a
        # shared machine after the user thinks they removed the product). Only
        # offered when the plugin wired the callback in; disabled mid-install.
        if self._on_remove_ai_data is not None:
            busy = False
            if self._is_busy_check is not None:
                try:
                    busy = bool(self._is_busy_check())
                except Exception:  # nosec B110
                    busy = False
            self._remove_btn = QPushButton(tr("Remove downloaded AI data"))
            self._remove_btn.setStyleSheet(_LINK_BTN)
            self._remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._remove_btn.setEnabled(not busy)
            if busy:
                self._remove_btn.setToolTip(
                    tr("Available once the current install or detection finishes."))
            self._remove_btn.clicked.connect(self._on_remove_ai_data_clicked)
            remove_row = QHBoxLayout()
            remove_row.setContentsMargins(0, 2, 0, 0)
            remove_row.addWidget(self._remove_btn)
            remove_row.addStretch()
            layout.addLayout(remove_row)

        return card

    def _on_remove_ai_data_clicked(self):
        """Confirm, then ask the plugin to delete the local venv + weights and
        clear the stored key + settings. Signs the user out on success."""
        from qgis.PyQt.QtWidgets import QMessageBox

        # Re-check busy at click time: an install can finish (or start) while
        # the modal dialog's event loop is running.
        if self._is_busy_check is not None:
            try:
                if self._is_busy_check():
                    QMessageBox.information(
                        self, tr("AI Segmentation"),
                        tr("An install or detection is still running. Wait for "
                           "it to finish, then try again."))
                    return
            except Exception:  # nosec B110
                pass

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(tr("Remove downloaded AI data"))
        box.setText(tr("Remove the downloaded AI data from this computer?"))
        box.setInformativeText(tr(
            "This deletes the local AI model files, signs you out, and resets "
            "the plugin. Your account and credits are not affected. Manual mode "
            "will download the files again next time you use it."))
        box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        if box.exec() != QMessageBox.StandardButton.Yes:
            return

        if hasattr(self, "_remove_btn"):
            self._remove_btn.setEnabled(False)
            self._remove_btn.setText(tr("Removing..."))
        try:
            ok, message = self._on_remove_ai_data()
        except Exception:  # nosec B110
            ok, message = False, tr("Could not remove the AI data. Try again.")

        if ok:
            QMessageBox.information(self, tr("AI Segmentation"), message)
            # The key is cleared and the user signed out: close the dialog.
            self.accept()
        else:
            QMessageBox.warning(self, tr("AI Segmentation"), message)
            if hasattr(self, "_remove_btn"):
                self._remove_btn.setEnabled(True)
                self._remove_btn.setText(tr("Remove downloaded AI data"))

    def _build_privacy_card(self) -> QFrame:
        """Anonymous usage telemetry with a clear, ON-by-default opt-out.

        Flips the shared TerraLab/telemetry_enabled flag (telemetry.py), so
        turning it off here also silences the sibling AI Edit plugin. Metrics
        are anonymous and carry no imagery, prompts, coordinates, layers, or
        project content.
        """
        from ..core.telemetry import is_telemetry_enabled, set_telemetry_enabled

        card = QFrame()
        card.setStyleSheet(_CARD_STYLE)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        title = QLabel(f"<b>{tr('Privacy')}</b>")
        title.setStyleSheet("font-size: 13px; color: palette(text);")
        layout.addWidget(title)

        self._telemetry_checkbox = QCheckBox(tr("Share anonymous usage statistics"))
        self._telemetry_checkbox.setChecked(is_telemetry_enabled())
        self._telemetry_checkbox.setCursor(Qt.CursorShape.PointingHandCursor)
        self._telemetry_checkbox.setStyleSheet("font-size: 12px; color: palette(text);")
        self._telemetry_checkbox.toggled.connect(set_telemetry_enabled)
        layout.addWidget(self._telemetry_checkbox)

        caption = QLabel(
            tr(
                "Helps us fix bugs faster. Never includes your data, layers or "
                "coordinates."
            )
        )
        caption.setWordWrap(True)
        caption.setStyleSheet("font-size: 11px; color: rgba(128,128,128,0.9);")
        layout.addWidget(caption)

        # Restore the dismissed in-app guidance tips. A quiet link-style action,
        # since it is a rare "I want the tips back" click. reset_hints() re-shows
        # any tip live if the dock is open, so the effect is immediate.
        guidance_row = QHBoxLayout()
        guidance_row.setContentsMargins(0, 4, 0, 0)
        self._reset_hints_btn = QPushButton(tr("Show guidance tips again"))
        self._reset_hints_btn.setStyleSheet(_LINK_BTN)
        self._reset_hints_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reset_hints_btn.clicked.connect(self._on_reset_hints)
        guidance_row.addWidget(self._reset_hints_btn)
        guidance_row.addStretch()
        layout.addLayout(guidance_row)

        return card

    def _on_reset_hints(self):
        from .dock.guidance import reset_hints

        reset_hints()
        if hasattr(self, "_reset_hints_btn"):
            self._reset_hints_btn.setText(tr("Guidance tips restored"))
            self._reset_hints_btn.setEnabled(False)

    @staticmethod
    def _open_install_folder(path: str):
        # The folder may not exist yet (deps never installed); fall back to the
        # nearest existing parent so the button never silently does nothing.
        target = path
        while target and not os.path.isdir(target):
            parent = os.path.dirname(target)
            if parent == target:
                break
            target = parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(target))

    @staticmethod
    def _format_dir_size(path: str) -> str:
        total = 0
        try:
            for root, _dirs, files in os.walk(path):
                for name in files:
                    try:
                        total += os.path.getsize(os.path.join(root, name))
                    except OSError:  # nosec B112
                        continue
        except OSError:
            return "-"
        mb = total / (1024 * 1024)
        if mb >= 1024:
            return f"{mb / 1024:.1f} GB"
        return f"{mb:.0f} MB"

    # -- Helpers --

    @staticmethod
    def _credits_bar(remaining: int, total: int, fill: str) -> QProgressBar:
        """Slim credits gauge under the credits row (AI Edit parity):
        6px tall, no text, brand-colored chunk on a faint neutral track."""
        bar = QProgressBar()
        bar.setRange(0, max(int(total), 1))
        bar.setValue(max(0, min(int(remaining), int(total))))
        bar.setTextVisible(False)
        bar.setFixedHeight(6)
        bar.setStyleSheet(
            "QProgressBar { background: rgba(128,128,128,0.15);"
            " border: none; border-radius: 3px; }"
            f"QProgressBar::chunk {{ background: {fill}; border-radius: 3px; }}"
        )
        return bar

    def _on_sign_out(self):
        from qgis.PyQt.QtWidgets import QMessageBox

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(tr("Sign out"))
        box.setText(tr("Sign out of AI Segmentation?"))
        box.setInformativeText(tr("You can sign back in anytime from QGIS."))
        box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        box.setDefaultButton(QMessageBox.StandardButton.No)
        if box.exec() != QMessageBox.StandardButton.Yes:
            return
        self.sign_out_requested.emit()
        self.accept()

    def done(self, result):  # noqa: N802 - Qt signature
        # accept()/reject() (Sign out, OK, Esc) dismiss the dialog without a
        # closeEvent, so cancel the in-flight loader here too rather than let it
        # complete with now-stale auth into a dismissed dialog.
        self._cancel_worker()
        super().done(result)

    def closeEvent(self, event):
        # The loader is a QgsTask now: no thread to wait on or terminate.
        # Disconnect + cancel so a late result can't fire into the closing
        # dialog; the task manager drains run() on its own. The old
        # wait(6000)+terminate() path crashed QGIS when the network was wedged.
        self._cancel_worker()
        super().closeEvent(event)
