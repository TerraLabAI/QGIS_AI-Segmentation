"""Account Settings dialog for AI Segmentation plugin.

Mirrors AI Edit's Settings: an account chip (avatar + email + status) with a
quiet inline Sign out link, a product card, a prominent Manage button, and a
discreet legal footer. The activation key lives only in the web dashboard, so
it is intentionally not shown here.
"""
from __future__ import annotations

import os
from typing import NamedTuple

from qgis.PyQt.QtCore import QRect, Qt, QTimer, QUrl, pyqtSignal
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..core.activation_manager import (
    PRODUCT_ID,
    get_dashboard_url,
    get_privacy_url,
    get_terms_url,
    get_upgrade_url,
)
from ..core.i18n import tr
from ..core.qt_compat import safe_disconnect
from ..core.server_dials import dial_copy
from ..workers.generic_request_task import GenericRequestTask
from .ai_segmentation_dockwidget import (
    BRAND_BLUE,
    BRAND_BLUE_HOVER,
    BRAND_GREEN,
    BRAND_GREEN_TEXT,
    BRAND_RED,
)
from .dock.font_scale import scale_px_length
from .dock.ui_refresh import format_quota_count
from .dock.upsell_card import UpsellCard

PRODUCT_NAME = "AI Segmentation"

# Measured install size per path, kept for the session: the walk costs seconds
# and the value barely moves, so re-opening Settings or pressing Retry must not
# pay for it again.
_DIR_SIZE_CACHE: dict[str, str] = {}

# Last resort for the removal: the window refuses to close while one runs, so a
# worker that never reports back (a thread the OS would not start, a signal
# lost with the plugin) leaves a modal dialog that only killing QGIS can shut.
# Far above any plausible delete, so a slow disk always wins the race.
_REMOVAL_WATCHDOG_MS = 600_000

# Height left free around the window when the cards are taller than the screen.
# The title bar is measured on its own, so this covers the taskbar and a strip
# of desktop, and keeps the window from looking wedged between the two edges.
_SCREEN_MARGIN_PX = 48

# Failure codes that mean the request never reached a server, so the card can
# tell "you are offline" from "something else went wrong". Anything outside it
# gets the general sentence.
_ACCOUNT_OFFLINE_CODES = frozenset({
    "NO_INTERNET", "DNS_ERROR", "CONNECTION_REFUSED", "PROXY_ERROR",
    "TIMEOUT", "SSL_ERROR",
})

_STATUS_DISPLAY = {
    "active": (tr("Active"), BRAND_GREEN_TEXT),
    # Design-system warning amber (taxonomy _MSG_TINTS['warning'] hue). No named
    # hex constant exists in styles.py for it, so the literal is used here.
    "trialing": (tr("Free trial"), "#f5a623"),
    "canceled": (tr("Cancelled"), BRAND_RED),
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


# Last-resort denominator for a paid plan when neither the usage response nor
# the account row carries one. Never used to grant or spend anything.
_PRO_MONTHLY_CREDITS_FALLBACK = 5000


class PlanCredits(NamedTuple):
    """What the plan card draws: who the user is, and what is left to spend."""

    is_subscriber: bool
    remaining: int | None
    total: int | None
    reset_date: str | None


def _as_int(value) -> int | None:
    """The value as an int, or None when the server sent null or a word."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_is_subscriber(usage: dict, sub: dict) -> bool:
    """Paid plan or free plan, from whichever signal the server sent.

    ``usage["plan"]`` is deliberately not read: the server hardcodes it to
    "pro" for every caller, so it says nothing about this account.
    """
    if "is_subscriber" in usage:
        return bool(usage["is_subscriber"])
    if "is_free_tier" in usage:
        return not bool(usage["is_free_tier"])
    return str(sub.get("plan", "")).lower() == "pro"


# Two server spellings reach this dialog and both stay live: older responses
# count images (images_used / images_limit / is_free_tier / period_end), newer
# ones count credits (remaining_credits / total_credits / is_subscriber /
# reset_date). Read the newer names first, derive them from the older ones when
# they are absent, and fall back to the account's own subscription row so the
# card still fills in when the usage call came back empty.
def resolve_plan_credits(usage: dict, sub: dict) -> PlanCredits:
    is_subscriber = _resolve_is_subscriber(usage, sub)
    remaining = _as_int(usage.get("remaining_credits"))
    total = _as_int(usage.get("total_credits"))
    if remaining is None or total is None:
        for limit, used in (
            (usage.get("images_limit"), usage.get("images_used")),
            (sub.get("quota_limit"), sub.get("usage_this_month")),
        ):
            limit_int = _as_int(limit)
            if limit_int is None:
                continue
            if total is None:
                total = limit_int
            if remaining is None:
                remaining = max(0, limit_int - (_as_int(used) or 0))
            break
    if total is None:
        # The free-tier total is a server dial, so read the getter instead of
        # restating the number here: a fleet-wide change must move the gauge.
        from ..core.detection_policy import free_monthly_allowance

        total = _PRO_MONTHLY_CREDITS_FALLBACK if is_subscriber else free_monthly_allowance()
    reset_date = (usage.get("reset_date") or usage.get("period_end")
                  or sub.get("current_period_end"))
    return PlanCredits(is_subscriber, remaining, total, reset_date)


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
    if not isinstance(account, dict):
        return {"error": "Invalid server response", "code": "SERVER_ERROR"}
    if "error" in account:
        return account
    if not isinstance(usage, dict) or "error" in usage:
        usage = {}
    return {"account": account, "usage": usage}


class AccountSettingsDialog(QDialog):

    sign_out_requested = pyqtSignal()
    # The usage payload this window just read, handed to whoever else shows a
    # balance. Without it the dock's footer ring keeps its own older figure and
    # the two surfaces contradict each other in front of the user.
    usage_loaded = pyqtSignal(dict)

    def __init__(self, client, auth, activation_key, parent=None,
                 on_remove_ai_data=None, is_busy_check=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Account Settings"))
        self.setModal(True)
        # Optional plugin callbacks. on_remove_ai_data(on_progress, on_finished)
        # -> (started, message) starts deleting the local venv + weights + key
        # + settings on a worker thread and calls the two back on the GUI
        # thread; is_busy_check() -> bool tells us not to offer removal while an
        # install/run is live. Both absent (MCP/headless construction) => the
        # removal action is simply not shown.
        self._on_remove_ai_data = on_remove_ai_data
        self._is_busy_check = is_busy_check
        # True between the confirmed removal and its result: the window refuses
        # to close then, so nobody is left with a half-deleted environment and
        # no report of what survived.
        self._removal_running = False
        # Bumped per removal so the watchdog armed for one can never end a
        # later one (the user can retry after a failure).
        self._removal_generation = 0
        self._remove_status = None
        self._remove_btn = None
        # Wide enough for the Dependencies + Privacy cards to sit side by side;
        # the old 400-500 window forced everything into one tall column.
        self.setMinimumWidth(600)
        self.setMaximumWidth(720)

        # ``activation_key`` is accepted for signature compatibility but no
        # longer displayed; key management happens on the web dashboard.
        del activation_key
        self._worker = None
        self._size_task = None
        self._size_label = None
        # The account picture: the badge currently on screen, the one loader
        # allowed per dialog open, and whether it has already been started (a
        # Retry rebuilds the card but must not buy a second request).
        self._avatar_label = None
        self._avatar_loader = None
        self._avatar_requested = False

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
        self._error_manage_btn.clicked.connect(
            lambda: self._open_dashboard("error_card"))
        self._error_manage_btn.setVisible(False)
        self._error_sign_out_btn = QPushButton(tr("Sign out"))
        self._error_sign_out_btn.setStyleSheet(_LINK_BTN)
        self._error_sign_out_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._error_sign_out_btn.clicked.connect(
            lambda: self._on_sign_out("error_card"))
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
        # The cards go in a scroll area so the window can always be made
        # shorter than the screen. Without it the layout minimum is the only
        # height there is, and on a laptop at 150% text scaling the Privacy
        # card and both footer links sit below the bottom edge, where no
        # scrollbar and no window drag can reach them.
        self._content_scroll = QScrollArea()
        self._content_scroll.setWidgetResizable(True)
        self._content_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._content_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._content_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollArea > QWidget > QWidget { background: transparent; }")
        self._content_scroll.setWidget(self._content_widget)
        self._content_scroll.setVisible(False)
        # The scroll area holds the only stretch factor in the window, so every
        # pixel the dialog has past its margins goes to the cards. A trailing
        # addStretch() used to hold that factor, and a box layout hands spare
        # height to the items carrying one first: the cards stayed at the scroll
        # area's own hint (capped at 24 text lines) behind a scrollbar while the
        # empty spacer took the rest of the window.
        self._layout.addWidget(self._content_scroll, 1)

        self._client = client
        self._auth = auth
        self._fetch_account()

    def _fetch_account(self):
        self._loading_label.setVisible(True)
        self._error_widget.setVisible(False)
        self._content_scroll.setVisible(False)

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
        # The size measurement belongs to the card the loader rebuilds, so it
        # follows the same lifetime.
        self._cancel_size_task()
        if self._worker is None:
            return
        # One guard per signal: batched, a raise on `succeeded` left `failed`
        # connected and a late failure still fired into the closed dialog.
        safe_disconnect(self._worker, "succeeded")
        safe_disconnect(self._worker, "failed")
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
        # Emitted before the cards are built, so a slow rebuild never delays
        # the figure reaching the dock. An empty dict means the usage call
        # failed (the window survives that), and carries nothing to share.
        if isinstance(usage_data, dict) and usage_data:
            self.usage_loaded.emit(usage_data)

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

        self._content_scroll.setVisible(True)
        from .dock.font_scale import apply_font_scale_to_tree

        apply_font_scale_to_tree(self)
        self.adjustSize()
        self._fit_to_screen()

    def _fit_to_screen(self) -> None:
        """Open at the height the cards need, capped by the screen, centred.

        The account arrives after the window is already up, so the layout
        grows downward from where an almost empty box was centred. Asking the
        window to adjust is not enough: a scroll area asks for a height of its
        own, capped at 24 text lines, which has nothing to do with how many
        cards it holds, and the user reads the first two behind a scrollbar.
        So measure the cards and open on them. How many there are moves with
        the account (Pro, free, an install on disk or not), hence a measure
        rather than a number.

        Clamped against the screen the window is actually on, not the main
        one: on a two-monitor desk those differ, and clamping to the wrong one
        either does nothing or shrinks a window that had room. Past the cap
        the scroll area takes over, as before.
        """
        available = self._available_screen_rect()
        if available is None:
            return
        width = min(self.width(), available.width())
        # resize() sets the client area, so the title bar and the borders come
        # off the cap; without that the bottom card lands under the taskbar.
        frame_extra = max(0, self.frameGeometry().height() - self.height())
        cap = available.height() - frame_extra - _SCREEN_MARGIN_PX
        height = max(self.minimumSizeHint().height(),
                     min(self._height_for_cards(width), cap))
        if height != self.height() or width != self.width():
            self.resize(width, height)
        # Centred on the window that opened it, so it lands over QGIS rather
        # than in the middle of a screen QGIS may be nowhere near. The screen
        # work area still bounds it, so it can never open off the desktop.
        frame = self.frameGeometry()
        frame.moveCenter(self._centre_anchor(available))
        left = min(max(available.left(), frame.left()),
                   max(available.left(), available.right() - frame.width()))
        top = min(max(available.top(), frame.top()),
                  max(available.top(), available.bottom() - frame.height()))
        self.move(left, top)

    def _centre_anchor(self, available: QRect):
        """The point this window centres on: the parent window, else the screen."""
        try:
            parent = self.parentWidget()
            if parent is not None and parent.isVisible():
                centre = parent.frameGeometry().center()
                if available.contains(centre):
                    return centre
        except (AttributeError, RuntimeError):
            pass  # nosec B110 -- placement is best-effort
        return available.center()

    def _available_screen_rect(self) -> QRect | None:
        """Work area of the screen this window sits on, None when Qt has none.

        A window Qt has not placed yet reports no screen, and an offscreen
        platform reports an empty rectangle. Either way there is no ruler to
        size against, and the caller leaves the window alone.
        """
        try:
            screen = self.screen()
        except (AttributeError, RuntimeError):
            screen = None
        if screen is None:
            from qgis.PyQt.QtGui import QGuiApplication

            screen = QGuiApplication.primaryScreen()
        if screen is None:
            return None
        available = screen.availableGeometry()
        if available.isEmpty():
            return None
        return available

    def _height_for_cards(self, width: int) -> int:
        """Client height that shows every card at once, margins included.

        A wrapped label only knows its height once it knows its width, so the
        cards are measured at the width the viewport will have, not at the one
        they would pick for themselves.
        """
        self._layout.activate()
        margins = self._layout.contentsMargins()
        border = self._content_scroll.frameWidth() * 2
        viewport_width = max(
            1, width - margins.left() - margins.right() - border)
        inner = self._content_widget
        needed = inner.sizeHint().height()
        if inner.hasHeightForWidth():
            needed = max(needed, inner.heightForWidth(viewport_width))
        return needed + margins.top() + margins.bottom() + border

    def _on_failed(self, message: str, code: str = ""):
        self._loading_label.setVisible(False)
        # The client's own text can be an HTTP status, raw server output or a
        # Python exception string. None of the three tells the user what to do,
        # so the detail goes to the log and the card gets one plain sentence.
        from qgis.core import Qgis

        from ..core.logging_utils import log

        log(f"Account load failed ({code or 'unknown'}): {str(message)[:200]}",
            Qgis.MessageLevel.Warning)
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
        elif (code or "").strip().upper() == "INVALID_KEY":
            # Retry sends the same rejected key and fails the same way. The one
            # move that clears it is signing in again.
            self._error_label.setText(tr(
                "This computer is no longer signed in. Sign out, then sign in "
                "again to reconnect it."))
            self._retry_btn.setVisible(False)
            self._error_manage_btn.setVisible(False)
        elif (code or "").strip().upper() == "DEVICE_LIMIT_EXCEEDED":
            # Same sentence the run path uses, so both surfaces name the same
            # action. Retry cannot free a slot, so it is not offered.
            self._error_label.setText(tr(
                "Your plan is already running on its maximum number of "
                "computers. Close AI Segmentation on one of them, then try "
                "again."))
            self._retry_btn.setVisible(False)
            self._error_manage_btn.setVisible(False)
        elif (code or "").strip().upper() in _ACCOUNT_OFFLINE_CODES:
            self._error_label.setText(tr(
                "Could not reach TerraLab. Check your internet connection, "
                "then try again."))
            self._retry_btn.setVisible(True)
            self._error_manage_btn.setVisible(False)
        else:
            self._error_label.setText(tr(
                "Could not load your account. Try again in a moment."))
            self._retry_btn.setVisible(True)
            self._error_manage_btn.setVisible(False)
        self._error_widget.setVisible(True)
        self._show_local_cards()

    def _show_local_cards(self) -> None:
        """Keep what works with no network on screen after a failed load.

        Hiding the whole content area took the telemetry opt-out and the
        Remove-AI-data action away with the account card. The privacy control
        is the one thing that must never depend on a server answering.
        """
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        two_up_row = QWidget()
        two_up = QHBoxLayout(two_up_row)
        two_up.setContentsMargins(0, 0, 0, 0)
        two_up.setSpacing(10)
        two_up.addWidget(self._build_dependencies_card(), 1)
        two_up.addWidget(self._build_privacy_card(), 1)
        self._content_layout.addWidget(two_up_row)
        self._content_scroll.setVisible(True)
        from .dock.font_scale import apply_font_scale_to_tree

        apply_font_scale_to_tree(self)
        self.adjustSize()
        self._fit_to_screen()

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

        diameter = scale_px_length(38)
        avatar = QLabel(email[:1].upper() if email and email != "-" else "?")
        avatar.setFixedSize(diameter, diameter)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        avatar.setStyleSheet(
            f"background: {BRAND_GREEN}; color: #14210A;"
            f" border-radius: {diameter // 2}px;"
            " font-size: 17px; font-weight: 700;"
        )
        chip_row.addWidget(avatar)
        self._avatar_label = avatar
        self._show_account_picture(data.get("avatar_url"), diameter)

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
        sign_out_btn.clicked.connect(lambda: self._on_sign_out("account_card"))
        chip_row.addWidget(sign_out_btn, 0, Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(chip)
        return card

    def _show_account_picture(self, url, diameter: int) -> None:
        """Swap the letter badge for the account's own picture, when there is one.

        A copy already on disk costs no request. Otherwise one request per
        dialog open, off the UI thread. No picture, no https, a failed read or
        bytes that do not decode: the letter badge stays.
        """
        from .account_avatar import (
            AccountAvatarLoader,
            cached_avatar_pixmap,
            is_avatar_url_usable,
        )

        if not is_avatar_url_usable(url):
            return
        pixmap = cached_avatar_pixmap(url, diameter)
        if pixmap is not None:
            self._paint_account_picture(pixmap)
            return
        if self._avatar_requested:
            return
        self._avatar_requested = True
        # Parented to the dialog: a window closed before the reply lands takes
        # the loader and its connection with it.
        self._avatar_loader = AccountAvatarLoader(self)
        self._avatar_loader.loaded.connect(self._paint_account_picture)
        self._avatar_loader.fetch(url, diameter)

    def _paint_account_picture(self, pixmap) -> None:
        """Draw the picture on the badge the card is showing right now."""
        label = self._avatar_label
        if label is None or pixmap is None or pixmap.isNull():
            return
        try:
            label.setText("")
            label.setStyleSheet("background: transparent; border: none;")
            label.setPixmap(pixmap)
        except RuntimeError:
            self._avatar_label = None  # the card was rebuilt

    def _cancel_avatar_load(self) -> None:
        """Drop an in-flight picture read so it cannot land in a closed dialog."""
        loader = self._avatar_loader
        self._avatar_loader = None
        if loader is None:
            return
        safe_disconnect(loader, "loaded")
        try:
            loader.abort()
        except RuntimeError:
            pass  # already gone with its parent

    def _open_dashboard(self, source: str = "account_card"):
        """Hand the user off to the web dashboard, where checkout lives. Both
        callers pass their card name through a lambda, because a bare
        clicked.connect would feed Qt's checked bool in as the source."""
        try:
            from ..core import telemetry_session_events
            telemetry_session_events.track_account_dashboard_opened(source)
        except Exception:
            pass  # nosec B110 -- telemetry never blocks the handoff
        QDesktopServices.openUrl(QUrl(get_dashboard_url()))

    def _build_subscription_card(self, sub: dict, usage: dict | None = None) -> QFrame:
        """Plan status, credits (Pro) or monthly free allowance (Free), with a
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
        manage_btn.clicked.connect(lambda: self._open_dashboard("account_card"))
        header.addWidget(manage_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        card_layout.addLayout(header)

        status = sub.get("status", "active")
        status_text, status_color = _STATUS_DISPLAY.get(
            status, (status.title(), BRAND_RED)
        )

        plan = resolve_plan_credits(usage, sub)
        is_subscriber = plan.is_subscriber
        remaining = plan.remaining
        total = plan.total
        reset_date = plan.reset_date
        # First source that answers at all, not first that answers non-zero:
        # `or` read a fresh 0 as "absent" and fell back to the account row, so
        # a user who had just spent their last free detection was shown the
        # figure from before the run.
        free_left = _as_int(usage.get("free_detections_remaining"))
        if free_left is None:
            free_left = _as_int(sub.get("free_detections_remaining"))

        # One quiet status line under the header (the old two-row labeled grid
        # read as a form; this is a summary, not a form).
        plan_name = tr("Pro plan") if is_subscriber else tr("Free plan")
        plan_status = QLabel(
            f"{plan_name} · <span style='color:{status_color};'>{status_text}</span>"
        )
        plan_status.setStyleSheet("font-size: 12px; color: palette(text);")
        card_layout.addWidget(plan_status)

        bar: QProgressBar | None = None
        # Both plans renew, so both get the date. A free user reading a spent
        # balance has to be able to tell whether to wait or to pay.
        from ..core.quota_reset_date import format_quota_reset_date
        reset_str = format_quota_reset_date(reset_date)
        # Native two-envelope rows whenever the account row carries them:
        # cloud objects (Semi-Auto saves) and km² of Automatic. Older servers
        # send none of the fields and the wallet rows below stand unchanged.
        from ..core.quota_envelopes import quota_envelopes_from_account_row
        envelopes = quota_envelopes_from_account_row(sub)
        if envelopes is not None and self._add_envelope_rows(
                card_layout, envelopes):
            pass
        elif is_subscriber and remaining is not None:
            # Credits line for Pro subscribers. Lime fill for the bar; the
            # darker green tone for the number so it stays AA-readable on the
            # light dialog (mirrors AI Edit); red across the board once
            # exhausted.
            fill = BRAND_GREEN if remaining > 0 else BRAND_RED
            text_color = "palette(text)" if remaining > 0 else BRAND_RED
            credits_text = tr("{remaining} / {total} cloud detections").format(
                remaining=format_quota_count(remaining),
                total=format_quota_count(total),
            )
            # The count is also the way out to the website: the user who cannot
            # find their credits here is the same one who cannot find the
            # dashboard, so the line they came to read carries the link.
            card_layout.addWidget(
                self._dashboard_quota_label(credits_text, text_color))
            bar = self._credits_bar(remaining, total, fill)
        elif not is_subscriber and free_left is not None:
            # Free-taste line (monthly free allowance)
            fill = BRAND_GREEN if free_left > 0 else BRAND_RED
            text_color = "palette(text)" if free_left > 0 else BRAND_RED
            # The monthly free total is a newer usage field; only draw the gauge
            # and the "of N" phrasing when the server reports it (older
            # responses omit it).
            free_total = usage.get("free_detections_total")
            if free_total:
                free_text = tr("{n} of {total} free cloud detections left").format(
                    n=format_quota_count(free_left),
                    total=format_quota_count(free_total))
                bar = self._credits_bar(free_left, int(free_total), fill)
            elif free_left == 1:
                free_text = tr("1 free cloud detection remaining")
            else:
                free_text = tr("{n} free cloud detections remaining").format(
                    n=format_quota_count(free_left))
            free_lbl = QLabel(free_text)
            free_lbl.setStyleSheet(f"font-size: 12px; color: {text_color};")
            card_layout.addWidget(free_lbl)

        if bar is not None:
            card_layout.addWidget(bar)

        # A small reset note (the renewal date) above the action buttons,
        # replacing the old inline "(resets Jul 3)" suffix on the credits row.
        if reset_str:
            reset_row = QHBoxLayout()
            reset_row.setContentsMargins(0, 2, 0, 0)
            reset_label = QLabel(tr("Resets {date}").format(date=reset_str))
            reset_label.setStyleSheet("font-size: 10px; color: palette(text);")
            reset_row.addWidget(reset_label)
            reset_row.addStretch()
            card_layout.addLayout(reset_row)

        # Upgrade CTA (Free accounts only): the shared upsell card, so this
        # screen reads as the same offer the dock shows. Pro accounts have no
        # upgrade block here; the header's Manage link covers plan management.
        if not is_subscriber:
            card_layout.addSpacing(6)
            upgrade_card = UpsellCard(
                "accountUpgradeCard", "star",
                on_cta=self._on_upgrade_clicked)
            upgrade_card.set_text(
                # Both lines quote the offer, so both are served: the monthly
                # envelope is a commercial figure that moves without waiting
                # for a plugin release.
                title=dial_copy(
                    "account.upgrade_title",
                    tr("300 km² of Automatic a month, on zones of any "
                       "size.")),
                body=dial_copy(
                    "account.upgrade_body",
                    tr("The same AI on every machine you work on.")),
                cta=dial_copy("upsell.cta", tr("Upgrade to Pro")),
                escape=dial_copy(
                    "upsell.cta_hint",
                    tr("39 EUR a month, cancel anytime.")),
                star=dial_copy(
                    "upsell.bullet_quota_manual",
                    tr("2,000 cloud objects every month in Semi-Auto")),
            )
            upgrade_card.button.setToolTip(dial_copy(
                "account.upgrade_tooltip",
                tr("Opens terra-lab.ai in your browser.")))
            card_layout.addWidget(upgrade_card)

        # One sentence, one key. Glued fragments cannot be reordered, and every
        # language that needs a different order lost it here. The address is
        # served so it can be moved without a plugin release.
        contact_email = dial_copy(
            "account.contact_email", "yvann.barbot@terra-lab.ai", max_chars=120,
            escape=True)
        contact = QLabel(
            tr("Team or organization? Write to us: {email}").format(
                email=f"<b>{contact_email}</b>")
        )
        contact.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        contact.setWordWrap(True)
        contact.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        contact.setStyleSheet("font-size: 10px; color: rgba(128,128,128,0.9);")
        card_layout.addWidget(contact)

        return card

    def _dashboard_quota_label(self, text: str, text_color: str) -> QLabel:
        """One quota row, linked to the dashboard, with its arrow and tooltip.

        The row a user comes to read is also the way out to the website, so
        every balance on this card carries the same link rather than one of
        them looking like the only clickable one.
        """
        label = QLabel(
            f'<a href="{get_dashboard_url()}" style="text-decoration: none;">'
            f'<span style="color: {text_color};">{text}</span>'
            f' <span style="color: {BRAND_BLUE};">↗</span></a>'
        )
        label.setOpenExternalLinks(True)
        label.setCursor(Qt.CursorShape.PointingHandCursor)
        label.setToolTip(tr(
            "Opens your terra-lab.ai dashboard: your plan, your cloud "
            "detections and your payment details."))
        label.setStyleSheet(f"font-size: 12px; color: {text_color};")
        return label

    def _add_envelope_rows(self, card_layout, env) -> bool:
        """Draw the objects and km² rows of the two-envelope quota.

        Both plans read the same way: what is left this month, and a bar that
        empties as the month is spent. A None cap or a None used figure drops
        its row rather than drawing a zero: None means unknown or exempt,
        never empty. Returns False when neither row can be drawn, so the
        caller falls back to the wallet rows.
        """
        from .dock.ui_refresh import format_km2_left

        rows_drawn = False
        if env.has_objects_gauge():
            objects_left = (env.objects_remaining
                            if env.objects_remaining is not None
                            else max(0, env.objects_cap - env.objects_used))
            spent = objects_left <= 0
            # The line reads in the normal text colour: it states a balance,
            # and a balance is not a status. Green on every row made the card
            # shout at a user who has done nothing wrong. Red only when the
            # month is spent, where the colour is the message.
            text_color = BRAND_RED if spent else "palette(text)"
            fill = BRAND_RED if spent else BRAND_GREEN
            objects_text = tr(
                "{n} of {total} cloud objects left in Semi-Auto this month").format(
                    n=format_quota_count(objects_left),
                    total=format_quota_count(env.objects_cap))
            card_layout.addWidget(
                self._dashboard_quota_label(objects_text, text_color))
            card_layout.addWidget(self._credits_bar(
                objects_left, env.objects_cap, fill))
            rows_drawn = True
        if env.has_km2_gauge():
            km2_left = (env.km2_remaining if env.km2_remaining is not None
                        else max(0.0, env.km2_cap - env.km2_used))
            km2_spent = km2_left <= 0
            km2_color = BRAND_RED if km2_spent else "palette(text)"
            km2_fill = BRAND_RED if km2_spent else BRAND_GREEN
            km2_text = tr(
                "{n} of {total} km² left in Automatic this month").format(
                    n=format_km2_left(km2_left),
                    total=format_km2_left(env.km2_cap))
            # Same shape as the objects row above it: two balances on one card
            # cannot be one a link and the other plain text, or the row that is
            # not underlined reads as the one with nowhere to go.
            card_layout.addWidget(
                self._dashboard_quota_label(km2_text, km2_color))
            # The km² gauge is the same 6 px bar as the objects one: two
            # resources, two bars, read the same way. The bar counts whole
            # units, so pass hundredths of a km² or 2.4 of 3 draws as 2 of 3.
            # A balance too small to fill one unit keeps one anyway: the line
            # above it reads "< 0.01" and still allows a run, and an empty bar
            # under it says the month is spent.
            km2_units = int(round(km2_left * 100))
            if km2_units <= 0 and km2_left > 0:
                km2_units = 1
            card_layout.addWidget(self._credits_bar(
                km2_units, int(round(env.km2_cap * 100)), km2_fill))
            rows_drawn = True
        return rows_drawn

    def _on_upgrade_clicked(self):
        """The account dialog's Upgrade CTA. Reports through the same event as
        every other upsell surface, so the card can be compared with the pill
        and the cards inside the dock."""
        try:
            from ..core import telemetry_session_events
            telemetry_session_events.track_pro_upsell_clicked(source="account_dialog")
        except Exception:
            pass  # nosec B110
        QDesktopServices.openUrl(QUrl(get_upgrade_url()))

    def _build_dependencies_card(self) -> QFrame:
        """Local AI dependencies: where they live, how big, and an Open button.

        The isolated venv and model weights (~1 GB) live outside QGIS under the
        user home. Surfacing the path + size + an Open button answers the common
        support needs (disk space, antivirus exclusions, manual cleanup) without
        the user having to hunt for a hidden folder.
        """
        from ..core.cache_paths import PLUGIN_CACHE_DIR
        from ..core.venv_manager import VENV_DIR

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
        cached_size = _DIR_SIZE_CACHE.get(PLUGIN_CACHE_DIR)
        if not installed:
            size_text = tr("Not installed")
        else:
            size_text = cached_size or tr("Calculating...")
        info = QLabel(f"{tr('On disk')}: {size_text}")
        info.setStyleSheet("font-size: 11px; color: rgba(128,128,128,0.9);")
        layout.addWidget(info)
        self._size_label = info
        if installed and cached_size is None:
            self._start_dir_size_task(PLUGIN_CACHE_DIR)

        path_lbl = QLabel(PLUGIN_CACHE_DIR)
        path_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        path_lbl.setWordWrap(True)
        path_lbl.setStyleSheet("font-size: 10px; color: rgba(128,128,128,0.7);")
        layout.addWidget(path_lbl)

        open_btn = QPushButton(tr("Open folder"))
        open_btn.setStyleSheet(_SECONDARY_BTN)
        open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_btn.clicked.connect(lambda: self._open_install_folder(PLUGIN_CACHE_DIR))
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
            # Working line for the removal: the delete runs on a worker thread,
            # so the only sign it is under way is this text. Hidden until then.
            self._remove_status = QLabel("")
            self._remove_status.setWordWrap(True)
            self._remove_status.setStyleSheet(
                "font-size: 11px; color: palette(text);")
            self._remove_status.setVisible(False)
            layout.addWidget(self._remove_status)
            if self._removal_running:
                # The card was rebuilt (account refresh) while a removal runs:
                # carry the working state onto the new widgets.
                self._remove_btn.setEnabled(False)
                self._remove_btn.setText(tr("Removing..."))
                self._set_remove_status(
                    tr("Removing the downloaded AI data..."))

        return card

    def _on_remove_ai_data_clicked(self):
        """Confirm, then ask the plugin to delete the local venv + weights and
        clear the stored key + settings. The delete runs off the GUI thread and
        reports back through _on_remove_ai_data_progress / _finished; the user
        is signed out on success."""
        from qgis.PyQt.QtWidgets import QMessageBox

        if self._removal_running:
            return

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
            "the plugin. Your account and your cloud detections are not "
            "affected. "
            "Semi-Auto mode will download the files again next time you "
            "use it."))
        box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        if box.exec() != QMessageBox.StandardButton.Yes:
            return

        self._removal_running = True
        self._removal_generation += 1
        generation = self._removal_generation
        if self._remove_btn is not None:
            self._remove_btn.setEnabled(False)
            self._remove_btn.setText(tr("Removing..."))
        self._set_remove_status(tr("Removing the downloaded AI data..."))
        try:
            started, message = self._on_remove_ai_data(
                self._on_remove_ai_data_progress,
                self._on_remove_ai_data_finished)
        except Exception:  # nosec B110
            started, message = False, tr("Could not remove the AI data. Try again.")

        if not started:
            # The delete did not start: put the card back and show the reason
            # (the message says whether anything was cleared on the way).
            self._removal_running = False
            self._set_remove_status("")
            self._reset_remove_button()
            QMessageBox.warning(self, tr("AI Segmentation"), message)
            return

        # The only way out of this window is the removal reporting back, and a
        # worker can die without ever reporting. Arm the last resort.
        try:
            QTimer.singleShot(
                _REMOVAL_WATCHDOG_MS,
                lambda g=generation: self._on_removal_watchdog(g))
        except (RuntimeError, AttributeError):
            pass

    def _on_removal_watchdog(self, generation: int) -> None:
        """The removal never reported back. Unblock the window: it refuses to
        close while one runs, so this is the only way out of a modal dialog
        whose worker died silently. A stale beat (the removal ended, or a later
        one is running) is dropped on the generation check."""
        if generation != getattr(self, "_removal_generation", 0):
            return
        if not self._removal_running:
            return
        try:
            self._on_remove_ai_data_finished(False, tr(
                "The removal did not finish. Close this window, then check "
                "the AI data folder before trying again."))
        except RuntimeError:
            # The dialog is already gone; the flag is all that is left to clear.
            self._removal_running = False

    def _on_remove_ai_data_progress(self, text: str):
        """Current step of the running removal, pushed from the plugin."""
        self._set_remove_status(text)

    def _on_remove_ai_data_finished(self, ok: bool, message: str):
        """The removal ended: report it and close (the user is signed out).

        Runs once per removal. A worker that answers after the watchdog gave up
        on it lands here a second time, and the user has already been told."""
        from qgis.PyQt.QtWidgets import QMessageBox

        if not self._removal_running:
            return
        self._removal_running = False
        self._set_remove_status("")
        self._reset_remove_button()
        if ok:
            # The measured size describes a tree that no longer exists.
            _DIR_SIZE_CACHE.clear()
            QMessageBox.information(self, tr("AI Segmentation"), message)
            # The key is cleared and the user signed out: close the dialog.
            self.accept()
        else:
            QMessageBox.warning(self, tr("AI Segmentation"), message)

    def _set_remove_status(self, text: str):
        """Show (or hide, on an empty text) the removal working line."""
        label = getattr(self, "_remove_status", None)
        if label is None:
            return
        try:
            label.setText(text)
            label.setVisible(bool(text))
        except RuntimeError:
            self._remove_status = None  # the card was rebuilt

    def _reset_remove_button(self):
        """Put the removal button back to its resting, clickable state."""
        btn = getattr(self, "_remove_btn", None)
        if btn is None:
            return
        try:
            btn.setEnabled(True)
            btn.setText(tr("Remove downloaded AI data"))
        except RuntimeError:
            self._remove_btn = None  # the card was rebuilt

    def _build_privacy_card(self) -> QFrame:
        """Usage telemetry with a clear, ON-by-default opt-out.

        Flips the shared TerraLab/telemetry_enabled flag (telemetry.py), so
        turning it off here also silences the sibling AI Edit plugin.

        The copy below says "linked to your account" and names the typed object
        because both are true: every event carries a device hash that outlives
        the session, the batch is posted under the account's key, and the
        Automatic prompt travels verbatim. It used to say "anonymous", which
        none of that supports. Keep this card, telemetry.py's module docstring
        and the public privacy FAQ saying the same thing.
        """
        from ..core.telemetry import is_telemetry_enabled

        card = QFrame()
        card.setStyleSheet(_CARD_STYLE)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        title = QLabel(f"<b>{tr('Privacy')}</b>")
        title.setStyleSheet("font-size: 13px; color: palette(text);")
        layout.addWidget(title)

        self._telemetry_checkbox = QCheckBox(
            tr("Share usage statistics with TerraLab"))
        self._telemetry_checkbox.setToolTip(tr("Helps us fix bugs faster."))
        self._telemetry_checkbox.setChecked(is_telemetry_enabled())
        self._telemetry_checkbox.setCursor(Qt.CursorShape.PointingHandCursor)
        self._telemetry_checkbox.setStyleSheet("font-size: 12px; color: palette(text);")
        self._telemetry_checkbox.toggled.connect(self._on_telemetry_toggled)
        layout.addWidget(self._telemetry_checkbox)

        caption = QLabel(
            tr(
                "Errors, versions and the words you type, linked to your "
                "account. Never your imagery, layers or coordinates."
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

    def _on_telemetry_toggled(self, enabled: bool):
        """Record the move, then apply it. Order matters on an opt-out: track()
        short-circuits the moment the flag is false, so the event has to leave
        first. It is a FLUSH_NOW event, so it ships rather than sitting in a
        batch the opt-out would silence. Turning telemetry back on reports
        after the flag, for the same reason."""
        from ..core.telemetry import set_telemetry_enabled
        from ..core.telemetry_session_events import track_telemetry_opt_changed

        if not enabled:
            try:
                track_telemetry_opt_changed(False)
            except Exception:
                pass  # nosec B110
        set_telemetry_enabled(enabled)
        if enabled:
            try:
                track_telemetry_opt_changed(True)
            except Exception:
                pass  # nosec B110

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
        from .plugin.shared import dir_size_label
        return dir_size_label(path)

    def _start_dir_size_task(self, path: str):
        """Measure the on-disk size off the main thread.

        The tree holds the isolated environment and the model weights, tens of
        thousands of files: walking it while the dialog is being built stalled
        the window for seconds, worse behind antivirus.
        """
        from qgis.core import QgsApplication

        from .plugin.shared import dir_size_label

        # The callable runs on a task thread and must not reach back into the
        # dialog, which may be closed by then.
        task = GenericRequestTask(
            tr("Measuring AI data size..."),
            lambda: {"path": path, "size": dir_size_label(path)},
            hidden=True,
        )
        task.succeeded.connect(self._on_dir_size_ready)
        self._size_task = task
        QgsApplication.taskManager().addTask(task)

    def _on_dir_size_ready(self, data: dict):
        self._size_task = None
        size = str(data.get("size") or "-")
        _DIR_SIZE_CACHE[str(data.get("path") or "")] = size
        label = self._size_label
        if label is None:
            return
        try:
            label.setText(f"{tr('On disk')}: {size}")
        except RuntimeError:
            # The card was rebuilt (Retry) and this label already deleted.
            self._size_label = None

    def _cancel_size_task(self):
        """Drop an in-flight size measurement so it cannot land late."""
        if self._size_task is None:
            return
        try:
            self._size_task.succeeded.disconnect()
        except (RuntimeError, TypeError):  # nosec B110
            pass
        try:
            self._size_task.cancel()
        except Exception:  # nosec B110
            pass
        self._size_task = None

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

    def _on_sign_out(self, source: str = "account_card"):
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
        # Track the confirmed sign-out, not the button: a cancelled dialog is
        # not churn. This is the last event this install sends until it signs
        # back in, so it goes out before the key is cleared.
        try:
            from ..core import telemetry_session_events
            telemetry_session_events.track_account_signed_out(source)
        except Exception:
            pass  # nosec B110
        self.sign_out_requested.emit()
        self.accept()

    def done(self, result):  # noqa: N802 - Qt signature
        # A removal is under way: refuse to leave. Dismissing here would hide
        # the only report of what could not be deleted, and the account card
        # behind it already describes an environment that is disappearing.
        if self._removal_running:
            self._set_remove_status(
                tr("Removing the downloaded AI data. This window closes when "
                   "it is done."))
            return
        # accept()/reject() (Sign out, OK, Esc) dismiss the dialog without a
        # closeEvent, so cancel the in-flight loader here too rather than let it
        # complete with now-stale auth into a dismissed dialog.
        self._cancel_worker()
        self._cancel_avatar_load()
        super().done(result)

    def closeEvent(self, event):
        if self._removal_running:
            self._set_remove_status(
                tr("Removing the downloaded AI data. This window closes when "
                   "it is done."))
            event.ignore()
            return
        # The loader is a QgsTask now: no thread to wait on or terminate.
        # Disconnect + cancel so a late result can't fire into the closing
        # dialog; the task manager drains run() on its own. The old
        # wait(6000)+terminate() path crashed QGIS when the network was wedged.
        self._cancel_worker()
        self._cancel_avatar_load()
        super().closeEvent(event)
