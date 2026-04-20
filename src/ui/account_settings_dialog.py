













from __future__ import annotations

from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtWidgets import QDialog, QHBoxLayout, QStackedWidget, QVBoxLayout, QWidget

from ..core.i18n import tr
from .account_settings_avatar import AccountAvatarMixin
from .account_settings_contact import AccountContactMixin
from .account_settings_deletion import (
    _DELETE_ACCOUNT_WATCHDOG_MS,
    AccountDeletionMixin,
    _format_purge_date,
)
from .account_settings_layout import _SCREEN_MARGIN_PX, AccountLayoutMixin
from .account_settings_plan import (
    _PRO_MONTHLY_CREDITS_FALLBACK,
    _STATUS_DISPLAY,
    AccountPlanMixin,
    PlanCredits,
    _as_int,
    _resolve_is_subscriber,
    resolve_plan_credits,
)
from .account_settings_privacy import AccountPrivacyMixin
from .account_settings_removal import _DIR_SIZE_CACHE, _REMOVAL_WATCHDOG_MS, AccountRemovalMixin
from .account_settings_session import (
    _ACCOUNT_OFFLINE_CODES,
    AccountSessionMixin,
    _load_account_and_usage,
)
from .dock.font_scale import apply_font_scale_to_tree
from .settings.account_page import AccountPageMixin
from .settings.billing_page import BillingPageMixin
from .settings.learn_pages import LearnPagesMixin
from .settings.local_model_page import LocalModelPageMixin
from .settings.settings_sidebar import SettingsSidebarMixin
from .settings.settings_widgets import WINDOW_QSS

PRODUCT_NAME = "AI Segmentation"


class AccountSettingsDialog(
    AccountSessionMixin,
    AccountLayoutMixin,
    AccountAvatarMixin,
    AccountPlanMixin,
    AccountContactMixin,
    AccountRemovalMixin,
    AccountDeletionMixin,
    AccountPrivacyMixin,
    SettingsSidebarMixin,
    AccountPageMixin,
    BillingPageMixin,
    LocalModelPageMixin,
    LearnPagesMixin,
    QDialog,
):

    sign_out_requested = pyqtSignal()



    account_deleted = pyqtSignal()


    usage_loaded = pyqtSignal(dict)

    def __init__(self, client, auth, activation_key, parent=None,
                 on_remove_ai_data=None, is_busy_check=None):
        super().__init__(parent)
        self.setObjectName("AISegmentationSettings")
        self.setWindowTitle(tr("AI Segmentation settings"))
        self.setModal(True)





        self._on_remove_ai_data = on_remove_ai_data
        self._is_busy_check = is_busy_check



        self._removal_running = False
        self._removal_generation = 0
        self._remove_status = None
        self._remove_btn = None
        self._delete_running = False
        self._delete_generation = 0
        self._delete_status = None
        self._delete_btn = None
        self._delete_task = None
        self._last_delete_error: dict | None = None


        self._account_email = ""


        del activation_key
        self._worker = None
        self._size_task = None
        self._size_label = None
        self._avatar_label = None
        self._avatar_loader = None
        self._avatar_requested = False
        self._account_state: dict = {}
        self._page_rows: dict = {}
        self._current_page_row = 0
        self._client = client
        self._auth = auth

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_sidebar())
        right = QWidget(self)
        right_col = QVBoxLayout(right)
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(0)
        self._pages = QStackedWidget(right)
        right_col.addWidget(self._pages, 1)
        root.addWidget(right, 1)
        self._right = right
        self._build_saved_hint(right)

        self._add_settings_page("account", "person", tr("Account"), self._build_account_page())
        self._add_settings_page("billing", "gem", tr("Billing"), self._build_billing_page())
        self._add_settings_page("local_model", "package", tr("Local model"),
                                self._build_local_model_page())
        self._add_settings_page("tutorials", "play", tr("Tutorials"), self._build_tutorials_page())
        self._add_settings_page("shortcuts", "terminal", tr("Keyboard shortcuts"),
                                self._build_shortcuts_page())
        self._add_settings_page("plugins", "puzzle", tr("More plugins"), self._build_plugins_page())

        self._add_settings_action("chat_bubble", tr("Contact us"), "contact")
        self._add_settings_action("warning", tr("Report a problem"), "report")
        self._nav.setCurrentRow(0)

        self.setStyleSheet(WINDOW_QSS)
        apply_font_scale_to_tree(self)
        self._fit_rail_to_text()
        self._fit_to_screen()
        self._fetch_account()

    def _close_refused(self) -> bool:




        if self._removal_running:
            self.show_settings_page("local_model")
            self._set_remove_status(
                tr("Removing the downloaded AI data. This window closes when "
                   "it is done."))
            return True
        if self._delete_running:
            self.show_settings_page("account")
            self._set_delete_status(
                tr("Scheduling the deletion. This window closes when the "
                   "service answers."))
            return True
        return False

    def _release_background_work(self) -> None:



        self._cancel_worker()
        self._cancel_size_task()
        self._cancel_delete_task()
        self._cancel_avatar_load()

    def done(self, result):  # noqa: N802
        if self._close_refused():
            return
        self._release_background_work()
        super().done(result)

    def closeEvent(self, event):
        if self._close_refused():
            event.ignore()
            return
        self._release_background_work()
        super().closeEvent(event)


__all__ = [
    "PRODUCT_NAME",
    "_DIR_SIZE_CACHE",
    "_REMOVAL_WATCHDOG_MS",
    "_DELETE_ACCOUNT_WATCHDOG_MS",
    "_SCREEN_MARGIN_PX",
    "_ACCOUNT_OFFLINE_CODES",
    "_STATUS_DISPLAY",
    "_format_purge_date",
    "_PRO_MONTHLY_CREDITS_FALLBACK",
    "PlanCredits",
    "_as_int",
    "_resolve_is_subscriber",
    "resolve_plan_credits",
    "_load_account_and_usage",
    "AccountSettingsDialog",
]
