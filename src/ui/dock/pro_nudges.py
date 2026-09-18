



















from __future__ import annotations

from datetime import date

from qgis.PyQt.QtCore import QSettings, Qt
from qgis.PyQt.QtWidgets import QLabel

from ...core.i18n import tr
from .font_scale import scale_qss_font_px
from .styles import FONT_HINT, LINK_INK, MUTED


PRO_AFTER_SUCCESS_HREF = "terralab:see-pro"

PRO_AFTER_SUCCESS_MONTH_KEY = "AI_Segmentation/pro_nudge_after_success_month"



_DISMISSED_PRO_CARDS: set[str] = set()

PRO_CARD_AUTO_LOW = "auto_low_credit"
PRO_CARD_MANUAL_LOW = "manual_low_credit"
PRO_CARD_EXEMPLAR_CAP = "exemplar_cap"
PRO_CARD_FREE_ZONE_CLIPPED = "free_zone_clipped"

_AFTER_SUCCESS_QSS = scale_qss_font_px(
    f"QLabel {{ font-size: {FONT_HINT}px; color: {MUTED};"

    " background: transparent; border: none; padding: 2px 0 0 0; }")


def pro_card_dismissed(key: str) -> bool:

    return key in _DISMISSED_PRO_CARDS


def build_pro_after_success_label(on_link) -> QLabel:


    label = QLabel()
    label.setObjectName("proAfterSuccessLine")
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setOpenExternalLinks(False)
    label.setStyleSheet(_AFTER_SUCCESS_QSS)
    label.linkActivated.connect(on_link)
    label.setVisible(False)
    return label


def _this_month() -> str:
    return date.today().strftime("%Y-%m")


class DockProNudgesMixin:




    def _known_free_account(self) -> bool:


        return bool(
            getattr(self, "_plugin_activated", False)
            and getattr(self, "_auto_credits", None) is not None
            and not getattr(self, "_auto_is_subscriber", False))



    def _sync_pro_pill(self) -> None:


        header = getattr(self, "_dock_header", None)
        if header is None:
            return
        show = self._known_free_account()
        try:
            header.set_pro_pill_visible(show)
        except (RuntimeError, AttributeError):
            return
        if show:
            self._track_pro_nudge_view("header_pill")
        else:



            self.hide_pro_after_success()

    def _on_header_pro_pill_clicked(self) -> None:

        if not self._known_free_account():
            return
        self.hide_pro_after_success()
        from ...core.pro_page_link import open_pro_page
        open_pro_page("plugin_header_pill", "header_pill", parent=self)

    @staticmethod
    def _track_pro_nudge_view(surface: str) -> None:

        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pro_upsell_viewed(
                trigger=surface, cta_source=surface)
        except Exception:  # noqa: BLE001
            pass  # nosec B110



    def show_pro_after_success(self, mode: str) -> None:


        try:
            if not self._known_free_account():
                return
            if getattr(self, "_auto_run_active", False):
                return
            label = getattr(
                self, "auto_pro_after_success" if mode == "auto"
                else "manual_pro_after_success", None)
            if label is None:
                return
            settings = QSettings()
            month = _this_month()
            if str(settings.value(PRO_AFTER_SUCCESS_MONTH_KEY, "") or "") == month:
                return
            settings.setValue(PRO_AFTER_SUCCESS_MONTH_KEY, month)




            lead = tr("Like the result?")
            link = tr("See Pro")
            label.setText(
                f'{lead} <a href="{PRO_AFTER_SUCCESS_HREF}" style="color: {LINK_INK};'
                f' text-decoration: none; font-weight: 600;">{link}</a>')
            label.setVisible(True)
        except (RuntimeError, AttributeError, TypeError):
            return
        self._track_pro_nudge_view("after_success")

    def hide_pro_after_success(self) -> None:

        for name in ("auto_pro_after_success", "manual_pro_after_success"):
            label = getattr(self, name, None)
            if label is None:
                continue
            try:
                label.setVisible(False)
            except RuntimeError:
                pass

    def _on_pro_after_success_link(self, href: str) -> None:
        if href != PRO_AFTER_SUCCESS_HREF:
            return
        self.hide_pro_after_success()
        from ...core.pro_page_link import open_pro_page
        open_pro_page("plugin_after_success", "after_success", parent=self)



    def _dismiss_pro_card(self, key: str, card) -> None:

        _DISMISSED_PRO_CARDS.add(key)
        try:
            card.setVisible(False)
        except (RuntimeError, AttributeError):
            pass  # nosec B110
