
















from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
from ...core.server_dials import dial_copy
from .contact_copy import CopyEmailLabel
from .font_scale import scale_px_length, scale_qss_font_px
from .styles import (
    _BTN_GHOST,
    _CARD_CHILD_BTN_RESET_QSS,
    _CARD_QSS,
    BTN_PILL_PX,
    FONT_BASE,
    FONT_HINT,
    INK,
    MUTED,
    _card_divider,
    _msg_label_qss,
    msg_rich,
    no_break_words,
)
from .ui_refresh import format_quota_count
from .upsell_card import UpsellCard, keep_working_cta
from .widgets import Mode


_CARD_LABEL_RESET_QSS = "QLabel { background: transparent; border: none; }"



_TITLE_QSS = scale_qss_font_px(
    f"font-size: {FONT_BASE}px; font-weight: 600; color: {INK};")
_QUIET_QSS = scale_qss_font_px(f"font-size: {FONT_HINT}px; color: {MUTED};")

_GATE_CARD_MARGINS = (16, 14, 16, 14)


def _warning_html(text: str) -> str:

    return msg_rich("warning", text or "")


class DockManualCreditGateMixin:


    def _setup_manual_credit_gate(self) -> None:

        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)




        self.manual_credit_notice = QLabel()
        self.manual_credit_notice.setWordWrap(True)
        self.manual_credit_notice.setTextFormat(Qt.TextFormat.RichText)
        self.manual_credit_notice.setStyleSheet(_msg_label_qss("warning"))
        self.manual_credit_notice.setVisible(False)
        layout.addWidget(self.manual_credit_notice)

        card = QWidget()
        card.setObjectName("manualCreditCard")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setStyleSheet(
            _CARD_QSS.format(name="manualCreditCard")
            + _CARD_LABEL_RESET_QSS
            + _CARD_CHILD_BTN_RESET_QSS)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(*_GATE_CARD_MARGINS)
        card_layout.setSpacing(4)


        self.manual_credit_title = QLabel()
        self.manual_credit_title.setWordWrap(True)
        self.manual_credit_title.setStyleSheet(_TITLE_QSS)
        card_layout.addWidget(self.manual_credit_title)




        self.manual_credit_reset = QLabel()
        self.manual_credit_reset.setWordWrap(True)
        self.manual_credit_reset.setStyleSheet(_QUIET_QSS)
        self.manual_credit_reset.setVisible(False)
        card_layout.addWidget(self.manual_credit_reset)

        card_layout.addSpacing(4)
        card_layout.addWidget(self._build_manual_credit_pro_lane())
        card_layout.addWidget(self._build_manual_credit_contact_lane())
        card_layout.addSpacing(6)
        for widget in self._build_manual_credit_free_way_out():
            if widget is self.manual_credit_offline_btn:
                card_layout.addWidget(widget, 0, Qt.AlignmentFlag.AlignLeft)
            else:
                card_layout.addWidget(widget)

        layout.addWidget(card)
        holder.setVisible(False)
        self.manual_credit_gate = holder
        self.main_layout.addWidget(holder)

    def _build_manual_credit_pro_lane(self) -> QWidget:


        lane = UpsellCard("manualCreditOffer", "star",
                          on_cta=self._on_upgrade_clicked, flat=True)


        self.manual_credit_upgrade_btn = lane.button


        self.manual_credit_pro_lane = lane




        star = dial_copy(
            "upsell.bullet_quota_manual",
            tr("500 cloud objects a month with Pro"))




        lane.set_text("", None, keep_working_cta(), star=star)
        lane.set_pro_offer("plugin_objects_wall")
        return lane

    def _build_manual_credit_contact_lane(self) -> QWidget:








        lane = UpsellCard("manualCreditContact", "wall",
                          on_cta=self._on_pro_contact_objects_wall)
        lane.set_ghost_button(True)
        lane.setVisible(False)
        self.manual_credit_contact_lane = lane
        return lane

    def _build_manual_credit_free_way_out(self) -> list[QWidget]:










        rule = _card_divider()

        self.manual_credit_free_note = QLabel()
        self.manual_credit_free_note.setWordWrap(True)
        self.manual_credit_free_note.setStyleSheet(_QUIET_QSS)



        self.manual_credit_offline_btn = QPushButton(dial_copy(
            "manual_gate.offline_cta", tr("Use my computer")))


        self.manual_credit_offline_btn.setFixedHeight(scale_px_length(BTN_PILL_PX))
        self.manual_credit_offline_btn.setAutoDefault(False)
        self.manual_credit_offline_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.manual_credit_offline_btn.setStyleSheet(_BTN_GHOST)
        self.manual_credit_offline_btn.clicked.connect(
            self._on_manual_credit_offline_clicked)



        self.manual_credit_custom_needs = CopyEmailLabel()
        return [rule, self.manual_credit_free_note,
                self.manual_credit_offline_btn,
                self.manual_credit_custom_needs]



    def _manual_credits_exhausted(self) -> bool:





        try:
            env = getattr(self, "_quota_envelopes", None)
            if env is not None and env.objects_remaining is not None:



                return env.objects_remaining <= 0
            left = getattr(self, "_auto_credits", None)
            return left is not None and int(left) <= 0
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return False

    def _manual_credit_gate_owns_page(self) -> bool:





        try:
            return bool(
                self._plugin_activated
                and self._mode == Mode.INTERACTIVE
                and not getattr(self, "_segmentation_active", False)
                and self._manual_cloud_route_picked()
                and self._manual_credits_exhausted()
                and self.layer_combo.count_layers() > 0
            )
        except (RuntimeError, AttributeError):
            return False

    def _refresh_manual_credit_gate(self) -> None:

        gate = getattr(self, "manual_credit_gate", None)
        if gate is None:
            return
        try:




            on_cloud = self._manual_cloud_route_picked()
            show = bool(
                self._plugin_activated
                and self._mode == Mode.INTERACTIVE
                and on_cloud
                and self._manual_credits_exhausted()



                and (self._segmentation_active
                     or self.layer_combo.count_layers() > 0)
            )
            gate.setVisible(show)
            if not show:
                return
            in_session = bool(getattr(self, "_segmentation_active", False))


            has_saved = bool(getattr(self, "_saved_polygon_count", 0) > 0)
            if in_session:
                self.manual_credit_notice.setText(_warning_html(
                    dial_copy(
                        "manual_gate.notice_saved",
                        tr("This polygon stays on the map, and Export still "
                           "works."))
                    if has_saved else
                    dial_copy(
                        "manual_gate.notice_unsaved",
                        tr("This polygon stays on the map, but it cannot be "
                           "saved."))))
            self.manual_credit_notice.setVisible(in_session)



            subscriber = bool(getattr(self, "_auto_is_subscriber", False))
            lane = getattr(self, "manual_credit_pro_lane", None)
            if lane is not None:
                lane.setVisible(not subscriber)
            self._refresh_manual_credit_contact_lane(subscriber)
            self._refresh_manual_credit_custom_needs(subscriber)
            env = getattr(self, "_quota_envelopes", None)
            if env is not None and env.objects_cap:



                self.manual_credit_title.setText(no_break_words(dial_copy(
                    "manual_gate.title_objects",
                    tr("{n} cloud objects used this month"),
                ).replace("{n}", format_quota_count(env.objects_cap))))
            else:
                self.manual_credit_title.setText(
                    dial_copy(
                        "manual_gate.title_exhausted_pro",
                        tr("Your cloud detections are used up"))
                    if getattr(self, "_auto_is_subscriber", False) else
                    dial_copy(
                        "manual_gate.title_exhausted_free",
                        tr("Your free cloud detections are used up")))
            self.manual_credit_reset.setText(self._manual_credit_reset_text())
            self.manual_credit_reset.setVisible(
                bool(self.manual_credit_reset.text()))


            self.manual_credit_offline_btn.setText(
                dial_copy("manual_gate.offline_cta_in_session",
                          tr("Stop and use my computer"))
                if in_session else
                dial_copy("manual_gate.offline_cta",
                          tr("Use my computer")))



            self.manual_credit_free_note.setText(
                self._manual_credit_free_note_text(in_session))
        except (RuntimeError, AttributeError):
            return
        self._note_manual_upsell_viewed()

    def _refresh_manual_credit_contact_lane(self, subscriber: bool) -> None:

        from ...core.pro_ceiling import pro_ceiling_enabled
        lane = getattr(self, "manual_credit_contact_lane", None)
        if lane is None:
            return
        show = subscriber and pro_ceiling_enabled()
        if show:
            body, cta = self._pro_ceiling_copy()
            lane.set_text("", body, cta, detail=self._pro_ceiling_detail())
        lane.setVisible(show)

    def _refresh_manual_credit_custom_needs(self, subscriber: bool) -> None:


        line = getattr(self, "manual_credit_custom_needs", None)
        if line is None:
            return
        if subscriber:
            line.set_email(None)
            return
        from ...core.pro_ceiling import pro_ceiling_contact_email
        line.set_email(pro_ceiling_contact_email())

    def _manual_credit_reset_text(self) -> str:






        reset_day = getattr(self, "_auto_reset_display", "")
        env = getattr(self, "_quota_envelopes", None)


        if env is not None and env.objects_cap:


            if reset_day:
                return dial_copy(
                    "manual_gate.reset_date",
                    tr("Back on {date}."),
                ).replace("{date}", reset_day)
            return ""
        try:
            total = int(getattr(self, "_auto_credits_total", 0) or 0)
        except (TypeError, ValueError):
            total = 0
        if total > 0 and reset_day:
            return dial_copy(
                "manual_gate.reset_used_all",
                tr("All {n} used. Back on {date}."),
            ).replace("{n}", format_quota_count(total)).replace("{date}", reset_day)
        if reset_day:
            return dial_copy(
                "manual_gate.reset_date",
                tr("Back on {date}."),
            ).replace("{date}", reset_day)
        return ""

    def _manual_credit_free_note_text(self, in_session: bool) -> str:










        if in_session:
            base = dial_copy(
                "manual_gate.free_note_in_session",
                tr("Or end this session and use a free AI on this "
                   "computer. Saved polygons stay."))
        else:


            base = dial_copy(
                "manual_gate.free_note_short",
                tr("Or use a smaller free AI on this computer."))
        if self._manual_engine_local_ready():
            return base
        try:
            from .manual_local_install_dialog import (
                local_install_disk_figures,
                local_install_minutes,
            )

            need, _free = local_install_disk_figures()
            minutes = str(local_install_minutes())
        except Exception:  # noqa: BLE001
            need = 0.0
            minutes = ""



        if need > 0 and minutes:
            return base + " " + dial_copy(
                "manual_gate.install_note",
                tr("First a {gb} GB download, about {n} minutes."),
            ).replace("{gb}", f"{need:g}").replace("{n}", minutes)
        if minutes:
            return base + " " + dial_copy(
                "manual_gate.install_note_no_size",
                tr("First a download, about {n} minutes."),
            ).replace("{n}", minutes)
        return base + " " + dial_copy(
            "manual_gate.install_note_no_time",
            tr("First a download."),
        )

    def _note_manual_upsell_viewed(self) -> None:

        if getattr(self, "_manual_upsell_seen", False):
            return
        self._manual_upsell_seen = True
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pro_upsell_viewed(
                trigger="manual_credits_exhausted", cta_source="upsell_card")
            env = getattr(self, "_quota_envelopes", None)
            telemetry_session_events.track_manual_objects_wall_hit(
                is_subscriber=bool(getattr(self, "_auto_is_subscriber", False)),
                in_session=bool(getattr(self, "_segmentation_active", False)),
                objects_cap=getattr(env, "objects_cap", None),
                objects_used=getattr(env, "objects_used", None),
            )
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _on_manual_credit_offline_clicked(self) -> None:






        if getattr(self, "_segmentation_active", False):
            self.stop_segmentation_requested.emit()





            if getattr(self, "_segmentation_active", False):
                return
        try:




            self._on_manual_engine_picked(False)
        except (RuntimeError, AttributeError):
            return
        self._update_full_ui()
