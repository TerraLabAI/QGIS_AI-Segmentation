




























from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
from ...core.server_dials import dial_copy
from ..icons import icon_for
from .auto_flow_look import _BTN_AUTO_CHIP, _BTN_AUTO_QUIET
from .styles import BTN_CHIP_PX
from .upsell_card import UpsellCard, keep_working_cta
from .widgets import Mode





_PAGE_EXEMPT = frozenset({"zone_too_large"})


def _takes_the_page(reason: str) -> bool:

    return reason not in _PAGE_EXEMPT


class DockAutoRunBlockMixin:


    def _setup_auto_run_block(self, parent_layout) -> None:







        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)





        self.auto_km2_block = UpsellCard(
            "autoKm2Block", "full", self._on_upgrade_clicked)


        self.auto_km2_block_upgrade = self.auto_km2_block.button
        self.auto_km2_block.setVisible(False)
        layout.addWidget(self.auto_km2_block)




        self.auto_run_block_card = UpsellCard("autoRunBlockCard", "full")
        self.auto_run_block_card.setVisible(False)
        layout.addWidget(self.auto_run_block_card)



        self.auto_run_block_redraw_btn = QPushButton(dial_copy(
            "run_block.redraw_cta", tr("Draw a smaller zone")))
        self.auto_run_block_redraw_btn.setMinimumHeight(BTN_CHIP_PX)
        self.auto_run_block_redraw_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_run_block_redraw_btn.setStyleSheet(_BTN_AUTO_CHIP)
        self.auto_run_block_redraw_btn.setIcon(
            icon_for(self.auto_run_block_redraw_btn, "polygon", 16))
        self.auto_run_block_redraw_btn.clicked.connect(
            self._on_auto_run_block_redraw)
        layout.addWidget(self.auto_run_block_redraw_btn)





        self.auto_run_block_exit_btn = QPushButton(tr("Exit"))
        self.auto_run_block_exit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_run_block_exit_btn.setStyleSheet(_BTN_AUTO_QUIET)
        self.auto_run_block_exit_btn.clicked.connect(
            self.auto_exit_requested.emit)
        exit_row = QHBoxLayout()
        exit_row.setContentsMargins(0, 0, 0, 0)
        exit_row.addStretch(1)
        exit_row.addWidget(self.auto_run_block_exit_btn, 0)
        exit_row.addStretch(1)
        layout.addLayout(exit_row)



        holder.setSizePolicy(QSizePolicy.Policy.Preferred,
                             QSizePolicy.Policy.Maximum)
        holder.setVisible(False)
        self.auto_run_block = holder
        parent_layout.addWidget(holder)



    def _auto_balance_spent(self) -> bool:






        try:
            if self._auto_is_subscriber:
                return (self._auto_credits is not None
                        and self._auto_credits <= 0)
            return (self._auto_free_left is not None
                    and self._auto_free_left <= 0)
        except (AttributeError, TypeError, ValueError):
            return False

    def _auto_service_available(self) -> bool:


        try:
            from ...core.activation_manager import is_automatic_mode_enabled

            return bool(is_automatic_mode_enabled())
        except Exception:  # noqa: BLE001
            return True

    def _auto_run_block_reason(self) -> str | None:







        try:
            if not self._plugin_activated or self._mode != Mode.AUTOMATIC:
                return None
            if self._auto_run_active or self._auto_review_active:
                return None
            if not self._auto_service_available():
                return "kill_switch"
            if not getattr(self, "_auto_started", False):


                return None
            if self._auto_balance_spent():
                return "credits"
            if not getattr(self, "_auto_zone_is_set", False):
                return None
            if getattr(self, "_auto_zone_too_large", False):
                return "zone_too_large"
            if getattr(self, "_auto_km2_exceeded", False):
                return "km2_envelope"
        except (RuntimeError, AttributeError):
            return None
        return None



    def _refresh_auto_run_block(self, suppressed: bool = False) -> bool:



        holder = getattr(self, "auto_run_block", None)
        if holder is None:
            return False
        try:
            reason = None if suppressed else self._auto_run_block_reason()
            if reason is not None and not _takes_the_page(reason):


                self._note_detect_blocked(reason)
                reason = None
            if reason is None:
                holder.setVisible(False)
                return False
            show_redraw = self._fill_auto_run_block_card(reason)
            self.auto_run_block_redraw_btn.setVisible(show_redraw)


            self.auto_run_block_exit_btn.setVisible(
                bool(getattr(self, "_auto_started", False)))
            holder.setVisible(True)
        except (RuntimeError, AttributeError):
            return False
        self._note_detect_blocked(reason)
        return True

    def _apply_auto_run_block_takeover(self) -> None:








        try:
            if not self._plugin_activated or self._mode != Mode.AUTOMATIC:



                return
            busy = bool(getattr(self, "_auto_run_active", False)
                        or getattr(self, "_auto_review_active", False))
            if self._is_free_exhausted() and not busy:
                return
            blocked = self._refresh_auto_run_block()
            self.auto_controls_section.setVisible(not blocked)
        except (RuntimeError, AttributeError):
            return

    def _fill_auto_run_block_card(self, reason: str) -> bool:







        km2 = getattr(self, "auto_km2_block", None)
        card = self.auto_run_block_card
        if reason == "km2_envelope":


            card.setVisible(False)
            if km2 is not None:
                km2.setVisible(True)


            left = self._auto_km2_left()
            show_redraw = left is not None and left > 0
            if km2 is not None and show_redraw:


                km2.escape.setVisible(False)
            return show_redraw
        if km2 is not None:
            km2.setVisible(False)
        card.setVisible(True)
        if reason == "kill_switch":
            return self._fill_auto_block_kill_switch(card)
        return self._fill_auto_block_credits(card)

    def _fill_auto_block_kill_switch(self, card) -> bool:




        card.set_tint("neutral")
        card.set_ghost_button(False)
        card.route_cta(self._on_auto_upsell_manual_clicked)
        card.set_text(

            dial_copy("run_block.offline_title",
                      tr("Automatic is unavailable right now.")).rstrip("."),
            dial_copy("run_block.offline_body",
                      tr("Try again in a few minutes. Your zone and your "
                         "settings are kept.")),
            dial_copy("upsell.manual_cta", tr("Use Semi-Auto")),
        )
        return False

    def _fill_auto_block_credits(self, card) -> bool:


        from ...core.pro_ceiling import pro_ceiling_enabled

        reset_day = getattr(self, "_auto_reset_display", "")
        subscriber = bool(getattr(self, "_auto_is_subscriber", False))


        title = dial_copy(
            "run_block.credits_title",
            tr("You used your Automatic surface for this month.")).rstrip(".")
        if reset_day:
            escape = dial_copy(
                "run_block.credits_escape_reset",
                tr("Semi-Auto still works, and Automatic comes back on "
                   "{date}.")).replace("{date}", reset_day.replace(" ", "\u00a0"))
        else:
            escape = dial_copy(
                "run_block.credits_escape",
                tr("Semi-Auto still works until it comes back."))
        if subscriber and pro_ceiling_enabled():


            card.set_tint("premium")


            card.set_ghost_button(True)
            body, cta = self._pro_ceiling_copy()
            card.route_cta(self._on_pro_contact_run_block)
            card.set_text(title, body, cta, escape=escape,
                          detail=self._pro_ceiling_detail())
            return False
        if subscriber:


            card.set_tint("neutral")
            card.set_ghost_button(True)
            card.route_cta(self._on_auto_upsell_manual_clicked)
            card.set_text(title, None,
                          dial_copy("upsell.manual_cta", tr("Use Semi-Auto")),
                          escape=escape)
            return False
        card.set_tint("premium")
        card.set_ghost_button(False)
        card.route_cta(self._on_upgrade_clicked)
        card.set_text(
            title,
            dial_copy("upsell.wall_body",
                      tr("Draw a whole city and let it run, at the finest "
                         "precision.")),
            keep_working_cta(),
            escape=escape,
        )
        card.set_pro_offer("plugin_run_block")
        return False



    def _on_auto_run_block_redraw(self) -> None:


        try:
            self.on_zone_deleted_from_canvas()
        except (RuntimeError, AttributeError):
            self._go_to_auto_step(1)

    def _note_detect_blocked(self, reason: str | None) -> None:



        if reason == getattr(self, "_detect_blocked_last", None):
            return
        self._detect_blocked_last = reason
        if not reason:
            return
        try:
            from ...core import telemetry_session_events

            telemetry_session_events.track_detect_blocked(reason=reason)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
