






from __future__ import annotations

from ...core.credit_gate import low_credit_ceiling as _low_credit_ceiling
from ...core.credit_gate import low_credit_threshold as _low_credit_threshold
from ...core.i18n import tr
from ...core.server_dials import dial_copy
from .font_scale import scale_qss_font_px
from .pro_nudges import PRO_CARD_AUTO_LOW, pro_card_dismissed
from .styles import (
    BTN_CHIP_PX,
    FONT_HINT,
    RED_TEXT,
)
from .upsell_card import UpsellCard, keep_working_cta
from .widgets import (
    Mode,
)



_KM2_DISPLAY_CAP = 10_000


def capped_zone_km2_text(km2) -> str:

    from .ui_refresh import format_km2_surface
    try:
        if float(km2) > _KM2_DISPLAY_CAP:
            return tr("over {n}").format(n=format_km2_surface(_KM2_DISPLAY_CAP))
    except (TypeError, ValueError):
        pass
    return format_km2_surface(km2)


class DockAutoCreditsMixin:



    def _on_upgrade_clicked(self) -> None:
        source, cta_source = "upsell_card", "plugin_upsell_card"
        try:
            sender = self.sender()
            if sender is getattr(self, "auto_exhausted_subscribe_link", None):
                source, cta_source = "exhausted_status", "plugin_km2_wall"
            elif sender is getattr(self, "auto_km2_block_upgrade", None):
                source, cta_source = "km2_block", "plugin_km2_block"
            elif sender is getattr(self, "manual_credit_upgrade_btn", None):
                source, cta_source = "upsell_card", "plugin_objects_wall"
            elif sender is getattr(self, "auto_upgrade_btn", None):
                source, cta_source = "upsell_card", "plugin_free_exhausted_wall"
        except Exception:
            pass  # nosec B110
        from ...core.pro_page_link import open_pro_page
        open_pro_page(cta_source, source, parent=self)

    def _build_upgrade_url(self, cta_source: str = "plugin_upsell_card") -> str:


        from ...core.activation_manager import get_pro_checkout_url
        return get_pro_checkout_url(cta_source)

    def set_auto_envelopes(self, envelopes) -> None:






        self._quota_envelopes = envelopes



        if not self.dock_content_built:
            return



        km2 = getattr(self, "_auto_zone_km2", None)
        if km2 is not None and self._auto_zone_is_set:
            left = self._auto_km2_left()
            if left is not None and km2 > left:
                self.set_auto_km2_block(km2, left)
            else:
                self.set_auto_km2_block(None)
        self._refresh_auto_credits_display()
        self._update_full_ui()

    def quota_envelopes(self):

        return getattr(self, "_quota_envelopes", None)

    def note_cloud_object_charged(self) -> None:






        env = getattr(self, "_quota_envelopes", None)
        if env is None:
            return
        used = env.objects_used + 1 if env.objects_used is not None else None
        if used is not None and env.objects_cap is not None:
            used = min(used, env.objects_cap)
        left = (max(0, env.objects_remaining - 1)
                if env.objects_remaining is not None else None)
        self.set_auto_envelopes(env._replace(
            objects_used=used, objects_remaining=left))

    def note_cloud_objects_exhausted(self) -> None:


        env = getattr(self, "_quota_envelopes", None)
        if env is None:
            return
        used = env.objects_used
        if env.objects_cap is not None:
            used = env.objects_cap if used is None else max(used, env.objects_cap)
        self.set_auto_envelopes(env._replace(
            objects_used=used, objects_remaining=0))

    def set_auto_credits(self, credits: int, reset_date: str,
                         is_subscriber: bool,
                         total: int | None = None) -> None:






        self._auto_credits = credits
        self._auto_credits_total = total
        self._auto_is_subscriber = is_subscriber
        self._auto_reset_date = reset_date or ""


        from ...core.quota_reset_date import format_quota_reset_date
        self._auto_reset_display = format_quota_reset_date(self._auto_reset_date)
        if not is_subscriber:
            self._auto_free_left = credits



        if not self.dock_content_built:
            return

        self._sync_pro_pill()
        self._refresh_auto_credits_display()





        _cost_label_free = not self._auto_run_active and not self._auto_review_active
        if self._auto_est_credits is not None and self._auto_zone_is_set and _cost_label_free:
            self.set_auto_credit_estimate(self._auto_est_credits)
        self._update_full_ui()

    def _set_credit_cost_style(self, qss: str) -> None:



        if getattr(self, "_auto_credit_cost_qss", None) == qss:
            return
        self._auto_credit_cost_qss = qss
        try:
            self.auto_credit_cost_label.setStyleSheet(qss)
        except (RuntimeError, AttributeError):
            pass

    def _auto_zone_too_large_text(self) -> str:







        fallback = tr("Zone too large - draw a smaller zone")
        try:
            from ..plugin.shared import max_tiles_per_run_cap, zone_too_large_message
            return zone_too_large_message(
                max_tiles_per_run_cap(getattr(self, "_auto_zone_km2", None)),
                fallback)
        except Exception:  # noqa: BLE001
            return fallback

    def _auto_zone_too_large_tooltip(self) -> str:



















        try:
            from ..plugin.shared import max_tiles_per_run_cap
            cap = int(max_tiles_per_run_cap(
                getattr(self, "_auto_zone_km2", None)))
        except Exception:  # noqa: BLE001
            cap = 0



        known_free = (self._auto_credits is not None
                      and not self._auto_is_subscriber)
        if cap > 0:
            if known_free:
                return dial_copy(
                    "zone.too_large_tooltip_free",
                    tr("This zone at this precision is more than one run "
                       "covers. Draw a smaller zone, or lower the precision. "
                       "Free runs stop well below that ceiling, so Pro keeps "
                       "more precision on a zone this size."),
                ).replace("{cap}", str(cap))
            return dial_copy(
                "zone.too_large_tooltip",
                tr("This zone at this precision is more than one run covers. "
                   "Draw a smaller zone, or lower the precision."),
            ).replace("{cap}", str(cap))
        if known_free:
            return dial_copy(
                "zone.too_large_tooltip_free_no_cap",
                tr("This zone at this precision is more than one run covers. "
                   "Draw a smaller zone, or lower the precision. Free runs "
                   "stop well below that ceiling, so Pro keeps more precision "
                   "on a zone this size."))
        return dial_copy(
            "zone.too_large_tooltip_no_cap",
            tr("This zone at this precision is more than one run covers. Draw "
               "a smaller zone, or lower the precision."))

    def _auto_km2_left(self) -> float | None:






        env = getattr(self, "_quota_envelopes", None)
        if env is None or not env.has_km2_gauge():
            return None
        if env.km2_remaining is not None:
            return max(0.0, float(env.km2_remaining))
        return max(0.0, float(env.km2_cap) - float(env.km2_used))

    def set_auto_zone_surface(self, km2: float | None) -> None:








        self._auto_zone_km2 = km2
        if getattr(self, "auto_credit_cost_label", None) is None:
            return
        if km2 is None or km2 <= 0:
            self.set_auto_km2_block(None)
            self._refresh_auto_cost_label()
            return
        try:


            self.auto_advanced_toggle_btn.setToolTip(tr(
                "Automatic is counted by surface. Precision changes how finely "
                "the zone is scanned, never the price. A run never costs more "
                "than the zone you drew."))
        except (RuntimeError, AttributeError):
            return
        self._refresh_auto_cost_label()
        left = self._auto_km2_left()
        if left is not None and km2 > left:
            self.set_auto_km2_block(km2, left)
        else:
            self.set_auto_km2_block(None)
        self._update_auto_detect_enabled()

    def _auto_cost_row_text(self, km2: float | None) -> str:









        if km2 is None or km2 <= 0:
            return ""
        try:
            from ...core.run_eta import friendly_run_eta_about
            from .ui_refresh import format_km2_surface
            surface = format_km2_surface(km2)
            tiles = getattr(self, "_auto_est_credits", None)
            prompt = ""
            box = getattr(self, "auto_prompt_input", None)
            if box is not None:
                prompt = (box.text() or "").strip()
            eta = (friendly_run_eta_about(
                tiles, seconds_per_tile=self._auto_quote_pace())
                if prompt and tiles is not None and tiles > 0 else "")
        except (RuntimeError, AttributeError):
            return ""
        if eta:
            return tr("{n} km² · {eta}").format(n=surface, eta=eta)
        return tr("{n} km²").format(n=surface)

    def refresh_auto_run_estimate(self) -> None:









        btn = getattr(self, "auto_detect_btn", None)
        try:
            if btn is not None and btn.text() != tr("Detect objects"):
                btn.setText(tr("Detect objects"))
        except (RuntimeError, AttributeError):
            pass
        self._refresh_auto_cost_label()

    def _refresh_auto_cost_label(self) -> None:

















        label = getattr(self, "auto_credit_cost_label", None)
        if label is None:
            return
        try:
            if self._auto_run_active or self._auto_review_active:
                return
            btn = getattr(self, "auto_detect_btn", None)
            if getattr(self, "_auto_zone_too_large", False):
                if btn is not None:
                    btn.setText(tr("Detect objects"))
                    btn.setToolTip("")
                label.setText(self._auto_zone_too_large_text())
                self._set_credit_cost_style(scale_qss_font_px(
                    f"color: {RED_TEXT}; font-size: {FONT_HINT}px;"))
                label.setToolTip(self._auto_zone_too_large_tooltip())
                label.setVisible(True)
                return
            km2 = getattr(self, "_auto_zone_km2", None)
            label.setText("")
            label.setToolTip("")
            label.setVisible(False)
            if km2 is None or km2 <= 0:
                if btn is not None:
                    btn.setText(tr("Detect objects"))
                    btn.setToolTip("")
                return



            from .ui_refresh import format_km2_surface
            if btn is not None:
                btn.setText(tr("Detect objects ({n} km²)").format(
                    n=format_km2_surface(km2)))
                btn.setToolTip(self._auto_cost_row_text(km2))
        except (RuntimeError, AttributeError):

            pass

    def set_auto_own_pace(self, seconds_per_tile: float | None) -> None:










        self._auto_own_pace_s = seconds_per_tile
        self.refresh_auto_run_estimate()

    def _auto_quote_pace(self) -> float | None:



        try:
            from ...core.run_pace_memory import own_machine_pace
            local = own_machine_pace()
        except Exception:  # noqa: BLE001
            local = None
        if local is not None:
            return local
        return getattr(self, "_auto_own_pace_s", None)

    def set_auto_km2_block(self, zone_km2: float | None,
                           left_km2: float = 0.0) -> None:










        card = getattr(self, "auto_km2_block", None)
        exceeded = zone_km2 is not None
        self._auto_km2_exceeded = exceeded
        if card is None:
            return
        try:
            if not exceeded:
                card.setVisible(False)
                return
            from .ui_refresh import format_km2_surface
            zone = capped_zone_km2_text(zone_km2)
            left = format_km2_surface(left_km2)



            free_user = (self._auto_credits is not None
                         and not self._auto_is_subscriber)
            reset_day = getattr(self, "_auto_reset_display", "")





            body = dial_copy(
                "km2_block.message",
                tr("Pro raises the month to 200 km² of Automatic."))
            title = dial_copy(
                "km2_block.title",
                tr("This zone is {zone} km². You have {left} km² left in Automatic this "
                   "month."))
            if left_km2 <= 0:


                if reset_day:
                    escape = dial_copy(
                        "km2_block.escape_spent_reset",
                        tr("No Automatic surface left this month. Semi-Auto "
                           "still works, and Automatic comes back on {date}."))
                else:
                    escape = dial_copy(
                        "km2_block.escape_spent",
                        tr("No Automatic surface left this month. Semi-Auto "
                           "still works until it comes back."))
            else:
                escape = dial_copy(
                    "km2_block.escape", tr("Or draw a smaller zone."))
            fill = (lambda text: text.replace("{zone}", zone)  # noqa: E731
                    .replace("{left}", left).replace("{date}", reset_day))



            from ...core.pro_ceiling import pro_ceiling_enabled
            pro_contact = (self._auto_is_subscriber
                           and self._auto_credits is not None
                           and pro_ceiling_enabled())




            card.set_ghost_button(pro_contact)
            self._paint_km2_redraw_fill(not free_user)
            if pro_contact:
                body, cta = self._pro_ceiling_copy()
                card.route_cta(self._on_pro_contact_km2_block)
                card.set_text(fill(title), body, cta, escape=fill(escape),
                              detail=self._pro_ceiling_detail())
            else:
                card.route_cta(self._on_upgrade_clicked)
                card.set_text(
                    fill(title),
                    fill(body) if free_user else None,


                    dial_copy("upsell.cta", keep_working_cta()),
                    escape=fill(escape),
                )
                if free_user:
                    card.set_pro_offer("plugin_km2_wall")


            self.auto_km2_block_upgrade.setVisible(free_user or pro_contact)
            card.setVisible(True)
        except (RuntimeError, AttributeError):
            return
        if free_user:
            try:
                from ...core import telemetry_session_events
                telemetry_session_events.track_pro_upsell_viewed(trigger="km2_block")
            except Exception:
                pass  # nosec B110

    def _paint_km2_redraw_fill(self, filled: bool) -> None:




        btn = getattr(self, "auto_run_block_redraw_btn", None)
        if btn is None or getattr(self, "_km2_redraw_filled", None) is bool(filled):
            return
        try:
            from qgis.PyQt.QtGui import QColor

            from ..icons import icon_for
            from .auto_flow_look import _BTN_AUTO_CHIP
            from .styles import _BTN_GREEN_STEP, BTN_PRIMARY_WIDE_PX, ON_ACCENT
            if filled:
                btn.setStyleSheet(_BTN_GREEN_STEP)
                btn.setMinimumHeight(BTN_PRIMARY_WIDE_PX)
                btn.setIcon(icon_for(btn, "polygon", 16, QColor(ON_ACCENT)))
            else:
                btn.setStyleSheet(_BTN_AUTO_CHIP)
                btn.setMinimumHeight(BTN_CHIP_PX)
                btn.setIcon(icon_for(btn, "polygon", 16))
            self._km2_redraw_filled = bool(filled)
        except (RuntimeError, AttributeError, ImportError):
            return

    def set_auto_credit_estimate(self, credits: int) -> None:














        self._auto_est_credits = credits





        if credits < 0 and not self._auto_detail_object_known():
            self._auto_zone_too_large = False
            self.set_auto_zone_fit_visible(False)
            self._refresh_auto_cost_label()
            return



        self._auto_zone_too_large = credits < 0


        self.set_auto_zone_fit_visible(self._auto_zone_too_large)




        self.refresh_auto_run_estimate()
        self._update_auto_detect_enabled()

    def set_auto_zone_rejected(self, area_km2: float | None) -> None:










        card = getattr(self, "_auto_zone_cap_label", None)
        if area_km2 is None:
            if card is not None:
                try:
                    card.setVisible(False)
                except (RuntimeError, AttributeError):
                    pass
            return
        if card is None:
            card = self._build_zone_cap_card()
            if card is None:
                return
            self._auto_zone_cap_label = card
        from ..plugin.shared import free_zone_cap_km2
        cap = f"{free_zone_cap_km2():g}"


        area = capped_zone_km2_text(area_km2)



        title = dial_copy("zone.free_cap_title", tr(
            "This zone is {area} km². Free runs stop at {max} km²."))
        body = dial_copy("zone.free_cap_body", tr(
            "Pro has no size limit and runs the zone as you drew it."))
        smaller = dial_copy("zone.free_cap_smaller", tr(
            "Or make the zone smaller and run it free."))
        fill = lambda t: t.replace("{area}", area).replace("{max}", cap)  # noqa: E731
        card.set_text(fill(title), fill(body),
                      dial_copy("upsell.cta", keep_working_cta()), fill(smaller))
        card.set_pro_offer("plugin_zone_cap")
        card.setVisible(True)


        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pro_upsell_viewed(trigger="zone_too_large")
        except Exception:
            pass  # nosec B110

    def _build_zone_cap_card(self):



        from .upsell_card import UpsellCard
        card = UpsellCard(
            "autoZoneCapCard", "full",
            on_cta=lambda: self._on_zone_cap_link_activated(
                self._build_upgrade_url("plugin_zone_cap")))
        try:
            self.auto_zone_hero.layout().addWidget(card)
        except (RuntimeError, AttributeError):
            return None
        return card

    def _on_zone_cap_link_activated(self, url: str) -> None:




        from ...core.pro_page_link import open_pro_page
        open_pro_page("plugin_zone_cap", "zone_too_large",
                      parent=self, fallback_url=url)

    def _update_auto_low_credit_note(self) -> None:










        remaining = self._auto_credits
        total = self._auto_credits_total
        show = self._mode == Mode.AUTOMATIC and self._plugin_activated
        show = show and not self._auto_is_subscriber

        show = show and not pro_card_dismissed(PRO_CARD_AUTO_LOW)



        try:
            show = show and self.auto_steps.currentIndex() == 0
        except (RuntimeError, AttributeError):
            pass  # nosec B110
        env = getattr(self, "_quota_envelopes", None)
        km2_gauge = env is not None and env.has_km2_gauge()
        if km2_gauge:



            km2_left = self._auto_km2_left()
            km2_cap = float(env.km2_cap)
            show = show and km2_left is not None
            show = show and 0 < km2_left <= km2_cap * _low_credit_threshold()
        else:
            show = show and remaining is not None and total and total > 0




            show = show and remaining <= _low_credit_ceiling()
            show = show and 0 < remaining <= total * _low_credit_threshold()
        line = getattr(self, "_auto_low_credit_line", None)
        if not show:
            if line is not None:
                try:
                    line.setVisible(False)
                except (RuntimeError, AttributeError):
                    pass
            return
        if line is None:
            line = self._build_auto_low_credit_line()
            if line is None:
                return
        line.route_cta(self._auto_low_credit_upgrade_cta)



        reset_day = getattr(self, "_auto_reset_display", "")
        if km2_gauge:


            from .ui_refresh import format_km2_left
            left = format_km2_left(km2_left)


            if reset_day:
                title = dial_copy(
                    "upsell.low_title_km2_reset",
                    tr("{n} km² of Automatic left, back on {date}."))
            else:
                title = dial_copy(
                    "upsell.low_title_km2",
                    tr("{n} km² of Automatic left this month."))
            title = title.replace("{n}", left).replace("{date}", reset_day)
        else:

            if reset_day:
                title = dial_copy(
                    "upsell.low_title_count_reset",
                    tr("{n} free cloud detections left, back on {date}."))
            else:
                title = dial_copy(
                    "upsell.low_title_count",
                    tr("{n} free cloud detections left."))
            title = (title.replace("{n}", str(remaining))
                          .replace("{date}", reset_day))



        line.set_text(title, None, keep_working_cta())
        line.set_pro_offer("plugin_low_credit_note")
        line.setVisible(True)





        if km2_gauge:
            return
        if not getattr(self, "_low_credit_note_seen", False) and remaining is not None and total:
            self._low_credit_note_seen = True
            try:
                from ...core import telemetry_session_events
                telemetry_session_events.track_low_credit_banner_viewed(int(remaining), int(total))
            except Exception:  # nosec B110
                pass

    def _build_auto_low_credit_line(self):







        layout = getattr(self, "low_credit_slot", None)
        if layout is None:
            return None
        self._auto_low_credit_upgrade_cta = (
            lambda: self._on_low_credit_link_activated(
                self._build_upgrade_url("plugin_low_credit_note")))
        card = UpsellCard(
            "autoLowCreditNote", "compact", self._auto_low_credit_upgrade_cta)

        card.enable_dismiss()
        card.dismissed.connect(
            lambda: self._dismiss_pro_card(PRO_CARD_AUTO_LOW, card))
        layout.addWidget(card)
        self._auto_low_credit_line = card
        return card

    def _on_low_credit_link_activated(self, url: str) -> None:




        from ...core.pro_page_link import open_pro_page
        open_pro_page("plugin_low_credit_note", "low_credit",
                      parent=self, fallback_url=url)

    def set_auto_exhausted_subscribe_visible(self, visible: bool) -> None:






        try:
            self.auto_exhausted_subscribe.setVisible(bool(visible))
        except (RuntimeError, AttributeError):
            return
        if visible:
            try:
                from ...core import telemetry_session_events
                telemetry_session_events.track_pro_upsell_viewed(trigger="exhausted_status")
            except Exception:
                pass  # nosec B110
