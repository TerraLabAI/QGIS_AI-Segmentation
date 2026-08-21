"""Automatic mode credits: the balance, the per-run cost estimate and its
gate, plus every upsell surface that sends the user to the dashboard.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.
"""
from __future__ import annotations

from ...core.credit_gate import low_credit_ceiling as _low_credit_ceiling
from ...core.credit_gate import low_credit_threshold as _low_credit_threshold
from ...core.i18n import tr
from ...core.server_dials import dial_copy
from .font_scale import scale_qss_font_px
from .styles import (
    ERROR_TEXT,
)
from .upsell_card import UpsellCard
from .widgets import (
    Mode,
)


class DockAutoCreditsMixin:
    """Automatic mode credits: the balance, the per-run cost estimate and its
    gate, plus every upsell surface that sends the user to the dashboard."""

    def _on_upgrade_clicked(self) -> None:
        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices
        cta_source = "plugin_upsell_card"
        try:
            from ...core import telemetry_session_events
            sender = self.sender()
            if sender is getattr(self, "_subscribe_pill", None):
                source, cta_source = "subscribe_pill", "plugin_subscribe_pill"
            elif sender is getattr(self, "auto_exhausted_subscribe_link", None):
                source, cta_source = "exhausted_status", "plugin_km2_wall"
            elif sender is getattr(self, "auto_km2_block_upgrade", None):
                source, cta_source = "km2_block", "plugin_km2_block"
            elif sender is getattr(self, "manual_credit_upgrade_btn", None):
                source, cta_source = "upsell_card", "plugin_objects_wall"
            elif sender is getattr(self, "auto_upgrade_btn", None):
                source, cta_source = "upsell_card", "plugin_free_exhausted_wall"
            else:
                source = "upsell_card"
            telemetry_session_events.track_pro_upsell_clicked(source=source)
        except Exception:
            pass  # nosec B110
        url = self._build_upgrade_url(cta_source)
        QDesktopServices.openUrl(QUrl(url))

    def _build_upgrade_url(self, cta_source: str = "plugin_upsell_card") -> str:
        from ...core.activation_manager import get_pro_checkout_url
        return get_pro_checkout_url(cta_source)

    def set_auto_envelopes(self, envelopes) -> None:
        """Store the account's two-envelope snapshot (or None) and repaint.

        ``envelopes`` is a core.quota_envelopes.QuotaEnvelopes, or None when
        the server sent no envelope fields: then every surface keeps the
        wallet display. Main thread only, like set_auto_credits.
        """
        self._quota_envelopes = envelopes
        # A new envelope can turn the drawn zone from allowed into refused (or
        # back), so the surface gate is re-run against the zone already on
        # screen instead of waiting for the next slider move.
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
        """The last stored QuotaEnvelopes snapshot, or None."""
        return getattr(self, "_quota_envelopes", None)

    def note_cloud_object_charged(self) -> None:
        """One Semi-Auto save was charged: move the objects envelope locally.

        The save response only carries the wallet gauge, and the next account
        read can be minutes away, so the count the user watches is bumped
        here and corrected by the next fetch.
        """
        env = getattr(self, "_quota_envelopes", None)
        if env is None:
            return
        used = env.objects_used + 1 if env.objects_used is not None else None
        left = (max(0, env.objects_remaining - 1)
                if env.objects_remaining is not None else None)
        self.set_auto_envelopes(env._replace(
            objects_used=used, objects_remaining=left))

    def note_cloud_objects_exhausted(self) -> None:
        """The server refused a save for the month: the objects envelope is
        spent, whatever the local count said."""
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
        """Called by plugin after loading usage data. Main thread only.

        ``total`` feeds the footer credit ring (remaining / total). Optional
        so older callers keep working; without it the ring stays hidden and
        only the count label shows.
        """
        self._auto_credits = credits
        self._auto_credits_total = total
        self._auto_is_subscriber = is_subscriber
        self._auto_reset_date = reset_date or ""
        # Format here, once per usage fetch, so no display path (tooltip,
        # note, upsell card) ever parses a date while painting.
        from ...core.quota_reset_date import format_quota_reset_date
        self._auto_reset_display = format_quota_reset_date(self._auto_reset_date)
        if not is_subscriber:
            self._auto_free_left = credits
        self._refresh_auto_credits_display()
        # A balance change (typically the post-run refresh that debits the tiles
        # just spent) must re-run the credit gate against the LAST estimate, so a
        # now-underfunded zone blocks Detect immediately instead of waiting for
        # the next slider move. Only while a zone is set and no run/review owns
        # the cost label, so this never force-shows it on the Start step.
        _cost_label_free = not self._auto_run_active and not self._auto_review_active
        if self._auto_est_credits is not None and self._auto_zone_is_set and _cost_label_free:
            self.set_auto_credit_estimate(self._auto_est_credits)
        self._update_full_ui()

    def _auto_zone_too_large_text(self) -> str:
        """The cost row's refusal, read from the served ``zone.too_large`` id.

        Same id as the log line and the MCP answer, so one deploy fixes every
        surface. The shipped fallback stays this row's own short wording: the
        row sits in the Detail header, where a longer sentence cannot wrap and
        widens the dock. Never raises: a slider move calls it.
        """
        fallback = tr("Zone too large - reduce the selection area")
        try:
            from ..plugin.shared import max_tiles_per_run_cap, zone_too_large_message
            return zone_too_large_message(max_tiles_per_run_cap(), fallback)
        except Exception:  # noqa: BLE001 -- served copy is best-effort
            return fallback

    def _auto_zone_too_large_tooltip(self) -> str:
        """The numbers the cost row has no width for.

        The row sits in the Detail header and can only carry "Zone too large",
        so the ceiling and the two ways back under it live here. The count
        itself never reaches the dock: the grid stops counting once it passes
        the ceiling. Never raises: a slider move calls it.

        Four served ids, one per form: free or subscriber, ceiling known or
        not. ``max_tiles_per_run_cap()`` is the same on every plan, so this
        refusal fires at the same size for everyone. A free zone is separately
        bounded by its own km² cap, which is why the free form still talks
        about precision. No shipped sentence quotes the ceiling, because it is
        a tile count and the user is charged by surface. The {cap} fill stays
        on the cap-known branch so a served
        override MAY quote it where it is meaningful. str.replace, never
        format(), so a stray brace in a served sentence cannot raise on a
        slider move.
        """
        try:
            from ..plugin.shared import max_tiles_per_run_cap
            cap = int(max_tiles_per_run_cap())
        except Exception:  # noqa: BLE001 -- a tooltip must never break a paint
            cap = 0
        # Known-free only. Before the usage fetch lands the tier is UNKNOWN,
        # not free (same rule as the zone-area guard), so a subscriber never
        # reads an upsell while their plan is still loading.
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
        """km² of Automatic left on the account this month, or None.

        None whenever the account did not tell us both figures. A gate that
        fires on an unknown envelope refuses a run the server would have
        accepted, so every caller fails open on None.
        """
        env = getattr(self, "_quota_envelopes", None)
        if env is None or not env.has_km2_gauge():
            return None
        if env.km2_remaining is not None:
            return max(0.0, float(env.km2_remaining))
        return max(0.0, float(env.km2_cap) - float(env.km2_used))

    def set_auto_zone_surface(self, km2: float | None) -> None:
        """The surface of the drawn zone, in km², or None when there is none.

        This is what an Automatic run is billed on, so it is the number the
        Precision header carries. Precision itself changes how finely the zone
        is scanned and never what it costs, which is why no tile count reaches
        this row any more. Also runs the monthly-envelope gate, because the
        zone and the balance are the two figures it compares.
        """
        self._auto_zone_km2 = km2
        label = getattr(self, "auto_credit_cost_label", None)
        if label is None:
            return
        if km2 is None or km2 <= 0:
            self.set_auto_km2_block(None)
            try:
                label.setText("")
                label.setToolTip("")
                label.setVisible(False)
            except (RuntimeError, AttributeError):
                pass
            return
        from .ui_refresh import format_km2_surface
        try:
            label.setText(tr("{n} km²").format(n=format_km2_surface(km2)))
            label.setToolTip(tr(
                "Automatic is counted by surface. Precision changes how finely "
                "the zone is scanned, never the price. A run never costs more "
                "than the zone you drew."))
            label.setVisible(True)
        except (RuntimeError, AttributeError):
            return
        left = self._auto_km2_left()
        if left is not None and km2 > left:
            self.set_auto_km2_block(km2, left)
        else:
            self.set_auto_km2_block(None)
        self._update_auto_detect_enabled()

    def set_auto_km2_block(self, zone_km2: float | None,
                           left_km2: float = 0.0) -> None:
        """Show (``zone_km2`` set) or hide (None) the offer card under the
        Precision header: the drawn zone is bigger than the surface the
        account has left this month.

        The shared offer card (upsell_card.py), the same shape as the zone-cap
        refusal in the draw hero, and it blocks Detect the same way. Only
        called with a zone when BOTH figures are known. A subscriber cannot buy
        their way past this one, so they read the fact and the way out, never
        the offer.
        """
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
            zone = format_km2_surface(zone_km2)
            left = format_km2_surface(left_km2)
            # Known-free only, the same rule as the zone tooltip: before the
            # usage fetch lands the tier is unknown, not free, and a subscriber
            # must never read an offer while their plan is still loading.
            free_user = (self._auto_credits is not None
                         and not self._auto_is_subscriber)
            reset_day = getattr(self, "_auto_reset_display", "")
            # Served copy for the Pro line: this refusal quotes the monthly
            # envelope, a commercial figure that can move any week. The two
            # fills stay on it, so a served sentence that still carries them
            # renders its numbers. str.replace, never format(), so a stray
            # brace in a served sentence cannot raise on the draw path.
            body = dial_copy(
                "km2_block.message",
                tr("Pro raises the month to 300 km² of Automatic."))
            title = dial_copy(
                "km2_block.title",
                tr("This zone is {zone} km². You have {left} km² left in Automatic this "
                   "month."))
            if left_km2 <= 0:
                # Nothing left this month, so no zone is small enough. The way
                # out is the other mode, or the day the month turns over.
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
            card.set_text(
                fill(title),
                fill(body) if free_user else None,
                dial_copy("upsell.cta", tr("Upgrade to Pro")),
                escape=fill(escape),
            )
            # The button is the one part a subscriber must not see. The way out
            # stays: it is the only move they have left on this card.
            self.auto_km2_block_upgrade.setVisible(free_user)
            card.setVisible(True)
        except (RuntimeError, AttributeError):
            return
        if free_user:
            try:
                from ...core import telemetry_session_events
                telemetry_session_events.track_pro_upsell_viewed(trigger="km2_block")
            except Exception:
                pass  # nosec B110

    def set_auto_credit_estimate(self, credits: int) -> None:
        """The tile estimate for the drawn zone (-1 = past the per-run cap).

        It does not price the run and it no longer gates one. Automatic is
        billed by the SURFACE of the zone (``set_auto_zone_surface`` owns that,
        and the monthly wall with it), while this number counts tiles, which
        precision moves and the bill never follows. Comparing the two refused
        runs that fitted: a 0.26 km2 zone with 0.5 km2 left was blocked because
        its grid held more tiles than the wallet held credits.

        What is left here is the MAX_TILES refusal, which is a real property of
        the tile count, and the styling it puts on the surface label.
        """
        # Remember the estimate: the detail-change telemetry reads it, and a
        # later balance change re-runs this method (see set_auto_credits).
        self._auto_est_credits = credits
        if credits < 0:
            self.auto_credit_cost_label.setText(self._auto_zone_too_large_text())
            self.auto_credit_cost_label.setStyleSheet(scale_qss_font_px(
                f"color: {ERROR_TEXT}; font-size: 11px;"))
            # By how much. The row can only fit "Zone too large", so the
            # tooltip carries the two numbers that say what a smaller zone has
            # to reach. Clearing it left nothing anywhere on screen saying it.
            self.auto_credit_cost_label.setToolTip(
                self._auto_zone_too_large_tooltip())
            self._auto_zone_too_large = True
        else:
            self.auto_credit_cost_label.setStyleSheet(scale_qss_font_px(
                "color: palette(text); font-size: 11px;"))
            self._auto_zone_too_large = False
            self.auto_credit_cost_label.setVisible(
                bool(self.auto_credit_cost_label.text()))
            self._update_auto_detect_enabled()
            return
        self.auto_credit_cost_label.setVisible(True)
        self._update_auto_detect_enabled()

    def set_auto_zone_rejected(self, area_km2: float | None) -> None:
        """Show (or hide with None) the free zone-cap card in the step-1 draw
        hero: the drawn zone was refused because it exceeds the free cap.

        This is the upsell a free user meets most, so it is a card, not a
        paragraph: a bold one-line fact (the two numbers), one line on what Pro
        removes, a filled Upgrade button, and the other way out in muted text.
        The old single amber label put the numbers, the pitch and the link in
        one four-line run, and the link was the smallest thing on it.
        The card clears as soon as a valid zone lands or the flow is exited
        (see set_auto_zone_state / reset_auto_to_start)."""
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
        from .ui_refresh import format_km2_surface
        cap = f"{free_zone_cap_km2():g}"
        # Same rounding as the cost row, so the refusal and the price name the
        # same surface. One decimal alone turned a small zone into "0.0 km²".
        area = format_km2_surface(area_km2)
        # Served copy for the two text lines, so a bad refusal is one deploy
        # away from fixed. str.replace, never format: a served brace must not
        # raise on the draw path.
        title = dial_copy("zone.free_cap_title", tr(
            "This zone is {area} km². Free runs stop at {max} km²."))
        body = dial_copy("zone.free_cap_body", tr(
            "Pro has no size limit and runs the zone as you drew it."))
        smaller = dial_copy("zone.free_cap_smaller", tr(
            "Or make the zone smaller and run it free."))
        fill = lambda t: t.replace("{area}", area).replace("{max}", cap)  # noqa: E731
        card.set_text(fill(title), fill(body),
                      dial_copy("upsell.cta", tr("Upgrade to Pro")), fill(smaller))
        card.setVisible(True)
        # Impression tracked so the click has a denominator. Deduped per
        # trigger, so once per session.
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pro_upsell_viewed(trigger="zone_too_large")
        except Exception:
            pass  # nosec B110

    def _build_zone_cap_card(self):
        """The zone-cap card, built once on first refusal: the shared premium
        UpsellCard in its full form. Returns None when the hero is not there
        to hold it."""
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
        """Subscribe link inside the zone-cap message: same destination as the
        footer pill, tracked with its own upsell source."""
        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pro_upsell_clicked(source="zone_too_large")
        except Exception:
            pass  # nosec B110
        QDesktopServices.openUrl(QUrl(url))

    def _update_auto_low_credit_note(self) -> None:
        """Free-tier low-credit nudge on the Automatic Start step (step 0).

        Shows a compact offer card, one line and a button, once a free user
        drops under the low-credit share of their free detections (and still
        has some left; a fully exhausted balance shows the wall instead). The
        share comes from credit_gate, so this note appears at the moment the
        footer credit ring turns amber.
        Subscribers never see it: the footer credit ring owns their balance.
        The card lives on the step-0 page, so it only shows on Start.
        """
        remaining = self._auto_credits
        total = self._auto_credits_total
        show = self._mode == Mode.AUTOMATIC and self._plugin_activated
        show = show and not self._auto_is_subscriber
        # The line sits at the foot of the dock now, not on the step-0 page, so
        # the page no longer hides it when the user moves on. Kept to Start on
        # purpose: mid-run and in the review the balance is not a decision.
        try:
            show = show and self.auto_steps.currentIndex() == 0
        except (RuntimeError, AttributeError):
            pass  # nosec B110 -- no stack yet means no run either
        env = getattr(self, "_quota_envelopes", None)
        km2_gauge = env is not None and env.has_km2_gauge()
        if km2_gauge:
            # The line prints a surface, so it has to be decided on one too.
            # The ceiling below counts detections and means nothing here, and
            # the wallet figure it reads is not what this card shows.
            km2_left = self._auto_km2_left()
            km2_cap = float(env.km2_cap)
            show = show and km2_left is not None
            show = show and 0 < km2_left <= km2_cap * _low_credit_threshold()
        else:
            show = show and remaining is not None and total and total > 0
            # Both rules, and the ceiling first: a fifth of a large allowance is
            # still dozens of runs, so the share alone started selling far too
            # early. Read from credit_gate, so Semi auto's line turns on at the
            # same moment (`_manual_engine_credits_low`).
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
        # Naming the renewal day turns "you are running out" into a choice
        # between waiting and paying. The date-free wording stays for servers
        # that send no period_end.
        reset_day = getattr(self, "_auto_reset_display", "")
        if km2_gauge:
            # Envelope-aware wording: the free Automatic allowance is an area,
            # and every gauge names what is left, never what is spent.
            from .ui_refresh import format_km2_left
            left = format_km2_left(km2_left)
            # Served like the offer line under it: the count sentence is what
            # a free user reads most, and it must be rewordable in a deploy.
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
            # Legacy accounts, counted in detections rather than in ground.
            if reset_day:
                title = dial_copy(
                    "upsell.low_title_count_reset",
                    tr("{n} free detections left, back on {date}."))
            else:
                title = dial_copy(
                    "upsell.low_title_count",
                    tr("{n} free detections left."))
            title = (title.replace("{n}", str(remaining))
                          .replace("{date}", reset_day))
        # Same rule as the Semi-Auto twin: the count says what is running out,
        # this line says what the other side holds. Served, because the Pro
        # allowance moves without waiting for a plugin release.
        body = dial_copy(
            "upsell.low_body_km2",
            tr("Pro gives you 300 km² of Automatic a month, so you keep "
               "working."))
        line.set_text(title, body, dial_copy("upsell.cta",
                                             tr("Upgrade to Pro")))
        line.setVisible(True)
        # Track the banner view once per session (the click was already
        # tracked, the view was not). Never on the surface envelope: the event
        # carries a detection count and its total, and the wallet figures this
        # object holds are not what the surface card is about. Reporting them
        # here would read as a balance nobody was shown.
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
        """Lazily create the running-low nudge and seat it in the dock's bottom
        slot, beside the credit ring it is talking about.

        The compact offer card (upsell_card.py): one bold line and an outline
        button on the same row, because this one is a nudge the user is free to
        ignore, not a refusal. Returns the card, or None if the slot is not
        built yet."""
        layout = getattr(self, "low_credit_slot", None)
        if layout is None:
            return None
        card = UpsellCard(
            "autoLowCreditNote", "compact",
            lambda: self._on_low_credit_link_activated(
                self._build_upgrade_url("plugin_low_credit_note")))
        layout.addWidget(card)
        self._auto_low_credit_line = card
        return card

    def _on_low_credit_link_activated(self, url: str) -> None:
        """Upgrade button on the low-credit note: same destination as the
        footer pill, tracked with its own upsell source."""
        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pro_upsell_clicked(source="low_credit")
        except Exception:
            pass  # nosec B110
        QDesktopServices.openUrl(QUrl(url))

    def set_auto_exhausted_subscribe_visible(self, visible: bool) -> None:
        """Show/hide the free-user offer shown under the status when a run
        stops on an exhausted allowance (Moment C).

        Reports its own impression: the surface had a click event and no
        matching view, so its conversion could not be told apart from the
        pill's."""
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
