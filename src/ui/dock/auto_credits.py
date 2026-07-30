"""Automatic mode credits: the balance, the per-run cost estimate and its
gate, plus every upsell surface that sends the user to the dashboard.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import Qt

from ...core.credit_gate import insufficient as _credit_insufficient
from ...core.credit_gate import low_credit_threshold as _low_credit_threshold
from ...core.i18n import tr
from .font_scale import scale_qss_font_px
from .styles import (
    ERROR_TEXT,
    _msg_label_qss,
    _msg_text,
)
from .widgets import (
    Mode,
)


class DockAutoCreditsMixin:
    """Automatic mode credits: the balance, the per-run cost estimate and its
    gate, plus every upsell surface that sends the user to the dashboard."""

    def _on_upgrade_clicked(self) -> None:
        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices
        try:
            from ...core import telemetry_session_events
            sender = self.sender()
            if sender is getattr(self, "_subscribe_pill", None):
                source = "subscribe_pill"
            elif sender is getattr(self, "auto_exhausted_subscribe_link", None):
                source = "exhausted_status"
            elif sender is getattr(self, "auto_credit_block_upgrade", None):
                source = "credit_block"
            else:
                source = "upsell_card"
            telemetry_session_events.track_pro_upsell_clicked(source=source)
        except Exception:
            pass  # nosec B110
        url = self._build_upgrade_url()
        QDesktopServices.openUrl(QUrl(url))

    def _build_upgrade_url(self) -> str:
        from ...core.activation_manager import get_dashboard_url
        # DASHBOARD_URL already carries its own UTM query string: strip it
        # before appending ours, or the URL ends up with two "?" and the
        # upsell attribution is swallowed into the previous utm_content value.
        base = get_dashboard_url().split("?")[0]
        return (
            f"{base}?utm_source=qgis&utm_medium=plugin"
            "&utm_campaign=ai-segmentation-pro&utm_content=upsell_card"
        )

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

    def set_auto_credit_estimate(self, credits: int) -> None:
        # Remember the estimate so a later balance change (e.g. after a run
        # consumes credits) can re-run the gate against it, see set_auto_credits.
        self._auto_est_credits = credits
        if credits < 0:
            self.auto_credit_cost_label.setText(self._auto_zone_too_large_text())
            self.auto_credit_cost_label.setStyleSheet(scale_qss_font_px(
                f"color: {ERROR_TEXT}; font-size: 11px;"))
            self.auto_credit_cost_label.setToolTip("")
            self._auto_zone_too_large = True
            self._auto_insufficient_credits = False
            self.set_auto_credit_block(None)
            self._set_auto_premium_gated(False)
        else:
            # Make the per-tile billing explicit right before Detect: the run
            # scans the zone tile by tile and spends 1 credit per tile, so the
            # count reads as the equation "N tiles = N credits" (same N on
            # purpose - that IS the lesson). The footer credit ring owns the
            # remaining balance, so no "M left" suffix here.
            if credits == 1:
                text = tr("≈ 1 tile = 1 credit")
            else:
                text = tr("≈ {n} tiles = {n} credits").format(n=credits)
            remaining = (self._auto_credits if self._auto_is_subscriber
                         else self._auto_free_left)
            # Hard credit gate: a run may never launch under-funded. When the
            # estimate exceeds the known balance, block Detect, turn the cost
            # line red and raise the callout above Detect that carries the
            # fix-it instruction. This replaces the old amber "will stop after
            # N" partial-run allowance, which let a run burn straight down to 0
            # and stop mid-zone.
            # credit_gate.insufficient owns the boundary: block only when the
            # estimate STRICTLY exceeds the balance (== is allowed), the same
            # rule as the auto_run pre-submit re-gate.
            insufficient = _credit_insufficient(credits, remaining)
            self._auto_insufficient_credits = insufficient
            # Free-plan per-run cap: the slider deliberately keeps its full
            # (Pro) travel, so past the cap the run is blocked HERE, with the
            # upgrade as the named fix. The balance gate wins when both apply
            # (an underfunded run can never launch regardless of plan).
            cap = self._auto_free_run_cap
            self._set_auto_premium_gated(
                not insufficient and not self._auto_is_subscriber and cap is not None and credits > cap)
            # The sentence and the upgrade link live in the callout above
            # Detect, never on this row.
            self.set_auto_credit_block(
                credits if insufficient else None,
                int(remaining) if remaining is not None else 0)
            if insufficient:
                # The cost line keeps the SHORT equation and only turns red.
                # It sits on the Detail header row, where a full sentence
                # cannot wrap: it dragged the panel's minimum width past a
                # normal dock and the page scrolled sideways.
                self.auto_credit_cost_label.setStyleSheet(scale_qss_font_px(
                    f"color: {ERROR_TEXT}; font-weight: bold; font-size: 11px;"))
            elif self._auto_premium_gated:
                # Red and starless, exactly like the not-enough-credits case:
                # both block Detect, so both have to look blocked. The blue star
                # read as an offer and the refusal went unnoticed. The line
                # keeps the SHORT equation (it sits on the Detail header row, a
                # long sentence would widen the dock); the box under the slider
                # carries the explanation and the upgrade link.
                self.auto_credit_cost_label.setStyleSheet(scale_qss_font_px(
                    f"color: {ERROR_TEXT}; font-weight: bold; font-size: 11px;"))
            else:
                self.auto_credit_cost_label.setStyleSheet(scale_qss_font_px(
                    "color: palette(text); font-size: 11px;"))
            self.auto_credit_cost_label.setText(text)
            _base_tip = tr(
                "Automatic mode scans your zone tile by tile. 1 tile = 1 credit, "
                "so this run costs about {n} credits. More precision splits the "
                "zone into more tiles, which costs more credits.").format(n=credits)
            _extra_tip = tr("1 credit ~ 0.17 km² at default precision.")
            # Say which way the estimate can be wrong. It counts every tile of
            # the grid, and tiles the layer holds no image for are dropped
            # before they are sent, so the run can cost less than this and never
            # more. Without the sentence a run quoted 40 and billed 32 looks
            # like a billing bug rather than the guard doing its job.
            _skip_tip = tr(
                "Tiles your layer has no image for are dropped before they are "
                "sent, so a run can cost less than this, never more.")
            self.auto_credit_cost_label.setToolTip(
                _base_tip + " " + _extra_tip + " " + _skip_tip)
            self._auto_zone_too_large = False
        self.auto_credit_cost_label.setVisible(True)
        self._update_auto_detect_enabled()

    def set_auto_credit_block(self, tiles: int | None, left: int = 0) -> None:
        """Show (``tiles`` set) or hide (``tiles`` None) the red callout that
        sits under the detail slider, right above Detect, when the run costs
        more credits than the balance covers.

        It is a card, not a line on the Detail header row: the sentence wraps
        over as many lines as the dock is narrow, so a small dock never widens
        the panel. The upgrade link shows for a free user only; a subscriber is
        already paying and can only pull the detail and zone levers.
        """
        card = getattr(self, "auto_credit_block", None)
        if card is None:
            return
        try:
            if tiles is None:
                card.setVisible(False)
                return
            self.auto_credit_block_label.setText(tr(
                "Not enough credits: {n} tiles, only {left} left. "
                "Lower the precision or the zone.").format(n=int(tiles), left=int(left)))
            free_user = not self._auto_is_subscriber
            self.auto_credit_block_upgrade.setVisible(free_user)
            card.setVisible(True)
        except (RuntimeError, AttributeError):
            return
        if free_user:
            # Same gap the other upsell surfaces closed: track the impression
            # so the click has a denominator. Deduped per trigger by telemetry.
            try:
                from ...core import telemetry_session_events
                telemetry_session_events.track_pro_upsell_viewed(trigger="credit_block")
            except Exception:
                pass  # nosec B110

    def set_auto_zone_rejected(self, area_km2: float | None) -> None:
        """Show (or hide with None) the free-trial zone-cap message in the
        step-1 draw hero: the drawn zone was refused because it exceeds
        FREE_TRIAL_MAX_ZONE_KM2. Contextual upsell touchpoint: the subscribe
        link opens the same dashboard upgrade URL as the footer pill. The
        message clears as soon as a valid zone lands or the flow is exited
        (see set_auto_zone_state / reset_auto_to_start)."""
        label = getattr(self, "_auto_zone_cap_label", None)
        if area_km2 is None:
            if label is not None:
                try:
                    label.setVisible(False)
                except (RuntimeError, AttributeError):
                    pass
            return
        if label is None:
            from qgis.PyQt.QtWidgets import QLabel
            label = QLabel()
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextBrowserInteraction)
            # Quiet warning card: translucent amber tint from the message
            # taxonomy, readable text on both themes (palette(text)).
            label.setStyleSheet(_msg_label_qss("warning"))
            label.linkActivated.connect(self._on_zone_cap_link_activated)
            try:
                self.auto_zone_hero.layout().addWidget(label)
            except (RuntimeError, AttributeError):
                return
            self._auto_zone_cap_label = label
        # Both lines come from the served ``zone.free_cap`` id, the same one the
        # MCP and headless answers read, so a confusing refusal is one deploy
        # away from fixed on every surface at once.
        from ..plugin.shared import zone_over_free_cap_lines
        line1, line2 = zone_over_free_cap_lines(
            area_km2, self._build_upgrade_url())
        label.setText("{}<br/>{}".format(_msg_text("warning", line1), line2))
        label.setVisible(True)
        # Same gap as the pill and the exhausted link: this card had a click
        # event and no impression, so its refusal-to-subscribe rate was not
        # measurable. Deduped per trigger, so once per session.
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pro_upsell_viewed(trigger="zone_too_large")
        except Exception:
            pass  # nosec B110

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

        Shows a discreet one-line "Running low" note with a Subscribe link once
        a free user drops under the low-credit share of their free detections
        (and still has some left; a fully exhausted balance shows the upsell
        card instead). The share comes from credit_gate, so this note and the
        footer credit ring turn amber at the same moment.
        Subscribers never see it: the footer credit ring owns their balance.
        The line lives on the step-0 page, so it only shows on Start.
        """
        remaining = self._auto_credits
        total = self._auto_credits_total
        show = self._mode == Mode.AUTOMATIC and self._plugin_activated
        show = show and not self._auto_is_subscriber
        show = show and remaining is not None and total and total > 0
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
        if reset_day:
            line.setText(tr(
                'Running low: {n} free detections left, back on {date}. '
                '<a href="{url}">Upgrade to Pro</a> to keep going.'
            ).format(n=remaining, date=reset_day,
                     url=self._build_upgrade_url()))
        else:
            line.setText(tr(
                'Running low: {n} free detections left. '
                '<a href="{url}">Upgrade to Pro</a> to keep going.'
            ).format(n=remaining, url=self._build_upgrade_url()))
        line.setVisible(True)
        # Track the banner view once per session (the click was already
        # tracked, the view was not).
        if not getattr(self, "_low_credit_note_seen", False):
            self._low_credit_note_seen = True
            try:
                from ...core import telemetry_session_events
                telemetry_session_events.track_low_credit_banner_viewed(int(remaining), int(total))
            except Exception:  # nosec B110
                pass

    def _build_auto_low_credit_line(self):
        """Lazily create the step-0 low-credit note (amber card) and slot it
        under the free-trial line. Returns the label, or None if the step-0
        page is not built yet."""
        from qgis.PyQt.QtWidgets import QLabel
        try:
            page = self.auto_steps.widget(0)
            layout = page.layout()
        except (RuntimeError, AttributeError):
            return None
        if layout is None:
            return None
        label = QLabel()
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        label.setOpenExternalLinks(False)
        label.setStyleSheet(_msg_label_qss("warning"))
        label.linkActivated.connect(self._on_low_credit_link_activated)
        anchor = getattr(self, "auto_start_caption", None)
        idx = layout.indexOf(anchor) if anchor is not None else -1
        if idx >= 0:
            layout.insertWidget(idx + 1, label)
        else:
            layout.addWidget(label)
        self._auto_low_credit_line = label
        return label

    def _on_low_credit_link_activated(self, url: str) -> None:
        """Subscribe link inside the low-credit note: same destination as the
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
        """Show/hide the free-user 'Subscribe to finish this zone' link shown
        under the status when a run stops on exhausted credits (Moment C).

        Reports its own impression: the link had a click event and no matching
        view, so its conversion could not be told apart from the pill's."""
        try:
            self.auto_exhausted_subscribe_link.setVisible(bool(visible))
        except (RuntimeError, AttributeError):
            return
        if visible:
            try:
                from ...core import telemetry_session_events
                telemetry_session_events.track_pro_upsell_viewed(trigger="exhausted_status")
            except Exception:
                pass  # nosec B110
