"""What the Automatic page shows when the run cannot happen at all.

One rule for the whole family of refusal cards: **a gate nothing on this page
can clear takes the page.** The card that names the gate is then the only
thing on screen, with one way out under it. A prompt box, an example row, a
precision fold and a green button the user cannot use are not context, they
are a form with an error stuck to it, and every one of them asks for work
that will be thrown away.

Four gates qualify, and they are the four ``_update_auto_detect_enabled``
already refuses on for reasons outside this page:

- ``kill_switch``: the service is off for the whole fleet.
- ``credits``: the account is known to have nothing left to spend.
- ``km2_envelope``: the drawn zone is bigger than the month's surface left.

What does NOT take the page: an unticked consent box, an unpicked layer, an
empty prompt, and a zone past the tile ceiling. Those are an unfinished form,
and the control that finishes them is on the page they would hide. The last
one is counted as a refusal all the same, it just does not get the page.

The free end-of-allowance wall (``auto_upsell_card``) is the fifth member of
the family and already owned the page before this file existed. It keeps its
own seat; ``_refresh_auto_run_block`` stands down for it, so the two can
never both be up.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py); methods
here are plain mixin members and widgets live on the dock instance.
"""
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
from .styles import _BTN_CHIP, _BTN_LINK_MUTED
from .upsell_card import UpsellCard
from .widgets import Mode

# Gates that refuse a run but are NOT worth the page. `zone_too_large` is the
# only one: the Precision slider clears it, the slider is on the page, and the
# tile ceiling now follows the zone drawn, so the slider's own top is the
# ceiling and a user can barely reach this state at all.
_PAGE_EXEMPT = frozenset({"zone_too_large"})


def _takes_the_page(reason: str) -> bool:
    """Whether ``reason`` is a dead end rather than an unfinished form."""
    return reason not in _PAGE_EXEMPT


class DockAutoRunBlockMixin:
    """The takeover card, its reason, and the two ways out under it."""

    def _setup_auto_run_block(self, parent_layout) -> None:
        """Build the takeover, hidden, above the controls it replaces.

        Two cards live in it and exactly one shows at a time: the monthly
        surface wall, which auto_credits.py fills with its own served copy,
        and the general card this file fills for the other three gates. One
        holder, so a single visibility rule covers the family.
        """
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # The monthly surface wall. It is built here rather than in the
        # Precision fold it used to sit in: a card that takes the page cannot
        # be a child of a control the takeover hides. Its copy, its Pro
        # routing and its telemetry stay in auto_credits.set_auto_km2_block.
        self.auto_km2_block = UpsellCard(
            "autoKm2Block", "full", self._on_upgrade_clicked)
        # The button keeps its old attribute name: the upgrade handler tells
        # this surface apart by the sender's identity.
        self.auto_km2_block_upgrade = self.auto_km2_block.button
        self.auto_km2_block.setVisible(False)
        layout.addWidget(self.auto_km2_block)

        # The other three gates share one card. They are refusals, not offers,
        # so the button is whatever the user's next move actually is: buy,
        # write to us, redraw, or switch mode.
        self.auto_run_block_card = UpsellCard("autoRunBlockCard", "full")
        self.auto_run_block_card.setVisible(False)
        layout.addWidget(self.auto_run_block_card)

        # Under the card, the way back to the zone step. A chip, because the
        # card above already carries the primary move.
        self.auto_run_block_redraw_btn = QPushButton(dial_copy(
            "run_block.redraw_cta", tr("Draw a smaller zone")))
        self.auto_run_block_redraw_btn.setMinimumHeight(30)
        self.auto_run_block_redraw_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_run_block_redraw_btn.setStyleSheet(_BTN_CHIP)
        self.auto_run_block_redraw_btn.clicked.connect(
            self._on_auto_run_block_redraw)
        layout.addWidget(self.auto_run_block_redraw_btn)

        # And the way out of the flow. A screen with no exit is the one thing
        # a takeover must never be. A quiet centred link, like the run's own
        # Cancel: two outline buttons stacked read as two equal choices, and
        # leaving is not the equal of fixing.
        self.auto_run_block_exit_btn = QPushButton(tr("Exit"))
        self.auto_run_block_exit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_run_block_exit_btn.setStyleSheet(_BTN_LINK_MUTED)
        self.auto_run_block_exit_btn.clicked.connect(
            self.auto_exit_requested.emit)
        exit_row = QHBoxLayout()
        exit_row.setContentsMargins(0, 0, 0, 0)
        exit_row.addStretch(1)
        exit_row.addWidget(self.auto_run_block_exit_btn, 0)
        exit_row.addStretch(1)
        layout.addLayout(exit_row)

        # The auto page absorbs the panel height; the takeover must not
        # stretch with it, or the card floats in the middle of a tall panel.
        holder.setSizePolicy(QSizePolicy.Policy.Preferred,
                             QSizePolicy.Policy.Maximum)
        holder.setVisible(False)
        self.auto_run_block = holder
        parent_layout.addWidget(holder)

    # -- state ----------------------------------------------------------------

    def _auto_balance_spent(self) -> bool:
        """True only when the account is KNOWN to have nothing left to run.

        An unread balance is not a spent one. A gate that fires on an unknown
        envelope refuses a run the server would have taken, so both branches
        fail open on None.
        """
        try:
            if self._auto_is_subscriber:
                return (self._auto_credits is not None
                        and self._auto_credits <= 0)
            return (self._auto_free_left is not None
                    and self._auto_free_left <= 0)
        except (AttributeError, TypeError, ValueError):
            return False

    def _auto_service_available(self) -> bool:
        """Whether the server still offers Automatic to the fleet. Fails open:
        an unreadable switch must never refuse a run."""
        try:
            from ...core.activation_manager import is_automatic_mode_enabled

            return bool(is_automatic_mode_enabled())
        except Exception:  # noqa: BLE001 -- a dead switch must not block a run
            return True

    def _auto_run_block_reason(self) -> str | None:
        """Why an Automatic run cannot happen at all right now, or None.

        The single source for both the takeover and the ``detect_blocked``
        signal, so the card on screen and the event in the funnel can never
        name different reasons. Ordered by how far the fix is from the user:
        the service first, then the account, then the zone they drew.
        """
        try:
            if not self._plugin_activated or self._mode != Mode.AUTOMATIC:
                return None
            if self._auto_run_active or self._auto_review_active:
                return None
            if not self._auto_service_available():
                return "kill_switch"
            if not getattr(self, "_auto_started", False):
                # Nothing is drawn yet, so nothing but the service switch can
                # refuse. The Start step stays the Start step.
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

    # -- painting -------------------------------------------------------------

    def _refresh_auto_run_block(self, suppressed: bool = False) -> bool:
        """Show the takeover for the current gate, and say whether it owns the
        page. ``suppressed`` is the free end-of-allowance wall claiming the
        seat first: two takeovers on one page is one too many."""
        holder = getattr(self, "auto_run_block", None)
        if holder is None:
            return False
        try:
            reason = None if suppressed else self._auto_run_block_reason()
            if reason is not None and not _takes_the_page(reason):
                # Still a refusal, still counted, but the control that clears
                # it is on the page: taking the page would hide the fix.
                self._note_detect_blocked(reason)
                reason = None
            if reason is None:
                holder.setVisible(False)
                return False
            show_redraw = self._fill_auto_run_block_card(reason)
            self.auto_run_block_redraw_btn.setVisible(show_redraw)
            # Before Start there is no flow to leave, and the exit would send
            # the user to the step they are already on.
            self.auto_run_block_exit_btn.setVisible(
                bool(getattr(self, "_auto_started", False)))
            holder.setVisible(True)
        except (RuntimeError, AttributeError):
            return False
        self._note_detect_blocked(reason)
        return True

    def _apply_auto_run_block_takeover(self) -> None:
        """Track the gates without a whole page pass.

        Every path that moves a gate (a zone drawn, a balance fetched, a
        precision moved, the kill switch flipping) ends in
        ``_update_auto_detect_enabled``, so the takeover is applied there and
        needs no call site of its own. The free end-of-allowance wall is left
        alone: ``_update_auto_page_state`` owns that seat.
        """
        try:
            if not self._plugin_activated or self._mode != Mode.AUTOMATIC:
                # Semi-Auto owns its own page. Touching the Automatic section
                # from here would move a widget on a page nobody is looking at,
                # and _update_auto_page_state repaints it on the way back.
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
        """Fill the card for ``reason``. Returns whether the redraw chip
        belongs under it.

        One card carries one primary move. The chip only appears when the
        card's own button is something else, so "Draw a smaller zone" is
        never on screen twice.
        """
        km2 = getattr(self, "auto_km2_block", None)
        card = self.auto_run_block_card
        if reason == "km2_envelope":
            # auto_credits.set_auto_km2_block already wrote every line of it,
            # including the Pro routing and the contact address.
            card.setVisible(False)
            if km2 is not None:
                km2.setVisible(True)
            # Nothing is small enough once the month is fully spent, so the
            # chip would point at a zone that cannot exist.
            left = self._auto_km2_left()
            show_redraw = left is not None and left > 0
            if km2 is not None and show_redraw:
                # The chip says this as a control. Left up, the card's own
                # line is the same sentence twice, one of them greyed out.
                km2.escape.setVisible(False)
            return show_redraw
        if km2 is not None:
            km2.setVisible(False)
        card.setVisible(True)
        if reason == "kill_switch":
            return self._fill_auto_block_kill_switch(card)
        return self._fill_auto_block_credits(card)

    def _fill_auto_block_kill_switch(self, card) -> bool:
        """The service is off. Nothing the user owns is wrong, so the card
        says when to come back and hands them the mode that still runs."""
        # Neutral, not the Pro tint: this card sells nothing, and a premium
        # box over an outage reads as an offer the reader cannot find.
        card.set_tint("neutral")
        card.route_cta(self._on_auto_upsell_manual_clicked)
        card.set_text(
            dial_copy("run_block.offline_title",
                      tr("Automatic is unavailable right now.")),
            dial_copy("run_block.offline_body",
                      tr("Try again in a few minutes. Your zone and your "
                         "settings are kept.")),
            dial_copy("upsell.manual_cta", tr("Use Semi-Auto")),
        )
        return False

    def _fill_auto_block_credits(self, card) -> bool:
        """The month is spent. A subscriber cannot buy past it, so the button
        writes to us; a free account still has the offer."""
        from ...core.pro_ceiling import pro_ceiling_enabled

        reset_day = getattr(self, "_auto_reset_display", "")
        subscriber = bool(getattr(self, "_auto_is_subscriber", False))
        # Served, and filled with str.replace: format() on a served sentence
        # raises on a stray brace, and this one paints the page.
        title = dial_copy(
            "run_block.credits_title",
            tr("You used your Automatic surface for this month."))
        if reset_day:
            escape = dial_copy(
                "run_block.credits_escape_reset",
                tr("Semi-Auto still works, and Automatic comes back on "
                   "{date}.")).replace("{date}", reset_day)
        else:
            escape = dial_copy(
                "run_block.credits_escape",
                tr("Semi-Auto still works until it comes back."))
        if subscriber and pro_ceiling_enabled():
            # A plan to negotiate is still something to offer, so this one
            # keeps the Pro tint.
            card.set_tint("premium")
            body, cta = self._pro_ceiling_copy()
            card.route_cta(self._on_pro_contact_run_block)
            card.set_text(title, body, cta, escape=escape,
                          detail=self._pro_ceiling_detail())
            return False
        if subscriber:
            # No offer to make and no address to give: the card is the fact
            # and the way on, and the button leaves for the mode that runs.
            card.set_tint("neutral")
            card.route_cta(self._on_auto_upsell_manual_clicked)
            card.set_text(title, None,
                          dial_copy("upsell.manual_cta", tr("Use Semi-Auto")),
                          escape=escape)
            return False
        card.set_tint("premium")
        card.route_cta(self._on_upgrade_clicked)
        card.set_text(
            title,
            dial_copy("upsell.wall_body",
                      tr("Draw a whole city and let it run, at the finest "
                         "precision.")),
            dial_copy("upsell.cta", tr("Upgrade to Pro")),
            escape=escape,
        )
        card.set_pro_offer("plugin_run_block")
        return False

    # -- handlers -------------------------------------------------------------

    def _on_auto_run_block_redraw(self) -> None:
        """Back to the zone step, which re-arms the draw tool through
        ``auto_step_changed``. The zone that was refused goes with it."""
        try:
            self.on_zone_deleted_from_canvas()
        except (RuntimeError, AttributeError):
            self._go_to_auto_step(1)

    def _note_detect_blocked(self, reason: str | None) -> None:
        """One ``detect_blocked`` per episode, from the one place that knows
        why. It used to wait for a typed prompt, which the takeover now hides:
        a refusal the user never got to type into counted as no refusal."""
        if reason == getattr(self, "_detect_blocked_last", None):
            return
        self._detect_blocked_last = reason
        if not reason:
            return
        try:
            from ...core import telemetry_session_events

            telemetry_session_events.track_detect_blocked(reason=reason)
        except Exception:  # noqa: BLE001 -- telemetry never blocks the UI
            pass  # nosec B110
