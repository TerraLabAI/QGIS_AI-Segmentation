"""What Semi-Auto shows when Cloud AI has no credits left to spend.

ONE card, and it owns the page: the layer picker and the Start button go away
while it is up (see ``_manual_credit_gate_owns_page``), because nothing on that
half of the screen can be acted on. Running out is not an error, so nothing
here is red. The card names the state, then offers the two ways on as peers,
in the same shape: what the AI does, then what it asks of you. That is the only
way a reader can tell what Pro buys that the free lane does not.

The rule this file exists to protect: **Export is never blocked.** A user who
runs out mid-session has already paid for every polygon on screen, and a
paywall that eats them is a refund request, not a sale. Only Save stops, and
mid-session the amber notice above the card says so.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py); methods
here are plain mixin members and widgets live on the dock instance.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
from ...core.server_dials import dial_copy
from .styles import (
    _BTN_CHIP,
    _CARD_CHILD_BTN_RESET_QSS,
    _CARD_QSS,
    _SUBCARD_MARGINS,
    _msg_label_qss,
    _msg_text,
)
from .ui_refresh import format_quota_count
from .upsell_card import UpsellCard
from .widgets import Mode

# Labels inside a _CARD_QSS card must not inherit its fill or its border.
_CARD_LABEL_RESET_QSS = "QLabel { background: transparent; border: none; }"

_TITLE_QSS = "font-size: 13px; font-weight: bold; color: palette(text);"
_QUIET_QSS = "font-size: 11px; color: rgba(128,128,128,0.95);"


class DockManualCreditGateMixin:
    """The out-of-credits card on the Semi-Auto page, and its two lanes."""

    def _setup_manual_credit_gate(self) -> None:
        """Build the notice and the card, both hidden."""
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Mid-session only: the one thing a refused Save has to say is what
        # happens to the polygon on screen. Amber, never red: nothing failed,
        # and the session carries on.
        self.manual_credit_notice = QLabel()
        self.manual_credit_notice.setWordWrap(True)
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
        card_layout.setContentsMargins(*_SUBCARD_MARGINS)
        card_layout.setSpacing(4)

        # The page names itself in its first card, never in floating text.
        self.manual_credit_title = QLabel()
        self.manual_credit_title.setWordWrap(True)
        self.manual_credit_title.setStyleSheet(_TITLE_QSS)
        card_layout.addWidget(self.manual_credit_title)

        # The OTHER way out, and the one a blocked free user is never told:
        # waiting. Filled from the served balance and period end, hidden when
        # the server sends neither.
        self.manual_credit_reset = QLabel()
        self.manual_credit_reset.setWordWrap(True)
        self.manual_credit_reset.setStyleSheet(_QUIET_QSS)
        self.manual_credit_reset.setVisible(False)
        card_layout.addWidget(self.manual_credit_reset)

        card_layout.addSpacing(4)
        card_layout.addWidget(self._build_manual_credit_pro_lane())
        card_layout.addSpacing(6)
        for widget in self._build_manual_credit_free_way_out():
            card_layout.addWidget(widget)

        layout.addWidget(card)
        holder.setVisible(False)
        self.manual_credit_gate = holder
        self.main_layout.addWidget(holder)

    def _build_manual_credit_pro_lane(self) -> QWidget:
        """The paid lane, in the shared offer card. Star variant: this is the
        end of an allowance and it owns the page."""
        lane = UpsellCard("manualCreditOffer", "star",
                          on_cta=self._on_upgrade_clicked)
        # The name the rest of the dock reads. `_on_upgrade_clicked` picks its
        # cta source off the sender, so this has to stay the button itself.
        self.manual_credit_upgrade_btn = lane.button
        # A subscriber has nothing to buy here, so the refresh hides the whole
        # lane rather than the button alone.
        self.manual_credit_pro_lane = lane

        # Its OWN served id, not the Automatic one: that card counts a whole
        # run, this one counts objects saved one click at a time. Both say
        # "cloud detections", which is what the counter in the footer counts.
        star = dial_copy(
            "upsell.bullet_quota_manual",
            tr("2,000 cloud objects every month in Semi-Auto"))
        # One line, and it answers the free lane in the same order: what the AI
        # does, then what it asks of you. Feature lists were tried here and none
        # of them let a reader compare the lanes.
        # The star line above already carries the 2,000 figure, so this line
        # must not repeat it: it says what the figure buys you.
        title = dial_copy(
            "upsell.manual_lane_title",
            tr("Keep clicking with the same cloud AI, nothing to install."))
        # The price ships in the sentence AND stays served. A number that only
        # lives on the server is absent on a cold cache, which is exactly the
        # screen where a buyer decides, and the price was missing from all six
        # Upgrade screens for that reason. The served id still wins, so a
        # change of price reaches the fleet the same day; the shipped line is
        # what a first launch with no configuration shows.
        lane.set_text(
            title,
            None,
            dial_copy("upsell.cta", tr("Upgrade to Pro")),
            escape=dial_copy(
                "upsell.cta_hint",
                tr("39 EUR a month, cancel anytime.")),
            star=star,
        )
        return lane

    def _build_manual_credit_free_way_out(self) -> list[QWidget]:
        """The free way out: a hairline, one grey sentence, one chip button.

        It has no card of its own. Given one it became a second offer beside
        the paid one, weighed the same, and the screen stopped selling
        anything. It still has to be here in words, because a paywall that
        hides the tier you are already on reads as a dead end to everyone who
        cannot pay today.
        """
        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        rule.setStyleSheet(
            "background: rgba(128,128,128,0.25); border: none; max-height: 1px;")

        self.manual_credit_free_note = QLabel()
        self.manual_credit_free_note.setWordWrap(True)
        self.manual_credit_free_note.setStyleSheet(_QUIET_QSS)

        # Served like the paid button beside it. _refresh_manual_credit_gate
        # rewrites this label per session state and reads the same two ids.
        self.manual_credit_offline_btn = QPushButton(dial_copy(
            "manual_gate.offline_cta", tr("Use my computer")))
        self.manual_credit_offline_btn.setMinimumHeight(30)
        self.manual_credit_offline_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.manual_credit_offline_btn.setStyleSheet(_BTN_CHIP)
        self.manual_credit_offline_btn.clicked.connect(
            self._on_manual_credit_offline_clicked)
        return [rule, self.manual_credit_free_note,
                self.manual_credit_offline_btn]

    # -- state --------------------------------------------------------------

    def _manual_credits_exhausted(self) -> bool:
        """True when the account cannot pay for one more saved object.

        An unknown balance is not exhausted: a plugin that has not heard from
        the account yet must never refuse work on a guess.
        """
        try:
            env = getattr(self, "_quota_envelopes", None)
            if env is not None and env.objects_remaining is not None:
                # Saves spend the objects envelope, never the km² one: the
                # wallet figure below can be the km² gauge in tile terms and
                # must not close Semi-Auto while objects remain.
                return env.objects_remaining <= 0
            left = getattr(self, "_auto_credits", None)
            return left is not None and int(left) <= 0
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return False

    def _manual_credit_gate_owns_page(self) -> bool:
        """True when the card replaces the Start view instead of sitting under it.

        Only before a session. Inside one the page belongs to the work already
        on the map, and the card is a notice beside it.
        """
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
        """Show the card only where it can change what happens next."""
        gate = getattr(self, "manual_credit_gate", None)
        if gate is None:
            return
        try:
            # The picked half is the whole answer, before a session and inside
            # one: nothing else stands between the switch and the wire, so a
            # session the machine is answering for free never gets a paywall
            # painted over it.
            on_cloud = self._manual_cloud_route_picked()
            show = bool(
                self._plugin_activated
                and self._mode == Mode.INTERACTIVE
                and on_cloud
                and self._manual_credits_exhausted()
                # Same guard as the engine card. With no imagery the page is
                # the empty-state hero, and a paywall stacked under it asks the
                # user to pay for a session they cannot start either way.
                and (self._segmentation_active
                     or self.layer_combo.count_layers() > 0)
            )
            gate.setVisible(show)
            if not show:
                return
            in_session = bool(getattr(self, "_segmentation_active", False))
            # Export is disabled until something is saved, so promising it
            # works on the first refused object points at a grey button.
            has_saved = bool(getattr(self, "_saved_polygon_count", 0) > 0)
            if in_session:
                self.manual_credit_notice.setText(_msg_text("warning", (
                    dial_copy(
                        "manual_gate.notice_saved",
                        tr("This polygon stays on the map, and Export still "
                           "works."))
                    if has_saved else
                    dial_copy(
                        "manual_gate.notice_unsaved",
                        tr("This polygon stays on the map, but it cannot be "
                           "saved.")))))
            self.manual_credit_notice.setVisible(in_session)
            # The paid lane sells the plan this user already has. A subscriber
            # who spent the month's objects reads the fact and the two ways
            # out, never an offer to buy it again.
            subscriber = bool(getattr(self, "_auto_is_subscriber", False))
            lane = getattr(self, "manual_credit_pro_lane", None)
            if lane is not None:
                lane.setVisible(not subscriber)
            env = getattr(self, "_quota_envelopes", None)
            if env is not None and env.objects_cap:
                # Open with what the month produced, in the envelope's unit.
                # Served, and filled with str.replace: format() on a served
                # sentence raises on a stray brace.
                self.manual_credit_title.setText(dial_copy(
                    "manual_gate.title_objects",
                    tr("You saved your {n} cloud objects in Semi-Auto this month"),
                ).replace("{n}", format_quota_count(env.objects_cap)))
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
            # Mid-session the offline AI cannot take over the open session, so
            # the button says what it will actually do: end this one.
            self.manual_credit_offline_btn.setText(
                dial_copy("manual_gate.offline_cta_in_session",
                          tr("Stop and use my computer"))
                if in_session else
                dial_copy("manual_gate.offline_cta",
                          tr("Use my computer")))
            # Never repeat the notice's own promise here. In a session the one
            # thing this lane has to say is what the button DOES: it ends the
            # session, because the route is fixed for its whole life.
            self.manual_credit_free_note.setText(
                self._manual_credit_free_note_text(in_session))
        except (RuntimeError, AttributeError):
            return
        self._note_manual_upsell_viewed()

    def _manual_credit_reset_text(self) -> str:
        """How many were spent and when they come back, or "" for neither.

        Waiting is the second way out and the only one with a date on it. The
        total is quoted only when the server sent one: a bare "you used all
        None" is worse than no sentence.
        """
        reset_day = getattr(self, "_auto_reset_display", "")
        env = getattr(self, "_quota_envelopes", None)
        # Served, and filled with str.replace: format() on a served sentence
        # raises on a stray brace, and this one paints the gate.
        if env is not None and env.objects_cap:
            # The title already carries the count; the wallet total below can
            # be the km² gauge in tile terms, which is not what a save spends.
            if reset_day:
                return dial_copy(
                    "manual_gate.reset_date",
                    tr("They come back on {date}."),
                ).replace("{date}", reset_day)
            return ""
        try:
            total = int(getattr(self, "_auto_credits_total", 0) or 0)
        except (TypeError, ValueError):
            total = 0
        if total > 0 and reset_day:
            return dial_copy(
                "manual_gate.reset_used_all",
                tr("You used all {n}. They come back on {date}."),
            ).replace("{n}", str(total)).replace("{date}", reset_day)
        if reset_day:
            return dial_copy(
                "manual_gate.reset_date",
                tr("They come back on {date}."),
            ).replace("{date}", reset_day)
        return ""

    def _manual_credit_free_note_text(self, in_session: bool) -> str:
        """The line under "My computer", download included when there is one.

        The lane sells free and unlimited, so the one thing it may not leave
        out is that the first run has to fetch the model. The figures come
        from the same reader the install window uses, so the two can never
        quote different numbers at the same user.
        """
        # Facts only. The way out is ranked by its grey text and its chip
        # button; running it down in words as well would read as bullying a
        # user who cannot pay.
        if in_session:
            base = dial_copy(
                "manual_gate.free_note_in_session",
                tr("Or end this session and work free on this computer. "
                   "Your saved polygons are kept."))
        else:
            base = dial_copy(
                "manual_gate.free_note",
                tr("Or keep clicking for free with a smaller AI on this "
                   "computer."))
        if self._manual_engine_local_ready():
            return base
        try:
            from .manual_local_install_dialog import local_install_disk_figures

            need, _free = local_install_disk_figures()
        except Exception:  # noqa: BLE001 -- an unknown floor drops the figure
            need = 0.0
        # Served whole, number included. The shipped English keeps its exact
        # wording because it is the lookup key for eleven translations, so the
        # minutes figure moves here by serving the sentence rather than by
        # rewriting it. {gb} is substituted after the read, so a served line
        # has to carry the placeholder.
        if need > 0:
            return base + " " + dial_copy(
                "manual_gate.install_note",
                tr("It downloads first: {gb} GB and about 10 minutes."),
            ).replace("{gb}", f"{need:g}")
        return base + " " + dial_copy(
            "manual_gate.install_note_no_size",
            tr("It downloads first, and takes about 10 minutes."),
        )

    def _note_manual_upsell_viewed(self) -> None:
        """One impression per session, so the click has a denominator."""
        if getattr(self, "_manual_upsell_seen", False):
            return
        self._manual_upsell_seen = True
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pro_upsell_viewed(
                trigger="manual_credits_exhausted")
        except Exception:  # noqa: BLE001 -- telemetry never blocks the UI  # nosec B110
            pass

    def _on_manual_credit_offline_clicked(self) -> None:
        """Take the free way out.

        Mid-session this stops the session first: the route is fixed for the
        life of a session, so the offline AI can only answer the NEXT one, and
        pretending otherwise would leave the user clicking into a dead socket.
        """
        if getattr(self, "_segmentation_active", False):
            self.stop_segmentation_requested.emit()
            # The stop handler asks before discarding open polygons, so a
            # session still live here means the user answered no. Moving the
            # route anyway hid this very block (it only shows on the cloud
            # route) and left every Save refusing with nothing on screen to
            # say why.
            if getattr(self, "_segmentation_active", False):
                return
        try:
            # The same handler the engine cards use, so this button cannot
            # move the switch onto a half with nothing behind it: with no
            # model on the machine it opens the install window first, and it
            # counts the change like every other engine choice.
            self._on_manual_engine_picked(False)
        except (RuntimeError, AttributeError):
            return
        self._update_full_ui()
