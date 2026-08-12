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
    _BTN_BLUE,
    _BTN_CHIP,
    _CARD_CHILD_BTN_RESET_QSS,
    _CARD_QSS,
    _PREMIUM_STAR,
    _SUBCARD_MARGINS,
    _msg_card_qss,
    _msg_label_qss,
    _msg_text,
)
from .widgets import Mode

# Labels inside a _CARD_QSS card must not inherit its fill or its border.
_CARD_LABEL_RESET_QSS = "QLabel { background: transparent; border: none; }"

_TITLE_QSS = "font-size: 13px; font-weight: bold; color: palette(text);"
_LANE_TITLE_QSS = "font-size: 12px; font-weight: bold; color: palette(text);"
_LANE_LINE_QSS = "font-size: 11px; color: palette(text);"
_QUIET_QSS = "font-size: 11px; color: rgba(128,128,128,0.95);"
_HINT_QSS = "font-size: 10px; color: rgba(128,128,128,0.95);"


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
        """The paid lane: tinted, one filled button, the price under it."""
        lane = QWidget()
        lane.setObjectName("manualCreditOffer")
        lane.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lane.setStyleSheet(
            _msg_card_qss("manualCreditOffer", "premium")
            + _CARD_CHILD_BTN_RESET_QSS)
        lane_layout = QVBoxLayout(lane)
        lane_layout.setContentsMargins(*_SUBCARD_MARGINS)
        lane_layout.setSpacing(3)

        # Its OWN served id, not the Automatic one: that card counts a whole
        # run, this one counts objects saved one click at a time. Both say
        # "cloud detections", which is what the counter in the footer counts.
        title = QLabel(f"{_PREMIUM_STAR}  " + dial_copy(
            "upsell.bullet_quota_manual",
            tr("Pro: 5,000 cloud detections a month")))
        title.setWordWrap(True)
        title.setStyleSheet(_LANE_TITLE_QSS)
        lane_layout.addWidget(title)

        # Two lines, and they answer the free lane's two lines in the same
        # order: what the AI does, then what it asks of you. Feature lists were
        # tried here and none of them let a reader compare the lanes.
        for line in (
            tr("The same cloud AI, and the cleanest shapes."),
            tr("Nothing to install, works right away."),
        ):
            label = QLabel(line)
            label.setWordWrap(True)
            label.setStyleSheet(_LANE_LINE_QSS)
            lane_layout.addWidget(label)

        self.manual_credit_upgrade_btn = QPushButton(
            dial_copy("upsell.cta", tr("Upgrade to Pro")))
        self.manual_credit_upgrade_btn.setMinimumHeight(34)
        self.manual_credit_upgrade_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.manual_credit_upgrade_btn.setStyleSheet(_BTN_BLUE)
        self.manual_credit_upgrade_btn.clicked.connect(self._on_upgrade_clicked)
        lane_layout.addWidget(self.manual_credit_upgrade_btn)

        # The price ships in the sentence AND stays served. A number that only
        # lives on the server is absent on a cold cache, which is exactly the
        # screen where a buyer decides, and the price was missing from all six
        # Upgrade screens for that reason. The served id still wins, so a
        # change of price reaches the fleet the same day; the shipped line is
        # what a first launch with no configuration shows.
        hint = QLabel(dial_copy(
            "upsell.cta_hint",
            tr("39 EUR a month, cancel anytime. "
               "Opens your TerraLab dashboard.")))
        hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        hint.setWordWrap(True)
        hint.setStyleSheet(_HINT_QSS)
        lane_layout.addWidget(hint)
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

        self.manual_credit_offline_btn = QPushButton(tr("Use my computer"))
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
            # Before a session, the picked half is the honest question: the
            # data notice comes at Start. Inside one, the route is fixed and
            # the strict answer is what says whether a save is billed at all,
            # so a session the machine is answering for free never gets a
            # paywall painted over it.
            on_cloud = (self._manual_cloud_route_on()
                        if getattr(self, "_segmentation_active", False)
                        else self._manual_cloud_route_picked())
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
                    tr("This polygon stays on the map, and Export still works.")
                    if has_saved else
                    tr("This polygon stays on the map, but it cannot be "
                       "saved."))))
            self.manual_credit_notice.setVisible(in_session)
            self.manual_credit_title.setText(
                tr("Your cloud detections are used up")
                if getattr(self, "_auto_is_subscriber", False) else
                tr("Your free cloud detections are used up"))
            self.manual_credit_reset.setText(self._manual_credit_reset_text())
            self.manual_credit_reset.setVisible(
                bool(self.manual_credit_reset.text()))
            # Mid-session the offline AI cannot take over the open session, so
            # the button says what it will actually do: end this one.
            self.manual_credit_offline_btn.setText(
                tr("Stop and use my computer") if in_session else
                tr("Use my computer"))
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
        try:
            total = int(getattr(self, "_auto_credits_total", 0) or 0)
        except (TypeError, ValueError):
            total = 0
        if total > 0 and reset_day:
            return tr("You used all {n}. They come back on {date}.").format(
                n=total, date=reset_day)
        if reset_day:
            return tr("They come back on {date}.").format(date=reset_day)
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
            base = tr("Or end this session and work free on this computer. "
                      "Your saved polygons are kept.")
        else:
            base = tr("Or work free with a smaller AI on this computer.")
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
            ).format(gb=f"{need:g}")
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
