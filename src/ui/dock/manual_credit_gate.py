"""What Semi-Auto shows when Cloud AI has no credits left to spend.

Two boxes, never one. The design system is explicit about why: a box that
blocks a button is an error, red and starless, or the refusal reads as an
invitation and the user keeps pressing a dead control. So the refusal is its
own red line, and the offer that follows is a separate card in the premium
blue, with the free way out named under it.

The rule this file exists to protect: **Export is never blocked.** A user who
runs out mid-session has already paid for every polygon on screen, and a
paywall that eats them is a refund request, not a sale. Only Save stops.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py); methods
here are plain mixin members and widgets live on the dock instance.
"""
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


class DockManualCreditGateMixin:
    """The out-of-credits refusal and its offer, on the Semi-Auto page."""

    def _setup_manual_credit_gate(self) -> None:
        """Build the refusal line and the offer card, both hidden."""
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # --- the refusal: red, starless, and it names the price ------------
        self.manual_credit_refusal = QLabel()
        self.manual_credit_refusal.setWordWrap(True)
        self.manual_credit_refusal.setStyleSheet(_msg_label_qss("error"))
        layout.addWidget(self.manual_credit_refusal)

        # --- the offer: the same served copy the Automatic upsell reads ----
        offer = QWidget()
        offer.setObjectName("manualCreditOffer")
        offer.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        offer.setStyleSheet(
            _msg_card_qss("manualCreditOffer", "premium")
            + _CARD_CHILD_BTN_RESET_QSS)
        offer_layout = QVBoxLayout(offer)
        offer_layout.setContentsMargins(*_SUBCARD_MARGINS)
        offer_layout.setSpacing(5)

        # The OTHER way out, and the one a blocked free user is never told:
        # waiting. It opens the offer card rather than floating between the
        # boxes, so the card reads as "wait, or do not wait". Filled from the
        # served period end, hidden when the server sends none.
        self.manual_credit_reset = QLabel()
        self.manual_credit_reset.setWordWrap(True)
        self.manual_credit_reset.setStyleSheet(
            "font-size: 11px; color: rgba(128,128,128,0.95);")
        self.manual_credit_reset.setVisible(False)
        offer_layout.addWidget(self.manual_credit_reset)

        # Its OWN served id, not the Automatic one: that card counts tiles
        # detected, this one counts objects saved, and one sentence cannot
        # be right for both.
        title = QLabel(f"{_PREMIUM_STAR}  " + dial_copy(
            "upsell.bullet_quota_manual", tr("5,000 objects every month")))
        title.setWordWrap(True)
        title.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: palette(text);")
        offer_layout.addWidget(title)

        for bullet_id, bullet in (
            ("upsell.bullet_objects",
             tr("Every building, tree, or road as clean polygons")),
            ("upsell.bullet_cancel",
             tr("Cancel anytime; your exported layers stay yours")),
        ):
            line = QLabel(dial_copy(bullet_id, bullet))
            line.setWordWrap(True)
            line.setStyleSheet("font-size: 11px; color: palette(text);")
            offer_layout.addWidget(line)

        self.manual_credit_upgrade_btn = QPushButton(
            dial_copy("upsell.cta", tr("Upgrade to Pro")))
        self.manual_credit_upgrade_btn.setMinimumHeight(34)
        self.manual_credit_upgrade_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.manual_credit_upgrade_btn.setStyleSheet(_BTN_BLUE)
        self.manual_credit_upgrade_btn.clicked.connect(self._on_upgrade_clicked)
        offer_layout.addWidget(self.manual_credit_upgrade_btn)

        hint = QLabel(dial_copy(
            "upsell.cta_hint", tr("Opens your TerraLab dashboard")))
        hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 10px; color: rgba(128,128,128,0.95);")
        offer_layout.addWidget(hint)

        layout.addWidget(offer)

        # The free way out, as its own named block rather than a link hanging
        # off the sale. Every paywall worth copying keeps the tier you are on
        # visible as a peer of the ones you are being sold; buried as a link it
        # reads as a footnote, and a user who cannot pay concludes they are
        # stuck. It stays quiet: no fill of its own, no second button.
        free_block = QWidget()
        free_block.setObjectName("manualCreditFree")
        free_block.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        free_block.setStyleSheet(
            _CARD_QSS.format(name="manualCreditFree")
            + "QLabel { background: transparent; border: none; }")
        free_layout = QVBoxLayout(free_block)
        free_layout.setContentsMargins(*_SUBCARD_MARGINS)
        free_layout.setSpacing(3)

        free_title = QLabel(tr("My computer, free and unlimited"))
        free_title.setWordWrap(True)
        free_title.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: palette(text);")
        free_layout.addWidget(free_title)

        self.manual_credit_free_note = QLabel(
            tr("The smaller model on your computer. No credits, no limit."))
        self.manual_credit_free_note.setWordWrap(True)
        self.manual_credit_free_note.setStyleSheet(
            "font-size: 11px; color: rgba(128,128,128,0.95);")
        free_layout.addWidget(self.manual_credit_free_note)

        self.manual_credit_offline_btn = QPushButton(
            tr("Use my computer instead"))
        self.manual_credit_offline_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.manual_credit_offline_btn.setStyleSheet(_BTN_CHIP)
        self.manual_credit_offline_btn.clicked.connect(
            self._on_manual_credit_offline_clicked)
        free_layout.addWidget(self.manual_credit_offline_btn)

        layout.addWidget(free_block)
        holder.setVisible(False)
        self.manual_credit_gate = holder
        self.main_layout.addWidget(holder)

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

    def _refresh_manual_credit_gate(self) -> None:
        """Show the two boxes only where they can change what happens next."""
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
            self.manual_credit_refusal.setText(_msg_text("error", (
                (tr("No credits left. This polygon stays on the map, and "
                    "Export still works.") if has_saved else
                 tr("No credits left. This polygon stays on the map, but it "
                    "cannot be saved."))
                if in_session else
                tr("No credits left. Each object you save with Cloud AI "
                   "costs one credit."))))
            # Mid-session the offline AI cannot take over the open session, so
            # the button says what it will actually do: end this one.
            self.manual_credit_offline_btn.setText(
                tr("Stop and use my computer instead")
                if in_session else
                tr("Use my computer instead"))
            # Never repeat the refusal's own promise here. In a session the one
            # thing this block has to say is what the button DOES: it ends the
            # session, because the route is fixed for its whole life.
            self.manual_credit_free_note.setText(
                self._manual_credit_free_note_text(in_session))
            # Waiting is the second way out, and the only one with a date on it.
            reset_day = getattr(self, "_auto_reset_display", "")
            if reset_day:
                self.manual_credit_reset.setText(tr(
                    "Your free credits come back on {date}.").format(
                        date=reset_day))
            self.manual_credit_reset.setVisible(bool(reset_day))
        except (RuntimeError, AttributeError):
            return
        self._note_manual_upsell_viewed()

    def _manual_credit_free_note_text(self, in_session: bool) -> str:
        """The line under "My computer", download included when there is one.

        The block sells free and unlimited, so the one thing it may not leave
        out is that the first run has to fetch the model. The figures come
        from the same reader the install window uses, so the two can never
        quote different numbers at the same user.
        """
        if in_session:
            base = tr("Ends this session. Your saved polygons are kept.")
        else:
            base = tr("The smaller model on your computer. No credits, "
                      "no limit.")
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
