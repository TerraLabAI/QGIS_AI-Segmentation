"""Which AI answers a Semi-Auto click, and the switch that picks it.

Two option cards, Cloud AI against the machine, over ONE line saying where the
picked one runs and what it costs.

It touches the Start button, above the caption that says what a click does.
Moving it to the foot of the panel was tried and failed: with the page stretch
between them, nobody read the picker and the button as one thing.

Two earlier builds are the reason it looks like this. A full chooser screen
stopped the user before the Start button and spent a whole state on one
decision. A single "Cloud AI" checkbox then hid the alternative: a tick names
one option, so a user who does not want the cloud cannot see what they would
get instead, and the choice reads as a guess. Both halves stay on screen.

Cloud is the half the mode opens on. Moving to the other half is one click,
always, and no wording shames it.

It stays readable on the install gate, because picking Cloud AI is exactly what
removes the download. It disappears the moment a session opens, because the
engine is fixed at Start and a control that cannot change anything is a lie.

Server withdrawal is the safe direction. With ``manual_cloud_route_offered()``
false there is no second engine, so nothing here is ever shown and the mode
behaves as it always did, on the machine, free.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py); methods
here are plain mixin members and widgets live on the dock instance.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
from ...core.manual_cloud_route import (
    manual_cloud_route_enabled,
    manual_cloud_route_offered,
    set_manual_cloud_route_enabled,
)
from ...core.manual_engine_choice import cloud_consent_given, set_cloud_consent
from ...core.server_dials import dial_copy
from .manual_local_install_dialog import local_install_minutes
from .styles import (
    _CARD_CHILD_BTN_RESET_QSS,
    _SUBCARD_MARGINS,
    _SUBCARD_QSS,
    _msg_card_qss,
    _msg_label_qss,
)
from .widgets import Mode, _EngineSwitch

# Splits the note into where it runs and what it costs. A middle dot rather
# than a full stop, so the two halves read as one line at a glance instead of
# as a paragraph of two sentences.
_DOT = "·"


def _fill_engine_line(text: str, **parts: str) -> str:
    """Fill the placeholders of one engine line, shipped or served.

    ``str.replace``, never ``str.format``: a served sentence may carry a stray
    brace, and these lines are written on a paint path. Filling after the
    reader rather than before is what lets served copy move the separator, the
    disk figure and the minutes wherever the sentence needs them.
    """
    out = text.replace("{dot}", _DOT)
    for name, value in parts.items():
        out = out.replace("{" + name + "}", value)
    return out


class DockManualEngineMixin:
    """Builds and drives the Semi-Auto engine switch and the note under it."""

    # -- construction -------------------------------------------------------

    def _setup_manual_engine(self) -> None:
        """Build the picker. Hidden until the page asks for it.

        No frame of its own. The two option cards already carry borders and
        the note carries one, so a third box around them added a line and no
        meaning: three nested rectangles read as a form, not as a choice.
        """
        card = QWidget()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Names the button above, because adjacency alone did not carry it: the
        # cards sat right under Start and were still read as an unrelated
        # setting. It borrows the word "segmentation" off the button on purpose,
        # so the eye joins the two without a line or an arrow between them.
        engine_header = QLabel(dial_copy(
            "engine.header", tr("Where the segmentation runs:")))
        engine_header.setStyleSheet(
            "font-size: 11px; color: rgba(128, 128, 128, 0.95);"
            " background: transparent;")
        layout.addWidget(engine_header)

        # Built on the recommended half. The stored answer lands on the first
        # refresh, before the card is ever shown, so nothing flashes.
        self.manual_engine_switch = _EngineSwitch(cloud=True)
        self.manual_engine_switch.engine_selected.connect(
            self._on_manual_engine_picked)
        layout.addWidget(self.manual_engine_switch)

        # The note gets its own box. Loose text under a control is the one
        # thing this design system never allows.
        #
        # ONE line, and never a headline over a paragraph. That was tried and
        # it put the same sentence on the screen three times: the card gloss,
        # the headline repeating it, and the paragraph repeating the headline.
        # The gloss on the card says what the engine gives. This line says the
        # two things the gloss cannot fit, where it runs and what it costs, and
        # it says them once.
        note_box = QWidget()
        note_box.setObjectName("manualEngineNote")
        note_box.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        note_layout = QVBoxLayout(note_box)
        note_layout.setContentsMargins(*_SUBCARD_MARGINS)
        note_layout.setSpacing(4)

        self.manual_engine_note = QLabel("")
        self.manual_engine_note.setWordWrap(True)
        self.manual_engine_note.setTextFormat(Qt.TextFormat.RichText)
        self.manual_engine_note.setStyleSheet(
            "font-size: 11px; color: palette(text); background: transparent;")
        note_layout.addWidget(self.manual_engine_note)

        self.manual_engine_note_box = note_box
        layout.addWidget(note_box)

        # The running-low warning, and its own card under the note rather than
        # inside it. Same widget, same wording and same amber as the Automatic
        # Start step (`_update_auto_low_credit_note`), because a user who meets
        # this moment in both modes must not have to read two different things
        # to learn one fact.
        #
        # Beside the description, never over it. The balance used to REPLACE the
        # engine copy, so the one card that says what Cloud AI does and what a
        # click costs disappeared exactly when the user was being asked to pay.
        self.manual_engine_low_line = QLabel("")
        self.manual_engine_low_line.setWordWrap(True)
        self.manual_engine_low_line.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.manual_engine_low_line.setTextFormat(Qt.TextFormat.RichText)
        self.manual_engine_low_line.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction)
        self.manual_engine_low_line.setOpenExternalLinks(False)
        self.manual_engine_low_line.setStyleSheet(_msg_label_qss("warning"))
        self.manual_engine_low_line.linkActivated.connect(
            self._on_manual_low_credit_link)
        self.manual_engine_low_line.setVisible(False)
        # NOT added here. It is seated at the foot of the dock, next to the
        # credit ring, by _setup_low_credit_slot. Leaving it in this card put an
        # amber warning in the middle of the engine decision and pushed the rest
        # of the page down. Out of the card it no longer hides with it, so
        # _update_manual_engine_card_visible has to put it away itself.

        card.setVisible(False)
        self.manual_engine_card = card
        # Which tint the note box is wearing, so a repaint only runs on a
        # change.
        self._manual_engine_card_tinted: str | None = None
        # Touching the Start button, and now the last thing in that container.
        # It was moved to the foot of the panel once, under the stretch, and
        # that failed on contact: 600 px of empty grey between the button and
        # the picker, and nobody read the two as one thing. Adjacency plus the
        # header above the cards is what says "this is where that button will
        # run".
        #
        # Inside start_container, which is what gets it hidden for free the
        # moment a session opens. The install gate no longer hides that
        # container while the picker exists (see _update_full_ui_interactive),
        # so picking Cloud AI is still one click away on a machine with
        # nothing installed.
        self.start_container.layout().addWidget(card)

    # -- questions the rest of the dock asks --------------------------------

    def _manual_cloud_route_picked(self) -> bool:
        """Which half of the switch the user is on.

        What the panel describes and what the next Start is about to open. It
        does NOT ask whether the data notice has been answered: that question
        is put at Start, and pressing the way out of it moves the switch here.

        The account is part of the answer: the route needs one, so with no
        account the on-device install is still what a session depends on.
        """
        try:
            return bool(
                self._plugin_activated
                and manual_cloud_route_enabled()
                and manual_cloud_route_offered()
            )
        except (RuntimeError, AttributeError):
            return False

    def _manual_cloud_route_on(self) -> bool:
        """Whether a click would really be answered off the machine.

        The same questions the click path asks, so the panel can never price a
        session the machine is answering for free: an account, the switch, the
        offer, and the data notice. The click path also reads a token, which
        only the plugin side holds; `_plugin_activated` follows it, since a
        sign-out and a failed key check both clear it.
        """
        if not self._manual_cloud_route_picked():
            return False
        try:
            return bool(cloud_consent_given())
        except Exception:  # noqa: BLE001 -- an unreadable answer is not a yes
            return False

    def _manual_engine_offered(self) -> bool:
        """Whether this install has a second engine to pick at all.

        Fail-closed, like the switch it reads: an install that has never heard
        from a server behaves exactly as it did before the engine card existed.
        """
        try:
            return bool(manual_cloud_route_offered())
        except Exception:  # noqa: BLE001 -- no configuration means no choice
            return False

    def _manual_engine_local_ready(self) -> bool:
        """Whether the offline AI is installed and usable right now."""
        return bool(getattr(self, "_dependencies_ok", False)
                    and getattr(self, "_checkpoint_ok", False))

    def _manual_install_running(self) -> bool:
        """Whether a download carrying the on-device AI is on the wire.

        Moving to Cloud AI mid-install starts a cloud session and leaves the
        download running, which is the right behaviour and the wrong thing to
        do silently: without this the setup section vanishes the instant cloud
        makes the page complete, and minutes of downloading disappear with it.
        The progress timer is the one thing that only runs while it runs.

        The timer alone answers a wider question than this one. EVERY install
        starts it, including the light one Automatic asks for, which fetches
        no model at all: the offline-install window then followed that bar to
        100%, closed, and nothing it exists for had been installed. So the
        timer AND an install this dock knows carries the model.
        """
        try:
            if not self._progress_timer.isActive():
                return False
        except (RuntimeError, AttributeError):
            return False
        return bool(getattr(self, "_manual_install_wants_model", False)
                    or getattr(self, "_auto_review_installing", False))

    # -- refresh ------------------------------------------------------------

    def _refresh_manual_engine_ui(self) -> None:
        """Show the card on the Semi-Auto Start view, and nowhere else."""
        card = getattr(self, "manual_engine_card", None)
        if card is None:
            return
        try:
            on_start_view = bool(
                self._plugin_activated
                and self._mode == Mode.INTERACTIVE
                and manual_cloud_route_offered()
                and not self._segmentation_active
                # Empty state shows the hero and nothing else. With no imagery
                # there is no session to pick an engine for.
                and self.layer_combo.count_layers() > 0
                # At zero the credit gate owns the page. It already names the
                # engine, the price and both ways on, and a card above it
                # still selling the cloud reads as a second offer arguing with
                # the refusal.
                and not (self._manual_cloud_route_picked()
                         and self._manual_credits_exhausted())
            )
            card.setVisible(on_start_view)
            if on_start_view:
                self._refresh_manual_engine_card_text()
            else:
                # The warning lives at the foot of the dock now, so hiding the
                # card no longer takes it away. Without this it survives into a
                # running session and into Automatic, which shows its own.
                self._write_manual_low_credit_line(False)
        except (RuntimeError, AttributeError):
            pass  # nosec B110 -- the widget can be gone during teardown

    def _refresh_manual_engine_card_text(self) -> None:
        """Write the box, its tint, its line and its offer for the state now."""
        cloud = self._manual_cloud_route_picked()
        low = cloud and self._manual_engine_credits_low()
        try:
            # Written from the stored answer, never from a user click, so the
            # write must not come back as one.
            self.manual_engine_switch.set_cloud(cloud)
            self._paint_manual_engine_card()
            self.manual_engine_note.setText(self._manual_engine_copy(cloud))
            self._write_manual_low_credit_line(low)
        except (RuntimeError, AttributeError):
            pass  # nosec B110 -- teardown
        if low:
            self._note_manual_low_credit_offer()

    def _note_manual_low_credit_offer(self) -> None:
        """One impression per session, so the top-up click has a denominator.

        Counted apart from the out-of-credits card: this one is read by a user
        who can still work, and the two convert nothing alike.
        """
        if getattr(self, "_manual_low_credit_seen", False):
            return
        self._manual_low_credit_seen = True
        try:
            from ...core import telemetry_session_events

            telemetry_session_events.track_pro_upsell_viewed(
                trigger="manual_credits_low")
        except Exception:  # noqa: BLE001 -- telemetry never blocks the UI  # nosec B110
            pass

    def _manual_engine_credits_low(self) -> bool:
        """Whether the balance has dropped into the warning band.

        Zero is not low, it is empty, and the credit gate owns that state with
        the full offer. This is only the stretch before it, where the user is
        still working and can still act. Same threshold as the footer ring,
        read from one place.

        A subscriber never sees it. Their balance refills on a plan they are
        already paying for, so an amber card and a top-up button would sell
        them what they bought.
        """
        try:
            from ...core.credit_gate import (
                low_credit_ceiling,
                low_credit_threshold,
            )

            if getattr(self, "_auto_is_subscriber", False):
                return False
            left = getattr(self, "_auto_credits", None)
            total = getattr(self, "_auto_credits_total", None)
            if left is None or not total or int(total) <= 0:
                return False
            if int(left) > low_credit_ceiling():
                return False
            return 0 < int(left) <= int(total) * low_credit_threshold()
        except Exception:  # noqa: BLE001 -- a tint must never break a paint
            return False

    def _paint_manual_engine_card(self) -> None:
        """The note box, quiet in every state.

        Deliberately no tint, for the picked engine or for a low balance. The
        premium fill is byte-identical to the armed one, so a box wearing it
        while nothing is happening reads as a tool waiting to be clicked. The
        push toward the cloud is carried by the picked card and its star.

        The amber used to live here, which put it on the description of what
        the mode does and left the user reading a warning where the price used
        to be. It has its own card under this one now
        (`manual_engine_low_line`), the same one Automatic shows, so the
        warning and the description stop competing for one box.
        """
        kind = ""
        if kind == self._manual_engine_card_tinted:
            return
        self._manual_engine_card_tinted = kind
        qss = (_msg_card_qss("manualEngineNote", kind) if kind
               else _SUBCARD_QSS.format(name="manualEngineNote")
               + "QLabel { background: transparent; border: none; }")
        try:
            self.manual_engine_note_box.setStyleSheet(
                qss + _CARD_CHILD_BTN_RESET_QSS)
        except (RuntimeError, AttributeError):
            pass  # nosec B110 -- teardown

    def _manual_engine_copy(self, cloud: bool) -> str:
        """The one line under the cards: where it runs, then what it costs.

        Where on the left of the dot, price on the right and in bold, so the
        cost can be read without reading the sentence. Never more than a line.
        A headline plus a paragraph was tried here and it said the same thing
        three times on one screen, counting the card gloss above it.

        Both states open on where it runs, because that is the whole difference
        and the thing a GIS user assumes. Most of them have never had a plugin
        send their imagery anywhere.

        The balance is NOT one of the states. It used to take this line over on
        the way to empty, which deleted the only sentence saying what a click
        costs at the exact moment the user was being asked to pay. It has its
        own line under this box now.
        """
        if cloud:
            # Served copy, under its own id. The old one carries a sentence
            # written for the chooser screen this replaced.
            #
            # Says the one thing the card above cannot, then what it costs. It
            # used to sell the size of the model, which is our problem and not
            # a reason to click anything. The reason to click is that this side
            # asks for nothing first: two of every three machines that reach
            # this page have never started the on-device install, and they open
            # the plugin again and again without ever getting an object out.
            #
            # Not where the machines are: that is disclosure, it belongs at the
            # moment of consent, and manual_cloud_consent.py carries it in full.
            #
            # One claim, not two. "Nothing to install, nothing to download" said
            # the same thing twice and spent the sentence on the word "nothing".
            #
            # It opens on the action, not on an absence. "No setup on this
            # computer" described a thing that does not happen, which is the
            # weakest way to say the strongest fact this side has: the user can
            # segment their first object now, without the wall the other card
            # puts in front of them.
            #
            # This id is SERVED, so the sentence a user reads is the site's, in
            # their own language. Reword it there, never here: the shipped one
            # below is the offline fallback and carries eleven translations that
            # a new English source would drop.
            return _fill_engine_line(dial_copy(
                "engine.cloud_line",
                tr("Start now, nothing to install {dot} "
                   "<b>1 cloud detection per object you save</b>")))
        if not self._manual_engine_local_ready():
            if self._manual_install_running():
                # Do not promise "runs on your computer" while it is still
                # being put there. The install window is modal and holds QGIS
                # until it ends, so this line cannot offer an action: the one
                # way to Cloud AI from here is the window's own Stop.
                return _fill_engine_line(dial_copy(
                    "engine.installing_line",
                    tr("Setting up on your computer {dot} <b>wait for it "
                       "to finish</b>")))
            if getattr(self, "_manual_install_failed", False):
                # The card under this one says the install failed. Selling the
                # ten-minute install over it read as if nothing had happened.
                return _fill_engine_line(dial_copy(
                    "engine.install_failed_line",
                    tr("The install did not finish {dot} <b>retry it, or "
                       "pick Cloud AI</b>")))
            return self._manual_engine_install_copy()
        # Where it runs, then what it costs, and nothing about what does or
        # does not leave the machine. Two versions have now made that mistake:
        # "Nothing is sent" named a risk in order to deny it, and "Everything
        # stays on this computer" said the same thing in the positive. Both
        # answer a question the user had not asked, and both make the other
        # half of the switch read as the risky one.
        return _fill_engine_line(dial_copy(
            "engine.local_line",
            tr("Runs on this computer {dot} <b>save as many as you like</b>")))

    def _on_manual_low_credit_link(self, url: str) -> None:
        """The Pro link inside this mode's running-low line.

        Same destination as Automatic's, counted under its own source. The two
        modes now show the same sentence, but a click here is a Semi auto user
        reaching the end of a balance one saved object at a time, and folding
        it into the other would leave the impression this mode already tracks
        with nothing to divide it by.
        """
        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pro_upsell_clicked(
                source="manual_credits_low")
        except Exception:  # noqa: BLE001 -- telemetry never blocks a click
            pass  # nosec B110
        QDesktopServices.openUrl(QUrl(url))

    def _write_manual_low_credit_line(self, low: bool) -> None:
        """The running-low warning under the engine note. Never raises.

        Word for word the Automatic Start step's line, with "objects you save"
        where that one says "detections", because that is what a credit buys
        here. The link names Pro rather than offering to "get more credits":
        what is on the other side is a paid plan, and the button that hid that
        was the one thing users had to click to find out.
        """
        line = getattr(self, "manual_engine_low_line", None)
        if line is None:
            return
        try:
            if not low:
                line.setVisible(False)
                return
            left = int(getattr(self, "_auto_credits", 0) or 0)
            reset_day = getattr(self, "_auto_reset_display", "")
            url = self._build_upgrade_url()
            if reset_day:
                line.setText(tr(
                    'Running low: {n} cloud detections left, back on {date}. '
                    '<a href="{url}">Upgrade to Pro</a> to keep going.'
                ).format(n=left, date=reset_day, url=url))
            else:
                line.setText(tr(
                    'Running low: {n} cloud detections left. '
                    '<a href="{url}">Upgrade to Pro</a> to keep going.'
                ).format(n=left, url=url))
            line.setVisible(True)
        except (RuntimeError, AttributeError):
            pass  # nosec B110 -- teardown

    def _manual_engine_install_copy(self) -> str:
        """What the offline AI asks for before Start is worth pressing.

        Time and disk, together. The install already refuses below a free-space
        floor, so a user who reads only the minutes can lose them to a wall this
        could have warned about. The floor is the served one, the same number
        the install itself checks.
        """
        # The minutes move by serving the sentence, not by rewriting it. These
        # two shipped strings are the lookup keys for eleven translations, so
        # they keep their exact wording, ten included; a served line is free to
        # quote another figure, in any language. {n} is filled anyway, so a
        # served value may carry the placeholder instead of a literal.
        minutes = str(local_install_minutes())
        fallback = _fill_engine_line(
            dial_copy("engine.install_line_no_disk",
                      tr("Everything stays on this computer {dot} <b>about "
                         "10 minutes to install</b>")),
            n=minutes)
        # `resolved_min_free_gb_full`, not `min_free_gb_full`: the second name
        # belongs to install_config and never existed here, so this import
        # raised on every refresh and the disk figure never once reached a
        # user. A memory read, no disk call: this runs on every redraw of the
        # Start view.
        try:
            from ...core.venv_manager import resolved_min_free_gb_full

            gb = float(resolved_min_free_gb_full())
        except Exception:  # noqa: BLE001 -- an unknown floor drops the figure
            return fallback
        if gb <= 0:
            return fallback
        return _fill_engine_line(
            dial_copy("engine.install_line",
                      tr("Everything stays on this computer {dot} <b>{gb} GB "
                         "and about 10 minutes to install</b>")),
            gb=f"{gb:g}", n=minutes)

    def _refresh_manual_engine_card_enabled(self) -> None:
        """Keep the Start button honest about the engine the switch is on.

        The switch itself is never disabled. Picking where the work runs does
        not depend on a raster or on the Terms, and greying the one control a
        card exists for is the wall of dead controls the design system bans.
        """
        try:
            from ...core.activation_manager import has_tos_accepted, has_tos_locked

            ready = bool(
                self.layer_combo.currentLayer() is not None
                and (has_tos_locked() or has_tos_accepted())
            )
            self._refresh_manual_start_action(ready)
        except (RuntimeError, AttributeError):
            pass  # nosec B110 -- teardown

    def _refresh_manual_start_action(self, ready: bool) -> None:
        """Make the Start button say what the picked engine turns it into.

        With the box cleared and nothing installed, pressing Start starts a
        download, not a session. Saying "Start" there and greying it out is the
        wall of dead controls the design system bans: the button keeps working
        and names the download instead.
        """
        button = getattr(self, "start_button", None)
        if button is None:
            return
        # A download on the wire this second, NOT "a download is needed".
        # _setup_section_wanted means the second one: it is set the moment the
        # panel learns the pieces are missing, so reading it here made
        # needs_install and not installing dead, and the button kept saying
        # Start on the one machine that cannot start.
        installing = self._manual_install_running()
        needs_install = bool(
            manual_cloud_route_offered()
            and not self._manual_cloud_route_picked()
            and not self._manual_engine_local_ready()
        )
        try:
            if needs_install and not installing:
                button.setText(tr("Install the offline AI"))
                button.setEnabled(ready and not self._segmentation_active)
            else:
                button.setText(tr("Start Semi-Auto AI Segmentation"))
            # Set from here, not once at build time. A tooltip that names the
            # Terms sits on an enabled button for the whole rest of the
            # session otherwise, and it is the only thing on screen that would
            # explain a disabled one. It also has to name the blocker it is
            # actually looking at: a user mid-install has a raster and has
            # accepted the Terms, and being told to go and do both reads as the
            # panel having lost track of them.
            button.setToolTip("" if button.isEnabled()
                              else self._manual_start_blocked_reason(ready))
        except (RuntimeError, AttributeError):
            pass  # nosec B110 -- teardown

    def _manual_start_blocked_reason(self, ready: bool) -> str:
        """Why Start is grey, in the order the user can act on.

        The install states come first: a user watching a download already has
        a raster and has accepted the Terms, so naming those two there sends
        them to look for something that is not missing.
        """
        if self._manual_install_running():
            return tr("The offline AI is still downloading.")
        if getattr(self, "_manual_install_failed", False):
            return tr("The install did not finish. Retry it, or pick Cloud AI.")
        if self._segmentation_active:
            return tr("A session is already running.")
        if not ready:
            return tr("Pick a raster layer and accept the Terms to start.")
        return tr("The offline AI is not installed yet.")

    def set_manual_cloud_session_spend(self, saved: int) -> None:
        """Record how many objects this cloud session has charged for.

        Called by the plugin on every billed save. The footer gauge is what
        shows the balance during a session; this only keeps the count the
        session recap reads.
        """
        self._manual_cloud_objects_saved = max(0, int(saved))

    # -- the clicks ---------------------------------------------------------

    def _track_manual_engine(self, what: str, accepted: bool = False,
                             from_install_gate: bool = False) -> None:
        """Report a choice. Never raises: telemetry does not gate a click."""
        try:
            from ...core import telemetry_session_events

            if what == "consent":
                telemetry_session_events.track_manual_cloud_consent(accepted)
                return
            telemetry_session_events.track_manual_engine_chosen(
                what,
                local_installed=self._manual_engine_local_ready(),
                from_install_gate=from_install_gate,
            )
        except Exception:  # noqa: BLE001 -- best effort  # nosec B110
            pass

    def _on_manual_engine_picked(self, cloud: bool) -> None:
        """The switch IS the choice, unless the picked half is not on the disk.

        Moving to the machine used to store the choice and let the panel grow
        an install card underneath, which is a download starting from a control
        that said "My computer". The window asks first (see
        manual_local_install.py) and stores the choice only on a yes, so the
        cards never sit on an engine that cannot answer a click.
        """
        if not cloud and not self._manual_engine_local_ready():
            self._open_manual_local_install()
            return
        self._set_manual_engine_cloud(bool(cloud))

    def _set_manual_engine_cloud(self, on: bool) -> None:
        """Move the route and tell the whole page about it."""
        # Worth telling apart: cloud picked while a download is running is a
        # user who did not want to wait it out, not a first choice.
        from_gate = bool(on and not self._manual_engine_local_ready())
        set_manual_cloud_route_enabled(on)
        self._track_manual_engine("cloud" if on else "offline",
                                  from_install_gate=from_gate)
        # Picking the cloud card is the earliest honest sign a click is coming,
        # and a machine takes about half a minute to start. Told here, the
        # plugin can ask for one while the user is still framing their object,
        # rather than at Start, which measured as too late: the click itself was
        # what started the machine in nearly every Semi-Auto session.
        try:
            self.manual_engine_changed.emit(bool(on))
        except (RuntimeError, AttributeError):
            pass  # nosec B110 -- teardown
        self._update_full_ui()

    def _manual_engine_gate_start(self) -> bool:
        """Answer the Start button before a Semi-Auto session opens.

        True means start now. False means the click was spent on something
        else and no session may open: the data notice came back refused, or
        the offline AI has to be downloaded before it can answer anything.
        """
        try:
            if self._mode != Mode.INTERACTIVE or not manual_cloud_route_offered():
                return True
            # The picked half, not the running one: the data notice is exactly
            # what this is about to ask, so a predicate that already required
            # it would send every first-time user down the other branch and
            # the question would never be put.
            cloud = self._manual_cloud_route_picked()
        except (RuntimeError, AttributeError):
            return True

        if cloud:
            if cloud_consent_given():
                return True
            from .manual_cloud_consent import ask_cloud_consent

            accepted = ask_cloud_consent(self)
            set_cloud_consent(accepted)
            self._track_manual_engine("consent", accepted=accepted)
            if accepted:
                return True
            # They read it and said no. Refusing is a valid answer, not an
            # error to recover from, so the switch moves to the engine that
            # asks no such question and the user presses Start again.
            self._set_manual_engine_cloud(False)
            return False

        if self._manual_engine_local_ready():
            return True
        # One door to the download, the same one the engine cards use. Start
        # only reaches it after an install was stopped or failed and left the
        # switch on a half with nothing behind it.
        self._open_manual_local_install()
        return False
