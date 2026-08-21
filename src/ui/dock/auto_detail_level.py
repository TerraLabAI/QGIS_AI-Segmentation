"""The Automatic Detail slider: its object gate, its seeding and cap, the
free-plan premium gate and the hint line under it.

The control is labelled "Precision" on screen. Every identifier here stays
``detail`` (widget names, the MCP API and the telemetry keys are bound to it),
so "Detail" in this file means the code, "Precision" means the words the user
reads.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.
"""
from __future__ import annotations

from qgis.PyQt.QtWidgets import (
    QGraphicsOpacityEffect,
)

from ...core.i18n import tr
from ...core.server_dials import dial_copy
from ...core.tile_manager import MAX_DETAIL_LEVEL
from .font_scale import scale_qss_font_px
from .styles import (
    _SECTION_TOGGLE_OPEN_QSS,
    _SECTION_TOGGLE_QSS,
    _msg_label_qss,
    _msg_text,
)


def _detail_hint_copy(state: str, fallback: str, obj: str = "") -> str:
    """The sentence for one Precision-slider state, server copy first.

    What the slider tells a user is advice about their object, and which advice
    is right for a class is the kind of thing a bad field report should be able
    to correct in an hour. Shipped in the binary it waits for the next release
    instead, on a control whose whole job is to be understood. So each state
    gets a flat ``copy.detail_hint.<state>`` id with the English sentence as the
    fallback.

    ``{obj}`` is filled with str.replace, never str.format: a served sentence is
    outside data and a single stray brace in it would raise here, on the path
    that paints the dock.
    """
    text = dial_copy(f"detail_hint.{state}", fallback)
    return text.replace("{obj}", obj) if obj else text


class DockAutoDetailLevelMixin:
    """The Automatic Detail slider: its Advanced settings fold, its object
    gate, its seeding and cap, the free-plan premium gate and the hint line
    under it."""

    def _refresh_auto_advanced_header(self) -> None:
        """Chevron + title on the fold's head (text swap only).

        The title lives in its own label, not the button's own text: see
        the comment where auto_advanced_toggle_title is built.
        """
        try:
            arrow = "\u25be" if self._auto_advanced_open else "\u25b8"
            self.auto_advanced_toggle_title.setText(
                arrow + " " + tr("Advanced settings"))
        except (RuntimeError, AttributeError):
            pass

    def _on_auto_advanced_toggle_clicked(self) -> None:
        """Head clicked: flip the fold and tell the plugin.

        The signal carries the new state because the tile grid follows it: the
        canvas draws the split only while the panel that explains it is open
        (see AutoZoneMixin._tile_grid_revealed). Pure setVisible here, so
        flipping the fold never emits a control signal of its own.
        """
        self.set_auto_advanced_open(not self._auto_advanced_open)

    def set_auto_advanced_open(self, open_: bool) -> None:
        """Open or shut the Advanced settings fold, and announce the state.

        The one way in, so the canvas and the panel can never disagree. Callers
        outside the click handler: none today, and it stays public for the MCP
        surface, which drives this dock the way a user does.
        """
        open_ = bool(open_)
        if open_ == getattr(self, "_auto_advanced_open", False):
            return
        try:
            self.auto_advanced_body.setVisible(open_)
            self.auto_advanced_toggle_btn.setStyleSheet(
                _SECTION_TOGGLE_OPEN_QSS if open_ else _SECTION_TOGGLE_QSS)
        except (RuntimeError, AttributeError):
            return
        self._auto_advanced_open = open_
        self._refresh_auto_advanced_header()
        self.auto_advanced_toggled.emit(open_)

    def is_auto_advanced_open(self) -> bool:
        """Whether the user has opened the Advanced settings fold."""
        return bool(getattr(self, "_auto_advanced_open", False))

    def _apply_auto_detail_gate(self, has_object: bool) -> None:
        """Grey the Precision controls until the object is defined (typed
        prompt or drawn example). The slider's default is object-aware, so an
        adjustment made BEFORE the object was named was thrown away by the
        prompt-commit re-seed: gating the control makes the order explicit.
        Disabling the container blocks every child, and the opacity dim makes
        the gate unmistakable (a same-color disabled slider read as broken);
        the slider QSS adds a grey :disabled track on top so no brand blue
        survives the dim. The programmatic seed still lands (setValue works
        while disabled). The one-line hint explains the greyed state instead
        of leaving a dead control unexplained."""
        try:
            # The fold's BODY, never the whole row: the head has to stay
            # clickable (a user may open it to look at the tiles before they
            # have named anything), and the surface, the envelope wall and the
            # cloud disclosure under it must never be dimmed.
            card = self.auto_advanced_body
            if card.isEnabled() == has_object:
                return
            card.setEnabled(has_object)
            if has_object:
                card.setGraphicsEffect(None)
                # Route through the shared refresher so a capped slider keeps
                # its capped wording (free-plan upsell or zone advice).
                self._refresh_auto_detail_hint()
            else:
                dim = QGraphicsOpacityEffect(card)
                dim.setOpacity(0.45)
                card.setGraphicsEffect(dim)
                self.auto_detail_hint.setStyleSheet(scale_qss_font_px(
                    "font-size: 10px; color: palette(text);"))
                self.auto_detail_hint.setText(tr(
                    "Name the object (or draw an example) first - Precision "
                    "then tunes itself to it."))
        except (RuntimeError, AttributeError):
            pass

    def _on_auto_detail_changed(self, value: int) -> None:
        # The slider now shows plain Coarse/Fine ends; the only numeric feedback
        # is the credit cost, which the plugin recomputes from the real grid.
        # The hint follows every tick of the drag; the signal that drives the
        # recompute settles first, so dragging across many levels fires it once.
        self._refresh_auto_detail_hint()
        self._auto_detail_pending_value = value
        timer = getattr(self, "_auto_detail_emit_timer", None)
        if timer is None:
            from qgis.PyQt.QtCore import QTimer
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._flush_auto_detail_changed)
            self._auto_detail_emit_timer = timer
        timer.start(150)

    def _flush_auto_detail_changed(self) -> None:
        """Emit the Precision change once the drag has settled."""
        value = getattr(self, "_auto_detail_pending_value", None)
        if value is not None:
            self.auto_detail_changed.emit(value)

    def _cancel_pending_auto_detail(self) -> None:
        """Drop a tick that has not been emitted yet.

        Every programmatic reseed calls this first. The debounce holds the
        value the USER last dragged to, and a reseed means that choice is about
        a zone or a prompt that is gone: letting it fire afterwards writes the
        old level back over the new one and marks it as chosen by hand.
        """
        timer = getattr(self, "_auto_detail_emit_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except (RuntimeError, AttributeError):
                pass  # nosec B110 -- a timer already gone holds nothing
        self._auto_detail_pending_value = None

    def set_auto_detail_value(self, n: int) -> None:
        """Seed the detail slider with a good default for a freshly drawn zone.

        Signal-free: the plugin recomputes the credit estimate (and the proper
        slider max) right after. Raises the max first if needed so the seeded
        value is not clamped by a previous zone's smaller cap.
        """
        self._cancel_pending_auto_detail()
        s = self.auto_detail_slider
        s.blockSignals(True)
        if s.maximum() < n:
            s.setMaximum(min(MAX_DETAIL_LEVEL, int(n)))
        s.setValue(max(s.minimum(), min(s.maximum(), int(n))))
        s.blockSignals(False)
        self._refresh_auto_detail_hint()

    def set_auto_detail_visible(self, visible: bool) -> None:
        """Show the detail slider whenever a zone is drawn; hidden while no zone is set."""
        self.auto_detail_row.setVisible(visible)

    def set_auto_detail_gsd_warning(self, coarse: bool) -> None:
        """Show the boxed amber warning when the chosen detail leaves the imagery
        too coarse for the cloud model (effective ground resolution >= ~0.5 m/px, where
        detection quality drops sharply). The detail seed now auto-raises past
        the soft tile budget, so this fires only when the USER dragged detail
        down (fix: raise it back) or the zone is so large even the slider max
        stays coarse (fix: a smaller zone; "raise detail" would be a dead end).
        The neutral hint hides while the warning shows so the two never stack."""
        if coarse:
            s = self.auto_detail_slider
            self.auto_detail_warning_label.setText(
                tr("This area is large for this precision. Raise the precision or zoom"
                   " in for sharper detections.")
                if s.value() < s.maximum() else
                tr("This zone is too large for sharp detections, even at full"
                   " precision. Draw a smaller zone for the best results."))
        self.auto_detail_warning.setVisible(coarse)
        self.auto_detail_hint.setVisible(not coarse)

    def set_auto_detail_range(
        self, lo: int, hi: int, object_bound: bool = False
    ) -> None:
        """Set the detail slider's travel to the useful band ``lo``-``hi``.

        The band is the levels worth offering for the object the user named
        (see ``ui/plugin/auto_detail_window.py``): under ``lo`` the object is
        too few pixels across to spot, past ``hi`` it stops fitting in a tile
        and comes back in fragments. ``object_bound`` says the fine end came
        from the OBJECT rather than from the zone or the source resolution,
        which is what the hint at the top of the travel explains.

        Clamps the current value into the band. Signal-free on purpose: the
        plugin calls this from _update_credit_estimate right before recomputing
        the grid, so the clamped value is picked up without a re-entrant signal.
        """
        hi = max(1, min(MAX_DETAIL_LEVEL, int(hi)))
        lo = max(1, min(hi, int(lo)))
        self._auto_detail_object_bound = bool(object_bound)
        self._cancel_pending_auto_detail()
        slider = self.auto_detail_slider
        slider.blockSignals(True)
        # Widen, then set the ends. Qt clamps each end against the CURRENT
        # other one, so a band sitting entirely above the old maximum (or below
        # the old minimum) collapses to a point unless the room is made first.
        slider.setMaximum(max(hi, slider.maximum()))
        slider.setMinimum(lo)
        slider.setMaximum(hi)
        value = min(max(slider.value(), lo), hi)
        if slider.value() != value:
            slider.setValue(value)
        slider.blockSignals(False)
        # One useful level is no choice, and a slider pinned to a single tick
        # reads as broken. Hide the control and its cost line; the hint below
        # says why and what would give the user a choice back.
        self._auto_detail_single_level = lo >= hi
        try:
            self.auto_detail_slider_row.setVisible(lo < hi)
            self.auto_detail_sub.setVisible(lo < hi)
        except (RuntimeError, AttributeError):
            pass
        self._refresh_auto_detail_hint()

    def _on_detail_cap_upgrade_link(self, _href: str = "") -> None:
        """Upgrade link inside the detail hint: same dashboard URL as every
        other upsell surface, its own telemetry source."""
        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pro_upsell_clicked(source="detail_cap")
        except Exception:
            pass  # nosec B110
        QDesktopServices.openUrl(QUrl(self._build_upgrade_url("plugin_detail_cap")))

    def set_auto_detail_feedback(self, state: str | None, object_word: str) -> None:
        """Live verdict for the CURRENT slider level against the named object
        and the drawn zone, computed by the plugin at the credit-estimate
        chokepoint. States: coarse / below / recommended / helps / above /
        over (None clears). Stored here and rendered by
        _refresh_auto_detail_hint, which owns the priority order."""
        word = (object_word or "").strip()
        if len(word) > 24:
            word = word[:24] + "..."
        self._auto_detail_feedback = (state, word) if state else None
        self._refresh_auto_detail_hint()

    def _refresh_auto_detail_hint(self) -> None:
        """Swap the muted line under the detail slider by state: the
        object-aware verdict when one is known, so the
        guidance moves live with the slider, the prompt and the zone; the
        handle sitting at a zone/native-capped maximum keeps the
        draw-a-larger-zone advice when raising detail is the (impossible)
        fix, so a slider that stops early never reads as broken. Same label,
        text swap only (no layout jump)."""
        s = self.auto_detail_slider
        capped = s.maximum() < MAX_DETAIL_LEVEL and s.value() >= s.maximum()
        feedback = getattr(self, "_auto_detail_feedback", None)
        _plain_hint = scale_qss_font_px("font-size: 10px; color: palette(text);")
        self.auto_detail_hint.setToolTip("")
        if getattr(self, "_auto_detail_single_level", False):
            word = feedback[1] if feedback else ""
            obj = f'"{word}"' if word else tr("your object")
            self.auto_detail_hint.setStyleSheet(_plain_hint)
            self.auto_detail_hint.setText(_detail_hint_copy(
                "single", tr(
                    "One precision level fits {obj} in a zone this size - draw"
                    " a larger zone for a choice."), obj))
            return
        if capped and getattr(self, "_auto_detail_object_bound", False):
            # The travel stops here because the OBJECT stops fitting in a tile,
            # not because the zone or the imagery ran out. That outranks the
            # live verdict below: a user who dragged to the end is asking why
            # it ends, and "draw a larger zone" (the other capped branch) would
            # send them to spend credits on the fragmenting they just avoided.
            word = feedback[1] if feedback else ""
            obj = f'"{word}"' if word else tr("your object")
            self.auto_detail_hint.setStyleSheet(_plain_hint)
            self.auto_detail_hint.setText(_detail_hint_copy(
                "objcap",
                tr("As fine as {obj} benefits from - finer splits them into"
                   " pieces."), obj))
            return
        if feedback and not (capped and feedback[0] in ("coarse", "below")):
            # "Raise the detail" advice is a dead end at a capped maximum;
            # the capped branch below gives the actionable fix instead.
            state, word = feedback
            obj = f'"{word}"' if word else tr("your object")
            if state == "coarse":
                self.auto_detail_hint.setStyleSheet(_msg_label_qss("warning"))
                self.auto_detail_hint.setText(_msg_text("warning", _detail_hint_copy(
                    "coarse", tr(
                        "At this precision {obj} is too small to spot - raise the"
                        " precision."), obj)))
            elif state == "over":
                # Quality fact only (large objects can fragment past this
                # point); never a nudge about credits - the cost line above
                # already says the price, guidance stays informational.
                self.auto_detail_hint.setStyleSheet(_msg_label_qss("warning"))
                self.auto_detail_hint.setText(_msg_text("warning", _detail_hint_copy(
                    "over", tr(
                        "Very fine for {obj} - large ones may come back split"
                        " in parts."), obj)))
            elif state == "above":
                self.auto_detail_hint.setStyleSheet(_plain_hint)
                self.auto_detail_hint.setText(_detail_hint_copy(
                    "above", tr(
                        "Sharper than {obj} usually needs - catches the smallest"
                        " ones."), obj))
            elif state == "helps":
                self.auto_detail_hint.setStyleSheet(_plain_hint)
                self.auto_detail_hint.setText(_detail_hint_copy(
                    "helps",
                    tr("More precision keeps helping {obj} in this zone."), obj))
            elif state == "below":
                self.auto_detail_hint.setStyleSheet(_plain_hint)
                self.auto_detail_hint.setText(_detail_hint_copy(
                    "below",
                    tr("Small {obj} may be missed at this level."), obj))
            else:  # recommended
                self.auto_detail_hint.setStyleSheet(_plain_hint)
                self.auto_detail_hint.setText("✓ " + _detail_hint_copy(
                    "recommended",
                    tr("Right level for {obj} in this zone."), obj))
            return
        if capped:
            self.auto_detail_hint.setStyleSheet(_plain_hint)
            self.auto_detail_hint.setText(_detail_hint_copy("capped", tr(
                "Max precision for this zone - draw a larger zone to go finer.")))
        else:
            self.auto_detail_hint.setStyleSheet(_plain_hint)
            self.auto_detail_hint.setText(_detail_hint_copy(
                "default", tr("More precision finds smaller objects.")))
