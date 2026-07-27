"""The Automatic Detail slider: its object gate, its seeding and cap, the
free-plan premium gate and the hint line under it.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.
"""
from __future__ import annotations

from qgis.PyQt.QtWidgets import (
    QGraphicsOpacityEffect,
)

from ...core.i18n import tr
from ...core.tile_manager import MAX_DETAIL_LEVEL
from .styles import (
    _PREMIUM_STAR,
    BRAND_BLUE,
    _msg_label_qss,
    _msg_text,
)


class DockAutoDetailLevelMixin:
    """The Automatic Detail slider: its object gate, its seeding and cap, the
    free-plan premium gate and the hint line under it."""

    def _apply_auto_detail_gate(self, has_object: bool) -> None:
        """Grey the whole Detail card until the object is defined (typed prompt
        or drawn example). The slider's default is object-aware, so an
        adjustment made BEFORE the object was named was thrown away by the
        prompt-commit re-seed: gating the control makes the order explicit.
        Disabling the container blocks every child, and the opacity dim makes
        the gate unmistakable (a same-color disabled slider read as broken);
        the slider QSS adds a grey :disabled track on top so no brand blue
        survives the dim. The programmatic seed still lands (setValue works
        while disabled). The one-line hint explains the greyed state instead
        of leaving a dead control unexplained."""
        try:
            card = self.auto_detail_row
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
                self.auto_detail_hint.setStyleSheet(
                    "font-size: 10px; color: palette(text);")
                self.auto_detail_hint.setText(tr(
                    "Name the object (or draw an example) first - Detail "
                    "then tunes itself to it."))
        except (RuntimeError, AttributeError):
            pass

    def _on_auto_detail_changed(self, value: int) -> None:
        # The slider now shows plain Coarse/Fine ends; the only numeric feedback
        # is the credit cost, which the plugin recomputes from the real grid.
        self._refresh_auto_detail_hint()
        self.auto_detail_changed.emit(value)

    def set_auto_detail_value(self, n: int) -> None:
        """Seed the detail slider with a good default for a freshly drawn zone.

        Signal-free: the plugin recomputes the credit estimate (and the proper
        slider max) right after. Raises the max first if needed so the seeded
        value is not clamped by a previous zone's smaller cap.
        """
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
                tr("This area is large for this detail level. Raise detail or zoom"
                   " in for sharper detections.")
                if s.value() < s.maximum() else
                tr("This zone is too large for sharp detections, even at maximum"
                   " detail. Draw a smaller zone for the best results."))
        self.auto_detail_warning.setVisible(coarse)
        self.auto_detail_hint.setVisible(not coarse)

    def set_auto_detail_max(self, n: int) -> None:
        """Cap the detail slider at ``n`` useful levels (1-MAX_DETAIL_LEVEL).

        Clamps the current value down if it now exceeds the cap. Signal-free
        on purpose: the plugin calls this from _update_credit_estimate right
        before recomputing the grid, so the clamped value is picked up
        immediately without a re-entrant signal.
        """
        n = max(1, min(MAX_DETAIL_LEVEL, int(n)))
        slider = self.auto_detail_slider
        slider.blockSignals(True)
        slider.setMaximum(n)
        if slider.value() > n:
            slider.setValue(n)
        slider.blockSignals(False)
        self._refresh_auto_detail_hint()

    def set_auto_free_run_cap(self, cap: int | None) -> None:
        """Per-run credit cap for the free plan (None = subscriber, uncapped).

        Set by the plugin from the credit-estimate chokepoint, right before
        the estimate itself lands. The slider keeps its full (Pro) travel; the
        cap gates DETECT instead: set_auto_credit_estimate compares the live
        estimate against it and flips the premium gate."""
        self._auto_free_run_cap = int(cap) if cap is not None else None

    def _set_auto_premium_gated(self, gated: bool) -> None:
        """Flip the free-plan premium gate (estimate above the per-run cap).

        Greys Detect (via _update_auto_detect_enabled, run by the caller) and
        swaps the detail hint to the upgrade link. The upsell view is tracked
        once per gate episode (rising edge)."""
        gated = bool(gated)
        if gated == self._auto_premium_gated:
            return
        self._auto_premium_gated = gated
        if gated and not self._detail_cap_upsell_tracked:
            self._detail_cap_upsell_tracked = True
            try:
                from ...core import telemetry
                telemetry.track_pro_upsell_viewed(trigger="detail_cap")
            except Exception:
                pass  # nosec B110
        elif not gated:
            # Next gate episode counts as a fresh upsell view.
            self._detail_cap_upsell_tracked = False
        self._refresh_auto_detail_hint()

    def _on_detail_cap_upgrade_link(self, _href: str = "") -> None:
        """Upgrade link inside the detail hint: same dashboard URL as every
        other upsell surface, its own telemetry source."""
        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices
        try:
            from ...core import telemetry
            telemetry.track_pro_upsell_clicked(source="detail_cap")
        except Exception:
            pass  # nosec B110
        QDesktopServices.openUrl(QUrl(self._build_upgrade_url()))

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
        """Swap the muted line under the detail slider by state. Premium-gated
        (the run costs more credits than the free plan allows in one go, which
        detail and zone size both drive) shows the upgrade link; then the
        object-aware verdict when one is known, so the
        guidance moves live with the slider, the prompt and the zone; the
        handle sitting at a zone/native-capped maximum keeps the
        draw-a-larger-zone advice when raising detail is the (impossible)
        fix, so a slider that stops early never reads as broken. Same label,
        text swap only (no layout jump)."""
        s = self.auto_detail_slider
        capped = s.maximum() < MAX_DETAIL_LEVEL and s.value() >= s.maximum()
        feedback = getattr(self, "_auto_detail_feedback", None)
        _plain_hint = "font-size: 10px; color: palette(text);"
        if self._auto_premium_gated:
            # Premium taxonomy: a dedicated blue-family line with the star
            # prefix and an underlined upgrade link (never inline in guidance).
            self.auto_detail_hint.setStyleSheet(_msg_label_qss("premium"))
            _hint = _PREMIUM_STAR + " "
            # Name the mechanism, which is a per-run credit ceiling, not a
            # locked slider position. Detail and zone both feed the estimate,
            # so both are offered as the fix: the old "this detail level is a
            # Pro feature" was simply false for a big zone at minimum detail.
            _cap = getattr(self, "_auto_free_run_cap", None)
            if _cap:
                _hint += tr(
                    "A free run covers up to {cap} credits. Lower the detail "
                    "or draw a smaller zone, or").format(cap=int(_cap))
            else:
                _hint += tr(
                    "This run costs more credits than a free run covers. "
                    "Lower the detail or draw a smaller zone, or")
            _hint += f' <a href="upgrade" style="color: {BRAND_BLUE};'
            _hint += ' text-decoration: underline;">'
            _hint += tr("upgrade to unlock it")
            _hint += "</a>."
            self.auto_detail_hint.setText(_hint)
            return
        if feedback and not (capped and feedback[0] in ("coarse", "below")):
            # "Raise the detail" advice is a dead end at a capped maximum;
            # the capped branch below gives the actionable fix instead.
            state, word = feedback
            obj = f'"{word}"' if word else tr("your object")
            if state == "coarse":
                self.auto_detail_hint.setStyleSheet(_msg_label_qss("warning"))
                self.auto_detail_hint.setText(_msg_text("warning", tr(
                    "At this detail {obj} is too small to spot - raise the"
                    " detail.").format(obj=obj)))
            elif state == "over":
                # Quality fact only (large objects can fragment past this
                # point); never a nudge about credits - the cost line above
                # already says the price, guidance stays informational.
                self.auto_detail_hint.setStyleSheet(_msg_label_qss("warning"))
                self.auto_detail_hint.setText(_msg_text("warning", tr(
                    "Very fine for {obj} - large ones may come back split"
                    " in parts.").format(obj=obj)))
            elif state == "above":
                self.auto_detail_hint.setStyleSheet(_plain_hint)
                self.auto_detail_hint.setText(tr(
                    "Sharper than {obj} usually needs - catches the smallest"
                    " ones.").format(obj=obj))
            elif state == "helps":
                self.auto_detail_hint.setStyleSheet(_plain_hint)
                self.auto_detail_hint.setText(tr(
                    "Extra detail keeps helping {obj} in this zone.").format(
                        obj=obj))
            elif state == "below":
                self.auto_detail_hint.setStyleSheet(_plain_hint)
                self.auto_detail_hint.setText(tr(
                    "Small {obj} may be missed at this level.").format(obj=obj))
            else:  # recommended
                self.auto_detail_hint.setStyleSheet(_plain_hint)
                self.auto_detail_hint.setText("✓ " + tr(
                    "Right level for {obj} in this zone.").format(obj=obj))
            return
        if capped:
            self.auto_detail_hint.setStyleSheet(_plain_hint)
            self.auto_detail_hint.setText(tr(
                "Max detail for this zone - draw a larger zone for finer detail."))
        else:
            self.auto_detail_hint.setStyleSheet(_plain_hint)
            self.auto_detail_hint.setText(tr("Finer detail finds smaller objects."))
