"""What a run says while it works: the progress card, the queue and warm-up
wait copy, the status banner and the zero-result rescue chips.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.
"""
from __future__ import annotations

import time

from qgis.PyQt.QtCore import Qt

from ...core.i18n import tr
from .prompt_guard import validate_prompt
from .styles import (
    _REPORT_HREF,
    _error_banner_html,
    _msg_label_qss,
    _msg_text,
)

# The run bar is scaled in permille, not in tiles. Answers arrive in bursts (the
# window holds many /predict calls and the whole cycle's replies are read back to
# back), and Qt compresses a burst of setValue calls into ONE repaint, so a
# tile-grained bar teleports: nothing, then a wide jump. A fine range plus the
# easing tick below let the fill travel that distance instead of skipping it.
_PROGRESS_SCALE = 1000
# ~30 fps. Enough to read as motion, cheap enough to run beside the geometry
# pump that folds tiles on this same thread.
_PROGRESS_EASE_INTERVAL_MS = 33
# Share of the remaining distance closed per tick, so the fill trails the true
# count by about a tenth of a second whatever the size of the jump.
_PROGRESS_EASE_FRACTION = 0.28


class DockAutoRunStatusMixin:
    """What a run says while it works: the progress card, the queue and warm-up
    wait copy, the status banner and the zero-result rescue chips."""

    def set_auto_run_found_count(self, obj: str, count: int) -> None:
        """Live in-run feedback: the running found-object count, shown in the
        progress card's Row 1 next to the tile count so a slow zone never feels
        dead. ``obj`` is the run's own object word (the prompt as sent, after
        any translation or rewrite the guard applied), quoted in the line so a
        long run says WHAT it is counting. Empty on a run carried by drawn
        examples alone, which has no word to show."""
        self._auto_found_count = max(0, count)
        self._auto_run_object_word = (obj or "").strip()
        if self.auto_progress_card.isVisible():
            self._refresh_auto_progress_readout()

    def _auto_found_so_far_text(self, found: int, prefix: str) -> str:
        """The "302 "building" found so far" tail: the object word quoted when
        the run has one, cut to what Row 1 has left after ``prefix`` (the tile
        count, which is already on the line)."""
        import html

        word = getattr(self, "_auto_run_object_word", "") or ""
        if not word:
            return tr("{n} found so far").format(n=found)
        template = tr('{n} "{object}" found so far')
        # Same template with an empty word: the part of the line whose width is
        # already spoken for, without inventing a second translatable string.
        fixed = f"{prefix} · " + template.format(n=found, object="")
        word = self._fit_run_object_word(word, fixed)
        return template.format(n=found, object=html.escape(word))

    def _fit_run_object_word(self, word: str, fixed: str) -> str:
        """Cut the object word to the width Row 1 has left for it.

        Measured against the progress CARD, not the label: the label carries
        no wrap, so a long word grows it and measuring it would compare the
        word against a width the word itself created. A character cap covers
        the first paint, where nothing has a width yet.
        """
        capped = word if len(word) <= 22 else word[:21] + "\u2026"
        try:
            from qgis.PyQt.QtGui import QFontMetrics

            free = self.auto_progress_card.width() - self.auto_progress_pct_label.width()
            # The row repaints many times a second while answers land, and both
            # the word and the width it is measured against hold still across
            # almost all of those, so the measurement is done once per pair.
            key = (capped, fixed, free)
            cached = getattr(self, "_auto_progress_elide_cache", None)
            if cached is not None and cached[0] == key:
                return cached[1]
            metrics = QFontMetrics(self.auto_progress_count_label.font())
            free -= metrics.horizontalAdvance(fixed)
            # Card margins plus the row spacing, and a little slack for the bold
            # face the stylesheet sets, which the plain font under-measures.
            free -= 34
            if free < 24:
                out = capped
            else:
                out = metrics.elidedText(
                    capped, Qt.TextElideMode.ElideRight, free)
            self._auto_progress_elide_cache = (key, out)
            return out
        except (RuntimeError, AttributeError, TypeError):
            return capped

    def _set_auto_progress_visible(self, visible: bool) -> None:
        """Show/hide the run progress card as one unit (count row + bar + the
        conditional status line)."""
        if not visible:
            # Card gone (review / idle / error): kill both heartbeats so no
            # stray tick repaints a torn-down card.
            self._stop_auto_warming_anim()
            self._stop_auto_progress_ease()
        self.auto_progress_card.setVisible(visible)

    def _paint_auto_finalize_card(self) -> None:
        """Put the run card in its hand-over state: same card, same line the
        last tiles were already showing, on the animated busy bar. Called when
        the run ends and the results start being turned into the review (see
        set_auto_finalizing), so the screen the user is looking at does not
        change until the review replaces it.

        The bar is indeterminate rather than parked full, for the reason
        _start_auto_warming_anim already gives: a bar that sits at 100% while
        work continues reads as a hang, and this stretch is the long one on a
        run whose objects are few but huge. Qt animates an indeterminate range
        on its own, so it keeps moving without a timer. The counters stay at
        full so anything that restores a determinate range finds them sane."""
        self.auto_status_banner.setVisible(False)
        self._set_auto_progress_visible(True)
        self._stop_auto_warming_anim()
        self._stop_auto_progress_ease()
        self._auto_progress_target = _PROGRESS_SCALE
        self._auto_progress_shown = _PROGRESS_SCALE
        self.auto_tile_progress.setRange(0, 0)
        self._auto_progress_dirty = False
        self._refresh_auto_progress_readout()
        self._render_auto_wait_label()

    def set_auto_billed_tile_total(self, total: int) -> None:
        """Remember the BILLED grid size for this run (the count the cost line
        quoted before Detect). Set once at run start; the worker's own total
        grows past it when dense tiles are re-split into free quadrants."""
        self._auto_billed_tile_total = max(0, int(total))
        # Which pass the card is showing (see _auto_progress_phase_pair).
        self._auto_progress_phase = "grid"
        # The pass's own measured pace, for the time-left line. One per pass:
        # the free re-scan starts its count from zero and runs at its own rate.
        self._auto_run_pace = None
        # The assembly pass has not started and owns nothing yet.
        self._auto_assemble_latched = False
        self._auto_assemble_tiles = (0, 0)

    def _auto_progress_phase_pair(self, current: int, total: int) -> tuple:
        """Split the worker's run-wide (completed, total) into the pass the card
        is showing, as ``(phase, done, of)``.

        A run has two passes and they are NOT the same size. The paid grid is
        submitted first and answers first (the worker's queue is FIFO, quadrants
        are appended behind it), then dense tiles are re-scanned as free
        quadrants, which on a city zone is most of the wall clock. Folding both
        into one 0-100 was what pinned the bar at 99% for the longer half of the
        run, so each pass gets the whole bar in turn and its own honest count.

        ``of`` for the second pass grows while parents keep answering; the row
        shows the real number rather than a guessed denominator.
        """
        current = max(0, current)
        if getattr(self, "_auto_finalizing", False):
            # The tiles are all in and the shapes are being built. Its own
            # phase, so it gets its own high-water mark: a count that arrives
            # here used to land under the pass that was already at its peak and
            # be swallowed, which froze the bar for the whole "Almost done"
            # window.
            return "finalize", current, total
        if getattr(self, "_auto_assemble_latched", False):
            # Every tile has been answered and the fold is still behind, so the
            # answered count has nothing left to say: it sat at a full bar for
            # the whole assembly on a large run, next to a time-left line that
            # never moved. From here the bar counts what is actually being
            # worked on.
            folded, given = getattr(self, "_auto_assemble_tiles", (0, 0))
            if given > 0:
                return "assemble", folded, given
            # Nothing was ever handed to a stitcher (the headless path runs
            # without one), so there is no assembly to count and the answered
            # split below stays the honest reading.
        billed = getattr(self, "_auto_billed_tile_total", 0)
        if not billed:
            # No quoted grid (headless/MCP path): nothing to split.
            return "grid", current, total
        refine_total = max(0, total - billed)
        refine_done = max(0, current - billed)
        if refine_total and current >= billed:
            return "refine", refine_done, refine_total
        return "grid", min(current, billed), billed

    def set_auto_assemble_tiles(self, done: int, given: int) -> None:
        """How many of the tiles handed to the live stitcher it has folded in.

        The bar follows the answered tiles while answering is the slow half.
        Once every tile has been answered and the fold is still behind, this
        pair takes the bar over and keeps it: a run whose long tail is assembly
        showed a nearly full bar and "Less than a minute left" for as long as
        the assembly took, which reads as a hang rather than as work.

        Latched, never un-latched before the finalize: the run's tile total
        grows when a dense tile queues its quadrants, so the answered pass can
        briefly look unfinished again, and a bar that swapped back would empty
        its fill and climb a second time.
        """
        pair = (max(0, int(done)), max(0, int(given)))
        if pair == getattr(self, "_auto_assemble_tiles", (0, 0)):
            return
        self._auto_assemble_tiles = pair
        if not getattr(self, "_auto_assemble_latched", False):
            return
        current, total = getattr(self, "_auto_progress_pair", (0, 0))
        self.set_auto_tile_progress(current, total)

    def note_auto_tiles_all_answered(self) -> None:
        """Hand the bar to the assembly count, once the wire has no more to
        give. Called by the run when the last tile is answered; the fold is
        normally still behind at that moment."""
        self._auto_assemble_latched = True

    def set_auto_finalize_tiles(self, done: int, total: int) -> None:
        """How many answered tiles the hand-over has folded in so far. Only
        the finalize row reads it; a run that never reports it shows the
        plain "Building the shapes"."""
        self._auto_finalize_tiles = (max(0, int(done)), max(0, int(total)))
        if getattr(self, "_auto_finalizing", False):
            self._refresh_auto_progress_readout()

    def _refresh_auto_progress_readout(self) -> None:
        """Rebuild the progress card's Row 1 (the pass count + live found count)
        and the right-aligned percent from the remembered pair + found count.

        The percent comes from the bar's own high-water target, never recomputed
        here, so the number beside the fill can never disagree with it."""
        current, total = getattr(self, "_auto_progress_pair", (0, 0))
        found = getattr(self, "_auto_found_count", 0)
        phase, current, total = self._auto_progress_phase_pair(current, total)
        # The tile count is the run's unit, not the user's: it lives in the
        # tooltip. The row says how much of the drawn surface is done.
        tooltip = ""
        if phase == "finalize":
            # Its own words: the tiles are done and nothing more is billed, so
            # the row must not keep counting a pass that has ended. While the
            # answered tiles are still being folded in, the row counts THAT,
            # so a long tail reads as work moving and not as a hang.
            done, of = getattr(self, "_auto_finalize_tiles", (0, 0))
            count_txt = tr("Building the shapes")
            if of > 0 and done < of:
                count_txt += " · " + tr("{current} of {total} tiles").format(
                    current=done, total=of)
        elif phase == "assemble":
            # Same words as the finalize row, because it is the same work: the
            # answers are in and their shapes are being built. Both strings are
            # already in every locale.
            count_txt = tr("Building the shapes")
            if total > 0 and current < total:
                count_txt += " · " + tr("{current} of {total} tiles").format(
                    current=current, total=total)
        elif phase == "refine":
            # Named and separate: the row is the only thing that says this
            # pass exists, and it is where the run spends most of its time.
            # "no extra cost" because a second pass over the same ground reads
            # as a second bill.
            count_txt = tr("Dense area · no extra cost")
            tooltip = tr("{current} of {total} tiles").format(
                current=current, total=total)
        else:
            count_txt = self._auto_surface_done_text(current, total)
            tooltip = tr("{current} of {total} tiles").format(
                current=current, total=total)
        if found > 0:
            found_txt = self._auto_found_so_far_text(found, count_txt)
            count_txt += ' <span style="color: rgba(128,128,128,0.95);">· ' + found_txt + "</span>"
        self.auto_progress_count_label.setText(count_txt)
        self.auto_progress_count_label.setToolTip(tooltip)
        if phase == "finalize":
            # The bar is the animated busy one here, so a percent beside it
            # would name a number nothing is measuring.
            self.auto_progress_pct_label.setVisible(False)
            return
        self.auto_progress_pct_label.setVisible(True)
        pct = getattr(self, "_auto_progress_target", 0) // (_PROGRESS_SCALE // 100)
        self.auto_progress_pct_label.setText(f"{max(0, min(100, pct))}%")

    def _auto_surface_done_text(self, done: int, of: int) -> str:
        """Row 1 of the paid pass: the surface scanned so far out of the zone,
        or the share done when the zone's surface is not known (headless)."""
        km2 = getattr(self, "_auto_zone_km2", None)
        if km2 is None or km2 <= 0 or not of:
            pct = int(100 * min(1.0, max(0.0, done / of))) if of else 0
            return tr("{pct}% done").format(pct=pct)
        from .ui_refresh import format_km2_surface
        done_km2 = km2 * min(1.0, max(0.0, done / of))
        return tr("{done} of {total} km²").format(
            done=format_km2_surface(done_km2), total=format_km2_surface(km2))

    def _note_auto_pace(self, phase: str, done: int, of: int) -> None:
        """Feed one progress report to the pass's pace tracker."""
        if phase == "finalize":
            return
        from ...core.run_eta import RunPace
        pace = getattr(self, "_auto_run_pace", None)
        if pace is None or getattr(self, "_auto_pace_phase", None) != phase:
            pace = RunPace()
            self._auto_run_pace = pace
            self._auto_pace_phase = phase
        pace.note(done, of, time.monotonic())

    def _auto_time_left_text(self) -> str:
        """"About 9 min left" from the run's own pace, or "" while unknown."""
        pace = getattr(self, "_auto_run_pace", None)
        if pace is None:
            return ""
        from ...core.run_eta import friendly_time_left
        left = pace.seconds_left()
        return friendly_time_left(left) if left is not None else ""

    def set_auto_tile_progress(self, current: int, total: int) -> None:
        self.auto_status_banner.setVisible(False)
        self.hide_auto_zero_assist()
        # Remembered so a cleared queue state can restore the live tile count.
        self._auto_progress_pair = (current, total)
        self._set_auto_progress_visible(True)
        phase, done, of = self._auto_progress_phase_pair(current, total)
        self._note_auto_pace(phase, done, of)
        if phase == "finalize":
            # The tiles are in and the hand-over card owns the bar (see
            # _paint_auto_finalize_card): it is the animated busy one, and a
            # late count must only refresh the row, never re-arm a determinate
            # fill that would then sit still for the whole "Almost done"
            # window.
            self._auto_progress_phase = phase
            self._refresh_auto_progress_readout()
            self._render_auto_wait_label()
            return
        if phase != getattr(self, "_auto_progress_phase", "grid"):
            # The paid grid is in and the free re-scan owns the bar now: empty
            # the fill so the second pass measures itself from zero, instead of
            # inheriting a high-water mark it could never move off.
            self._auto_progress_phase = phase
            self._stop_auto_progress_ease()
            self._auto_progress_target = 0
            self._auto_progress_shown = 0
            self.auto_tile_progress.setValue(0)
        ratio = (done / of) if of and done > 0 else 0.0
        self._auto_progress_ratio = ratio
        # The total GROWS mid-run: a saturated tile queues quadrants that become
        # new tiles (worker _drain_subtiles), so done/of can fall between two
        # answers. Within a pass the fill and the percent take the high-water
        # mark, because a bar that walks backwards reads as a bug, not as extra
        # work.
        target = (_PROGRESS_SCALE if of and done >= of
                  else int(round(min(1.0, max(0.0, ratio)) * _PROGRESS_SCALE)))
        self._auto_progress_target = max(
            getattr(self, "_auto_progress_target", 0), target)
        if done <= 0 and phase == "grid":
            # No tile has landed yet (a cold service can take about a minute to answer):
            # keep the bar ALIVE instead of a frozen 0%. _ensure_auto_warming_anim
            # switches the bar to indeterminate (Qt animates it) and runs a 1s
            # timer that evolves the label. Row-3 copy is chosen by the shared
            # renderer below.
            self._stop_auto_progress_ease()
            self._auto_progress_shown = 0
            self._ensure_auto_warming_anim()
            self._refresh_auto_progress_readout()
            self._render_auto_wait_label()
            return
        # Real progress: an honest determinate bar. The warming animation
        # (if it was running) has done its job.
        self._stop_auto_warming_anim()
        if self.auto_tile_progress.maximum() != _PROGRESS_SCALE:
            # Leaving the indeterminate range: paint the fill we believe in, or
            # the first frame shows whatever Qt kept from the busy animation.
            self.auto_tile_progress.setRange(0, _PROGRESS_SCALE)
            self.auto_tile_progress.setValue(
                getattr(self, "_auto_progress_shown", 0))
        if of and done >= of:
            # Last answer in: land on a full bar at once. The card is about to be
            # replaced by the review, and easing here would hide it mid-slide.
            self._stop_auto_progress_ease()
            self._auto_progress_shown = self._auto_progress_target
            self.auto_tile_progress.setValue(self._auto_progress_shown)
            self._refresh_auto_progress_readout()
            self._render_auto_wait_label()
            return
        # Leave the fill AND the text to the tick: a burst of answers would
        # otherwise rebuild the rich-text row once per tile, which is layout work
        # taken from the geometry pump on this same thread, and only the last of
        # those rebuilds would ever be painted.
        self._auto_progress_dirty = True
        self._ensure_auto_progress_ease()

    # ------------------------------------------------------------------
    # Fill easing (one tick, shared by the bar and the coalesced text)
    # ------------------------------------------------------------------

    def _ensure_auto_progress_ease(self) -> None:
        """Start (or keep) the tick that slides the fill toward the true count.
        Idempotent - safe to call on every answer."""
        if getattr(self, "_auto_progress_ease_timer", None) is None:
            from qgis.PyQt.QtCore import QTimer
            timer = QTimer(self)
            timer.setInterval(_PROGRESS_EASE_INTERVAL_MS)
            timer.timeout.connect(self._on_auto_progress_ease_tick)
            self._auto_progress_ease_timer = timer
        if not self._auto_progress_ease_timer.isActive():
            self._auto_progress_ease_timer.start()

    def _stop_auto_progress_ease(self) -> None:
        timer = getattr(self, "_auto_progress_ease_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()

    def _on_auto_progress_ease_tick(self) -> None:
        """One frame: flush the count row if answers landed since the last one,
        then move the fill a step closer to them. Stops itself once the fill has
        caught up, so an idle run costs no timer."""
        if not getattr(self, "_auto_run_active", False):
            self._stop_auto_progress_ease()
            return
        if getattr(self, "_auto_progress_dirty", False):
            self._auto_progress_dirty = False
            self._refresh_auto_progress_readout()
            self._render_auto_wait_label()
        shown = getattr(self, "_auto_progress_shown", 0)
        target = getattr(self, "_auto_progress_target", 0)
        if shown >= target:
            self._stop_auto_progress_ease()
            return
        # The floor guarantees the last permille always lands, so the fill can
        # never stall a hair short of the count it is showing.
        step = max(1, int((target - shown) * _PROGRESS_EASE_FRACTION))
        self._auto_progress_shown = min(target, shown + step)
        self.auto_tile_progress.setValue(self._auto_progress_shown)

    def set_auto_queue_state(self, position: int, depth: int, eta_s: int) -> None:
        """Honest launch-spike feedback on the progress bar's label. The server
        answers a saturated moment with a real place in its fair queue; showing
        that place (and watching it move) is what keeps a user from reading the
        wait as a hang. position >= 1 = known place in line; -1 = busy but no
        position known (older server / platform rejection / cold start);
        (0, 0, 0) = flowing again, restore the tile count. The bar itself is
        never animated on a timer: only real state changes repaint it."""
        if not self.auto_progress_card.isVisible():
            return
        self._auto_queue_position = position
        self._auto_queue_eta = eta_s if eta_s and eta_s > 0 else 0
        if position == 0 and depth == 0:
            # Flowing again: restore the live tile count (which re-arms the
            # warming animation if we are still at zero tiles).
            current, total = getattr(self, "_auto_progress_pair", (0, 0))
            self.set_auto_tile_progress(current, total)
            return
        # A busy/queued answer means we are still pre-first-tile: keep the bar
        # animated and let the shared renderer pick the right copy (real place
        # in line vs generic "waking up").
        self._ensure_auto_warming_anim()
        self._render_auto_wait_label()

    # ------------------------------------------------------------------
    # Pre-first-tile "waking up" animation (single timer-driven renderer)
    # ------------------------------------------------------------------

    def _ensure_auto_warming_anim(self) -> None:
        """Start (or keep) the pre-first-tile feedback: an indeterminate
        (Qt-animated) bar plus a 1s timer that evolves the label. Idempotent -
        safe to call from every progress/queue update."""
        if self._auto_warming_since is None:
            self._auto_warming_since = time.monotonic()
        # An indeterminate range is Qt's built-in animated busy bar: it always
        # moves, so the wait can never read as frozen. (This deliberately
        # overrides the old "never animate on a timer" rule, per the request
        # for constant motion during cold starts.)
        self.auto_tile_progress.setRange(0, 0)
        if self._auto_warmup_timer is None:
            from qgis.PyQt.QtCore import QTimer
            self._auto_warmup_timer = QTimer(self)
            self._auto_warmup_timer.setInterval(1000)
            self._auto_warmup_timer.timeout.connect(self._on_auto_warming_tick)
        if not self._auto_warmup_timer.isActive():
            self._auto_warmup_timer.start()

    def _stop_auto_warming_anim(self) -> None:
        """End the pre-first-tile animation once tiles flow or the run ends."""
        self._auto_warming_since = None
        if self._auto_warmup_timer is not None and self._auto_warmup_timer.isActive():
            self._auto_warmup_timer.stop()

    def _on_auto_warming_tick(self) -> None:
        """1s heartbeat while no tile has landed: re-assert the animated bar and
        recount the label so the wait is visibly progressing."""
        if not getattr(self, "_auto_run_active", False):
            self._stop_auto_warming_anim()
            return
        current, _total = getattr(self, "_auto_progress_pair", (0, 0))
        if current > 0:
            self._stop_auto_warming_anim()
            return
        if self.auto_tile_progress.maximum() != 0:
            self.auto_tile_progress.setRange(0, 0)
        self._render_auto_wait_label()

    def set_auto_wait_phase(self, phase: str) -> None:
        """Which work the pre-first-tile wait belongs to, from the worker:
        "imagery" while the basemap is still being fetched, "detecting" from the
        first tile on. Only the wording changes; the bar is untouched."""
        if getattr(self, "_auto_wait_phase", "") == phase:
            return
        self._auto_wait_phase = phase
        self._render_auto_wait_label()

    def set_auto_link_slow(self, slow: bool, local: bool = False) -> None:
        """Whether the run has gone quiet long enough to say so on the card.

        A mid-run silence and a hang look the same on a progress bar, and the
        run that is merely slow is the common one. Saying it keeps the user
        waiting instead of cancelling work they have already paid for.
        ``local`` says the silence is this machine's own work (answers being
        turned into shapes), so the line must not blame the connection."""
        if (bool(slow) == getattr(self, "_auto_link_slow", False)
                and bool(local) == getattr(self, "_auto_link_local", False)):
            return
        self._auto_link_slow = bool(slow)
        self._auto_link_local = bool(local)
        self._render_auto_wait_label()

    def _render_auto_wait_label(self) -> None:
        """Single source of truth for the Row-3 status line. Priority: the
        hand-over to the review, then the cancel note, then the post-last-tile
        note, then a mid-run silence, then a real place in the server queue,
        then the pre-first-tile copy. Hidden while tiles flow normally,
        INCLUDING through the free re-scan pass: that pass names and counts
        itself in Row 1, so a standing sentence under it only repeated what the
        row already said."""
        current, total = getattr(self, "_auto_progress_pair", (0, 0))
        if getattr(self, "_auto_finalizing", False):
            # The run is over and the results are being turned into the review.
            # Same words as the last stretch of the run, because it is the same
            # work: the screen must not change until the review takes it.
            text = tr("Almost done - building the shapes...")
        elif self._auto_cancelling:
            text = tr("Stopping - keeping everything already found...")
        elif total and current >= total:
            # Every answer is in, the run has not ended: the shapes are still
            # being folded on this thread. Without a line here the card is a full
            # bar and silence, which reads as a hang, and the faster the answers
            # arrive the longer this stretch lasts.
            text = tr("Almost done - building the shapes...")
        elif current > 0:
            if not getattr(self, "_auto_link_slow", False):
                # Tiles are flowing: the line under the bar is the time left,
                # measured on this run's own pace, and nothing until it is
                # known.
                text = self._auto_time_left_text()
                if not text:
                    self.auto_progress_label.setVisible(False)
                    return
                self.auto_progress_label.setText(text)
                self.auto_progress_label.setVisible(True)
                return
            # Tiles landed and then stopped. The bar alone reads as a hang, and
            # the work already billed is lost if that reading makes the user
            # cancel, so name the cause and say the run is still on. When the
            # cause is this machine turning answers into shapes, say that:
            # "connection" would send the user to check a link that is fine.
            if getattr(self, "_auto_link_local", False):
                text = tr("Building the shapes on this computer - still "
                          "working, everything already found is kept...")
            else:
                text = tr("Connection is slow - still working, everything already "
                          "found is kept...")
        else:
            pos = getattr(self, "_auto_queue_position", 0)
            eta_s = getattr(self, "_auto_queue_eta", 0)
            if pos == 1:
                text = tr("You're next · starting now...")
            elif pos > 1:
                if 0 < eta_s < 10:
                    text = tr("Spot reserved · starting in a few seconds...")
                else:
                    eta = self._friendly_eta(eta_s)
                    text = (tr("Spot reserved · starting in ~{eta}").format(eta=eta)
                            if eta else tr("Spot reserved · starting soon..."))
            else:
                text = self._warming_message()
        self.auto_progress_label.setText(text)
        self.auto_progress_label.setVisible(True)

    def _warming_message(self) -> str:
        """Evolving pre-first-tile copy with a live elapsed count, so the wait
        is visibly moving even before the first tile answers.

        A run opens by downloading the imagery it is about to read, and on a
        slow link that is most of the wait. Naming the AI through it points at
        the wrong component: the user checks a service that is not busy yet
        instead of their connection."""
        since = self._auto_warming_since
        elapsed = int(time.monotonic() - since) if since is not None else 0
        if getattr(self, "_auto_wait_phase", "") == "imagery":
            if elapsed < 6:
                return tr("Loading the imagery...")
            if elapsed < 22:
                return tr("Loading the imagery... {n}s").format(n=elapsed)
            return tr("The imagery is loading slowly... {n}s").format(n=elapsed)
        if elapsed < 6:
            return tr("Sending to the AI...")
        if elapsed < 22:
            return tr("Waking up the AI... {n}s").format(n=elapsed)
        return tr("The AI is starting up, almost there... {n}s").format(n=elapsed)

    @staticmethod
    def _friendly_eta(eta_s: int) -> str:
        """Rounded, human wait estimate ('' when unknown or tiny). Never
        false-precise: seconds snap to 5s steps, past a minute whole minutes."""
        if eta_s is None or eta_s < 10:
            return ""
        if eta_s < 60:
            return tr("{s} seconds").format(s=int(round(eta_s / 5.0) * 5))
        return tr("{m} min").format(m=int((eta_s + 59) // 60))

    def set_auto_status(
        self, kind: str, message: str = "",
        report_payload: tuple | None = None,
    ) -> None:
        """Single surface for run feedback. kind: 'idle', 'progress', 'info',
        'error'. Exactly one of progress bar / status banner is visible at a
        time; 'idle' hides both and clears any stale text.

        ``report_payload`` is ``(error_title, error_message, error_code)`` for a
        terminal 'error' status: the banner then renders as RichText with a
        persistent "Report this problem" link that re-opens the copy-logs/email
        dialog seeded with that payload. Every non-error (or payload-less) call
        resets the banner to PlainText and drops the stash, so a later status
        can neither misrender markup nor fire a stale report link.
        """
        # Between the last tile and the review the run card owns the screen
        # (see set_auto_finalizing): clearing it here would blank the dock for
        # the length of the hand-over and flash the setup step back. A status
        # with something to SAY still replaces the card, as always.
        if kind == "idle" and getattr(self, "_auto_finalizing", False):
            return
        # Any new status replaces the context the zero-result chips belonged
        # to; the plugin re-shows them explicitly after the zero status.
        self.hide_auto_zero_assist()
        # Reset the actionable-error state on EVERY call up front (see docstring):
        # only the error+payload branch below re-arms it.
        self._auto_status_report_payload = None
        self.auto_status_banner.setTextFormat(Qt.TextFormat.PlainText)
        if kind == "progress":
            self.auto_status_banner.setVisible(False)
            return  # set_auto_tile_progress drives the bar itself
        self._set_auto_progress_visible(False)
        if kind == "idle" or not message:
            self.auto_status_banner.setVisible(False)
            self.auto_status_banner.setText("")
            return
        if kind == "error":
            self.auto_status_banner.setStyleSheet(_msg_label_qss("error"))
        else:
            self.auto_status_banner.setStyleSheet(_msg_label_qss("info"))
        if kind == "error" and report_payload is not None:
            self._auto_status_report_payload = tuple(report_payload)
            self.auto_status_banner.setTextFormat(Qt.TextFormat.RichText)
            self.auto_status_banner.setText(
                _error_banner_html(message, tr("Report this problem")))
        else:
            self.auto_status_banner.setText(
                _msg_text("error" if kind == "error" else "info", message))
        self.auto_status_banner.setVisible(True)

    def _on_auto_status_link_activated(self, href: str) -> None:
        """A link clicked inside the run status banner. Only the report sentinel
        is handled: it re-opens the copy-logs/email dialog seeded with the
        failed run's stashed (title, message, code). The failure was already
        tracked when the banner was raised, so this never re-tracks."""
        if href != _REPORT_HREF:
            return
        payload = getattr(self, "_auto_status_report_payload", None)
        if not payload:
            return
        from ..error_report_dialog import show_error_report
        show_error_report(self, *payload, track=False)

    def show_auto_zero_assist(self, object_word: str,
                              has_examples: bool = False) -> None:
        """Show the zero-result rescue under the status banner, exemplar first.

        Called by the plugin right after the zero-detection status is posted
        (never on the network-failure variant, where the levers do not apply).
        The example call is the hero (the proven rescue for a zero-result);
        the synonym chip only shows when the server steer table knows a
        stronger word for this prompt, so the suggestion stays server-tunable
        with no plugin release. Labels must stay short: QPushButton text does
        not wrap. ``has_examples`` switches the label to "another" when the
        run already carried drawn examples."""
        obj = (object_word or "").strip()
        if has_examples:
            label = tr("Add another example - more references detect more")
        elif obj:
            # Name the object in the call to action: "Draw one 'dam'" reads as a
            # concrete next step, not a generic instruction. Kept short (the
            # button text does not wrap) and &-escaped (Qt eats a lone & as a
            # mnemonic marker).
            short = obj if len(obj) <= 18 else obj[:17] + "…"
            short = short.replace("&", "&&")
            label = tr("Draw one '{object}' - the AI finds the rest").format(
                object=short)
        else:
            label = tr("Draw one example - the AI finds the rest")
        self.auto_zero_example_chip.setText("✎  " + label)
        self.auto_zero_example_chip.setToolTip(tr(
            "Outline ONE example of the object on the map, then run again. "
            "Runs with a drawn example return far fewer empty results."))
        suggestion = ""
        if obj:
            try:
                ok, reason, extra = validate_prompt(obj)
                if ok and reason == "steer" and extra:
                    suggestion = str(extra)
            except Exception:  # nosec B110 -- the chip is best-effort rescue UI
                suggestion = ""
        self._auto_zero_synonym = suggestion
        if suggestion:
            # &-escaped like the example chip above: Qt eats a lone & in button
            # text as a mnemonic marker. The stored word keeps its single &.
            shown = suggestion.replace("&", "&&")
            self.auto_zero_synonym_chip.setText(
                "→  " + tr('Try "{word}" instead').format(word=shown))
        self.auto_zero_synonym_chip.setVisible(bool(suggestion))
        self.auto_zero_assist_row.setVisible(True)
        # The rescue sits under the Detect row, which is below the fold on a
        # small dock: bring it into view once the layout has settled, or the
        # whole state reads as "nothing happened".
        try:
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(0, self._scroll_auto_zero_assist_into_view)
        except (RuntimeError, AttributeError):
            pass

    def _scroll_auto_zero_assist_into_view(self) -> None:
        try:
            if self.auto_zero_assist_row.isVisible():
                self._dock_scroll_area.ensureWidgetVisible(
                    self.auto_zero_assist_row, 0, 24)
        except (RuntimeError, AttributeError):
            pass

    def hide_auto_zero_assist(self) -> None:
        try:
            self.auto_zero_assist_row.setVisible(False)
        except (RuntimeError, AttributeError):
            pass

    def show_auto_rerun_guard(self) -> bool:
        """Show the identical-re-run tip under the credit estimate. Advisory
        only: Detect still runs. The plugin decides when it applies (same
        prompt, detail and example count as the last run).

        Returns whether it is on screen: a user who closed the tip with its x
        never sees it again until guidance is restored, and the caller uses the
        answer to keep the "hint shown" telemetry honest."""
        from .guidance import HINT_RERUN_SAME_SETUP, is_hint_dismissed

        try:
            self._auto_rerun_guard_applies = True
            if is_hint_dismissed(HINT_RERUN_SAME_SETUP):
                return False
            self.auto_rerun_guard_hint.setVisible(True)
            return self.auto_rerun_guard_hint.isVisible()
        except (RuntimeError, AttributeError):
            return False

    def hide_auto_rerun_guard(self) -> None:
        try:
            self._auto_rerun_guard_applies = False
            self.auto_rerun_guard_hint.setVisible(False)
        except (RuntimeError, AttributeError):
            pass

    def _should_show_rerun_guard(self) -> bool:
        """Gate for the guidance reset: the tip comes back only while the next
        Detect would still repeat the last run."""
        return bool(getattr(self, "_auto_rerun_guard_applies", False))
