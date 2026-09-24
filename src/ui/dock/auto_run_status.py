






from __future__ import annotations

import time

from qgis.PyQt.QtCore import Qt

from ...core.i18n import tr
from ...core.server_dials import dial_in_range
from .prompt_guard import validate_prompt
from .styles import (
    _REPORT_HREF,
    FIELD,
    _error_banner_html,
    _msg_label_qss,
    category_ink,
    msg_rich,
)








_PROGRESS_HUE_START = "sky"
_PROGRESS_HUE_DONE = "green"
_PROGRESS_BLEND_FROM = 0.7


def _blend_hex(start_hex: str, end_hex: str, t: float) -> str:

    t = max(0.0, min(1.0, t))
    start = start_hex.lstrip("#")
    end = end_hex.lstrip("#")
    out = []
    for i in (0, 2, 4):
        a = int(start[i:i + 2], 16)
        b = int(end[i:i + 2], 16)
        out.append(f"{round(a + (b - a) * t):02x}")
    return "#" + "".join(out)




RUN_BAR_PX = 6


def _auto_progress_bar_qss(ratio: float | None) -> str:







    if ratio is None:
        colour = category_ink(_PROGRESS_HUE_START)
    else:
        blend = max(0.0, (max(0.0, min(1.0, ratio)) - _PROGRESS_BLEND_FROM)
                    / (1.0 - _PROGRESS_BLEND_FROM))
        colour = _blend_hex(category_ink(_PROGRESS_HUE_START),
                            category_ink(_PROGRESS_HUE_DONE), blend)
    return (
        f"QProgressBar {{ background: {FIELD}; border: none;"
        f" border-radius: {RUN_BAR_PX // 2}px; max-height: {RUN_BAR_PX}px;"
        f" min-height: {RUN_BAR_PX}px; }}"
        f"QProgressBar::chunk {{ background: {colour};"
        f" border-radius: {RUN_BAR_PX // 2}px; }}"
    )







_PROGRESS_SCALE = 1000


_PROGRESS_EASE_INTERVAL_MS = 33


_PROGRESS_EASE_FRACTION = 0.28


_WARMUP_TICK_MS = 1000


class DockAutoRunStatusMixin:



    def set_auto_run_found_count(self, obj: str, count: int) -> None:






        self._auto_found_count = max(0, count)
        self._auto_run_object_word = (obj or "").strip()
        if self.auto_progress_card.isVisible():
            self._refresh_auto_progress_readout()

    def _auto_found_so_far_text(self, found: int) -> str:



        return tr("{n} found so far").format(n=found)

    def _elide_row1_text(self, text: str) -> str:




        try:
            from qgis.PyQt.QtGui import QFontMetrics

            chrome = (self.auto_progress_dots.width()
                      + self.auto_progress_clock.width()
                      + self.auto_progress_pct_label.width())
            free = self.auto_progress_card.width() - chrome - (24 + 3 * 8 + 6)
            if free < 40:
                return text
            metrics = QFontMetrics(self.auto_progress_count_label.font())
            return metrics.elidedText(text, Qt.TextElideMode.ElideRight, free)
        except (RuntimeError, AttributeError, TypeError):
            return text

    def _set_auto_progress_visible(self, visible: bool) -> None:


        if not visible:


            self._stop_auto_warming_anim()
            self._stop_auto_progress_ease()
        self.auto_progress_card.setVisible(visible)

        dots = getattr(self, "auto_progress_dots", None)
        if dots is not None:
            try:
                if visible:
                    dots.start()
                else:
                    dots.stop()
            except RuntimeError:
                pass  # nosec B110

    def _paint_auto_finalize_card(self) -> None:












        self.auto_status_banner.setVisible(False)
        self._set_auto_progress_visible(True)
        self._stop_auto_warming_anim()
        self._stop_auto_progress_ease()
        self._auto_progress_target = _PROGRESS_SCALE
        self._auto_progress_shown = _PROGRESS_SCALE
        self.auto_tile_progress.setRange(0, 0)
        self.auto_tile_progress.setStyleSheet(_auto_progress_bar_qss(None))
        self._auto_progress_dirty = False
        self._refresh_auto_progress_readout()
        self._render_auto_wait_label()

    def set_auto_billed_tile_total(self, total: int) -> None:



        self._auto_billed_tile_total = max(0, int(total))

        self._auto_progress_phase = "grid"


        self._auto_run_pace = None

        self._auto_assemble_latched = False
        self._auto_assemble_tiles = (0, 0)

    def _auto_progress_phase_pair(self, current: int, total: int) -> tuple:













        current = max(0, current)
        if getattr(self, "_auto_finalizing", False):





            return "finalize", current, total
        if getattr(self, "_auto_assemble_latched", False):





            folded, given = getattr(self, "_auto_assemble_tiles", (0, 0))
            if given > 0:
                return "assemble", folded, given



        billed = getattr(self, "_auto_billed_tile_total", 0)
        if not billed:

            return "grid", current, total
        refine_total = max(0, total - billed)
        refine_done = max(0, current - billed)
        if refine_total and current >= billed:
            return "refine", refine_done, refine_total
        return "grid", min(current, billed), billed

    def set_auto_assemble_tiles(self, done: int, given: int) -> None:













        pair = (max(0, int(done)), max(0, int(given)))
        if pair == getattr(self, "_auto_assemble_tiles", (0, 0)):
            return
        self._auto_assemble_tiles = pair
        if not getattr(self, "_auto_assemble_latched", False):
            return
        current, total = getattr(self, "_auto_progress_pair", (0, 0))
        self.set_auto_tile_progress(current, total)

    def note_auto_tiles_all_answered(self) -> None:



        self._auto_assemble_latched = True

    def set_auto_finalize_tiles(self, done: int, total: int) -> None:



        self._auto_finalize_tiles = (max(0, int(done)), max(0, int(total)))
        if getattr(self, "_auto_finalizing", False):
            self._refresh_auto_progress_readout()

    def set_auto_finalize_phase(self, text: str) -> None:










        text = str(text or "")
        if text == getattr(self, "_auto_finalize_phase_text", ""):
            return
        self._auto_finalize_phase_text = text
        if getattr(self, "_auto_finalizing", False):
            self._refresh_auto_progress_readout()

    def _refresh_auto_progress_readout(self) -> None:





        current, total = getattr(self, "_auto_progress_pair", (0, 0))
        found = getattr(self, "_auto_found_count", 0)
        phase, current, total = self._auto_progress_phase_pair(current, total)


        tooltip = ""
        if phase == "finalize":




            done, of = getattr(self, "_auto_finalize_tiles", (0, 0))



            count_txt = (getattr(self, "_auto_finalize_phase_text", "")
                         or tr("Building the shapes"))
            if of > 0 and done < of:
                count_txt += " · " + tr("{current} of {total} tiles").format(
                    current=done, total=of)
        elif phase == "assemble":



            count_txt = tr("Building the shapes")
            if total > 0 and current < total:
                count_txt += " · " + tr("{current} of {total} tiles").format(
                    current=current, total=total)
        elif phase == "refine":




            count_txt = tr("Dense area · no extra cost")
            tooltip = tr("{current} of {total} tiles").format(
                current=current, total=total)
        else:




            coverage_txt = self._auto_surface_done_text(current, total)
            tile_tooltip = tr("{current} of {total} tiles").format(
                current=current, total=total)
            if found > 0:
                count_txt = self._auto_found_so_far_text(found)
                tooltip = f"{coverage_txt} · {tile_tooltip}"
            else:
                count_txt = coverage_txt
                tooltip = tile_tooltip
        if found > 0 and phase != "grid":


            count_txt = self._auto_found_so_far_text(found) + " \u00b7 " + count_txt



        count_txt = self._elide_row1_text(count_txt)
        self.auto_progress_count_label.setText(count_txt)
        self.auto_progress_count_label.setToolTip(tooltip)
        if phase == "finalize":


            self.auto_progress_pct_label.setVisible(False)
            return
        if (getattr(self, "_auto_warming_since", None) is not None
                and getattr(self, "_auto_progress_pair", (0, 0))[0] <= 0):


            self.auto_progress_pct_label.setVisible(False)
            return
        self.auto_progress_pct_label.setVisible(True)
        pct = getattr(self, "_auto_progress_target", 0) // (_PROGRESS_SCALE // 100)
        self.auto_progress_pct_label.setText(f"{max(0, min(100, pct))}%")

    def _auto_surface_done_text(self, done: int, of: int) -> str:


        km2 = getattr(self, "_auto_zone_km2", None)
        if km2 is None or km2 <= 0 or not of:
            pct = int(100 * min(1.0, max(0.0, done / of))) if of else 0
            return tr("{pct}% done").format(pct=pct)
        from .ui_refresh import format_km2_surface
        from .ui_refresh_credits import _grouped_decimal
        done_km2 = km2 * min(1.0, max(0.0, done / of))



        decimals = (4 if km2 < 0.005 else 2 if km2 < 1 else 1 if km2 < 10 else 0)
        return tr("{done} of {total} km²").format(
            done=_grouped_decimal(round(done_km2, decimals), decimals),
            total=format_km2_surface(km2))

    def _note_auto_pace(self, phase: str, done: int, of: int) -> None:

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

        pace = getattr(self, "_auto_run_pace", None)
        if pace is None:
            return ""
        from ...core.run_eta import friendly_time_left
        left = pace.seconds_left()
        return friendly_time_left(left) if left is not None else ""

    def set_auto_tile_progress(self, current: int, total: int) -> None:








        if not (self._auto_run_active or getattr(self, "_auto_finalizing", False)):
            return
        self.auto_status_banner.setVisible(False)
        self.hide_auto_zero_assist()

        self._auto_progress_pair = (current, total)
        self._set_auto_progress_visible(True)
        phase, done, of = self._auto_progress_phase_pair(current, total)
        self._note_auto_pace(phase, done, of)
        if phase == "finalize":





            self._auto_progress_phase = phase
            self._refresh_auto_progress_readout()
            self._render_auto_wait_label()
            return
        if phase != getattr(self, "_auto_progress_phase", "grid"):



            self._auto_progress_phase = phase
            self._stop_auto_progress_ease()
            self._auto_progress_target = 0
            self._auto_progress_shown = 0
            self.auto_tile_progress.setValue(0)
        ratio = (done / of) if of and done > 0 else 0.0
        self._auto_progress_ratio = ratio





        target = (_PROGRESS_SCALE if of and done >= of
                  else int(round(min(1.0, max(0.0, ratio)) * _PROGRESS_SCALE)))
        self._auto_progress_target = max(
            getattr(self, "_auto_progress_target", 0), target)
        if done <= 0 and phase == "grid":





            self._stop_auto_progress_ease()
            self._auto_progress_shown = 0
            self._ensure_auto_warming_anim()
            self._refresh_auto_progress_readout()
            self._render_auto_wait_label()
            return


        self._stop_auto_warming_anim()
        if self.auto_tile_progress.maximum() != _PROGRESS_SCALE:


            self.auto_tile_progress.setRange(0, _PROGRESS_SCALE)
            shown = getattr(self, "_auto_progress_shown", 0)
            self.auto_tile_progress.setValue(shown)
            self.auto_tile_progress.setStyleSheet(
                _auto_progress_bar_qss(shown / _PROGRESS_SCALE))
        if of and done >= of:


            self._stop_auto_progress_ease()
            self._auto_progress_shown = self._auto_progress_target
            self.auto_tile_progress.setValue(self._auto_progress_shown)
            self.auto_tile_progress.setStyleSheet(
                _auto_progress_bar_qss(self._auto_progress_shown / _PROGRESS_SCALE))
            self._refresh_auto_progress_readout()
            self._render_auto_wait_label()
            return




        self._auto_progress_dirty = True
        self._ensure_auto_progress_ease()





    def _ensure_auto_progress_ease(self) -> None:


        if getattr(self, "_auto_progress_ease_timer", None) is None:
            from qgis.PyQt.QtCore import QTimer
            timer = QTimer(self)
            timer.setInterval(dial_in_range(
                "tuning.auto.progress_ease_interval_ms",
                _PROGRESS_EASE_INTERVAL_MS, 16, 200))
            timer.timeout.connect(self._on_auto_progress_ease_tick)
            self._auto_progress_ease_timer = timer
        if not self._auto_progress_ease_timer.isActive():
            self._auto_progress_ease_timer.start()

    def _stop_auto_progress_ease(self) -> None:
        timer = getattr(self, "_auto_progress_ease_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()

    def _on_auto_progress_ease_tick(self) -> None:



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


        ease_fraction = dial_in_range(
            "tuning.auto.progress_ease_fraction", _PROGRESS_EASE_FRACTION,
            0.05, 0.9)
        step = max(1, int((target - shown) * ease_fraction))
        self._auto_progress_shown = min(target, shown + step)
        self.auto_tile_progress.setValue(self._auto_progress_shown)
        self.auto_tile_progress.setStyleSheet(
            _auto_progress_bar_qss(self._auto_progress_shown / _PROGRESS_SCALE))

    def set_auto_queue_state(self, position: int, depth: int, eta_s: int) -> None:







        if not self.auto_progress_card.isVisible():
            return
        self._auto_queue_position = position
        self._auto_queue_eta = eta_s if eta_s and eta_s > 0 else 0
        if position == 0 and depth == 0:


            current, total = getattr(self, "_auto_progress_pair", (0, 0))
            self.set_auto_tile_progress(current, total)
            return



        self._ensure_auto_warming_anim()
        self._render_auto_wait_label()





    def _ensure_auto_warming_anim(self) -> None:



        if self._auto_warming_since is None:
            self._auto_warming_since = time.monotonic()




        self.auto_tile_progress.setRange(0, 0)
        self.auto_tile_progress.setStyleSheet(_auto_progress_bar_qss(None))
        if self._auto_warmup_timer is None:
            from qgis.PyQt.QtCore import QTimer
            self._auto_warmup_timer = QTimer(self)
            self._auto_warmup_timer.setInterval(dial_in_range(
                "tuning.auto.warmup_tick_ms", _WARMUP_TICK_MS, 250, 5000))
            self._auto_warmup_timer.timeout.connect(self._on_auto_warming_tick)
        if not self._auto_warmup_timer.isActive():
            self._auto_warmup_timer.start()

    def _stop_auto_warming_anim(self) -> None:

        self._auto_warming_since = None
        if self._auto_warmup_timer is not None and self._auto_warmup_timer.isActive():
            self._auto_warmup_timer.stop()

    def _on_auto_warming_tick(self) -> None:


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



        if getattr(self, "_auto_wait_phase", "") == phase:
            return
        self._auto_wait_phase = phase
        self._render_auto_wait_label()

    def set_auto_link_slow(self, slow: bool, local: bool = False) -> None:







        if (bool(slow) == getattr(self, "_auto_link_slow", False)
                and bool(local) == getattr(self, "_auto_link_local", False)):
            return
        self._auto_link_slow = bool(slow)
        self._auto_link_local = bool(local)
        self._render_auto_wait_label()

    def _render_auto_wait_label(self) -> None:







        current, total = getattr(self, "_auto_progress_pair", (0, 0))
        if self._auto_row1_says_building():



            self.auto_progress_label.setVisible(False)
            return
        if getattr(self, "_auto_finalizing", False):



            text = tr("Almost done - building the shapes...")
        elif self._auto_cancelling:
            text = tr("Stopping - keeping everything already found...")
        elif total and current >= total:




            text = tr("Almost done - building the shapes...")
        elif current > 0:
            if not getattr(self, "_auto_link_slow", False):






                parts = []
                phase, done, of = self._auto_progress_phase_pair(current, total)
                if (phase == "grid" and of
                        and getattr(self, "_auto_found_count", 0) > 0):
                    parts.append(self._auto_surface_done_text(done, of))
                left = self._auto_time_left_text()
                if left:
                    parts.append(left)
                text = " \u00b7 ".join(parts)
                if not text:
                    self.auto_progress_label.setVisible(False)
                    return
                self.auto_progress_label.setText(text)
                self.auto_progress_label.setVisible(True)
                return





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

    def _auto_row1_says_building(self) -> bool:




        if getattr(self, "_auto_cancelling", False):
            return False
        if getattr(self, "_auto_finalizing", False):
            return not getattr(self, "_auto_finalize_phase_text", "")
        if getattr(self, "_auto_assemble_latched", False):
            _folded, given = getattr(self, "_auto_assemble_tiles", (0, 0))
            return given > 0
        return False

    def _warming_message(self) -> str:







        since = self._auto_warming_since
        elapsed = int(time.monotonic() - since) if since is not None else 0



        if getattr(self, "_auto_wait_phase", "") == "imagery":
            if elapsed < 22:
                return tr("Loading the imagery...")
            return tr("The imagery is loading slowly...")
        if elapsed < 6:
            return tr("Sending to the AI...")
        if elapsed < 22:
            return tr("Waking up the AI...")
        return tr("The AI is starting up, almost there...")

    @staticmethod
    def _friendly_eta(eta_s: int) -> str:


        if eta_s is None or eta_s < 10:
            return ""
        seconds = int(round(eta_s / 5.0) * 5)
        if seconds < 60:

            return tr("{s} seconds").format(s=seconds)
        return tr("{m} min").format(m=int((eta_s + 59) // 60))

    def set_auto_status(
        self, kind: str, message: str = "",
        report_payload: tuple | None = None,
    ) -> None:















        if kind == "idle" and getattr(self, "_auto_finalizing", False):
            return


        self.hide_auto_zero_assist()


        self._auto_status_report_payload = None
        self.auto_status_banner.setTextFormat(Qt.TextFormat.PlainText)
        if kind == "progress":
            self.auto_status_banner.setVisible(False)
            return
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
            self.auto_status_banner.setTextFormat(Qt.TextFormat.RichText)
            self.auto_status_banner.setText(
                msg_rich("error" if kind == "error" else "info", message))
        self.auto_status_banner.setVisible(True)

    def _on_auto_status_link_activated(self, href: str) -> None:




        if href != _REPORT_HREF:
            return
        payload = getattr(self, "_auto_status_report_payload", None)
        if not payload:
            return
        from ..error_report_dialog import show_error_report
        show_error_report(self, *payload, track=False)

    def show_auto_zero_assist(self, object_word: str,
                              has_examples: bool = False) -> None:










        obj = (object_word or "").strip()
        if has_examples:
            label = tr("Add another example - more references detect more")
        elif obj:




            short = obj if len(obj) <= 18 else obj[:17] + "…"
            short = short.replace("&", "&&")
            label = tr("Draw one '{object}' - the AI finds the rest").format(
                object=short)
        else:
            label = tr("Draw one example - the AI finds the rest")


        self.auto_zero_example_chip.setText(label)
        self.auto_zero_example_chip.setToolTip(tr(
            "Outline ONE example of the object on the map, then run again. "
            "Runs with a drawn example return far fewer empty results."))
        suggestion = ""
        if obj:
            try:
                ok, reason, extra = validate_prompt(obj)
                if ok and reason == "steer" and extra:
                    suggestion = str(extra)
            except Exception:  # nosec B110
                suggestion = ""
        self._auto_zero_synonym = suggestion
        if suggestion:


            shown = suggestion.replace("&", "&&")
            self.auto_zero_synonym_chip.setText(
                tr('Try "{word}" instead').format(word=shown))
        self.auto_zero_synonym_chip.setVisible(bool(suggestion))
        self.auto_zero_assist_row.setVisible(True)



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


        return bool(getattr(self, "_auto_rerun_guard_applies", False))
