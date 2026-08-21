"""Automatic mode flow: mode switching, credits, warmup, detail/grid math.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

from ...core.i18n import tr
from ...core.qt_compat import DistanceMeters
from ...core.telemetry_errors import slot_guard
from .shared import (
    _WEBMERC_MUPP_Z0,
    _debounce_timer,
    clip_served_hint,
)


def _crs_run_identifier(crs) -> str:
    """A string the rest of the run can rebuild this CRS from.

    The authority code whenever there is one. A custom or WKT-only raster CRS
    has none, and the empty string it returns reads as EPSG:4326 further down,
    which stamps projected metres as degrees and lands the saved polygons
    thousands of km away. So fall back to the CRS object's own WKT, which
    QgsCoordinateReferenceSystem parses back into the same CRS.
    """
    try:
        if crs is None or not crs.isValid():
            return ""
        return crs.authid() or crs.toWkt()
    except (RuntimeError, AttributeError):
        return ""


class AutoFlowMixin:
    """Automatic mode flow: mode switching, credits, warmup, detail/grid math."""

    def _on_mode_changed(self, mode) -> None:
        from ..ai_segmentation_dockwidget import Mode
        # Whatever the switch leads to, the shape under the cursor belongs to
        # the mode being left.
        self._stop_hover_preview("mode changed")
        # Before anything else, including the handoff return below: arriving on
        # Semi-Auto with the cloud engine picked is a click coming, and the ping
        # has to overlap the user rather than the click. Debounced and silent.
        if mode == Mode.INTERACTIVE:
            self._warmup_if_manual_cloud()
        # Refine-in-Manual handoff: preserve both the held _auto_review and the
        # manual session across the Manual<->Auto hop, so the usual destructive
        # reset below MUST be skipped. The flag stays set for the whole round trip
        # (it is cleared only on Finish / Exit / discard).
        if self._refine_handoff_active:
            if mode == Mode.INTERACTIVE:
                self._ensure_interactive_setup()
                self._enter_manual_refine_session()
            else:  # back to Automatic review
                self._restore_auto_review_after_handoff()
            return
        # Capture the pre-reset state for telemetry (the resets below clear it).
        to_mode = "auto" if mode == Mode.AUTOMATIC else "manual"
        has_mask = getattr(self, "current_mask", None) is not None
        has_frozen_sessions = getattr(self, "_frozen_sessions", None)
        has_unfrozen_display = getattr(self, "_unfrozen_display_polygon", None) is not None
        has_saved_polygons = getattr(self, "saved_polygons", None)
        had_unsaved_manual = bool(has_mask or has_frozen_sessions or has_unfrozen_display or has_saved_polygons)
        auto_step = None
        try:
            auto_step = int(self.dock_widget.auto_steps.currentIndex())
        except (RuntimeError, AttributeError):
            pass
        # Switching modes returns each flow to its pre-Start base state, so no
        # mid-flow state (a drawn zone, a pending review, a running detection)
        # leaks across the switch. The Automatic reset is always safe (its
        # results are transient or already exported); the Manual reset is skipped
        # when there is unsaved manual work, to never silently discard it.
        self._reset_auto_flow_to_start(exit_path="mode_switch")
        self._reset_manual_flow_to_start()
        # The footer gauge is on in both modes now, so both refresh it. The
        # fetch is a hidden task and drops out while one is already in flight.
        if self.dock_widget:
            self._refresh_auto_credits()
        if mode == Mode.INTERACTIVE and self.dock_widget:
            self._ensure_interactive_setup()
            # After the resets above, or one of them would park it again on
            # this same switch: a session parked on the way to Automatic is
            # armed here, on the raster and the shapes it still holds.
            self._resume_parked_manual_session()
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_mode_switched(
                to_mode=to_mode,
                had_unsaved_manual=had_unsaved_manual,
                auto_step=auto_step,
            )
        except Exception:
            pass  # nosec B110

    def _reset_auto_flow_to_start(self, exit_path: str = "other") -> None:
        """Return the Automatic flow to its pre-Start base: stop any run, drop
        the zone / review / exemplars / canvas visuals, and land on the Start
        step. Idempotent, so it is safe to call when nothing is active.
        ``exit_path`` names the leave path for the abandonment telemetry."""
        self._stop_auto_detection()  # hard teardown of a live worker; no-op if idle
        self._discard_auto_review(exit_path=exit_path)
        self._restore_maptool_after_zone()
        # Drop any pending/stored run plan (and its attribute filters) so it
        # cannot leak across the switch.
        self._auto_run_plan = None
        self._auto_attribute_filters = []
        self._cancel_task("_auto_run_plan_task")
        self._cancel_task("_auto_token_task")
        self._auto_zone = None
        self._auto_zone_polygon = None
        self._clear_auto_canvas()
        if self.dock_widget:
            try:
                self.dock_widget.reset_auto_to_start()
            except (RuntimeError, AttributeError):
                pass

    def _reset_manual_flow_to_start(self) -> None:
        """Return the Manual flow to its pre-Start base when switching modes.

        Skips the reset when there is unsaved manual work (a live mask, frozen
        sessions, an unfrozen polygon, or saved-but-unexported polygons), so a
        mode toggle never silently discards it; the user stops Manual explicitly
        in that case. Exported layers are always untouched."""
        # getattr defaults: the frozen/unfrozen fields only exist once a manual
        # session has started, and a mode switch can fire before that.
        has_unsaved = getattr(self, "current_mask", None) is not None
        has_unsaved = has_unsaved or getattr(self, "_frozen_sessions", None)
        has_unsaved = has_unsaved or getattr(self, "_unfrozen_display_polygon", None) is not None
        has_unsaved = has_unsaved or getattr(self, "saved_polygons", None)
        if has_unsaved:
            # The work stays, the tool does not: left armed it kept answering
            # clicks from under the Automatic page, on a session the page does
            # not show. Coming back to Semi-Auto arms the same session again.
            self._park_manual_session()
            return
        try:
            canvas = self.iface.mapCanvas()
            tool = getattr(self, "map_tool", None)
            if tool is not None and canvas.mapTool() == tool:
                # The tool announces every deactivate, and the handler puts it
                # straight back one event-loop turn later unless this flag is
                # up. Without it the switch to Automatic re-arms the Semi-Auto
                # tool and flips the page to "session active".
                self._stopping_segmentation = True
                try:
                    canvas.unsetMapTool(tool)
                finally:
                    self._stopping_segmentation = False
        except (RuntimeError, AttributeError):
            pass
        self._reset_session()
        if self.dock_widget:
            try:
                self.dock_widget.reset_session()
            except (RuntimeError, AttributeError):
                pass

    def _refresh_auto_credits(self) -> None:
        """Fetch usage/credits off the main thread and reflect them in the dock.

        A refresh asked for while one is already in flight is REMEMBERED, never
        dropped. The in-flight read left before the charges this call is meant
        to show, so its answer is the balance from before the run, and skipping
        the new read leaves that pre-run figure on screen. The remembered
        refresh fires as soon as the older read settles, and its newer stamp
        wins in _apply_usage_payload.
        """
        from ...core.activation_manager import get_auth_header, is_plugin_activated
        if not self.dock_widget or not is_plugin_activated():
            return
        if self._usage_fetch_task is not None and self._usage_fetch_task.is_active():
            self._usage_refresh_pending = True
            return
        auth = get_auth_header()
        if not auth:
            return
        self._usage_refresh_pending = False
        from qgis.core import QgsApplication

        from ...api.terralab_client import TerraLabClient
        from ...workers.generic_request_task import GenericRequestTask
        client = TerraLabClient()
        # Stamp the read with the moment it leaves, not the moment it lands: a
        # charge that goes out after it can answer first, and the older figure
        # must not overwrite the newer one. See _apply_usage_payload.
        import time as _time
        fetched_at = _time.monotonic()
        task = GenericRequestTask(
            tr("Refreshing your cloud detections"),
            # One bundled read: the usage payload (wallet gauge, old servers)
            # plus the account rows that carry the two-envelope fields.
            lambda: client.get_usage_with_account(auth=auth),
            hidden=True,
        )
        self._usage_fetch_task = task
        # Both callbacks carry the task they belong to: they release the
        # in-flight reference, and an older read must not release a newer one.
        task.succeeded.connect(
            lambda usage, at=fetched_at, t=task: self._on_usage_fetched(usage, at, t))
        # A usage error (network or server) keeps the dock's last figures rather
        # than blanking the ring; matches the old "empty dict on error" contract.
        # A lapsed subscription (failed payment) is called out clearly instead.
        task.failed.connect(
            lambda msg, code, t=task: self._on_usage_failed(msg, code, t))
        QgsApplication.taskManager().addTask(task)

    def _settle_usage_fetch(self, task=None) -> None:
        """Release the in-flight usage reference and fire a refresh that was
        asked for while this read was out.

        Only the read that OWNS the reference may clear it: an older answer
        landing after a newer read went out would otherwise drop the live task,
        leaving it uncancelled at unload and letting the next call fire a
        second one in parallel. ``task`` None keeps the old unconditional
        behaviour for any caller that has no task to name.
        """
        if task is None or self._usage_fetch_task is task:
            self._usage_fetch_task = None
        if getattr(self, "_usage_refresh_pending", False) and self._usage_fetch_task is None:
            self._usage_refresh_pending = False
            self._refresh_auto_credits()

    def _on_usage_failed(self, message: str = "", code: str = "", task=None) -> None:
        """Credits refresh failed. Keep the last-known figures (never blank the
        ring on a transient error). If the server reports the subscription is no
        longer active (a failed renewal payment: the key still authenticates
        locally but every call is rejected), surface a clear one-time notice
        pointing at the account page, so the user is not left with stale credits
        and a mysterious failed run. Throttled to once per session."""
        self._settle_usage_fetch(task)
        if code == "SUBSCRIPTION_INACTIVE" and not self._billing_warning_shown:
            self._billing_warning_shown = True
            try:
                from qgis.core import Qgis
                self.iface.messageBar().pushMessage(
                    "AI Segmentation",
                    tr("There's a problem with your subscription. Open Settings "
                       "to update your payment method or review your plan."),
                    level=Qgis.MessageLevel.Warning,
                )
            except (RuntimeError, AttributeError):
                pass

    def _on_usage_fetched(self, usage: dict, fetched_at: float | None = None,
                          task=None) -> None:
        """Main thread: push fetched usage data into the dock's credit display.

        ``fetched_at`` is the monotonic time this read went out (see
        _refresh_auto_credits); ``task`` is the read itself, so the in-flight
        reference is released by its owner only.

        Accepts both payload shapes: the bundled {"usage", "account"} answer
        of get_usage_with_account, and a bare usage dict from any older
        caller."""
        self._settle_usage_fetch(task)
        account = None
        if isinstance(usage, dict) and isinstance(usage.get("usage"), dict):
            account = usage.get("account")
            usage = usage["usage"]
        self._apply_usage_payload(usage, fetched_at=fetched_at)
        self._apply_account_envelopes(account, fetched_at=fetched_at)

    def _apply_account_envelopes(self, account, fetched_at: float | None = None) -> None:
        """Hand the /account two-envelope fields to the dock, when they came.

        None (the account half failed or was never fetched) keeps the last
        known envelopes: a transient error must not blank a gauge. A row
        WITHOUT the envelope keys (an older server) clears them, so the dock
        falls back to the wallet display it shows today. Stamped like the
        usage payload, so an older read can never overwrite a newer one.
        """
        if account is None or not self.dock_widget:
            return
        import time as _time
        stamp = _time.monotonic() if fetched_at is None else float(fetched_at)
        applied_at = getattr(self, "_envelopes_applied_at", None)
        if applied_at is not None and stamp < applied_at:
            return
        self._envelopes_applied_at = stamp
        try:
            from ...core.quota_envelopes import (
                pick_segmentation_account_row,
                quota_envelopes_from_account_row,
            )
            row = pick_segmentation_account_row(account)
            snapshot = quota_envelopes_from_account_row(row) if row else None
            self.dock_widget.set_auto_envelopes(snapshot)
        except (RuntimeError, AttributeError):
            pass

    def _apply_usage_payload(self, usage: dict, fetched_at: float | None = None) -> None:
        """Reflect a usage payload in the dock, whoever fetched it.

        Split from the task callback because the account window fetches the
        same payload for its own card and hands it here, rather than leaving
        the footer ring on a figure the user can see is out of date. It must
        never touch the in-flight task reference: that belongs to the task,
        and clearing it from here would leave a live request uncancelled at
        unload.

        ``fetched_at`` is the monotonic time the payload was ASKED for. Two
        writers reach this: a read of the balance, and the charge that lowers
        it. A read fired before the charge can land after it and put the spent
        credit back, so a payload older than the last one applied is dropped. A
        caller with nothing to stamp (the charge response, the account window)
        counts as now and always wins.
        """
        if not usage or not isinstance(usage, dict) or not self.dock_widget:
            return
        import time as _time
        stamp = _time.monotonic() if fetched_at is None else float(fetched_at)
        applied_at = getattr(self, "_usage_applied_at", None)
        if applied_at is not None and stamp < applied_at:
            return
        self._usage_applied_at = stamp
        # Cache the last-known usage so run telemetry can report credits_before /
        # is_free_tier without an extra fetch on the critical path. Capture the
        # previous free balance first so a drop can emit free_taste_consumed.
        prev = self._last_usage or {}
        prev_free = prev.get("free_detections_remaining")
        self._last_usage = dict(usage)
        is_free = usage.get("is_free_tier", True)
        # A plan that flips to Pro mid-session is the payment landing. Say it
        # once: the user is coming back from a browser to a panel that was
        # asking them to subscribe. Never on the first payload of a session
        # (empty prev), which carries no change at all.
        if prev and bool(prev.get("is_free_tier", True)) and not is_free:
            self._announce_plan_upgrade()
        free_left = usage.get("free_detections_remaining")
        if is_free and prev_free is not None and free_left is not None and free_left < prev_free:
            try:
                from ...core import telemetry_session_events
                telemetry_session_events.track_free_taste_consumed(
                    remaining=free_left,
                    # Best effort: the refresh follows the run that consumed it.
                    run_id=self._auto_run_id or "",
                )
            except Exception:
                pass  # nosec B110
        if is_free:
            # None travels. It means "the account has not said", which the gauge
            # hides and the gates read as not-exhausted. Turning it into 0 here
            # told the whole panel the user was out of credits: Start went grey
            # and the paywall came up on a funded account, on nothing worse than
            # a response that left the field out.
            credits = free_left
            # Lifetime free-taste allowance. Newer servers echo the total; older
            # deploys omit it, so fall back to the allowance dial for the ring
            # gauge. Read the getter, never a second copy of the number: a
            # literal here would let the gauge disagree with the credit gate.
            from ...core.detection_policy import free_monthly_allowance
            total = int(usage.get("free_detections_total") or free_monthly_allowance())
        else:
            used = usage.get("images_used", 0) or 0
            limit = usage.get("images_limit", 0) or 0
            credits = max(0, limit - used)
            total = limit or None
        reset_date = usage.get("period_end") or ""
        self.dock_widget.set_auto_credits(credits, reset_date,
                                          is_subscriber=not is_free,
                                          total=total)
        # The plan gates the detail slider (free-run cap): re-derive the cap
        # now that the plan is known, so a subscriber's slider uncaps (and a
        # free user's caps) without waiting for the next slider move. Idle
        # zone-set state only; a live run or review owns the cost label.
        if (self._auto_zone is not None and self._auto_worker is None and self._auto_review is None):
            self._update_credit_estimate()

    def _auto_credit_snapshot(self) -> tuple[int | None, bool]:
        """Return (credits_before, is_free_tier) from the last fetched usage.

        Best-effort for telemetry: (None, True) when usage was never fetched."""
        # credit_gate.credit_snapshot owns the usage -> (credits, is_free) rule.
        from ...core.credit_gate import credit_snapshot

        return credit_snapshot(self._last_usage or {})

    def _auto_zone_area_km2(self) -> float:
        """Geodesic area (km2) of the run's billable zone; 0.0 if unknown.

        The clip polygon when a polygon was drawn, the zone rectangle
        otherwise, so the MCP and headless paths report a surface too
        (_auto_billable_zone_geometry owns that choice)."""
        try:
            geom, crs = self._auto_billable_zone_geometry()
            if geom is None or crs is None:
                return 0.0
            from ...core.layer_conventions import make_area_measurer
            from ...core.zone_crs_check import zone_fits_declared_crs
            # A zone whose coordinates cannot be in the CRS the run declared
            # measures metres as degrees. Unknown beats wrong here: the caller
            # treats 0.0 as "no surface" and sends nothing.
            if not zone_fits_declared_crs(geom, crs):
                return 0.0
            da = make_area_measurer(crs)
            return max(0.0, da.measureArea(geom) / 1_000_000.0)
        except Exception:
            return 0.0

    def _auto_duration_ms(self) -> int:
        """Elapsed ms since the detection phase started (0 if not started)."""
        try:
            import time as _time
            if not self._auto_detect_t0:
                return 0
            return int((_time.monotonic() - self._auto_detect_t0) * 1000)
        except Exception:
            return 0

    @staticmethod
    def _classify_auto_error(msg: str) -> str:
        """Map an auto-detection error message to a telemetry error_class enum:
        NETWORK / AUTH / CREDITS_EXHAUSTED / SERVER / CANCELLED / TIMEOUT / UNKNOWN.

        The ladder itself lives in core/error_policy.py, where the server can
        put a rule in front of it: a code the backend starts sending tomorrow
        reaches the right banner on the day it ships instead of at the next
        release. Read there for why the order is what it is.
        """
        from ...core.error_policy import classify_run_error

        return classify_run_error(msg)

    def _open_auto_error_report(
        self, title: str, message: str, error_code: str, *, track: bool,
    ) -> None:
        """Open the copy-logs/email report dialog for a terminal Automatic
        failure, at most once per run: a run can reach more than one terminal
        surface, and the popup must never stack. No-op in headless/MCP runs
        (no human to see it). ``track`` is False when the same failure was
        already counted, so one failure emits exactly one plugin_error."""
        if getattr(self, "_auto_headless_run", False):
            return
        if getattr(self, "_auto_error_dialog_shown", False):
            return
        self._auto_error_dialog_shown = True
        try:
            from ..error_report_dialog import show_error_report
            show_error_report(
                self.iface.mainWindow(), title, message, error_code, track=track)
        except Exception:  # nosec B110 -- the dialog must never mask the error
            pass

    def _track_manual_run_failed(self) -> None:
        """Emit an unsampled segmentation_run(success=False) on a manual predict error."""
        try:
            import time as _time

            from ...core import telemetry_session_events
            start_ts = getattr(self, "_segmentation_start_ts", None)
            duration_ms = int((_time.time() - start_ts) * 1000) if start_ts else None
            telemetry_session_events.track_segmentation_run(success=False, duration_ms=duration_ms)
        except Exception:
            pass  # nosec B110

    # Minimum gap between cold-start pings: one warmup keeps the backend hot
    # well past this, so re-firing on every step/zone/arm event is wasteful.
    _WARMUP_MIN_INTERVAL_S = 300.0

    def _maybe_warmup_auto(self) -> None:
        """Best-effort cloud cold-start ping, off the GUI thread.

        Fired at each rising-intent step of the Automatic flow so the idle
        backend overlaps the user's setup and is warm by the time they hit
        Detect: entering the draw-zone step (Start clicked), the zone drawn,
        the prompt committed, and a detail-slider move. The prompt-commit
        trigger is the universal one (every run passes through it); the others
        buy extra runway when they fire first.

        Also fired when a Semi-Auto session opens on the cloud route. Same
        backend, same host, so one warm instance answers both, and without it
        the session's first click is the one that waits for a machine to start.

        Debounced to at most once per _WARMUP_MIN_INTERVAL_S and gated on cloud
        access (an activated key); every failure is silent. Never blocks the
        main thread.
        """
        import time

        from ...core.activation_manager import get_auth_header, is_plugin_activated

        # Only warmup for users who can actually use the cloud Automatic flow.
        if not is_plugin_activated():
            return
        # A run already keeps the backend hot (the worker hammers it); skip
        # redundant pings while one is in flight.
        if self._auto_worker is not None:
            return
        auth = get_auth_header()
        if not auth:
            return

        now = time.monotonic()
        if now - self._last_warmup_monotonic < self._WARMUP_MIN_INTERVAL_S:
            return
        # A ping already in flight covers the cold start; don't pile on.
        if self._warmup_task is not None and self._warmup_task.is_active():
            return

        self._last_warmup_monotonic = now
        try:
            from qgis.core import QgsApplication

            from ...api.terralab_client import TerraLabClient
            from ...workers.generic_request_task import GenericRequestTask
            client = TerraLabClient()
            self._warm_click_connection(client)
            self._warmup_task = GenericRequestTask(
                tr("Warming up AI Segmentation"),
                lambda: client.warmup(auth=auth),
                hidden=True,
            )
            self._warmup_task.succeeded.connect(self._on_warmup_finished)
            self._warmup_task.failed.connect(lambda *_a: self._on_warmup_finished(False))
            QgsApplication.taskManager().addTask(self._warmup_task)
        except Exception:
            # Warmup is purely cosmetic; never let it disrupt the flow.
            self._warmup_task = None

    def _warm_click_connection(self, client) -> None:
        """Open, from THIS thread, the connection a click will travel on.

        The ping below runs on a task thread and the crop hand-over on a worker
        of its own, and Qt keeps its connections per thread, so neither leaves
        anything behind for the click: the first click of a session opens its
        own, on the thread that draws, while the user waits. This sends a
        throwaway probe from here so that cost is paid before the click.

        Only worth anything when the clicks go straight to the detection host,
        which is the one case where the click and this probe share a host.
        Silent and best effort throughout.
        """
        try:
            if not getattr(client, "detection_direct", False):
                return
            from ...api.click_transport import warm_click_connection

            warm_click_connection(f"{client.detection_base_url}/health")
        except Exception:  # noqa: BLE001 -- a warm must never disrupt the flow  # nosec B110
            pass

    def _on_warmup_finished(self, ok: object = None) -> None:
        """Main thread: release the finished warmup task. A failed ping rewinds
        the debounce stamp, so the next rising-intent event may retry at once
        instead of leaving the whole interval cold."""
        self._warmup_task = None
        if ok is False:
            self._last_warmup_monotonic = 0.0

    @slot_guard(stage="segment", user_message=tr(
        "Something went wrong starting the detection. Please try again."))
    def _on_auto_detect_requested(self) -> None:
        # Commit-time guard rail: both entry points (Detect click and Enter)
        # funnel here, so this is the single gate where an off-rails prompt
        # blocks the run and shows its guidance under the prompt box.
        dock = self.dock_widget
        if dock is not None and not dock.confirm_prompt_for_detect():
            return
        # First committed Detect seals Terms + Privacy consent (the checkbox
        # sits right above the button; Detect stays disabled until it is
        # ticked, so reaching this line means consent was given).
        if dock is not None:
            try:
                dock.seal_tos_consent()
            except (RuntimeError, AttributeError):
                pass
        # Every Detect is a fresh run. The resume/re-detect-the-same-zone flow
        # was removed (it confused more than it helped): a stopped run keeps its
        # partial results in the review instead.
        self._start_auto_detection()

    def _refresh_rerun_guard(self) -> None:
        """Toggle the non-blocking 'identical re-run' note under the credit
        estimate. It appears when the next Detect would repeat the last run
        exactly (same prompt, detail and example count) and clears the moment
        any of the three changes. It never blocks Detect: a user who wants to
        re-run still can (the iterate-until-right path exports the most). Hidden
        during a run/review and while the zero-result rescue is up, so the flow
        keeps one note per state. Fires the hint telemetry once per occurrence."""
        dock = self.dock_widget
        if dock is None:
            return
        try:
            busy = (getattr(dock, "_auto_run_active", False) or getattr(dock, "_auto_review_active", False))
            row = getattr(dock, "auto_zero_assist_row", None)
            zero_up = bool(row is not None and row.isVisible())
            if busy or zero_up:
                dock.hide_auto_rerun_guard()
                return
            last = getattr(self, "_auto_last_run_sig", None)
            if not last:
                dock.hide_auto_rerun_guard()
                return
            cur = (dock.auto_prompt_input.text().strip(),
                   self._get_auto_detail_level(),
                   self._auto_exemplar_store.count())
        except (RuntimeError, AttributeError):
            return
        if cur != last:
            dock.hide_auto_rerun_guard()
            return
        # A tip the user closed stays closed, so nothing was shown to report.
        if not dock.show_auto_rerun_guard():
            return
        if getattr(self, "_rerun_guard_emitted_sig", None) != cur:
            self._rerun_guard_emitted_sig = cur
            try:
                from ...core import telemetry_run_events
                telemetry_run_events.track_auto_prompt_hint_shown(
                    kind="identical_rerun", prompt=cur[0])
            except Exception:
                pass  # nosec B110 -- guard telemetry is best-effort

    def _on_auto_library_clicked(self) -> None:
        """Open the segment library gallery; drop the chosen English token into
        the prompt box. The token is the literal cloud-model prompt (labels are
        localized, tokens are not).

        A server switch can withdraw the library. It fails open, so no
        configuration means it opens exactly as it always has."""
        from ...core.server_dials import feature_enabled

        if not feature_enabled("library"):
            return
        try:
            from ..dialogs.segment_library_dialog import SegmentLibraryDialog
        except Exception as err:  # noqa: BLE001
            from qgis.core import Qgis

            from ...core.logging_utils import log
            log(f"Segment library unavailable: {err}", Qgis.MessageLevel.Warning)
            return
        from ...core.presets import segment_history
        # While a run (or review) is in flight the library opens view-only: the
        # user can still browse, but picking / re-running is disabled (mirrors AI
        # Edit). A picked token can never overwrite the locked in-run prompt.
        run_active = getattr(self.dock_widget, "_auto_run_active", False)
        review_active = getattr(self.dock_widget, "_auto_review_active", False)
        view_only = bool(run_active or review_active)
        dlg = SegmentLibraryDialog(
            self.dock_widget, recent=segment_history.get_recent(), plugin=self,
            view_only=view_only)
        if dlg.exec() and not view_only:
            token = dlg.get_selected_prompt()
            if token:
                self.dock_widget.set_prompt_text(token)

    def _on_zone_draw_requested(self) -> None:
        """Handle the dock's 'Draw zone' button (wired by plan #76)."""
        self._activate_zone_drawing()

    def _on_auto_detail_changed(self, _value: int) -> None:
        """Detail slider moved: the grid, overlay and credit estimate change.

        This fires only on a genuine USER move: programmatic seeds go through
        set_auto_detail_value / set_auto_detail_range, which block the slider's
        signal. So a real move here means the user is overriding the auto pick;
        latch that so the object-aware re-seed stops fighting them for this zone.
        The prompt the override was made for is recorded with it: a manual
        tweak stands for THAT object only, a different prompt re-seeds.
        """
        self._auto_detail_user_locked = True
        self._auto_detail_lock_prompt = self._resolved_auto_object_class().lower()
        # Debounce the cost/grid recompute: each tick runs compute_grid + up to
        # MAX_TILES polygon-clip GEOS calls on the GUI thread, so dragging the
        # slider across many levels would stutter. Coalesce to ~130ms of
        # inactivity (other callers still update immediately).
        self._schedule_credit_estimate()
        # NO wake ping here, deliberately. Moving this slider says nothing about
        # whether a run is coming: a user reads the cost at three levels and
        # closes the dock. Every one of those moves used to wake a machine and
        # hold it, and the measurement upstream is blunt about it: the pings
        # alone held the card for a third of its billed life. The triggers that
        # do carry intent still fire, and the prompt commit is the one every
        # real run passes through, so nothing loses its runway.
        self._schedule_detail_telemetry("user")

    def _schedule_credit_estimate(self) -> None:
        """Debounce _update_credit_estimate for rapid slider moves (~130ms of
        inactivity), so a drag recomputes the grid/cost once it settles instead
        of on every intermediate tick."""
        _debounce_timer(self, "_credit_est_timer", self.dock_widget, 130,
                        self._update_credit_estimate)

    def _schedule_detail_telemetry(self, source: str) -> None:
        """Debounce detail_changed telemetry to ~1s of slider inactivity."""
        self._detail_tel_source = source
        _debounce_timer(self, "_detail_tel_timer", self.dock_widget, 1000,
                        self._emit_detail_telemetry)

    def _emit_detail_telemetry(self) -> None:
        try:
            from ...core import telemetry_run_events
            tiles = getattr(self, "_auto_est_tiles", -1)
            # The band travels with the level: without it a level reads as a
            # free choice when it may have been the only one on offer.
            slider = self.dock_widget.auto_detail_slider
            telemetry_run_events.track_detail_changed(
                detail=self._get_auto_detail_level(),
                tiles=tiles if tiles is not None else -1,
                source=getattr(self, "_detail_tel_source", "user"),
                band_lo=int(slider.minimum()),
                band_hi=int(slider.maximum()),
                object_bound=bool(getattr(
                    self.dock_widget, "_auto_detail_object_bound", False)),
            )
        except Exception:
            pass  # nosec B110

    def _on_auto_layer_combo_changed(self, _layer) -> None:
        """Raster picked (or gone) in the Automatic combo: the cost, the
        slider visibility and the grid preview all depend on the layer."""
        self._update_credit_estimate()

    def _on_auto_step_changed(self, index: int) -> None:
        """Arm the zone drawing tool whenever the zone step opens bare.

        Drawing is the only way to pick the detection area (there is no Full
        image button), so the cross-hair tool activates by itself instead of
        waiting for a button click. Leaving the step disarms the tool unless
        a zone was actually drawn.
        """
        if self._auto_worker is not None:
            return
        if index == 1:
            # Entering the draw-zone step (Start clicked, or re-draw): warm the
            # backend now so the cold start overlaps the user picking a zone.
            self._maybe_warmup_auto()
            # Re-entering the zone step means re-choosing the zone: drop any
            # previous rectangle so the next drag starts clean.
            if self._auto_zone is not None:
                self._on_zone_cleared()
            self._activate_zone_drawing()
        else:
            # Leaving the draw step without (re)drawing: disarm the zone tool and
            # hand the pan tool back rather than leaving a bare cursor.
            self._restore_maptool_after_zone()

    # ---- Pro Auto mode: detection worker (plan #78) -------------------------

    def _get_auto_detail_level(self) -> int:
        """Current detail slider value (grid side n, 1-7). Defaults to 1."""
        if self.dock_widget is None:
            return 1
        try:
            return max(1, int(self.dock_widget.auto_detail_slider.value()))
        except (RuntimeError, AttributeError):
            return 1

    def _online_native_mupp(self, layer) -> float:
        """Finest native resolution (layer-CRS units per pixel) of an online tiled
        source, so the detail slider never renders finer than what the source
        actually serves (no upsampled blurry tiles, no wasted credits). Returns
        0.0 (no clamp) when the native resolution is genuinely unknown, which
        preserves the prior behaviour for non-tiled sources and, by failing OPEN,
        can never over-restrict legitimate detail or make the plugin crash.

        Covered:
          - XYZ basemaps (Google, OSM, ...), WITH or WITHOUT an explicit zmax.
          - WMTS (its tile matrix set defines the finest level's resolution).
          - Any raster provider that reports native resolutions, any CRS.
        Returns 0.0 for non-tiled WMS / WCS / anything with no fixed native grid.
        """
        if layer is None:
            return 0.0
        # Primary path: ask the provider for its native resolutions. QGIS builds
        # this from the tile pyramid for XYZ (derived from zmin/zmax, so it works
        # even when the URI carries no explicit zmax) and from the tile matrix set
        # for WMTS. The values are already in the provider/layer CRS units per
        # pixel - exactly what `_grid_for_detail` compares against - so no CRS math
        # is needed here (a Web Mercator zoom's resolution is latitude-independent
        # in projected units, and any other matrix set reports its own CRS units).
        # The finest level = the smallest value; clamping the render to it only
        # blocks upsampling, never real detail.
        try:
            provider = layer.dataProvider()
            if provider is not None:
                # nativeResolutions(): QGIS >= 3.8 (our floor is 3.22).
                native = provider.nativeResolutions()
                finest = self._finest_native_resolution(native)
                if finest > 0:
                    return finest
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass  # provider does not expose it; fall through to the URI fallback

        # Fallback: derive the deepest-zoom resolution from an XYZ URI's zmax using
        # the Web Mercator slippy-map pyramid. Only valid when the tiling is Web
        # Mercator (EPSG:3857); for any other CRS we cannot assume that formula, so
        # we fail open (0.0) rather than risk an over- or under-clamp.
        try:
            if self._layer_is_web_mercator(layer):
                return self._xyz_zmax_mupp_from_source(layer.source() or "")
        except (RuntimeError, AttributeError, ValueError, TypeError):
            pass
        return 0.0

    @staticmethod
    def _finest_native_resolution(resolutions) -> float:
        """Finest (smallest positive) resolution, in layer-CRS units per pixel,
        from a provider's ``nativeResolutions()`` list. Skips non-numeric or
        non-positive entries individually and returns 0.0 for an empty/invalid
        list (fail open: no clamp)."""
        best = 0.0
        try:
            items = list(resolutions or [])
        except TypeError:
            return 0.0
        for r in items:
            try:
                v = float(r)
            except (TypeError, ValueError):
                continue
            if v > 0 and (best == 0.0 or v < best):
                best = v
        return best

    @staticmethod
    def _layer_is_web_mercator(layer) -> bool:
        """True when the layer's CRS is Web Mercator, so the slippy-map zoom
        formula (``_WEBMERC_MUPP_Z0 / 2**z``) yields correct EPSG:3857 units/px.
        Covers the common historical aliases for spherical Web Mercator."""
        try:
            authid = (layer.crs().authid() or "").upper()
        except (RuntimeError, AttributeError):
            return False
        return authid in (
            "EPSG:3857", "EPSG:900913", "EPSG:102100", "EPSG:102113",
        )

    @staticmethod
    def _xyz_zmax_mupp_from_source(source: str) -> float:
        """Deepest-zoom resolution (EPSG:3857 units per pixel) parsed from an XYZ
        URI's ``zmax``, via the Web Mercator tile pyramid. Returns 0.0 when there
        is no usable zmax: a plain WMS/WMTS URI has none, and an XYZ without an
        explicit zmax is left unclamped on purpose (QGIS's default max zoom is
        provider-internal and version-dependent, so guessing one would over- or
        under-clamp; the runtime nativeResolutions() path already covers it).
        Never raises."""
        try:
            import re  # noqa: PLC0415 - cheap, cached; keeps the top imports lean
            # QGIS raster URIs are '&'-delimited key=value pairs; the tile 'url'
            # value carries its own ampersands URL-encoded (%26), so a top-level
            # '&'/'?' split never cuts through it. Match zmax as a standalone key.
            m = re.search(r"(?:^|[?&])zmax=(\d+)", source or "")
            if not m:
                return 0.0
            zmax = int(m.group(1))
            if zmax <= 0:
                return 0.0
            return _WEBMERC_MUPP_Z0 / float(2 ** zmax)
        except (ValueError, TypeError):
            return 0.0

    def _layer_units_to_run_units(self, layer, zone_in_run) -> tuple[float, float]:
        """How many run-CRS units one layer-CRS unit spans, along x and along y.

        (1.0, 1.0) whenever the run stayed in the layer's own CRS, and on any
        failure. A source reports its native resolution in layer units while
        the grid counts run units, so the two have to be brought to one unit
        before they can be compared at all.
        """
        from qgis.core import QgsCoordinateTransform, QgsProject

        from ...core.layer_conventions import ground_unit_metres

        run_crs = self._run_crs_now(layer)
        try:
            layer_crs = layer.crs()
            if run_crs is None or run_crs == layer_crs:
                return 1.0, 1.0
            centre_run = zone_in_run.center()
            run_mx, run_my = ground_unit_metres(
                run_crs, centre_run.x(), centre_run.y())
            if run_mx <= 0 or run_my <= 0:
                return 1.0, 1.0
            centre_layer = QgsCoordinateTransform(
                run_crs, layer_crs, QgsProject.instance()).transform(centre_run)
            layer_mx, layer_my = ground_unit_metres(
                layer_crs, centre_layer.x(), centre_layer.y())
            # Each axis against its own, so the y factor stays right even if a
            # run CRS is ever allowed whose two axes do not agree.
            return layer_mx / run_mx, layer_my / run_my
        except (RuntimeError, AttributeError, TypeError, ValueError,
                ZeroDivisionError):
            return 1.0, 1.0

    def _grid_for_detail(self, layer, zone_in_layer, detail_n: int):
        """Pixel grid for one detail level. Single source of truth for the
        detail-slider math, shared by `_compute_auto_grid` (the real run),
        `_max_useful_detail` (the slider cap) and the m/px hint.

        The zone's longer side is rendered as exactly ``detail_n`` tiles
        (snap formula: TILE_SIZE + (n-1) * stride pixels). Local rasters AND
        online Web Mercator XYZ basemaps are clamped near their native
        resolution: the render may oversample up to NATIVE_OVERSAMPLE_MAX
        (2x linear) past it, because the cloud model's fixed processing window
        means a finer grid still enlarges small objects in model space, but
        never further (pure interpolation, credits for nothing).

        ``zone_in_layer`` is the zone already reprojected to the run CRS.

        Returns ``(pixel_w, pixel_h, mupp, tile_count)`` where ``mupp`` is the
        rendered ground resolution in run-CRS units per pixel (identical on
        both axes) and ``tile_count`` is the credit cost (``-1`` when the grid
        exceeds MAX_TILES). Returns None when the zone has no extent or the
        layer cannot be read.
        """
        from ...core.tile_manager import (
            NATIVE_OVERSAMPLE_MAX,
            OVERLAP_FRACTION,
            TILE_SIZE,
        )

        longer_side = max(zone_in_layer.width(), zone_in_layer.height())
        if longer_side <= 0:
            return None
        try:
            layer_w = layer.width()
            layer_h = layer.height()
            ext = layer.extent()
        except (RuntimeError, AttributeError):
            return None

        use_online = layer_w <= 0 or layer_h <= 0 or self._needs_canvas_render(layer)
        # Same integer stride as TileManager. compute_grid() emits one tile at
        # offset 0 then one per full stride, so a span of TILE_SIZE + k * stride
        # pixels yields exactly k + 1 tiles per axis.
        stride = int(TILE_SIZE * (1.0 - OVERLAP_FRACTION))
        target_px = TILE_SIZE + (max(1, detail_n) - 1) * stride
        mupp = longer_side / target_px
        # Both clamps allow up to NATIVE_OVERSAMPLE_MAX past the source's
        # native resolution: upsampling adds no pixels, but a finer grid
        # enlarges small objects inside the model's fixed processing window.
        # Past the clamp it is pure interpolation, so the clamp holds there.
        # Both clamps read the source, which reports in layer units, while mupp
        # counts run units. The two are the same unit unless the run moved, so
        # a run that did not move keeps exactly the arithmetic it always had.
        # Where they differ, the coarser of the source's two axes sets the
        # clamp: the render is one resolution on both axes now, and billing
        # tiles to interpolate the coarse axis is what the clamp exists to stop.
        to_run_x, to_run_y = self._layer_units_to_run_units(layer, zone_in_layer)
        if not use_online and ext.width() > 0 and ext.height() > 0:
            native_mupp = max(ext.width() / layer_w * to_run_x,
                              ext.height() / layer_h * to_run_y)
            mupp = max(mupp, native_mupp / NATIVE_OVERSAMPLE_MAX)
        elif use_online:
            # Online XYZ basemap: clamp near its deepest-zoom native resolution
            # so a finer detail level cannot burn unbounded credits on ever
            # blurrier upsamples. No-op for WMS / non-Mercator.
            online_mupp = self._online_native_mupp(layer)
            if online_mupp > 0:
                native_mupp = online_mupp * max(to_run_x, to_run_y)
                mupp = max(mupp, native_mupp / NATIVE_OVERSAMPLE_MAX)

        pixel_w = max(1, int(zone_in_layer.width() / mupp))
        pixel_h = max(1, int(zone_in_layer.height() / mupp))
        # Do NOT snap up to clean tile boundaries: snapping grew the grid past
        # the drawn zone, so the preview and the tiles spilled into whatever lay
        # outside the selection (e.g. desert). Tile the zone exactly instead;
        # compute_grid's last row/column is a partial edge tile that stays fully
        # inside the zone, and the 20% overlap means the adjacent full tile
        # already covers most of that strip.
        tile_count = self._tile_manager.estimate_credits(pixel_w, pixel_h)
        return pixel_w, pixel_h, mupp, tile_count

    def _max_useful_detail(self, layer, zone_in_layer) -> int:
        """Largest detail level (1-MAX_DETAIL_LEVEL) that still changes the grid.

        Two regimes bound the slider, and crossing either makes higher levels
        inert (the cursor moves with no visible effect):
          - the MAX_TILES cap (a finer grid would exceed it), and
          - the native-resolution clamp: on a local raster, past some level
            the render is pinned to native pixels, so a higher level repeats
            the exact same mupp and the same tiles.
        Returns the largest n whose grid fits MAX_TILES and whose mupp is
        still strictly finer than n-1's. Always >= 1.
        """
        from ...core.tile_manager import MAX_DETAIL_LEVEL
        from .shared import max_tiles_per_run_cap

        # The run gate already uses the server-dialed cap; the slider ceiling
        # must read the SAME value or a lowered cap would leave inert levels
        # the Detect gate then refuses.
        max_tiles = max_tiles_per_run_cap()

        # One-entry memo: the debounced slider/cost ticks re-ask for the SAME
        # layer+zone many times, and each walk re-runs up to MAX_DETAIL_LEVEL
        # grid computations (ellipsoidal math included).
        try:
            key = (
                layer.id(), max_tiles,
                zone_in_layer.xMinimum(), zone_in_layer.yMinimum(),
                zone_in_layer.xMaximum(), zone_in_layer.yMaximum(),
            )
        except (RuntimeError, AttributeError):
            key = None
        cached = getattr(self, "_max_detail_cache", None)
        if key is not None and cached is not None and cached[0] == key:
            return cached[1]

        best = 1
        prev_mupp = None
        for n in range(1, MAX_DETAIL_LEVEL + 1):
            sized = self._grid_for_detail(layer, zone_in_layer, n)
            if sized is None:
                break
            _pw, _ph, mupp, tiles = sized
            if tiles == -1 or tiles > max_tiles:
                break  # grids only grow with n; nothing finer will fit
            if prev_mupp is not None and mupp >= prev_mupp:
                break  # clamped to native: this level renders no finer
            best = n
            prev_mupp = mupp
        if key is not None:
            self._max_detail_cache = (key, best)
        return best

    def _free_run_tile_cap(self) -> int | None:
        """Per-run tile (credit) cap for free-tier users; None for subscribers.

        A single free run may only cost a fraction of the lifetime free
        allowance (server policy `seed.free_run_fraction`), so one Detect can
        never drain the trial: the remaining runs are where the product proves
        itself. The cap gates DETECT, never the slider: the user drags through
        the full Pro range and sees an upgrade message past the cap (the locked
        capability stays visible). The auto seeds also respect it, so the
        suggested default never lands in the gated range. Before the usage
        fetch lands the user is treated as free (the safe default); the
        estimate re-runs when the fetch arrives, so a subscriber ungates the
        moment their plan is known.
        """
        from ...core.credit_gate import free_run_tile_cap
        from ...core.detection_policy import free_run_fraction

        usage = self._last_usage or {}
        if not usage.get("is_free_tier", True):
            return None
        # credit_gate.free_run_tile_cap owns the cap arithmetic (the round(),
        # the older-server total fallback, and the >= 1 floor). The subscriber
        # None stays here: it is a tier decision, not cap arithmetic.
        return free_run_tile_cap(usage.get("free_detections_total"), free_run_fraction())

    def _seed_tile_cap_for_plan(self) -> int:
        """The hard seed tile cap, tightened to the free-run cap on free tier,
        so the auto-picked default never proposes a level the plan gates."""
        from ...core.detection_policy import seed_tile_cap

        cap = seed_tile_cap()
        free_cap = self._free_run_tile_cap()
        return cap if free_cap is None else min(cap, free_cap)

    def _default_detail_for_zone(self, layer, zone_in_layer) -> int:
        """Cheapest detail level whose ground resolution reaches the seed target.

        Walks detail levels coarse -> fine and returns the first level reaching
        the seed resolution within the soft tile budget. Past that budget it
        returns the cheapest level still inside the sweet-spot band (the credit
        cost is displayed live and Detect is the user's confirmation), and never
        spends past the hard seed cap. Falls back to the cheapest in-budget
        level when even the slider maximum stays coarser. Always >= 1, so a
        fresh zone never sits at a too-coarse 1x1 grid when a finer level is the
        better default. The object-aware seed refines this per prompt.
        """
        from ...core.detection_policy import (
            soft_tile_budget,
            sweet_spot_max_mupp,
            zone_seed_mupp,
        )

        seed_mupp = zone_seed_mupp()
        tile_cap = self._seed_tile_cap_for_plan()
        soft_budget = soft_tile_budget()
        sweet_max = sweet_spot_max_mupp()
        best_within_budget = 1
        for n in range(1, self._max_useful_detail(layer, zone_in_layer) + 1):
            sized = self._grid_for_detail(layer, zone_in_layer, n)
            if sized is None:
                break
            _pw, _ph, mupp, tiles = sized
            if tiles == -1:
                break
            if tiles > tile_cap:
                break  # the default never spends past the hard seed cap
            ground_mupp = self._mupp_to_meters(layer, zone_in_layer, mupp)
            if tiles > soft_budget:
                if 0 < ground_mupp <= sweet_max:
                    return n  # cheapest ADEQUATE level beyond the soft budget
                continue
            best_within_budget = n
            if 0 < ground_mupp <= seed_mupp:
                return n  # cheapest level reaching the seed target
        return best_within_budget

    def _object_detail_profile(self, object_class: str) -> tuple[float, float]:
        """(typical ground size m, target ground resolution m/px) for the
        object named by ``object_class``.

        The per-object targets are provided by the server configuration;
        without it a generic object at the default seed resolution applies. See
        ``core.detection_policy.object_profile``.
        """
        from ...core.detection_policy import object_profile

        return object_profile(object_class)

    def _auto_detail_for_object(self, layer, zone_in_layer, object_class) -> int:
        """Cheapest detail level whose ground resolution suits the OBJECT.

        The object maps to a target ground resolution (`_object_detail_profile`)
        so distinct objects land on distinctly different levels. Walks detail
        levels coarse -> fine and returns the cheapest reaching the target, past
        the soft tile budget when needed (the credit cost is displayed live and
        Detect is the user's confirmation) but never past the hard seed cap.

        Fallbacks, in order, when no level reaches the target: the cheapest
        level where the object still renders at the object minimum pixel size
        (resolvable, if not ideal), else the finest level within the soft tile
        budget; on any layer read error, the resolution-only zone default.
        Always >= 1.
        """
        try:
            obj_m, target_mupp = self._object_detail_profile(object_class)
            return self._auto_detail_for_target(
                layer, zone_in_layer, obj_m, target_mupp,
                self._detail_window_profile(object_class)[1])
        except (RuntimeError, AttributeError, ValueError):
            return self._default_detail_for_zone(layer, zone_in_layer)

    def _auto_detail_for_target(
        self, layer, zone_in_layer, obj_m: float, target_mupp: float,
        floor_m: float = 0.0,
    ) -> int:
        """Cheapest detail level whose ground resolution reaches ``target_mupp``.

        The shared coarse -> fine walk behind `_auto_detail_for_object` (blob
        tier target) and the async run-plan re-seed (server target), so the two
        paths pick levels identically. ``obj_m`` is the object's typical ground
        size, used only for the resolvable fallback.

        ``floor_m`` is the smallest tile ground side the object is known to
        detect at (``detection_policy.object_tile_floor_m``, 0.0 = none) and it
        OUTRANKS the target. Some objects are read through a fixed wide window,
        so a tile under that width is padding rather than context and the answer
        degrades however sharp the pixels are. The target only says "fine
        enough", which on some zone sizes is first reached one level BELOW the
        floor; taking it there would make the plugin recommend the setting the
        floor exists to forbid, and the slider band, which must contain the
        recommendation, would follow it down.

        When no level inside the seed's tile cap reaches the target, the pick
        is the FINEST of three floors, so a named object never seeds coarser
        than the same zone with no prompt at all: the finest level inside the
        soft tile budget, the cheapest level inside the adequate-quality band
        (the zone default crosses the soft budget for this too), and the
        cheapest level where the object still spans the minimum pixels. Each is
        collected only on levels that clear ``floor_m``, so the fallback cannot
        reintroduce what the walk stopped at. Always >= 1.
        """
        from ...core.detection_policy import (
            object_min_px,
            soft_tile_budget,
            sweet_spot_max_mupp,
        )
        from ...core.tile_manager import TILE_SIZE

        cap = self._max_useful_detail(layer, zone_in_layer)
        min_px = object_min_px()
        tile_cap = self._seed_tile_cap_for_plan()
        soft_budget = soft_tile_budget()
        sweet_max = sweet_spot_max_mupp()
        resolvable_n = 0
        adequate_n = 0
        finest_in_budget = 1
        for n in range(1, cap + 1):
            sized = self._grid_for_detail(layer, zone_in_layer, n)
            if sized is None:
                break
            _pw, _ph, mupp, tiles = sized
            if tiles == -1:
                break
            if tiles > tile_cap:
                break  # tiles grow with n: the seed never spends past the cap
            ground_mupp = self._mupp_to_meters(layer, zone_in_layer, mupp)
            if ground_mupp <= 0:
                continue
            if floor_m > 0 and TILE_SIZE * ground_mupp < floor_m:
                break  # tiles shrink with n: no finer level clears the floor
            if ground_mupp <= target_mupp:
                return n
            if tiles <= soft_budget:
                finest_in_budget = n
            if not adequate_n and ground_mupp <= sweet_max:
                adequate_n = n
            if not resolvable_n and obj_m / ground_mupp >= min_px:
                resolvable_n = n
        return max(1, finest_in_budget, adequate_n, resolvable_n)

    def _current_auto_object_class(self) -> str:
        """The object class currently entered in the Automatic prompt box."""
        if self.dock_widget is None:
            return ""
        try:
            return self.dock_widget.auto_prompt_input.text().strip()
        except (RuntimeError, AttributeError):
            return ""

    def _resolve_object_token(self, raw: str) -> str:
        """English cloud-model token for a possibly-localized prompt.

        The offline lexicon resolves a localized object word to its English
        token synchronously, so the detail seed and every prompt-keyed policy
        lookup key on the SAME token the run will send ("jardin" seeds like
        "garden"). An English or lexicon-missing word passes through unchanged;
        a miss beyond the offline lexicon is resolved asynchronously by the
        server fallback. Cached per prompt so the debounced commit never
        re-queries the catalogue on every keystroke settle.
        """
        raw = (raw or "").strip()
        if not raw:
            return ""
        cache = getattr(self, "_auto_token_cache", None)
        if cache is None:
            cache = {}
            self._auto_token_cache = cache
        key = raw.lower()
        if key in cache:
            return cache[key]
        try:
            from ..dock.prompt_guard import resolve_object_token
            token = resolve_object_token(raw)
        except Exception:  # noqa: BLE001 -- resolution is best-effort
            token = raw
        cache[key] = token
        return token

    def _resolved_auto_object_class(self) -> str:
        """The current prompt resolved to its English token. The box keeps
        showing the user's own words; the seed and policy lookups key on this
        so a localized prompt behaves like its English equivalent."""
        return self._resolve_object_token(self._current_auto_object_class())

    def _seed_auto_detail_value(self, detail: int) -> None:
        """Push a plugin-picked detail level to the slider and remember it.

        Every automatic seed goes through here, so the run-started event can
        tell a level the plugin recommended from one the user picked. What is
        remembered is the value the slider ends up holding, not the value asked
        for: the setter clamps to the slider's own travel.
        """
        self.dock_widget.set_auto_detail_value(detail)
        self._auto_detail_seeded = self._get_auto_detail_level()

    def _apply_default_detail(self, zone_rect) -> None:
        """Seed the detail slider with a good default for a freshly drawn zone.

        A freshly drawn zone is a fresh start, so the manual-override lock is
        cleared: the object-aware re-seed may drive the slider again until the
        user next moves it themselves. When an object class is already entered,
        seed with the object-aware pick; otherwise fall back to the
        resolution-only default.
        """
        # New zone = fresh default: let the auto picker own the slider again.
        self._auto_detail_user_locked = False
        self._auto_detail_lock_prompt = ""
        # The old zone's seed says nothing about this one; it stands again only
        # once a picker below actually lands a value.
        self._auto_detail_seeded = None
        if not self.dock_widget:
            return
        layer = self._get_active_raster_layer()
        if layer is None:
            return
        try:
            zone_in_layer = self._reproject_zone_to_run_crs(zone_rect, layer)
            object_class = self._resolved_auto_object_class()
            if object_class:
                detail = self._auto_detail_for_object(
                    layer, zone_in_layer, object_class)
            else:
                detail = self._default_detail_for_zone(layer, zone_in_layer)
            self._seed_auto_detail_value(detail)
        except (RuntimeError, AttributeError):
            pass

    def _reseed_auto_detail_for_object(self, object_class: str = "") -> None:
        """Prompt settled (debounced): re-seed the detail default from the blob
        NOW (must land instantly), then fire the async server run-plan fetch.

        The synchronous blob seed is the source of truth the moment the prompt
        settles; the run plan, when it lands and still matches this prompt,
        refines it (see _on_auto_run_plan_ready).

        A localized prompt is resolved to its English token first (offline
        lexicon), so the object tiers match ("jardin" seeds like "garden"). The
        box keeps the user's own words; only the lookups key on the token."""
        raw = (object_class or "").strip()
        token = self._resolve_object_token(raw)
        self._reseed_auto_detail_from_blob(token)
        self._fetch_auto_run_plan(token)
        # A localized word the offline lexicon does not cover (a rare language)
        # cannot seed the object tiers yet: ask the server for its English token
        # and re-seed when it lands, if this prompt is still the one shown.
        if raw and token == raw:
            self._fetch_auto_token(raw)
        # A committed prompt is the highest-intent, most universal pre-Detect
        # signal: it is the ONE step every run passes through (the zone-draw and
        # slider triggers miss a user who lands on the prompt step with a zone
        # already set and runs without touching the slider). Warm here so the
        # backend is spinning up while they read the estimate and reach Detect.
        # Self-limited: debounced, no-op mid-run and when already warm.
        self._maybe_warmup_auto()
        # A committed prompt may re-match (or stop matching) the last run's
        # signature, so re-evaluate the identical-re-run note here.
        self._refresh_rerun_guard()

    def _reseed_auto_detail_from_blob(self, object_class: str = "") -> None:
        """Re-pick the detail level when the object class changes (debounced).

        Respects a manual override for the SAME prompt: once the user has
        moved the slider, their value stands for the object it was tuned for.
        A different prompt is a new sizing problem, so it releases the
        override and re-seeds (the user can still re-adjust afterwards).
        Requires a drawn zone and a non-empty object class. Sets the slider
        programmatically (signal free via set_auto_detail_value's
        blockSignals), so it never trips the user-lock itself.
        """
        if not self.dock_widget or self._auto_zone is None:
            return
        # Never fight a run/review or override while one is in flight.
        if self._auto_worker is not None or self._auto_review is not None:
            return
        object_class = (object_class or "").strip()
        if not object_class:
            return
        if self._auto_detail_user_locked:
            locked_for = getattr(self, "_auto_detail_lock_prompt", "")
            if object_class.lower() == locked_for:
                return
            self._auto_detail_user_locked = False
            self._auto_detail_lock_prompt = ""
        layer = self._get_active_raster_layer()
        if layer is None:
            return
        try:
            zone_in_layer = self._reproject_zone_to_run_crs(self._auto_zone, layer)
            detail = self._auto_detail_for_object(layer, zone_in_layer, object_class)
            self._seed_auto_detail_value(detail)
            self._update_credit_estimate()
        except (RuntimeError, AttributeError):
            pass

    # ---- Async server run plan (Phase 2) ----------------------------------

    def _active_run_plan(self, prompt: str) -> dict | None:
        """The stored run plan IF it was fetched for ``prompt`` (case/space
        insensitive), else None. The single gate every apply site goes through,
        so a stale plan for a different prompt never leaks into a run/review."""
        rp = getattr(self, "_auto_run_plan", None)
        if not isinstance(rp, dict):
            return None
        if (rp.get("prompt") or "").strip().lower() != (prompt or "").strip().lower():
            return None
        plan = rp.get("plan")
        return plan if isinstance(plan, dict) else None

    def _fetch_auto_run_plan(self, prompt: str) -> None:
        """Fire-and-forget: fetch the server run plan for the committed prompt.

        Off the GUI thread via a hidden task; never blocks, fails open (the
        blob/generic path stands on any error or timeout). A new commit clears
        the previous plan so a stale one can never apply between commit and
        reply."""
        prompt = (prompt or "").strip()
        # Every commit invalidates the previous plan (it was for the old prompt),
        # and with it any attribute filters that plan carried.
        self._auto_run_plan = None
        self._auto_attribute_filters = []
        if not prompt or not self.dock_widget:
            return
        # Never fetch mid-run/review: the prompt is locked then and a late
        # apply must not fight an in-flight run.
        if self._auto_worker is not None or self._auto_review is not None:
            return
        from ...core.activation_manager import get_auth_header, is_plugin_activated
        if not is_plugin_activated():
            return
        auth = get_auth_header()
        if not auth:
            return
        # Only one in flight; a newer commit supersedes the older request.
        self._cancel_task("_auto_run_plan_task")
        zone_area_m2, native_mupp = self._auto_run_plan_inputs()
        try:
            from qgis.core import QgsApplication

            from ...api.terralab_client import TerraLabClient
            from ...workers.generic_request_task import GenericRequestTask
            client = TerraLabClient()
            task = GenericRequestTask(
                tr("Planning AI Segmentation run"),
                lambda: client.get_seg_run_plan(
                    prompt, zone_area_m2, native_mupp, auth=auth),
                hidden=True,
            )
            task.succeeded.connect(
                lambda plan, p=prompt: self._on_auto_run_plan_ready(p, plan))
            task.failed.connect(lambda *_a: self._on_auto_run_plan_failed())
            self._auto_run_plan_task = task
            QgsApplication.taskManager().addTask(task)
        except Exception:  # noqa: BLE001 -- planning is best-effort
            self._auto_run_plan_task = None  # nosec B110

    def _auto_run_plan_inputs(self) -> tuple[float | None, float | None]:
        """Best-effort (zone_area_m2, native_mupp) for the run-plan request.

        Either may be None (both optional server-side). One cheap geodesic
        measurement each, on the debounced commit; never raises."""
        zone_area_m2: float | None = None
        native_mupp: float | None = None
        try:
            layer = self._get_active_raster_layer()
        except (RuntimeError, AttributeError):
            layer = None
        if layer is None:
            return zone_area_m2, native_mupp
        zone_in_layer = None
        if self._auto_zone is not None:
            try:
                zone_in_layer = self._reproject_zone_to_run_crs(self._auto_zone, layer)
            except (RuntimeError, AttributeError):
                zone_in_layer = None
        if zone_in_layer is not None:
            try:
                from qgis.core import QgsGeometry

                from ...core.layer_conventions import make_area_measurer
                # The rectangle is in the run CRS, so the measurer has to be
                # too. Measuring run coordinates through the layer's CRS reads
                # an easting as a longitude and returns an area of nowhere.
                da = make_area_measurer(self._run_crs_now(layer) or layer.crs())
                area = da.measureArea(QgsGeometry.fromRect(zone_in_layer))
                if area and area > 0:
                    zone_area_m2 = float(area)
            except Exception:  # noqa: BLE001 -- best-effort  # nosec B110
                zone_area_m2 = None
        try:
            layer_w = layer.width()
            layer_h = layer.height()
            ext = layer.extent()
            # A source reports its native resolution in layer units, and
            # _mupp_to_meters measures in run units, so the two are brought
            # together the same way the grid math does it. Both factors are
            # 1.0 for a run that stayed in the layer's own CRS.
            ref = zone_in_layer if zone_in_layer is not None else ext
            to_run_x, to_run_y = self._layer_units_to_run_units(layer, ref)
            if self._needs_canvas_render(layer) or layer_w <= 0 or layer_h <= 0:
                native_units = self._online_native_mupp(layer) * max(to_run_x, to_run_y)
            elif ext.width() > 0 and ext.height() > 0:
                native_units = max(ext.width() / layer_w * to_run_x,
                                   ext.height() / layer_h * to_run_y)
            else:
                native_units = 0.0
            if native_units and native_units > 0:
                meters = self._mupp_to_meters(layer, ref, native_units)
                if meters and meters > 0:
                    native_mupp = float(meters)
        except (RuntimeError, AttributeError, ValueError):
            native_mupp = None
        return zone_area_m2, native_mupp

    def _on_auto_run_plan_ready(self, prompt: str, plan: object) -> None:
        """Main thread: store the plan under its prompt and refine the detail
        seed from it, but only while that prompt is still the committed one."""
        self._auto_run_plan_task = None
        if not isinstance(plan, dict) or plan.get("error"):
            return
        # A late plan response must never act while a run, its review or a
        # correction batch is active: the prompt is not editable then, and a
        # swap or detail reseed would fight the in-flight or reviewed result.
        if self._auto_worker is not None or self._auto_review is not None:
            return
        prompt = (prompt or "").strip()
        if not prompt:
            return
        if prompt.lower() != self._resolved_auto_object_class().strip().lower():
            return  # the user moved on to a different object since the fetch
        self._auto_run_plan = {"prompt": prompt, "plan": plan}
        self._reseed_auto_detail_from_plan(prompt, plan)
        # Optional server prompt_rewrite block (additive, fail-open). When it
        # owns the prompt-info line (a rewrite swapped in, or a decline nudge
        # shown), the generic plan hint below yields to it.
        if self._apply_prompt_rewrite(prompt, plan):
            return
        # Optional per-prompt guidance string. Additive and fail-open: absent or
        # malformed on older servers = exactly today's behavior. Shown verbatim
        # (server-authored English) only when no higher-priority prompt message
        # owns the line (the dock enforces that precedence).
        hint = plan.get("hint")
        if isinstance(hint, str) and self.dock_widget is not None:
            hint = clip_served_hint(hint)
            if hint:
                try:
                    if self.dock_widget.show_auto_prompt_hint(hint):
                        from ...core import telemetry_run_events
                        telemetry_run_events.track_auto_prompt_hint_shown(
                            kind="plan_hint", prompt=prompt)
                except Exception:  # noqa: BLE001 -- guidance is best-effort
                    pass  # nosec B110

    def _apply_prompt_rewrite(self, prompt: str, plan: dict) -> bool:
        """Act on the optional server ``prompt_rewrite`` block. Returns True when
        it owns the prompt-info line (a rewrite was swapped in, or a decline
        nudge shown), so the caller skips the generic plan hint.

        Additive and fail-open: an absent or malformed block leaves exactly
        today's behavior. The caller already checked this prompt is still the
        committed one (the box unedited since the fetch), the same guard the
        seed/hint path uses, so a rewrite here is safe. The rewrite is applied
        verbatim (the server preserves any attributes) through the dock's
        visible swap-and-tell channel ONLY; a decline shows a non-blocking
        guard-style nudge and never blocks Detect. Attribute filters are stored
        for the run but not yet used for filtering in this pass."""
        from ...core.prompt_rewrite import parse_prompt_rewrite

        action, payload, filters = parse_prompt_rewrite(plan.get("prompt_rewrite"))
        # Attribute filters are informational this pass: store them (cleared on
        # the next commit / run reset) so a later honest-count UI can read them.
        self._auto_attribute_filters = filters
        dock = self.dock_widget
        if dock is None:
            return False
        if action == "rewrite":
            # Route through the shared swap-and-tell channel (setText + quiet
            # note + rewrite telemetry): the box swaps to the rewritten phrase
            # and its own commit re-seeds the detail for the new token.
            try:
                return bool(dock.apply_prompt_swap(payload, "server_rewrite"))
            except (RuntimeError, AttributeError):
                return False
        if action == "decline":
            # Show the model's reason as a non-blocking nudge; never blocks the
            # run (the user can Detect regardless).
            try:
                return bool(dock.show_auto_prompt_decline(payload))
            except (RuntimeError, AttributeError):
                return False
        return False

    def _on_auto_run_plan_failed(self) -> None:
        """Run-plan fetch failed/timed out: keep the blob path silently."""
        self._auto_run_plan_task = None

    # ---- Async server token resolution (offline-lexicon miss) --------------

    def _fetch_auto_token(self, raw: str) -> None:
        """Fire-and-forget: resolve a prompt the offline lexicon could not to
        its English token via the server, so the detail seed can key on it
        without waiting for Detect.

        Off the GUI thread via a hidden task; never blocks, fails open (the raw
        prompt keeps the generic seed on any error). A no-op for an already
        English object word (nothing to translate) and mid-run/review (the
        prompt is locked then). A late answer re-seeds only while the prompt is
        still the one shown (see _on_auto_token_ready)."""
        raw = (raw or "").strip()
        if not raw or not self.dock_widget:
            return
        if self._auto_worker is not None or self._auto_review is not None:
            return
        try:
            from ..dock.prompt_guard import is_known_object
            if is_known_object(raw):
                return  # already an English object word: nothing to resolve
        except Exception:  # noqa: BLE001 -- best-effort gate  # nosec B110
            pass
        # Only one in flight; a newer commit supersedes the older request.
        self._cancel_task("_auto_token_task")
        try:
            from qgis.core import QgsApplication

            from ...api.prompt_translation import resolve_english_prompt
            from ...workers.generic_request_task import GenericRequestTask
            task = GenericRequestTask(
                tr("Resolving object name"),
                lambda: {"token": resolve_english_prompt(raw)},
                hidden=True,
            )
            task.succeeded.connect(
                lambda res, r=raw: self._on_auto_token_ready(r, res))
            task.failed.connect(lambda *_a: self._on_auto_token_failed())
            self._auto_token_task = task
            QgsApplication.taskManager().addTask(task)
        except Exception:  # noqa: BLE001 -- translation is best-effort
            self._auto_token_task = None  # nosec B110

    def _on_auto_token_ready(self, raw: str, result: object) -> None:
        """Main thread: cache the server-resolved token and re-seed the detail,
        but only while ``raw`` is still the committed prompt. The blob re-seed
        enforces the manual slider lock, so a user who took over the slider is
        never overridden."""
        self._auto_token_task = None
        token = result.get("token") if isinstance(result, dict) else None
        if not isinstance(token, str) or not token:
            return
        raw = (raw or "").strip()
        if not raw or token.strip().lower() == raw.lower():
            return  # server confirmed it was already English: nothing to swap
        if raw.lower() != self._current_auto_object_class().strip().lower():
            return  # the user moved on to a different object since the fetch
        cache = getattr(self, "_auto_token_cache", None)
        if cache is None:
            cache = {}
            self._auto_token_cache = cache
        cache[raw.lower()] = token
        # Re-seed with the resolved token and refresh the run plan under it.
        self._reseed_auto_detail_from_blob(token)
        self._fetch_auto_run_plan(token)

    def _on_auto_token_failed(self) -> None:
        """Token resolution failed/timed out: keep the raw-prompt path."""
        self._auto_token_task = None

    def _reseed_auto_detail_from_plan(self, prompt: str, plan: dict) -> None:
        """Re-seed the detail slider from the plan's target resolution, reusing
        the same coarse -> fine walk as the blob seed. No-op while a run/review
        is in flight or when the user already took over the slider for this
        prompt."""
        if not self.dock_widget or self._auto_zone is None:
            return
        if self._auto_worker is not None or self._auto_review is not None:
            return
        prompt = (prompt or "").strip()
        if not prompt or prompt.lower() != self._resolved_auto_object_class().strip().lower():
            return
        if self._auto_detail_user_locked and getattr(self, "_auto_detail_lock_prompt", "") == prompt.lower():
            return
        target_mupp = plan.get("target_mupp")
        if not isinstance(target_mupp, (int, float)) or isinstance(target_mupp, bool) or target_mupp <= 0:
            return
        obj_m = plan.get("object_size_m")
        is_valid_obj_m = isinstance(obj_m, (int, float)) and not isinstance(obj_m, bool) and obj_m > 0
        obj_m = float(obj_m) if is_valid_obj_m else 10.0
        layer = self._get_active_raster_layer()
        if layer is None:
            return
        try:
            zone_in_layer = self._reproject_zone_to_run_crs(self._auto_zone, layer)
            # The plan is already stored under this prompt, so the shared
            # profile reader resolves the tile floor the same way the slider
            # band does: the plan's when it names one, else the blob tier's.
            detail = self._auto_detail_for_target(
                layer, zone_in_layer, obj_m, float(target_mupp),
                self._detail_window_profile(prompt)[1])
            self._seed_auto_detail_value(detail)
            self._update_credit_estimate()
        except (RuntimeError, AttributeError):
            pass

    def _push_detail_feedback(self, layer, zone_in_layer, ground_mupp: float) -> None:
        """Classify the CURRENT detail level against the named object and this
        zone, and hand the verdict to the dock's one-line slider guidance.

        The verdict compares the level the user chose with the level the seed
        logic recommends for the same prompt + zone (so the guidance moves with
        the object AND the zone), plus two physical bounds: the object no
        longer resolvable at this resolution (too coarse to spot), and the
        split-risk band, where the object's own ground size approaches the
        tile's ground side.

        The Fine-end warning is anchored on the recommended level, so it can
        never fire on the setting the plugin itself proposes. The Coarse-end
        one still can, on a zone so large the seed's tile cap leaves the object
        under the minimum pixel size: there it is true, and raising the slider
        is the fix. Runs at the debounced credit-estimate chokepoint, so it
        tracks every slider drag, prompt commit, zone redraw and layer switch.
        """
        if self.dock_widget is None:
            return
        if ground_mupp <= 0:
            self.dock_widget.set_auto_detail_feedback(None, "")
            return
        obj = self._current_auto_object_class()
        # Key the sizing verdict on the resolved English token (so a localized
        # prompt is judged like its English equivalent), but display the user's
        # own word in the hint.
        obj_token = self._resolve_object_token(obj)
        has_reference = False
        try:
            has_reference = self._auto_exemplar_store.count() > 0
        except (RuntimeError, AttributeError):
            pass
        if not obj and not has_reference:
            # No object defined: the detail card is gated with its own hint.
            self.dock_widget.set_auto_detail_feedback(None, "")
            return
        try:
            from ...core.detection_policy import object_min_px

            obj_m, target_mupp = self._object_detail_profile(obj_token)
            value = self._get_auto_detail_level()
            recommended = self._recommended_detail_now(layer, zone_in_layer, obj_token)
            target_met = ground_mupp <= target_mupp
            if obj_m / ground_mupp < object_min_px():
                state = "coarse"
            elif value > recommended:
                if self._detail_splits_objects(
                        layer, zone_in_layer, ground_mupp, obj_m,
                        target_mupp, recommended):
                    state = "over"
                else:
                    # Past a budget-capped recommendation extra detail genuinely
                    # helps; past a target-met one it is mostly extra cost.
                    state = "above" if target_met else "helps"
            elif value < recommended:
                state = "below"
            else:
                state = "recommended"
            self.dock_widget.set_auto_detail_feedback(state, obj)
        except (RuntimeError, AttributeError, ValueError, ZeroDivisionError):
            pass

    def _detail_splits_objects(
        self, layer, zone_in_layer, ground_mupp: float, obj_m: float,
        target_mupp: float, recommended: int,
    ) -> bool:
        """Whether this level really risks returning the object in pieces, the
        one claim the amber Fine-end warning makes.

        Both conditions must hold. The object's typical ground size has to
        reach the split-risk share of a tile's ground side
        (TILE_SIZE * m/px): a small object on a wide tile comes back whole at
        any resolution, so on those the warning would simply be false. And the
        level has to sit past the point where finer pixels stop paying, taken
        against the resolution of the RECOMMENDED level rather than a fixed
        per-object target. The slider steps are coarse (one step near the
        Coarse end nearly doubles the resolution), so a fixed target left the
        warning one notch above the plugin's own pick; anchoring it on that
        pick cannot.
        """
        from ...core.detection_policy import (
            detail_over_ratio,
            detail_over_ratio_free,
            split_risk_tile_frac,
        )
        from ...core.tile_manager import TILE_SIZE

        if ground_mupp <= 0 or obj_m <= 0:
            return False
        if obj_m < split_risk_tile_frac() * TILE_SIZE * ground_mupp:
            return False
        # Free runs spend scarce trial credits, so their nudge fires earlier.
        is_subscriber = bool(getattr(
            self.dock_widget, "_auto_is_subscriber", False))
        over_ratio = (detail_over_ratio() if is_subscriber
                      else detail_over_ratio_free())
        rec_mupp = self._ground_mupp_for_detail(
            layer, zone_in_layer, recommended)
        # A recommendation held back by the tile budget is coarser than the
        # target; anchoring on it there would warn early again, so take the
        # finer of the two.
        anchor = min(rec_mupp, target_mupp) if rec_mupp > 0 else target_mupp
        return ground_mupp < anchor * over_ratio

    def _ground_mupp_for_detail(
        self, layer, zone_in_layer, detail_n: int
    ) -> float:
        """Ground resolution (m/px) one detail level renders at; 0.0 when the
        grid or the ground measurement cannot be computed."""
        sized = self._grid_for_detail(layer, zone_in_layer, detail_n)
        if sized is None:
            return 0.0
        return self._mupp_to_meters(layer, zone_in_layer, sized[2])

    def _recommended_detail_now(self, layer, zone_in_layer, object_class: str) -> int:
        """The level the seed logic recommends RIGHT NOW for this prompt + zone.

        Recomputed live (never cached) so a layer or zone change can never
        leave the guidance comparing against a stale pick. Prefers the active
        server run plan's target when one matches the prompt, exactly like the
        seed appliers, so the guidance never disagrees with the seed."""
        plan = self._active_run_plan(object_class) if object_class else None
        if plan:
            target_mupp = plan.get("target_mupp")
            if isinstance(target_mupp, (int, float)) and not isinstance(target_mupp, bool) and target_mupp > 0:
                obj_m = plan.get("object_size_m")
                is_valid_obj_m = isinstance(obj_m, (int, float)) and not isinstance(obj_m, bool) and obj_m > 0
                obj_m = float(obj_m) if is_valid_obj_m else 10.0
                return self._auto_detail_for_target(
                    layer, zone_in_layer, obj_m, float(target_mupp),
                    self._detail_window_profile(object_class)[1])
        if object_class:
            return self._auto_detail_for_object(layer, zone_in_layer, object_class)
        return self._default_detail_for_zone(layer, zone_in_layer)

    def _mupp_to_meters(self, layer, zone_in_layer, mupp: float) -> float:
        """Convert a run-CRS resolution (CRS units per pixel) to meters per
        pixel, measured at the zone center.

        An ellipsoidal measurement (WGS84) handles every CRS uniformly:
        projected layers in metres pass through, projected feet convert, and
        geographic layers in degrees get a real ground distance instead of a
        meaningless degree value. Returns 0.0 when it cannot be computed.
        """
        try:
            from qgis.core import (
                QgsDistanceArea,
                QgsPointXY,
                QgsProject,
            )

            measure_crs = self._run_crs_now(layer) or layer.crs()
            da = QgsDistanceArea()
            da.setSourceCrs(measure_crs, QgsProject.instance().transformContext())
            da.setEllipsoid("WGS84")
            cx = (zone_in_layer.xMinimum() + zone_in_layer.xMaximum()) / 2.0
            cy = (zone_in_layer.yMinimum() + zone_in_layer.yMaximum()) / 2.0
            # Measure BOTH axes and take the coarser ground distance. For a
            # geographic CRS (EPSG:4326) a pixel is square in degrees but a
            # degree of longitude shrinks with latitude, so a horizontal-only
            # measurement underestimates the effective GSD at high latitude and
            # skews the detail seed. Seeding on the coarser axis stays conservative
            # (never over-tiles). Projected metric CRSes are isotropic, so both
            # axes agree and this is a no-op there.
            dist_x = da.measureLine(QgsPointXY(cx, cy), QgsPointXY(cx + mupp, cy))
            dist_y = da.measureLine(QgsPointXY(cx, cy), QgsPointXY(cx, cy + mupp))
            dist = max(dist_x, dist_y)
            return da.convertLengthMeasurement(dist, DistanceMeters)
        except (RuntimeError, AttributeError, ValueError):
            return 0.0

    def _compute_auto_grid(self, layer) -> dict | None:
        """Compute the pixel grid for one automatic-detection run.

        Returns a dict with keys:
            pixel_w  (int)  -- total image width in pixels
            pixel_h  (int)  -- total image height in pixels
            zone_x   (int)  -- always 0 (bbox carries the extent for both paths)
            zone_y   (int)  -- always 0
            bbox     (tuple) -- (minx, miny, maxx, maxy) in the run CRS
            crs      (str)  -- authid of the run CRS, the one the bbox, the
                               render, the tiles and every detection use
            online   (bool) -- True when the map renderer path must be used
                               (always True once a zone is drawn, regardless of
                               layer type; False only for the no-zone local path)

        Returns None when a grid cannot be computed (e.g. no zone for an online layer).

        Slider-driven path (zone set, any layer type):
            Resolution is driven by the detail slider: the zone's longer side is
            rendered as exactly n tiles (snap formula: TILE_SIZE + (n-1) * stride
            pixels), so the user directly chooses how finely the zone is divided
            and therefore the credit cost. Local rasters are clamped to their
            native resolution so the render never upsamples the source pixels.
            online=True. zone_x and zone_y are 0 (the bbox IS the zone extent).

        Native path (no zone, local raster only):
            Derives pixel dimensions from the layer's own pixel grid. online=False.
            Online providers cannot run without a zone and return None here.
        """
        if self._tile_manager is None:
            self._setup_auto_mode()

        try:
            layer_w = layer.width()
            layer_h = layer.height()
            ext = layer.extent()
        except (RuntimeError, AttributeError):
            return None

        use_online = layer_w <= 0 or layer_h <= 0 or self._needs_canvas_render(layer)

        if self._auto_zone is not None:
            # Slider-driven path for EVERY layer once a zone is drawn: the user
            # picks the grid subdivision n (zone's longer side = n tiles), and
            # therefore the credit cost. The actual sizing lives in
            # _grid_for_detail so the slider cap and the m/px hint share it.
            zone_in_layer = self._reproject_zone_to_run_crs(self._auto_zone, layer)
            detail_n = self._get_auto_detail_level()
            sized = self._grid_for_detail(layer, zone_in_layer, detail_n)
            if sized is None:
                return None
            pixel_w, pixel_h, mupp, _tiles = sized

            # Extend the geo bbox so it matches the snapped pixel grid at the
            # SAME mupp on both axes. snap_dimensions() rounds each axis up
            # independently, so the snapped pixel aspect no longer equals the
            # zone aspect; rendering the raw zone into that grid stretches the
            # image (up to ~1.8x) and the model sees deformed objects. Anchor
            # at the top-left corner (image origin) and grow right/down. The
            # forward render and the worker's inverse mapping both read this
            # bbox, so detections stay correctly georeferenced; the only
            # visible effect is the preview grid spilling slightly past the
            # drawn zone (blank margin at a raster edge renders harmlessly).
            minx = zone_in_layer.xMinimum()
            maxy = zone_in_layer.yMaximum()
            maxx = minx + pixel_w * mupp
            miny = maxy - pixel_h * mupp

            run_crs = self._run_crs_now(layer)
            return {
                "pixel_w": pixel_w,
                "pixel_h": pixel_h,
                "zone_x": 0,
                "zone_y": 0,
                "bbox": (minx, miny, maxx, maxy),
                "crs": (_crs_run_identifier(run_crs)
                        or _crs_run_identifier(layer.crs())),
                "online": True,
            }

        # No zone (MCP / full-raster path): keep the native pixel grid. Online
        # providers cannot run without a zone.
        if use_online:
            return None
        if ext.width() <= 0:
            return None
        pixel_w, pixel_h = layer_w, layer_h
        zone_x, zone_y = 0, 0
        bbox = (ext.xMinimum(), ext.yMinimum(), ext.xMaximum(), ext.yMaximum())
        # No zone: the grid IS the raster's own pixel grid, read straight off
        # the layer, so this path stays in the layer's CRS whatever it is.
        return {
            "pixel_w": pixel_w, "pixel_h": pixel_h,
            "zone_x": zone_x, "zone_y": zone_y,
            "bbox": bbox, "crs": _crs_run_identifier(layer.crs()),
            "online": False,
        }
