







from __future__ import annotations

from ...core.i18n import tr


class AutoFlowCreditsMixin:


    def _refresh_auto_credits(self) -> None:









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



        import time as _time
        fetched_at = _time.monotonic()
        task = GenericRequestTask(
            tr("Refreshing your cloud detections"),


            lambda: client.get_usage_with_account(auth=auth),
            hidden=True,
        )
        self._usage_fetch_task = task


        task.succeeded.connect(
            lambda usage, at=fetched_at, t=task: self._on_usage_fetched(usage, at, t))



        task.failed.connect(
            lambda msg, code, t=task: self._on_usage_failed(msg, code, t))
        QgsApplication.taskManager().addTask(task)

    def _settle_usage_fetch(self, task=None) -> None:









        if task is None or self._usage_fetch_task is task:
            self._usage_fetch_task = None
        if getattr(self, "_usage_refresh_pending", False) and self._usage_fetch_task is None:
            self._usage_refresh_pending = False
            self._refresh_auto_credits()

    def _on_usage_failed(self, message: str = "", code: str = "", task=None) -> None:






        self._settle_usage_fetch(task)
        normalized = (code or "").strip().upper()
        if (getattr(self, "_key_revalidate_pending", False)
                or normalized in ("INVALID_KEY", "DEVICE_LIMIT_EXCEEDED")):



            self._on_key_revalidate_failed(message, code)
            return
        if code == "SUBSCRIPTION_INACTIVE" and not self._billing_warning_shown:
            self._billing_warning_shown = True
            try:
                from qgis.core import Qgis

                from ...core.server_dials import dial_copy
                self.iface.messageBar().pushMessage(
                    "AI Segmentation",
                    dial_copy("copy.credits.subscription_inactive", tr(
                        "There's a problem with your subscription. Open Settings "
                        "to update your payment method or review your plan.")),
                    level=Qgis.MessageLevel.Warning,
                )
            except (RuntimeError, AttributeError):
                pass

    def _on_usage_fetched(self, usage: dict, fetched_at: float | None = None,
                          task=None) -> None:









        self._settle_usage_fetch(task)
        if getattr(self, "_key_revalidate_pending", False):
            self._on_key_revalidate_ok(usage)
        account = None
        if isinstance(usage, dict) and isinstance(usage.get("usage"), dict):
            account = usage.get("account")
            usage = usage["usage"]



        try:
            self._note_credits_reading(
                self._credits_reading_fingerprint(usage, account))
        except (RuntimeError, AttributeError):
            pass
        self._apply_usage_payload(usage, fetched_at=fetched_at)
        self._apply_account_envelopes(account, fetched_at=fetched_at)

    def _apply_account_envelopes(self, account, fetched_at: float | None = None) -> None:








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
















        if not usage or not isinstance(usage, dict) or not self.dock_widget:
            return
        import time as _time
        stamp = _time.monotonic() if fetched_at is None else float(fetched_at)
        applied_at = getattr(self, "_usage_applied_at", None)
        if applied_at is not None and stamp < applied_at:
            return
        self._usage_applied_at = stamp



        prev = self._last_usage or {}
        prev_free = prev.get("free_detections_remaining")
        self._last_usage = dict(usage)
        is_free = usage.get("is_free_tier", True)




        if prev and bool(prev.get("is_free_tier", True)) and not is_free:
            self._announce_plan_upgrade()

            try:
                self.dock_widget.set_auto_zone_rejected(None)
            except (RuntimeError, AttributeError):
                pass
        free_left = usage.get("free_detections_remaining")
        if is_free and prev_free is not None and free_left is not None and free_left < prev_free:
            try:
                from ...core import telemetry_session_events
                telemetry_session_events.track_free_taste_consumed(
                    remaining=free_left,

                    run_id=self._auto_run_id or "",
                )
            except Exception:
                pass  # nosec B110
        if is_free:





            credits = free_left




            from ...core.detection_policy import free_monthly_allowance



            served_total = usage.get("free_detections_total")
            total = (int(served_total)
                     if isinstance(served_total, (int, float)) and served_total > 0
                     else free_monthly_allowance())
        else:


            used = usage.get("images_used", 0)
            limit = usage.get("images_limit", 0)
            used = used if isinstance(used, (int, float)) and not isinstance(used, bool) else 0
            limit = limit if isinstance(limit, (int, float)) and not isinstance(limit, bool) else 0
            credits = max(0, limit - used)
            total = limit or None




        reset_date = usage.get("reset_date") or usage.get("period_end") or ""
        self.dock_widget.set_auto_credits(credits, reset_date,
                                          is_subscriber=not is_free,
                                          total=total)




        if (self._auto_zone is not None and self._auto_worker is None and self._auto_review is None):
            self._update_credit_estimate()

    def _account_is_envelope_gated(self) -> bool:






        try:
            env = self.dock_widget.quota_envelopes() if self.dock_widget else None
        except (RuntimeError, AttributeError):
            return True
        if env is None:
            return False
        return bool(env.has_km2_gauge() or env.has_objects_gauge()
                    or env.km2_remaining is not None
                    or env.objects_remaining is not None)

    def _apply_tile_balance(self, balance) -> bool:











        if not isinstance(balance, dict) or not self.dock_widget:
            return False
        if self._account_is_envelope_gated():
            return False
        last = self._last_usage
        if not isinstance(last, dict) or not last:


            return False
        merged = dict(last)
        if bool(last.get("is_free_tier", True)):
            free_left = balance.get("free_detections_remaining")
            if not isinstance(free_left, int):
                return False
            merged["free_detections_remaining"] = max(0, free_left)
        else:
            left = balance.get("credits_remaining")
            limit = last.get("images_limit")
            if not isinstance(left, int) or not isinstance(limit, (int, float)) or limit <= 0:
                return False
            merged["images_used"] = max(0, int(limit) - max(0, left))
        self._apply_usage_payload(merged)
        return True

    def _auto_credit_snapshot(self) -> tuple[int | None, bool]:




        from ...core.credit_gate import credit_snapshot

        return credit_snapshot(self._last_usage or {})

    def _auto_zone_area_km2(self) -> float:





        try:
            geom, crs = self._auto_billable_zone_geometry()
            if geom is None or crs is None:
                return 0.0
            from ...core.layer_conventions import make_area_measurer
            from ...core.zone_crs_check import zone_fits_declared_crs



            if not zone_fits_declared_crs(geom, crs):
                return 0.0
            da = make_area_measurer(crs)
            return max(0.0, da.measureArea(geom) / 1_000_000.0)
        except Exception:
            return 0.0

    def _auto_zone_tile_cap(self) -> int:











        from .shared import max_tiles_per_run_cap
        cap = max_tiles_per_run_cap(self._auto_zone_area_km2() or None)
        tiles = getattr(self, "_tile_manager", None)
        if tiles is not None:
            try:
                tiles.max_tiles = cap
            except (RuntimeError, AttributeError):
                pass  # nosec B110
        return cap

    def _auto_duration_ms(self) -> int:

        try:
            import time as _time
            if not self._auto_detect_t0:
                return 0
            return int((_time.monotonic() - self._auto_detect_t0) * 1000)
        except Exception:
            return 0
