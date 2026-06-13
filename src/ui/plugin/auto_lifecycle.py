





from __future__ import annotations

import contextlib
import os

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsGeometry,
    QgsMessageLog,
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
)

from ...core.i18n import tr
from .shared import (
    _apply_fast_render,
    release_worker_ref,
)












EXAMPLE_MATCH_CLASS = "Example match"


class AutoLifecycleMixin:


    def _autosave_pending_auto_review(self, exit_path: str = "other") -> bool:















        if self._land_cover_autosave_pending(exit_path) is False:
            return False
        try:


            self._finish_auto_review_export_offload()
            review = self._auto_review




            if not (review and self._auto_objects):
                return True


            self._track_review_abandoned(exit_path)
            exported = self._export_auto_review(include_hidden=True, autosave=True)



            if exported and exported[0] and exported[1] > 0 and self.dock_widget:
                name, count = exported
                try:



                    template = (
                        tr("Saved the 1 object found to {name}.") if count == 1
                        else tr("Saved all {n} objects found to {name}, "
                                "including any the Confidence slider hid."))
                    self.dock_widget.set_auto_status(
                        "info", template.format(n=count, name=name or ""))
                except (RuntimeError, AttributeError):
                    pass
        except Exception:  # nosec B110
            try:
                QgsMessageLog.logMessage(
                    "Auto review autosave failed", "AI Segmentation",
                    level=Qgis.MessageLevel.Warning)
            except Exception:  # nosec B110
                pass
        return True

    def _forget_auto_last_run_km2(self) -> None:


        try:
            if self.dock_widget is not None:
                self.dock_widget._auto_last_run_km2 = None
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _end_superseded_finalize(self, show_notice: bool = True) -> None:









        state = getattr(self, "_auto_finalize_state", None)
        if not isinstance(state, dict) or state.get("mode") != "finalize":
            return
        if state.get("superseded"):
            return
        state["superseded"] = True


        if getattr(self, "_auto_tel_stop_reason", None) in (None, "completed"):
            try:
                from ...core import telemetry_run_events
                from .auto_client_profile import client_profile_props
                done = int(state.get("tiles_succeeded", 0) or 0)
                telemetry_run_events.track_auto_detect_cancelled(
                    run_id=self._auto_run_id or "",
                    tiles_done=done,
                    tiles_total=int((self._auto_run_ctx or {}).get("total", done) or done),
                    salvaged_to_review=False,
                    duration_ms=self._auto_duration_ms(),
                    client_profile=client_profile_props(self),
                )
            except Exception:  # noqa: BLE001  # nosec B110
                pass
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_finalizing(False)
            except (RuntimeError, AttributeError):
                pass
        if show_notice:
            try:
                self.iface.messageBar().pushMessage(
                    "AI Segmentation",
                    tr("Your results are kept in Library > History."),
                    level=Qgis.MessageLevel.Info, duration=8)
            except Exception:  # noqa: BLE001  # nosec B110
                pass

    def _discard_auto_review(self, exit_path: str = "other",
                             keep_run_policy: bool = False) -> bool:









        if self._land_cover_autosave_pending(exit_path) is False:
            return False
        self._forget_auto_last_run_km2()
        self._end_superseded_finalize(show_notice=exit_path != "unload")





        self._abort_qgis_edit_bridge_if_active()





        try:
            if getattr(self, "_refine_handoff_active", False) and self.saved_polygons:
                self._collect_manual_refine_into_review()
            else:
                self._abandon_fix_session_for_discard()
        except Exception as exc:  # noqa: BLE001


            try:
                from ...core.telemetry_errors import track_plugin_error
                track_plugin_error(stage="segment",
                                   error_code="review_fold_edits_failed",
                                   message=type(exc).__name__)
            except Exception:  # noqa: BLE001  # nosec B110
                pass
        self._autosave_pending_auto_review(exit_path)
        self._land_cover_teardown()
        self._auto_review = None


        if not keep_run_policy:
            _release_run_policy()
        self._clear_free_zone_review_outline()




        self._auto_finalize_gen += 1
        self._auto_finalize_state = None



        self._stop_review_refine_thread()
        self._remove_auto_selection_layer()
        self._auto_manual_removed = set()



        self._release_local_ai_install()




        try:
            self._end_correct_focus()
        except (RuntimeError, AttributeError):
            pass
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_review_active(False)
            except (RuntimeError, AttributeError):
                pass
        return True



    def _autosave_billed_results(self, merged_ided: list) -> None:














        if self._auto_headless_run:
            return
        if self._start_billed_autosave(merged_ided):
            return
        try:
            from ...core import run_autosave
            ctx = self._auto_run_ctx or {}
            prompt = str(ctx.get("prompt") or "").strip()
            source_layer = None
            layer_id = ctx.get("layer_id")
            if layer_id:
                source_layer = QgsProject.instance().mapLayer(layer_id)
            info = run_autosave.write_autosave(
                merged_ided, self._auto_crs_authid or "EPSG:4326", prompt,
                self._auto_run_id or "", source_layer=source_layer)
            if not info:
                return
            run_autosave.record_pending(info)
            QgsMessageLog.logMessage(
                "Auto detection: autosaved {n} object(s) to disk before "
                "review".format(n=info.get("count", 0)),
                "AI Segmentation", level=Qgis.MessageLevel.Info,
            )



            if not getattr(self, "_auto_retain_raw", False):
                self._auto_raw_fragments = None
        except Exception:  # noqa: BLE001
            try:
                QgsMessageLog.logMessage(
                    "Auto detection: pre-review autosave failed",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
            except Exception:  # nosec B110
                pass







    def _on_auto_progress(self, completed: int, total: int) -> None:

        self._note_auto_progress()
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_tile_progress(completed, total)
                if total > 0 and completed >= total:



                    self.dock_widget.note_auto_tiles_all_answered()
                    self._push_auto_assemble_progress()
            except (RuntimeError, AttributeError):
                pass

    def _on_auto_queue_state(self, position: int, depth: int, eta_s: int) -> None:







        import time as _time
        if position or depth or eta_s:
            if getattr(self, "_auto_warming_t0", None) is None:
                self._auto_warming_t0 = _time.monotonic()
        else:
            t0 = getattr(self, "_auto_warming_t0", None)
            if t0 is not None:
                self._auto_warming_ms = getattr(self, "_auto_warming_ms", 0) + int((_time.monotonic() - t0) * 1000)
                self._auto_warming_t0 = None
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_queue_state(position, depth, eta_s)
            except (RuntimeError, AttributeError):
                pass

    def _auto_warming_wait_ms(self) -> int:


        try:
            import time as _time
            total = getattr(self, "_auto_warming_ms", 0)
            t0 = getattr(self, "_auto_warming_t0", None)
            if t0 is not None:
                total += int((_time.monotonic() - t0) * 1000)
            return total
        except Exception:
            return 0

    def _on_auto_warning(self, msg: str) -> None:
        QgsMessageLog.logMessage(
            f"Auto detection warning: {msg}",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )


        low = (msg or "").lower()
        if "timeout" in low or "timed out" in low:
            self._auto_timeout_tiles += 1
        elif "skip" in low:
            self._auto_skipped_tiles += 1



            if "could not process result" in low:
                self._auto_convert_failed_tiles = (
                    getattr(self, "_auto_convert_failed_tiles", 0) + 1)

    def _on_auto_nothing_found_yet(self, tiles: int) -> None:





        QgsMessageLog.logMessage(
            f"Auto detection: {tiles} tiles, nothing found yet",
            "AI Segmentation", level=Qgis.MessageLevel.Info,
        )

    def _auto_account_refusal_line(self, msg: str) -> str:













        from ...core.error_policy import account_refusal_reason

        reason = account_refusal_reason(msg)
        if reason == "SUBSCRIPTION":
            return tr("Your plan is not active, so the cloud AI cannot run. "
                      "Check your subscription to keep detecting.")
        if reason == "NO_ACCOUNT":
            return tr("No account matches this sign-in. Sign in with the "
                      "account that has your plan.")
        return tr("Session expired. Sign in again to continue.")





    def _store_auto_resume_state(self, worker) -> bool:




        try:
            ctx = self._auto_run_ctx or {}
            state = worker.resume_state()
            from ...core.activation_manager import auth_revision

            if state.get("auth_revision") != auth_revision():
                self._auto_resume_state = None
                return False
            if not state["missing"] or not ctx.get("layer_id"):
                self._auto_resume_state = None
                return False
            state["layer_id"] = ctx.get("layer_id")
            state["prompt"] = ctx.get("prompt")
            state["layer_source"] = ctx.get("layer_source")
            state["layer_provider"] = ctx.get("layer_provider")
            state["layer_dataset_identity"] = ctx.get("layer_dataset_identity")
            self._auto_resume_state = state
            if not getattr(self, "_auto_resume_wired", False):

                QgsProject.instance().cleared.connect(self._clear_auto_resume_state)
                self._auto_resume_wired = True
            return True
        except (RuntimeError, AttributeError, KeyError, TypeError):
            self._auto_resume_state = None
            return False

    def _clear_auto_resume_state(self) -> None:
        self._auto_resume_state = None
        self._auto_resume_armed = False
        self._auto_resume_backup = None
        self._pop_auto_resume_item()

    def _auto_resume_available(self) -> bool:

        state = getattr(self, "_auto_resume_state", None)
        if not state or getattr(self, "_auto_zone", None) is None:
            return False
        return QgsProject.instance().mapLayer(state.get("layer_id") or "") is not None

    def _push_auto_resume_offer(self, banner: str) -> None:


        from qgis.PyQt.QtWidgets import QPushButton

        state = self._auto_resume_state
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_auto_resume(
                True, state["run_id"], len(state["missing"]),
                state["answers"] is not None)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        try:
            bar = self.iface.messageBar()
            item = bar.createMessage("AI Segmentation", banner)
            button = QPushButton(tr("Continue missing tiles"))
            button.setAutoDefault(False)
            button.clicked.connect(lambda _=False, it=item: self._on_auto_resume_clicked(it))
            item.layout().addWidget(button)
            bar.pushWidget(item, Qgis.MessageLevel.Warning)
            self._auto_resume_item = item
        except (RuntimeError, AttributeError):
            pass

    def _on_auto_resume_clicked(self, item=None) -> None:




        if (not self._auto_resume_available() or self.dock_widget is None
                or getattr(self, "_auto_resume_busy", False)
                or getattr(self, "_auto_start_in_progress", False)):


            if not self._auto_resume_available():
                self._clear_auto_resume_state()
            return
        if self._auto_worker is not None and self._auto_worker.isRunning():
            return
        state = self._auto_resume_state
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_auto_resume(
                False, state["run_id"], len(state["missing"]),
                state["answers"] is not None)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        self._auto_resume_item = item
        self._auto_resume_armed = True
        self._auto_resume_busy = True
        try:
            self._start_auto_detection()
        finally:
            self._auto_resume_busy = False



            if getattr(self, "_auto_imagery_probe", None) is None:
                self._auto_resume_armed = False

    def _commit_auto_resume(self, state: dict) -> None:





        import time as _time
        import types

        self._pop_auto_resume_item()
        self._auto_resume_backup = state
        self._discard_review_without_autosave(exit_path="new_run", keep_run_policy=True)
        try:
            self.dock_widget.set_auto_review_active(False)
            self._set_zone_band_fill_visible(True)
            self.dock_widget.set_auto_zone_state("zone_set")
            self._restore_tile_grid_after_run()
            self.dock_widget.set_auto_status("idle")
        except (RuntimeError, AttributeError):
            pass



        self._auto_density_forced = {
            "run_id": state["run_id"],
            "decision": types.SimpleNamespace(side_m=0.0),
            "click": _time.monotonic(),
            "from_m": 0.0,
        }

    def _pop_auto_resume_item(self) -> None:
        item = getattr(self, "_auto_resume_item", None)
        self._auto_resume_item = None
        if item is None:
            return
        try:
            self.iface.messageBar().popWidget(item)
        except (RuntimeError, AttributeError):
            pass

    def _restore_auto_resume_offer(self) -> None:



        state = getattr(self, "_auto_resume_backup", None)
        self._auto_resume_backup = None
        if not state:
            return
        self._auto_resume_state = state
        self._push_auto_resume_offer(
            tr("The run did not start. Continue to try again."))

    def _auto_resume_plan_ok(self, tiles, prompt, layer, *,
                             geo_transform=None, crs_authid=None) -> bool:



        if not getattr(self, "_auto_resume_armed", False):
            return True
        state = self._auto_resume_state
        plan_len = int(state["plan_len"]) if state else -1
        current_context = {
            "geo_transform": geo_transform,
            "crs_authid": crs_authid,
            "prompt": prompt,
        }
        from ...core.activation_manager import auth_revision
        from ...core.raster_dataset_cache import dataset_identity

        if (state is not None and layer.id() == state.get("layer_id")
                and state.get("auth_revision") == auth_revision()
                and prompt == state.get("prompt")
                and layer.source() == state.get("layer_source")
                and layer.providerType() == state.get("layer_provider")
                and dataset_identity(layer.source()) == state.get("layer_dataset_identity")
                and current_context == state.get("request_context")
                and list(tiles) == list(state["tiles"][:plan_len])):
            self._commit_auto_resume(state)
            return True
        self._clear_auto_resume_state()
        try:
            self.iface.messageBar().pushWarning(
                "AI Segmentation",
                tr("The zone or settings changed, so this run cannot be "
                   "continued. Run Detect again."))
        except (RuntimeError, AttributeError):
            pass
        return False

    def _apply_pending_resume(self, worker) -> None:


        armed = getattr(self, "_auto_resume_armed", False)
        state = getattr(self, "_auto_resume_state", None)
        self._auto_resume_armed = False
        self._auto_resume_state = None
        if not armed or not state:

            self._pop_auto_resume_item()
            self._auto_resume_backup = None
            return
        try:
            worker.apply_resume(state)
        except (ValueError, KeyError, TypeError):


            import uuid
            self._auto_run_id = str(uuid.uuid4())
            worker._run_id = self._auto_run_id

    def _on_auto_error(self, msg: str) -> None:
        from ...core.error_policy import REPORTABLE_ERROR_CLASSES
        QgsMessageLog.logMessage(
            f"Auto detection error: {msg}",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
        self._set_zone_badge_enabled(True)



        self._set_zone_band_fill_visible(True)



        error_class = self._classify_auto_error(msg)
        is_auth = error_class == "AUTH"











        salvaged_tiles = getattr(self._auto_worker, "tiles_succeeded", 0)
        if is_auth:
            banner = self._auto_account_refusal_line(msg)
        elif error_class == "DEVICE_LIMIT":


            banner = tr("Your plan is already running on its maximum number of "
                        "computers. Close AI Segmentation on one of them, then "
                        "run Detect again.")
        elif salvaged_tiles > 0 and not self._auto_headless_run:



            banner = tr("Detection stopped early. Everything found is kept "
                        "below and stays yours.")
        elif error_class == "SERVER":



            banner = tr("The detection service had a problem and the run "
                        "stopped. Please try again.")
        elif error_class == "TIMEOUT":
            banner = tr("The detection service is busy right now. "
                        "Please try again in a moment.")
        elif error_class == "NETWORK":
            banner = tr("Detection failed. Check your connection and try again.")
        else:
            banner = tr("Detection failed. Run Detect again, and lower the "
                        "precision if it fails a second time.")



        code = (self._auto_run_id or "")[:8]
        if code and error_class in ("SERVER", "TIMEOUT", "UNKNOWN"):
            banner = banner + "\n" + tr("Support code: {code}").format(code=code)


        report_payload = None
        if error_class in REPORTABLE_ERROR_CLASSES:
            report_payload = (
                tr("Automatic detection failed"), msg,
                "auto_detect_" + error_class.lower())
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_run_active(False)
                self.dock_widget.set_auto_status(
                    "error", banner, report_payload=report_payload)
            except (RuntimeError, AttributeError):
                pass

        self._last_auto_result = {"status": "error", "message": msg}
        self._auto_tel_stop_reason = "error"
        try:
            from ...core import telemetry_errors, telemetry_run_events
            from .auto_client_profile import client_profile_props, snapshot_worker_profile
            snapshot_worker_profile(self, self._auto_worker)
            telemetry_run_events.track_auto_detect_failed(
                run_id=self._auto_run_id or "",
                error_class=error_class,
                tiles_done=getattr(self._auto_worker, "tiles_succeeded", 0),
                duration_ms=self._auto_duration_ms(),
                warming_ms=self._auto_warming_wait_ms(),
                client_profile=client_profile_props(self),
            )




            if error_class in ("SERVER", "TIMEOUT", "UNKNOWN"):
                telemetry_errors.track_plugin_error(
                    stage="segment",
                    error_code="auto_detect_" + error_class.lower(),
                    message=msg,
                )
        except Exception:
            pass  # nosec B110
        try:
            from ...core.run_log_capture import send_run_log
            send_run_log("failed")
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        worker = self._auto_worker



        release_worker_ref(worker)
        self._auto_worker = None
        self._drop_auto_tile_bridge()
        self._capture_auto_mask_gsd(worker)
        tiles_succeeded = getattr(worker, "tiles_succeeded", 0)
        offer_resume = False
        if (tiles_succeeded > 0 and getattr(worker, "stopped_offline", False)
                and not self._auto_headless_run):
            offer_resume = self._store_auto_resume_state(worker)
        if tiles_succeeded > 0:







            self._finalize_auto_results(tiles_succeeded)




            if offer_resume and not is_auth:
                self._push_auto_resume_offer(banner)
            elif not is_auth:
                try:
                    self.iface.messageBar().pushWarning("AI Segmentation", banner)
                except (RuntimeError, AttributeError):
                    pass
        else:





            self._reset_auto_live_pipeline()
            self._stop_auto_stall_watchdog()
            self._remove_auto_selection_layer()

            _release_run_policy()



            if self.dock_widget and not self._auto_headless_run:
                try:
                    self._restore_tile_grid_after_run()
                except (RuntimeError, AttributeError):
                    pass




            if report_payload is not None:
                self._open_auto_error_report(
                    *report_payload, track=(error_class == "NETWORK"))
        if is_auth and not self._auto_headless_run:


            try:



                self.iface.messageBar().pushWarning("AI Segmentation", banner)
            except (RuntimeError, AttributeError):
                pass
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(0, self._on_settings_clicked)




        was_headless = self._auto_headless_run
        self._auto_headless_run = False
        self._auto_review_preset_overrides = None



        if was_headless and getattr(self, "_auto_review", None) is None:
            try:
                from ...core.detection_policy_core import release_run_policy
                release_run_policy()
            except Exception:  # noqa: BLE001  # nosec B110
                pass

    @staticmethod
    def _auto_quota_refusal(worker) -> dict:

        try:
            return worker.quota_refusal_detail() if worker is not None else {}
        except (RuntimeError, AttributeError):
            return {}

    @staticmethod
    def _quota_refusal_is_zone_size(refusal: dict) -> bool:






        try:
            used = float(refusal.get("used"))
            limit = float(refusal.get("limit"))
        except (TypeError, ValueError):
            return False
        return limit > 0 and used < limit

    def _quota_stop_banner(
        self, zone_too_large: bool, tiles_succeeded: int, tiles_total
    ) -> str:






        if zone_too_large:


            return tr("This zone is larger than the surface you have left "
                      "this month. Draw a smaller zone, or get Pro for a "
                      "larger monthly surface.")
        if tiles_succeeded <= 0:
            return tr("Your monthly allowance ran out, so this run did not "
                      "start.")


        return tr("Your monthly allowance ran out before the end of the "
                  "zone. Everything found so far is kept below and stays "
                  "yours.")

    def _on_auto_credits_exhausted(self, remaining: int) -> None:
        QgsMessageLog.logMessage(
            f"Auto detection: credits exhausted (remaining={remaining})",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
        self._set_zone_badge_enabled(True)



        worker = self._auto_worker



        release_worker_ref(worker)
        self._auto_worker = None
        self._drop_auto_tile_bridge()
        self._capture_auto_mask_gsd(worker)
        tiles_succeeded = getattr(worker, "tiles_succeeded", 0)
        tiles_total = (self._auto_run_ctx or {}).get("total", tiles_succeeded)
        _, is_free_tier = self._auto_credit_snapshot()
        refusal = self._auto_quota_refusal(worker)
        zone_too_large = self._quota_refusal_is_zone_size(refusal)
        banner = self._quota_stop_banner(
            zone_too_large, tiles_succeeded, tiles_total)


        if refusal.get("message"):
            QgsMessageLog.logMessage(
                f"Auto detection: quota refusal - {refusal['message']}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )



        self._auto_quota_stop_banner = banner if tiles_succeeded <= 0 else None
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_run_active(False)

                self.dock_widget.set_auto_status("info", banner)



                self.dock_widget.set_auto_exhausted_subscribe_visible(
                    is_free_tier and not zone_too_large)
            except (RuntimeError, AttributeError):
                pass




        if tiles_succeeded > 0 and not self._auto_headless_run:
            try:
                self.iface.messageBar().pushWarning("AI Segmentation", banner)
            except (RuntimeError, AttributeError):
                pass
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_credits_exhausted(
                run_id=self._auto_run_id or "",
                tiles_done=tiles_succeeded,
                tiles_total=tiles_total,
                is_free_tier=is_free_tier,
            )
        except Exception:
            pass  # nosec B110
        self._auto_tel_stop_reason = "exhausted"

        self._refresh_auto_credits()






        self._last_auto_result = {"status": "credits_exhausted", "credits_remaining": remaining}
        self._finalize_auto_results(tiles_succeeded)

    def _salvage_headless_timeout(self) -> dict | None:






        import time as _t

        from qgis.PyQt.QtCore import QEventLoop, QTimer

        from .auto_run import _headless_cancel_grace_ms, _headless_cancel_poll_ms

        self._on_auto_cancel_clicked()
        deadline = _t.monotonic() + _headless_cancel_grace_ms() / 1000.0
        while self._last_auto_result is None and _t.monotonic() < deadline:
            wait = QEventLoop()
            QTimer.singleShot(_headless_cancel_poll_ms(), wait.quit)
            wait.exec()
        return self._last_auto_result

    def _on_auto_cancelled(self, reason: str = "user", worker=None) -> None:











        stalled = reason == "stalled"
        if worker is not None and worker is not self._auto_worker:
            return
        worker = self._auto_worker
        if worker is None or self._auto_merger is None:












            release_worker_ref(worker)
            self._auto_worker = None
            self._drop_auto_tile_bridge()
            if getattr(self, "_auto_review", None) is None:
                _release_run_policy()
            return



        from .auto_client_profile import client_profile_props, snapshot_worker_profile
        snapshot_worker_profile(self, worker)
        release_worker_ref(worker)
        self._auto_worker = None
        self._drop_auto_tile_bridge()
        self._capture_auto_mask_gsd(worker)
        self._set_zone_badge_enabled(True)
        tiles_succeeded = getattr(worker, "tiles_succeeded", 0)
        tiles_total = (self._auto_run_ctx or {}).get("total", tiles_succeeded)








        warming_ms = self._auto_warming_wait_ms()
        try:
            health = worker.run_health_summary() if worker is not None else {}
        except (RuntimeError, AttributeError):
            health = {}
        submit_retries = int(health.get("submit_retries", 0) or 0)
        skipped_network = int(health.get("tiles_skipped_network", 0) or 0)



        timed_out = int(health.get("tiles_timed_out", 0) or 0)
        from .shared import backend_stalled_flag
        backend_stalled = backend_stalled_flag(
            tiles_succeeded, warming_ms, submit_retries, skipped_network,
            timed_out)
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_run_active(False)
                self.dock_widget.set_auto_status("idle")
            except (RuntimeError, AttributeError):
                pass
        try:
            from ...core import telemetry_run_events
            if stalled:



                telemetry_run_events.track_auto_detect_failed(
                    run_id=self._auto_run_id or "",
                    error_class="TIMEOUT",
                    tiles_done=tiles_succeeded,
                    duration_ms=self._auto_duration_ms(),
                    warming_ms=warming_ms,
                    client_profile=client_profile_props(self),
                )
            else:
                telemetry_run_events.track_auto_detect_cancelled(
                    run_id=self._auto_run_id or "",
                    tiles_done=tiles_succeeded,
                    tiles_total=tiles_total,
                    salvaged_to_review=tiles_succeeded > 0,
                    duration_ms=self._auto_duration_ms(),
                    warming_ms=warming_ms,
                    backend_stalled=backend_stalled,
                    submit_retries=submit_retries,
                    client_profile=client_profile_props(self),
                )
        except Exception:
            pass  # nosec B110
        try:
            from ...core.run_log_capture import send_run_log
            send_run_log("stalled" if stalled else "cancelled")
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        self._auto_tel_stop_reason = "stalled" if stalled else "cancelled"




        self._refresh_auto_credits()



        if stalled:



            if tiles_succeeded <= 0:
                msg = tr("The detection stopped responding before anything came "
                         "back. Check your connection, then run Detect again "
                         "(nothing was charged).")
            else:


                msg = tr("The detection stopped responding. Keeping what was "
                         "already found.")
            try:
                self.iface.messageBar().pushWarning("AI Segmentation", msg)
            except (RuntimeError, AttributeError):
                pass




        stop_status = "stalled" if stalled else "cancelled"
        self._last_auto_result = {"status": stop_status}


        self._finalize_auto_results(tiles_succeeded)


        merged = self._last_auto_result
        if isinstance(merged, dict) and merged:




            merged["status"] = stop_status
        else:
            self._last_auto_result = {"status": stop_status}
        QgsMessageLog.logMessage(
            "Auto detection: stalled" if stalled else "Auto detection: cancelled",
            "AI Segmentation", level=Qgis.MessageLevel.Info,
        )

    def _notice_auto_raster_removed(self, worker) -> None:





        try:
            kept = (self._auto_worker is worker
                    or self._auto_review is not None
                    or getattr(self, "_auto_finalize_state", None) is not None)
            if kept:
                msg = tr("The selected raster was removed. "
                         "Keeping what was already found.")
            else:
                msg = tr("The selected raster was removed.")
            self.iface.messageBar().pushInfo("AI Segmentation", msg)
        except (RuntimeError, AttributeError):
            pass

    def _on_layers_will_be_removed(self, layer_ids) -> None:











        try:
            ids = set(layer_ids or [])
        except TypeError:
            return
        if not ids:
            return



        probe = getattr(self, "_auto_imagery_probe", None)
        early = getattr(self, "_auto_imagery_early", None)
        if ((probe is not None and (probe.get("signature") or ("",))[0] in ids)
                or (early is not None
                    and (early.get("signature") or ("",))[0] in ids)):
            with contextlib.suppress(Exception):
                self._abandon_imagery_probe()


        with contextlib.suppress(Exception):
            self._stop_hover_preview("layer gone")





        with contextlib.suppress(Exception):
            project = QgsProject.instance()
            if any(isinstance(project.mapLayer(lid), QgsRasterLayer) for lid in ids):
                from ...core.raster_dataset_cache import release_raster_datasets
                release_raster_datasets()
        dock = self.dock_widget

        run_layer_id = (self._auto_run_ctx or {}).get("layer_id")
        dock_mid_flow = dock is not None and getattr(dock, "_auto_started", False)
        no_active_auto_run = self._auto_worker is None and self._auto_review is None
        if self._auto_worker is not None and run_layer_id in ids:
            worker = self._auto_worker
            self._on_auto_cancel_clicked()





            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(
                0, lambda w=worker: self._notice_auto_raster_removed(w))
        elif dock_mid_flow and no_active_auto_run and not self._refine_handoff_active:


            try:
                locked = dock.auto_layer_combo.currentLayer()
                locked_id = locked.id() if locked is not None else None
            except (RuntimeError, AttributeError):
                locked_id = None
            if locked_id and locked_id in ids:
                self._reset_auto_flow_to_start(exit_path="raster_removed")
                try:
                    dock.set_auto_status(
                        "info", tr("The selected raster was removed."))
                except (RuntimeError, AttributeError):
                    pass

        if self._refine_handoff_active:
            return
        if dock is None or not getattr(dock, "_segmentation_active", False):
            return
        try:
            manual_id = (self._current_layer.id()
                         if self._is_layer_valid() else None)
        except RuntimeError:
            manual_id = None
        if manual_id and manual_id in ids:
            has_manual_edits = (
                self.saved_polygons or self.current_mask is not None)
            has_frozen_display = (
                self._frozen_sessions or self._unfrozen_display_polygon is not None)
            had_work = bool(has_manual_edits or has_frozen_display)
            self._stop_manual_session(keep_saves=True)
            msg = (tr("The raster was removed. Your polygons were saved to a layer.")
                   if had_work else tr("The selected raster was removed."))
            try:
                self.iface.messageBar().pushInfo("AI Segmentation", msg)
            except (RuntimeError, AttributeError):
                pass



    def _export_auto_detections(
        self,
        deduped_geoms: list,
        crs: QgsCoordinateReferenceSystem,
        source_layer_name: str,
        prompt_label: str,
        scores: list | None = None,
        confidence_applied: float | None = None,
        det_ids: list | None = None,
    ) -> str | None:




























        export = self._prepare_auto_export(
            deduped_geoms, crs, source_layer_name, prompt_label,
            scores=scores, confidence_applied=confidence_applied, det_ids=det_ids)
        if export is None:
            return None
        from ...core.run_export_job import run_export_job

        return self._adopt_auto_export(export, run_export_job(export["job"]))

    def _prepare_auto_export(
        self,
        deduped_geoms: list,
        crs: QgsCoordinateReferenceSystem,
        source_layer_name: str,
        prompt_label: str,
        scores: list | None = None,
        confidence_applied: float | None = None,
        det_ids: list | None = None,
        row_classes: list | None = None,
        class_colors: list | None = None,
        land_cover: dict | None = None,
    ) -> dict | None:



        from ...core.run_export_job import prepare_run_export_job



        self._auto_export_layer_id = ""
        self._auto_export_feature_count = 0
        self._auto_export_failure = ""



        self._auto_exported_area_m2 = 0.0
        if not deduped_geoms:
            self._auto_export_failure = "nothing_visible"
            return None



        prompt_label = (prompt_label or "").strip() or EXAMPLE_MATCH_CLASS
        if scores is not None and len(scores) != len(deduped_geoms):
            scores = None
        if det_ids is not None and len(det_ids) != len(deduped_geoms):
            det_ids = None



        source_layer = None
        run_ctx = self._auto_run_ctx or {}
        run_layer_id = run_ctx.get("layer_id")
        if run_layer_id:
            source_layer = QgsProject.instance().mapLayer(run_layer_id)



        if source_layer is None and not run_ctx.get("restored"):
            source_layer = self._get_active_raster_layer()
        try:
            job = prepare_run_export_job(
                deduped_geoms, crs, prompt_label, scores=scores,
                det_ids=det_ids, source_layer=source_layer,
                row_classes=row_classes, class_colors=class_colors,
                land_cover=land_cover)
        except Exception:  # noqa: BLE001
            self._auto_export_failure = "file_refused"
            return None

        basemap_label = ""
        try:
            from ...core.basemap_label import online_basemap_credit
            if source_layer is not None:
                basemap_label = online_basemap_credit(source_layer)
        except (RuntimeError, AttributeError):
            basemap_label = ""
        return {
            "job": job,
            "crs": crs,
            "source_layer_name": source_layer_name,
            "prompt_label": prompt_label,
            "confidence_applied": confidence_applied,
            "basemap_label": basemap_label,
        }

    def _adopt_auto_export(self, export: dict, result: dict | None) -> str | None:




        from ...core import output_store
        from ...core.layer_conventions import (
            apply_output_conventions,
            make_class_categorized_renderer,
            make_committed_renderer,
        )
        from ...core.output_metadata import output_timestamp_iso

        result = result or {}
        self._auto_exported_area_m2 = float(result.get("area_m2") or 0.0)
        failure = result.get("failure") or ""
        if failure == "no_shapes":
            QgsMessageLog.logMessage(
                "Export refused: the layer provider took no features",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        if failure:
            self._auto_export_failure = failure
            return None
        job = export["job"]
        prompt_label = export["prompt_label"]
        crs = export["crs"]
        source_layer_name = export["source_layer_name"]
        written_count = int(result.get("count") or 0)
        overlapping_pairs = result.get("overlapping_pairs")
        result = output_store.load_written_run_table(
            result.get("written"), job["plan"]["friendly"])
        if result is None:
            self._auto_export_failure = "file_refused"
            return None
        result_layer = result.layer
        layer_name = result_layer.name()

        if result.used_fallback:
            try:
                self.iface.messageBar().pushWarning(
                    "AI Segmentation",
                    tr("Could not write to {name}. Saved to a separate "
                       "file instead.").format(name=os.path.basename(
                           result.intended_path or output_store.GPKG_FILENAME)))
            except (RuntimeError, AttributeError):
                pass











        object_class = job.get("object_class") or ""
        if job.get("class_colors"):
            from ...core.layer_conventions import make_legend_categorized_renderer
            class_renderer = make_legend_categorized_renderer(job["class_colors"])
        else:
            class_renderer = (
                make_class_categorized_renderer(result_layer)
                if not object_class else None)
        if class_renderer is not None:
            result_layer.setRenderer(class_renderer)
        else:
            result_layer.setRenderer(make_committed_renderer(
                color=output_store.committed_color_for_prompt(prompt_label)))


        _apply_fast_render(result_layer)
        try:
            plugin_version = self._read_plugin_version()
        except (RuntimeError, AttributeError):
            plugin_version = ""
        source_authid = ""
        try:
            source_authid = str(crs.authid() or "")
        except (RuntimeError, AttributeError):
            source_authid = ""
        confidence_applied = export.get("confidence_applied")
        apply_output_conventions(
            result_layer, source_layer_name,
            prompt=prompt_label,
            detail=(self._auto_run_ctx or {}).get("detail"),
            confidence=(confidence_applied if confidence_applied is not None
                        else self._auto_confidence),
            created_iso=output_timestamp_iso(),
            plugin_version=plugin_version,
            basemap_label=export.get("basemap_label") or "",
            source_crs_authid=source_authid,
            overlapping_pairs=overlapping_pairs,
        )





        try:
            from .canvas_redraw_handover import hold_map_picture_during_redraw
            hold_map_picture_during_redraw(self.iface.mapCanvas())
        except (RuntimeError, AttributeError):  # nosec B110
            pass







        output_store.add_committed_layer(result_layer, source_name=source_layer_name)

















        self._auto_export_layer_id = result_layer.id()


        self._auto_export_feature_count = written_count







        from ...core.qt_compat import safe_single_shot
        clip = getattr(self, "_auto_clip_polygon", None)
        history_context = {
            "clip": QgsGeometry(clip) if clip is not None else None,
            "run_id": str(getattr(self, "_auto_run_id", "") or ""),
            "account_dir": (self._auto_run_ctx or {}).get("account_dir"),
            "auth_revision": (self._auto_run_ctx or {}).get("auth_revision"),
        }
        safe_single_shot(
            0, self.dock_widget or self.iface.mainWindow(),
            lambda: self._record_detection_history(
                prompt_label, layer_name, written_count, crs, result_layer,
                history_context=history_context))

        QgsMessageLog.logMessage(
            f"Auto detection: saved {written_count} polygon(s) to {result.gpkg_path} "
            f"(table {result.table_name})",
            "AI Segmentation", level=Qgis.MessageLevel.Info,
        )

        return layer_name



    def _record_detection_history(
        self,
        prompt_label: str,
        layer_name: str,
        count: int,
        crs: QgsCoordinateReferenceSystem,
        result_layer: QgsVectorLayer,
        *,
        history_context: dict | None = None,
    ) -> None:










        try:
            from ...core import detection_history
            from ...core.activation_manager import auth_revision

            rect = None
            context = history_context if history_context is not None else {
                "clip": getattr(self, "_auto_clip_polygon", None),
                "run_id": str(getattr(self, "_auto_run_id", "") or ""),
                "account_dir": (getattr(self, "_auto_run_ctx", None) or {}).get("account_dir"),
                "auth_revision": (getattr(self, "_auto_run_ctx", None) or {}).get("auth_revision"),
            }
            revision = context.get("auth_revision")
            if revision is not None and revision != auth_revision():
                return
            account_dir = context.get("account_dir", detection_history.account_history_dir())
            if account_dir is None:
                account_dir = detection_history.account_history_dir()
            if detection_history.account_history_dir() != account_dir:
                return
            clip = context["clip"]
            if clip is not None and not clip.isEmpty():
                rect = clip.boundingBox()
            if rect is None or rect.isEmpty():





                rect = result_layer.extent()
                try:
                    from qgis.core import QgsCoordinateTransform

                    out_crs = result_layer.crs()
                    if (crs is not None and crs.isValid()
                            and out_crs.isValid() and out_crs != crs):
                        rect = QgsCoordinateTransform(
                            out_crs, crs, QgsProject.instance()
                        ).transformBoundingBox(rect)
                except Exception:  # noqa: BLE001
                    rect = None
            extent = None
            if rect is not None and not rect.isEmpty():
                extent = (rect.xMinimum(), rect.yMinimum(),
                          rect.xMaximum(), rect.yMaximum())



            run_id = context["run_id"]

            def _store(thumb: str | None) -> None:
                if ((revision is not None and revision != auth_revision())
                        or detection_history.account_history_dir() != account_dir):
                    return
                try:
                    detection_history.add_entry(
                        prompt=prompt_label,
                        layer_name=layer_name,
                        objects=count,
                        extent=extent,
                        crs_authid=crs.authid() if crs is not None else "",
                        thumb=thumb,
                        zone_wkt=(clip.asWkt() if clip is not None
                                  and not clip.isEmpty() else None),
                        run_id=run_id,
                    )
                except Exception as e:  # noqa: BLE001
                    QgsMessageLog.logMessage(
                        f"Detection history skipped: {e}",
                        "AI Segmentation", level=Qgis.MessageLevel.Info)

            if extent is None:
                _store(None)
            else:
                self._render_history_thumbnail(rect, crs, result_layer, _store)
        except Exception as e:  # noqa: BLE001
            QgsMessageLog.logMessage(
                f"Detection history skipped: {e}",
                "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _render_history_thumbnail(
        self,
        rect,
        crs: QgsCoordinateReferenceSystem,
        result_layer: QgsVectorLayer,
        on_done,
    ) -> None:















        try:
            from qgis.core import (
                QgsMapRendererSequentialJob,
                QgsMapSettings,
                QgsRectangle,
            )
            from qgis.PyQt.QtCore import QSize

            from ...core import detection_history

            if rect.width() <= 0 or rect.height() <= 0:
                on_done(None)
                return
            canvas = self.iface.mapCanvas()
            account_dir = detection_history.account_history_dir()


            layers = [result_layer] + [
                lyr for lyr in canvas.layers()
                if lyr is not None and lyr.id() != result_layer.id()]
            padded = QgsRectangle(rect)
            padded.scale(1.1)
            try:
                from ...core.server_dials import dial_in_range
                width = int(dial_in_range(
                    "tuning.library.history_thumb_width_px", 256, 96, 512))
            except Exception:  # noqa: BLE001
                width = 256
            height = int(round(width * padded.height() / padded.width()))
            height = max(64, min(height, 512))
            settings = QgsMapSettings()
            settings.setLayers(layers)
            settings.setDestinationCrs(crs)
            settings.setTransformContext(
                QgsProject.instance().transformContext())
            settings.setExtent(padded)
            settings.setOutputSize(QSize(width, height))
            settings.setBackgroundColor(canvas.canvasColor())
            job = QgsMapRendererSequentialJob(settings)

            def _finished() -> None:
                name = None
                try:
                    image = job.renderedImage()
                    if (not image.isNull()
                            and detection_history.account_history_dir() == account_dir):
                        candidate = detection_history.new_thumb_filename()
                        path = os.path.join(
                            detection_history.history_dir(), candidate)
                        if image.save(path, "PNG"):
                            name = candidate
                except Exception as err:  # noqa: BLE001
                    QgsMessageLog.logMessage(
                        f"Detection thumbnail skipped: {err}",
                        "AI Segmentation", level=Qgis.MessageLevel.Info)
                if getattr(self, "_history_thumb_job", None) is job:
                    self._history_thumb_job = None
                on_done(name)

            job.finished.connect(_finished)


            self._cancel_history_thumbnail()
            self._history_thumb_job = job
            job.start()
        except Exception as e:  # noqa: BLE001
            QgsMessageLog.logMessage(
                f"Detection thumbnail skipped: {e}",
                "AI Segmentation", level=Qgis.MessageLevel.Info)
            on_done(None)

    def _cancel_history_thumbnail(self) -> None:





        job = getattr(self, "_history_thumb_job", None)
        self._history_thumb_job = None
        if job is None:
            return
        try:
            job.cancel()
        except (RuntimeError, AttributeError):
            pass


def _release_run_policy() -> None:

    try:
        from ...core.detection_policy_core import release_run_policy

        release_run_policy()
    except Exception:  # noqa: BLE001  # nosec B110
        pass
