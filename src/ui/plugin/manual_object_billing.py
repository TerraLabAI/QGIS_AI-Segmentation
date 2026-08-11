



















from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog

from ...core.i18n import tr


class ManualObjectBillingMixin:




    def _start_manual_credit_session(self) -> None:










        self._manual_credit_ledger = None


        self._manual_cloud_objects_charged = 0
        self._tell_dock_manual_spend(0)



        if not self._cloud_correct_predictor_active():
            return
        try:
            from ...core.manual_object_credit import ManualObjectLedger

            self._manual_credit_ledger = ManualObjectLedger()
        except Exception as err:  # noqa: BLE001
            QgsMessageLog.logMessage(
                f"Semi-Auto: the object ledger did not open ({err})",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return



        try:
            setter = getattr(getattr(self, "predictor", None), "set_session_id", None)
            if setter is not None:
                setter(self._manual_credit_ledger.session_id)
        except Exception:  # noqa: BLE001  # nosec B110
            pass



        try:
            self._refresh_auto_credits()
        except (RuntimeError, AttributeError):
            pass

    def _end_manual_credit_session(self) -> None:



        self._manual_credit_ledger = None

    def _note_manual_cloud_answer(self) -> None:


        ledger = getattr(self, "_manual_credit_ledger", None)
        if ledger is not None:
            ledger.note_remote_answer()



    def _manual_credit_balance(self):










        try:
            dock = self.dock_widget
            envelopes = dock.quota_envelopes() if dock is not None else None
            if envelopes is not None and envelopes.objects_remaining is not None:
                return envelopes.objects_remaining
        except (RuntimeError, AttributeError):
            pass
        try:
            from ...core.credit_gate import credit_snapshot

            balance, _is_free = credit_snapshot(getattr(self, "_last_usage", None) or {})
            return balance
        except Exception:  # noqa: BLE001
            return None

    def _manual_save_is_billable(self, det_id) -> bool:


















        ledger = getattr(self, "_manual_credit_ledger", None)
        if ledger is None:
            return False
        reworks_existing = bool(getattr(self, "_active_refine_origin_entry", None))
        adding = bool(getattr(self, "_refine_add_mode_active", False))
        if reworks_existing and not adding:
            return False
        try:
            return bool(ledger.object_is_billable(det_id))
        except Exception:  # noqa: BLE001
            return False

    def _manual_save_refused_for_credits(self, det_id) -> bool:







        if not self._manual_save_is_billable(det_id):
            return False
        from ...core.manual_object_credit import save_affordable

        if save_affordable(self._manual_credit_balance()):
            return False



        try:
            self._refresh_auto_credits()
        except (RuntimeError, AttributeError):
            pass
        self._track_manual_charge("exhausted", error_code="balance_empty")



        if getattr(self, "_refine_handoff_active", False) and \
                self._degrade_correct_ai_to_manual(
                    "balance empty", failure_class="CREDITS_EXHAUSTED"):
            return True
        self._say_manual_credits_exhausted()
        return True

    def _track_manual_charge(self, outcome: str, error_code: str = "") -> None:


        try:
            from ...core import telemetry_session_events

            telemetry_session_events.track_manual_object_charged(
                outcome=outcome,
                objects_charged=int(getattr(self, "_manual_cloud_objects_charged", 0)),
                error_code=error_code,
            )
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _say_manual_credits_exhausted(self) -> None:







        try:
            self.dock_widget._update_full_ui()
        except (RuntimeError, AttributeError):
            pass  # nosec B110
        try:
            from ...core.server_dials import dial_in_range

            notice_duration_s = dial_in_range(
                "tuning.credits.exhausted_notice_duration_s", 8, 1, 30)
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("You saved your cloud objects for this month. Switch "
                   "to your own computer to keep working free, or upgrade "
                   "from the panel."),
                level=Qgis.MessageLevel.Warning,
                duration=notice_duration_s,
            )
        except (RuntimeError, AttributeError):
            pass





    _CHARGE_WKT_MAX_CHARS = 50_000

    def _manual_charge_extras(self, det_id, geom=None, crs_authid=None) -> dict:











        try:
            import math



            if getattr(self, "_is_non_georeferenced_mode", False):
                self._say_charge_has_no_surface(
                    "the session is not georeferenced")
                return {}
            if geom is None:
                geom, entry_crs = self._saved_polygon_for_charge(det_id)
                if not crs_authid:
                    crs_authid = entry_crs
            if geom is None or geom.isEmpty():
                self._say_charge_has_no_surface("the saved object has no shape")
                return {}
            crs = self._manual_charge_crs(crs_authid)
            if crs is None:
                self._say_charge_has_no_surface(
                    "the session CRS did not resolve")
                return {}
            from ...core.layer_conventions import make_area_measurer

            extras: dict = {}
            area = float(make_area_measurer(crs).measureArea(geom))
            if math.isfinite(area) and area > 0:
                extras["area_m2"] = round(area, 1)
            else:
                self._say_charge_has_no_surface(
                    f"the measured area is {area}")
            wkt = self._polygon_wgs84_wkt(geom, crs)
            if not wkt:
                self._say_charge_has_no_surface(
                    "the outline did not reach EPSG:4326")
            elif len(wkt) > self._CHARGE_WKT_MAX_CHARS:
                self._say_charge_has_no_surface(
                    f"the outline is {len(wkt)} characters, over the "
                    f"{self._CHARGE_WKT_MAX_CHARS} a charge carries")
            else:
                extras["polygon_wkt"] = wkt
            return extras
        except Exception as err:  # noqa: BLE001
            self._say_charge_has_no_surface(f"they could not be built ({err})")
            return {}

    def _say_charge_has_no_surface(self, reason: str) -> None:

        try:
            QgsMessageLog.logMessage(
                f"Semi-Auto: the saved object carries no ground surface, {reason}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        except Exception:  # noqa: BLE001
            pass  # nosec B110

    def _manual_charge_crs(self, crs_authid=None):






        from qgis.core import QgsCoordinateReferenceSystem

        if not crs_authid:
            crs_authid = self._manual_charge_crs_authid()
        if crs_authid:
            crs = QgsCoordinateReferenceSystem(str(crs_authid))
            if crs.isValid():
                return crs
        try:
            layer = getattr(self, "_current_layer", None)
            if layer is not None and layer.crs().isValid():
                return layer.crs()
        except RuntimeError:
            return None
        return None

    def _saved_polygon_for_charge(self, det_id):


        from qgis.core import QgsGeometry

        for entry in reversed(getattr(self, "saved_polygons", None) or []):
            if entry.get("det_id") != det_id:
                continue
            geom = entry.get("geom_obj")
            if geom is None:
                wkt = entry.get("geometry_wkt")
                geom = QgsGeometry.fromWkt(wkt) if wkt else None
            transform_info = entry.get("transform_info") or {}
            return geom, transform_info.get("crs")
        return None, None

    def _manual_charge_crs_authid(self):


        transform_info = getattr(self, "current_transform_info", None) or {}
        value = transform_info.get("crs")
        if isinstance(value, str) and value.strip():
            return value
        try:
            layer = getattr(self, "_current_layer", None)
            if layer is not None and layer.crs().isValid():
                return layer.crs().authid() or None
        except RuntimeError:
            return None
        return None

    @staticmethod
    def _polygon_wgs84_wkt(geom, crs) -> str | None:


        from qgis.core import (
            QgsCoordinateReferenceSystem,
            QgsCoordinateTransform,
            QgsGeometry,
            QgsProject,
        )

        from ...core.qt_compat import geometry_op_succeeded

        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        out = QgsGeometry(geom)
        if crs.authid() != wgs84.authid():
            transform = QgsCoordinateTransform(
                crs, wgs84, QgsProject.instance().transformContext())
            if not geometry_op_succeeded(out.transform(transform)):
                return None

        return out.asWkt(7) or None

    def _charge_manual_saved_object(self, det_id, geom=None,
                                    crs_authid=None) -> None:











        ledger = getattr(self, "_manual_credit_ledger", None)
        if ledger is None or not self._manual_save_is_billable(det_id):
            return
        try:
            from ...core.activation_manager import get_auth_header



            auth = get_auth_header()
        except Exception:  # noqa: BLE001
            auth = None
        if not auth:
            return
        try:
            from qgis.core import QgsApplication

            from ...api.terralab_client import TerraLabClient
            from ...workers.generic_request_task import GenericRequestTask

            client = TerraLabClient()
            session_id = ledger.session_id
            index = ledger.wire_index(det_id)


            extras = self._manual_charge_extras(
                det_id, geom=geom, crs_authid=crs_authid)
            task = GenericRequestTask(
                tr("Saving object"),
                lambda: client.charge_saved_object(
                    auth, session_id, index,
                    area_m2=extras.get("area_m2"),
                    polygon_wkt=extras.get("polygon_wkt")),
                hidden=True,
            )
            task.succeeded.connect(
                lambda payload, oid=det_id: self._on_manual_charge_done(oid, payload))
            task.failed.connect(
                lambda message, code, oid=det_id:
                    self._on_manual_charge_failed(oid, message, code))
            self._manual_charge_tasks.append(task)
            QgsApplication.taskManager().addTask(task)
        except Exception as err:  # noqa: BLE001
            QgsMessageLog.logMessage(
                f"Semi-Auto: the object charge did not go out ({err})",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)

    def _tell_dock_manual_spend(self, saved: int) -> None:


        try:
            self.dock_widget.set_manual_cloud_session_spend(int(saved))
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _on_manual_charge_done(self, det_id, payload) -> None:

        self._forget_manual_charge_tasks()
        ledger = getattr(self, "_manual_credit_ledger", None)
        if ledger is not None:
            ledger.mark_charged(det_id)



            self._manual_cloud_objects_charged = int(
                getattr(self, "_manual_cloud_objects_charged", 0)) + 1
            self._tell_dock_manual_spend(self._manual_cloud_objects_charged)
        self._track_manual_charge("charged")
        try:





            import time as _time
            now = _time.monotonic()
            self._envelopes_applied_at = now


            self._usage_applied_at = now
            self.dock_widget.note_cloud_object_charged()
        except (RuntimeError, AttributeError):
            pass
        usage = (payload or {}).get("usage") if isinstance(payload, dict) else None
        if isinstance(usage, dict) and usage:
            try:
                self._apply_usage_payload(usage)
            except (RuntimeError, AttributeError):
                pass

    def _on_manual_charge_failed(self, det_id, message: str, code: str) -> None:








        from ...core.error_policy import EXHAUSTED_CODES, RUN_FATAL_CODES
        self._forget_manual_charge_tasks()
        exhausted = code in EXHAUSTED_CODES
        QgsMessageLog.logMessage(
            f"Semi-Auto: object charge refused ({code or 'unknown'})",
            "AI Segmentation",
            level=Qgis.MessageLevel.Warning if exhausted else Qgis.MessageLevel.Info)
        self._track_manual_charge("exhausted" if exhausted else "refused",
                                  error_code=code or "unknown")
        if not exhausted:
            if code in RUN_FATAL_CODES:



                try:
                    self._on_key_revalidate_failed(message, code)
                except (RuntimeError, AttributeError):
                    pass
                self._end_cloud_click_session()
            return
        try:
            self.dock_widget.note_cloud_objects_exhausted()
            import time as _time
            self._envelopes_applied_at = _time.monotonic()
        except (RuntimeError, AttributeError):
            pass



        if getattr(self, "_refine_handoff_active", False):
            self._end_manual_credit_session()
            if self._degrade_correct_ai_to_manual(
                    code or "credits exhausted", failure_class="CREDITS_EXHAUSTED"):
                try:
                    self._refresh_auto_credits()
                except (RuntimeError, AttributeError):
                    pass
                return
        try:


            self._end_cloud_click_session(say_signed_out=False)
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        self._end_manual_credit_session()
        self._say_manual_credits_exhausted()
        try:
            self._refresh_auto_credits()
        except (RuntimeError, AttributeError):
            pass

    def _cancel_manual_charge_tasks(self) -> None:







        from ...core.qt_compat import safe_disconnect

        for task in getattr(self, "_manual_charge_tasks", []) or []:
            safe_disconnect(task, "succeeded")
            safe_disconnect(task, "failed")
            try:
                if task.is_active():
                    task.cancel()
            except Exception:  # nosec B110
                pass
        self._manual_charge_tasks = []

    def _forget_manual_charge_tasks(self) -> None:





        try:
            self._manual_charge_tasks = [
                task for task in getattr(self, "_manual_charge_tasks", [])
                if task.is_active()
            ]
        except (RuntimeError, AttributeError):
            self._manual_charge_tasks = []
