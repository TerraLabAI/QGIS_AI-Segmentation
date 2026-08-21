"""Spending one credit for one object saved in Semi-Auto on TerraLab's servers.

The rule is in ``core/manual_object_credit.py``; what is here is the wiring:
when a session opens a ledger, when a Save is refused for want of credits, and
when the charge goes out.

Three properties this file exists to keep true:

- **A session answered on the user's own machine never reaches any of it.** No
  ledger, no gate, no request. The offline mode stays free and unlimited.
- **The charge follows the save, it never blocks it.** The request goes out on a
  background task, so the polygon lands on the canvas at the speed it always
  did. The gate that CAN refuse runs before the save and reads a balance the
  plugin already holds, so it costs no network either.
- **The server's answer is what moves the credit ring.** Nothing here subtracts
  one locally: a figure the user reads has to be one the account agrees with.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); methods here are
plain mixin members and state lives on the plugin instance.
"""
from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog

from ...core.i18n import tr


class ManualObjectBillingMixin:
    """Opens the session ledger, gates a Save, and sends the charge."""

    # -- session lifetime ---------------------------------------------------

    def _start_manual_credit_session(self) -> None:
        """Open a ledger for a session whose clicks travel.

        Called at Start, after the route is decided. A session that stays on the
        machine gets no ledger at all, which is what makes every question below
        answer "free" without a single check of its own.

        Two kinds of session reach here, and both pay the same way: a Semi-Auto
        session on Cloud AI, and one fix in the Automatic review's Correct step,
        whose AI lane is answered off the machine and has no on-device side.
        """
        self._manual_credit_ledger = None
        # The engine row counts what THIS session spent, so it starts at zero
        # even when the account balance did not move.
        self._manual_cloud_objects_charged = 0
        self._tell_dock_manual_spend(0)
        # Asks the predictor in hand, not the mode: the Semi-Auto check is the
        # narrower of the two and would refuse the bare remote predictor the
        # Correct lane holds.
        if not self._cloud_correct_predictor_active():
            return
        try:
            from ...core.manual_object_credit import ManualObjectLedger

            self._manual_credit_ledger = ManualObjectLedger()
        except Exception as err:  # noqa: BLE001 -- no ledger means no billing
            QgsMessageLog.logMessage(
                f"Semi-Auto: the object ledger did not open ({err})",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return
        # The gate below judges against the balance the plugin already holds, so
        # a session that starts on a figure from an hour ago would refuse a Save
        # the account can afford. One read at Start, off the GUI thread.
        try:
            self._refresh_auto_credits()
        except (RuntimeError, AttributeError):
            pass

    def _end_manual_credit_session(self) -> None:
        """Close the ledger. What it recorded is spent and does not carry over:
        a new session is a new key, and an object saved again in it is a new
        object as far as the account is concerned."""
        self._manual_credit_ledger = None

    def _note_manual_cloud_answer(self) -> None:
        """One click came back from the network. Runs on the thread that
        answered the click, so it writes one bool and nothing else."""
        ledger = getattr(self, "_manual_credit_ledger", None)
        if ledger is not None:
            ledger.note_remote_answer()

    # -- the gate -----------------------------------------------------------

    def _manual_credit_balance(self):
        """Credits this account has left, or ``None`` when it is not known.

        Read from what the plugin already holds, never from the network: this
        answers inside a Save handler, and a Save may not wait on a request.

        The objects envelope answers first, exactly as the gate card and the
        engine card do. A save spends objects, and the wallet figure below can
        be the km2 gauge counted in tiles, which would refuse a save while
        objects remain.
        """
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
        except Exception:  # noqa: BLE001 -- an unreadable balance is unknown
            return None

    def _manual_save_is_billable(self, det_id) -> bool:
        """Whether saving this object right now would spend a credit.

        ADDING an object costs one. CORRECTING an object that already exists
        costs nothing, in Semi-Auto and in the Automatic review alike. That is
        the rule the product is sold on, and the public copy says it in twelve
        languages, so this function is where it has to be true.

        Between v2.4.0 and now the gate above widened to cover the review's
        remote predictor, which was right for the Add lane and wrong for the
        Correct lane: it started charging a reshape of a detection the user had
        already paid for once as a tile.

        Two conditions, not one. `_active_refine_origin_entry` says this save
        reworks a shape that was already on the map. `_refine_add_mode_active`
        says the Add lane is armed, and an Add keep can weld itself into an
        overlapping detection and inherit that entry, so the origin entry alone
        would hand a genuine add away for free.
        """
        ledger = getattr(self, "_manual_credit_ledger", None)
        if ledger is None:
            return False
        reworks_existing = bool(getattr(self, "_active_refine_origin_entry", None))
        adding = bool(getattr(self, "_refine_add_mode_active", False))
        if reworks_existing and not adding:
            return False
        try:
            return bool(ledger.object_is_billable(det_id))
        except Exception:  # noqa: BLE001 -- an unreadable ledger bills nothing
            return False

    def _manual_save_refused_for_credits(self, det_id) -> bool:
        """True when this Save must not happen because the balance is empty.

        The shape on screen is left exactly where it is. The user has three ways
        on from here and the line below names two of them: put the clicks back
        on their own computer, or add credits. The third is to do nothing, and
        nothing is lost by it.
        """
        if not self._manual_save_is_billable(det_id):
            return False
        from ...core.manual_object_credit import save_affordable

        if save_affordable(self._manual_credit_balance()):
            return False
        # The figure this refusal stands on can be minutes old, and the usual
        # reason it is wrong is the user having just paid. Read it again so the
        # next attempt is judged on a fresh one.
        try:
            self._refresh_auto_credits()
        except (RuntimeError, AttributeError):
            pass
        self._track_manual_charge("exhausted", error_code="balance_empty")
        # The review's Correct step has a free way on that Semi-Auto does not:
        # the Manual lane beside it. Hand the step over and say so once, rather
        # than leaving the user on a lane that can no longer answer.
        if getattr(self, "_refine_handoff_active", False) and \
                self._degrade_correct_ai_to_manual(
                    "balance empty", failure_class="CREDITS_EXHAUSTED"):
            return True
        self._say_manual_credits_exhausted()
        return True

    def _track_manual_charge(self, outcome: str, error_code: str = "") -> None:
        """Report what the account did with one object's credit. Best effort:
        a counter is never a reason for a save to fail."""
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
        """Refuse the save, in the panel and once in the message bar.

        The panel is what carries it: the refusal box names the price and the
        offer under it names the two ways on. The message bar line stays as the
        thing that catches an eye already on the map, where the click was.
        Never a dialog: the shape is still there and nothing is lost.
        """
        try:
            self.dock_widget._update_full_ui()
        except (RuntimeError, AttributeError):
            pass  # nosec B110 -- no dock, or one being torn down
        try:
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("You saved your cloud objects for this month. Switch "
                   "to your own computer to keep working free, or upgrade "
                   "from the panel."),
                level=Qgis.MessageLevel.Warning,
                duration=8,
            )
        except (RuntimeError, AttributeError):
            pass

    # -- the charge ---------------------------------------------------------

    # A saved outline above this many WKT characters stays out of the charge
    # body; the area still travels. Keeps the request small.
    _CHARGE_WKT_MAX_CHARS = 50_000

    def _manual_charge_extras(self, det_id, geom=None, crs_authid=None) -> dict:
        """Optional ground-surface facts for one charged object: its geodesic
        area in m2 and its outline as EPSG:4326 WKT.

        Informational only, and strictly best-effort: any failure answers {}
        and the charge goes out exactly as before. A caller that holds the
        saved shape passes it; otherwise the saved list is searched. GUI
        thread only (the area measurer reads the project)."""
        try:
            import math

            # A pixel-grid session sits on no ellipsoid, so neither figure
            # means anything there.
            if getattr(self, "_is_non_georeferenced_mode", False):
                return {}
            if geom is None:
                geom, entry_crs = self._saved_polygon_for_charge(det_id)
                if not crs_authid:
                    crs_authid = entry_crs
            if geom is None or geom.isEmpty():
                return {}
            if not crs_authid:
                crs_authid = self._manual_charge_crs_authid()
            from qgis.core import QgsCoordinateReferenceSystem

            crs = (QgsCoordinateReferenceSystem(str(crs_authid))
                   if crs_authid else None)
            if crs is None or not crs.isValid():
                return {}
            from ...core.layer_conventions import make_area_measurer

            extras: dict = {}
            area = float(make_area_measurer(crs).measureArea(geom))
            if math.isfinite(area) and area > 0:
                extras["area_m2"] = round(area, 1)
            wkt = self._polygon_wgs84_wkt(geom, crs)
            if wkt and len(wkt) <= self._CHARGE_WKT_MAX_CHARS:
                extras["polygon_wkt"] = wkt
            return extras
        except Exception:  # noqa: BLE001 -- extras never touch the charge
            return {}

    def _saved_polygon_for_charge(self, det_id):
        """(geometry, crs_authid) of the newest saved entry carrying this id,
        or (None, None). Reads the same list the Save wrote to."""
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
        """The session's raster CRS authid, or None. Same lookup order the
        Manual export uses: the live transform info, then the layer itself."""
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
        """The polygon as EPSG:4326 WKT, or None when it cannot be brought
        there. Works on a copy: the saved geometry is never moved."""
        from qgis.core import (
            QgsCoordinateReferenceSystem,
            QgsCoordinateTransform,
            QgsGeometry,
            QgsProject,
        )

        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        out = QgsGeometry(geom)
        if crs.authid() != wgs84.authid():
            transform = QgsCoordinateTransform(
                crs, wgs84, QgsProject.instance().transformContext())
            if int(out.transform(transform)) != 0:
                return None
        # 7 decimals of a degree resolve to about a centimetre.
        return out.asWkt(7) or None

    def _charge_manual_saved_object(self, det_id, geom=None,
                                    crs_authid=None) -> None:
        """Send the credit for an object that has just been saved.

        Fire and forget by design: the polygon is already on the canvas, and a
        user waiting on a request to see their own shape is the thing this mode
        exists to avoid. A charge that never lands leaves the object billable,
        so the next Save of it tries again.

        ``geom`` and ``crs_authid`` are optional: a caller whose object never
        reached the saved list passes the shape so the charge can carry its
        ground surface. They change nothing about what is charged or deduped.
        """
        ledger = getattr(self, "_manual_credit_ledger", None)
        if ledger is None or not self._manual_save_is_billable(det_id):
            return
        try:
            from ...core.activation_manager import get_auth_header

            # Read on the GUI thread. The key store is not something to reach
            # for from the task below.
            auth = get_auth_header()
        except Exception:  # noqa: BLE001 -- no key, no charge
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
            # Computed here on the GUI thread; {} on any failure, and the
            # charge below is the same request either way.
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
        except Exception as err:  # noqa: BLE001 -- a charge never breaks a save
            QgsMessageLog.logMessage(
                f"Semi-Auto: the object charge did not go out ({err})",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)

    def _tell_dock_manual_spend(self, saved: int) -> None:
        """Push the session's billed count to the engine row. Best-effort: a
        counter is never a reason for a save to fail."""
        try:
            self.dock_widget.set_manual_cloud_session_spend(int(saved))
        except (RuntimeError, AttributeError):
            pass  # nosec B110 -- no dock, or one being torn down

    def _on_manual_charge_done(self, det_id, payload) -> None:
        """The account took the credit. Record it and show the new balance."""
        self._forget_manual_charge_tasks()
        ledger = getattr(self, "_manual_credit_ledger", None)
        if ledger is not None:
            ledger.mark_charged(det_id)
            # Count the charges the account CONFIRMED, never the saves we sent:
            # the row is a spend figure, and a figure the user reads has to be
            # one the account agrees with.
            self._manual_cloud_objects_charged = int(
                getattr(self, "_manual_cloud_objects_charged", 0)) + 1
            self._tell_dock_manual_spend(self._manual_cloud_objects_charged)
        self._track_manual_charge("charged")
        try:
            # The save response only carries the wallet gauge; move the
            # objects envelope locally so the count the user watches moves
            # with the save. The next account read corrects it. The stamp
            # makes this move count as now, so an account read already in
            # flight cannot land later and put the spent object back.
            self.dock_widget.note_cloud_object_charged()
            import time as _time
            self._envelopes_applied_at = _time.monotonic()
        except (RuntimeError, AttributeError):
            pass
        usage = (payload or {}).get("usage") if isinstance(payload, dict) else None
        if isinstance(usage, dict) and usage:
            try:
                self._apply_usage_payload(usage)
            except (RuntimeError, AttributeError):
                pass

    def _on_manual_charge_failed(self, det_id, message: str, code: str) -> None:
        """The charge did not stand.

        A network or server failure is left alone: the object stays billable and
        the user keeps their polygon. An account that is out of credits is the
        one case worth acting on, because every click after it would be work
        nobody is paying for. The session's clicks go back to the machine, and
        the user is told in one line.
        """
        self._forget_manual_charge_tasks()
        exhausted = code in ("FREE_DETECTIONS_EXHAUSTED", "QUOTA_EXCEEDED",
                             "TRIAL_EXHAUSTED")
        QgsMessageLog.logMessage(
            f"Semi-Auto: object charge refused ({code or 'unknown'})",
            "AI Segmentation",
            level=Qgis.MessageLevel.Warning if exhausted else Qgis.MessageLevel.Info)
        self._track_manual_charge("exhausted" if exhausted else "refused",
                                  error_code=code or "unknown")
        if not exhausted:
            return
        try:
            self.dock_widget.note_cloud_objects_exhausted()
            import time as _time
            self._envelopes_applied_at = _time.monotonic()
        except (RuntimeError, AttributeError):
            pass
        # A Correct fix hands its step to the Manual lane. Semi-Auto has no such
        # lane, so it ends the cloud session and puts the clicks back on the
        # machine. Same ending, two different ways there.
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
            self._end_cloud_click_session()
        except Exception:  # noqa: BLE001 -- the session stands either way  # nosec B110
            pass
        self._end_manual_credit_session()
        self._say_manual_credits_exhausted()
        try:
            self._refresh_auto_credits()
        except (RuntimeError, AttributeError):
            pass

    def _cancel_manual_charge_tasks(self) -> None:
        """Teardown: let go of every charge still out.

        Signals first, so a result landing after the controller has come apart
        cannot reach a torn-down dock. The task itself is only asked to stop and
        is never terminated: the manager drains a network-bound task on its own,
        and killing one wedged in a socket crashes QGIS.
        """
        from ...core.qt_compat import safe_disconnect

        for task in getattr(self, "_manual_charge_tasks", []) or []:
            safe_disconnect(task, "succeeded")
            safe_disconnect(task, "failed")
            try:
                if task.is_active():
                    task.cancel()
            except Exception:  # nosec B110 -- teardown must never raise
                pass
        self._manual_charge_tasks = []

    def _forget_manual_charge_tasks(self) -> None:
        """Drop the references to charges that have finished.

        The plugin holds them only so a task is not garbage-collected while the
        task manager still owns it. Anything no longer running is free to go.
        """
        try:
            self._manual_charge_tasks = [
                task for task in getattr(self, "_manual_charge_tasks", [])
                if task.is_active()
            ]
        except (RuntimeError, AttributeError):
            self._manual_charge_tasks = []
