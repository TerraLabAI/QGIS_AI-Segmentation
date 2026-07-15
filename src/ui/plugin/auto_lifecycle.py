"""Run wind-down: autosave/discard, worker signal handlers, detection export.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

import os

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsField,
    QgsMessageLog,
    QgsProject,
    QgsVectorLayer,
)

from ...core.i18n import tr
from .shared import (
    _FIELD_TYPE_DOUBLE,
    _FIELD_TYPE_STRING,
    _add_features_fast,
    _apply_fast_render,
    park_orphaned_worker,
)


class AutoLifecycleMixin:
    """Run wind-down: autosave/discard, worker signal handlers, detection export."""

    def _autosave_pending_auto_review(self, exit_path: str = "other") -> None:
        """Persist a still-pending review to a layer so a paid detection is never
        lost when the user leaves the flow WITHOUT clicking Finish (Exit, mode
        switch, new run, dropping the zone, or quitting QGIS). A detection costs
        credits, so it must survive any exit, not only the happy path.

        No-op when there is nothing uncommitted (Finish already cleared it).
        Never raises: it runs on teardown paths (including unload) where a
        failure must not break the rest of the cleanup. _export_auto_review()
        clears the review and removes the selection layer on success.

        Safety net: exports with include_hidden so detections the Confidence
        cutoff is hiding are saved too (the visible set may be empty), since the
        user never got to review them before leaving.
        """
        try:
            review = self._auto_review
            # The visible set (review["geoms"]) may be EMPTY when Confidence is
            # hiding billed detections, but those must still not be lost: gate on
            # whether the run found any objects at all, then export the full
            # found set (include_hidden) below.
            if not (review and self._auto_objects):
                return
            # The user is leaving without Finish: record the abandonment (with
            # how) BEFORE the rescue export clears the review state.
            self._track_review_abandoned(exit_path)
            exported = self._export_auto_review(include_hidden=True, autosave=True)
            if exported and exported[1] > 0 and self.dock_widget:
                name, count = exported
                try:
                    self.dock_widget.set_auto_status(
                        "info",
                        tr("Saved {n} polygon(s) to {name}").format(
                            n=count, name=name or ""))
                except (RuntimeError, AttributeError):
                    pass
        except Exception:  # nosec B110 -- teardown safety: never propagate
            try:
                QgsMessageLog.logMessage(
                    "Auto review autosave failed", "AI Segmentation",
                    level=Qgis.MessageLevel.Warning)
            except Exception:  # nosec B110
                pass

    def _discard_auto_review(self, exit_path: str = "other") -> None:
        """Auto-save then clear the post-run review state.

        Called when a new run starts, the zone is dropped, the mode changes,
        Exit is clicked, or the plugin is unloaded. The detection is billed, so
        we never silently throw it away: any still-pending review is committed
        to a layer first (see _autosave_pending_auto_review), then the in-memory
        review state is cleared. ``exit_path`` names the leave path for the
        abandonment telemetry.
        """
        self._autosave_pending_auto_review(exit_path)
        self._auto_review = None
        # Supersede any in-flight cooperative finalize/reslice (same gen-bump +
        # state-null as _reset_auto_live_pipeline): every caller here is leaving
        # the run (Exit, zone drop, mode switch, new run, unload), so a late
        # _complete_auto_finalize must not resurrect a review over the reset flow.
        self._auto_finalize_gen += 1
        self._auto_finalize_state = None
        self._remove_auto_selection_layer()
        self._auto_protected_geoms = []
        self._auto_manual_removed = set()
        # Orphan any background install started from the review (D1): a late
        # predictor load must not open a handoff on a review that is gone.
        self._clear_refine_install_pending()
        if self.dock_widget:
            try:
                self.dock_widget.set_protected_note(False)
                self.dock_widget.set_auto_review_active(False)
            except (RuntimeError, AttributeError):
                pass

    def _on_auto_progress(self, completed: int, total: int) -> None:
        """Slot: update progress bar in dock."""
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_tile_progress(completed, total)
            except (RuntimeError, AttributeError):
                pass

    def _on_auto_queue_state(self, position: int, depth: int, eta_s: int) -> None:
        """Slot: the worker hit (or left) the server's waiting room. Flips the
        progress bar's label to an honest "you're in line" state so a launch
        spike never reads as a frozen run; (0, 0, 0) restores the tile count.

        Also accumulates the wall time spent waiting, reported as warming_ms on
        the run's terminal telemetry: the server only sees per-request timing,
        never the user-perceived wait across the whole waiting-room stretch."""
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
        """Total waiting-room wall time for the current run, closing any still
        open stretch (a run can end while the warming banner is up)."""
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
            "Auto detection warning: {}".format(msg),
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
        # Aggregate degraded-tile warnings for the auto_tiles_degraded event
        # emitted once per run at finalize (timeouts vs other skips).
        low = (msg or "").lower()
        if "timeout" in low or "timed out" in low:
            self._auto_timeout_tiles += 1
        elif "skip" in low:
            self._auto_skipped_tiles += 1

    def _on_auto_error(self, msg: str) -> None:
        QgsMessageLog.logMessage(
            "Auto detection error: {}".format(msg),
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
        self._set_zone_badge_enabled(True)
        # The run start dropped the zone fill (live/review visual parity); a
        # failed run lands back on the prompt step, where the kept zone should
        # read plainly again.
        self._set_zone_band_fill_visible(True)
        # T20: an expired/invalid session gets its own copy + the sign-in
        # surface, and the billed partial results are salvaged into the review
        # (mirrors the credits-exhausted tail) instead of being dropped.
        error_class = self._classify_auto_error(msg)
        is_auth = error_class == "AUTH"
        # Build the user-facing banner once so it can be reused for a persistent
        # message-bar warning when partial results open the review (opening the
        # review swaps the dock status line to idle, wiping this banner).
        # Be honest about whose fault it is: only a real connectivity failure
        # blames the user's connection. A server or timeout is ours, so we say so
        # and point at the right next step.
        if is_auth:
            banner = tr("Session expired. Sign in again to continue.")
        elif error_class == "SERVER":
            banner = tr("The detection service had a problem. Your credits "
                        "for the failed tiles were refunded. Please try again.")
        elif error_class == "TIMEOUT":
            banner = tr("The detection service is busy right now. "
                        "Please try again in a moment.")
        elif error_class == "NETWORK":
            banner = tr("Detection failed. Check your connection and try again.")
        else:
            banner = tr("Detection failed. Please try again.")
        # Our-fault failures (server/timeout/unknown) get a short support code the
        # user can quote: it is the head of the run id the server archives this
        # run under, so support can match the report to the exact run.
        code = (self._auto_run_id or "")[:8]
        if code and error_class in ("SERVER", "TIMEOUT", "UNKNOWN"):
            banner = banner + "\n" + tr("Support code: {code}").format(code=code)
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_run_active(False)
                self.dock_widget.set_auto_status("error", banner)
            except (RuntimeError, AttributeError):
                pass
        # Record result for MCP/headless callers waiting on QEventLoop.
        self._last_auto_result = {"status": "error", "message": msg}
        self._auto_tel_stop_reason = "error"
        try:
            from ...core import telemetry
            telemetry.track_auto_detect_failed(
                run_id=self._auto_run_id or "",
                error_class=error_class,
                tiles_done=getattr(self._auto_worker, "tiles_succeeded", 0),
                duration_ms=self._auto_duration_ms(),
                warming_ms=self._auto_warming_wait_ms(),
            )
            # Pro-path failures that are OUR fault (server/timeout/unknown, not
            # a user NETWORK/AUTH/CANCELLED) also emit a plugin_error so the
            # existing server-side error spike/persistent detectors cover the
            # paid path, not just the Manual segmentation_run event.
            if error_class in ("SERVER", "TIMEOUT", "UNKNOWN"):
                telemetry.track_plugin_error(
                    stage="segment",
                    error_code="auto_detect_" + error_class.lower(),
                    message=msg,
                )
        except Exception:
            pass  # nosec B110
        worker = self._auto_worker
        # A terminal signal fires from inside the worker's run loop, which may
        # not have returned yet: dropping the last ref to a still-running QThread
        # aborts QGIS, so park a strong ref (released on finished) first.
        if worker is not None and worker.isRunning():
            park_orphaned_worker(worker)
        self._auto_worker = None
        self._drop_auto_tile_bridge()
        self._capture_auto_mask_gsd(worker)
        tiles_succeeded = getattr(worker, "tiles_succeeded", 0)
        if tiles_succeeded > 0 and not self._auto_headless_run:
            # Billed partials survive ANY terminal error (auth, a non-retryable
            # tile code, an offline abort mid-run): route them into the review so
            # nothing already charged is dropped, mirroring the credits-exhausted,
            # user-cancel and auth paths. Only the account panel below stays
            # auth-specific.
            self._finalize_auto_results(tiles_succeeded)
            # Opening the review swaps the dock status banner to idle, wiping the
            # error line. Re-post it to the message bar (AUTH pushes its own
            # below) so a salvaged-into-review failure still explains what
            # happened and keeps the quotable support code on screen.
            if not is_auth:
                try:
                    self.iface.messageBar().pushWarning("AI Segmentation", banner)
                except (RuntimeError, AttributeError):
                    pass
        else:
            self._remove_auto_selection_layer()
            # Zero-success failure lands back on the prompt step with the zone
            # kept: the run cleared the tile-grid preview but the cost label still
            # shows the old estimate. Redraw the grid + cost (interactive only).
            if self.dock_widget and not self._auto_headless_run:
                try:
                    self._update_credit_estimate()
                except (RuntimeError, AttributeError):
                    pass
        if is_auth and not self._auto_headless_run:
            # Persist the message (the review wind-down clears the status
            # banner) and open the account panel so signing in is one click.
            try:
                self.iface.messageBar().pushWarning(
                    "AI Segmentation",
                    tr("Session expired. Sign in again to continue."))
            except (RuntimeError, AttributeError):
                pass
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(0, self._on_settings_clicked)

    def _on_auto_credits_exhausted(self, remaining: int) -> None:
        QgsMessageLog.logMessage(
            "Auto detection: credits exhausted (remaining={})".format(remaining),
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
        self._set_zone_badge_enabled(True)
        # The worker no longer emits cancelled after exhaustion. The billed
        # partial results must not vanish: route them straight into the review
        # so the user can refine and Finish them (the resume flow was removed).
        worker = self._auto_worker
        # The exhausted terminal fires from the worker's run loop, which may not
        # have returned yet: park a strong ref before dropping ours so a live
        # QThread is never garbage-collected (which aborts QGIS).
        if worker is not None and worker.isRunning():
            park_orphaned_worker(worker)
        self._auto_worker = None
        self._drop_auto_tile_bridge()
        self._capture_auto_mask_gsd(worker)
        tiles_succeeded = getattr(worker, "tiles_succeeded", 0)
        tiles_total = (self._auto_run_ctx or {}).get("total", tiles_succeeded)
        _, is_free_tier = self._auto_credit_snapshot()
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_run_active(False)
                # Tiles-only message: the review header carries the kept count.
                self.dock_widget.set_auto_status(
                    "info",
                    tr("Out of credits after {done}/{total} tiles. "
                       "Your detections are kept below.").format(
                        done=tiles_succeeded, total=tiles_total),
                )
                # Free users get a one-click path to finish the zone.
                self.dock_widget.set_auto_exhausted_subscribe_visible(is_free_tier)
            except (RuntimeError, AttributeError):
                pass
        try:
            from ...core import telemetry
            telemetry.track_credits_exhausted(
                run_id=self._auto_run_id or "",
                tiles_done=tiles_succeeded,
                tiles_total=tiles_total,
                is_free_tier=is_free_tier,
            )
        except Exception:
            pass  # nosec B110
        self._auto_tel_stop_reason = "exhausted"
        # Refresh the footer ring / upsell card with the real balance.
        self._refresh_auto_credits()
        # Record the exhausted result BEFORE finalize: the headless finalize
        # exports the billed partials and records a completed result carrying
        # the saved layer_name, and _complete_auto_finalize merges the two so
        # the MCP caller learns about BOTH the layer and the quota. Writing it
        # after finalize used to clobber the layer info, so the caller saw a
        # bare error and could orphan or retry already-charged work.
        self._last_auto_result = {"status": "credits_exhausted", "credits_remaining": remaining}
        self._finalize_auto_results(tiles_succeeded)

    def _on_auto_cancelled(self) -> None:
        worker = self._auto_worker
        if worker is None or self._auto_merger is None:
            # Hard stop (_stop_auto_detection): the run state was already torn
            # down (merger nulled, selection layer removed, UI restored), and
            # only the worker reference may still be alive so unload can join
            # the thread. Harvesting now would finalize an EMPTY merger and
            # post a stale "No detection in this zone." over the reset flow,
            # dropping any billed partials silently. Just release the refs.
            self._auto_worker = None
            self._drop_auto_tile_bridge()
            return
        # The cancelled signal fires from the worker's run loop, which may still
        # be winding down: park a strong ref before dropping ours so a live
        # QThread is never garbage-collected (which aborts QGIS).
        if worker is not None and worker.isRunning():
            park_orphaned_worker(worker)
        self._auto_worker = None
        self._drop_auto_tile_bridge()
        self._capture_auto_mask_gsd(worker)
        self._set_zone_badge_enabled(True)
        tiles_succeeded = getattr(worker, "tiles_succeeded", 0)
        tiles_total = (self._auto_run_ctx or {}).get("total", tiles_succeeded)
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_run_active(False)
                self.dock_widget.set_auto_status("idle")
            except (RuntimeError, AttributeError):
                pass
        try:
            from ...core import telemetry
            telemetry.track_auto_detect_cancelled(
                run_id=self._auto_run_id or "",
                tiles_done=tiles_succeeded,
                tiles_total=tiles_total,
                salvaged_to_review=tiles_succeeded > 0,
                duration_ms=self._auto_duration_ms(),
                warming_ms=self._auto_warming_wait_ms(),
            )
        except Exception:
            pass  # nosec B110
        self._auto_tel_stop_reason = "cancelled"
        # Keep whatever was found so far: drop the user into the review of the
        # billed partial results (the resume flow was removed).
        self._finalize_auto_results(tiles_succeeded)
        # Record result for MCP/headless callers waiting on QEventLoop.
        self._last_auto_result = {"status": "cancelled"}
        QgsMessageLog.logMessage(
            "Auto detection: cancelled",
            "AI Segmentation", level=Qgis.MessageLevel.Info,
        )

    def _on_layers_will_be_removed(self, layer_ids) -> None:
        """A layer is about to leave the project: end any flow that depends
        on it (wired to QgsProject.layersWillBeRemoved in initGui).

        T15: the Automatic source raster removed MID-RUN -> soft-cancel, so
        the billed partial results are salvaged into the review.
        T16: removed on the draw/prompt steps -> reset the flow to Start.
        T17: the Manual session raster removed -> end the session cleanly
        without the confirm dialog, saving the work first (never lost, I2).
        A pending review is layer-independent (geoms + CRS are stored), so it
        is deliberately left untouched.
        """
        try:
            ids = set(layer_ids or [])
        except TypeError:
            return
        if not ids:
            return
        dock = self.dock_widget
        # ---- Automatic flow (T15 / T16) ----
        run_layer_id = (self._auto_run_ctx or {}).get("layer_id")
        dock_mid_flow = dock is not None and getattr(dock, "_auto_started", False)
        no_active_auto_run = self._auto_worker is None and self._auto_review is None
        if self._auto_worker is not None and run_layer_id in ids:
            self._on_auto_cancel_clicked()
            msg = tr("The selected raster was removed. "
                     "Keeping what was already found.")
            if dock:
                try:
                    dock.set_auto_status("info", msg)
                except (RuntimeError, AttributeError):
                    pass
            try:
                self.iface.messageBar().pushInfo("AI Segmentation", msg)
            except (RuntimeError, AttributeError):
                pass
        elif dock_mid_flow and no_active_auto_run and not self._refine_handoff_active:
            # T16: the flow is on the draw/prompt steps; if the LOCKED raster
            # is the one leaving, the zone would dangle -> back to Start.
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
        # ---- Manual session (T17) ----
        if self._refine_handoff_active:
            return  # the handoff session is owned by the review round trip
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

    # ---- Auto detection export ----------------------------------------------

    def _export_auto_detections(
        self,
        deduped_geoms: list,
        crs: "QgsCoordinateReferenceSystem",
        source_layer_name: str,
        prompt_label: str,
        scores: list | None = None,
    ) -> str | None:
        """Write deduplicated geometries into the project GeoPackage as a new
        table and add it to the AI Segmentation group.

        Per-feature schema follows the detection-output convention (class
        identity + confidence score + measures; COCO/Deepness-style): the
        object class survives layer merges and shapefile round-trips where
        the layer name and QgsLayerMetadata are lost, and the score is the
        one per-feature fact a geomatician filters/audits on. Run-level
        provenance (raster, date, params) stays in the layer metadata, not
        repeated per row. ``scores`` is parallel to ``deduped_geoms`` (None,
        or None entries, when unknown, e.g. after a Manual-refine dissolve).

        Returns the (friendly) layer name of the created layer, or None on
        failure.
        """
        from datetime import datetime

        from ...core import output_store
        from ...core.layer_conventions import (
            apply_output_conventions,
            make_area_measurer,
            make_committed_renderer,
            repair_polygon,
            to_multipolygon,
        )

        if not deduped_geoms:
            return

        # Example-only runs have no text prompt: fall back to a stable name so the
        # class attribute, layer name, table name and colour are never empty. The
        # English token stays "Example match" (localized for display).
        prompt_label = (prompt_label or "").strip() or tr("Example match")

        # Build a temporary memory layer.
        temp_layer = QgsVectorLayer("MultiPolygon", "auto_export", "memory")
        if not temp_layer.isValid():
            return
        temp_layer.setCrs(crs)

        pr = temp_layer.dataProvider()
        pr.addAttributes([
            QgsField("label", _FIELD_TYPE_STRING),
            QgsField("class", _FIELD_TYPE_STRING),
            QgsField("score", _FIELD_TYPE_DOUBLE),
            QgsField("area_m2", _FIELD_TYPE_DOUBLE),
            QgsField("perimeter_m", _FIELD_TYPE_DOUBLE),
        ])
        temp_layer.updateFields()

        object_class = (prompt_label or "").strip()
        if scores is not None and len(scores) != len(deduped_geoms):
            scores = None  # misaligned parallel list: honest NULLs over mislabels

        # One measurer for the whole batch: setEllipsoid loads from the SRS DB, so
        # rebuilding it per feature cost seconds on a big run.
        measurer = make_area_measurer(crs)
        features_to_add = []
        for index, geom in enumerate(deduped_geoms):
            if geom is None or geom.isEmpty():
                continue
            geom = to_multipolygon(repair_polygon(geom) or geom)
            if geom is None or geom.isEmpty():
                continue
            score = scores[index] if scores is not None else None
            feat = QgsFeature(temp_layer.fields())
            feat.setGeometry(geom)
            feat.setAttributes([
                # label stays empty (the user's own annotation column, the
                # Deepness/SCP norm); the machine fact lives in `class`.
                "",
                object_class,
                round(float(score), 3) if score is not None else None,
                float(measurer.measureArea(geom)),
                float(measurer.measurePerimeter(geom)),
            ])
            features_to_add.append(feat)

        if not features_to_add:
            return

        _add_features_fast(pr, features_to_add)
        temp_layer.updateExtents()

        # The run's raster only steers the fallback output directory; the run
        # may outlive the layer (T15), so a missing layer is fine.
        source_layer = None
        run_layer_id = (self._auto_run_ctx or {}).get("layer_id")
        if run_layer_id:
            source_layer = QgsProject.instance().mapLayer(run_layer_id)
        if source_layer is None:
            source_layer = self._get_active_raster_layer()

        result = output_store.write_run_table(
            temp_layer,
            prompt=prompt_label,
            source_layer=source_layer,
            fallback_stem=prompt_label or "detection",
        )
        if result is None:
            return
        result_layer = result.layer
        layer_name = result_layer.name()

        if result.used_fallback:
            try:
                self.iface.messageBar().pushWarning(
                    "AI Segmentation",
                    tr("Could not write to {name}. Saved to a separate "
                       "file instead.").format(name=output_store.GPKG_FILENAME))
            except (RuntimeError, AttributeError):
                pass

        # Apply style and provenance, then add to project. One stable color
        # per object prompt = committed.
        result_layer.setRenderer(make_committed_renderer(
            color=output_store.committed_color_for_prompt(prompt_label)))
        # Smooth pan/zoom on a dense result: render-time simplification (the GPKG
        # already ships a spatial index from the OGR writer).
        _apply_fast_render(result_layer)
        try:
            plugin_version = self._read_plugin_version()
        except (RuntimeError, AttributeError):
            plugin_version = ""
        apply_output_conventions(
            result_layer, source_layer_name,
            prompt=prompt_label,
            detail=(self._auto_run_ctx or {}).get("detail"),
            confidence=self._auto_confidence,
            created_iso=datetime.now().astimezone().isoformat(timespec="seconds"),
            plugin_version=plugin_version,
        )
        result_layer.triggerRepaint()

        output_store.add_committed_layer(result_layer, source_name=source_layer_name)
        self.iface.mapCanvas().refresh()

        # Local run history for the library's Recent tab: prompt + zone extent
        # + layer name + thumbnail, so a recent card can bring the user back.
        self._record_detection_history(
            prompt_label, layer_name, len(features_to_add), crs, result_layer)

        QgsMessageLog.logMessage(
            "Auto detection: saved {} polygon(s) to {} (table {})".format(
                len(features_to_add), result.gpkg_path, result.table_name),
            "AI Segmentation", level=Qgis.MessageLevel.Info,
        )

        return layer_name

    # ---- local detection history (library Recent tab) -----------------------

    def _record_detection_history(
        self,
        prompt_label: str,
        layer_name: str,
        count: int,
        crs: "QgsCoordinateReferenceSystem",
        result_layer: "QgsVectorLayer",
    ) -> None:
        """Remember this committed run locally (prompt, zone extent + CRS,
        exported layer name, object count, thumbnail) so the Segment library's
        Recent tab can bring the user straight back to it.

        Local-only state (see core/detection_history.py), NEVER telemetry.
        Fail-safe: any problem is logged quietly and the export is untouched.
        """
        try:
            from ...core import detection_history

            rect = None
            clip = getattr(self, "_auto_clip_polygon", None)
            if clip is not None and not clip.isEmpty():
                rect = clip.boundingBox()  # zone polygon, already in run CRS
            if rect is None or rect.isEmpty():
                # Rectangle/MCP zones carry no clip polygon: the exported
                # layer's extent covers the same footprint.
                rect = result_layer.extent()
            extent = None
            thumb = None
            if rect is not None and not rect.isEmpty():
                extent = (rect.xMinimum(), rect.yMinimum(),
                          rect.xMaximum(), rect.yMaximum())
                thumb = self._render_history_thumbnail(rect, crs, result_layer)
            detection_history.add_entry(
                prompt=prompt_label,
                layer_name=layer_name,
                objects=count,
                extent=extent,
                crs_authid=crs.authid() if crs is not None else "",
                thumb=thumb,
            )
        except Exception as e:  # noqa: BLE001 -- history must never break Finish
            QgsMessageLog.logMessage(
                "Detection history skipped: {}".format(e),
                "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _render_history_thumbnail(
        self,
        rect,
        crs: "QgsCoordinateReferenceSystem",
        result_layer: "QgsVectorLayer",
    ) -> str | None:
        """Render a small PNG of the zone (padded ~10%) from the current canvas
        layers, one-shot on the main thread (a ~256px sequential render is a
        few tens of ms). Returns the saved filename inside the history dir, or
        None; never raises, a missing thumbnail just means the recent card
        shows the no-preview placeholder."""
        try:
            from qgis.core import (
                QgsMapRendererSequentialJob,
                QgsMapSettings,
                QgsRectangle,
            )
            from qgis.PyQt.QtCore import QSize

            from ...core import detection_history

            if rect.width() <= 0 or rect.height() <= 0:
                return None
            canvas = self.iface.mapCanvas()
            # The freshly exported layer on top of whatever the canvas shows,
            # so the thumbnail reads as "your detections on your imagery".
            layers = [result_layer] + [
                lyr for lyr in canvas.layers()
                if lyr is not None and lyr.id() != result_layer.id()]
            padded = QgsRectangle(rect)
            padded.scale(1.1)
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
            job.start()
            job.waitForFinished()
            image = job.renderedImage()
            if image.isNull():
                return None
            name = detection_history.new_thumb_filename()
            path = os.path.join(detection_history.history_dir(), name)
            if not image.save(path, "PNG"):
                return None
            return name
        except Exception as e:  # noqa: BLE001 -- the thumbnail is optional
            QgsMessageLog.logMessage(
                "Detection thumbnail skipped: {}".format(e),
                "AI Segmentation", level=Qgis.MessageLevel.Info)
            return None
