






from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsGeometry,
    QgsMessageLog,
)


class AutoReviewOpenMixin:


    def _run_export_crs(self, source_layer) -> QgsCoordinateReferenceSystem:









        authid = str(getattr(self, "_auto_crs_authid", None) or "").strip()
        if authid:
            crs = QgsCoordinateReferenceSystem(authid)
            if crs.isValid():
                return crs
        layer_crs = None
        try:
            if source_layer is not None:
                layer_crs = source_layer.crs()
        except (RuntimeError, AttributeError):
            layer_crs = None
        if layer_crs is not None and layer_crs.isValid():
            QgsMessageLog.logMessage(
                "Auto review: the run carries no usable CRS id; the export "
                "takes the source layer's CRS.",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return layer_crs
        QgsMessageLog.logMessage(
            "Auto review: no CRS could be resolved for the run; the export "
            "falls back to EPSG:4326 and may be misplaced.",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return QgsCoordinateReferenceSystem("EPSG:4326")

    def _complete_auto_finalize(self, visible: list, tiles_succeeded: int,
                                scores: list | None = None,
                                ids: list | None = None) -> None:






        from ...core import run_timeline
        run_timeline.mark("review_opening")

        source_layer = self._get_active_raster_layer()
        source_layer_name = ""
        try:
            if source_layer is not None:
                source_layer_name = source_layer.name()
        except (RuntimeError, AttributeError):
            pass
        crs = self._run_export_crs(source_layer)


        prompt_text = ""
        try:
            if self.dock_widget:
                prompt_text = self.dock_widget.auto_prompt_input.text().strip()
        except (RuntimeError, AttributeError):
            pass

        if self._auto_headless_run:


            if not visible:



                self._remove_auto_selection_layer()
                self._record_auto_zero_result(tiles_succeeded)
                return
            exported_layer_name = self._export_auto_detections(
                visible, crs, source_layer_name, prompt_text, scores=scores)
            if not exported_layer_name:


                self._auto_headless_run = False
                self._complete_auto_finalize(visible, tiles_succeeded, scores=scores, ids=ids)
                self._last_auto_result = {
                    "status": "error",
                    "message": "The detections could not be saved. They remain in the panel for export.",
                    "instances": len(visible),
                    "tiles_processed": tiles_succeeded,
                    "layer_name": None,
                }
                return
            self._remove_auto_selection_layer()
            result = {
                "status": "completed",
                "instances": len(visible),
                "tiles_processed": tiles_succeeded,
                "layer_name": exported_layer_name,
            }
            prior = self._last_auto_result
            if isinstance(prior, dict) and prior.get("status") == "credits_exhausted":




                result["status"] = "credits_exhausted"
                result["credits_remaining"] = prior.get("credits_remaining", 0)
            self._last_auto_result = result
            QgsMessageLog.logMessage(
                f"Auto detection: exported {len(visible)} polygon(s)",
                "AI Segmentation", level=Qgis.MessageLevel.Info,
            )
            return




        if not self._auto_objects:
            self._record_auto_zero_result(tiles_succeeded)
            return

        pixel_size = self._auto_refine_pixel_size()
        self._auto_review = {
            "geoms": visible,
            "scores": scores or [],
            "ids": ids or [],
            "crs": crs,
            "source_layer_name": source_layer_name,
            "prompt": prompt_text,
            "pixel_size": pixel_size,


            "stamp": ("acc", (self._auto_reslice_cache or {}).get("key")),
        }


        try:
            self._reset_auto_corrections()
        except (RuntimeError, AttributeError):
            pass

        self._last_auto_result = {
            "status": "completed",
            "instances": len(visible),
            "tiles_processed": tiles_succeeded,
            "layer_name": None,
        }



        if getattr(self, "_auto_dense_tiles", 0):
            QgsMessageLog.logMessage(
                f"Auto detection: {self._auto_dense_tiles} tile(s) still at the max masks per "
                "inference after re-split; denser tiling (higher Detail) may "
                "catch more objects.",
                "AI Segmentation", level=Qgis.MessageLevel.Info,
            )





        if self.dock_widget:
            try:
                self.dock_widget.set_auto_review_active(
                    True, count=len(visible), preset=self._auto_review_preset())



                self.dock_widget.set_boundary_snap_offered(
                    self._boundary_snap_offered())


                self._set_zone_band_fill_visible(False)
            except (RuntimeError, AttributeError):
                pass




        self._warm_local_ai_for_review()



        self._seed_review_display_mode()
        if self._auto_selection_layer is not None:
            self._apply_review_display_mode(self._auto_selection_layer)
        self._refresh_auto_review_preview()
        self._review_conf_moves = 0


        try:
            from ...core import telemetry_run_events
            ctx = self._auto_run_ctx or {}
            total = ctx.get("total", tiles_succeeded)
            instances_found = len(self._auto_objects)
            visible_n = len(visible)
            start_pct = int(round((self._auto_confidence or 0.0) * 100))
            if self._auto_tel_stop_reason in (None, "completed"):
                from .auto_client_profile import client_profile_props
                blob_armed, blob_dropped, tile_m = self._auto_blob_guard_stats()
                telemetry_run_events.track_auto_detect_completed(
                    run_id=self._auto_run_id or "",
                    duration_ms=self._auto_duration_ms(),
                    tiles_done=tiles_succeeded,
                    tiles_failed=max(0, total - tiles_succeeded),
                    instances_found=instances_found,
                    instances_visible_at_default=visible_n,
                    zero_at_default=visible_n == 0,
                    stop_reason="completed",
                    warming_ms=self._auto_warming_wait_ms(),
                    merge_mode_final="separate" if self._auto_merge_separate else "map",
                    blob_armed=blob_armed,
                    blob_dropped=blob_dropped,
                    tile_ground_m=tile_m,
                    client_profile=client_profile_props(self),
                )
            telemetry_run_events.track_review_opened(
                run_id=self._auto_run_id or "",
                instances_found=instances_found,
                visible_at_start=visible_n,
                start_confidence=start_pct,
                auto_lowered=start_pct < int(round(self._effective_confidence_default() * 100)),
            )
        except Exception:
            pass  # nosec B110



        self._review_tel_refined = False
        self._review_tel_conf_changed = False
        self._review_abandon_tracked = False
        self._offer_closed_canopy_advice(tiles_succeeded)


        self._offer_free_zone_fit_upsell()
        self._remember_auto_run_pace(tiles_succeeded)
        QgsMessageLog.logMessage(
            f"Auto detection: {len(visible)} object(s) ready for review",
            "AI Segmentation", level=Qgis.MessageLevel.Info,
        )
        self._note_review_open_on_timeline()

    def _note_review_open_on_timeline(self) -> None:


        from ...core import run_timeline
        if not run_timeline.enabled():
            return
        run_timeline.mark("review_open")

        def _idle() -> None:
            run_timeline.mark("review_idle")
            QgsMessageLog.logMessage(
                "Auto detection: timeline - " + run_timeline.summary_line(),
                "AI Segmentation", level=Qgis.MessageLevel.Info)

        try:
            from ...core.qt_compat import safe_single_shot
            safe_single_shot(0, self.dock_widget, _idle)
        except (RuntimeError, AttributeError):
            _idle()

    def _remember_auto_run_pace(self, tiles_succeeded: int) -> None:






        started = getattr(self, "_auto_run_started_mono", None)
        self._auto_run_started_mono = None
        if started is None:
            return
        try:
            import time as _time

            from ...core.run_pace_memory import own_machine_pace, remember_run
            remember_run(int(tiles_succeeded or 0), _time.monotonic() - started)
            pace = own_machine_pace()
            if pace is not None and self.dock_widget is not None:
                self.dock_widget.set_auto_own_pace(pace)
        except Exception:  # noqa: BLE001
            pass  # nosec B110

    def _offer_closed_canopy_advice(self, tiles_succeeded: int) -> None:



        if not self.dock_widget:
            return
        try:
            from ...core.detection_policy import closed_canopy_signature, prompt_suggests_canopy
            prompt = str((self._auto_run_ctx or {}).get("prompt") or "")


            try:
                prompt = str(self._resolve_object_token(prompt) or prompt)
            except (RuntimeError, AttributeError, TypeError):
                pass  # nosec B110
            span_dropped = int(getattr(self, "_auto_blob_split", (0, 0, 0))[1])
            on = (
                bool(getattr(self, "_auto_merge_separate", True))
                and prompt_suggests_canopy(prompt)
                and closed_canopy_signature(
                    int(getattr(self, "_auto_raw_count", 0) or 0),
                    int(tiles_succeeded or 0), span_dropped)
            )
            if on:
                QgsMessageLog.logMessage(
                    f"Auto detection: closed-canopy signature ({span_dropped} "
                    f"spanning mask(s) dropped over {tiles_succeeded} tile(s)); "
                    "review shows the forest advice",
                    "AI Segmentation", level=Qgis.MessageLevel.Info,
                )
            self.dock_widget.set_closed_canopy_advice(on)
        except Exception:  # noqa: BLE001
            pass  # nosec B110

    def _archive_auto_default_export(self) -> None:












        review = getattr(self, "_auto_review", None)
        if not review or getattr(self, "_auto_headless_run", False):
            return
        run_id = getattr(self, "_auto_run_id", None)
        if not run_id:
            return


        if getattr(self, "_auto_default_export_run_id", None) == run_id:
            return
        self._auto_default_export_run_id = run_id
        try:
            geoms = review.get("geoms") or []
            scores = review.get("scores")
            if scores is not None and len(scores) != len(geoms):
                scores = None
            refined, refined_scores = [], []
            for index, g in enumerate(geoms):
                if g is None or g.isEmpty():
                    continue
                refined.append(QgsGeometry(g))
                refined_scores.append(scores[index] if scores else None)
            if not refined:
                return
            from .run_export_upload import queue_run_export_upload





            try:
                default_confidence = float(self._review_start_confidence())
            except Exception:  # noqa: BLE001
                default_confidence = None
            queue_run_export_upload(
                self, review, refined, refined_scores,
                export_path="review_open", confidence_applied=default_confidence)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
