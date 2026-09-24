







from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsMessageLog,
    QgsProject,
    QgsRectangle,
)

from ...core.i18n import tr
from .shared import (
    _RECALL_FLOOR,
    _RECALL_FLOOR_EXEMPLAR_ONLY,
    _provider_name_for_log,
)


class AutoRunStartMixin:


    def _start_auto_detection(self) -> None:












        if getattr(self, "_auto_start_in_progress", False):
            return


        if getattr(self, "_auto_imagery_probe", None) is not None:
            return






        if (getattr(self, "_auto_imagery_resume", None) is None
                and getattr(self, "_auto_density_forced", None) is None):
            import time as _time
            self._auto_click_mono = _time.monotonic()
        self._auto_start_in_progress = True
        try:
            self._start_auto_detection_body()
        finally:
            self._auto_start_in_progress = False
            self._density_after_start()

    def _start_auto_detection_body(self) -> None:

        import uuid as _uuid

        from ...core import run_timeline
        from ...core.activation_manager import get_auth_header, is_plugin_activated

        run_timeline.mark("start_body")







        if not self.dock_widget:



            self._headless_error = tr(
                "The AI Segmentation panel is closed, so there is nothing to "
                "detect from. Open it and try again.")
            return




        self._auto_last_run_sig = None







        try:
            from ...core.cloud_detection import visible_extent_for
        except (ImportError, OSError) as err:


            self._tel_detect_blocked("deps_missing")
            deps_msg = tr(
                "Automatic mode needs a small one-time setup before it can "
                "read your imagery. It takes about a minute."
            )


            self._headless_error = deps_msg
            try:
                self.dock_widget.set_auto_status("error", deps_msg)
            except (RuntimeError, AttributeError):
                pass
            QgsMessageLog.logMessage(
                f"Auto detection: local packages unavailable ({err})",
                "AI Segmentation", level=Qgis.MessageLevel.Critical,
            )


            if not self._auto_headless_run:
                self._offer_automatic_setup(deps_msg)
            return




        if self._auto_worker is not None and self._auto_worker.isRunning():
            self._tel_detect_blocked("worker_busy")




            self._headless_error = tr(
                "A zone detection is already running. Wait for it to finish, "
                "or stop it, before starting another."
            )
            if self.dock_widget:
                try:





                    self.dock_widget.set_auto_status("info", tr(
                        "Finishing the previous run, please wait a moment..."))
                except (RuntimeError, AttributeError):
                    pass
            return


        self._discard_auto_review(exit_path="new_run")




        self._restore_maptool_after_exemplar()





        self._restore_maptool_after_zone()

        layer = self._get_active_raster_layer()
        if layer is None:
            self._tel_detect_blocked("no_layer")
            self._headless_error = tr(
                "Pick a raster layer at the top of the panel first.")
            QgsMessageLog.logMessage(
                "Auto detection: no raster layer selected",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return






        guard_msg = self._auto_raster_guard_message(layer)
        if guard_msg is not None:
            self._tel_detect_blocked(
                getattr(self, "_auto_raster_guard_reason", "raster_shape"))
            try:
                self.dock_widget.set_auto_status("error", guard_msg)
            except (RuntimeError, AttributeError):
                pass
            self._headless_error = guard_msg
            self._push_auto_warning(guard_msg)
            QgsMessageLog.logMessage(
                "Auto detection: raster shape guard blocked the run",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return



        if not self._auto_headless_run:
            self._warn_local_raster_quality(layer)
            self._warn_drawn_map_basemap(layer)


        if not is_plugin_activated():
            self._tel_detect_blocked("not_activated")
            self._headless_error = tr("Sign in to run Automatic.")
            QgsMessageLog.logMessage(
                "Auto detection: plugin not activated",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return


        from ...core.activation_manager import is_automatic_mode_enabled
        if not is_automatic_mode_enabled():
            self._tel_detect_blocked("kill_switch")
            kill_msg = tr(
                "Automatic detection is temporarily unavailable. Please try again later.")
            self._headless_error = kill_msg
            self._push_auto_warning(kill_msg)
            return

        auth = get_auth_header()
        if not auth:
            self._tel_detect_blocked("no_auth")
            self._headless_error = tr("Sign in to run Automatic.")
            QgsMessageLog.logMessage(
                "Auto detection: no auth token available",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return

        if self._tile_manager is None:
            self._setup_auto_mode()

        grid = self._compute_auto_grid(layer)
        if grid is None:
            is_online = self._needs_canvas_render(layer)
            try:
                layer_w = layer.width()
                layer_h = layer.height()
            except (RuntimeError, AttributeError):
                layer_w = 0
                layer_h = 0
            if is_online or min(layer_w, layer_h) <= 0:

                msg = tr(
                    "Draw a zone first. Automatic detection on online layers needs a zone."
                )
                if self.dock_widget:
                    try:
                        self.dock_widget.set_auto_status("info", msg)
                    except (RuntimeError, AttributeError):
                        pass
                self._headless_error = msg
                self._push_auto_warning(msg)
                QgsMessageLog.logMessage(
                    "Auto detection: online layer requires a zone; aborting",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning,
                )
            else:
                self._headless_error = tr(
                    "Could not read the pixel grid of this raster. Check the "
                    "layer opens and shows in QGIS, then try again.")
                QgsMessageLog.logMessage(
                    "Auto detection: could not compute pixel grid for layer",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning,
                )
            return

        pixel_w = grid["pixel_w"]
        pixel_h = grid["pixel_h"]
        geo_bbox = grid["bbox"]








        has_exemplars = self._auto_exemplar_store.count() > 0





        if not self._needs_canvas_render(layer) and self._auto_zone is not None:
            zone_rect = QgsRectangle(geo_bbox[0], geo_bbox[1], geo_bbox[2], geo_bbox[3])






            layer_extent = self._layer_extent_in_run_crs(layer, grid["crs"])
            if layer_extent is not None and not zone_rect.intersects(layer_extent):
                self._abort_zone_outside_layer()
                return




        self._auto_source_is_online = self._needs_canvas_render(layer)






        self._auto_transform_context = self._read_project_transform_context()









        probe = self._probe_imagery_behind_banner(layer, grid)
        if probe is None:


            return
        mupp_floor, probe_msg = probe



        self._retire_early_imagery_probe()
        if probe_msg is None and mupp_floor > 0:




            coarser = self._compute_auto_grid(layer, mupp_floor=mupp_floor)
            if coarser is not None:
                grid = coarser
                pixel_w = grid["pixel_w"]
                pixel_h = grid["pixel_h"]
                geo_bbox = grid["bbox"]
                self._note_imagery_backoff()
                QgsMessageLog.logMessage(
                    "Auto detection: the layer serves no imagery at the detail "
                    "asked for; the run falls back to a coarser one",
                    "AI Segmentation", level=Qgis.MessageLevel.Info,
                )
        if probe_msg is not None:



            self._headless_error = probe_msg
            try:
                self.dock_widget.set_auto_status("error", probe_msg)
            except (RuntimeError, AttributeError):
                pass
            self._push_auto_warning(probe_msg)
            QgsMessageLog.logMessage(
                "Auto detection: the layer serves no imagery at this detail over "
                "the zone; aborting before billing",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return





        tiles = self._tile_manager.compute_grid(pixel_w, pixel_h, apply_cap=False)
        if tiles is not None:


            before = len(tiles)
            tiles = self._tiles_in_polygon(
                tiles, geo_bbox, pixel_w, pixel_h, layer, grid.get("crs"))
            if len(tiles) > self._auto_zone_tile_cap():
                tiles = None
            elif not tiles:



                self._abort_zone_outside_layer()
                return
            elif len(tiles) != before:
                QgsMessageLog.logMessage(
                    f"Auto detection: zone cull kept {len(tiles)} of {before} tiles",
                    "AI Segmentation", level=Qgis.MessageLevel.Info,
                )
        if tiles is None:
            from .shared import zone_too_large_message
            cap = self._auto_zone_tile_cap()
            self._headless_error = zone_too_large_message(cap)
            QgsMessageLog.logMessage(
                f"Auto detection: zone too large (exceeds {cap} tiles)",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return









        if not self._auto_headless_run and self.dock_widget is not None:
            try:
                left_km2 = self.dock_widget._auto_km2_left()
            except (RuntimeError, AttributeError):
                left_km2 = None
            zone_km2 = self._auto_zone_area_km2()



            if left_km2 is not None and zone_km2 > 0 and zone_km2 > left_km2:
                self._tel_detect_blocked("cost_over_balance")
                try:
                    self.dock_widget.set_auto_zone_surface(zone_km2)
                except (RuntimeError, AttributeError):
                    pass
                QgsMessageLog.logMessage(
                    f"Auto detection: zone of {zone_km2:.2f} km2 over the "
                    f"{left_km2:.2f} km2 left this month; aborting before billing",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning,
                )
                return




        crs_authid = grid.get("crs") or layer.crs().authid()

















        import time as _time
        zone_extent = QgsRectangle(geo_bbox[0], geo_bbox[1], geo_bbox[2], geo_bbox[3])



        actual_extent = visible_extent_for(zone_extent, pixel_w, pixel_h)


        geo_bbox = (
            actual_extent.xMinimum(), actual_extent.yMinimum(),
            actual_extent.xMaximum(), actual_extent.yMaximum(),
        )
        geo_transform = {
            "bbox": geo_bbox,
            "img_shape": (pixel_h, pixel_w),
            "crs": crs_authid,
        }


        self._auto_gsd = (geo_bbox[2] - geo_bbox[0]) / pixel_w if pixel_w > 0 else 0.0



        self._auto_gsd_m = self._mupp_to_meters(layer, zone_extent, self._auto_gsd)


        self._auto_mask_gsd = 0.0





        self._auto_render_ms = 0
        self._auto_detect_t0 = _time.monotonic()
        run_timeline.mark("guards_passed")


        try:
            from ...core.run_log_capture import start_run_log
            start_run_log(self._auto_run_id or "")
        except Exception:  # noqa: BLE001
            pass  # nosec B110



        self._auto_live_draw_ms = 0.0
        self._auto_live_draw_ticks = 0
        QgsMessageLog.logMessage(
            f"Auto detection: per-tile JIT render, zone {pixel_w}x{pixel_h}px, {len(tiles)} tile(s) "
            f"(provider={_provider_name_for_log(layer)})",
            "AI Segmentation", level=Qgis.MessageLevel.Info,
        )


        try:
            prompt = self.dock_widget.auto_prompt_input.text().strip()
        except (RuntimeError, AttributeError):
            prompt = ""









        if prompt:
            self._auto_merge_separate = self._default_merge_separate(prompt)
            self._auto_merge_mode_source = "prompt"
        else:
            self._auto_merge_separate = False
            self._auto_merge_mode_source = "signal"







        exemplar_payload = (
            self._compute_exemplar_pixel_boxes(layer, geo_bbox, pixel_w, pixel_h)
            if has_exemplars else None
        )








        if not prompt and not has_exemplars:

            msg = tr("Type what to find, or draw an example of it.")
            try:
                self.dock_widget.set_auto_run_active(False)
                self.dock_widget.set_auto_status("error", msg)
            except (RuntimeError, AttributeError):
                pass
            self._headless_error = msg


            if self.dock_widget is None:
                self._push_auto_warning(msg)
            QgsMessageLog.logMessage(
                "Auto detection: empty prompt and no exemplars; aborting before "
                "any credit is spent",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            self._auto_gsd = 0.0
            self._auto_run_id = None
            return





        from ...core.detect_gate import can_detect
        positives = self._auto_exemplar_store.positives()
        if not can_detect(bool(prompt), positives):
            msg = tr("Type what to find, or draw an example of it.")
            try:
                self.dock_widget.set_auto_run_active(False)
                self.dock_widget.set_auto_status("error", msg)
            except (RuntimeError, AttributeError):
                pass
            self._headless_error = msg


            if self.dock_widget is None:
                self._push_auto_warning(msg)
            QgsMessageLog.logMessage(
                "Auto detection: no prompt; aborting before any credit "
                "is spent",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            self._auto_gsd = 0.0
            self._auto_run_id = None
            return



        exemplar_stamps = None

        if has_exemplars:





            if not exemplar_payload:
                msg = tr(
                    "Could not place the example on the image. Redraw the "
                    "example box inside the zone and try again."
                )
                try:
                    self.dock_widget.set_auto_run_active(False)
                    self.dock_widget.set_auto_status("error", msg)
                except (RuntimeError, AttributeError):
                    pass
                self._headless_error = msg
                self._push_auto_warning(msg)
                self._auto_gsd = 0.0
                self._auto_run_id = None
                return





            smallest_px = min(
                min(b["box"][2] - b["box"][0], b["box"][3] - b["box"][1])
                for b in exemplar_payload
            )
            QgsMessageLog.logMessage(
                "Auto detection (exemplar): composite-per-tile, full image "
                f"{pixel_w}x{pixel_h}px, {len(exemplar_payload)} example(s), smallest example {smallest_px:.0f}px",
                "AI Segmentation", level=Qgis.MessageLevel.Info,
            )











            exemplar_stamps = self._build_exemplar_stamps(
                layer, geo_bbox, pixel_w, pixel_h, has_prompt=bool(prompt))
            if not exemplar_stamps and prompt:



                QgsMessageLog.logMessage(
                    "Auto detection: no usable example (render failed, no "
                    "in-situ box); continuing on the text prompt alone",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning,
                )
                exemplar_stamps = None
            elif not exemplar_stamps:
                msg = tr(
                    "Could not place the example on the image. Redraw the "
                    "example box inside the zone and try again."
                )
                try:
                    self.dock_widget.set_auto_run_active(False)
                    self.dock_widget.set_auto_status("error", msg)
                except (RuntimeError, AttributeError):
                    pass
                self._headless_error = msg
                self._push_auto_warning(msg)
                QgsMessageLog.logMessage(
                    "Auto detection (exemplar): all example renders failed; "
                    "aborting before any credit is spent",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning,
                )
                self._auto_gsd = 0.0
                self._auto_run_id = None
                return






        from qgis.PyQt.QtCore import QEventLoop
        from qgis.PyQt.QtWidgets import QApplication

        self._clear_zone_tile_grid()
        if not self._auto_headless_run:




            self._auto_grid_suppressed = True





        self._set_zone_band_fill_visible(False)
        try:
            self.dock_widget.set_auto_run_active(True)



            self.dock_widget.set_auto_status("info", tr("Preparing your zone..."))
        except (RuntimeError, AttributeError):
            pass





        QApplication.processEvents(
            QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        run_timeline.mark("ui_flipped")

        from ...core.polygon_exporter import IncrementalMerger



        self._reset_auto_live_pipeline()


        forced = getattr(self, "_auto_density_forced", None)
        forced_run_id = forced.get("run_id") if isinstance(forced, dict) else None
        self._auto_run_id = forced_run_id or str(_uuid.uuid4())

        self._reset_credits_backoff()


        try:
            from ...core import telemetry
            telemetry.set_last_run_id(self._auto_run_id)
        except Exception:
            pass  # nosec B110








        from ...core import detection_policy
        self._auto_merge_scalars = detection_policy.merge_scalars()









        self._auto_restore_partitions = detection_policy.restore_partitions_for(
            prompt, exemplar_only=bool(has_exemplars) and not prompt)
        self._auto_merger = IncrementalMerger(
            seam_min_dim=self._auto_seam_min_dim(),
            select_duplicates=self._auto_merge_separate,
            gsd=self._auto_gsd,


            restore_partitions=(self._auto_merge_separate and self._auto_restore_partitions),



            **detection_policy.merge_scalar_kwargs(
                IncrementalMerger, self._auto_merge_scalars),
        )












        from ...core.review_defaults import AUTO_DEFAULT_CONFIDENCE
        try:
            spin_conf = (self.dock_widget.get_auto_confidence()
                         if self.dock_widget is not None else None)
            if spin_conf is not None and abs(
                    float(spin_conf) - AUTO_DEFAULT_CONFIDENCE) > 1e-9:
                self._auto_confidence = float(spin_conf)
            else:




                self._auto_confidence = self._snap_review_start_confidence(
                    self._confidence_default_for(
                        prompt, bool(has_exemplars) and not prompt))
        except (RuntimeError, AttributeError, TypeError, ValueError):
            self._auto_confidence = AUTO_DEFAULT_CONFIDENCE
        self._auto_raw_count = 0
        self._auto_dense_tiles = 0
        self._auto_objects = []
        self._auto_preview_geoms = []
        self._reset_review_refine_cache()








        from ...core.tile_manager import TILE_SIZE
        self._auto_is_exemplar_only = bool(has_exemplars) and not prompt
        self._auto_collect_raw = self._auto_is_exemplar_only
        self._auto_retain_raw = self._auto_collect_raw
        self._auto_raw_fragments = [] if self._auto_retain_raw else None
        self._auto_raw_n_total = 0
        self._auto_raw_cov_sum = 0.0
        self._auto_raw_cov_sq_sum = 0.0


        self._auto_tile_ground_area = (
            (TILE_SIZE * self._auto_gsd) ** 2
            if self._auto_retain_raw and self._auto_gsd > 0 else 0.0)
        self._auto_manual_removed = set()




        self._auto_correction_removed = set()
        self._auto_manual_object_ids = set()


        self._auto_crs_authid = crs_authid



        self._auto_clip_polygon = self._polygon_in_run_crs(layer)




        if self._auto_clip_polygon is not None:
            try:
                from qgis.core import QgsCoordinateReferenceSystem as _QgsCrs
                from qgis.core import QgsCoordinateTransform as _QgsTransform
                from qgis.core import QgsGeometry as _QgsGeometry
                from qgis.core import QgsProject as _QgsProject


                data_extent = layer.extent()
                run_crs = _QgsCrs(crs_authid)
                if run_crs.isValid() and run_crs != layer.crs():
                    data_extent = _QgsTransform(
                        layer.crs(), run_crs, _QgsProject.instance()
                    ).transformBoundingBox(data_extent)
                data_rect = _QgsGeometry.fromRect(data_extent)
                clipped = self._auto_clip_polygon.intersection(data_rect)
                if clipped is not None and not clipped.isEmpty() and clipped.area() > 0:
                    self._auto_clip_polygon = clipped
            except Exception:  # noqa: BLE001  # nosec B110
                pass



        self._auto_clip_engine = self._prepare_clip_engine(self._auto_clip_polygon)






        self._seed_review_display_mode()

        self._remove_auto_selection_layer()
        self._auto_selection_layer = self._create_auto_selection_layer(layer)
        run_timeline.mark("selection_layer")







        self._auto_mask_scale = detection_policy.mask_scale_for_run(
            prompt, getattr(self, "_auto_gsd_m", 0.0))




        self._auto_run_ctx = {
            "tiles": tiles,
            "geo_transform": geo_transform,
            "crs_authid": crs_authid,
            "prompt": prompt,
            "layer_id": layer.id(),
            "zone": QgsRectangle(self._auto_zone) if self._auto_zone is not None else None,
            "detail": self._get_auto_detail_level(),
            "detection_threshold": self.dock_widget.get_auto_confidence(),
            "exemplars": exemplar_payload,
            "mask_scale": self._auto_mask_scale,
            "total": len(tiles),
        }





        from .shared import AutoRerunSignature, auto_rerun_scope
        self._auto_last_run_sig = AutoRerunSignature(
            self,
            (prompt, self._get_auto_detail_level(),
             self._auto_exemplar_store.count()),
            auto_rerun_scope(self))










        from ...workers.auto_detection_worker import TileRenderBridge
        self._auto_tile_bridge = TileRenderBridge(layer, geo_transform)





        recall_text = detection_policy.recall_floor(_RECALL_FLOOR)
        recall_exemplar = detection_policy.recall_floor_exemplar_only(
            _RECALL_FLOOR_EXEMPLAR_ONLY)
        plan = self._active_run_plan(prompt)
        if plan is not None:
            pv = plan.get("recall_floor")
            if isinstance(pv, (int, float)) and not isinstance(pv, bool):
                recall_text = float(pv)
            pv = plan.get("recall_floor_exemplar_only")
            if isinstance(pv, (int, float)) and not isinstance(pv, bool):
                recall_exemplar = float(pv)
        detection_threshold = (
            recall_text if (prompt or "").strip() else recall_exemplar)





        from ...core.cloud_detection import should_request_semantic
        return_semantic = should_request_semantic(
            detection_policy.semantic_rescue_enabled(),
            bool(prompt),
            self._auto_merge_separate,
        )





        client_meta = self._build_auto_client_meta()

        density_probe = self._density_probe_plan(
            layer,
            self._reproject_zone_to_run_crs(self._auto_zone, layer)
            if self._auto_zone is not None else None,
            prompt, tiles, bool(has_exemplars))
        run_timeline.mark("launch")





        self._launch_auto_worker(
            tile_renderer=self._auto_tile_bridge.render_tile,
            tiles=tiles,
            geo_transform=geo_transform,
            crs_authid=crs_authid,
            prompt=prompt,
            auth=auth,
            run_id=self._auto_run_id,





            max_concurrent=detection_policy.max_concurrent(),






            detection_threshold=detection_threshold,
            exemplar_stamps=exemplar_stamps,


            merge_scalars=self._auto_merge_scalars,
            subdivide_budget=self._auto_subdivide_budget(
                len(tiles), bool(exemplar_stamps)),




            collect_raw=self._auto_collect_raw,


            return_semantic=return_semantic,


            gate_config=self._auto_gate_config(
                prompt, bool(exemplar_stamps), len(tiles)),
            client_meta=client_meta,
            density_probe=density_probe,
        )
        restarted = isinstance(forced, dict)



        self._auto_tel_stop_reason = None
        self._auto_skipped_tiles = 0
        self._auto_timeout_tiles = 0




        self._auto_convert_failed_tiles = 0


        self._auto_error_dialog_shown = False


        self._auto_skipped_blank_tiles = 0
        self._auto_render_failed_tiles = 0
        self._auto_unavailable_tiles = 0



        self._auto_prefiltered_tiles = 0
        self._auto_gate_skipped_tiles = 0


        self._auto_warming_t0 = None
        self._auto_warming_ms = 0


        try:

            tile_props = self._tile_plan_run_props(
                layer, self._reproject_zone_to_run_crs(self._auto_zone, layer)
                if self._auto_zone is not None else None,
                getattr(self, "_auto_gsd_m", 0.0))
            if not restarted:
                from ...core import telemetry_run_events
                credits_before, is_free_tier = self._auto_credit_snapshot()
                telemetry_run_events.track_auto_detect_started(
                    run_id=self._auto_run_id,
                    tiles=len(tiles),
                    zone_km2=self._auto_zone_area_km2(),



                    object_class=prompt or "Example match",
                    detail=self._get_auto_detail_level(),
                    detail_seeded=getattr(self, "_auto_detail_seeded", None),
                    exemplar_count=self._auto_exemplar_store.count(),
                    est_credits=len(tiles),
                    credits_before=credits_before,
                    is_free_tier=bool(is_free_tier),
                    merge_mode="separate" if self._auto_merge_separate else "map",
                    merge_mode_source=getattr(self, "_auto_merge_mode_source", "prompt"),
                    tile_props=tile_props,
                )
        except Exception:
            pass  # nosec B110


        self._density_clear_forced()

    @staticmethod
    def _layer_extent_in_run_crs(layer, crs_authid: str):






        try:
            extent = layer.extent()
            run_crs = QgsCoordinateReferenceSystem(crs_authid)
            if not run_crs.isValid():



                return extent if crs_authid == layer.crs().authid() else None
            if run_crs == layer.crs():
                return extent
            return QgsCoordinateTransform(
                layer.crs(), run_crs, QgsProject.instance()
            ).transformBoundingBox(extent)
        except Exception:  # noqa: BLE001
            return None
