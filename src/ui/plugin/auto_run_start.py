







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
    _provider_name_for_log,
)


def zone_edge_keep_margin_m(prompt: str, merge_separate: bool) -> float:




    if not prompt or not merge_separate:
        return 0.0
    try:
        from ...core.detection_policy import (
            object_profile,
            zone_edge_margin_mult,
            zone_edge_whole_objects,
        )
        if not zone_edge_whole_objects():
            return 0.0
        size_m, _mupp = object_profile(prompt)
        margin = float(size_m) * zone_edge_margin_mult()
    except Exception:  # noqa: BLE001
        return 0.0
    return margin if margin > 0 else 0.0


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
        worker_before = getattr(self, "_auto_worker", None)
        from ...core.served_config import ServedConfigMissing
        try:
            self._start_auto_detection_body()
        except ServedConfigMissing as err:




            QgsMessageLog.logMessage(
                f"Auto detection: served setting missing ({err.key})",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            if getattr(self, "_auto_worker", None) is worker_before:
                self._refuse_start_without_settings()
        finally:
            self._auto_start_in_progress = False

            if (getattr(self, "_auto_worker", None) is worker_before
                    and getattr(self, "_auto_review", None) is None):
                from ...core.detection_policy_core import release_run_policy
                release_run_policy()

                self._late_plan_clear()
            self._density_after_start()

    def _start_auto_detection_body(self) -> None:







        from ...core import run_timeline

        run_timeline.mark("start_body")









        try:
            dock = self.dock_widget
            if dock is not None and dock.auto_target_mode() == "land_cover":
                from ..dock.auto_target_mode import LAND_COVER_WORD, LAND_COVER_WORDS
                if dock.auto_prompt_input.text().strip().lower() not in LAND_COVER_WORDS:
                    dock.auto_prompt_input.blockSignals(True)
                    dock.auto_prompt_input.setText(LAND_COVER_WORD)
                    dock.auto_prompt_input.blockSignals(False)
        except (RuntimeError, AttributeError):  # nosec B110
            pass

        if not self._auto_headless_run and self._my_classes_wanted():
            self._start_my_classes_run()
            return
        visible_extent_for = self._start_step_panel_and_deps()
        if visible_extent_for is None:
            return
        if not self._start_step_worker_not_busy():
            return
        if not self._start_step_served_settings():
            return
        if not self._start_step_land_cover_plan():
            return
        if self._start_step_release_ui_tools() is False:
            return
        layer = self._start_step_pick_layer()
        if layer is None:
            return
        auth = self._start_step_sign_in()
        if not auth:
            return
        grid = self._start_step_pixel_grid(layer)
        if grid is None:
            return

        pixel_w = grid["pixel_w"]
        pixel_h = grid["pixel_h"]
        geo_bbox = grid["bbox"]








        has_exemplars = self._auto_exemplar_store.count() > 0
        if not self._start_step_zone_on_raster(layer, grid, geo_bbox):
            return
        imagery = self._start_step_imagery_check(layer, grid, pixel_w, pixel_h, geo_bbox)
        if imagery is None:
            return
        grid, pixel_w, pixel_h, geo_bbox = imagery
        tiles = self._start_step_tile_plan(layer, grid, pixel_w, pixel_h, geo_bbox)
        if tiles is None:
            return
        if not self._start_step_month_surface():
            return
        crs_authid, geo_bbox, geo_transform = self._start_step_run_geometry(
            layer, grid, geo_bbox, pixel_w, pixel_h, tiles, visible_extent_for, run_timeline)
        prompt = self._start_step_read_prompt()


        if not has_exemplars and self._land_cover_plan_for(prompt) is None:
            tiles = self._zone_edge_grid_tiles(layer, tiles, geo_transform)
        exemplar_payload = self._start_step_example_boxes(
            layer, geo_bbox, pixel_w, pixel_h, has_exemplars)
        if not self._start_step_query_guards(prompt, has_exemplars):
            return
        exemplar_stamps, examples_ok = self._start_step_prepare_examples(
            layer, geo_bbox, pixel_w, pixel_h, prompt, has_exemplars, exemplar_payload)
        if not examples_ok:
            return
        from ...core.activation_manager import auth_revision

        if auth_revision() != self._auto_start_auth_revision:
            self._headless_error = tr("Session expired. Sign in again to continue.")
            self._push_auto_warning(self._headless_error)
            return
        if not self._auto_resume_plan_ok(
                tiles, prompt, layer, geo_transform=geo_transform, crs_authid=crs_authid):
            return
        self._start_step_flip_ui_to_run(run_timeline)
        forced = self._start_step_open_run(prompt, has_exemplars)
        self._start_step_confidence_and_counters(prompt, has_exemplars)
        self._start_step_clip_polygon(layer, crs_authid)
        self._note_zone_margin_for_replay()
        self._start_step_live_layer(layer, run_timeline)
        self._start_step_mask_scale(prompt)
        self._start_step_self_exemplar()
        self._start_step_run_context(
            layer, prompt, tiles, geo_transform, crs_authid, exemplar_payload)
        detection_threshold, return_semantic, client_meta, density_probe = (
            self._start_step_launch_dials(layer, prompt, tiles, has_exemplars))


        land_cover = self._land_cover_plan_for(prompt)
        if self._land_cover_begin_run(land_cover, tiles, geo_transform) is False:
            return
        if land_cover is not None:
            return_semantic = False
            density_probe = None
            self._auto_self_exemplar = None
        run_timeline.mark("launch")

        from ...core import detection_policy





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
            exemplar_stamps=None if land_cover is not None else exemplar_stamps,


            merge_scalars=self._auto_merge_scalars,
            subdivide_budget=0 if land_cover is not None else self._auto_subdivide_budget(
                len(tiles), bool(exemplar_stamps)),




            collect_raw=self._auto_collect_raw,


            return_semantic=return_semantic,


            gate_config=None if land_cover is not None else self._auto_gate_config(
                prompt, bool(exemplar_stamps), len(tiles)),
            client_meta=client_meta,
            density_probe=density_probe,
        )
        self._start_step_announce_started(layer, tiles, prompt, forced)

    def _start_step_panel_and_deps(self) -> object:


        if not self.dock_widget:



            self._headless_error = tr(
                "The AI Segmentation panel is closed, so there is nothing to "
                "detect from. Open it and try again.")
            return None




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
            return None
        return visible_extent_for

    def _start_step_served_settings(self) -> bool:






        try:
            prompt = self.dock_widget.auto_prompt_input.text().strip()
        except (RuntimeError, AttributeError):
            prompt = ""
        from ...core.detection_policy_core import capture_run_policy, release_run_policy
        from ...core.run_decisions import neutral_run_decisions, parse_run_decisions
        from ...core.served_config import served_config_ready

        self._late_plan_clear()
        plan = self._active_run_plan(prompt)


        self._auto_run_decisions = parse_run_decisions(plan)
        if served_config_ready():
            capture_run_policy(plan)
            if self._served_policy_preflight():
                if self._auto_run_decisions is None:
                    self._auto_run_decisions = neutral_run_decisions()
                    if plan is None:
                        self._late_plan_begin(prompt)
                    else:
                        QgsMessageLog.logMessage(
                            "Auto detection: run plan carries no decisions; "
                            "neutral choices", "AI Segmentation",
                            level=Qgis.MessageLevel.Warning)
                return True
            release_run_policy()
        self._refuse_start_without_settings(prompt)
        return False

    def _refuse_start_without_settings(self, prompt: str | None = None) -> None:



        if prompt is None:
            try:
                prompt = self.dock_widget.auto_prompt_input.text().strip()
            except (RuntimeError, AttributeError):
                prompt = ""
        self._tel_detect_blocked("settings_not_loaded")
        self._headless_error = tr("Connecting to load settings")
        self._headless_error_code = "settings_not_loaded"
        self._request_served_settings(prompt)
        if not getattr(self, "_auto_headless_run", False):
            self._show_served_settings_missing()

    @staticmethod
    def _served_policy_preflight() -> bool:



        from ...core import boundary_snap
        from ...core import detection_policy as dp
        from ...core.served_config import ServedConfigMissing

        try:
            dp.merge_scalars()
            dp.map_likeness_min_share()
            for reader in (
                    dp.max_masks_per_tile, dp.mask_cap_trigger_frac,
                    dp.max_tile_coverage, dp.hard_tile_coverage,
                    dp.hard_cover_shape_escape, dp.compact_min_fill,
                    dp.tile_span_fraction, dp.min_keep_px,
                    dp.subdiv_max_depth, dp.resplit_time_ratio,
                    dp.subdivide_overlap_fraction, dp.subdivide_min_parent_px,
                    dp.min_keep_floor_m2):
                reader()
            for reader in (
                    dp.zone_seed_mupp, dp.object_min_px,
                    dp.detail_coarse_travel_ratio, dp.detail_fine_travel_ratio,
                    dp.drawn_object_tile_frac, dp.split_risk_tile_frac,
                    dp.tile_fit_object_frac,
                    dp.tile_plan_half_steps, dp.tile_plan_step_ratio,
                    dp.exemplar_context_pad, dp.exemplar_context_pad_px_cap,
                    dp.exemplar_min_paste_scale,
                    dp.semantic_rescue_coverage_floor,
                    dp.gate_prefilter_band_eps,
                    dp.recall_floor, dp.recall_floor_exemplar_only,
                    boundary_snap.boundary_snap_tolerance_m,
                    boundary_snap.boundary_snap_max_area_change,
                    boundary_snap.boundary_snap_min_keep_share,
                    boundary_snap.boundary_snap_max_objects):
                reader()
            if dp.gate_enabled():
                dp.gate_group()
                dp.gate_max_group()
                dp.gate_min_pixels()
                dp.gate_min_tiles()
            dp.density_probe_config()
        except ServedConfigMissing as err:
            QgsMessageLog.logMessage(
                f"Auto detection: served setting missing ({err.key})",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return False
        return True

    def _request_served_settings(self, prompt: str) -> None:


        from ...core.served_config import served_config_ready

        if not served_config_ready():
            try:
                self._refresh_config_for_account_change()
            except Exception:  # noqa: BLE001  # nosec B110
                pass
        token = self._resolve_object_token(prompt) if prompt else ""


        if self._run_plan_fetch_in_flight(token):
            return
        try:
            if prompt or self._exemplar_size_for_plan() is not None:
                self._fetch_auto_run_plan(token)
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _show_served_settings_missing(self) -> None:


        try:
            from qgis.PyQt.QtWidgets import QPushButton

            bar = self.iface.messageBar()
            old = getattr(self, "_served_settings_message", None)
            if old is not None:
                try:
                    bar.popWidget(old)
                except (RuntimeError, TypeError):
                    pass
            item = bar.createMessage(tr("Connecting to load settings"))
            button = QPushButton(tr("Retry"))
            button.setAutoDefault(False)
            button.clicked.connect(self._on_served_settings_retry)
            item.layout().addWidget(button)
            self._served_settings_message = item
            bar.pushWidget(item, Qgis.MessageLevel.Info, 10)
        except (RuntimeError, AttributeError):
            self._served_settings_message = None

    def _on_served_settings_retry(self) -> None:


        old = getattr(self, "_served_settings_message", None)
        self._served_settings_message = None
        if old is not None:
            try:
                self.iface.messageBar().popWidget(old)
            except (RuntimeError, TypeError, AttributeError):
                pass
        try:
            prompt = self.dock_widget.auto_prompt_input.text().strip()
        except (RuntimeError, AttributeError):
            prompt = ""
        self._request_served_settings(prompt)

    def _start_step_worker_not_busy(self) -> bool:




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
            return False
        return True

    def _start_step_release_ui_tools(self) -> bool:






        if not getattr(self, "_auto_resume_armed", False):
            if self._discard_auto_review(exit_path="new_run", keep_run_policy=True) is False:
                return False




        self._restore_maptool_after_exemplar()





        self._restore_maptool_after_zone()
        return True

    def _start_step_pick_layer(self) -> object:

        layer = self._get_active_raster_layer()
        if layer is None:
            self._tel_detect_blocked("no_layer")
            self._headless_error = tr(
                "Pick a raster layer at the top of the panel first.")
            QgsMessageLog.logMessage(
                "Auto detection: no raster layer selected",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return None






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
            return None



        if not self._auto_headless_run:
            self._warn_local_raster_quality(layer)
            self._warn_drawn_map_basemap(layer)
        return layer

    def _start_step_sign_in(self) -> object:

        from ...core.activation_manager import auth_revision, get_auth_header, is_plugin_activated


        if not is_plugin_activated():
            self._tel_detect_blocked("not_activated")
            self._headless_error = tr("Sign in to run Automatic.")
            QgsMessageLog.logMessage(
                "Auto detection: plugin not activated",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return None


        from ...core.activation_manager import is_automatic_mode_enabled
        if not is_automatic_mode_enabled():
            self._tel_detect_blocked("kill_switch")
            kill_msg = tr(
                "Automatic detection is temporarily unavailable. Please try again later.")
            self._headless_error = kill_msg
            self._push_auto_warning(kill_msg)
            return None

        auth = get_auth_header()
        self._auto_start_auth_revision = auth_revision()
        if not auth:
            self._tel_detect_blocked("no_auth")
            self._headless_error = tr("Sign in to run Automatic.")
            QgsMessageLog.logMessage(
                "Auto detection: no auth token available",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return None
        return auth

    def _start_step_pixel_grid(self, layer) -> object:

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
            return None
        return grid

    def _start_step_zone_on_raster(self, layer, grid, geo_bbox) -> bool:





        if not self._needs_canvas_render(layer) and self._auto_zone is not None:
            zone_rect = QgsRectangle(geo_bbox[0], geo_bbox[1], geo_bbox[2], geo_bbox[3])






            layer_extent = self._layer_extent_in_run_crs(layer, grid["crs"])
            if layer_extent is not None and not zone_rect.intersects(layer_extent):
                self._abort_zone_outside_layer()
                return False
        return True

    def _start_step_imagery_check(self, layer, grid, pixel_w, pixel_h, geo_bbox) -> object:





        self._auto_source_is_online = self._needs_canvas_render(layer)






        self._auto_transform_context = self._read_project_transform_context()









        probe = self._probe_imagery_behind_banner(layer, grid)
        if probe is None:


            return None
        mupp_floor, probe_msg = probe


        self._auto_imagery_floor_ratio = 0.0


        probe_grid = grid



        self._retire_early_imagery_probe()
        if probe_msg is None and mupp_floor > 0:




            coarser = self._compute_auto_grid(layer, mupp_floor=mupp_floor)
            if coarser is not None:
                grid = coarser
                pixel_w = grid["pixel_w"]
                pixel_h = grid["pixel_h"]
                geo_bbox = grid["bbox"]
                from ...core.online_zoom_reach import floor_upsample_enabled
                grid_mupp = self._grid_mupp(grid)
                if grid_mupp > 0 and floor_upsample_enabled():
                    self._auto_imagery_floor_ratio = float(mupp_floor) / grid_mupp
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
            return None



        if not self._auto_imagery_notice_passes(layer, probe_grid, mupp_floor):
            return None
        return grid, pixel_w, pixel_h, geo_bbox

    def _start_step_tile_plan(self, layer, grid, pixel_w, pixel_h, geo_bbox) -> object:





        tiles = self._tile_manager.compute_grid(pixel_w, pixel_h, apply_cap=False)
        if tiles is not None:


            before = len(tiles)
            tiles = self._tiles_in_polygon(
                tiles, geo_bbox, pixel_w, pixel_h, layer, grid.get("crs"))
            if len(tiles) > self._auto_zone_tile_cap():
                tiles = None
            elif not tiles:



                self._abort_zone_outside_layer()
                return None
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
            return None
        return tiles

    def _start_step_month_surface(self) -> bool:









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
                return False
        return True

    def _start_step_run_geometry(
            self, layer, grid, geo_bbox, pixel_w, pixel_h, tiles, visible_extent_for,
            run_timeline) -> tuple:





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
        return crs_authid, geo_bbox, geo_transform

    def _start_step_read_prompt(self) -> str:


        try:
            prompt = self.dock_widget.auto_prompt_input.text().strip()
        except (RuntimeError, AttributeError):
            prompt = ""









        if prompt:
            self._auto_merge_separate = bool(
                (getattr(self, "_auto_run_decisions", None) or {}).get("merge_separate", True))
            self._auto_merge_mode_source = "prompt"
        else:
            self._auto_merge_separate = False
            self._auto_merge_mode_source = "signal"
        self._auto_zone_keep_margin_m = self._zone_keep_margin_m(prompt)
        return prompt

    def _zone_keep_margin_m(self, prompt: str) -> float:









        return zone_edge_keep_margin_m(prompt, bool(self._auto_merge_separate))

    def _note_zone_margin_for_replay(self) -> None:



        margin_m = float(getattr(self, "_auto_zone_keep_margin_m", 0.0) or 0.0)
        if margin_m <= 0.0 or getattr(self, "_auto_clip_polygon", None) is None:
            return
        from .run_restore import note_run_zone_margin
        note_run_zone_margin(str(self._auto_run_id or ""), margin_m)

    def _zone_edge_grid_tiles(self, layer, tiles: list, geo_transform: dict) -> list:











        margin_m = float(getattr(self, "_auto_zone_keep_margin_m", 0.0) or 0.0)
        gsd_m = float(getattr(self, "_auto_gsd_m", 0.0) or 0.0)
        if margin_m <= 0.0 or gsd_m <= 0.0 or not tiles:
            return tiles
        try:
            import math

            from ...core.tile_manager import margin_axis_span, margin_grid_tiles
            img_h, img_w = (int(v) for v in geo_transform["img_shape"][:2])
            minx, miny, maxx, maxy = (float(v) for v in geo_transform["bbox"])
            m = int(math.ceil(margin_m / gsd_m))
            px_w = (maxx - minx) / img_w
            px_h = (maxy - miny) / img_h
            reach = (-m, -m, img_w + m, img_h + m)
            if not self._needs_canvas_render(layer):
                extent = self._layer_extent_in_run_crs(layer, geo_transform.get("crs"))
                if extent is None:
                    return tiles
                reach = (
                    max(-m, min(0, int(math.ceil((extent.xMinimum() - minx) / px_w)))),
                    max(-m, min(0, int(math.ceil((maxy - extent.yMaximum()) / px_h)))),
                    min(img_w + m, max(img_w, int(math.floor((extent.xMaximum() - minx) / px_w)))),
                    min(img_h + m, max(img_h, int(math.floor((maxy - extent.yMinimum()) / px_h)))),
                )



            zx0, zy0, zx1, zy1 = 0.0, 0.0, float(img_w), float(img_h)
            zone = getattr(self, "_auto_zone", None)
            if zone is not None:
                box = self._reproject_zone_to_run_crs(zone, layer)
                if box is not None and not box.isEmpty():
                    zx0 = (box.xMinimum() - minx) / px_w
                    zx1 = (box.xMaximum() - minx) / px_w
                    zy0 = (maxy - box.yMaximum()) / px_h
                    zy1 = (maxy - box.yMinimum()) / px_h
            x0, x1 = margin_axis_span(zx0, zx1, img_w, m, (reach[0], reach[2]))
            y0, y1 = margin_axis_span(zy0, zy1, img_h, m, (reach[1], reach[3]))
            grown = margin_grid_tiles(self._tile_manager, (x0, y0, x1, y1))
            if not grown:
                return tiles
            grown = self._tiles_in_polygon(
                grown, (minx, miny, maxx, maxy), img_w, img_h, layer,
                geo_transform.get("crs"))
            drawn_km2 = float(self._auto_zone_area_km2() or 0.0)





            per_tile_max_km2 = 0.0016
            if self._auto_zone_wkt_wgs84() is None:
                over_bill = len(grown) > len(tiles)
            else:
                over_bill = len(grown) * per_tile_max_km2 > drawn_km2
            if (not grown or len(grown) > self._auto_zone_tile_cap()
                    or over_bill):
                QgsMessageLog.logMessage(
                    f"Auto detection: zone edge margin skipped ({len(grown or [])} "
                    f"tiles for {drawn_km2:.4f} km2)", "AI Segmentation",
                    level=Qgis.MessageLevel.Info)
                return tiles
        except Exception:  # noqa: BLE001
            return tiles
        QgsMessageLog.logMessage(
            f"Auto detection: zone edge margin {m} px ({margin_m:.0f} m), "
            f"tiles {len(tiles)} -> {len(grown)}", "AI Segmentation",
            level=Qgis.MessageLevel.Info)
        return grown

    def _start_step_example_boxes(self, layer, geo_bbox, pixel_w, pixel_h, has_exemplars) -> object:







        return (
            self._compute_exemplar_pixel_boxes(layer, geo_bbox, pixel_w, pixel_h)
            if has_exemplars else None
        )

    def _start_step_query_guards(self, prompt, has_exemplars) -> bool:








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
            return False





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
            return False
        return True

    def _start_step_prepare_examples(
            self, layer, geo_bbox, pixel_w, pixel_h, prompt, has_exemplars,
            exemplar_payload) -> tuple:




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
                return None, False





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
                return None, False
        return exemplar_stamps, True

    def _start_step_flip_ui_to_run(self, run_timeline) -> None:






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

    def _start_step_open_run(self, prompt, has_exemplars) -> object:


        import uuid as _uuid

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





        self._auto_restore_partitions = bool(
            (getattr(self, "_auto_run_decisions", None) or {}).get("restore_partitions", False))
        self._auto_merger = IncrementalMerger(
            seam_min_dim=self._auto_seam_min_dim(),
            select_duplicates=self._auto_merge_separate,
            gsd=self._auto_gsd,


            restore_partitions=(self._auto_merge_separate and self._auto_restore_partitions),



            **detection_policy.merge_scalar_kwargs(
                IncrementalMerger, self._auto_merge_scalars),
        )
        return forced

    def _start_step_confidence_and_counters(self, prompt, has_exemplars) -> None:













        from ...core.review_defaults import AUTO_DEFAULT_CONFIDENCE

        self._auto_start_confidence_default = None
        try:
            self._auto_start_confidence_default = self._confidence_default_for(
                prompt, bool(has_exemplars) and not prompt)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            self._auto_start_confidence_default = None
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


        if self._late_plan_pending():
            self._auto_retain_raw = True
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

    def _start_step_clip_polygon(self, layer, crs_authid) -> None:



        self._auto_crs_authid = crs_authid


        run_crs = QgsCoordinateReferenceSystem(crs_authid)
        context = getattr(self, "_auto_transform_context", None)
        if context is None:
            context = QgsProject.instance().transformContext()
        self._auto_clip_polygon = self._polygon_in_run_crs(
            layer, target_crs=run_crs, transform_context=context)
        if self._auto_clip_polygon is None and getattr(self, "_auto_zone_polygon", None) is None:
            zone_rect = getattr(self, "_auto_zone", None)
            if zone_rect is not None and not zone_rect.isEmpty():
                from qgis.core import QgsGeometry

                from ...core.qt_compat import geometry_op_succeeded
                geom = QgsGeometry.fromRect(zone_rect)
                try:
                    source_crs = self._zone_source_crs(zone_rect)
                    target_crs = QgsCoordinateReferenceSystem(crs_authid)
                    if (source_crs is None or not source_crs.isValid() or not target_crs.isValid()
                            or source_crs != target_crs and not geometry_op_succeeded(
                                geom.transform(QgsCoordinateTransform(
                                    source_crs, target_crs, context)))):
                        geom = None
                except Exception:  # noqa: BLE001
                    geom = None
                self._auto_clip_polygon = geom




        if self._auto_clip_polygon is not None:
            try:
                from qgis.core import QgsCoordinateReferenceSystem as _QgsCrs
                from qgis.core import QgsCoordinateTransform as _QgsTransform
                from qgis.core import QgsGeometry as _QgsGeometry


                data_extent = layer.extent()
                run_crs = _QgsCrs(crs_authid)
                if run_crs.isValid() and run_crs != layer.crs():
                    data_extent = _QgsTransform(
                        layer.crs(), run_crs, context
                    ).transformBoundingBox(data_extent)
                data_rect = _QgsGeometry.fromRect(data_extent)
                clipped = self._auto_clip_polygon.intersection(data_rect)
                if clipped is not None and not clipped.isEmpty() and clipped.area() > 0:
                    self._auto_clip_polygon = clipped
            except Exception:  # noqa: BLE001  # nosec B110
                pass



        self._auto_clip_engine = self._prepare_clip_engine(self._auto_clip_polygon)

    def _start_step_live_layer(self, layer, run_timeline) -> None:






        self._seed_review_display_mode()

        self._remove_auto_selection_layer()
        self._auto_selection_layer = self._create_auto_selection_layer(layer)
        run_timeline.mark("selection_layer")

    def _start_step_mask_scale(self, prompt) -> None:

        from ...core.run_decisions import coarse_mask_scale






        threshold = (getattr(self, "_auto_run_decisions", None) or {}).get(
            "coarse_mask_max_mupp", 0.0)
        self._auto_mask_scale = coarse_mask_scale(
            threshold, getattr(self, "_auto_gsd_m", 0.0))

    def _start_step_self_exemplar(self) -> None:



        from ...core.self_exemplar import resolve_self_exemplar_settings
        from ...core.server_dials import dial_bool

        self._auto_self_exemplar = resolve_self_exemplar_settings(
            dial_bool("features.tree_self_exemplar", False),
            getattr(self, "_auto_run_decisions", None))

    def _start_step_run_context(self, layer, prompt, tiles, geo_transform, crs_authid, exemplar_payload) -> None:




        from ...core.activation_manager import auth_revision
        from ...core.detection_history import account_history_dir
        from ...core.raster_dataset_cache import dataset_identity

        self._auto_run_ctx = {
            "auth_revision": getattr(self, "_auto_start_auth_revision", auth_revision()),
            "account_dir": account_history_dir(),
            "tiles": tiles,
            "geo_transform": geo_transform,
            "crs_authid": crs_authid,
            "prompt": prompt,
            "layer_id": layer.id(),
            "layer_source": layer.source(),
            "layer_provider": layer.providerType(),
            "layer_dataset_identity": dataset_identity(layer.source()),
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
        self._auto_tile_bridge = TileRenderBridge(
            layer, geo_transform,
            floor_ratio=getattr(self, "_auto_imagery_floor_ratio", 0.0),
            transform_context=getattr(self, "_auto_transform_context", None))
        self._auto_imagery_floor_ratio = 0.0

    def _start_step_launch_dials(self, layer, prompt, tiles, has_exemplars) -> tuple:


        from ...core import detection_policy





        recall_text = detection_policy.recall_floor()
        recall_exemplar = detection_policy.recall_floor_exemplar_only()
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
        return detection_threshold, return_semantic, client_meta, density_probe

    def _start_step_announce_started(self, layer, tiles, prompt, forced) -> None:

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
