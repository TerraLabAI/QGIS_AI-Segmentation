







from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsGeometry,
    QgsMessageLog,
    QgsProject,
)

from ...core.i18n import tr


class AutoRunHandoffMixin:


    def _auto_gate_config(
        self, prompt: str, has_exemplars: bool, n_tiles: int
    ) -> dict | None:






        try:
            from ...core import detection_policy
            if has_exemplars or not (prompt or "").strip():
                return None
            if not detection_policy.gate_enabled():
                return None
            if n_tiles < detection_policy.gate_min_tiles():
                return None
            from ...core.review_presets import shape_class_for





            rule = detection_policy.gate_class_rule(
                detection_policy.gate_class_for_prompt(
                    prompt, shape_class_for(prompt)))
            if not rule:
                return None
            config = {
                "group": detection_policy.gate_group(),



                "max_group": rule.get(
                    "max_group", detection_policy.gate_max_group()),
                "min_score": rule["min_score"],
                "min_pixels": detection_policy.gate_min_pixels(),
            }
            if "max_scan_mupp" in rule:
                config["max_scan_mupp"] = rule["max_scan_mupp"]
            return config
        except Exception:  # noqa: BLE001  # nosec B110
            return None

    def _tel_detect_blocked(self, reason: str) -> None:

        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_detect_blocked(reason)
        except Exception:
            pass  # nosec B110

    def _auto_subdivide_budget(self, base_tiles: int, has_stamps: bool) -> int:






        if has_stamps:
            return 0




        from ...core.credit_gate import subdivide_budget, subdivide_cap
        from ...core.detection_policy import resplit_charge_every

        every = resplit_charge_every()
        if every <= 0:
            return subdivide_cap(base_tiles)
        credits, _is_free = self._auto_credit_snapshot()
        return subdivide_budget(credits, base_tiles, every)

    def _build_auto_client_meta(self) -> dict:







        import math

        from ...core import detection_policy



        prompt_mode = (
            "count" if getattr(self, "_auto_merge_separate", True) else "map")
        meta: dict = {
            "plugin_version": self._read_plugin_version(),
            "policy_rev": detection_policy.policy_rev(),
            "prompt_mode": prompt_mode,
        }
        basemap = self._auto_basemap_label()
        if basemap:
            meta["basemap"] = basemap
        zone = self._auto_zone_geojson()
        if zone is not None:
            meta["zone_geojson"] = zone

        zone_wkt = self._auto_zone_wkt_wgs84()
        if zone_wkt is not None:
            meta["zone_wkt"] = zone_wkt

        zone_km2 = self._auto_zone_area_km2()
        if math.isfinite(zone_km2) and zone_km2 > 0:
            meta["zone_km2"] = round(zone_km2, 4)

        try:
            _zone_area_m2, native_mupp = self._auto_run_plan_inputs()
        except Exception:  # noqa: BLE001
            native_mupp = None
        if (native_mupp is not None and math.isfinite(native_mupp)
                and native_mupp > 0):
            meta["native_mupp"] = round(float(native_mupp), 4)
        return meta

    def _auto_basemap_label(self) -> str | None:





        try:
            from ...core.basemap_label import detect_basemap_label

            layer_id = (getattr(self, "_auto_run_ctx", None) or {}).get("layer_id")
            layer = QgsProject.instance().mapLayer(layer_id) if layer_id else None
            if layer is None:
                layer = self._get_active_raster_layer()
            if layer is None:
                return None
            return detect_basemap_label(layer)
        except Exception:  # noqa: BLE001
            return None

    def _billable_zone_layer(self):









        try:
            if getattr(self, "_auto_worker", None) is not None:
                layer_id = (getattr(self, "_auto_run_ctx", None) or {}).get("layer_id")
                if layer_id:
                    layer = QgsProject.instance().mapLayer(layer_id)
                    if layer is not None:
                        return layer
            return self._get_active_raster_layer()
        except (RuntimeError, AttributeError):
            return None

    def _read_project_transform_context(self):






        try:
            from qgis.core import QgsCoordinateTransformContext, QgsProject
            return QgsCoordinateTransformContext(
                QgsProject.instance().transformContext())
        except (RuntimeError, AttributeError, TypeError):
            return None

    def _zone_clipped_to_data(self, geom, crs):








        try:
            layer = self._billable_zone_layer()
            authid = crs.authid() if crs is not None else ""
            if layer is None or not authid:
                return geom
            extent = self._layer_extent_in_run_crs(layer, authid)
            if extent is None or extent.isEmpty():
                return geom
            clipped = geom.intersection(QgsGeometry.fromRect(extent))
            if clipped is None or clipped.isEmpty() or clipped.area() <= 0:
                return geom
            return clipped
        except Exception:  # noqa: BLE001
            return geom

    def _auto_billable_zone_geometry(self):














        try:
            from qgis.core import QgsCoordinateReferenceSystem, QgsGeometry

            poly = getattr(self, "_auto_clip_polygon", None)
            authid = getattr(self, "_auto_crs_authid", None)
            if poly is not None and not poly.isEmpty() and authid:
                run_crs = QgsCoordinateReferenceSystem(authid)
                return self._zone_clipped_to_data(QgsGeometry(poly), run_crs), run_crs
            rect = getattr(self, "_auto_zone", None)
            if rect is None or rect.isEmpty():
                return None, None
            crs = self._zone_source_crs(rect)
            if crs is None or not crs.isValid():
                crs = (QgsCoordinateReferenceSystem(authid) if authid else None)
            if crs is None or not crs.isValid():
                return None, None



            drawn = getattr(self, "_auto_zone_polygon", None)
            if drawn is not None and not drawn.isEmpty():
                return self._zone_clipped_to_data(QgsGeometry(drawn), crs), crs
            return self._zone_clipped_to_data(QgsGeometry.fromRect(rect), crs), crs
        except Exception:  # noqa: BLE001
            return None, None

    def _auto_zone_wkt_wgs84(self) -> str | None:



        geom, src = self._auto_billable_zone_geometry()
        if geom is None or src is None:
            return None
        try:
            from qgis.core import (
                QgsCoordinateReferenceSystem,
                QgsCoordinateTransform,
                QgsProject,
            )

            from ...core.zone_crs_check import zone_fits_declared_crs

            dst = QgsCoordinateReferenceSystem("EPSG:4326")
            if not src.isValid() or not dst.isValid():
                return None



            if not zone_fits_declared_crs(geom, src):
                return None
            copy = geom
            if src != dst:
                from ...core.qt_compat import geometry_op_succeeded
                transform = QgsCoordinateTransform(
                    src, dst, QgsProject.instance().transformContext())
                if not geometry_op_succeeded(copy.transform(transform)):
                    return None





            from ...core.zone_antimeridian import fold_into_lonlat_range
            copy = fold_into_lonlat_range(copy)
            wkt = copy.asWkt(7)
            if not wkt or len(wkt) > 100_000:
                return None
            return wkt
        except Exception:  # noqa: BLE001
            return None

    def _auto_zone_geojson(self) -> dict | None:



        poly = getattr(self, "_auto_clip_polygon", None)
        if poly is None:
            return None
        try:
            import json

            return json.loads(poly.asJson(6))
        except Exception:  # noqa: BLE001
            return None

    def _build_auto_worker(
        self,
        *,
        tile_renderer=None,
        tiles: list,
        geo_transform: dict,
        crs_authid: str,
        prompt: str,
        auth: dict,
        run_id: str,
        max_concurrent: int = 6,
        detection_threshold: float = 0.30,
        progress_offset: int = 0,
        progress_total: int | None = None,
        exemplar_stamps: list | None = None,
        merge_scalars: dict | None = None,
        subdivide_budget: int = 0,
        collect_raw: bool = False,
        return_semantic: bool = False,
        gate_config: dict | None = None,
        mask_scale: int | None = None,
        client_meta: dict | None = None,
    ):








        from ...workers.auto_detection_worker import AutoDetectionWorker





        clip_polygon_wkb = None
        if self._auto_clip_polygon is not None:
            try:
                clip_polygon_wkb = bytes(self._auto_clip_polygon.asWkb())
            except (RuntimeError, AttributeError):
                clip_polygon_wkb = None

        return AutoDetectionWorker(
            tile_renderer=tile_renderer,
            tiles=tiles,
            geo_transform=geo_transform,
            crs_authid=crs_authid,
            prompt=prompt,
            auth=auth,
            run_id=run_id,
            max_concurrent=max_concurrent,
            detection_threshold=detection_threshold,
            exemplar_stamps=exemplar_stamps,
            progress_offset=progress_offset,
            progress_total=progress_total,
            clip_polygon_wkb=clip_polygon_wkb,
            gsd=self._auto_gsd,
            merge_separate=self._auto_merge_separate,
            seam_min_dim=self._auto_seam_min_dim(),
            merge_scalars=merge_scalars,
            subdivide_budget=subdivide_budget,
            collect_raw=collect_raw,
            return_semantic=return_semantic,
            gate_config=gate_config,
            mask_scale=(getattr(self, "_auto_mask_scale", 1)
                        if mask_scale is None else mask_scale),
            client_meta=client_meta,




            source_is_online=bool(getattr(self, "_auto_source_is_online", False)),


            transform_context=getattr(self, "_auto_transform_context", None),
        )

    def _wind_down_unstarted_auto_worker(self, worker) -> None:

        if self._auto_worker is worker:
            self._auto_worker = None
        self._auto_cancelled_slot = None
        if self.dock_widget is not None:
            try:
                self.dock_widget.set_auto_run_active(False)
                self.dock_widget.set_auto_status("idle")
                self._set_zone_badge_enabled(True)
            except (RuntimeError, AttributeError):
                pass
        self._reset_auto_live_pipeline()
        self._drop_auto_tile_bridge()

    def _launch_auto_worker(
        self,
        *,
        tile_renderer=None,
        tiles: list,
        geo_transform: dict,
        crs_authid: str,
        prompt: str,
        auth: dict,
        run_id: str,
        max_concurrent: int = 6,
        detection_threshold: float = 0.30,
        progress_offset: int = 0,
        progress_total: int | None = None,
        exemplar_stamps: list | None = None,
        merge_scalars: dict | None = None,
        subdivide_budget: int = 0,
        collect_raw: bool = False,
        return_semantic: bool = False,
        gate_config: dict | None = None,
        client_meta: dict | None = None,
    ) -> None:








        self._last_auto_result = None
        self._auto_quota_stop_banner = None
        self._auto_worker = self._build_auto_worker(
            tile_renderer=tile_renderer,
            tiles=tiles,
            geo_transform=geo_transform,
            crs_authid=crs_authid,
            prompt=prompt,
            auth=auth,
            run_id=run_id,
            max_concurrent=max_concurrent,
            detection_threshold=detection_threshold,
            progress_offset=progress_offset,
            progress_total=progress_total,
            exemplar_stamps=exemplar_stamps,
            merge_scalars=merge_scalars,
            subdivide_budget=subdivide_budget,
            collect_raw=collect_raw,
            return_semantic=return_semantic,
            gate_config=gate_config,



            mask_scale=getattr(self, "_auto_mask_scale", 1),
            client_meta=client_meta,
        )
        worker = self._auto_worker



        self._start_auto_stitcher(geo_transform)






        from qgis.PyQt.QtCore import Qt
        _queued = Qt.ConnectionType.QueuedConnection
        self._auto_worker.tile_completed.connect(self._on_auto_tile_completed, _queued)
        self._auto_worker.all_tiles_finished.connect(self._on_auto_all_finished, _queued)
        self._auto_worker.progress.connect(self._on_auto_progress, _queued)
        self._auto_worker.warning.connect(self._on_auto_warning, _queued)
        self._auto_worker.nothing_found_yet.connect(
            self._on_auto_nothing_found_yet, _queued)
        self._auto_worker.error.connect(self._on_auto_error, _queued)
        self._auto_worker.credits_exhausted.connect(self._on_auto_credits_exhausted, _queued)



        self._auto_cancelled_slot = (
            lambda w=self._auto_worker: self._on_auto_cancelled(worker=w))
        self._auto_worker.cancelled.connect(self._auto_cancelled_slot, _queued)
        self._auto_worker.queue_state.connect(self._on_auto_queue_state, _queued)
        self._auto_worker.rescan_state.connect(self._on_auto_rescan_state, _queued)
        self._auto_worker.run_phase.connect(self._on_auto_run_phase, _queued)






        total_display = progress_total or len(tiles)


        if self.dock_widget is not None:
            try:
                self.dock_widget.set_auto_run_active(True)
                self._set_zone_badge_enabled(False)




                self.dock_widget.set_auto_billed_tile_total(total_display)
                self.dock_widget.set_auto_tile_progress(progress_offset, total_display)

                self.dock_widget.auto_detect_btn.setText(tr("Detect objects"))
                self.dock_widget.set_auto_status("progress")
            except (RuntimeError, AttributeError):
                QgsMessageLog.logMessage(
                    "Auto detection: the panel went away before the run "
                    "started; aborting before any tile is sent",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning,
                )



                try:
                    self.dock_widget.set_auto_run_active(False)
                    self._set_zone_badge_enabled(True)
                except (RuntimeError, AttributeError):
                    pass
                self._auto_worker = None
                self._auto_cancelled_slot = None
                self._reset_auto_live_pipeline()
                self._drop_auto_tile_bridge()
                return

        if (worker is None or self._auto_worker is not worker
                or bool(getattr(worker, "_stop_requested", False))):
            self._wind_down_unstarted_auto_worker(worker)
            return
        worker.start()


        import time as _time
        self._auto_run_started_mono = _time.monotonic()


        try:
            from ...core.run_log_capture import note_run_id
            from .auto_client_profile import reset_run_profile
            reset_run_profile(self)
            note_run_id(self._auto_run_id or "")
        except Exception:  # noqa: BLE001
            pass  # nosec B110

        self._start_auto_stall_watchdog()




        self._pause_preview_jobs()
