






from __future__ import annotations

from qgis.core import QgsCoordinateReferenceSystem


class AutoObjectBuildMixin:




    def _capture_auto_mask_gsd(self, worker) -> None:







        obs = getattr(worker, "observed_mask_gsd", 0.0)
        if obs > 0:
            self._auto_mask_gsd = obs
        self._auto_skipped_blank_tiles = int(
            getattr(worker, "tiles_skipped_blank", 0) or 0)
        self._auto_render_failed_tiles = int(
            getattr(worker, "tiles_render_failed", 0) or 0)
        self._auto_unavailable_tiles = int(
            getattr(worker, "tiles_unavailable", 0) or 0)



        self._auto_prefiltered_tiles = int(
            getattr(worker, "tiles_prefiltered", 0) or 0)



        self._auto_gate_skipped_tiles = int(
            getattr(worker, "tiles_gate_skipped", 0) or 0)




        self._auto_blob_dropped = int(
            getattr(worker, "masks_dropped_whole_tile", 0) or 0)



        self._auto_blob_armed = int(
            getattr(worker, "masks_whole_tile_armed", 0) or 0)
        self._auto_blob_kept_map = int(
            getattr(worker, "masks_whole_tile_kept_map", 0) or 0)
        self._auto_blob_map_lowscore = int(
            getattr(worker, "masks_dropped_map_lowscore", 0) or 0)


        self._auto_map_cover_scores = list(
            getattr(worker, "map_cover_scores", ()) or ())
        self._auto_blob_split = (
            int(getattr(worker, "masks_dropped_hard_cover", 0) or 0),
            int(getattr(worker, "masks_dropped_tile_span", 0) or 0),
            int(getattr(worker, "masks_dropped_not_compact", 0) or 0),
        )



        raw_total = int(getattr(worker, "raw_detections_total", 0) or 0)
        if raw_total > self._auto_raw_count:
            self._auto_raw_count = raw_total




        self._auto_dense_tiles = int(
            getattr(worker, "tiles_capped_final", 0) or 0)
        self._auto_subdiv_tiles = int(
            getattr(worker, "tiles_subdivided", 0) or 0)

    def _auto_blob_guard_stats(self) -> tuple[int, int, int]:








        armed = int(getattr(self, "_auto_blob_armed", 0) or 0)
        dropped = int(getattr(self, "_auto_blob_dropped", 0) or 0)
        area = float(getattr(self, "_auto_tile_ground_area", 0.0) or 0.0)
        if area <= 0.0 and self._auto_gsd > 0:
            from ...core.tile_manager import TILE_SIZE
            area = (TILE_SIZE * self._auto_gsd) ** 2
        return armed, dropped, int(round(area ** 0.5)) if area > 0.0 else 0

    def _auto_refine_pixel_size(self) -> float:















        worker = self._auto_worker
        if worker is not None:
            obs = getattr(worker, "observed_mask_gsd", 0.0)
            if obs > 0:
                return obs
        obs = getattr(self, "_auto_mask_gsd", 0.0)
        if obs > 0:
            return obs
        if self._auto_gsd > 0:
            return self._auto_gsd
        return self._auto_source_pixel_size()

    def _auto_source_pixel_size(self) -> float:



        source_layer = self._get_active_raster_layer()
        try:
            if source_layer is not None:
                ext = source_layer.extent()
                w = source_layer.width()
                if w > 0 and ext.width() > 0:





                    in_run = self._layer_extent_in_run_crs(
                        source_layer, getattr(self, "_auto_crs_authid", "") or "")
                    if in_run is not None and in_run.width() > 0:
                        ext = in_run
                    return ext.width() / w
        except (RuntimeError, AttributeError):
            pass
        return 1.0

    def _make_auto_area_measurer(self):


        try:
            from ...core.layer_conventions import make_area_measurer
            crs = QgsCoordinateReferenceSystem(self._auto_crs_authid or "EPSG:4326")
            return make_area_measurer(crs)
        except Exception:  # noqa: BLE001
            return None

    def _object_area_m2(self, geom, measurer) -> float:


        try:
            if measurer is not None:
                return float(measurer.measureArea(geom))
            return float(geom.area())
        except (RuntimeError, AttributeError):
            try:
                return float(geom.area())
            except (RuntimeError, AttributeError):
                return 0.0

    def _review_noise_floor(self) -> float:






        from ...core.detection_policy import review_noise_floor
        return review_noise_floor()

    def _auto_fp_rules(self) -> list:







        prompt = str((self._auto_run_ctx or {}).get("prompt") or "").strip()
        if not prompt:
            return []
        try:
            from ...core.detection_policy import fp_rules
            from ...core.review_presets import shape_class_for
            return fp_rules(shape_class_for(prompt))
        except Exception:  # noqa: BLE001
            return []

    def _object_is_fp(self, geom, area: float, rules: list, measurer) -> bool:




        if not rules:
            return False
        try:
            from ...core.geometry_attrs import matches_drop_rule, polygon_attributes
            return matches_drop_rule(
                polygon_attributes(geom, area_m2=area, measurer=measurer), rules)
        except Exception:  # noqa: BLE001
            return False



    def _auto_footprint_align_sweep(self, rows):






        try:
            if not rows:
                return None
            prompt = str((self._auto_run_ctx or {}).get("prompt") or "").strip()
            if not prompt:
                return None
            from ...core.detection_policy import auto_regularize_settings
            from ...core.review_presets import shape_class_for
            settings = auto_regularize_settings(shape_class_for(prompt))
            if settings is None:
                return None
            from ...core.footprint_alignment import (
                FootprintAlignSweep,
                compile_alignment_params,
                run_frame_scale,
            )
            scale = run_frame_scale(
                rows, self._make_auto_area_measurer(),
                self._auto_crs_authid or "")
            if scale is None:
                return None
            gsd_m = self._auto_refine_pixel_size() * (scale[0] * scale[1]) ** 0.5
            params = compile_alignment_params(settings, gsd_m)
            return FootprintAlignSweep(rows, params, scale)
        except Exception:  # noqa: BLE001
            return None

    def _align_auto_footprints_now(self, rows, max_objects: int = 0) -> list:









        if max_objects > 0 and len(rows) > max_objects:
            try:
                from qgis.core import Qgis, QgsMessageLog
                QgsMessageLog.logMessage(
                    f"Auto detection: footprint alignment skipped on "
                    f"{len(rows)} objects (over the {max_objects} this caller "
                    f"can wait for)",
                    "AI Segmentation", level=Qgis.MessageLevel.Info)
            except Exception:  # noqa: BLE001  # nosec B110
                pass
            return rows
        sweep = self._auto_footprint_align_sweep(rows)
        if sweep is None:
            return rows
        try:
            while not sweep.step(256):
                pass
            self._log_footprint_alignment(sweep)
            return sweep.result()
        except Exception:  # noqa: BLE001
            return rows

    def _log_footprint_alignment(self, sweep) -> None:

        try:
            from qgis.core import Qgis, QgsMessageLog
            QgsMessageLog.logMessage(
                f"Auto detection: footprint alignment kept "
                f"{sweep.aligned_count} shape(s), reverted "
                f"{sweep.reverted_count} ({sweep.simplified_count} of them "
                f"simplified), skipped {sweep.skipped_count}, "
                f"{sweep.circle_count} circle(s)",
                "AI Segmentation", level=Qgis.MessageLevel.Info)
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _build_auto_objects(self, merged_ided) -> list:







        measurer = self._make_auto_area_measurer()
        floor = self._review_noise_floor()
        fp_rules = self._auto_fp_rules()
        out = []
        fids = []
        for fid, geom, score in merged_ided:
            if geom is None or geom.isEmpty():
                continue
            if float(score) < floor:
                continue
            area = self._object_area_m2(geom, measurer)
            if self._object_is_fp(geom, area, fp_rules, measurer):
                continue
            out.append((geom, float(score), area))
            fids.append(fid)
        self._auto_object_fids = fids
        return out

    def _object_fid_for(self, idx: int) -> int:




        fids = getattr(self, "_auto_object_fids", None)
        if fids is not None and 0 <= idx < len(fids):
            return fids[idx]
        return idx
