










from __future__ import annotations

import logging
import time

from .convert_pool import _convert_failure_reason

__all__ = [
    "AutoMaskGeometryMixin",
    "_COMPACT_MIN_FILL",
    "_HARD_COVER_SHAPE_ESCAPE",
    "_HARD_TILE_COVERAGE",
    "_MASK_CAP_TRIGGER_FRAC",
    "_MAX_MASKS_PER_TILE",
    "_MAX_TILE_COVERAGE",
    "_MIN_KEEP_PX",
    "_TILE_SPAN_FRACTION",
    "logger",
]

logger = logging.getLogger(__name__)





_MAX_MASKS_PER_TILE = 200











_MASK_CAP_TRIGGER_FRAC = 0.95









_MAX_TILE_COVERAGE = 0.55





_HARD_TILE_COVERAGE = 0.80














_HARD_COVER_SHAPE_ESCAPE = True





_COMPACT_MIN_FILL = 0.85








_TILE_SPAN_FRACTION = 0.95





_MIN_KEEP_PX = 1.5


class AutoMaskGeometryMixin:


    def _emit_completed(
        self,
        response: dict,
        tile_idx: int,
        tile_w: int,
        tile_h: int,
        tile_transform: dict,
    ) -> bool:
















        job = self._plan_completed(
            response, tile_idx, tile_w, tile_h, tile_transform)
        try:
            detections = self._convert_completed(job)
        except Exception as exc:  # noqa: BLE001
            return self._settle_converted(False, job, exc)
        return self._settle_converted(True, job, detections)

    def _plan_completed(
        self,
        response: dict,
        tile_idx: int,
        tile_w: int,
        tile_h: int,
        tile_transform: dict,
    ) -> dict:








        from ...core.cloud_detection import detection_mask_count







        tile_w, tile_h = self._tile_outsize.get(tile_idx, (tile_w, tile_h))


        self._release_tile_clean_image(tile_idx)

        self._note_flowing()



        decoded_count = detection_mask_count(response, self._score_threshold)


        if self._tile_depth.get(tile_idx, 0) == 0:
            self._paid_tiles_done += 1
            paid_grid_done = not self._resplit_deadline and self._paid_tiles_done >= self._paid_tiles_total
            if paid_grid_done and self._resplit_time_ratio > 0:
                spent = max(0.0, time.monotonic() - self._run_started_at)
                self._resplit_deadline = (
                    time.monotonic() + spent * self._resplit_time_ratio)



        resplit = False
        if decoded_count >= self._mask_cap_trigger:
            self._hit_mask_cap = True
            self.tiles_mask_capped += 1






            resplit = self._maybe_subdivide(tile_idx)
            if not resplit:
                self.tiles_capped_final += 1
        return {
            "response": response,
            "tile_idx": tile_idx,
            "tile_w": tile_w,
            "tile_h": tile_h,
            "transform": tile_transform,
            "count": decoded_count,
            "resplit": resplit,
        }

    def _convert_completed(self, job: dict) -> list:












        from ...core.mask_crops import iter_detection_crops





        mask_iter = iter_detection_crops(
            job["response"], job["tile_w"], job["tile_h"],
            self._score_threshold, strict=True,
        )
        convert_t0 = time.monotonic()
        out = self._detections_to_geoms(
            self._iter_kept_masks(
                mask_iter, job["response"], job["tile_idx"],
                job["tile_w"], job["tile_h"], job["count"],
            ),
            job["transform"],
        )
        with self._stat_lock:
            self.phase_convert_s += time.monotonic() - convert_t0
        return out

    def _settle_converted(self, ok: bool, job: dict, payload) -> bool:






        tile_idx = job["tile_idx"]
        reason = ""
        if not ok:


            reason = _convert_failure_reason(payload)
            if not self._convert_fail_reason:
                self._convert_fail_reason = reason


            rescued = self._retry_convert_in_process(job)
            if rescued is not None:
                ok, payload = True, rescued



        self._settle_rescanning(tile_idx)
        if not ok:
            logger.warning(
                "AutoDetectionWorker: tile %d decode/convert failed: %s",
                tile_idx, payload,
            )
            self.tiles_convert_failed += 1
            self._emit_warning(
                f"Tile {tile_idx}: could not process result ({reason}); skipping"
            )
            return False

        detections = payload
        logger.debug(
            "AutoDetectionWorker: tile %d completed with %d detection(s)",
            tile_idx, len(detections),
        )
        if job["resplit"]:





            self._withheld[tile_idx] = detections
            detections = []
        else:
            parent = self._parent_of.get(tile_idx)
            if parent is not None and detections:


                self._parents_with_child_results.add(parent)

        self._note_tile_outcome(bool(detections))
        try:
            self.tile_completed.emit(tile_idx, detections)
        except RuntimeError:






            pass  # nosec B110
        return True

    def _iter_kept_masks(
        self, mask_iter, response: dict, tile_idx: int, tile_w: int, tile_h: int,
        instance_count: int,
    ):















        stamp = self._tile_stamp_norm.get(tile_idx)
        for mask, score, box in mask_iter:
            if stamp and self._centroid_in_stamp(box, mask, stamp):
                continue
            yield (mask, score)






        yield from self._semantic_rescue_masks(
            response, instance_count, tile_w, tile_h)

    def _semantic_rescue_masks(
        self, response: dict, instance_count: int, tile_w: int, tile_h: int,
    ) -> list:









        if not self._return_semantic:
            return []
        from ...core.cloud_detection import (
            decode_rle_to_mask,
            parse_semantic_fields,
            should_rescue_with_semantic,
        )

        rle, coverage, _presence = parse_semantic_fields(response)
        if not should_rescue_with_semantic(
            instance_count, coverage, rle is not None,
            self._return_semantic, self._semantic_coverage_floor,
        ):
            return []



        srv_w = response.get("width")
        srv_h = response.get("height")
        decode_w = int(srv_w) if srv_w is not None else tile_w
        decode_h = int(srv_h) if srv_h is not None else tile_h
        mask = decode_rle_to_mask(rle, decode_h, decode_w)
        if not mask.any():
            return []
        return [(mask, float(coverage))]

    def _make_clip_pair(self):





        if not self._clip_polygon_wkb:
            return None, None
        from qgis.core import QgsGeometry

        geom = QgsGeometry()
        geom.fromWkb(self._clip_polygon_wkb)
        if geom.isEmpty():



            logger.warning(
                "AutoDetectionWorker: zone clip polygon could not be rebuilt")
            self._emit_warning(
                "Zone clip could not be rebuilt; results may extend past the "
                "drawn zone")
            return None, None
        try:
            engine = QgsGeometry.createGeometryEngine(geom.constGet())
            engine.prepareGeometry()
        except Exception:  # noqa: BLE001
            engine = None
        return geom, engine

    def _build_clip_engine(self) -> None:





        self._clip_geom, self._clip_engine = self._make_clip_pair()

    def _clip_for_thread(self):








        if not self._clip_polygon_wkb:
            return None, None
        pair = getattr(self._clip_local, "pair", None)
        if pair is None:
            pair = self._make_clip_pair()
            self._clip_local.pair = pair
        return pair

    def _detections_to_geoms(self, kept, tile_transform) -> list:









        import numpy as np

        from ...core.cloud_detection import (
            mask_cell_size,
            pinhole_fill_limit_px,
            tile_simplify_tolerance,
        )
        from ...core.hypothesis_nms import select_tile_hypotheses
        from ...core.layer_conventions import repair_polygon, to_multipolygon
        from ...core.mask_crops import as_crop, crop_has_no_holes
        from ...core.polygon_exporter import (
            fill_small_holes,
            masks_to_polygons_packed,
        )



















        area_scale = self._ground_area_scale()
        length_scale = self._ground_length_scale()
        min_keep_area = (
            max((self._min_keep_px * self._gsd) ** 2,
                self._min_keep_floor_m2 / area_scale)
            if self._gsd > 0 else 0.0
        )



        bbox = tile_transform.get("bbox", (0.0, 1.0, 0.0, 1.0))
        ground_w = float(bbox[1] - bbox[0])
        ground_h = float(bbox[3] - bbox[2])




        clip_geom, clip_engine = self._clip_for_thread()
        observed_cell = 0.0






        n_blob_armed = 0
        n_blob_hard = 0
        n_blob_span = 0
        n_blob_shape = 0
        n_blob_kept_map = 0
        n_blob_map_lowscore = 0


        map_cover_scores: list[float] = []
        out = []







        pending_crops: dict = {}
        pending_meta: dict = {}
        for mask, score in kept:









            crop = as_crop(mask)
            full_h, full_w = crop.full_shape






            cell = mask_cell_size(ground_w, ground_h, full_w, full_h)
            if cell > observed_cell:
                observed_cell = cell
            set_pixels = crop.set_pixels
            if set_pixels == 0:
                continue
            row0, row1 = crop.row0, crop.row1
            col0, col1 = crop.col0, crop.col1








            coverage = set_pixels / float(full_h * full_w)
            blob_check = False


            if self._merge_separate and not self._collect_raw and coverage > self._max_tile_coverage:



                n_blob_armed += 1
                if (coverage > self._hard_tile_coverage
                        and not self._hard_cover_shape_escape):
                    n_blob_hard += 1
                    continue






                span = self._tile_span_fraction
                if (col1 - col0 + 1 >= span * full_w and row1 - row0 + 1 >= span * full_h):
                    n_blob_span += 1
                    continue
                blob_check = True
            elif coverage > self._max_tile_coverage and not self._collect_raw and (
                    self._map_cover_score_floor > 0.0
                    and float(score) < self._map_cover_score_floor):








                n_blob_map_lowscore += 1
                map_cover_scores.append(float(score))
                continue
            elif coverage > self._max_tile_coverage:










                n_blob_kept_map += 1
                map_cover_scores.append(float(score))






            sub = crop.padded()

















            if crop_has_no_holes(sub):
                sub = sub.astype(np.uint8)
            else:
                sub = fill_small_holes(
                    sub,
                    pinhole_fill_limit_px(
                        self._gsd * length_scale, cell * length_scale,
                        self._pinhole_m))




            key = (
                (full_h, full_w),
                tile_simplify_tolerance(
                    self._gsd, cell, self._tile_simplify_mult),
            )
            pending_crops.setdefault(key, []).append((sub, (row0 - 1, col0 - 1)))
            pending_meta.setdefault(key, []).append((float(score), blob_check))

        for key, crops in pending_crops.items():
            full_shape, simplify_tolerance = key
            polygon_lists = masks_to_polygons_packed(
                crops, tile_transform, full_shape,
                simplify_tolerance=simplify_tolerance,
            )
            for (score, blob_check), geoms in zip(pending_meta[key], polygon_lists):
                for geom in geoms:
                    if geom is None or geom.isEmpty():
                        continue







                    unclipped = clip_geom is None
                    if clip_geom is not None:
                        inside = False
                        if clip_engine is not None:
                            try:
                                inside = clip_engine.contains(geom.constGet())
                            except Exception:  # noqa: BLE001
                                inside = False
                        if not inside:
                            geom = geom.intersection(clip_geom)
                        if geom is None or geom.isEmpty() or geom.area() <= 0:
                            continue
                        unclipped = inside









                    if not unclipped:
                        geom = repair_polygon(geom) or geom
                    geom = to_multipolygon(geom)
                    if geom is None or geom.isEmpty():
                        continue




                    if min_keep_area > 0.0 and geom.area() < min_keep_area:
                        continue



                    if blob_check and not self._is_compact_shape(
                            geom, self._compact_min_fill):
                        n_blob_shape += 1
                        continue
                    out.append((geom, score))






        with self._stat_lock:
            self.raw_detections_total += len(out)
            self.masks_dropped_whole_tile += (
                n_blob_hard + n_blob_span + n_blob_shape)
            self.masks_whole_tile_armed += n_blob_armed
            self.masks_dropped_hard_cover += n_blob_hard
            self.masks_dropped_tile_span += n_blob_span
            self.masks_dropped_not_compact += n_blob_shape
            self.masks_whole_tile_kept_map += n_blob_kept_map
            self.masks_dropped_map_lowscore += n_blob_map_lowscore
            self.map_cover_scores.extend(map_cover_scores)
            if observed_cell > self.observed_mask_gsd:
                self.observed_mask_gsd = observed_cell
















        if self._merge_separate or self._collect_raw or self._map_hypothesis_nms:




            ms = self._merge_scalars
            sup_kwargs = {k: ms[k] for k in (
                "ios_threshold", "dup_ios_floor", "dup_centroid_frac") if k in ms}
            out = select_tile_hypotheses(out, **sup_kwargs)
        if not (self._merge_separate or self._collect_raw):









            out = self._premerge_map_fragments(out)
        return [(bytes(geom.asWkb()), score) for geom, score in out]

    def _premerge_map_fragments(self, out: list) -> list:








        if len(out) < 2:
            return out
        from ...core.polygon_exporter import IncrementalMerger

        ms = self._merge_scalars
        merge_kwargs = {k: ms[k] for k in (
            "merge_ios", "dedup_ios", "dup_ios_floor", "dup_centroid_frac",
            "seam_span_ios", "seam_span_tol", "jitter_area_frac",
            "score_floor_frac") if k in ms}
        merger = IncrementalMerger(
            seam_min_dim=self._seam_min_dim,
            select_duplicates=False,
            gsd=self._gsd,
            **merge_kwargs,
        )
        for geom, score in out:
            merger.add(geom, float(score))
        return merger.result_scored()

    @staticmethod
    def _is_compact_shape(geom, min_fill: float = _COMPACT_MIN_FILL) -> bool:








        try:
            _obb, obb_area, _angle, _w, _h = geom.orientedMinimumBoundingBox()
            if obb_area and obb_area > 0.0:
                return geom.area() / obb_area >= min_fill
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return False

    @staticmethod
    def _centroid_in_stamp(box, mask, stamp) -> bool:





        nx0, ny0, nx1, ny1 = stamp[0], stamp[1], stamp[2], stamp[3]
        if box and len(box) == 4 and (box[2] > 0 or box[3] > 0):
            cx, cy = box[0], box[1]
            return nx0 <= cx <= nx1 and ny0 <= cy <= ny1
        try:
            import numpy as np
            crop = getattr(mask, "crop", None)
            if crop is not None:


                ys, xs = np.nonzero(crop)
                ys = ys + mask.row0
                xs = xs + mask.col0
                shape = mask.full_shape
            else:
                ys, xs = np.nonzero(mask)
                shape = mask.shape
            if xs.size == 0:
                return False
            h = max(1, shape[0])
            w = max(1, shape[1])
            cx = float(xs.mean()) / w
            cy = float(ys.mean()) / h
            return nx0 <= cx <= nx1 and ny0 <= cy <= ny1
        except Exception:  # noqa: BLE001
            return False
