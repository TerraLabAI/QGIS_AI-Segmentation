










from __future__ import annotations

import logging
import time

from .convert_pool import _convert_failure_reason

__all__ = [
    "AutoMaskGeometryMixin",
    "_MAX_MASKS_PER_TILE",
    "logger",
]

logger = logging.getLogger(__name__)






_MAX_MASKS_PER_TILE = None


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
        from ...core.tile_filter_answer import (
            STEP_WHOLE_TILE,
            read_tile_filters,
            tile_saturated,
        )







        tile_w, tile_h = self._tile_outsize.get(tile_idx, (tile_w, tile_h))


        self._release_tile_clean_image(tile_idx)

        self._note_flowing()

        land_cover = bool(getattr(self, "_land_cover", False))


        verdict = read_tile_filters(response)
        if verdict.raw_count is not None:
            decoded_count = verdict.raw_count
        else:


            decoded_count = detection_mask_count(response, self._score_threshold)
        self.masks_received += int(decoded_count or 0)
        if not decoded_count:
            self.tiles_answered_empty += 1
        if self.first_answer_mono is None:
            self.first_answer_mono = time.monotonic()
        if not verdict.filtered and not land_cover:
            self.tiles_filters_missing += 1
        if verdict.filtered:
            self._fold_tile_verdict(verdict)

        self._density_note_count(tile_idx, decoded_count)


        if self._tile_depth.get(tile_idx, 0) == 0:
            self._paid_tiles_done += 1
            paid_grid_done = not self._resplit_deadline and self._paid_tiles_done >= self._paid_tiles_total
            if paid_grid_done and self._resplit_time_ratio > 0:
                spent = max(0.0, time.monotonic() - self._run_started_at)
                self._resplit_deadline = (
                    time.monotonic() + spent * self._resplit_time_ratio)



        resplit = False

        if not land_cover and tile_saturated(verdict, decoded_count, self._max_masks):
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


            "whole_tile_done": STEP_WHOLE_TILE in verdict.applied,





            "stamp": self._tile_stamp_norm.get(tile_idx),
        }

    def _convert_completed(self, job: dict) -> list:












        from ...core.mask_crops import iter_detection_crops

        if getattr(self, "_land_cover", False):
            return self._land_cover_tile_payload(job)




        mask_iter = iter_detection_crops(
            job["response"], job["tile_w"], job["tile_h"],
            self._score_threshold, strict=True,
        )
        convert_t0 = time.monotonic()
        out = self._detections_to_geoms(
            self._iter_kept_masks(mask_iter, job.get("stamp")),
            job["transform"],
            whole_tile_done=bool(job.get("whole_tile_done")),
        )
        with self._stat_lock:
            self.phase_convert_s += time.monotonic() - convert_t0
        return out

    def _land_cover_tile_payload(self, job: dict) -> list:





        from ...core.land_cover import (
            decode_tile_labels,
            normalize_legend,
            response_mask_classes,
        )

        response = job["response"]
        if not any(c is not None for c in response_mask_classes(response)):
            return []
        tile_idx = job["tile_idx"]
        rect = self._tiles[tile_idx] if 0 <= tile_idx < len(self._tiles) else None
        if rect is None:
            return []
        _x, _y, w, h = rect
        labels = decode_tile_labels(response, int(w), int(h))
        return [{
            "land_cover_rect": tuple(int(v) for v in rect),
            "labels": labels,
            "class_legend": normalize_legend(response.get("class_legend")),
            "preview": _land_cover_preview(labels, job["transform"]),
        }]

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

    def _iter_kept_masks(self, mask_iter, stamp):
















        for mask, score, box in mask_iter:
            if stamp and self._centroid_in_stamp(box, mask, stamp):
                continue
            yield (mask, score)

    def _fold_tile_verdict(self, verdict) -> None:


        dropped = verdict.dropped
        hard = dropped.get("hard_cover", 0)
        span = dropped.get("tile_span", 0)
        shape = dropped.get("not_compact", 0)
        with self._stat_lock:
            self.masks_dropped_whole_tile += hard + span + shape
            self.masks_whole_tile_armed += verdict.armed
            self.masks_dropped_hard_cover += hard
            self.masks_dropped_tile_span += span
            self.masks_dropped_not_compact += shape
            self.masks_whole_tile_kept_map += verdict.kept_map
            self.masks_dropped_map_lowscore += dropped.get("map_lowscore", 0)
            self.map_cover_scores.extend(verdict.map_cover_scores)

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

    def _detections_to_geoms(self, kept, tile_transform,
                             whole_tile_done: bool = False) -> list:












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
        from ...core.polygon_trace import trace_crops_rings
        from ...core.tile_filter_answer import mask_spans_tile

















        length_scale = self._ground_length_scale()
        min_keep_area = (
            max((getattr(self, "_min_keep_px", 1.0) * self._gsd) ** 2,
                getattr(self, "_min_keep_floor_m2", 0.0) / self._ground_area_scale())
            if self._gsd > 0 else 0.0
        )



        bbox = tile_transform.get("bbox", (0.0, 1.0, 0.0, 1.0))
        ground_w = float(bbox[1] - bbox[0])
        ground_h = float(bbox[3] - bbox[2])




        clip_geom, clip_engine = self._clip_for_thread()

        keep_margin = (
            float(getattr(self, "_zone_keep_margin_m", 0.0) or 0.0)
            / length_scale if length_scale > 0 else 0.0)
        observed_cell = 0.0



        n_blob_span = 0


        span_test = (self._merge_separate and not self._collect_raw
                     and not whole_tile_done)
        out = []







        pending_crops: dict = {}
        pending_meta: dict = {}


        pending_outlines: dict = {}


        queued: list = []
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



            if span_test and mask_spans_tile(col0, col1, row0, row1, full_w, full_h):
                n_blob_span += 1
                continue






            sub = crop.padded()













            fill_limit = pinhole_fill_limit_px(
                self._gsd * length_scale, cell * length_scale, self._pinhole_m)




            key = (
                (full_h, full_w),
                tile_simplify_tolerance(
                    self._gsd, cell, self._tile_simplify_mult),
            )
            queued.append((sub, row0, col0, key, float(score), fill_limit))












        outlines = trace_crops_rings([q[0] for q in queued], saddles=False)
        for (sub, row0, col0, key, score, fill_limit), outline in zip(
                queued, outlines):
            if outline is not None and outline.has_holes:
                outline = outline.after_pinhole_fill(
                    fill_limit,
                    lambda unfilled, limit=fill_limit: fill_small_holes(unfilled, limit))
            if outline is not None:
                if outline.pixels is None:
                    sub = sub.astype(np.uint8)
            elif crop_has_no_holes(sub):
                sub = sub.astype(np.uint8)
            else:
                sub = fill_small_holes(sub, fill_limit)
            pending_crops.setdefault(key, []).append((sub, (row0 - 1, col0 - 1)))
            pending_meta.setdefault(key, []).append(score)
            pending_outlines.setdefault(key, []).append(outline)



        path_counts: dict = {}
        for key, crops in pending_crops.items():
            full_shape, simplify_tolerance = key
            polygon_lists = masks_to_polygons_packed(
                crops, tile_transform, full_shape,
                simplify_tolerance=simplify_tolerance,
                outlines=pending_outlines[key],


                skip_below_area=min_keep_area,
                path_counts=path_counts,


                max_side=getattr(self, "_pack_max_side", None),
            )
            for score, geoms in zip(pending_meta[key], polygon_lists):
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
                        if not inside and keep_margin > 0.0:




                            try:
                                gap = geom.distance(clip_geom)
                            except Exception:  # noqa: BLE001
                                gap = -1.0
                            if 0.0 <= gap <= keep_margin:
                                inside = True
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
                    out.append((geom, score))






        with self._stat_lock:
            self.raw_detections_total += len(out)
            self.masks_dropped_whole_tile += n_blob_span
            self.masks_dropped_tile_span += n_blob_span
            if observed_cell > self.observed_mask_gsd:
                self.observed_mask_gsd = observed_cell
            if path_counts:
                self.polygonized_gdal += path_counts.get("gdal", 0)
                self.polygonized_tracer += path_counts.get("tracer", 0)
                self.polygonized_fallback += path_counts.get("fallback", 0)
                self.polygonized_fallback_fast += path_counts.get("fallback_fast", 0)
















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
        from ...core.detection_policy import merge_scalar_kwargs
        from ...core.polygon_exporter import IncrementalMerger

        merger = IncrementalMerger(
            seam_min_dim=self._seam_min_dim,
            select_duplicates=False,
            gsd=self._gsd,
            **merge_scalar_kwargs(IncrementalMerger, self._merge_scalars),
        )
        for geom, score in out:
            merger.add(geom, float(score))
        return merger.result_scored()

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


def _land_cover_preview(labels, tile_transform: dict, step: int = 4) -> list:



    try:
        from ...core.land_cover import class_patches_wkb

        minx, maxx, miny, maxy = tile_transform["bbox"]
        h, w = labels.shape
        px_w = (maxx - minx) / max(1, w) * step
        px_h = (maxy - miny) / max(1, h) * step
        return class_patches_wkb(labels[::step, ::step],
                                 (minx, px_w, 0.0, maxy, 0.0, -px_h))
    except Exception:  # noqa: BLE001
        return []
