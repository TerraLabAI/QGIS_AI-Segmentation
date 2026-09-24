










from __future__ import annotations

import itertools
import logging
import time
from collections import deque

__all__ = [
    "AutoTileRenderMixin",
    "_PREFETCH_DEPTH",
    "_PREFETCH_HOLDOFF_S",
    "_RENDER_RETRY_DELAY_S",
    "_RENDER_RETRY_MAX",
    "_RENDER_SLOW_S",
    "logger",
]

logger = logging.getLogger(__name__)









_PREFETCH_DEPTH = 2












_RENDER_RETRY_MAX = 3


_RENDER_RETRY_DELAY_S = 1.5






_PREFETCH_HOLDOFF_S = 4.0









_RENDER_SLOW_S = 8.0


def _repeat_single_positive(valid: list) -> list:





    positives = [v for v in valid if v[0] is not None and v[1] == 1 and not v[4]]
    if len(positives) != 1:
        return valid
    try:
        from ...core.detection_policy import exemplar_single_repeat  # noqa: PLC0415
        repeat = exemplar_single_repeat(1)
    except Exception:  # noqa: BLE001
        repeat = 1
    crop, label, obj_box, _full, _region = positives[0]
    return valid + [(crop, label, obj_box, None, False)] * (repeat - 1)


class AutoTileRenderMixin:


    def _prepare_stamps(self) -> None:














        self._stamps = []
        self._stamp_full_boxes = []
        self._stamp_regions = []
        from qgis.PyQt.QtCore import Qt as _Qt

        from ...core.cloud_detection import stamp_size_cap

        if not self._exemplar_stamps_in:
            return








        valid: list = []
        for item in self._exemplar_stamps_in:
            if len(item) >= 3:
                crop, label, obj_box = item[0], item[1], item[2]
            else:
                crop, label = item[0], item[1]
                obj_box = None
            full_box = item[3] if len(item) > 3 else None
            region = bool(item[4]) if len(item) > 4 else False
            if crop is not None and crop.isNull():
                crop = None
            if region:
                crop = None
            if crop is None and full_box is None:

                continue
            valid.append((crop, int(label), obj_box, full_box, region))
        if not valid:
            return
        valid = _repeat_single_positive(valid)










        cap = stamp_size_cap(len(valid))
        for crop, label, obj_box, full_box, region in valid:
            if crop is not None and max(crop.width(), crop.height()) > cap:
                prev_w = crop.width()
                crop = crop.scaled(
                    cap, cap,
                    _Qt.AspectRatioMode.KeepAspectRatio,
                    _Qt.TransformationMode.SmoothTransformation)

                if obj_box is not None and prev_w > 0:
                    s = crop.width() / prev_w
                    obj_box = [float(v) * s for v in obj_box]
            self._stamps.append((crop, int(label), obj_box))
            self._stamp_full_boxes.append(full_box)
            self._stamp_regions.append(region)
        self._resolve_top_row_band_edge()

    def _split_stamps_for_tile(self, tx: int, ty: int, tw: int, th: int,
                               bottom: bool) -> tuple[list, list]:












        from ...core.cloud_detection import in_situ_exemplar_box, region_exemplar_box
        from ...core.tile_manager import OVERLAP_FRACTION, TILE_SIZE

        paste: list = []
        in_situ: list = []
        region_boxes: list = []
        for i, stamp in enumerate(self._stamps):
            full_box = (self._stamp_full_boxes[i]
                        if i < len(self._stamp_full_boxes) else None)
            if i < len(self._stamp_regions) and self._stamp_regions[i]:



                local = region_exemplar_box(full_box, tx, ty, tw, th)
                if local is not None:
                    region_boxes.append((stamp, local))
                continue
            if stamp[0] is not None and int(stamp[1]) == 1:
                paste.append(stamp)
                continue
            local = in_situ_exemplar_box(full_box, tx, ty, tw, th)
            if local is None:



                if stamp[0] is not None:
                    paste.append(stamp)
            else:
                in_situ.append((stamp, local))
        if paste and in_situ:


            band_h = min(th, int(TILE_SIZE * OVERLAP_FRACTION))
            kept = []
            for stamp, local in in_situ:
                under_band = (
                    local[3] > th - band_h if bottom else local[1] < band_h)
                if under_band:



                    if stamp[0] is not None:
                        paste.append(stamp)
                else:
                    kept.append((stamp, local))
            in_situ = kept
        if paste and region_boxes:



            band_h = min(th, int(TILE_SIZE * OVERLAP_FRACTION))
            shaved = []
            for stamp, local in region_boxes:
                x0, y0, x1, y1 = local
                if bottom:
                    y1 = min(y1, float(th - band_h))
                else:
                    y0 = max(y0, float(band_h))
                if (x1 - x0) >= 8.0 and (y1 - y0) >= 8.0:
                    shaved.append((stamp, [x0, y0, x1, y1]))
            region_boxes = shaved
        boxes = [
            {"box": [float(v) for v in local], "label": int(stamp[1])}
            for stamp, local in in_situ + region_boxes
        ]
        return paste, boxes

    def _resolve_top_row_band_edge(self) -> None:











        self._top_stamp_ty = None
        self._stamp_bottom_top_row = False
        if not self._stamps or not self._tiles:
            return
        from ...core.cloud_detection import top_row_bottom_stamp_ok

        tys = sorted({int(t[1]) for t in self._tiles})
        if len(tys) < 2:
            return
        top_ty = tys[0]
        self._top_stamp_ty = top_ty
        top_th = max(int(t[3]) for t in self._tiles if int(t[1]) == top_ty)
        next_ty = tys[1]
        pasted = [crop for crop, _l, _b in self._stamps if crop is not None]
        if not pasted:

            self._top_stamp_ty = None
            return
        band_content_h = max(int(crop.height()) for crop in pasted)
        self._stamp_bottom_top_row = top_row_bottom_stamp_ok(
            top_ty, top_th, next_ty, band_content_h)

    def _pump_render_deferred(self, pending: deque) -> None:



        if not self._render_deferred:
            return
        now = time.monotonic()
        matured = [e for e in self._render_deferred if e[0] <= now]
        if not matured:
            return
        for entry in matured:
            self._render_deferred.remove(entry)
            pending.appendleft((entry[1], entry[2]))

    def _request_render_prefetch(self, pending: deque) -> None:











        if self._render_request is None or self._stop_requested:
            return
        if time.monotonic() < getattr(self, "_prefetch_holdoff_until", 0.0):
            return


        width = self._render_window.cap




        limit = 1 if self._render_ramp_pending else width


        for tile_idx, spec in list(itertools.islice(pending, width)):
            if len(self._prefetched) >= limit:
                return
            if tile_idx in self._prefetched:
                continue





            if self._tile_needs_no_render(tile_idx):
                continue
            tx, ty, tw, th = spec
            out_w, out_h = self._tile_outsize.get(tile_idx, (0, 0))
            seq = self._render_request(tx, ty, tw, th, out_w, out_h)
            if seq is None:
                return
            self._prefetched[tile_idx] = seq
            ahead = self._encode_ahead_for_run()
            if ahead is not None and not (out_w and out_h):
                ahead.expect(seq, tile_idx, tw, th, self._render_ready)

    def _open_render_ramp(self) -> None:



        self._render_ramp_pending = False
        pending = self._stream_pending
        if pending is not None:
            self._request_render_prefetch(pending)

    def _tile_needs_no_render(self, tile_idx: int) -> bool:




        return (
            tile_idx in self._gate_skip
            or tile_idx in self._prefilter_skip
            or tile_idx in self._gate_tile_bytes
        )

    def _pop_next_pending(self, pending: deque, allow_wait: bool = True) -> tuple | None:






















        ready = self._render_ready
        if ready is not None and pending:
            width = self._render_window.cap
            waiting_on_a_render = False
            for tile_idx, spec in itertools.islice(pending, width):
                if self._tile_needs_no_render(tile_idx):
                    pending.remove((tile_idx, spec))
                    return tile_idx, spec
                seq = self._prefetched.get(tile_idx)
                if seq is None:
                    continue

                job = (self._encode_ahead.handed_over(tile_idx)
                       if self._encode_ahead is not None else None)
                if job.done() if job is not None else ready(seq):
                    pending.remove((tile_idx, spec))
                    return tile_idx, spec
                waiting_on_a_render = True
            if not allow_wait and waiting_on_a_render:
                return None
        return pending.popleft()

    def _note_render_health(self, waited_s: float, got_pixels: bool) -> None:










        window = getattr(self, "_render_window", None)
        if window is None:
            return
        if not got_pixels or waited_s >= self._render_slow_s:
            window.on_setback()
            self.renders_slow += 1
        else:
            window.on_clean_cycle()
        self.render_window_floor = min(self.render_window_floor, window.cap)

    def _encode_ahead_for_run(self):




        if self._encode_ahead is not None:
            return self._encode_ahead


        enabled = getattr(self, "_encode_ahead_enabled", None)
        if enabled is None:
            from ...core.server_dials import feature_enabled
            enabled = feature_enabled("encode_ahead")
            self._encode_ahead_enabled = enabled
        if not enabled:
            return None
        if self._render_bridge is None or self._stamps or self._stop_requested:
            return None
        try:
            from ...core.cloud_detection import encode_tile_png
            from ..tile_encode_ahead import TileEncodeAhead

            ahead = TileEncodeAhead(
                self._render_bridge.collect_render_timed, encode_tile_png)
            self._render_bridge.set_landed_hook(ahead.render_landed)
        except Exception:  # noqa: BLE001
            logger.debug("AutoDetectionWorker: no encode-ahead", exc_info=True)
            self._render_bridge = None
            return None
        self._encode_ahead = ahead
        return ahead

    def _close_encode_ahead(self) -> None:
        ahead, self._encode_ahead = self._encode_ahead, None
        if ahead is None:
            return
        try:
            if self._render_bridge is not None:
                self._render_bridge.set_landed_hook(None)
            ahead.close()
        except Exception:  # noqa: BLE001
            pass  # nosec B110

    def _discard_prefetch(self, tile_idx: int) -> None:










        seq = self._prefetched.pop(tile_idx, None)
        if seq is None or self._render_collect is None:
            return
        ahead = self._encode_ahead
        if ahead is not None and ahead.release(tile_idx, seq):
            return
        try:
            self._render_collect(seq)
        except Exception:  # nosec B110
            pass

    def _encode_or_defer(self, tile_idx: int, spec) -> tuple:

























        cached = self._gate_tile_bytes.pop(tile_idx, None)
        if cached is not None:


            self._discard_prefetch(tile_idx)
            return ("ok", cached)
        tx, ty, tw, th = spec
        encode_t0 = time.monotonic()
        self._last_render_wait_s = 0.0
        status, payload = self._encode_tile(tile_idx, tx, ty, tw, th)

        self.phase_encode_s += max(
            0.0, time.monotonic() - encode_t0 - self._last_render_wait_s)
        if status == "ok":
            self._render_attempts.pop(tile_idx, None)
            return ("ok", payload)
        if status == "empty":



            self.tiles_prefiltered += 1
            self._render_attempts.pop(tile_idx, None)
            return ("empty", None)
        if status in ("blank", "render", "unavailable"):
            self._prefetch_holdoff_until = time.monotonic() + self._prefetch_holdoff_s
        if status in ("blank", "render", "unavailable") and not self._stop_requested:
            attempts = self._render_attempts.get(tile_idx, 0)
            if attempts < self._render_retry_max:
                self._render_attempts[tile_idx] = attempts + 1
                delay = self._render_retry_delay_s * (2 ** attempts)
                self._render_deferred.append(
                    (time.monotonic() + delay, tile_idx, spec))
                return ("defer", None)

        if status == "blank":
            self.tiles_skipped_blank += 1
        elif status == "render":
            self.tiles_render_failed += 1
        elif status == "unavailable":
            self.tiles_unavailable += 1
        self._render_attempts.pop(tile_idx, None)
        return ("skip", None)

    def _encode_tile(self, tile_idx: int, tx: int, ty: int, tw: int, th: int):
























        from ...core.cloud_detection import (
            composite_tile_with_stamps,
            encode_tile_archive_copy,
            encode_tile_png,
            tile_is_blank,
            tile_is_degenerate,
            tile_is_unavailable,
            tile_png_to_base64,
        )

        try:






            if self._tile_renderer is None:
                return ("skip", None)
            if self._stop_requested:
                return ("skip", None)





            out_w, out_h = self._tile_outsize.get(tile_idx, (0, 0))



            prefetch_seq = self._prefetched.pop(tile_idx, None)
            render_t0 = time.monotonic()
            ahead_job = (self._encode_ahead.claim(tile_idx, prefetch_seq)
                         if self._encode_ahead is not None
                         and prefetch_seq is not None else None)
            ahead_render_s = 0.0
            ahead_encoded = None
            if ahead_job is not None:
                tile_img, ahead_render_s, ahead_encoded = ahead_job.result()
            elif prefetch_seq is not None and self._render_collect is not None:
                tile_img = self._render_collect(prefetch_seq)
            else:
                tile_img = self._tile_renderer(tx, ty, tw, th, out_w, out_h)
            if self._render_ramp_pending:
                self._open_render_ramp()
            got_pixels = tile_img is not None and not tile_img.isNull()
            if not got_pixels and self._stop_requested:





                return ("skip", None)
            render_wait_s = time.monotonic() - render_t0
            self._last_render_wait_s = render_wait_s
            self.phase_render_s += render_wait_s




            render_health_s = max(render_wait_s, ahead_render_s)
            if (ahead_job is None and prefetch_seq is not None
                    and self._render_duration is not None):
                try:
                    render_health_s = max(render_wait_s, float(self._render_duration()))
                except Exception:  # noqa: BLE001
                    render_health_s = render_wait_s
            self._note_render_health(render_health_s, got_pixels)
            if not got_pixels:




                return ("render", None)







            if (
                self._prefilter is not None and not (out_w and out_h) and tile_is_degenerate(
                    tile_img,
                    self._prefilter["nodata_frac"],
                    self._prefilter["band_eps"],
                    self._prefilter["nodata_rgb_eps"],
                    self._prefilter["min_valid_px"],
                )
            ):
                return ("empty", None)






            if tile_is_blank(tile_img):
                return ("blank", None)




            if self._skip_unavailable_tiles and tile_is_unavailable(tile_img):
                return ("unavailable", None)
            src_x, src_y = 0, 0

            if out_w and out_h:





                encoded = encode_tile_png(tile_img, 0, 0, out_w, out_h)
                if encoded is None:
                    return ("skip", None)
                _crop, data = encoded
                return ("ok", ((tx, ty, tw, th), data))

            if self._stamps:



                bottom = bool(
                    self._stamp_bottom_top_row and ty == self._top_stamp_ty)



                paste, insitu_boxes = self._split_stamps_for_tile(
                    tx, ty, tw, th, bottom)
                out = composite_tile_with_stamps(
                    tile_img, src_x, src_y, tw, th, paste, bottom=bottom)
                if out is None:
                    return ("skip", None)
                (_sx, _sy, cw, ch), data, ex_boxes, stamp_norm = out



                boxes = (ex_boxes + insitu_boxes)[:8]
                if boxes:
                    self._tile_exemplars[tile_idx] = boxes



                if stamp_norm:
                    self._tile_stamp_norm[tile_idx] = stamp_norm








                    if self._client_meta is not None:
                        clean = encode_tile_archive_copy(
                            tile_img, src_x, src_y, tw, th)
                        if clean is not None:
                            self._tile_clean_image[tile_idx] = (
                                tile_png_to_base64(clean))
                return ("ok", ((tx, ty, cw, ch), data))
            encoded = ahead_encoded
            if encoded is None:
                encoded = encode_tile_png(tile_img, src_x, src_y, tw, th)
            if encoded is None:
                return ("skip", None)
            (_sx, _sy, cw, ch), data = encoded
            return ("ok", ((tx, ty, cw, ch), data))
        except Exception as exc:
            logger.warning("AutoDetectionWorker: tile encode failed at (%d,%d): %s",
                           tx, ty, exc)
            return ("skip", None)
