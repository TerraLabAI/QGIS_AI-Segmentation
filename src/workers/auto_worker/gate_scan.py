









from __future__ import annotations

import logging
import time
from collections import deque

from qgis.core import Qgis

__all__ = [
    "AutoGateScanMixin",
    "_GATE_RENDER_CACHE_MAX",
    "_GATE_SCAN_RENDER_TRIES",
    "logger",
]

logger = logging.getLogger(__name__)








_GATE_RENDER_CACHE_MAX = 64


_GATE_SCAN_RENDER_TRIES = 2


class AutoGateScanMixin:






    def gate_summary(self) -> dict:


        return dict(self._gate_stats)

    def _gate_ground_mupp(self) -> float | None:


        try:
            tx, ty, tw, th = self._tiles[0]
            if tw <= 0:
                return None
            transform = self._make_tile_transform(tx, ty, tw, th)
            width_m = self._bbox_ground_width_m(transform["bbox_native"])
            if width_m is None or width_m <= 0:
                return None
            return width_m / tw
        except Exception:  # noqa: BLE001
            return None

    def _run_gate_scan(self) -> None:














        cfg = self._gate_config
        if not cfg:
            return
        stats = {"scans": 0, "blocks": 0, "skipped": 0, "prepaid": 0,
                 "unscanned": 0, "prefiltered": 0, "fallback": None,
                 "scan_ms": 0, "aborted": 0}
        self._gate_stats = stats
        if self._stamps or self._collect_raw or not (self._prompt or "").strip():
            stats["fallback"] = "not_text_run"
            self._track_gate_scan(stats, 0)
            return
        from ...core import scan_gate
        from ..gate_scan_phase import apply_scan_result, drain_scan_replies

        try:
            base_group = int(cfg.get("group", 2))
            max_group = int(cfg.get("max_group", base_group))
            min_score = float(cfg.get("min_score", 0.0))
            min_px = max(1, int(cfg.get("min_pixels", 8)))
        except (TypeError, ValueError):
            stats["fallback"] = "bad_config"
            self._track_gate_scan(stats, 0)
            return
        if base_group < 2 or not 0.0 < min_score <= 1.0:
            stats["fallback"] = "bad_config"
            self._track_gate_scan(stats, 0)
            return
        cap = cfg.get("max_scan_mupp")
        if not isinstance(cap, (int, float)) or isinstance(cap, bool) or cap <= 0:
            cap = None





        mupp = self._gate_ground_mupp() if cap is not None else None
        group = scan_gate.scan_group(
            base_group, max_group, None if cap is None else float(cap), mupp)
        if group == 0:
            stats["fallback"] = "resolution"
            self._track_gate_scan(stats, base_group)
            return



        blocks = [b for b in scan_gate.group_tiles(self._tiles, group)
                  if len(b) >= 2]
        if not blocks:
            stats["fallback"] = "no_blocks"
            self._track_gate_scan(stats, group)
            return
        stats["blocks"] = len(blocks)

        from qgis.PyQt.QtCore import QCoreApplication, QEventLoop

        _ef = getattr(QEventLoop, "ProcessEventsFlag", QEventLoop)
        _wait = _ef.WaitForMoreEvents | _ef.AllEvents

        t0 = time.monotonic()
        pending: deque = deque(enumerate(blocks))
        resubmit: deque = deque()
        in_flight: dict = {}
        submit_attempts: dict[int, int] = {}
        busy_since: dict[int, float] = {}
        exhausted_payload = None

        def fire() -> bool:
            while resubmit or pending:
                now = time.monotonic()
                ready_i = None
                for i, entry in enumerate(resubmit):
                    if entry[3] <= now:
                        ready_i = i
                        break
                if ready_i is not None:
                    block_i, block, submission, _ = resubmit[ready_i]
                    del resubmit[ready_i]
                elif pending:
                    block_i, block = pending.popleft()
                    submission, block = self._build_scan_submission(
                        block_i, block, group)
                    if submission is None:
                        continue
                else:
                    return False
                reply = self._client.post_detection_async(
                    submission, self._auth)
                in_flight[reply] = (block_i, block, submission)
                return True
            return False

        while (in_flight or resubmit or pending) and not self._stop_requested:



            while (not self._stop_requested
                   and len(in_flight) < max(1, self._aimd.cap) and fire()):
                pass
            if not in_flight:
                if resubmit or pending:
                    self._interruptible_sleep(0.25)
                    continue
                break
            QCoreApplication.processEvents(_wait, 250)
            if self._stop_requested:
                break


            read_replies: list = []
            cycle_setback = False
            cycle_progress = False
            for reply in [r for r in in_flight if self._reply_is_finished(r)]:
                block_i, block, submission = in_flight.pop(reply)
                read_replies.append(reply)
                response = self._read_reply(-(block_i + 1), reply)
                outcome = self._classify_submit_response(
                    -(block_i + 1), response, {})
                kind = outcome[0]
                if kind == "completed_inline":
                    self._fastfail.reset()
                    apply_scan_result(
                        self, block, outcome[1], group, min_score, min_px,
                        stats)
                    cycle_progress = True
                elif kind == "retry":
                    give_up, delay, setback = self._retry_decision(
                        -(block_i + 1), outcome, busy_since, submit_attempts)
                    cycle_setback = cycle_setback or setback
                    if give_up:
                        stats["unscanned"] += 1
                    else:
                        resubmit.append((block_i, block, submission,
                                         time.monotonic() + delay))
                elif kind == "exhausted":




                    exhausted_payload = outcome
                    break
                else:




                    stats["unscanned"] += 1



            self._free_read_replies(read_replies)
            self._settle_concurrency(cycle_setback, cycle_progress)
            if read_replies:



                self._emit_progress(0, len(self._tiles))
            if exhausted_payload is not None:
                break



            if self._fastfail.tripped and self.tiles_succeeded == 0:
                stats["fallback"] = "offline"
                break




        drain_scan_replies(
            self, in_flight, _wait, group, min_score, min_px, stats)
        unread = list(in_flight)
        stats["aborted"] = len(unread)
        for reply in unread:
            try:
                if not self._reply_is_finished(reply):
                    reply.abort()
            except (RuntimeError, AttributeError):
                pass
        in_flight.clear()
        self._free_read_replies(unread)

        if exhausted_payload is not None:
            self._emit_stop(exhausted_payload)



        for idx in list(self._gate_tile_bytes):
            if idx in self._gate_skip:
                self._gate_tile_bytes.pop(idx, None)
        stats["skipped"] = len(self._gate_skip)
        stats["prepaid"] = len(self._gate_prepaid)
        stats["prefiltered"] = len(self._prefilter_skip)
        stats["scan_ms"] = int((time.monotonic() - t0) * 1000)
        self.tiles_gate_skipped = len(self._gate_skip)
        logger.debug(
            "AutoDetectionWorker: gate scan %d blocks -> %d scans, "
            "%d skipped, %d prepaid, %d prefiltered (%d ms)",
            stats["blocks"], stats["scans"], stats["skipped"],
            stats["prepaid"], stats["prefiltered"], stats["scan_ms"],
        )




        if self._gate_skip:
            try:
                from qgis.core import QgsMessageLog

                QgsMessageLog.logMessage(
                    f"Auto detection: scan gate skipped "
                    f"{len(self._gate_skip)} of {len(self._tiles)} tile(s) "
                    f"as empty at 1/{group} resolution "
                    f"({stats['scans']} scan(s), {stats['scan_ms']} ms). "
                    f"These were charged and got no detection pass.",
                    "AI Segmentation", level=Qgis.MessageLevel.Info,
                )
            except Exception:  # noqa: BLE001
                pass  # nosec B110
        self._track_gate_scan(stats, group)

    def _track_gate_scan(self, stats: dict, group: int) -> None:



        try:
            from ...core.telemetry_run_events import track_auto_gate_scan
            track_auto_gate_scan(
                run_id=self._run_id,
                tiles=len(self._tiles),
                group=group,
                scans=int(stats.get("scans", 0)),
                blocks=int(stats.get("blocks", 0)),
                tiles_skipped=int(stats.get("skipped", 0)),
                tiles_prepaid=int(stats.get("prepaid", 0)),
                tiles_unscanned=int(stats.get("unscanned", 0)),
                tiles_prefiltered=int(stats.get("prefiltered", 0)),
                fallback=str(stats.get("fallback") or ""),
                scan_ms=int(stats.get("scan_ms", 0)),
            )
        except Exception:  # noqa: BLE001
            pass  # nosec B110

    def _build_scan_submission(self, block_i: int, block: list, group: int):












        from qgis.PyQt.QtCore import QRect
        from qgis.PyQt.QtGui import QImage, QPainter

        from ...core.cloud_detection import (
            encode_tile_png,
            tile_is_blank,
            tile_is_degenerate,
            tile_png_to_base64,
        )
        from ...core.qt_compat import resolve_qt_enum
        from ...core.tile_manager import TILE_SIZE
        from ..gate_scan_phase import release_scan_renders, request_scan_renders

        cell_px = max(1, TILE_SIZE // group)
        canvas_px = cell_px * group


        fmt = resolve_qt_enum(QImage, "Format", "Format_RGB32")
        canvas = QImage(canvas_px, canvas_px, fmt)
        canvas.fill(0xFF808080)
        painter = QPainter(canvas)
        try:
            hint = resolve_qt_enum(QPainter, "RenderHint", "SmoothPixmapTransform")
            painter.setRenderHint(hint, True)
            scanned: list = []




            bbox_union: list | None = None
            for idx, _qr, _qc in block:
                tx, ty, tw, th = self._tiles[idx]
                bn = self._make_tile_transform(tx, ty, tw, th)["bbox_native"]
                if bbox_union is None:
                    bbox_union = list(bn)
                else:
                    bbox_union = [
                        min(bbox_union[0], bn[0]), min(bbox_union[1], bn[1]),
                        max(bbox_union[2], bn[2]), max(bbox_union[3], bn[3]),
                    ]




            queued = request_scan_renders(
                self._render_request, self._render_collect, self._tiles,
                block, self._stop_requested)
            for idx, qr, qc in block:
                tx, ty, tw, th = self._tiles[idx]
                tile_img = None
                seq = queued.pop(idx, None)
                for _ in range(self._gate_scan_render_tries):
                    if self._stop_requested:
                        release_scan_renders(
                            self._render_collect, queued, seq)
                        return None, None
                    if seq is not None:
                        img = self._render_collect(seq)
                        seq = None
                    else:
                        img = self._tile_renderer(tx, ty, tw, th)
                    if img is None or img.isNull():
                        continue




                    if self._prefilter is not None and tile_is_degenerate(
                        img,
                        self._prefilter["nodata_frac"],
                        self._prefilter["band_eps"],
                        self._prefilter["nodata_rgb_eps"],
                        self._prefilter["min_valid_px"],
                    ):
                        self._prefilter_skip.add(idx)
                        self.tiles_prefiltered += 1
                        break
                    if not tile_is_blank(img):
                        tile_img = img
                        break
                if idx in self._prefilter_skip:
                    continue
                if tile_img is None:
                    continue


                if len(self._gate_tile_bytes) < self._gate_render_cache_max:
                    encoded = encode_tile_png(tile_img, 0, 0, tw, th)
                    if encoded is not None:
                        (_sx, _sy, cw, ch), data = encoded
                        self._gate_tile_bytes[idx] = ((tx, ty, cw, ch), data)
                painter.drawImage(
                    QRect(qc * cell_px, qr * cell_px, cell_px, cell_px),
                    tile_img)
                scanned.append((idx, qr, qc))
        finally:
            painter.end()
        if len(scanned) < 2:




            return None, None
        packed = encode_tile_png(canvas, 0, 0, canvas_px, canvas_px)
        if packed is None:
            return None, None
        _crop, data = packed
        submission = {
            "run_id": self._run_id,
            "prompt": self._prompt,
            "image_b64": tile_png_to_base64(data),


            "tile_index": -(block_i + 1),
            "crs_authid": self._crs_authid,
            "tile_bbox_wgs84": self._tile_bbox_wgs84(bbox_union),
            "tile_bbox_native": None if bbox_union is None else {
                "xmin": bbox_union[0], "ymin": bbox_union[1],
                "xmax": bbox_union[2], "ymax": bbox_union[3],
            },
            "pixel_size_m": None,
            "max_masks": self._max_masks,

            "threshold": self._detection_threshold,
            "mask_threshold": None,
            "exemplars": None,
            "parent_tile_index": None,



            "charge_tiles": len(scanned),
        }



        self._apply_client_meta(submission)
        return submission, scanned
