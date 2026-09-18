









from __future__ import annotations

import math

from ...core import transport_dials as _td
from ...core.error_policy import (
    BACKEND_UNAVAILABLE_CODES,
    EXHAUSTED_CODES,
    LINK_FAILURE_CODES,
    RUN_FATAL_CODES,
    TRANSIENT_CODES,
)
from ...core.server_dials import dial_in_range
from .retry_policy import (
    _BACKEND_UNAVAILABLE_DELAY_S,
    _HANDOFF_MIN_DELAY_S,
    _HANDOFF_OPEN_WINDOW_MAX_S,
    HANDOFF_CODE,
    HANDOFF_OVERLOAD_CODE,
)


def _as_int(value, default: int = -1) -> int:

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _handoff_wait_s(response: dict):







    if not response.get("retry_after_header"):
        return None
    try:
        return max(0.0, float(response.get("retry_after") or 0.0))
    except (TypeError, ValueError):
        return None


def _as_float(value, default: float) -> float:



    try:
        return float(value)
    except (TypeError, ValueError):
        return default




_SERVICE_WARMING_RETRY_S = 5.0




_TRANSIENT_RETRY_BASE_S = 2.0


class AutoTileSubmitMixin:


    def _run_transform_context(self):





        ctx = self._transform_context
        if ctx is not None:
            return ctx
        from qgis.core import QgsCoordinateTransformContext
        return QgsCoordinateTransformContext()

    def _bbox_ground_width_m(self, bbox_native) -> float | None:


        try:
            da = self._distance_area
            if da is None:
                from qgis.core import (
                    QgsCoordinateReferenceSystem,
                    QgsDistanceArea,
                )
                da = QgsDistanceArea()
                da.setSourceCrs(
                    QgsCoordinateReferenceSystem(self._crs_authid),
                    self._run_transform_context(),
                )
                da.setEllipsoid("WGS84")
                self._distance_area = da
            from qgis.core import QgsPointXY

            from ...core.qt_compat import DistanceMeters

            xmin, ymin, xmax, ymax = bbox_native
            ymid = (ymin + ymax) / 2.0
            width = da.measureLine(
                QgsPointXY(xmin, ymid), QgsPointXY(xmax, ymid))
            width_m = da.convertLengthMeasurement(width, DistanceMeters)







            if not math.isfinite(width_m) or width_m <= 0:
                return None
            return width_m
        except Exception:  # nosec B110
            return None

    def _resolve_ground_unit_scale(self) -> None:











        try:
            if not self._measure_ground_unit_scale(self._tiles[0]):
                return
            if len(self._tiles) > 1:
                near_kx, near_ky = self._ground_kx, self._ground_ky
                if self._measure_ground_unit_scale(self._tiles[-1]):
                    self._ground_kx = (self._ground_kx + near_kx) / 2.0
                    self._ground_ky = (self._ground_ky + near_ky) / 2.0
                else:
                    self._ground_kx, self._ground_ky = near_kx, near_ky
        except Exception:  # nosec B110
            self._ground_kx = 1.0
            self._ground_ky = 1.0

    def _measure_ground_unit_scale(self, tile) -> bool:





        try:
            tx, ty, tw, th = tile
            transform = self._make_tile_transform(tx, ty, tw, th)
            xmin, ymin, xmax, ymax = transform["bbox_native"]
            span_x = float(xmax - xmin)
            span_y = float(ymax - ymin)
            if span_x <= 0 or span_y <= 0:
                return False
            width_m = self._bbox_ground_width_m((xmin, ymin, xmax, ymax))
            if width_m is None:
                return False
            from qgis.core import QgsPointXY

            from ...core.qt_compat import DistanceMeters

            da = self._distance_area
            if da is None:
                return False
            xmid = (xmin + xmax) / 2.0
            height = da.measureLine(
                QgsPointXY(xmid, ymin), QgsPointXY(xmid, ymax))
            height_m = da.convertLengthMeasurement(height, DistanceMeters)
            if not math.isfinite(height_m) or height_m <= 0:
                return False
            self._ground_kx = width_m / span_x
            self._ground_ky = height_m / span_y
            return True
        except Exception:  # nosec B110
            return False

    def _ground_area_scale(self) -> float:

        scale = self._ground_kx * self._ground_ky
        return scale if math.isfinite(scale) and scale > 0 else 1.0

    def _ground_length_scale(self) -> float:




        return math.sqrt(self._ground_area_scale())

    def _tile_pixel_size_m(self, bbox_native, png_bytes) -> float | None:







        try:
            from ...core.cloud_detection import encoded_image_size

            size = encoded_image_size(png_bytes)
            if not size or size[0] <= 0:
                return None
            width_m = self._bbox_ground_width_m(bbox_native)
            if width_m is None:
                return None
            ratio = width_m / size[0]
            return round(ratio, 4) if math.isfinite(ratio) else None
        except Exception:  # nosec B110
            return None

    def _apply_client_meta(self, submission: dict) -> None:














        meta = self._client_meta
        if not meta:
            return
        tile_idx = submission.get("tile_index")
        for key in ("plugin_version", "policy_rev", "prompt_mode", "basemap"):
            val = meta.get(key)
            if val is not None:
                submission[key] = val
        if self._run_fields_tile_index is None:
            for key in ("zone_geojson", "zone_wkt", "zone_km2", "native_mupp"):
                val = meta.get(key)
                if val is not None:
                    submission[key] = val
            self._run_fields_tile_index = tile_idx
        clean = self._tile_clean_image.get(tile_idx)
        if clean is not None:
            submission["clean_image"] = clean

    def _release_run_fields(self, tile_idx: int) -> None:



        if self._run_fields_tile_index == tile_idx:
            self._run_fields_tile_index = None

    def _tile_bbox_wgs84(self, bbox_native) -> dict | None:



        if bbox_native is None or self._wgs84_transform_failed:
            return None
        try:
            if self._wgs84_transform is None:
                from qgis.core import (
                    QgsCoordinateReferenceSystem,
                    QgsCoordinateTransform,
                )
                src = QgsCoordinateReferenceSystem(self._crs_authid)
                dst = QgsCoordinateReferenceSystem("EPSG:4326")
                if not src.isValid() or not dst.isValid():
                    self._wgs84_transform_failed = True
                    return None
                self._wgs84_transform = QgsCoordinateTransform(
                    src, dst, self._run_transform_context())
            from qgis.core import QgsRectangle

            rect = QgsRectangle(
                float(bbox_native[0]), float(bbox_native[1]),
                float(bbox_native[2]), float(bbox_native[3]))
            out = self._wgs84_transform.transformBoundingBox(rect)
            if out is None or out.isEmpty():
                return None
            return {
                "xmin": out.xMinimum(), "ymin": out.yMinimum(),
                "xmax": out.xMaximum(), "ymax": out.yMaximum(),
            }
        except Exception:  # noqa: BLE001
            return None

    def _release_tile_clean_image(self, tile_idx: int) -> None:





        self._tile_clean_image.pop(tile_idx, None)

    def _build_submission(self, tile_idx: int, tile_spec, png_bytes) -> tuple[dict, dict]:


        from ...core.cloud_detection import mask_scale_field, tile_png_to_base64

        tile_x, tile_y, tile_w, tile_h = tile_spec
        tile_transform = self._make_tile_transform(tile_x, tile_y, tile_w, tile_h)
        bbox_native = tile_transform["bbox_native"]
        submission = {
            "run_id": self._run_id,
            "prompt": self._prompt,
            "image_b64": tile_png_to_base64(png_bytes),
            "tile_index": tile_idx,



            "tiles_total": int(self._paid_tiles_total or 0),
            "crs_authid": self._crs_authid,
            "tile_bbox_wgs84": self._tile_bbox_wgs84(bbox_native),
            "tile_bbox_native": {
                "xmin": bbox_native[0], "ymin": bbox_native[1],
                "xmax": bbox_native[2], "ymax": bbox_native[3],
            },
            "pixel_size_m": self._tile_pixel_size_m(bbox_native, png_bytes),
            "max_masks": self._max_masks,
            "threshold": self._detection_threshold,
            "mask_threshold": None,
            "exemplars": self._tile_exemplars.get(tile_idx) or None,



            "parent_tile_index": self._billed_ancestor_of(tile_idx),
        }


        if self._return_semantic:
            submission["return_semantic"] = True



        run_mask_scale = mask_scale_field(self._mask_scale)
        if run_mask_scale is not None:
            submission["mask_scale"] = run_mask_scale




        if tile_idx in self._gate_prepaid:
            submission["charge_tiles"] = 0


        self._apply_client_meta(submission)



        self.upload_bytes += (
            len(submission["image_b64"]) + len(submission.get("clean_image") or ""))
        self.uploads_sent += 1
        return submission, tile_transform

    def _submit_batch(self, batch: list) -> list:




















        submissions = []
        transforms = []
        for tile_idx, tile_spec, png_bytes in batch:
            submission, tile_transform = self._build_submission(
                tile_idx, tile_spec, png_bytes)
            transforms.append(tile_transform)
            submissions.append(submission)

        responses = self._client.submit_detection_many(
            submissions, self._auth, should_abort=self._should_abort)
        for response in responses:


            self._apply_window_hint(response)
            self._note_tile_balance(response)
        outcomes = []
        for (tile_idx, _spec, _png), response, tile_transform in zip(
            batch, responses, transforms
        ):
            outcomes.append(
                self._classify_submit_response(tile_idx, response, tile_transform)
            )
        return outcomes

    def _note_quota_refusal(self, response: dict) -> None:






        try:
            self._quota_refusal = {
                "code": str(response.get("code") or ""),
                "message": str(response.get("error") or ""),
                "envelope": str(response.get("envelope") or ""),
                "used": response.get("used"),
                "limit": response.get("limit"),
            }
        except (TypeError, ValueError, AttributeError):
            self._quota_refusal = None

    def quota_refusal_detail(self) -> dict:

        return dict(self._quota_refusal or {})

    def _classify_submit_response(self, tile_idx: int, response: dict, tile_transform: dict) -> tuple:



        code = response.get("code", "")
        handoff = _handoff_wait_s(response)


        status = response.get("http_status")
        if status == 429:
            self.http_429 = getattr(self, "http_429", 0) + 1
        elif status == 503:
            self.http_503 = getattr(self, "http_503", 0) + 1
        if "error" in response:







            if code != "RATE_LIMITED" and handoff is None:
                self._release_run_fields(tile_idx)
            if code in EXHAUSTED_CODES:
                self._note_quota_refusal(response)
                return ("exhausted", _as_int(response.get("credits_remaining"), 0))
            if code == "RATE_LIMITED":
                try:
                    delay = float(response.get("retry_after", 0) or 0)
                except (TypeError, ValueError):
                    delay = 0.0




                position = _as_int(response.get("queue_position"))
                depth = _as_int(response.get("queue_depth"))
                eta = _as_int(response.get("eta_seconds"))
                self._note_busy(position, depth, eta)


                return ("retry", delay if delay > 0 else 5.0, True, "RATE_LIMITED")
            if handoff is not None:







                self._note_busy(-1, -1, 0)
                delay = max(handoff, _td.handoff_min_delay_s(_HANDOFF_MIN_DELAY_S))
                delay = min(delay, self._queue_retry_budget_s)
                short = handoff <= _td.handoff_open_window_max_s(_HANDOFF_OPEN_WINDOW_MAX_S)
                return ("retry", delay, True,
                        HANDOFF_CODE if short else HANDOFF_OVERLOAD_CODE)
            if code in TRANSIENT_CODES:
                if code == "SERVICE_WARMING":






                    self._note_busy(-1, -1, 0)
                    return ("retry", dial_in_range(
                        "tuning.network.service_warming_retry_s",
                        _SERVICE_WARMING_RETRY_S, 1.0, 30.0), True, code)
                if code in LINK_FAILURE_CODES:





                    self._note_flowing()
                else:







                    self._note_busy(-1, -1, 0)



                return ("retry", dial_in_range(
                    "tuning.network.transient_retry_base_s",
                    _TRANSIENT_RETRY_BASE_S, 0.5, 10.0), False, code)
            if code in BACKEND_UNAVAILABLE_CODES:







                self._note_busy(-1, -1, 0)
                return ("retry", _BACKEND_UNAVAILABLE_DELAY_S, False, code)



            if code in RUN_FATAL_CODES:





                detail = str(response.get("error") or "").strip()[:200]
                return ("fatal", code, detail)
            return ("tile_fatal", code)







        if response.get("status") == "completed":
            return ("completed_inline", response, tile_transform)

        request_id = response.get("request_id", "")
        if not request_id:



            self._release_run_fields(tile_idx)
            self._emit_warning(
                f"Tile {tile_idx}: submit response missing request_id; skipping"
            )
            return ("skip",)

        poll_interval = _as_float(response.get("poll_interval"), self._poll_interval_s)
        max_wait = _as_float(response.get("max_wait"), self._poll_max_wait_s)


        if max_wait > 3600:
            max_wait = max_wait / 1000.0
        return ("ok", request_id, poll_interval, max_wait, tile_transform)

    def _make_tile_transform(
        self, tile_x: int, tile_y: int, tile_w: int, tile_h: int
    ) -> dict:









        src_bbox = self._geo_transform.get("bbox", (0.0, 0.0, 1.0, 1.0))
        img_shape = self._geo_transform.get("img_shape", (1, 1))
        img_h, img_w = img_shape[0], img_shape[1]


        img_w = max(img_w, 1)
        img_h = max(img_h, 1)

        src_minx, src_miny, src_maxx, src_maxy = src_bbox

        px_w = (src_maxx - src_minx) / img_w
        px_h = (src_maxy - src_miny) / img_h

        tile_minx = src_minx + tile_x * px_w
        tile_maxx = src_minx + (tile_x + tile_w) * px_w

        tile_miny = src_maxy - (tile_y + tile_h) * px_h
        tile_maxy = src_maxy - tile_y * px_h

        return {

            "bbox": (tile_minx, tile_maxx, tile_miny, tile_maxy),

            "bbox_native": (tile_minx, tile_miny, tile_maxx, tile_maxy),
            "img_shape": (tile_h, tile_w),
            "crs": self._crs_authid,
        }
