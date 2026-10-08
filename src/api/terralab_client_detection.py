







from __future__ import annotations

import json
import math
import time

from qgis.core import Qgis, QgsMessageLog

from ..core import transport_dials as _td
from ..core.gui_thread import on_gui_thread
from ..core.i18n import tr
from ..core.server_dials import dial_in_range
from .request_compression import answer_refused_the_body, note_gzip_request_refused, packed_request_body
from .terralab_client_errors import (
    _answer_from_failed_transfer,
    _answer_from_status_and_body,
    _classify_qt_error,
)
from .terralab_client_primitives import (
    _TIMEOUT_POLL_DETECTION,
    _TIMEOUT_RUN_EXPORT,
    _TIMEOUT_SUBMIT_DETECTION,
    _TIMEOUT_SUBMIT_DETECTION_DIRECT,
    _TIMEOUT_WARMUP,
    _apply_redirect_policy,
    _log_warning,
    _NoError,
    note_server_contact,
    server_reached_recently,
)
from .terralab_client_retry import UNAVAILABLE_STATUS, client_attempt_headers, unavailable_retry_pause_s


class TerraLabDetectionMixin:







    def _detection_predict_url(self) -> str:


        if self.detection_direct:
            return f"{self.detection_base_url}/predict"
        return f"{self.detection_base_url}/api/ai-segmentation/predict"

    def _detection_refine_url(self) -> str:


        if self.detection_direct:
            return f"{self.detection_base_url}/refine"
        return f"{self.detection_base_url}/api/ai-segmentation/refine"

    def _detection_run_export_url(self) -> str:


        return f"{self.detection_base_url}/run-export"

    def refine_register_takes_crop_tiles(self) -> bool:



        try:
            from ..core.server_dials import read_value

            routes = read_value("features.crop_tiles_routes")
            if not isinstance(routes, (list, tuple)):
                return False
            route = "direct" if getattr(self, "detection_direct", False) else "relay"
            return route in routes
        except Exception:  # noqa: BLE001
            return False

    def refine_endpoint_url(self) -> str:





        return self._resolve_url(self._detection_refine_url())

    def submit_refine(self, payload: dict, auth: dict, cancel_check=None,
                      timeout_ms: int | None = None) -> dict:



















        body = json.dumps(payload, allow_nan=False).encode("utf-8")




        kept = self._refine_while_drawing(body, auth, cancel_check, timeout_ms)
        if kept is not None:
            return kept
        if cancel_check is not None and cancel_check():
            return {"error": tr("The click was cancelled."), "code": "CANCELLED"}
        return self._request(
            "POST",
            self._detection_refine_url(),
            auth=auth,
            body=body,
            timeout_ms=timeout_ms or self._submit_timeout(),
            retry_unavailable=True,
        )

    def _refine_while_drawing(self, body: bytes, auth: dict,
                              cancel_check=None,
                              timeout_ms: int | None = None) -> dict | None:



















        started = time.monotonic()
        payload, packed = packed_request_body(body)
        hints: dict = {}
        answer, http_status, _body_was_json = self._refine_once_while_drawing(
            payload, packed, auth, cancel_check, timeout_ms, hints=hints)
        if answer is not None and packed and answer_refused_the_body(http_status):
            _log_warning("A compressed request body was refused; sending "
                         "them plain for the rest of the session")
            note_gzip_request_refused()
            payload, packed = body, False
            answer, http_status, _ = self._refine_once_while_drawing(
                payload, packed, auth, cancel_check, timeout_ms, hints=hints)
        if answer is None:
            return None
        return self._refine_again_while_drawing(
            answer, http_status, hints, started, payload, packed, auth,
            cancel_check, timeout_ms)

    def _refine_again_while_drawing(
        self, answer: dict, http_status: int | None, hints: dict, started: float,
        body: bytes, packed: bool, auth: dict, cancel_check=None,
        timeout_ms: int | None = None,
    ) -> dict:





        if http_status != UNAVAILABLE_STATUS:
            return answer
        try:
            from .click_transport import ClickPostAbandoned, click_wait_max_ms, wait_until_done

            budget_ms = min(timeout_ms or self._submit_timeout(), click_wait_max_ms())
            pause = unavailable_retry_pause_s(
                http_status, hints.get("retry_after_s") or 0.0, budget_ms / 1000.0,
                time.monotonic() - started)
            if pause is None:
                return answer
            _log_warning("The service was unavailable (HTTP 503); sending "
                         "the click once more")
            until = time.monotonic() + pause
            if not wait_until_done(lambda: time.monotonic() >= until,
                                   int(pause * 1000) + 1000, cancel_check=cancel_check):
                return self._refine_abandoned(ClickPostAbandoned(cancelled=True))
            left_ms = int(budget_ms - (time.monotonic() - started) * 1000.0)
        except Exception:  # noqa: BLE001
            return answer
        if left_ms <= 0:
            return answer
        again, _, _ = self._refine_once_while_drawing(
            body, packed, auth, cancel_check, left_ms, attempt=2)
        return answer if again is None else again

    def _refine_once_while_drawing(
        self, body: bytes, packed: bool, auth: dict, cancel_check=None,
        timeout_ms: int | None = None, attempt: int = 1,
        hints: dict | None = None,
    ) -> tuple[dict | None, int | None, bool]:









        from ..core.server_dials import feature_enabled
        if not feature_enabled("click_keep_painting"):
            return None, None, False
        try:
            from .click_transport import ClickPostAbandoned
        except Exception:  # noqa: BLE001
            return None, None, False


        try:
            if not on_gui_thread():


                return None, None, False
            from .click_transport import post_and_keep_painting

            url = self._resolve_url(self._detection_refine_url())
            timeout_ms = timeout_ms or self._submit_timeout()
        except Exception:  # noqa: BLE001
            return None, None, False
        try:
            taken = post_and_keep_painting(
                url, body, auth, timeout_ms, _apply_redirect_policy,
                cancel_check=cancel_check, packed=packed,
                extra_headers=client_attempt_headers(attempt), hints=hints)
        except ClickPostAbandoned as gone:
            return self._refine_abandoned(gone), None, False
        except Exception:  # noqa: BLE001






            _log_warning("The click wait ended in an unexpected way; "
                         "the click is not being re-sent")
            return self._refine_abandoned(ClickPostAbandoned()), None, False
        if taken is None:
            return None, None, False
        raw, http_status, qt_error = taken
        raw_body = raw.decode("utf-8", "replace")
        if http_status is not None:
            note_server_contact()

        if qt_error != _NoError:
            answer, body_was_json = _answer_from_failed_transfer(
                raw_body, http_status,
                lambda: _classify_qt_error(
                    qt_error, "", http_status,
                    service_reachable=server_reached_recently()))
        else:
            answer, body_was_json = _answer_from_status_and_body(raw_body, http_status)
        if (isinstance(answer, dict) and "error" in answer
                and http_status is not None and http_status >= 400
                and "http_status" not in answer):
            answer = dict(answer)
            answer["http_status"] = int(http_status)
        return answer, http_status, body_was_json

    @staticmethod
    def _refine_abandoned(gone) -> dict:



        if getattr(gone, "cancelled", False):
            return {"error": tr("The click was cancelled."), "code": "CANCELLED"}
        message = tr("Request timed out. Check your connection or try again.")
        try:
            from ..core.server_dials import dial_copy

            message = dial_copy("network.request_timed_out", message)
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return {"error": message, "code": "TIMEOUT"}

    def submit_refine_register(self, payload: dict, auth: dict,
                               timeout_ms: int | None = None,
                               cancel_feedback=None) -> dict:












        body = json.dumps(payload, allow_nan=False).encode("utf-8")

        def _send() -> dict:
            return self._request(
                "POST",
                self._detection_refine_url() + "/register",
                auth=auth,
                body=body,
                timeout_ms=timeout_ms or self._submit_timeout(),
                wall_clock=True,
                retry_unavailable=True,
            )

        if cancel_feedback is None:
            return _send()



        from .request_feedback import request_feedback

        with request_feedback(cancel_feedback):
            return _send()

    def post_run_export_body(self, body: bytes, auth: dict) -> dict:
















        if not self.detection_direct:
            QgsMessageLog.logMessage(
                "Run export summary not sent: this route does not take one",
                "AI Segmentation", level=Qgis.MessageLevel.Info)
            return {"error": "run export not available on this route",
                    "code": "NOT_AVAILABLE"}
        return self._request(
            "POST", self._detection_run_export_url(), auth=auth, body=body,
            timeout_ms=_td.run_export_timeout_ms(_TIMEOUT_RUN_EXPORT),
            retry_unavailable=True,
        )

    def _submit_timeout(self) -> int:








        if not self.detection_direct:
            return dial_in_range(
                "tuning.network.submit_detection_timeout_ms",
                _TIMEOUT_SUBMIT_DETECTION, 10000, 300000)
        try:
            from ..core.detection_policy import submit_timeout_ms

            return submit_timeout_ms(_TIMEOUT_SUBMIT_DETECTION_DIRECT)
        except Exception:  # noqa: BLE001
            return _TIMEOUT_SUBMIT_DETECTION_DIRECT

    def submit_detection(
        self,
        run_id: str,
        prompt: str,
        image_b64: str,
        tile_index: int,
        crs_authid: str,
        tile_bbox_wgs84: dict | None,
        tile_bbox_native: dict | None,
        pixel_size_m: float | None,
        max_masks: int | None,
        auth: dict,
        threshold: float | None = None,
        mask_threshold: float | None = None,
        exemplars: list[dict] | None = None,
    ) -> dict:
































        body = self._build_predict_body(
            run_id, prompt, image_b64, tile_index, crs_authid,
            tile_bbox_wgs84, tile_bbox_native, pixel_size_m,
            max_masks, threshold, mask_threshold, exemplars,
        )
        return self._request(
            "POST",
            self._detection_predict_url(),
            auth=auth,
            body=body,
            timeout_ms=self._submit_timeout(),
            extra_headers=self._predict_attempt_headers(
                {"run_id": run_id, "tile_index": tile_index}),
        )

    def _predict_attempt_headers(self, submission: dict) -> dict:







        run_id = submission.get("run_id")
        if run_id != self._predict_sends_run:
            self._predict_sends_run = run_id
            self._predict_sends = {}
        key = (submission.get("tile_index"), submission.get("parent_tile_index"))
        sends = self._predict_sends.get(key, 0) + 1
        self._predict_sends[key] = sends
        return client_attempt_headers(sends)






































    _PREDICT_EXTRA_FIELDS = (
        "return_semantic", "charge_tiles", "mask_scale",
        "plugin_version", "policy_rev", "prompt_mode", "basemap",
        "zone_geojson", "zone_wkt", "zone_km2", "native_mupp", "zone_keep_margin_m",
        "clean_image", "tiles_total", "self_exemplar_of", "land_cover",
        "tile_filters", "collect_raw", "run_pixel_size_m",
    )

    @classmethod
    def _predict_extras(cls, submission: dict) -> dict | None:

        extra = {
            k: submission[k]
            for k in cls._PREDICT_EXTRA_FIELDS
            if submission.get(k) is not None
        }
        return extra or None

    @staticmethod
    def _build_predict_body(
        run_id, prompt, image_b64, tile_index, crs_authid,
        tile_bbox_wgs84, tile_bbox_native, pixel_size_m,
        max_masks, threshold, mask_threshold, exemplars,
        parent_tile_index=None, extra=None,
    ) -> bytes:




        payload: dict = {
            "image": image_b64,
            "run_id": run_id,
            "prompt": prompt,
            "tile_index": tile_index,
            "crs_authid": crs_authid,
        }
        if extra:
            payload.update(extra)
        if parent_tile_index is not None:

            payload["parent_tile_index"] = int(parent_tile_index)
        if tile_bbox_wgs84 is not None:
            payload["tile_bbox_wgs84"] = tile_bbox_wgs84
        if tile_bbox_native is not None:
            payload["tile_bbox_native"] = tile_bbox_native





        if pixel_size_m is not None and math.isfinite(pixel_size_m):
            payload["pixel_size_m"] = pixel_size_m
        if max_masks is not None:
            payload["max_masks"] = max_masks
        if threshold is not None:
            payload["threshold"] = threshold
        if mask_threshold is not None:
            payload["mask_threshold"] = mask_threshold
        if exemplars:
            payload["exemplars"] = exemplars






        try:
            return json.dumps(payload, allow_nan=False).encode("utf-8")
        except ValueError:
            pass
        for key in ("pixel_size_m", "tile_bbox_native", "tile_bbox_wgs84"):
            payload.pop(key, None)
        _log_warning(
            "Dropped non-finite optional fields from a tile payload; the "
            "detection itself is unaffected."
        )
        try:
            return json.dumps(payload, allow_nan=False).encode("utf-8")
        except ValueError:


            return json.dumps(payload).encode("utf-8")

    def submit_detection_many(
        self, submissions: list[dict], auth: dict, should_abort=None
    ) -> list[dict]:









        specs = []
        for s in submissions:
            body = self._build_predict_body(
                s["run_id"], s["prompt"], s["image_b64"], s["tile_index"],
                s["crs_authid"], s.get("tile_bbox_wgs84"), s.get("tile_bbox_native"),
                s.get("pixel_size_m"), s.get("max_masks"), s.get("threshold"),
                s.get("mask_threshold"), s.get("exemplars"),
                s.get("parent_tile_index"), self._predict_extras(s),
            )
            specs.append({
                "method": "POST",
                "path": self._detection_predict_url(),
                "auth": auth,
                "body": body,
                "timeout_ms": self._submit_timeout(),
                "extra_headers": self._predict_attempt_headers(s),
            })
        return self.request_many(specs, should_abort=should_abort)

    def warmup(self, auth: dict) -> bool:












        from ..core.server_dials import dial_in_range
        warmup_timeout_ms = dial_in_range(
            "tuning.network.warmup_timeout_ms", _TIMEOUT_WARMUP, 2000, 30000)
        try:
            if self.detection_direct:







                result = self._request(
                    "GET",
                    f"{self.detection_base_url}/health",
                    auth=auth,
                    timeout_ms=warmup_timeout_ms,
                    retry_get_failures=False,
                )



                from ..core import cloud_warming_state
                if result.get("status") == "ok":
                    cloud_warming_state.mark_ready()
                    return True
                if result.get("http_status") == 503:
                    cloud_warming_state.mark_warming()
                    return True
                return False
            result = self._request(
                "POST",
                "/api/ai-segmentation/warmup",
                auth=auth,
                body=b"{}",
                timeout_ms=warmup_timeout_ms,
            )
            return result.get("ok") is True
        except Exception:
            return False

    def detection_health(self, auth: dict, timeout_ms: int = 5000) -> str:





        from ..core import cloud_warming_state
        try:
            if not self.detection_direct:
                result = self._request(
                    "POST", "/api/ai-segmentation/warmup", auth=auth,
                    body=b"{}", timeout_ms=timeout_ms)
                if result.get("ok") is True:
                    cloud_warming_state.mark_ready()
                    return "ready"
                return "error"
            result = self._request(
                "GET", f"{self.detection_base_url}/health", auth=auth,
                timeout_ms=timeout_ms, retry_get_failures=False)
            if result.get("status") == "ok":
                cloud_warming_state.mark_ready()
                return "ready"
            if result.get("http_status") == 503:
                cloud_warming_state.mark_warming()
                return "warming"
            return "error"
        except Exception:  # noqa: BLE001
            return "error"

    def end_detection_session(self, auth: dict, timeout_ms: int | None = None) -> bool:






        if not self.detection_direct:
            return False
        try:
            from .detection_session import SESSION_END_PATH

            if timeout_ms is None:
                from ..core.server_dials import dial_in_range
                timeout_ms = dial_in_range(
                    "tuning.network.session_end_timeout_ms", 2000, 500, 10000)
            result = self._request(
                "POST",
                f"{self.detection_base_url}{SESSION_END_PATH}",
                auth=auth,
                body=b"{}",
                timeout_ms=timeout_ms,
                wall_clock=True,
            )
            return isinstance(result, dict) and result.get("stopping") is True
        except Exception:  # noqa: BLE001
            return False

    def get_detection_status(self, request_id: str, auth: dict) -> dict:















        from urllib.parse import quote

        path = f"/api/ai-segmentation/predict/status?request_id={quote(request_id, safe='')}"
        return self._request(
            "GET", path, auth=auth, timeout_ms=_td.poll_detection_timeout_ms(_TIMEOUT_POLL_DETECTION)
        )

    def get_detection_status_many(
        self, request_ids: list[str], auth: dict, should_abort=None
    ) -> list[dict]:







        from urllib.parse import quote

        specs = [
            {
                "method": "GET",
                "path": f"/api/ai-segmentation/predict/status?request_id={quote(rid, safe='')}",
                "auth": auth,
                "timeout_ms": _td.poll_detection_timeout_ms(_TIMEOUT_POLL_DETECTION),
            }
            for rid in request_ids
        ]
        return self.request_many(specs, should_abort=should_abort)
