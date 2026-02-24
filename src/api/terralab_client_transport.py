







from __future__ import annotations

from qgis.core import QgsFeedback, QgsNetworkAccessManager
from qgis.PyQt.QtCore import QByteArray
from qgis.PyQt.QtNetwork import QNetworkRequest

from ..core import transport_dials as _td
from ..core.gil_safe_qobject import prime as _gil_safe
from ..core.i18n import tr
from ..core.streamed_download import sleep_unless_cancelled
from .json_request import build_json_request
from .request_compression import answer_refused_the_body, note_gzip_request_refused, packed_request_body
from .request_feedback import current_request_feedback as _current_feedback
from .terralab_client_errors import (
    _answer_from_failed_transfer,
    _answer_from_status_and_body,
    _classify_network_error,
    _classify_qt_error,
    _unreadable_answer,
)
from .terralab_client_nam_pool import (
    _CONNECTIONS_PER_MANAGER,
)
from .terralab_client_primitives import (
    _TIMEOUT_API,
    _http_status_of,
    _log_warning,
    _NoError,
    _reply_was_packed,
    _WallClockGuard,
    note_server_contact,
    server_reached_recently,
)
from .terralab_client_retry import (
    _HANDOFF_STATUSES,
    _note_retry_after,
    _note_window_hint,
    _response_etag,
    _retry_after_s,
    _retry_pause_s,
    _worth_asking_again,
)




_NOT_MODIFIED_STATUS = 304


def _valid_request_timeout(value) -> bool:


    return (isinstance(value, int) and not isinstance(value, bool)
            and 0 < value < 2_147_478_647)


def _may_pack_body(method: str, path: str, body: bytes | None) -> bool:






    return bool(method == "POST" and body
                and path.startswith(("http://", "https://")))


def _request_cancelled() -> bool:
    feedback = _current_feedback()
    if feedback is None:
        return False
    try:
        return bool(feedback.isCanceled())
    except (AttributeError, RuntimeError):
        return False


def _cancelled_answer() -> dict:
    return {"error": tr("Cancelled"), "code": "CANCELLED"}


def _transfer_failed(method: str, reply) -> tuple[bool, str]:






    if reply is None:
        return True, ""
    try:
        error = reply.error()
    except (AttributeError, RuntimeError):
        return True, ""
    if getattr(error, "value", error) != getattr(_NoError, "value", _NoError):
        return True, ""
    if method == "GET":
        try:
            if not bytes(reply.content()):
                return True, "empty response"
        except (AttributeError, RuntimeError, TypeError):
            return False, ""
    return False, ""


class TerraLabTransportMixin:







    def _resolve_url(self, path_or_url: str) -> str:


        if path_or_url.startswith(("http://", "https://")):
            url = path_or_url
        else:
            url = f"{self.base_url}{path_or_url}"
        self._reject_cleartext_remote(url)
        if getattr(self, "detection_direct", False) and url.startswith(
                f"{self.detection_base_url}/"):
            from .detection_session import note_use

            note_use(url)
        return url

    @staticmethod
    def _reject_cleartext_remote(url: str) -> None:




        from ..core.server_dials import cleartext_remote_url

        if cleartext_remote_url(url):
            raise ValueError(
                "Refusing to send an authenticated request over plain HTTP to a "
                "remote host (the key would travel in cleartext). Use HTTPS."
            )

    def _make_qnetwork_request(self, auth: dict | None, timeout_ms: int, path: str,
                               packed: bool = False) -> QNetworkRequest:





        return build_json_request(self._resolve_url(path), auth, timeout_ms, packed=packed)

    def _request(
        self,
        method: str,
        path: str,
        auth: dict | None = None,
        body: bytes | None = None,
        timeout_ms: int | None = None,
        allow_list: bool = False,
        require_body: bool = False,
        wall_clock: bool = False,
        extra_headers: dict | None = None,
        retry_get_failures: bool = True,
    ) -> dict | list:








































        if _request_cancelled():
            return _cancelled_answer()
        if timeout_ms is None:
            timeout_ms = _td.api_timeout_ms(_TIMEOUT_API)
        if not _valid_request_timeout(timeout_ms):
            return {"error": "Invalid request timeout", "code": "CLIENT_ERROR"}
        payload, packed = body, False
        if _may_pack_body(method, path, body):
            payload, packed = packed_request_body(body)
        answer, http_status, _body_was_json = self._request_once(
            method, path, auth, payload, packed, timeout_ms, allow_list,
            require_body, wall_clock, extra_headers)
        if _request_cancelled():
            return _cancelled_answer()
        if packed and answer_refused_the_body(http_status):
            _log_warning("A compressed request body was refused; sending "
                         "them plain for the rest of the session")
            note_gzip_request_refused()
            answer, http_status, _ = self._request_once(
                method, path, auth, body, False, timeout_ms, allow_list,
                require_body, wall_clock, extra_headers)
        if (method == "GET" and retry_get_failures
                and _worth_asking_again(answer, http_status)):


            sleep_unless_cancelled(
                self._pending_retry_after_s or _retry_pause_s(), _request_cancelled, slice_s=0.1)
            if _request_cancelled():
                return _cancelled_answer()
            answer, http_status, _ = self._request_once(
                method, path, auth, payload, packed, timeout_ms, allow_list,
                require_body, wall_clock, extra_headers)




        if (isinstance(answer, dict) and "error" in answer
                and http_status is not None and "http_status" not in answer):
            answer = dict(answer)
            answer["http_status"] = int(http_status)
        return answer

    def _request_once(
        self,
        method: str,
        path: str,
        auth: dict | None,
        body: bytes | None,
        packed: bool,
        timeout_ms: int,
        allow_list: bool,
        require_body: bool,
        wall_clock: bool,
        extra_headers: dict | None = None,
    ) -> tuple[dict | list, int | None, bool]:







        self._pending_retry_after_s = 0.0
        try:
            url = self._resolve_url(path)
        except ValueError as err:


            return ({"error": str(err), "code": "CLIENT_ERROR"}, None, False)


        req = build_json_request(url, auth, timeout_ms, packed=packed, extra_headers=extra_headers)

        if method not in ("GET", "POST"):
            return ({"error": f"Unsupported method: {method}",
                     "code": "CLIENT_ERROR"}, None, False)








        outer = _gil_safe(_current_feedback())
        own = _gil_safe(QgsFeedback()) if wall_clock else None
        guard = _WallClockGuard(own, timeout_ms, outer) if own is not None else None
        feedback = own if own is not None else outer
        try:
            if method == "GET":
                reply = QgsNetworkAccessManager.blockingGet(req, "", True, feedback)
            else:
                payload = QByteArray(body) if body else QByteArray()
                reply = QgsNetworkAccessManager.blockingPost(req, payload, "", True, feedback)
        finally:
            if guard is not None:
                guard.stop()

        if _request_cancelled():
            return _cancelled_answer(), None, False
        failed, detail = _transfer_failed(method, reply)
        if failed:
            http_status = _http_status_of(reply)
            if http_status is not None:
                note_server_contact()
            if http_status == _NOT_MODIFIED_STATUS:


                return {"not_modified": True}, http_status, False
            if http_status in _HANDOFF_STATUSES and reply is not None:
                self._pending_retry_after_s = _retry_after_s(reply)
            raw_body = ""
            if reply is not None and http_status is not None and http_status >= 400:


                raw_body = bytes(reply.content()).decode("utf-8", "replace")
            answer, body_was_json = _answer_from_failed_transfer(
                raw_body, http_status, lambda: _classify_network_error(reply, detail))
            return answer, http_status, body_was_json

        if reply is None:
            return _unreadable_answer(), None, False
        http_status = _http_status_of(reply)
        raw_body = bytes(reply.content()).decode("utf-8", "replace")
        if http_status is None and not raw_body:




            return ({"error": tr("Request timed out. Check your connection or try again."),
                     "code": "TIMEOUT"}, None, False)
        note_server_contact()
        if http_status == _NOT_MODIFIED_STATUS:



            return {"not_modified": True}, http_status, False
        if http_status in _HANDOFF_STATUSES:
            self._pending_retry_after_s = _retry_after_s(reply)

        answer, body_was_json = _answer_from_status_and_body(
            raw_body, http_status, require_body=require_body, allow_list=allow_list)
        if body_was_json and isinstance(answer, dict) and http_status == 200:




            etag = _response_etag(reply)
            if etag:
                answer = dict(answer)
                answer["etag"] = etag
        return answer, http_status, body_was_json

    def _parse_reply(self, reply, require_body: bool = False,
                     allow_list: bool = False) -> dict | list:
















        answer, _body_was_json = self._parse_reply_once(
            reply, require_body, allow_list)



        if isinstance(answer, dict) and "error" in answer:
            status = _http_status_of(reply)
            if status is not None:
                answer = dict(answer)
                answer["http_status"] = int(status)
        if _reply_was_packed(reply) and answer_refused_the_body(
                _http_status_of(reply)):
            _log_warning("A compressed request body was refused; sending "
                         "them plain for the rest of the session")
            note_gzip_request_refused()



            if isinstance(answer, dict) and not answer.get("code"):
                answer = dict(answer)
                answer["code"] = "SERVER_ERROR"
        return _note_window_hint(_note_retry_after(answer, reply), reply)

    def _parse_reply_once(self, reply, require_body: bool = False,
                          allow_list: bool = False) -> tuple[dict | list, bool]:






        if not reply.isFinished():



            return ({"error": tr("Request timed out. Check your connection or try again."),
                     "code": "TIMEOUT"}, False)
        qt_error = reply.error()
        http_status = _http_status_of(reply)
        raw_body = bytes(reply.readAll()).decode("utf-8", "replace")
        if qt_error == _NoError and http_status is None and not raw_body:
            return ({"error": tr("Request timed out. Check your connection or try again."),
                     "code": "TIMEOUT"}, False)
        if http_status is not None or (qt_error == _NoError and raw_body):
            note_server_contact()

        if qt_error != _NoError:
            return _answer_from_failed_transfer(
                raw_body, http_status,
                lambda: _classify_qt_error(
                    qt_error, reply.errorString(), http_status,
                    service_reachable=server_reached_recently()))
        return _answer_from_status_and_body(
            raw_body, http_status, require_body=require_body,
            allow_list=allow_list)

    def parse_reply(self, reply) -> dict:


        return self._parse_reply(reply)

    def request_many(self, specs: list[dict], should_abort=None,
                     retry_reads: bool = True) -> list[dict]:



































        from qgis.PyQt.QtCore import QEventLoop, QTimer

        if not specs:
            return []

        def _already_aborting():
            if _request_cancelled():
                return True
            if should_abort is None:
                return False
            try:
                return bool(should_abort())
            except Exception:  # noqa: BLE001
                return False




        loop = QEventLoop()
        replies = []
        packed_flags: list[bool] = []
        failures: dict[int, dict] = {}
        default_timeout = _td.api_timeout_ms(_TIMEOUT_API)
        max_timeout = 0
        for slot, spec in enumerate(specs):
            if _already_aborting():
                replies.append(None)
                packed_flags.append(False)
                continue
            reply = None
            packed = False
            try:
                method = spec.get("method", "GET")
                if method not in ("GET", "POST"):
                    raise ValueError("Unsupported request method")
                timeout_ms = spec.get("timeout_ms", default_timeout)
                if not _valid_request_timeout(timeout_ms):
                    raise ValueError("Invalid request timeout")
                body = spec.get("body") or b""
                if _may_pack_body(method, spec["path"], body):
                    body, packed = packed_request_body(body)
                req = self._make_qnetwork_request(
                    spec.get("auth"), timeout_ms, spec["path"], packed)
                nam = self._predict_nam_at(slot // _CONNECTIONS_PER_MANAGER)
                payload = QByteArray(body)


                if not _already_aborting():
                    max_timeout = max(max_timeout, timeout_ms)
                    reply = nam.post(req, payload) if method == "POST" else nam.get(req)
                    if reply is None:
                        raise RuntimeError("Request returned no reply")
            except (AttributeError, KeyError, TypeError, ValueError):
                failures[slot] = {"error": "Invalid request specification", "code": "CLIENT_ERROR"}
            except RuntimeError:
                failures[slot] = {"error": "Request could not be completed", "code": "TIMEOUT"}
            replies.append(reply)
            packed_flags.append(packed)

        remaining = [sum(reply is not None for reply in replies)]

        def _on_one_finished():
            remaining[0] -= 1
            if remaining[0] <= 0:
                loop.quit()

        for reply in replies:
            if reply is None:
                continue
            if reply.isFinished():


                remaining[0] -= 1
            else:
                reply.finished.connect(_on_one_finished)






        safety_timer = QTimer()
        safety_timer.setSingleShot(True)
        safety_timer.setInterval(max_timeout + 5_000)
        safety_timer.timeout.connect(loop.quit)





        abort_timer = None




        if remaining[0] > 0:
            safety_timer.start()
            if should_abort is not None or _current_feedback() is not None:
                abort_timer = QTimer()
                abort_timer.setInterval(250)
                abort_timer.timeout.connect(
                    lambda: loop.quit() if _already_aborting() else None)
                abort_timer.start()


            if not _already_aborting():
                loop.exec()
            if abort_timer is not None:
                abort_timer.stop()
        safety_timer.stop()


        safety_timer.deleteLater()
        if abort_timer is not None:
            abort_timer.deleteLater()

        results = []
        statuses: list[int | None] = []
        retry_delays: dict[int, float] = {}
        for index, (spec, reply) in enumerate(zip(specs, replies)):
            if reply is None:
                results.append(failures.get(index) or _cancelled_answer())
                statuses.append(None)
                continue






            try:
                result = (_cancelled_answer() if _already_aborting() and not reply.isFinished()
                          else self._parse_reply(
                              reply, require_body=bool(spec.get("require_body")),
                              allow_list=bool(spec.get("allow_list"))))
                results.append(result)
                statuses.append(_http_status_of(reply))
                if statuses[-1] in _HANDOFF_STATUSES:
                    retry_delays[index] = _retry_after_s(reply)
            except RuntimeError:
                results.append({
                    "error": tr("Request timed out. Check your connection or try again."),
                    "code": "TIMEOUT"})
                statuses.append(None)
            finally:
                try:
                    if not reply.isFinished():
                        reply.abort()
                    reply.deleteLater()
                except RuntimeError:
                    pass  # nosec B110



        refused = [i for i, packed in enumerate(packed_flags)
                   if packed and specs[i].get("method") == "POST"
                   and answer_refused_the_body(statuses[i])]
        if refused and not _already_aborting():
            _log_warning("A compressed request body was refused; sending "
                         "them plain for the rest of the session")
            note_gzip_request_refused()
            plain = self.request_many(
                [specs[i] for i in refused], should_abort=should_abort,
                retry_reads=False)
            for slot, answer in zip(refused, plain):
                results[slot] = answer
        if retry_reads and not _already_aborting():
            again = [i for i, spec in enumerate(specs)
                     if isinstance(spec, dict) and spec.get("method", "GET") == "GET"
                     and _worth_asking_again(results[i], statuses[i])]
            if again:
                delay = max(retry_delays.get(i, 0.0) for i in again) or _retry_pause_s()

                sleep_unless_cancelled(
                    delay, _already_aborting, slice_s=0.1 if should_abort is not None else max(delay, 0.001))
                if not _already_aborting():
                    second = self.request_many(
                        [specs[i] for i in again], should_abort=should_abort,
                        retry_reads=False)
                    for slot, answer in zip(again, second):
                        results[slot] = answer
        return results

    def post_detection_async(self, submission: dict, auth: dict, should_abort=None):















        from qgis.PyQt.QtCore import QByteArray

        def aborted() -> bool:
            if _request_cancelled():
                return True
            try:
                return bool(should_abort is not None and should_abort())
            except Exception:  # noqa: BLE001
                return False

        if aborted():
            return None
        body = self._build_predict_body(
            submission["run_id"], submission["prompt"], submission["image_b64"],
            submission["tile_index"], submission["crs_authid"],
            submission.get("tile_bbox_wgs84"), submission.get("tile_bbox_native"),
            submission.get("pixel_size_m"), submission.get("max_masks"),
            submission.get("threshold"), submission.get("mask_threshold"),
            submission.get("exemplars"), submission.get("parent_tile_index"),
            self._predict_extras(submission),
        )
        body, packed = packed_request_body(body)
        req = self._make_qnetwork_request(
            auth, self._submit_timeout(), self._detection_predict_url(), packed
        )




        nam = self._next_predict_nam()
        payload = QByteArray(body)
        if aborted():
            return None
        return nam.post(req, payload)
