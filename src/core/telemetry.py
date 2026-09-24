








































from __future__ import annotations

import json
import platform
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

from qgis.core import (
    Qgis,
    QgsApplication,
    QgsBlockingNetworkRequest,
    QgsFeedback,
    QgsTask,
)
from qgis.PyQt.QtCore import QByteArray, QSettings, QThread, QTimer, QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest

from . import transport_dials as _td
from .gil_safe_qobject import prime as _gil_safe
from .qt_compat import HttpStatusCodeAttribute, silent_task_flags
from .telemetry_events import FLUSH_NOW, NO_CONSENT_EVENTS, REGISTRY_VERSION
from .telemetry_payload import _scrub_telemetry_properties, scrub_payload_value  # noqa: F401

_TIMEOUT_MS = 5_000
_BATCH_MAX = 10




TELEMETRY_PRODUCT_ID = "ai-segmentation"
_PENDING_PRE_AUTH_MAX = 50
_TELEMETRY_ENABLED_KEY = "TerraLab/telemetry_enabled"



_BATCH_HARD_MAX = 200


_POST_MAX_BYTES = 128 * 1024
_INFLIGHT_MAX = 8

_FLUSH_INTERVAL_S = 60

_RETRY_BACKOFF_S = 2.0












def _batch_max() -> int:

    try:
        from .server_dials import dial_in_range

        return int(dial_in_range("telemetry.batch_max", _BATCH_MAX, 1, 200))
    except Exception:  # noqa: BLE001
        return _BATCH_MAX


def _inflight_max() -> int:

    try:
        from .server_dials import dial_in_range

        return int(dial_in_range("telemetry.inflight_max", _INFLIGHT_MAX, 1, 32))
    except Exception:  # noqa: BLE001
        return _INFLIGHT_MAX


def _timeout_ms() -> int:






    try:
        from .server_dials import dial_in_range

        return int(dial_in_range("telemetry.timeout_ms", _TIMEOUT_MS, 500, 10_000))
    except Exception:  # noqa: BLE001
        return _TIMEOUT_MS


def _event_sampled_in(event: str, urgent: bool = False) -> bool:















    try:
        from .server_dials import read_value

        key = "telemetry.urgent_sample_rates" if urgent else "telemetry.sample_rates"
        rates = read_value(key)
        if not isinstance(rates, dict):
            return True
        rate = rates.get(event)
        if not isinstance(rate, (int, float)) or isinstance(rate, bool):
            return True
        if not 0.0 <= rate < 1.0:
            return True
        import random  # noqa: PLC0415

        return random.random() < rate  # nosec B311
    except Exception:  # noqa: BLE001
        return True




_lock = threading.Lock()
_batch: list[dict] = []
_pending_pre_auth: list[dict] = []
_inflight: list = []
_session_id = uuid.uuid4().hex

_flush_timer = None




_last_run_id: str | None = None




_plugin_version_memo: dict[str, str] = {}


_enabled_memo: dict = {"value": None, "at": 0.0}
_ENABLED_CACHE_S = 5.0





def is_telemetry_enabled() -> bool:










    now = time.monotonic()
    held = _enabled_memo["value"]
    if held is not None and (now - _enabled_memo["at"]) < _ENABLED_CACHE_S:
        return held
    try:
        value = bool(QSettings().value(_TELEMETRY_ENABLED_KEY, True, type=bool))
    except Exception:  # nosec B110
        value = False
    _enabled_memo["value"], _enabled_memo["at"] = value, now
    return value


def _forget_enabled_cache() -> None:

    _enabled_memo["value"] = None


def set_telemetry_enabled(enabled: bool) -> None:







    try:
        QSettings().setValue(_TELEMETRY_ENABLED_KEY, bool(enabled))
    except Exception:  # nosec B110
        pass
    _forget_enabled_cache()
    if not enabled:
        drop_queued_events()


def drop_queued_events() -> None:










    with _lock:
        _batch.clear()
        _pending_pre_auth.clear()
        pending_tasks = list(_inflight)
        _inflight.clear()
    for task in pending_tasks:
        try:
            task.cancel()
        except Exception:  # nosec B110
            pass


def new_session() -> None:

    global _session_id
    _session_id = uuid.uuid4().hex


def current_session_id() -> str:


    return _session_id


def set_last_run_id(run_id: str | None) -> None:


    global _last_run_id
    _last_run_id = run_id or None


def get_last_run_id() -> str | None:

    return _last_run_id







_WINDOWS_11_FIRST_BUILD = 22000


def _os_release() -> str:

    release = platform.release()
    if sys.platform != "win32" or release != "10":
        return release
    try:
        if sys.getwindowsversion().build >= _WINDOWS_11_FIRST_BUILD:
            return "11"
    except (AttributeError, OSError):
        pass
    return release


def _base_properties() -> dict:

    try:
        qgis_version = Qgis.QGIS_VERSION
    except Exception:
        qgis_version = "unknown"
    props = {
        "product_id": TELEMETRY_PRODUCT_ID,
        "plugin_version": _read_plugin_version(),
        "os": platform.system(),
        "os_version": _os_release(),
        "arch": platform.machine(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "qgis_version": qgis_version,
        "session_id": _session_id,
        "registry_version": REGISTRY_VERSION,
    }
    try:
        from .device_id import get_device_hash
        props["device_hash"] = get_device_hash()
    except Exception:  # nosec B110
        pass
    return props


def _read_plugin_version() -> str:
    cached = _plugin_version_memo.get("version")
    if cached is not None:
        return cached
    import os
    version = "unknown"
    try:
        plugin_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        metadata_path = os.path.join(plugin_dir, "metadata.txt")
        with open(metadata_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("version="):
                    version = line.strip().split("=", 1)[1]
                    break
    except Exception:
        pass  # nosec B110


    if version != "unknown":
        _plugin_version_memo["version"] = version
    return version


def _build_base_url() -> str:


    from .env_local import terralab_base_url

    return terralab_base_url()


def _get_auth_header() -> dict | None:

    try:
        from .activation_manager import get_auth_header
        hdr = get_auth_header()
        if hdr and hdr.get("Authorization"):
            return hdr
    except Exception:
        pass  # nosec B110
    return None


def _has_consent() -> bool:









    try:
        from .activation_manager import has_tos_accepted, has_tos_locked
        return bool(has_tos_accepted() or has_tos_locked())
    except Exception:
        return False


def on_main_thread() -> bool:
    try:
        app = QgsApplication.instance()
        return app is not None and QThread.currentThread() == app.thread()
    except Exception:
        return False





class _TelemetryFlushTask(QgsTask):


    def __init__(self, events: list, auth: dict):
        super().__init__("AI Segmentation telemetry flush", silent_task_flags())
        self._events = events
        self._auth = dict(auth)
        from .activation_manager import auth_revision
        self._auth_revision = auth_revision()



        self._feedback = _gil_safe(QgsFeedback())

    def cancel(self) -> None:
        try:
            self._feedback.cancel()
        except Exception:  # nosec B110
            pass
        super().cancel()

    def run(self) -> bool:
        if self.isCanceled():
            return False


        if not self._post() and not self.isCanceled():
            self._wait_before_retry(_td.telemetry_retry_backoff_s(_RETRY_BACKOFF_S))
            if self.isCanceled():
                return False
            self._post()
        return True

    def _wait_before_retry(self, seconds: float) -> None:

        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self.isCanceled():
                return
            time.sleep(0.1)

    def _post(self) -> bool:
        try:
            _forget_enabled_cache()
            if self.isCanceled() or not is_telemetry_enabled():
                return True
            from .activation_manager import auth_revision
            if not _has_consent() or auth_revision() != self._auth_revision:
                return True
            payload = json.dumps({"events": self._events}).encode("utf-8")
            url = f"{_build_base_url().rstrip('/')}/api/plugin/track"
            from .server_dials import cleartext_remote_url

            if cleartext_remote_url(url):


                return True
            req = QNetworkRequest(QUrl(url))
            req.setRawHeader(b"Content-Type", b"application/json")
            if hasattr(req, "setTransferTimeout"):
                req.setTransferTimeout(_timeout_ms())
            for k, v in self._auth.items():
                req.setRawHeader(k.encode("utf-8"), v.encode("utf-8"))


            blocker = _gil_safe(QgsBlockingNetworkRequest())
            try:
                err = blocker.post(req, QByteArray(payload), False, self._feedback)
            except TypeError:


                err = blocker.post(req, QByteArray(payload))


            if int(err) != 0:
                return False







            status = self._http_status(blocker)
            if status is None or status < 400:
                return True
            return not (status >= 500 or status == 429)
        except Exception:
            return False  # nosec B110

    @staticmethod
    def _http_status(blocker) -> int | None:


        if HttpStatusCodeAttribute is None:
            return None
        try:
            reply = blocker.reply()
            if reply is None:
                return None
            attr = reply.attribute(HttpStatusCodeAttribute)
        except (RuntimeError, AttributeError):
            return None
        if attr is None:
            return None
        try:
            return int(attr)
        except (TypeError, ValueError):
            return None

    def finished(self, result: bool) -> None:
        return


def _drop_inflight(task: _TelemetryFlushTask) -> None:
    with _lock:
        try:
            _inflight.remove(task)
        except ValueError:
            pass





def _event_timestamp() -> str:

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def track(event: str, properties: dict | None = None, flush_now: bool = False) -> None:




    if not is_telemetry_enabled():
        return
    if not isinstance(event, str) or not event or len(event) > 128:
        return
    urgent = flush_now or event in FLUSH_NOW
    if not _event_sampled_in(event, urgent):
        return
    try:
        evt = {
            "event": event,



            "timestamp": _event_timestamp(),
            "properties": _scrub_telemetry_properties(
                {**_base_properties(), **(properties or {})}),
        }




        evt["properties"]["product_id"] = TELEMETRY_PRODUCT_ID
    except Exception:  # nosec B110
        return
    with _lock:
        _batch.append(evt)
        _trim_batch_locked()
        should_flush = urgent or len(_batch) >= _batch_max()
    _arm_flush_timer()
    if should_flush:
        flush()


def _trim_batch_locked() -> None:







    drop = len(_batch) - _td.telemetry_batch_hard_max(_BATCH_HARD_MAX)
    if drop <= 0:
        return
    kept: list[dict] = []
    for evt in _batch:
        if drop > 0 and evt.get("event") not in FLUSH_NOW:
            drop -= 1
            continue
        kept.append(evt)
    if drop > 0:
        kept = kept[drop:]
    _batch[:] = kept


def _on_flush_timer() -> None:


    try:
        flush()
    except Exception:  # nosec B110
        pass


def _arm_flush_timer() -> None:







    global _flush_timer
    if _flush_timer is not None or not on_main_thread():
        return
    try:
        timer = QTimer()
        timer.setInterval(_td.telemetry_flush_interval_s(_FLUSH_INTERVAL_S) * 1000)
        timer.timeout.connect(_on_flush_timer)
        timer.start()
        _flush_timer = timer
    except Exception:  # nosec B110
        _flush_timer = None


def stop_flush_timer() -> None:






    global _flush_timer
    timer = _flush_timer
    _flush_timer = None
    if timer is None:
        return
    try:
        timer.stop()
        timer.timeout.disconnect(_on_flush_timer)
    except Exception:  # nosec B110
        pass
    try:
        timer.deleteLater()
    except Exception:  # nosec B110
        pass


def _split_for_post(events: list[dict]) -> list[list[dict]]:





    chunks: list[list[dict]] = []
    current: list[dict] = []
    envelope_bytes = len(b'{"events": []}')
    size = envelope_bytes
    post_max = _td.telemetry_post_max_bytes(_POST_MAX_BYTES)
    for evt in events:
        try:
            evt_bytes = len(json.dumps(evt).encode("utf-8"))
        except (TypeError, ValueError, OverflowError, RecursionError):

            continue
        if envelope_bytes + evt_bytes > post_max:
            continue
        separator_bytes = 2 if current else 0
        if current and size + separator_bytes + evt_bytes > post_max:
            chunks.append(current)
            current = []
            size = envelope_bytes
            separator_bytes = 0
        current.append(evt)
        size += separator_bytes + evt_bytes
    if current:
        chunks.append(current)
    return chunks


def flush() -> None:



    if not on_main_thread():
        return




    _forget_enabled_cache()
    if not is_telemetry_enabled():
        drop_queued_events()
        return
    with _lock:
        if not _batch and not _pending_pre_auth:
            return
        available = max(0, _inflight_max() - len(_inflight))
        if not available:
            return






    auth = _get_auth_header()
    consented = _has_consent()
    with _lock:
        if not _batch and not _pending_pre_auth:
            return
        if not auth or not consented:
            pre_auth_max = _td.telemetry_pending_pre_auth_max(_PENDING_PRE_AUTH_MAX)
            for evt in _batch:
                if evt["event"] in NO_CONSENT_EVENTS and len(_pending_pre_auth) < pre_auth_max:
                    _pending_pre_auth.append(evt)
            _batch.clear()
            return
        events_to_send = list(_pending_pre_auth) + list(_batch)
        _batch.clear()
        _pending_pre_auth.clear()
    if not events_to_send:
        return


    chunks = _split_for_post(events_to_send)
    tasks = [_TelemetryFlushTask(chunk, dict(auth)) for chunk in chunks[:available]]
    with _lock:
        _inflight.extend(tasks)
        for chunk in chunks[available:]:
            _batch.extend(chunk)
        _trim_batch_locked()
    for task in tasks:
        try:
            task.taskCompleted.connect(lambda t=task: _drop_inflight(t))
            task.taskTerminated.connect(lambda t=task: _drop_inflight(t))
        except Exception:  # nosec B110
            pass
        try:
            accepted = QgsApplication.taskManager().addTask(task)
            if accepted is False:
                _drop_inflight(task)
        except Exception:  # noqa: BLE001
            _drop_inflight(task)
