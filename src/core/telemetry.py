"""Batched telemetry for the AI Segmentation plugin.

Design principles (do not deviate):
- Global opt-out: the shared TerraLab/telemetry_enabled QSettings key. When
  disabled, nothing is even queued. Fail-closed on read errors.
- Events batch in memory and flush once per generation cycle (or immediately
  for FLUSH_NOW milestones/failures). Batching collapses the power-user
  "hundreds of clicks per session" volume into one POST per cycle.
- Lifecycle events (NO_CONSENT_EVENTS) ship as soon as the plugin is activated;
  everything else additionally requires the user to have accepted the ToS.
  Pre-auth lifecycle events park in _pending_pre_auth until the first
  authenticated flush drains them.
- Relay model: plugin -> POST {base}/api/plugin/track -> our analytics relay. No analytics
  key in the plugin. Body shape {"events": [...]} matches the shared route.
- The typed per-event wrappers do NOT live here. They are in
  telemetry_session_events.py (lifecycle, install, pairing, account, manual,
  monetization, library), telemetry_run_events.py (everything an Automatic run
  and its review report) and telemetry_errors.py (plugin_error plus the slot
  and worker boundaries that catch an exception). This module is the transport:
  consent, batching, scrubbing, and the one POST.
- MAIN THREAD ONLY: flush() ends in QgsApplication.taskManager().addTask(),
  which is main-thread-only. Worker threads must only track(); the next
  main-thread flush ships the batch.
- Errors in telemetry never affect plugin functionality (fail silently).
- Payloads carry no paths, coordinates, layer names, urls or emails: every
  string leaving the machine goes through scrub_payload_value first.
- They are NOT anonymous, and nothing here should say they are. Each event
  carries a device_hash that is stable for the life of the install, and the
  batch is posted under the account's bearer key, so the relay can and does
  attach the account to it. track_auto_prompt_committed also sends what the
  user typed, verbatim. The Privacy card in the account dialog has to describe
  this honestly, and the public FAQ has to match it.

HTTP stack: QgsBlockingNetworkRequest (inside the flush QgsTask), so the
relay POST inherits QGIS proxy/TLS settings. No raw requests/urllib.
"""

from __future__ import annotations

import json
import platform
import sys
import threading
import uuid

from qgis.core import (
    Qgis,
    QgsApplication,
    QgsBlockingNetworkRequest,
    QgsTask,
)
from qgis.PyQt.QtCore import QByteArray, QSettings, QThread, QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest

from .qt_compat import HttpStatusCodeAttribute, silent_task_flags
from .telemetry_events import FLUSH_NOW, NO_CONSENT_EVENTS, REGISTRY_VERSION

_TIMEOUT_MS = 5_000
_BATCH_MAX = 10
_PENDING_PRE_AUTH_MAX = 50
_TELEMETRY_ENABLED_KEY = "TerraLab/telemetry_enabled"

# How much telemetry travels is a server dial: batch size, transport timeout,
# and a per-event sample rate for throttling one noisy event during an
# incident. All three default to exactly what ships.
#
# WHAT the plugin is allowed to send is NOT a dial and must never become one.
# The opt-out default, NO_CONSENT_EVENTS and the scrubbing patterns are
# privacy commitments published in the FAQ; a server that could widen them
# could collect, from an installed plugin, something the user never agreed to.
# Volume can be tuned remotely. Scope cannot.


def _batch_max() -> int:
    """Events queued before a flush goes out."""
    try:
        from .server_dials import dial_in_range

        return int(dial_in_range("telemetry.batch_max", _BATCH_MAX, 1, 200))
    except Exception:  # noqa: BLE001 -- telemetry must never break a caller
        return _BATCH_MAX


def _timeout_ms() -> int:
    """Transport timeout on one relay call."""
    try:
        from .server_dials import dial_in_range

        return int(dial_in_range("telemetry.timeout_ms", _TIMEOUT_MS, 500, 60_000))
    except Exception:  # noqa: BLE001 -- telemetry must never break a caller
        return _TIMEOUT_MS


def _event_sampled_in(event: str) -> bool:
    """Whether this occurrence of ``event`` is kept.

    ``telemetry.sample_rates`` is a ``{event_name: rate}`` map, every event
    defaulting to 1.0 (keep everything, today's behaviour). A rate below one
    drops that share of occurrences at random, which is how one noisy event
    gets throttled during an incident without shipping a release. Only a
    number in [0, 1] counts; anything else keeps the event.

    A dropped event is never queued, so this cannot be used to collect more.
    """
    try:
        from .server_dials import read_value

        rates = read_value("telemetry.sample_rates")
        if not isinstance(rates, dict):
            return True
        rate = rates.get(event)
        if not isinstance(rate, (int, float)) or isinstance(rate, bool):
            return True
        if not 0.0 <= rate < 1.0:
            return True
        import random  # noqa: PLC0415 -- only needed when a rate is served

        return random.random() < rate  # nosec B311 -- sampling, not security
    except Exception:  # noqa: BLE001 -- telemetry must never break a caller
        return True


# Guards _batch / _pending_pre_auth / _inflight: track() can run on a worker
# thread while the main thread flushes, so the list mutations must not race.
_lock = threading.Lock()
_batch: list[dict] = []
_pending_pre_auth: list[dict] = []
_inflight: list = []
_session_id = uuid.uuid4().hex

# Most-recent Automatic run correlation id (the id the server archives each
# billed run under). Kept so the error report can quote it for support; None
# until the first run this process. Not telemetry state, just a breadcrumb.
_last_run_id: str | None = None

# The plugin version and relay base URL come from static bundled files that
# never change during a session; memoize them so track()/flush() do not re-read
# metadata.txt (and .env.local) from disk on every event.
_plugin_version_cache: str | None = None
_base_url_cache: str | None = None


# --- Opt-out --------------------------------------------------------------


def is_telemetry_enabled() -> bool:
    """Whether usage telemetry is enabled. Opt-out: defaults to True.

    Reads the shared TerraLab/telemetry_enabled QSettings key (shared with
    AI Edit so the user opts out once). Fail-closed: if the preference cannot be
    read, do NOT send (privacy over a data point)."""
    try:
        return bool(QSettings().value(_TELEMETRY_ENABLED_KEY, True, type=bool))
    except Exception:  # nosec B110
        return False


def set_telemetry_enabled(enabled: bool) -> None:
    """Persist the global telemetry opt-out flag (shared across TerraLab plugins)."""
    try:
        QSettings().setValue(_TELEMETRY_ENABLED_KEY, bool(enabled))
    except Exception:  # nosec B110
        pass


def new_session() -> None:
    """Rotate the per-session id. Call on dock open so events group by session."""
    global _session_id
    _session_id = uuid.uuid4().hex


def set_last_run_id(run_id: str | None) -> None:
    """Remember the most recent Automatic run's correlation id so a later error
    report can quote it. Best-effort breadcrumb, never raises."""
    global _last_run_id
    _last_run_id = run_id or None


def get_last_run_id() -> str | None:
    """The most recent Automatic run id this process, or None if no run yet."""
    return _last_run_id


# --- Payload helpers ------------------------------------------------------


def _base_properties() -> dict:
    """Properties shared by every event (computed once per call, cheap)."""
    try:
        qgis_version = Qgis.QGIS_VERSION
    except Exception:
        qgis_version = "unknown"
    props = {
        "product_id": "ai-segmentation",
        "plugin_version": _read_plugin_version(),
        "os": platform.system(),
        "os_version": platform.release(),
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
    global _plugin_version_cache
    if _plugin_version_cache is not None:
        return _plugin_version_cache
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
    # Cache only a real read: a transient failure ("unknown") is retried next
    # call rather than pinned for the whole session.
    if version != "unknown":
        _plugin_version_cache = version
    return version


def _build_base_url() -> str:
    global _base_url_cache
    if _base_url_cache is not None:
        return _base_url_cache
    import os
    plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env_path = os.path.join(plugin_dir, ".env.local")
    base = "https://terra-lab.ai"
    if os.path.isfile(env_path):
        try:
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("TERRALAB_BASE_URL="):
                        base = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        except Exception:
            pass  # nosec B110
    _base_url_cache = base
    return base


def _get_auth_header() -> dict | None:
    """Return {'Authorization': 'Bearer <key>'} if the plugin is activated."""
    try:
        from .activation_manager import get_auth_header, is_plugin_activated
        if not is_plugin_activated():
            return None
        hdr = get_auth_header()
        if hdr and hdr.get("Authorization"):
            return hdr
    except Exception:
        pass  # nosec B110
    return None


def _has_consent() -> bool:
    """Non-lifecycle events additionally require ToS acceptance (raw message
    fields can carry path fragments; the ToS is the user's data gate)."""
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


# --- Background flush task ------------------------------------------------


class _TelemetryFlushTask(QgsTask):
    """Sends one batch. Failures swallowed: telemetry must never break the plugin."""

    def __init__(self, events: list, auth: dict):
        super().__init__("AI Segmentation telemetry flush", silent_task_flags())
        self._events = events
        self._auth = auth

    def run(self) -> bool:
        if self.isCanceled():
            return False
        # One retry with a short backoff covers a transient network blip without
        # a disk queue; a hard-offline session still loses the batch (accepted).
        if not self._post() and not self.isCanceled():
            import time
            time.sleep(2)
            if self.isCanceled():
                return False
            self._post()
        return True

    def _post(self) -> bool:
        try:
            payload = json.dumps({"events": self._events}).encode("utf-8")
            url = f"{_build_base_url().rstrip('/')}/api/plugin/track"
            req = QNetworkRequest(QUrl(url))
            req.setRawHeader(b"Content-Type", b"application/json")
            if hasattr(req, "setTransferTimeout"):
                req.setTransferTimeout(_timeout_ms())
            for k, v in self._auth.items():
                req.setRawHeader(k.encode("utf-8"), v.encode("utf-8"))
            blocker = QgsBlockingNetworkRequest()
            err = blocker.post(req, QByteArray(payload))
            # ErrorCode 0 = NoError. Ignoring the result made run()'s single
            # retry dead code: a failed batch returned True and was dropped.
            if int(err) != 0:
                return False
            # A transport-level NoError still covers a 4xx/5xx response (the POST
            # reached the relay but it rejected the batch): treat those as a
            # failure too so run()'s single retry fires. An unreadable status
            # (None) stays a success, since the transport itself succeeded.
            status = self._http_status(blocker)
            return status is None or status < 400
        except Exception:
            return False  # nosec B110 - telemetry must never break the plugin

    @staticmethod
    def _http_status(blocker) -> int | None:
        """HTTP status of the reply, or None. Never raises (Qt can hand back a
        non-numeric attribute that int() would choke on)."""
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


# --- Core API -------------------------------------------------------------


def track(event: str, properties: dict | None = None, flush_now: bool = False) -> None:
    """Queue an event. Global opt-out short-circuits before anything is queued.

    Ships immediately when flush_now is True, the event is a FLUSH_NOW milestone,
    or the batch is full; otherwise it waits for the next flush()."""
    if not is_telemetry_enabled():
        return
    # A milestone always travels: sampling is for volume, not for losing the
    # events the funnel is measured on.
    if not (flush_now or event in FLUSH_NOW) and not _event_sampled_in(event):
        return
    try:
        evt = {
            "event": event,
            "properties": {**_base_properties(), **(properties or {})},
        }
    except Exception:  # nosec B110
        return
    with _lock:
        _batch.append(evt)
        should_flush = flush_now or event in FLUSH_NOW or len(_batch) >= _batch_max()
    if should_flush:
        flush()


def flush() -> None:
    """Ship the queued batch. MAIN THREAD ONLY (no-ops off it). Lifecycle events
    ship pre-consent; everything else requires consent. Pre-auth lifecycle events
    park in _pending_pre_auth until the first authenticated flush."""
    if not on_main_thread():
        return
    task = None
    with _lock:
        if not _batch and not _pending_pre_auth:
            return
        auth = _get_auth_header()
        if not auth:
            for evt in _batch:
                if evt["event"] in NO_CONSENT_EVENTS and len(_pending_pre_auth) < _PENDING_PRE_AUTH_MAX:
                    _pending_pre_auth.append(evt)
            _batch.clear()
            return
        consented = _has_consent()
        events_to_send = list(_pending_pre_auth) + [
            e for e in _batch
            if consented or e["event"] in NO_CONSENT_EVENTS
        ]
        _batch.clear()
        _pending_pre_auth.clear()
        if not events_to_send:
            return
        task = _TelemetryFlushTask(events_to_send, auth)
        _inflight.append(task)
    try:
        task.taskCompleted.connect(lambda t=task: _drop_inflight(t))
        task.taskTerminated.connect(lambda t=task: _drop_inflight(t))
    except Exception:  # nosec B110
        pass
    QgsApplication.taskManager().addTask(task)


# --- Payload scrubbing (kept as-is) ---------------------------------------


_COORD_PATTERN = None
_URL_PATTERN = None
_EMAIL_PATTERN = None


def scrub_payload_value(value: str) -> str:
    """Strip path-like tokens, coordinate tuples, URLs and email addresses
    from telemetry strings.

    Applied defensively to any string leaving the machine. We already call
    log_scrub.anonymize_paths for filesystem paths, but this pass also catches
    leftover coordinate-like artefacts (crop bounds, click tuples, bbox
    extents) plus URLs/emails: the unhandled-error catch-all forwards raw
    third-party exception text, which can embed a host or an address, and the
    telemetry contract is no URLs and no emails, so redact rather than trust
    the source.
    """
    import re as _re
    global _COORD_PATTERN, _URL_PATTERN, _EMAIL_PATTERN
    if _COORD_PATTERN is None:
        _COORD_PATTERN = _re.compile(
            r"(?:[-+]?\d+(?:\.\d+)?)(?:\s*,\s*[-+]?\d+(?:\.\d+)?){1,}"
        )
        _URL_PATTERN = _re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s'\"]+")
        _EMAIL_PATTERN = _re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
    # core-to-core import, deliberately NOT wrapped in try/except: a broken
    # scrubber must fail loudly, never silently ship unscrubbed paths.
    from .log_scrub import anonymize_paths
    value = anonymize_paths(value)
    value = _URL_PATTERN.sub("<URL>", value or "")
    value = _EMAIL_PATTERN.sub("<EMAIL>", value)
    return _COORD_PATTERN.sub("<COORDS>", value)
