"""Minimal telemetry for AI Segmentation plugin.

Design principles (do not deviate):
- All telemetry respects a global opt-out: the QSettings key
  `TerraLab/telemetry_enabled` (default on, shared across TerraLab plugins so
  disabling it in one disables it everywhere). When off, `_send` drops every
  event before any network call. Users flip it from the account settings dialog.
- All events are anonymous: the payload carries only OS, plugin version, QGIS
  version, a success bool, counts, and durations. No personal data, no content,
  no coordinates. The lifecycle events (plugin_opened, plugin_activated,
  segmentation_run, session_summary, and the install/download/export/Pro events)
  carry no user-generated content, so they need no gate beyond the opt-out above.
- `plugin_error` is gated additionally, because its scrubbed message field could
  carry path fragments. The gate is the existing ToS acceptance
  the user gives via the dock checkbox before their first segmentation
  (`tos_accepted` in activation_manager.py).
- File paths are anonymized (<USER> placeholder) via error_report_dialog's
  _anonymize_paths before leaving the machine.
- Errors in telemetry never affect plugin functionality (fail silently).
- Non-blocking: POSTs run on a background QThread, with a single ~2s-backoff
  retry inside the worker. A hard-offline session loses events (no disk queue).
- Event names come from telemetry_events.py constants only, never raw strings.

sample_rate semantics: an integer N means "1 in N kept". 1 means unsampled.
Dashboards multiply counts by sample_rate. An event without sample_rate is
uninterpretable, so segmentation_run ALWAYS sends it.
"""

from __future__ import annotations

import json
import platform
import sys
import time

from qgis.core import Qgis, QgsBlockingNetworkRequest
from qgis.PyQt.QtCore import QByteArray, QThread, QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest

from . import telemetry_events as ev

_TIMEOUT_MS = 5_000
_RETRY_BACKOFF_S = 2.0

# Global opt-out switch. Shared across TerraLab plugins (same namespace as
# TerraLab/device_seed) so the user only has to disable telemetry once.
_TELEMETRY_ENABLED_KEY = "TerraLab/telemetry_enabled"


def is_telemetry_enabled() -> bool:
    """Whether anonymous usage telemetry is enabled. Opt-out: defaults to True.

    Reads the shared TerraLab/telemetry_enabled QSettings key. Fails closed: if
    the preference cannot be read, we do NOT send (privacy takes precedence over
    a data point).
    """
    try:
        from qgis.PyQt.QtCore import QSettings
        return bool(QSettings().value(_TELEMETRY_ENABLED_KEY, True, type=bool))
    except Exception:  # nosec B110
        return False


def set_telemetry_enabled(enabled: bool) -> None:
    """Persist the global telemetry opt-out flag (shared across TerraLab plugins)."""
    try:
        from qgis.PyQt.QtCore import QSettings
        QSettings().setValue(_TELEMETRY_ENABLED_KEY, bool(enabled))
    except Exception:
        pass  # nosec B110 — a settings write failure must not break the UI


# Anonymous lifecycle events that carry no user-generated content, so they need
# no gate beyond the global opt-out (plugin_error is gated separately).
_NO_CONTENT_EVENTS = frozenset({
    ev.PLUGIN_OPENED,
    ev.PLUGIN_ACTIVATED,
    ev.ACTIVATION_ATTEMPTED,
    ev.DEPENDENCIES_INSTALLED,
    ev.MODEL_DOWNLOAD_COMPLETED,
    ev.SEGMENTATION_RUN,
    ev.SESSION_SUMMARY,
    ev.EXPORT_COMPLETED,
    ev.PRO_PANEL_VIEWED,
    ev.SUBSCRIBE_LINK_CLICKED,
    ev.TRIAL_EXHAUSTED_VIEWED,
})


# --- Payload helpers ------------------------------------------------------


def _base_properties() -> dict:
    """Properties shared by every event (computed once per call, cheap)."""
    try:
        qgis_version = Qgis.QGIS_VERSION
    except Exception:
        qgis_version = "unknown"
    return {
        "product_id": "ai-segmentation",
        "source": "qgis_plugin",
        "plugin_version": _read_plugin_version(),
        "os": platform.system(),
        "os_version": platform.release(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "qgis_version": qgis_version,
    }


def _read_plugin_version() -> str:
    import os
    try:
        plugin_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        metadata_path = os.path.join(plugin_dir, "metadata.txt")
        with open(metadata_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("version="):
                    return line.strip().split("=", 1)[1]
    except Exception:
        pass  # nosec B110
    return "unknown"


def _build_base_url() -> str:
    import os
    plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env_path = os.path.join(plugin_dir, ".env.local")
    if os.path.isfile(env_path):
        try:
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("TERRALAB_BASE_URL="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass  # nosec B110
    return "https://terra-lab.ai"


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


# --- Background HTTP worker ----------------------------------------------


class _TelemetryWorker(QThread):
    """Fire-and-forget single-event HTTP POST on a background thread.

    One retry with a short backoff on transport failure. Never blocks the UI
    thread, never raises into it.
    """

    def __init__(self, url: str, headers: dict, payload: bytes):
        super().__init__()
        self._url = url
        self._headers = headers
        self._payload = payload

    def _post_once(self) -> bool:
        req = QNetworkRequest(QUrl(self._url))
        req.setRawHeader(b"Content-Type", b"application/json")
        if hasattr(req, "setTransferTimeout"):
            req.setTransferTimeout(_TIMEOUT_MS)
        for k, v in self._headers.items():
            req.setRawHeader(k.encode("utf-8"), v.encode("utf-8"))
        blocker = QgsBlockingNetworkRequest()
        err = blocker.post(req, QByteArray(self._payload))
        return err == QgsBlockingNetworkRequest.ErrorCode.NoError

    def run(self):
        try:
            if self._post_once():
                return
        except Exception:
            pass  # nosec B110 — telemetry must never break the plugin
        try:
            time.sleep(_RETRY_BACKOFF_S)
            self._post_once()
        except Exception:
            pass  # nosec B110


_workers: list = []


def _on_worker_finished(worker: _TelemetryWorker):
    try:
        _workers.remove(worker)
    except ValueError:
        pass
    worker.deleteLater()


# --- Public API -----------------------------------------------------------


def _send(event: str, properties: dict | None = None) -> bool:
    """Build payload + ship. Returns True once the background worker has been queued.

    Anonymous events in `_NO_CONTENT_EVENTS` ship as long as the plugin is
    activated; everything else additionally requires the user to have accepted
    the ToS (via the dock checkbox before first segmentation).

    The global opt-out is checked first: if the user disabled telemetry, no
    event is ever built or sent.
    """
    if not is_telemetry_enabled():
        return False
    auth = _get_auth_header()
    if not auth:
        return False
    if event not in _NO_CONTENT_EVENTS:
        try:
            from .activation_manager import has_tos_accepted, has_tos_locked
            if not (has_tos_accepted() or has_tos_locked()):
                return False
        except Exception:
            return False
    try:
        payload = json.dumps({
            "event": event,
            "properties": {
                **_base_properties(),
                **(properties or {}),
            },
        }).encode("utf-8")
        url = f"{_build_base_url().rstrip('/')}/api/plugin/track"
        worker = _TelemetryWorker(url, auth, payload)
        _workers.append(worker)
        worker.finished.connect(lambda w=worker: _on_worker_finished(w))
        worker.start()
        return True
    except Exception:
        return False  # nosec B110


def track_plugin_opened() -> bool:
    """Fire once per dock-open. Returns True only if the event was actually
    queued so callers can gate their 'emitted once' flag correctly."""
    return _send(ev.PLUGIN_OPENED)


def track_plugin_activated() -> bool:
    """Fire when the activation key is validated."""
    return _send(ev.PLUGIN_ACTIVATED)


def track_activation_attempted(success: bool, error_code: str | None = None) -> bool:
    """Fire on every activation-key validation attempt (success and failure).

    error_code is a short machine id on failure (never the raw server message)."""
    props: dict = {"success": bool(success)}
    if not success and error_code:
        props["error_code"] = error_code
    return _send(ev.ACTIVATION_ATTEMPTED, props)


def track_dependencies_installed(duration_ms: int | None = None) -> bool:
    """Fire after a successful dependency install. Failures emit plugin_error."""
    return _send(
        ev.DEPENDENCIES_INSTALLED,
        {"success": True, "duration_ms": duration_ms},
    )


def track_model_download_completed(
    duration_ms: int | None = None, model_id: str | None = None
) -> bool:
    """Fire after a successful model checkpoint download.

    model_id is an abstract family id (e.g. "sam2"/"sam1"), never a file name."""
    return _send(
        ev.MODEL_DOWNLOAD_COMPLETED,
        {"duration_ms": duration_ms, "model_id": model_id},
    )


# segmentation_run is sampled 1-in-N. session_summary (unsampled) carries the
# exact totals, so per-run sampling never loses the headline numbers.
_SEGMENTATION_SAMPLE_RATE = 10
_segmentation_run_sent_this_session = False


def track_segmentation_run(
    success: bool,
    duration_ms: int | None = None,
    error_code: str | None = None,
    mode: str = "local",
) -> bool:
    """Fire when a segmentation run finishes.

    success=True is a saved selection; success=False is a crop/encoding/
    prediction failure mid-run (today those only surfaced as plugin_error).

    Sampling: power users click hundreds of times per session, making the
    success path the bulk of telemetry volume. The first success of a session
    is always sent (sample_rate: 1) so light users stay represented;
    after that, 1-in-N with sample_rate set so dashboards re-weight. Failures
    are rare and precious, so they are ALWAYS sent unsampled (sample_rate: 1)
    regardless of the session counter.

    sample_rate is ALWAYS present (an event without it is uninterpretable)."""
    global _segmentation_run_sent_this_session
    import random

    if not success:
        sample_rate = 1
    elif _segmentation_run_sent_this_session:
        if random.random() >= (1.0 / _SEGMENTATION_SAMPLE_RATE):  # nosec B311 - sampling, not crypto
            return True
        sample_rate = _SEGMENTATION_SAMPLE_RATE
    else:
        _segmentation_run_sent_this_session = True
        sample_rate = 1

    props: dict = {
        "success": bool(success),
        "duration_ms": duration_ms,
        "sample_rate": sample_rate,
        "mode": mode,
    }
    if not success and error_code:
        props["error_code"] = error_code
    return _send(ev.SEGMENTATION_RUN, props)


def track_session_summary(
    run_count: int,
    success_count: int,
    export_count: int,
    session_duration_ms: int | None = None,
) -> bool:
    """Fire once on dock close / plugin unload. Unsampled exact totals.

    Caller guards against double emission per session; a hard QGIS kill loses
    this event, which is accepted."""
    return _send(
        ev.SESSION_SUMMARY,
        {
            "run_count": int(run_count),
            "success_count": int(success_count),
            "export_count": int(export_count),
            "session_duration_ms": session_duration_ms,
        },
    )


def track_export_completed(feature_count: int, format: str = "gpkg") -> bool:
    """Fire after a successful vector export. format is "gpkg" or "shp"."""
    return _send(
        ev.EXPORT_COMPLETED,
        {"format": format, "feature_count": int(feature_count)},
    )


def track_pro_panel_viewed() -> bool:
    """Fire when the Pro upsell panel is first shown in a session."""
    return _send(ev.PRO_PANEL_VIEWED)


def track_subscribe_link_clicked(source: str) -> bool:
    """Fire on any upgrade/subscribe CTA click. source names the CTA location."""
    return _send(ev.SUBSCRIBE_LINK_CLICKED, {"source": source})


def track_trial_exhausted_viewed(is_free_tier: bool) -> bool:
    """Fire when a quota/trial exhausted state is shown (Pro)."""
    return _send(ev.TRIAL_EXHAUSTED_VIEWED, {"is_free_tier": bool(is_free_tier)})


_COORD_PATTERN = None


def _scrub_payload_value(value: str) -> str:
    """Strip path-like tokens and numeric coordinate tuples from telemetry strings.

    Applied defensively to any string leaving the machine. We already call
    _anonymize_paths upstream for filesystem paths, but this pass also catches
    leftover coordinate-like artefacts that could appear in logs (crop bounds,
    click tuples, bbox extents, etc.).
    """
    import re as _re
    global _COORD_PATTERN
    if _COORD_PATTERN is None:
        # Matches sequences of 2+ comma-separated signed floats/ints — e.g.
        # "(12.345, -67.89)" or "bbox=100,200,300,400". Replaces with <COORDS>.
        _COORD_PATTERN = _re.compile(
            r"(?:[-+]?\d+(?:\.\d+)?)(?:\s*,\s*[-+]?\d+(?:\.\d+)?){1,}"
        )
    try:
        from ..ui.error_report_dialog import _anonymize_paths
        value = _anonymize_paths(value)
    except Exception:
        pass  # nosec B110
    return _COORD_PATTERN.sub("<COORDS>", value or "")


def track_plugin_error(
    stage: str,
    error_code: str,
    message: str,
    include_log_tail: bool = False,
):
    """Fire when an error is shown to the user or an exception is caught.

    stage: install | download | activate | segment | export | other
    error_code: short machine-friendly id (e.g. "PIP_TIMEOUT", "RUNTIME_ERROR")
    message: first line of the error, truncated to 200 chars, path + coord scrubbed
    include_log_tail: OFF by default. When True, the last 20 anonymized log lines
        are capped to ~4KB and coordinate-scrubbed before being sent. We default
        off because QGIS runtime logs frequently contain click coords, crop
        bounds, and export extents that we promised never to collect.
    """
    props = {
        "stage": stage,
        "error_code": error_code,
        "error_message": _scrub_payload_value((message or "")[:200]),
    }
    if include_log_tail:
        try:
            from ..ui.error_report_dialog import _get_recent_logs
            tail_lines = _get_recent_logs().splitlines()[-20:]
            scrubbed = _scrub_payload_value("\n".join(tail_lines))
            props["last_log_lines"] = scrubbed.encode("utf-8")[:4096].decode(
                "utf-8", errors="ignore"
            )
        except Exception:
            pass  # nosec B110
    _send(ev.PLUGIN_ERROR, props)
