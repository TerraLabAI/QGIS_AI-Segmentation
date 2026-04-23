"""Minimal telemetry for AI Segmentation plugin.

Design principles (do not deviate):
- Only fires after the user has activated AND given consent. No telemetry
  on QGIS startup, no background polling, no pre-activation tracking.
- Four events only: plugin_opened, plugin_activated, segmentation_run,
  plugin_error.
- No PII, no image data, no click coordinates, no prompts, no layer names.
- File paths are anonymized (<USER> placeholder) via error_report_dialog's
  _anonymize_paths before leaving the machine.
- Errors in telemetry never affect plugin functionality (fail silently).
- Non-blocking: POSTs run on a background QThread.
"""

from __future__ import annotations

import json
import platform
import sys

from qgis.core import Qgis, QgsBlockingNetworkRequest
from qgis.PyQt.QtCore import QByteArray, QSettings, QThread, QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest

_TIMEOUT_MS = 5_000
# Keep the same QSettings namespace as the rest of the plugin
# (see src/core/activation_manager.py) so the whole plugin state
# lives under one root.
_SETTINGS_PREFIX = "AISegmentation/telemetry"


# --- Consent --------------------------------------------------------------


def has_consent() -> bool:
    """True only after the user has explicitly accepted telemetry."""
    s = QSettings()
    return s.value(f"{_SETTINGS_PREFIX}/consent", False, type=bool)


def set_consent(granted: bool):
    """Persist user's telemetry choice."""
    s = QSettings()
    s.setValue(f"{_SETTINGS_PREFIX}/consent", bool(granted))


# --- Payload helpers ------------------------------------------------------


def _base_properties() -> dict:
    """Properties shared by every event (computed once per call, cheap)."""
    try:
        qgis_version = Qgis.QGIS_VERSION
    except Exception:
        qgis_version = "unknown"
    return {
        "product_id": "ai-segmentation",
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
    """Fire-and-forget single-event HTTP POST on a background thread."""

    def __init__(self, url: str, headers: dict, payload: bytes):
        super().__init__()
        self._url = url
        self._headers = headers
        self._payload = payload

    def run(self):
        try:
            req = QNetworkRequest(QUrl(self._url))
            req.setRawHeader(b"Content-Type", b"application/json")
            if hasattr(req, "setTransferTimeout"):
                req.setTransferTimeout(_TIMEOUT_MS)
            for k, v in self._headers.items():
                req.setRawHeader(k.encode("utf-8"), v.encode("utf-8"))
            blocker = QgsBlockingNetworkRequest()
            blocker.post(req, QByteArray(self._payload))
        except Exception:
            pass  # nosec B110 — telemetry must never break the plugin


_workers: list = []


def _on_worker_finished(worker: _TelemetryWorker):
    try:
        _workers.remove(worker)
    except ValueError:
        pass
    worker.deleteLater()


# --- Public API -----------------------------------------------------------


def _send(event: str, properties: dict | None = None) -> bool:
    """Build payload + ship. No-op (returns False) if consent missing or plugin
    not activated. Returns True once the background worker has been queued."""
    if not has_consent():
        return False
    auth = _get_auth_header()
    if not auth:
        # Never track an unactivated user — matches the "jamais quand le
        # plugin n'est pas utilisé" rule + avoids any anonymous-identity debate.
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
    return _send("plugin_opened")


def track_plugin_activated() -> bool:
    """Fire when the activation key is validated."""
    return _send("plugin_activated")


def track_segmentation_run(success: bool, duration_ms: int | None = None) -> bool:
    """Fire when a segmentation run completes (success = saved polygon)."""
    return _send(
        "segmentation_run",
        {"success": bool(success), "duration_ms": duration_ms},
    )


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
    message: first line of the error, truncated to 500 chars, path + coord scrubbed
    include_log_tail: OFF by default. When True, the last 20 anonymized log lines
        are capped to ~4KB and coordinate-scrubbed before being sent. We default
        off because QGIS runtime logs frequently contain click coords, crop
        bounds, and export extents that we promised never to collect.
    """
    props = {
        "stage": stage,
        "error_code": error_code,
        "message": _scrub_payload_value((message or "")[:500]),
    }
    if include_log_tail:
        try:
            from ..ui.error_report_dialog import _get_recent_logs
            tail_lines = _get_recent_logs().splitlines()[-20:]
            scrubbed = _scrub_payload_value("\n".join(tail_lines))
            # Hard byte cap: PostHog silently drops oversized properties.
            props["last_log_lines"] = scrubbed.encode("utf-8")[:4096].decode(
                "utf-8", errors="ignore"
            )
        except Exception:
            pass  # nosec B110
    _send("plugin_error", props)
