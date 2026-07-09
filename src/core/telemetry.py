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
- MAIN THREAD ONLY: flush() ends in QgsApplication.taskManager().addTask(),
  which is main-thread-only. Worker threads must only track(); the next
  main-thread flush ships the batch.
- Errors in telemetry never affect plugin functionality (fail silently).
- Payloads carry only OS/version/timing/counts/enums, never PII: no paths,
  coordinates, layer names, urls, or emails.

HTTP stack: QgsBlockingNetworkRequest (inside the flush QgsTask), so the
relay POST inherits QGIS proxy/TLS settings. No raw requests/urllib.
"""

from __future__ import annotations

import functools
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

from . import telemetry_events as ev
from .telemetry_events import FLUSH_NOW, NO_CONSENT_EVENTS, REGISTRY_VERSION

_TIMEOUT_MS = 5_000
_BATCH_MAX = 10
_PENDING_PRE_AUTH_MAX = 50
_TELEMETRY_ENABLED_KEY = "TerraLab/telemetry_enabled"

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


# --- Opt-out --------------------------------------------------------------


def is_telemetry_enabled() -> bool:
    """Whether anonymous usage telemetry is enabled. Opt-out: defaults to True.

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


def _has_consent() -> bool:
    """Non-lifecycle events additionally require ToS acceptance (raw message
    fields can carry path fragments; the ToS is the user's data gate)."""
    try:
        from .activation_manager import has_tos_accepted, has_tos_locked
        return bool(has_tos_accepted() or has_tos_locked())
    except Exception:
        return False


def _silent_task_flags():
    """CanCancel plus Hidden/Silent when the running QGIS exposes them.

    Hidden/Silent landed in QGIS 3.26; resolve defensively so the task-manager
    widget never fills with "AI Segmentation telemetry" rows on modern builds,
    and degrades to a plain cancellable task on older ones."""
    flags = QgsTask.Flag.CanCancel
    for name in ("Hidden", "Silent"):
        flag = getattr(QgsTask.Flag, name, None)
        if flag is not None:
            flags = flags | flag
    return flags


def _on_main_thread() -> bool:
    try:
        app = QgsApplication.instance()
        return app is not None and QThread.currentThread() == app.thread()
    except Exception:
        return False


# --- Background flush task ------------------------------------------------


class _TelemetryFlushTask(QgsTask):
    """Sends one batch. Failures swallowed: telemetry must never break the plugin."""

    def __init__(self, events: list, auth: dict):
        super().__init__("AI Segmentation telemetry flush", _silent_task_flags())
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
                req.setTransferTimeout(_TIMEOUT_MS)
            for k, v in self._auth.items():
                req.setRawHeader(k.encode("utf-8"), v.encode("utf-8"))
            blocker = QgsBlockingNetworkRequest()
            err = blocker.post(req, QByteArray(payload))
            # ErrorCode 0 = NoError. Ignoring the result made run()'s single
            # retry dead code: a failed batch returned True and was dropped.
            return int(err) == 0
        except Exception:
            return False  # nosec B110 - telemetry must never break the plugin

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
    try:
        evt = {
            "event": event,
            "properties": {**_base_properties(), **(properties or {})},
        }
    except Exception:  # nosec B110
        return
    with _lock:
        _batch.append(evt)
        should_flush = flush_now or event in FLUSH_NOW or len(_batch) >= _BATCH_MAX
    if should_flush:
        flush()


def flush() -> None:
    """Ship the queued batch. MAIN THREAD ONLY (no-ops off it). Lifecycle events
    ship pre-consent; everything else requires consent. Pre-auth lifecycle events
    park in _pending_pre_auth until the first authenticated flush."""
    if not _on_main_thread():
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


def _scrub_payload_value(value: str) -> str:
    """Strip path-like tokens, coordinate tuples, URLs and email addresses
    from telemetry strings.

    Applied defensively to any string leaving the machine. We already call
    _anonymize_paths upstream for filesystem paths, but this pass also catches
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
    try:
        from ..ui.error_report_dialog import _anonymize_paths
        value = _anonymize_paths(value)
    except Exception:
        pass  # nosec B110
    value = _URL_PATTERN.sub("<URL>", value or "")
    value = _EMAIL_PATTERN.sub("<EMAIL>", value)
    return _COORD_PATTERN.sub("<COORDS>", value)


# --- Public typed wrappers (delegate to track) ----------------------------
#
# Every event constant has a wrapper so call sites stay readable and prop names
# are centralized. flush_now is decided by FLUSH_NOW membership inside track().


# Lifecycle -----------------------------------------------------------------


_FIRST_OPEN_KEY = "AI_Segmentation/first_open_sent"


def track_plugin_first_open() -> None:
    """Fire exactly once, ever, the first time the dock is opened on this machine.

    Guarded by a persistent QSettings flag so an install -> first-open ->
    activation funnel has a clean entry marker. No consent needed (a lifecycle
    ping with no user content); parks pre-auth like plugin_opened."""
    try:
        settings = QSettings()
        if bool(settings.value(_FIRST_OPEN_KEY, False, type=bool)):
            return
        settings.setValue(_FIRST_OPEN_KEY, True)
    except Exception:  # nosec B110 - never break the open path on a settings error
        return
    track(ev.PLUGIN_FIRST_OPEN)


def track_plugin_opened() -> None:
    """Fire once per dock-open."""
    track(ev.PLUGIN_OPENED)


def track_plugin_activated() -> None:
    """Fire when the activation key is validated."""
    track(ev.PLUGIN_ACTIVATED)


def track_mode_switched(to_mode: str, had_unsaved_manual: bool = False,
                        auto_step: int | None = None) -> None:
    """Fire when the user changes the Manual / Automatic toggle."""
    track(ev.MODE_SWITCHED, {
        "to_mode": to_mode,
        "had_unsaved_manual": bool(had_unsaved_manual),
        "auto_step": auto_step,
    })


def track_install_started() -> None:
    track(ev.INSTALL_STARTED)


def track_install_completed(duration_ms: int | None = None,
                            python_minor: int | None = None,
                            retry_count: int | None = None) -> None:
    track(ev.INSTALL_COMPLETED, {
        "duration_ms": duration_ms,
        "python_minor": python_minor,
        "retry_count": retry_count,
    })


def track_install_failed(error_class: str, duration_ms: int | None = None,
                         python_minor: int | None = None,
                         retry_count: int | None = None) -> None:
    track(ev.INSTALL_FAILED, {
        "error_class": error_class,
        "duration_ms": duration_ms,
        "python_minor": python_minor,
        "retry_count": retry_count,
    })


def track_model_download_completed(model: str, duration_ms: int | None = None) -> None:
    """model is "sam1" or "sam2" ONLY (never a checkpoint URL or file name)."""
    track(ev.MODEL_DOWNLOAD_COMPLETED, {"model": model, "duration_ms": duration_ms})


# Automatic funnel ----------------------------------------------------------


def track_auto_start_clicked(layer_kind: str, has_credits_known: bool = False) -> None:
    track(ev.AUTO_START_CLICKED, {
        "layer_kind": layer_kind,
        "has_credits_known": bool(has_credits_known),
    })


def track_zone_drawn(vertices: int, area_km2: float, zone_kind: str = "polygon") -> None:
    track(ev.ZONE_DRAWN, {
        "vertices": vertices,
        "area_km2": round(area_km2, 1),
        "zone_kind": zone_kind,
    })


def track_auto_zone_too_large(area_km2: float) -> None:
    """Free-trial zone cap hit: the zone exceeds the free-tier limit. Only
    the rounded area is sent, never coordinates."""
    track(ev.AUTO_ZONE_TOO_LARGE, {"area_km2": round(area_km2, 1)})


def track_auto_prompt_committed(prompt: str, from_library: bool = False) -> None:
    """prompt is the validated 1-2 word object class (no PII by construction)."""
    track(ev.AUTO_PROMPT_COMMITTED, {"prompt": prompt, "from_library": bool(from_library)})


def track_tutorial_opened(source: str) -> None:
    """A tutorial/guide open. source is the touchpoint id (footer_tutorial,
    post_signin, zero_results); no PII by construction."""
    track(ev.TUTORIAL_OPENED, {"source": source})


def track_exemplar_added(count_after: int) -> None:
    track(ev.EXEMPLAR_ADDED, {"count_after": count_after})


def track_exemplar_removed(count_after: int) -> None:
    track(ev.EXEMPLAR_REMOVED, {"count_after": count_after})


def track_detail_changed(detail: int, tiles: int, source: str) -> None:
    """source: "auto_seeded" or "user"."""
    track(ev.DETAIL_CHANGED, {"detail": detail, "tiles": tiles, "source": source})


def track_auto_detect_started(run_id: str, tiles: int, zone_km2: float,
                              object_class: str, detail: int, exemplar_count: int,
                              est_credits: int, credits_before: int | None,
                              is_free_tier: bool) -> None:
    track(ev.AUTO_DETECT_STARTED, {
        "run_id": run_id,
        "tiles": tiles,
        "zone_km2": round(zone_km2, 2),
        "object_class": object_class,
        "detail": detail,
        "exemplar_count": exemplar_count,
        "est_credits": est_credits,
        "credits_before": credits_before,
        "is_free_tier": bool(is_free_tier),
    })


def track_auto_detect_completed(run_id: str, duration_ms: int, tiles_done: int,
                                tiles_failed: int, instances_found: int,
                                instances_visible_at_default: int, zero_at_default: bool,
                                p50_tile_ms: int | None = None,
                                p95_tile_ms: int | None = None,
                                stop_reason: str = "completed") -> None:
    track(ev.AUTO_DETECT_COMPLETED, {
        "run_id": run_id,
        "duration_ms": duration_ms,
        "tiles_done": tiles_done,
        "tiles_failed": tiles_failed,
        "instances_found": instances_found,
        "instances_visible_at_default": instances_visible_at_default,
        "zero_at_default": bool(zero_at_default),
        "p50_tile_ms": p50_tile_ms,
        "p95_tile_ms": p95_tile_ms,
        "stop_reason": stop_reason,
    })


def track_auto_detect_failed(run_id: str, error_class: str, tiles_done: int,
                             duration_ms: int | None = None) -> None:
    """error_class: NETWORK/AUTH/CREDITS_EXHAUSTED/SERVER/CANCELLED/TIMEOUT/UNKNOWN."""
    track(ev.AUTO_DETECT_FAILED, {
        "run_id": run_id,
        "error_class": error_class,
        "tiles_done": tiles_done,
        "duration_ms": duration_ms,
    })


def track_auto_detect_cancelled(run_id: str, tiles_done: int, tiles_total: int,
                                salvaged_to_review: bool) -> None:
    track(ev.AUTO_DETECT_CANCELLED, {
        "run_id": run_id,
        "tiles_done": tiles_done,
        "tiles_total": tiles_total,
        "salvaged_to_review": bool(salvaged_to_review),
    })


def track_credits_exhausted(run_id: str, tiles_done: int, tiles_total: int,
                            is_free_tier: bool) -> None:
    track(ev.CREDITS_EXHAUSTED, {
        "run_id": run_id,
        "tiles_done": tiles_done,
        "tiles_total": tiles_total,
        "is_free_tier": bool(is_free_tier),
    })


def track_auto_tiles_degraded(run_id: str, skipped_tiles: int, timeout_tiles: int,
                              blank_tiles: int = 0,
                              render_failed_tiles: int = 0) -> None:
    track(ev.AUTO_TILES_DEGRADED, {
        "run_id": run_id,
        "skipped_tiles": skipped_tiles,
        "timeout_tiles": timeout_tiles,
        # Pre-submit, uncharged drops: blank/nodata skips (credits saved) and
        # render/provider holes (possible coverage gap). Additive; older keys
        # unchanged.
        "blank_tiles": blank_tiles,
        "render_failed_tiles": render_failed_tiles,
    })


def track_auto_zero_result(run_id: str, tiles: int, object_class: str,
                           had_exemplar: bool) -> None:
    track(ev.AUTO_ZERO_RESULT, {
        "run_id": run_id,
        "tiles": tiles,
        "object_class": object_class,
        "had_exemplar": bool(had_exemplar),
    })


# Review / refine -----------------------------------------------------------


def track_review_opened(run_id: str, instances_found: int, visible_at_start: int,
                        start_confidence: int, auto_lowered: bool) -> None:
    track(ev.REVIEW_OPENED, {
        "run_id": run_id,
        "instances_found": instances_found,
        "visible_at_start": visible_at_start,
        "start_confidence": start_confidence,
        "auto_lowered": bool(auto_lowered),
    })


def track_review_confidence_final(run_id: str, final_pct: int, visible_count: int,
                                  moves: int) -> None:
    track(ev.REVIEW_CONFIDENCE_FINAL, {
        "run_id": run_id,
        "final_pct": final_pct,
        "visible_count": visible_count,
        "moves": moves,
    })


def track_review_display_mode(mode: str) -> None:
    track(ev.REVIEW_DISPLAY_MODE, {"mode": mode})


def track_review_shape_adjusted(control: str, value) -> None:
    track(ev.REVIEW_SHAPE_ADJUSTED, {"control": control, "value": value})


def track_refine_in_manual_entered(run_id: str, instances: int) -> None:
    track(ev.REFINE_IN_MANUAL_ENTERED, {"run_id": run_id, "instances": instances})


def track_refine_in_manual_back(run_id: str, validated_count: int,
                                duration_ms: int | None = None) -> None:
    track(ev.REFINE_IN_MANUAL_BACK, {
        "run_id": run_id,
        "validated_count": validated_count,
        "duration_ms": duration_ms,
    })


def track_auto_export_done(run_id: str, exported_count: int, visible_pct_of_found: int,
                           final_confidence: int, display_mode: str,
                           refined_in_manual: bool) -> None:
    track(ev.AUTO_EXPORT_DONE, {
        "run_id": run_id,
        "exported_count": exported_count,
        "visible_pct_of_found": visible_pct_of_found,
        "final_confidence": final_confidence,
        "display_mode": display_mode,
        "refined_in_manual": bool(refined_in_manual),
    })


def track_auto_retry_clicked(run_id: str, discarded_count: int, confirmed: bool) -> None:
    track(ev.AUTO_RETRY_CLICKED, {
        "run_id": run_id,
        "discarded_count": discarded_count,
        "confirmed": bool(confirmed),
    })


def track_auto_exit_clicked(from_step: int, autosaved_count: int) -> None:
    track(ev.AUTO_EXIT_CLICKED, {
        "from_step": from_step,
        "autosaved_count": autosaved_count,
    })


# Manual --------------------------------------------------------------------


_segmentation_run_sent_this_session = False


def track_segmentation_run(success: bool, duration_ms: int | None = None) -> None:
    """Fire when a manual segmentation run completes (or fails).

    Success runs are sampled 1-in-10 (power users click hundreds of times per
    session); the first run per session is always sent (sample_rate 1). Failures
    are ALWAYS sent unsampled (sample_rate 1) so the failure rate is real."""
    global _segmentation_run_sent_this_session
    import random

    if not success:
        # Failures are never sampled: the failure signal must be complete.
        track(ev.SEGMENTATION_RUN, {
            "success": False, "duration_ms": duration_ms, "sample_rate": 1,
        })
        return

    if _segmentation_run_sent_this_session:
        if random.random() >= 0.1:  # nosec B311 - sampling, not crypto
            return
        sample_rate = 10
    else:
        _segmentation_run_sent_this_session = True
        sample_rate = 1
    track(ev.SEGMENTATION_RUN, {
        "success": True, "duration_ms": duration_ms, "sample_rate": sample_rate,
    })


def track_manual_export_done(polygon_count: int, refine_used: bool) -> None:
    track(ev.MANUAL_EXPORT_DONE, {
        "polygon_count": polygon_count,
        "refine_used": bool(refine_used),
    })


def track_manual_session_summary(saves: int, undos: int,
                                 duration_ms: int | None = None) -> None:
    track(ev.MANUAL_SESSION_SUMMARY, {
        "saves": saves,
        "undos": undos,
        "duration_ms": duration_ms,
    })


# Monetization --------------------------------------------------------------


_upsell_viewed_this_session = False
_low_credit_banner_viewed_this_session = False


def track_pro_upsell_viewed(trigger: str = "free_exhausted") -> None:
    """Fire at most once per session when the upsell card first renders."""
    global _upsell_viewed_this_session
    if _upsell_viewed_this_session:
        return
    _upsell_viewed_this_session = True
    track(ev.PRO_UPSELL_VIEWED, {"trigger": trigger})


def track_pro_upsell_clicked(source: str = "upsell_card") -> None:
    """source: upsell_card / subscribe_pill / low_credit_banner / exhausted_status."""
    track(ev.PRO_UPSELL_CLICKED, {"source": source})


def track_free_taste_consumed(remaining: int) -> None:
    """remaining = free detections left after this one."""
    track(ev.FREE_TASTE_CONSUMED, {"remaining": remaining})


def track_low_credit_banner_viewed(remaining: int, total: int) -> None:
    """Fire at most once per session when the low-credit banner first shows."""
    global _low_credit_banner_viewed_this_session
    if _low_credit_banner_viewed_this_session:
        return
    _low_credit_banner_viewed_this_session = True
    track(ev.LOW_CREDIT_BANNER_VIEWED, {"remaining": remaining, "total": total})


def track_detect_blocked(reason: str) -> None:
    """reason: credits / zone_too_large / cost_over_balance."""
    track(ev.DETECT_BLOCKED, {"reason": reason})


# Errors --------------------------------------------------------------------


def track_plugin_error(
    stage: str,
    error_code: str,
    message: str,
    include_log_tail: bool = False,
    traceback_hash: str | None = None,
    module: str | None = None,
) -> None:
    """Fire when an error is shown to the user or an exception is caught.

    stage: install | download | activate | segment | export | other
    error_code: short machine-friendly id (e.g. "PIP_TIMEOUT", "RUNTIME_ERROR")
    message: first line of the error, truncated to 500 chars, path + coord scrubbed
    include_log_tail: OFF by default. When True, the last 20 anonymized log lines
        are capped to ~4KB and coordinate-scrubbed before being sent.
    traceback_hash: optional short sha of the normalized traceback (groups
        recurrences of the same crash). Additive; omitted when unknown.
    module: optional source module the exception was caught in. Additive.
    """
    props = {
        "stage": stage,
        "error_code": error_code,
        "message": _scrub_payload_value((message or "")[:500]),
    }
    if traceback_hash:
        props["traceback_hash"] = traceback_hash
    if module:
        props["module"] = module
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
    track(ev.PLUGIN_ERROR, props)


# --- Error capture (top-level slots + worker bodies) ----------------------
#
# The only error -> telemetry path used to be an explicit show_error_report()
# call, so an uncaught exception in a Qt slot or worker run() body produced NO
# telemetry at all. These helpers give a standard capture pattern: track a
# plugin_error with a stable English error_code, a traceback_hash (so the
# analytics backend groups recurrences of the same crash), and the source
# module; log a line; and
# show a dialog ONLY when an explicit user_message is passed. No sys.excepthook
# (that conflicts with QGIS and other plugins) - wrap the entry points instead.


def _short_traceback_hash(exc: BaseException) -> str:
    """A short, path-free fingerprint of an exception's traceback.

    Each frame contributes basename:lineno:function; the exception class name
    is appended. Filenames are reduced to their basename so the hash is stable
    across machines. Returns "" if anything goes wrong (never raises)."""
    import hashlib
    import os as _os
    import traceback as _tb
    try:
        parts = [
            "{}:{}:{}".format(_os.path.basename(fr.filename), fr.lineno, fr.name)
            for fr in _tb.extract_tb(exc.__traceback__)
        ]
        parts.append(exc.__class__.__name__)
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""


def report_exception(
    exc: BaseException,
    stage: str,
    module: str = "",
    user_message: str | None = None,
    parent=None,
) -> None:
    """Capture an unhandled exception: track it, log it, optionally show it.

    Always tracks a plugin_error (error_code = exception class name, plus a
    traceback_hash and module) and writes one QgsMessageLog line. Shows the
    error-report dialog ONLY when user_message is given AND we are on the main
    thread. Never raises: this runs on failure paths where a second failure
    must stay invisible."""
    error_code = ""
    tb_hash = ""
    first_line = ""
    try:
        error_code = exc.__class__.__name__
        tb_hash = _short_traceback_hash(exc)
        text = str(exc)
        first_line = text.splitlines()[0] if text else ""
    except Exception:  # nosec B110
        pass
    try:
        track_plugin_error(
            stage=stage,
            error_code=error_code or "Exception",
            message=first_line,
            traceback_hash=tb_hash,
            module=module or None,
        )
    except Exception:  # nosec B110
        pass
    try:
        from qgis.core import Qgis, QgsMessageLog
        QgsMessageLog.logMessage(
            "Unhandled {code} in {mod} ({stage}) [{h}]".format(
                code=error_code or "Exception", mod=module or "?",
                stage=stage, h=tb_hash or "-"),
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
    except Exception:  # nosec B110
        pass
    if user_message and _on_main_thread():
        try:
            from ..ui.error_report_dialog import ErrorReportDialog
            dialog = ErrorReportDialog(user_message, user_message, parent)
            dialog.exec()
        except Exception:  # nosec B110
            pass


def slot_guard(stage: str, user_message: str | None = None):
    """Decorator for a top-level Qt slot: catch any unhandled exception, report
    it (telemetry + log, and a dialog only when user_message is given), and
    swallow it so a stray crash never leaves QGIS's console handler as the only
    trace. Do NOT stack on slots that already surface their own errors."""
    def deco(fn):
        module = (fn.__module__ or "").rsplit(".", 1)[-1]

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            try:
                return fn(self, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - top-level slot boundary
                parent = None
                if user_message is not None:
                    try:
                        parent = self.iface.mainWindow()
                    except Exception:  # nosec B110
                        parent = None
                report_exception(
                    exc, stage=stage, module=module,
                    user_message=user_message, parent=parent,
                )
                return None
        return wrapper
    return deco


# NOTE: worker run() bodies report inline (see AutoDetectionWorker.run): they
# pair the report with worker-specific cleanup (power inhibit, error signal),
# which a generic context manager cannot know about. slot_guard above is the
# one shared boundary; an unused error_boundary context manager was removed.
