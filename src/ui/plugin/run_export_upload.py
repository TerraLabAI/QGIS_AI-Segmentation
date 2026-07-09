"""Fire-and-forget upload of a finished Automatic run's FINAL output.

When the user hits Finish (or a headless run commits), the reviewed set they
actually kept - confidence-filtered, shape-refined, possibly hand-edited in the
Refine-in-Manual detour - exists only in their local project file. The per-tile
run record on the service holds the raw model output at the recall floor, so
without this step the user's real result (and the confidence threshold they
chose) is invisible to run quality analysis.

This module posts a compact run summary (chosen confidence cutoff, refine
settings, kept/found counts) plus the exported FeatureCollection when it is
small enough, over the same authenticated channel the run itself used. It is
strictly best-effort: built and queued AFTER the local export succeeded, runs
on a hidden background task, swallows every failure, and never blocks or fails
the user's export. Payload carries no paths, no layer names, no user identity
beyond the auth the run already used.
"""

from __future__ import annotations

import json

from qgis.core import QgsApplication, QgsTask

# Geometry ceiling for the uploaded FeatureCollection. Above this the summary
# row is still sent, just without the geometry (mirrors the server-side cap).
_MAX_GEOJSON_BYTES = 20_000_000

# Keep strong refs to in-flight tasks (QgsTaskManager holds only a weak ref).
_inflight: list[QgsTask] = []


def _silent_task_flags():
    flags = QgsTask.Flag(0)
    for name in ("Hidden", "Silent"):
        flag = getattr(QgsTask.Flag, name, None)
        if flag is not None:
            flags = flags | flag
    return flags


class _RunExportUploadTask(QgsTask):
    """POST one run-export payload. Failures swallowed end to end."""

    def __init__(self, payload: dict, auth: dict):
        super().__init__("AI Segmentation run summary", _silent_task_flags())
        self._payload = payload
        self._auth = auth

    def run(self) -> bool:  # noqa: D102 - QgsTask contract
        if self.isCanceled():
            return False
        try:
            from ...api.terralab_client import TerraLabClient

            TerraLabClient().post_run_export(self._payload, self._auth)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        return True

    def finished(self, result: bool) -> None:  # noqa: D102 - QgsTask contract
        try:
            _inflight.remove(self)
        except ValueError:
            pass


def _geometry_feature(geom, score, precision: int) -> dict | None:
    """One exported object as a GeoJSON Feature (score kept, nothing else)."""
    try:
        gj = json.loads(geom.asJson(precision))
    except Exception:  # noqa: BLE001
        return None
    if not gj:
        return None
    props: dict = {}
    if score is not None:
        try:
            props["score"] = round(float(score), 4)
        except (TypeError, ValueError):
            pass
    return {"type": "Feature", "geometry": gj, "properties": props}


def build_run_export_payload(
    plugin, review: dict, refined: list, refined_scores: list
) -> dict | None:
    """Assemble the run-export payload from the review state, BEFORE the export
    clears it. Returns None when there is no run to attach it to."""
    run_id = getattr(plugin, "_auto_run_id", None)
    if not run_id:
        return None

    crs = review.get("crs")
    crs_authid = None
    precision = 3  # projected units: mm-ish, keeps multi-thousand-object runs lean
    try:
        if crs is not None and crs.isValid():
            crs_authid = crs.authid() or None
            if crs.isGeographic():
                precision = 8
    except (RuntimeError, AttributeError):
        pass

    try:
        params = plugin._widget_review_params()
    except (RuntimeError, AttributeError):
        params = {}
    refine_params = {
        k: params.get(k)
        for k in (
            "simplify_px",
            "smooth",
            "expand_px",
            "fill_holes",
            "open_px",
            "ortho",
            "min_a",
            "max_a",
        )
        if k in params
    }

    try:
        default_confidence = float(plugin._review_start_confidence())
    except Exception:  # noqa: BLE001
        default_confidence = None

    payload: dict = {
        "run_id": run_id,
        "prompt": (review.get("prompt") or "").strip() or None,
        "final_confidence": float(getattr(plugin, "_auto_confidence", 0.0) or 0.0),
        "default_confidence": default_confidence,
        "refined_in_manual": bool(getattr(plugin, "_auto_refined_in_manual", False)),
        "exported_count": len(refined),
        "total_found": len(getattr(plugin, "_auto_objects", []) or []),
        "refine_params": refine_params or None,
        "crs_authid": crs_authid,
    }

    features = []
    for index, geom in enumerate(refined):
        score = refined_scores[index] if index < len(refined_scores) else None
        feature = _geometry_feature(geom, score, precision)
        if feature is not None:
            features.append(feature)
    if features:
        collection = {"type": "FeatureCollection", "features": features}
        try:
            if len(json.dumps(collection, separators=(",", ":"))) <= _MAX_GEOJSON_BYTES:
                payload["geojson"] = collection
        except (TypeError, ValueError):
            pass
    return payload


def queue_run_export_upload(
    plugin, review: dict, refined: list, refined_scores: list
) -> None:
    """Build the payload on the GUI thread (widget/state reads), then hand the
    network POST to a hidden background task. Never raises."""
    try:
        payload = build_run_export_payload(plugin, review, refined, refined_scores)
        if payload is None:
            return
        from ...core.activation_manager import get_auth_header

        auth = get_auth_header()
        if not auth:
            return
        task = _RunExportUploadTask(payload, auth)
        _inflight.append(task)
        QgsApplication.taskManager().addTask(task)
    except Exception:  # noqa: BLE001
        pass  # nosec B110
