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

The GUI thread hands the task WKB bytes and never GeoJSON. Turning one polygon
into GeoJSON costs a few hundred microseconds, so a run with tens of thousands
of them held the interface for seconds on the Finish click, and again the
moment the review opened. A binary dump is a fraction of that, it is plain
immutable bytes so the task owns its copy and shares nothing with the map, and
the text is built on the task thread, which the geometry calls leave free.
"""

from __future__ import annotations

import json

from qgis.core import QgsApplication, QgsTask

from ...core.qt_compat import silent_task_flags

# Geometry ceiling for the uploaded FeatureCollection. Above this the summary
# row is still sent, just without the geometry. It mirrors the cap on the
# receiving side, so it is a server dial (network.max_geojson_bytes) read in
# queue_run_export_upload; this is the client fallback.
_MAX_GEOJSON_BYTES = 20_000_000

# Ceiling on the WKB the GUI thread collects. GeoJSON text of the same
# coordinates is never much smaller than the binary, so a set past this is
# certain to blow the GeoJSON cap: dropping it on an exact byte count is
# both cheaper and surer than extrapolating from a sample, and it spares the
# task thread a dense run's conversion that was always going to be discarded.
# Server dial (network.max_wkb_bytes) too, so the pair moves together.
_MAX_WKB_BYTES = 24_000_000

# What the FeatureCollection wrapper adds around the joined features.
_COLLECTION_OVERHEAD = len('{"type":"FeatureCollection","features":[]}')

# Keep strong refs to in-flight tasks (QgsTaskManager holds only a weak ref).
_inflight: list[QgsTask] = []


class _RunExportUploadTask(QgsTask):
    """Encode one run-export body and POST it. Failures swallowed end to end."""

    def __init__(self, summary: dict, geometry_rows: list, precision: int, auth: dict,
                 max_geojson_bytes: int = _MAX_GEOJSON_BYTES):
        super().__init__("AI Segmentation run summary", silent_task_flags())
        self._summary = summary
        self._geometry_rows = geometry_rows
        self._precision = precision
        self._auth = auth
        # Resolved on the GUI thread by the caller, so the task thread never
        # touches the policy cache.
        self._max_geojson_bytes = max_geojson_bytes

    def run(self) -> bool:  # noqa: D102 - QgsTask contract
        if self.isCanceled():
            return False
        try:
            body = encode_run_export_body(
                self._summary, self._geometry_rows, self._precision,
                self._max_geojson_bytes)
            self._geometry_rows = []  # the WKB is spent: drop it before the POST
            from ...api.terralab_client import TerraLabClient

            TerraLabClient().post_run_export_body(body, self._auth)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        return True

    def finished(self, result: bool) -> None:  # noqa: D102 - QgsTask contract
        try:
            _inflight.remove(self)
        except ValueError:
            pass


def json_precision(crs) -> int:
    """Decimal places for the uploaded coordinates: mm-ish in projected units,
    8 places on a geographic CRS, where a degree is 111 km."""
    try:
        if crs is not None and crs.isValid() and crs.isGeographic():
            return 8
    except (RuntimeError, AttributeError):
        pass
    return 3


def geometry_rows_for_upload(refined: list, refined_scores: list,
                             max_wkb_bytes: int = _MAX_WKB_BYTES) -> list[tuple]:
    """(WKB, score) per exported object, for the task thread to serialize.

    This is the ONLY geometry work the GUI thread does for the upload, so it
    stays a straight binary dump: no GeoJSON, no validity pass. Returns an
    empty list once the set passes ``max_wkb_bytes``, which is the summary-only
    case the GeoJSON cap already produced, decided here on an exact byte count.
    """
    rows: list[tuple] = []
    budget = max_wkb_bytes
    for index, geom in enumerate(refined):
        try:
            wkb = bytes(geom.asWkb())
        except (RuntimeError, AttributeError, TypeError, ValueError):
            continue
        if not wkb:
            continue
        budget -= len(wkb)
        if budget < 0:
            return []
        score = refined_scores[index] if index < len(refined_scores) else None
        rows.append((wkb, score))
    return rows


def _feature_json(geom, score, precision: int) -> str | None:
    """One exported object as GeoJSON Feature TEXT (score kept, nothing else).

    Text, not a dict: the collection is spliced straight into the request body,
    so a dense run is never parsed into Python objects and dumped back out
    twice over.
    """
    try:
        geometry = geom.asJson(precision)
    except Exception:  # noqa: BLE001
        return None
    if not geometry or geometry == "null":
        return None
    props = ""
    if score is not None:
        try:
            props = '"score":' + json.dumps(round(float(score), 4))
        except (TypeError, ValueError):
            props = ""
    return '{"type":"Feature","geometry":' + geometry + ',"properties":{' + props + "}}"


def encode_run_export_body(summary: dict, geometry_rows: list, precision: int,
                           max_geojson_bytes: int = _MAX_GEOJSON_BYTES) -> bytes:
    """The POST body: the summary, plus the FeatureCollection when it fits.

    Task-thread only. The cap is measured as the text grows, never by
    serializing the finished collection to read its length, so a run past it
    stops converting instead of building a payload nobody will read.
    """
    head = json.dumps(summary, separators=(",", ":"))
    if not geometry_rows or not head.endswith("}"):
        return head.encode("utf-8")
    from qgis.core import QgsGeometry

    pieces: list[str] = []
    budget = max_geojson_bytes - _COLLECTION_OVERHEAD
    for wkb, score in geometry_rows:
        geom = QgsGeometry()
        try:
            geom.fromWkb(wkb)
        except (RuntimeError, TypeError, ValueError):
            continue
        piece = _feature_json(geom, score, precision)
        if piece is None:
            continue
        budget -= len(piece) + 1  # the joining comma
        if budget < 0:
            return head.encode("utf-8")
        pieces.append(piece)
    if not pieces:
        return head.encode("utf-8")
    opening = head[:-1] + ',"geojson":{"type":"FeatureCollection","features":['
    return (opening + ",".join(pieces) + "]}}").encode("utf-8")


def _corrections_summary(plugin) -> dict | None:
    """A compact summary of the review's edit journal: {"count": n, "kinds":
    {kind: n, ...}}, or None when there is no journal or it holds no edits.
    Defensive by design (the whole module swallows failures): never raises."""
    journal = getattr(plugin, "_auto_correct_journal", None)
    if journal is None:
        return None
    try:
        entries = list(journal)
    except Exception:  # noqa: BLE001
        return None
    if not entries:
        return None
    kinds: dict[str, int] = {}
    for entry in entries:
        kind = getattr(entry, "kind", None)
        if isinstance(kind, str):
            kinds[kind] = kinds.get(kind, 0) + 1
    return {"count": len(entries), "kinds": kinds}


def build_run_export_payload(
    plugin, review: dict, refined: list, refined_scores: list,
    export_path: str = "finish", confidence_applied: float | None = None,
) -> dict | None:
    """Assemble the run-export SUMMARY from the review state, BEFORE the export
    clears it. Returns None when there is no run to attach it to. The exported
    geometry is NOT part of it: that travels as WKB and becomes text on the
    task thread (see encode_run_export_body).

    ``export_path`` is which leg committed the run (finish, exit_save,
    autosave) and ``confidence_applied`` is the confidence that actually
    filtered the exported set, which is 0.0 whenever a safety-net leg fell
    back to the full found set. Both exist because final_confidence alone
    cannot tell a curated export from a rescue: it records where the slider
    was left, not the gate that ran.
    """
    run_id = getattr(plugin, "_auto_run_id", None)
    if not run_id:
        return None

    crs = review.get("crs")
    crs_authid = None
    try:
        if crs is not None and crs.isValid():
            crs_authid = crs.authid() or None
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
            "points_pct",
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

    # Per-run provenance: which client build and policy revision produced the
    # run. Best-effort; a failure leaves the field None (never blocks export).
    try:
        plugin_version = plugin._read_plugin_version()
    except (RuntimeError, AttributeError):
        plugin_version = None
    try:
        from ...core import detection_policy

        policy_rev = detection_policy.policy_rev()
    except Exception:  # noqa: BLE001
        policy_rev = None

    return {
        "run_id": run_id,
        "prompt": (review.get("prompt") or "").strip() or None,
        "final_confidence": float(getattr(plugin, "_auto_confidence", 0.0) or 0.0),
        "default_confidence": default_confidence,
        "refined_in_manual": bool(getattr(plugin, "_auto_refined_in_manual", False)),
        "export_path": export_path,
        "confidence_applied": (
            float(confidence_applied) if confidence_applied is not None else None),
        "exported_count": len(refined),
        "total_found": len(getattr(plugin, "_auto_objects", []) or []),
        "refine_params": refine_params or None,
        "crs_authid": crs_authid,
        "plugin_version": plugin_version,
        "policy_rev": policy_rev,
        "corrections": _corrections_summary(plugin),
    }


def queue_run_export_upload(
    plugin, review: dict, refined: list, refined_scores: list,
    export_path: str = "finish", confidence_applied: float | None = None,
) -> None:
    """Read the run state on the GUI thread, then hand the encoding AND the
    network POST to a hidden background task. Never raises.

    Order matters for the click the user is waiting on: the auth lookup comes
    before the geometry, so a signed-out user pays nothing at all.

    Both size caps are resolved here, once, and carried into the task: they
    mirror a cap on the receiving side, and one upload must use one pair.
    """
    try:
        summary = build_run_export_payload(
            plugin, review, refined, refined_scores,
            export_path=export_path, confidence_applied=confidence_applied)
        if summary is None:
            return
        from ...core.activation_manager import get_auth_header

        auth = get_auth_header()
        if not auth:
            return
        from ...core.detection_policy import max_geojson_bytes, max_wkb_bytes

        rows = geometry_rows_for_upload(
            refined, refined_scores, max_wkb_bytes(_MAX_WKB_BYTES))
        task = _RunExportUploadTask(
            summary, rows, json_precision(review.get("crs")), auth,
            max_geojson_bytes(_MAX_GEOJSON_BYTES))
        _inflight.append(task)
        QgsApplication.taskManager().addTask(task)
    except Exception:  # noqa: BLE001
        pass  # nosec B110
