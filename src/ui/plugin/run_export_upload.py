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
import math

from qgis.core import QgsApplication, QgsTask

from ...core.qt_compat import geometry_op_succeeded, silent_task_flags
from .run_zone_clip import ZONE_WKT_CRS_AUTHID

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


def cancel_inflight_uploads() -> None:
    """Cancel every queued upload and drop the strong refs. Call from unload.

    A reload re-imports this module, so `_inflight` becomes a fresh list while
    the tasks started by the previous instance keep resolving `finished()`
    against the old module dict. Nothing then reaps them, and each one still
    holds a run's WKB. Never raises: unload must finish whatever happens here.
    """
    tasks = list(_inflight)
    del _inflight[:]
    for task in tasks:
        try:
            task.cancel()
        except (RuntimeError, AttributeError):
            pass


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


# Ceiling on the zone outline carried in the run summary, in WKT characters.
# The run row is read back by the library to point a re-run at the same
# ground, and a traced coastline is tens of thousands of characters on a row
# that is otherwise a few hundred bytes. Past this the summary travels without
# it and the reader falls back to the tile union, as it always did.
_MAX_ZONE_WKT_CHARS = 64_000


def zone_outline_for_upload(plugin) -> tuple[str, str] | None:
    """(zone WKT in WGS84, "EPSG:4326") of the polygon THIS run looked at, or
    None.

    The tiles a run billed cover its bounding box, not the shape the user
    drew, so a run row that keeps only the tiles cannot say where the run was
    confined. This carries the shape itself, so a restore can clip to it and a
    re-run can point at it.

    Always WGS84, never the run's own CRS: the run's predict call already
    writes the same column in WGS84, and one column cannot hold two
    conventions. A reader has no way to tell which one a row uses, and every
    row written before the CRS field existed answers nothing at all.

    None when the run had no drawn polygon (a rectangle or a headless zone),
    when the outline cannot be written or moved to WGS84, or when it is past
    the ceiling above. Best-effort like the rest of this module: never raises.
    """
    try:
        from qgis.core import (
            QgsCoordinateReferenceSystem,
            QgsCoordinateTransform,
            QgsGeometry,
            QgsProject,
        )

        polygon = getattr(plugin, "_auto_clip_polygon", None)
        if polygon is None or polygon.isEmpty():
            return None
        authid = str(getattr(plugin, "_auto_crs_authid", "") or "")
        if not authid:
            return None
        outline = QgsGeometry(polygon)  # transform() edits its geometry
        if authid != ZONE_WKT_CRS_AUTHID:
            source = QgsCoordinateReferenceSystem(authid)
            target = QgsCoordinateReferenceSystem(ZONE_WKT_CRS_AUTHID)
            if not source.isValid() or not target.isValid():
                return None
            # Anything but success leaves half-moved coordinates, and a
            # zone on the wrong ground clips a later restore down to nothing,
            # so the row travels without an outline instead.
            if not geometry_op_succeeded(outline.transform(QgsCoordinateTransform(
                    source, target,
                    QgsProject.instance().transformContext()))):
                return None
            if outline.isEmpty():
                return None
        # Seven decimals of a degree is about a centimetre on the ground, and
        # full precision doubles a traced coastline for nothing.
        wkt = str(outline.asWkt(7) or "")
        if not wkt or len(wkt) > _MAX_ZONE_WKT_CHARS:
            return None
        return wkt, ZONE_WKT_CRS_AUTHID
    except Exception:  # noqa: BLE001 -- an unreadable zone is simply absent
        return None


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


def _geoms_fit_declared_crs(refined: list, crs) -> bool:
    """False only when the exported set provably cannot be in ``crs``.

    The check works on bounds, so the whole set is asked at once through its
    union box. True on anything it cannot decide, including an empty set, so
    the caller keeps its behaviour and only loses figures that are wrong.
    """
    try:
        from qgis.core import QgsGeometry

        from ...core.zone_crs_check import zone_fits_declared_crs

        box = None
        for geom in refined:
            if geom is None or geom.isEmpty():
                continue
            rect = geom.boundingBox()
            if box is None:
                box = rect
            else:
                box.combineExtentWith(rect)
        if box is None or box.isEmpty():
            return True
        return zone_fits_declared_crs(QgsGeometry.fromRect(box), crs)
    except Exception:  # noqa: BLE001
        return True


def _exported_area_m2(refined: list, crs, plugin=None) -> float | None:
    """Geodesic area (m2) of the whole exported set, or None when it cannot be
    measured. One measurer for the batch, GUI thread only (the measurer reads
    the project). Best-effort like everything in this module: never raises.

    The export that just ran measured every geometry it wrote, so its total is
    read back off ``plugin`` when it is there: measuring the same set again is
    a second geodesic pass over every object, on the click the user is waiting
    on. Falls through to the measurement when no total was recorded.

    Same guard as the zone surface next door (_auto_zone_area_km2): a set whose
    coordinates cannot be in the CRS the run declared measures degrees as
    metres, and unknown beats wrong, because the caller omits an unknown
    surface and stores a wrong one.
    """
    if not _geoms_fit_declared_crs(refined, crs):
        return None
    if plugin is not None:
        try:
            total = float(getattr(plugin, "_auto_exported_area_m2", 0.0) or 0.0)
        except (TypeError, ValueError):
            total = 0.0
        if math.isfinite(total) and total > 0:
            return total
    if not refined:
        return None
    try:
        if crs is None or not crs.isValid():
            # Without a real CRS the figure would be in unknown units.
            return None
        from ...core.layer_conventions import make_area_measurer

        measurer = make_area_measurer(crs)
        total = 0.0
        for geom in refined:
            if geom is None or geom.isEmpty():
                continue
            area = float(measurer.measureArea(geom))
            if math.isfinite(area) and area > 0:
                total += area
        if math.isfinite(total) and total > 0:
            return total
    except Exception:  # noqa: BLE001
        pass  # nosec B110
    return None


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

    payload = {
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
    # Ground surface, both optional and omitted when unknown: the run zone's
    # geodesic area in km2 and the geodesic area of the exported set in m2.
    try:
        zone_km2 = float(plugin._auto_zone_area_km2())
    except Exception:  # noqa: BLE001
        zone_km2 = 0.0
    if math.isfinite(zone_km2) and zone_km2 > 0:
        payload["zone_km2"] = round(zone_km2, 4)
    # The drawn outline, so a restore can confine the run to it and a re-run
    # can point at it instead of at its bounding box. Both optional, both
    # omitted when the run had no polygon.
    outline = zone_outline_for_upload(plugin)
    if outline is not None:
        payload["zone_wkt"], payload["zone_crs_authid"] = outline
    exported_area = _exported_area_m2(refined, crs, plugin)
    if exported_area is not None:
        payload["exported_area_m2"] = round(exported_area, 1)
    return payload


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
