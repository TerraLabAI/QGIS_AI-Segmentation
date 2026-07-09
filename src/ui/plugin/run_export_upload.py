
























from __future__ import annotations

import json
import math

from qgis.core import QgsApplication, QgsTask

from ...core.qt_compat import geometry_op_succeeded, silent_task_flags
from .run_zone_clip import ZONE_WKT_CRS_AUTHID









_MAX_GEOJSON_BYTES = 25_000_000







_MAX_WKB_BYTES = 30_000_000


_COLLECTION_OVERHEAD = len('{"type":"FeatureCollection","features":[]}')


_inflight: list[QgsTask] = []


def cancel_inflight_uploads() -> None:







    tasks = list(_inflight)
    del _inflight[:]
    for task in tasks:
        try:
            task.cancel()
        except (RuntimeError, AttributeError):
            pass


class _RunExportUploadTask(QgsTask):


    def __init__(self, summary: dict, geometry_rows: list, precision: int, auth: dict,
                 max_geojson_bytes: int = _MAX_GEOJSON_BYTES):
        super().__init__("AI Segmentation run summary", silent_task_flags())
        self._summary = summary
        self._geometry_rows = geometry_rows
        self._precision = precision
        self._auth = auth


        self._max_geojson_bytes = max_geojson_bytes

    def run(self) -> bool:  # noqa: D102
        if self.isCanceled():
            return False
        try:
            body = encode_run_export_body(
                self._summary, self._geometry_rows, self._precision,
                self._max_geojson_bytes)
            self._geometry_rows = []
            from ...api.terralab_client import TerraLabClient

            TerraLabClient().post_run_export_body(body, self._auth)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        return True

    def finished(self, result: bool) -> None:  # noqa: D102
        try:
            _inflight.remove(self)
        except ValueError:
            pass


def json_precision(crs) -> int:


    try:
        if crs is not None and crs.isValid() and crs.isGeographic():
            return 8
    except (RuntimeError, AttributeError):
        pass
    return 3


def geometry_rows_for_upload(refined: list, refined_scores: list,
                             max_wkb_bytes: int = _MAX_WKB_BYTES) -> list[tuple]:







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


def _summary_without_geometry(summary: dict, dropped: int, stage: str) -> dict:








    marked = dict(summary)
    marked["geometry_omitted"] = True
    marked["geometry_omitted_count"] = int(dropped)
    marked["geometry_omitted_stage"] = str(stage)
    return marked


def _log_geometry_omitted(dropped: int, stage: str, limit: int) -> None:




    try:
        from qgis.core import Qgis, QgsMessageLog

        QgsMessageLog.logMessage(
            f"Run summary sent without its geometry: {dropped} objects are "
            f"past the {limit} byte {stage} ceiling, so this run cannot be "
            f"replayed from the summary.",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
    except Exception:  # noqa: BLE001
        pass  # nosec B110


def encode_run_export_body(summary: dict, geometry_rows: list, precision: int,
                           max_geojson_bytes: int = _MAX_GEOJSON_BYTES) -> bytes:









    if not geometry_rows:
        return json.dumps(summary, separators=(",", ":")).encode("utf-8")
    head = json.dumps(summary, separators=(",", ":"))
    if not head.endswith("}"):
        return head.encode("utf-8")
    from qgis.core import QgsGeometry

    def dropped_body() -> bytes:
        _log_geometry_omitted(len(geometry_rows), "GeoJSON", max_geojson_bytes)
        marked = _summary_without_geometry(
            summary, len(geometry_rows), "geojson")
        return json.dumps(marked, separators=(",", ":")).encode("utf-8")

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
        budget -= len(piece) + 1
        if budget < 0:
            return dropped_body()
        pieces.append(piece)
    if not pieces:
        return dropped_body()
    opening = head[:-1] + ',"geojson":{"type":"FeatureCollection","features":['
    return (opening + ",".join(pieces) + "]}}").encode("utf-8")







_MAX_ZONE_WKT_CHARS = 64_000


def zone_outline_for_upload(plugin) -> tuple[str, str] | None:

















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
        outline = QgsGeometry(polygon)
        if authid != ZONE_WKT_CRS_AUTHID:
            source = QgsCoordinateReferenceSystem(authid)
            target = QgsCoordinateReferenceSystem(ZONE_WKT_CRS_AUTHID)
            if not source.isValid() or not target.isValid():
                return None



            if not geometry_op_succeeded(outline.transform(QgsCoordinateTransform(
                    source, target,
                    QgsProject.instance().transformContext()))):
                return None
            if outline.isEmpty():
                return None



        from ...core.zone_antimeridian import fold_into_lonlat_range
        outline = fold_into_lonlat_range(outline)


        wkt = str(outline.asWkt(7) or "")
        from ...core.server_dials import dial_in_range
        max_chars = dial_in_range(
            "tuning.export.zone_wkt_max_chars", _MAX_ZONE_WKT_CHARS, 8_000, 500_000)
        if not wkt or len(wkt) > max_chars:
            return None
        return wkt, ZONE_WKT_CRS_AUTHID
    except Exception:  # noqa: BLE001
        return None


def _corrections_summary(plugin) -> dict | None:



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


def _run_tile_counts(plugin) -> tuple[int | None, int | None]:






    done = total = None
    try:
        result = getattr(plugin, "_last_auto_result", None) or {}
        value = result.get("tiles_processed")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            done = value
    except (RuntimeError, AttributeError):
        done = None
    try:
        ctx = getattr(plugin, "_auto_run_ctx", None) or {}
        value = ctx.get("total")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            total = value
    except (RuntimeError, AttributeError):
        total = None
    return done, total


def build_run_export_payload(
    plugin, review: dict, refined: list, refined_scores: list,
    export_path: str = "finish", confidence_applied: float | None = None,
) -> dict | None:












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


    try:
        zone_km2 = float(plugin._auto_zone_area_km2())
    except Exception:  # noqa: BLE001
        zone_km2 = 0.0
    if math.isfinite(zone_km2) and zone_km2 > 0:
        payload["zone_km2"] = round(zone_km2, 4)






    tiles_done, tiles_total = _run_tile_counts(plugin)
    if tiles_done is not None:
        payload["tiles_completed"] = tiles_done
    if tiles_total is not None:
        payload["tiles_total"] = tiles_total



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

        wkb_ceiling = max_wkb_bytes(_MAX_WKB_BYTES)
        rows = geometry_rows_for_upload(refined, refined_scores, wkb_ceiling)
        if refined and not rows:


            _log_geometry_omitted(len(refined), "WKB", wkb_ceiling)
            summary = _summary_without_geometry(summary, len(refined), "wkb")
        task = _RunExportUploadTask(
            summary, rows, json_precision(review.get("crs")), auth,
            max_geojson_bytes(_MAX_GEOJSON_BYTES))
        _inflight.append(task)
        QgsApplication.taskManager().addTask(task)
    except Exception:  # noqa: BLE001
        pass  # nosec B110
