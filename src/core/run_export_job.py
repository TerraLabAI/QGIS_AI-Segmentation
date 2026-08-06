













from __future__ import annotations

import threading

from qgis.core import QgsFeature, QgsFeatureSink, QgsField, QgsGeometry, QgsVectorLayer

from . import output_store
from .layer_conventions import (
    make_area_measurer,
    measure_field,
    repair_polygon,
    round_measure,
    to_multipolygon,
)
from .output_values import confidence_value
from .qt_compat import field_type_string



_EXPORT_FAST_INSERT = getattr(
    getattr(QgsFeatureSink, "Flag", QgsFeatureSink), "FastInsert", None)


class RunExportProgress:


    def __init__(self, total: int) -> None:
        self._lock = threading.Lock()
        self._done = 0
        self.total = max(0, int(total))

    def advance(self, count: int = 1) -> None:
        with self._lock:
            self._done += count

    def fraction(self) -> float:
        with self._lock:
            done = self._done
        return min(1.0, done / self.total) if self.total else 0.0


def prepare_run_export_job(geoms: list, crs, prompt_label: str, *,
                           scores: list | None, det_ids: list | None,
                           source_layer, row_classes: list | None = None,
                           class_colors: list | None = None,
                           land_cover: dict | None = None) -> dict:














    if scores is not None and len(scores) != len(geoms):
        scores = None
    if det_ids is not None and len(det_ids) != len(geoms):
        det_ids = None
    if row_classes is not None and len(row_classes) != len(geoms):
        row_classes = None
    if land_cover is not None and len(land_cover.get("class_codes") or []) != len(geoms):
        raise ValueError("Land cover classes must match the exported geometries")
    rows = []
    kept_classes = []
    kept_codes = []
    for index, geom in enumerate(geoms):
        if geom is None or geom.isEmpty():
            continue
        score = scores[index] if scores is not None else None
        rows.append((
            str(det_ids[index] if det_ids is not None else index),
            bytes(geom.asWkb()),
            confidence_value(score),
        ))
        if row_classes is not None:
            kept_classes.append(str(row_classes[index]))
        if land_cover is not None:
            kept_codes.append(int(land_cover["class_codes"][index]))
    lc = None
    if land_cover is not None:
        lc = {"class_codes": kept_codes, "run_id": str(land_cover.get("run_id") or ""),
              "min_area_m2": float(land_cover.get("min_area_m2") or 0.0)}
    return {
        "land_cover": lc,
        "rows": rows,
        "crs": crs,
        "object_class": None if row_classes is not None else (
            (prompt_label or "").strip() or None),
        "row_classes": kept_classes if row_classes is not None else None,
        "class_colors": list(class_colors) if class_colors else None,


        "measurer": make_area_measurer(crs),
        "plan": output_store.plan_run_table(
            prompt_label, source_layer, prompt_label or "detection"),
    }


def run_export_job(job: dict, progress: RunExportProgress | None = None) -> dict:







    from .polygon_exporter import count_overlapping_pairs

    out = {"written": None, "count": 0, "area_m2": 0.0,
           "overlapping_pairs": None, "failure": ""}
    if job.get("crs") is None or not job["crs"].isValid():
        out["failure"] = "invalid_crs"
        return out
    if job.get("land_cover"):
        return _run_land_cover_export(job, out, progress)
    try:
        temp_layer = QgsVectorLayer("MultiPolygon", "auto_export", "memory")
        if not temp_layer.isValid():
            out["failure"] = "no_working_layer"
            return out
        temp_layer.setCrs(job["crs"])
        provider = temp_layer.dataProvider()




        if not provider.addAttributes([
            QgsField("det_id", field_type_string()),
            QgsField("class", field_type_string()),
            measure_field("confidence", decimals=3),
            measure_field("area_m2"),
            measure_field("perimeter_m"),
        ]):
            out["failure"] = "no_working_layer"
            return out
        temp_layer.updateFields()
        fields = temp_layer.fields()
        measurer = job["measurer"]
        object_class = job.get("object_class")
        row_classes = job.get("row_classes")
        features = []
        written_geoms = []
        area_total = 0.0
        for row_index, (det_id, wkb, score) in enumerate(job["rows"]):
            geom = QgsGeometry()
            geom.fromWkb(wkb)
            geom = to_multipolygon(repair_polygon(geom) or geom)
            if progress is not None:
                progress.advance()
            if geom is None or geom.isEmpty():
                out["failure"] = "unwritable_geometry"
                return out
            feat = QgsFeature(fields)
            feat.setGeometry(geom)
            area_m2 = float(measurer.measureArea(geom))
            if area_m2 > 0:
                area_total += area_m2
            feat.setAttributes([
                det_id,
                row_classes[row_index] if row_classes is not None else object_class,
                confidence_value(score),
                round_measure(area_m2),
                round_measure(measurer.measurePerimeter(geom)),
            ])
            features.append(feat)
            written_geoms.append(geom)
        if not features:
            out["failure"] = "no_shapes"
            return out
        if _EXPORT_FAST_INSERT is not None:
            added = provider.addFeatures(features, _EXPORT_FAST_INSERT)
        else:
            added = provider.addFeatures(features)
        if isinstance(added, tuple):
            added = bool(added[0]) if added else False
        if not added or temp_layer.featureCount() != len(features):

            out["failure"] = "no_shapes"
            return out
        temp_layer.updateExtents()
        out["count"] = len(features)
        out["area_m2"] = area_total
        out["overlapping_pairs"] = count_overlapping_pairs(written_geoms)
        written = output_store.write_planned_run_table(temp_layer, job["plan"])
        if written is None:
            out["failure"] = "file_refused"
            return out
        out["written"] = written
        return out
    except Exception:  # noqa: BLE001
        out["failure"] = out["failure"] or "file_refused"
        out["written"] = None
        return out


def _run_land_cover_export(job: dict, out: dict,
                           progress: RunExportProgress | None) -> dict:



    from .qt_compat import field_type_int

    try:
        temp_layer = QgsVectorLayer("Polygon", "auto_export", "memory")
        if not temp_layer.isValid():
            out["failure"] = "no_working_layer"
            return out
        temp_layer.setCrs(job["crs"])
        provider = temp_layer.dataProvider()
        if not provider.addAttributes([
            QgsField("det_id", field_type_string()),
            QgsField("class", field_type_string()),
            QgsField("class_code", field_type_int()),
            measure_field("area_m2"),
            measure_field("perimeter_m"),
            QgsField("run_id", field_type_string()),
            measure_field("min_area_m2"),
        ]):
            out["failure"] = "no_working_layer"
            return out
        temp_layer.updateFields()
        fields = temp_layer.fields()
        measurer = job["measurer"]
        lc = job["land_cover"]
        classes = job.get("row_classes") or []
        features = []
        area_total = 0.0
        for index, (det_id, wkb, _score) in enumerate(job["rows"]):
            geom = QgsGeometry()
            geom.fromWkb(wkb)
            geom = repair_polygon(geom) or geom
            if progress is not None:
                progress.advance()
            if geom is None or geom.isEmpty():
                out["failure"] = "unwritable_geometry"
                return out
            from .land_cover import polygon_parts
            parts = polygon_parts(geom)
            if not parts:
                out["failure"] = "unwritable_geometry"
                return out
            for n, part in enumerate(parts):
                feat = QgsFeature(fields)
                feat.setGeometry(part)
                area_m2 = float(measurer.measureArea(part))
                area_total += max(0.0, area_m2)
                feat.setAttributes([
                    det_id if n == 0 else f"{det_id}-{n + 1}",
                    classes[index] if index < len(classes) else None,
                    int(lc["class_codes"][index]),
                    round_measure(area_m2),
                    round_measure(measurer.measurePerimeter(part)),
                    lc["run_id"],
                    round_measure(lc["min_area_m2"]),
                ])
                features.append(feat)
        if not features:
            out["failure"] = "no_shapes"
            return out
        added = provider.addFeatures(features)
        if isinstance(added, tuple):
            added = bool(added[0]) if added else False
        if not added or temp_layer.featureCount() != len(features):
            out["failure"] = "no_shapes"
            return out
        temp_layer.updateExtents()
        out["count"] = len(features)
        out["area_m2"] = area_total
        out["overlapping_pairs"] = 0
        written = output_store.write_planned_run_table(temp_layer, job["plan"])
        if written is None:
            out["failure"] = "file_refused"
            return out
        out["written"] = written
        return out
    except Exception:  # noqa: BLE001
        out["failure"] = out["failure"] or "file_refused"
        out["written"] = None
        return out
