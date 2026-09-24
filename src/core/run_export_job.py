













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
                           source_layer) -> dict:






    rows = []
    for index, geom in enumerate(geoms):
        if geom is None or geom.isEmpty():
            continue
        score = scores[index] if scores is not None else None
        rows.append((
            str(det_ids[index] if det_ids is not None else index),
            bytes(geom.asWkb()),
            None if score is None else float(score),
        ))
    return {
        "rows": rows,
        "crs": crs,
        "object_class": (prompt_label or "").strip() or None,


        "measurer": make_area_measurer(crs),
        "plan": output_store.plan_run_table(
            prompt_label, source_layer, prompt_label or "detection"),
    }


def run_export_job(job: dict, progress: RunExportProgress | None = None) -> dict:







    from .polygon_exporter import count_overlapping_pairs

    out = {"written": None, "count": 0, "area_m2": 0.0,
           "overlapping_pairs": None, "failure": ""}
    try:
        temp_layer = QgsVectorLayer("MultiPolygon", "auto_export", "memory")
        if not temp_layer.isValid():
            out["failure"] = "no_working_layer"
            return out
        temp_layer.setCrs(job["crs"])
        provider = temp_layer.dataProvider()




        provider.addAttributes([
            QgsField("det_id", field_type_string()),
            QgsField("class", field_type_string()),
            measure_field("confidence", decimals=3),
            measure_field("area_m2"),
            measure_field("perimeter_m"),
        ])
        temp_layer.updateFields()
        fields = temp_layer.fields()
        measurer = job["measurer"]
        object_class = job.get("object_class")
        features = []
        written_geoms = []
        area_total = 0.0
        for det_id, wkb, score in job["rows"]:
            geom = QgsGeometry()
            geom.fromWkb(wkb)
            geom = to_multipolygon(repair_polygon(geom) or geom)
            if progress is not None:
                progress.advance()
            if geom is None or geom.isEmpty():
                continue
            feat = QgsFeature(fields)
            feat.setGeometry(geom)
            area_m2 = float(measurer.measureArea(geom))
            if area_m2 > 0:
                area_total += area_m2
            feat.setAttributes([
                det_id,
                object_class,
                round(float(score), 3) if score is not None else None,
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
        if not added:

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
