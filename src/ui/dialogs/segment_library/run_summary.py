









from __future__ import annotations

import calendar
import time
from datetime import date

from qgis.PyQt.QtCore import QDate, QLocale, QTime

from ....core.i18n import tr
from ....core.qt_compat import resolve_qt_enum
from .common import _fmt_count, _iso_norm, _run_key
from .run_marks import run_mark


_HECTARE_CEILING_M2 = 1_000_000.0


_SHORT_FORMAT = resolve_qt_enum(QLocale, "FormatType", "ShortFormat")


def run_started_at(run: dict) -> str:

    return _iso_norm(run.get("started_at") or run.get("created_at"))


def run_title_text(run: dict) -> str:

    prompt = (run.get("prompt") or "").strip()
    if prompt:
        return prompt
    return tr("Drawn examples") if run.get("has_exemplars") else tr("Older detection")


def zone_wkt_area_m2(run: dict) -> float:













    from ...plugin.run_zone_clip import ZONE_WKT_CRS_AUTHID, zone_polygon_from_wkt

    geom = zone_polygon_from_wkt(run.get("zone_wkt"))
    if geom is None:
        return 0.0
    authid = str(run.get("zone_crs_authid") or "").strip() or ZONE_WKT_CRS_AUTHID
    try:
        from qgis.core import (
            QgsCoordinateReferenceSystem,
            QgsCoordinateTransformContext,
        )

        from ....core.layer_conventions import make_area_measurer

        crs = QgsCoordinateReferenceSystem(authid)
        if not crs.isValid():
            return 0.0
        measurer = make_area_measurer(
            crs, QgsCoordinateTransformContext(), "EPSG:7030")
        return max(0.0, float(measurer.measureArea(geom)))
    except Exception:  # noqa: BLE001
        return 0.0


def tile_bbox_area_m2(run: dict) -> float:






    box = run.get("bbox_wgs84")
    if not isinstance(box, dict):
        return 0.0
    try:
        xmin = float(box["xmin"])
        ymin = float(box["ymin"])
        xmax = float(box["xmax"])
        ymax = float(box["ymax"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    if xmax <= xmin or ymax <= ymin or not -90.0 < (ymin + ymax) / 2.0 < 90.0:
        return 0.0
    try:
        from qgis.core import (
            QgsCoordinateReferenceSystem,
            QgsCoordinateTransformContext,
            QgsGeometry,
            QgsRectangle,
        )

        from ....core.layer_conventions import make_area_measurer

        measurer = make_area_measurer(
            QgsCoordinateReferenceSystem("EPSG:4326"),
            QgsCoordinateTransformContext(), "EPSG:7030")
        rect = QgsGeometry.fromRect(QgsRectangle(xmin, ymin, xmax, ymax))
        return max(0.0, float(measurer.measureArea(rect)))
    except Exception:  # noqa: BLE001
        return 0.0


def run_zone_area_m2(run: dict) -> float:


















    outline = zone_wkt_area_m2(run)
    box = tile_bbox_area_m2(run)
    if outline > 0 and box > 0:
        return min(outline, box)
    return outline if outline > 0 else box


def run_zone_area_text(run: dict) -> str:

    area = run_zone_area_m2(run)
    if area <= 0:
        return ""
    if area < _HECTARE_CEILING_M2:
        hectares = area / 10_000.0
        value = f"{hectares:.1f}" if hectares < 10 else str(int(round(hectares)))
        return tr("{n} ha").format(n=value)
    km2 = area / _HECTARE_CEILING_M2
    value = f"{km2:.1f}" if km2 < 10 else str(int(round(km2)))


    return tr("{n} km²").format(n=value)


def run_objects_text(run: dict) -> str:

    objects = int(run.get("objects") or 0)
    if objects == 1:
        return tr("1 object")
    return tr("{n} objects").format(n=_fmt_count(objects))


def run_detections_text(run: dict) -> str:





    tiles = int(run.get("tiles") or 0)
    credits = int(run.get("credits") or 0)
    if tiles <= 0:
        return ""
    text = (tr("1 cloud detection") if tiles == 1
            else tr("{n} cloud detections").format(n=_fmt_count(tiles)))
    if credits != tiles:
        text += "  ·  " + (tr("1 charged") if credits == 1
                           else tr("{n} charged").format(n=_fmt_count(credits)))
    return text


def run_status_text(run: dict) -> str:






    key = _run_key(run)
    if key:
        mark = run_mark(key)
        if mark == "exported":
            return tr("Exported")
        if mark == "restored":
            return tr("Restored")
    if int(run.get("objects") or 0) <= 0:
        return tr("Found nothing")
    return ""


def _run_local_time(run: dict):






    stamp = run_started_at(run)
    if not stamp:
        return None
    try:
        return time.localtime(
            calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")))
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def run_time_text(run: dict) -> str:





    local = _run_local_time(run)
    if local is None:
        return ""
    try:
        return QLocale().toString(
            QTime(local.tm_hour, local.tm_min), _SHORT_FORMAT)
    except (TypeError, ValueError):
        return time.strftime("%H:%M", local)


def run_day_key(run: dict) -> str:





    local = _run_local_time(run)
    if local is None:
        return ""
    return time.strftime("%Y-%m-%d", local)


def run_day_label(day_key: str) -> str:

    if not day_key:
        return tr("Undated")
    try:
        parts = [int(p) for p in day_key.split("-")]
        then = date(parts[0], parts[1], parts[2])
        days = (date.today() - then).days
    except (ValueError, IndexError, TypeError):
        return day_key
    if days <= 0:
        return tr("Today")
    if days == 1:
        return tr("Yesterday")
    try:
        return QLocale().toString(QDate(then.year, then.month, then.day))
    except (TypeError, ValueError):
        return day_key


def group_runs_by_day(runs: list[dict]) -> list[tuple[str, list[dict]]]:






    groups: list[tuple[str, list[dict]]] = []
    index: dict[str, list[dict]] = {}
    for run in runs:
        key = run_day_key(run)
        bucket = index.get(key)
        if bucket is None:
            bucket = []
            index[key] = bucket
            groups.append((key, bucket))
        bucket.append(run)
    return [(run_day_label(key), bucket) for key, bucket in groups]
