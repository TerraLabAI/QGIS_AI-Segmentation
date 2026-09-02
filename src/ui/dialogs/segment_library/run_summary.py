"""What one past run says about itself, in words a card can print.

Every fact here is read from the history payload the server already sends
(started_at, prompt, objects, tiles, credits, bbox_wgs84, has_exemplars) or
from the local marks store. Nothing depends on a field the server may not
carry: a missing field prints nothing rather than a zero.

Kept apart from the widgets so the wording is one place, shared by the run
cards and the run detail popup.
"""
from __future__ import annotations

import calendar
import time
from datetime import date

from qgis.PyQt.QtCore import QDate, QLocale, QTime

from ....core.i18n import tr
from ....core.qt_compat import resolve_qt_enum
from .common import _fmt_count, _iso_norm, _run_key
from .run_marks import run_mark

# Below this a zone reads better in hectares: "0.04 km2" says less than "4 ha".
_HECTARE_CEILING_M2 = 1_000_000.0
# Resolved through qt_compat so the scoped/flat enum split stays out of the
# Qt6 static check.
_SHORT_FORMAT = resolve_qt_enum(QLocale, "FormatType", "ShortFormat")


def run_started_at(run: dict) -> str:
    """The run's own timestamp, normalized to the UTC shape the rest parses."""
    return _iso_norm(run.get("started_at") or run.get("created_at"))


def run_title_text(run: dict) -> str:
    """The card's headline: the prompt, or what stands in for one."""
    prompt = (run.get("prompt") or "").strip()
    if prompt:
        return prompt
    return tr("Drawn examples") if run.get("has_exemplars") else tr("Older detection")


def zone_wkt_area_m2(run: dict) -> float:
    """Geodesic ground area (m2) of the outline the run was billed on, 0.0 when
    the row carries no readable outline.

    The row's ``zone_wkt`` is the polygon the user drew, in WGS84, and it is
    the shape the account is charged for. The bounding box next door is the
    union of the tile grid, which covers the zone's rectangle and overhangs it
    on every edge: on a drawn shape it reads about twice the ground the run
    cost. Measuring the outline is the only way this card and the panel that
    quoted the run can print the same number.

    One measurer per call and no project read (the ellipsoid is named here),
    so this is safe wherever the card is built.
    """
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
    except Exception:  # noqa: BLE001 -- an unmeasurable outline is simply absent
        return 0.0


def tile_bbox_area_m2(run: dict) -> float:
    """Geodesic ground area (m2) of the box the run's completed tiles cover.

    The payload's ``bbox_wgs84`` is the union of the tile grid, so this is the
    ground the run actually worked over, box and all. It answers 0.0 for a row
    that carries no box.
    """
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
    except Exception:  # noqa: BLE001 -- an unmeasurable box is simply absent
        return 0.0


def run_zone_area_m2(run: dict) -> float:
    """Ground area of the run's zone in square metres, 0.0 when unknown.

    Two readings, and the smaller wins.

    The drawn outline is the ground the run was billed on, and on a run that
    finished it is the number to print. The tile box is the ground the run
    actually worked over, and on a run that finished it is bigger, because the
    grid covers the zone's rectangle and overhangs it on every edge: about
    2.4x the drawn zone at the median. So the outline wins there.

    On a run the user cancelled, or one that stopped on an error, the tiles
    covered only part of the zone, and then the box is the smaller of the two
    and wins instead. That is the same rule the account applies to the bill, so
    the card and the bill say the same thing about a stopped run rather than
    the card quoting ground nobody looked at.

    0.0 when the row carries neither, and the caller prints nothing.
    """
    outline = zone_wkt_area_m2(run)
    box = tile_bbox_area_m2(run)
    if outline > 0 and box > 0:
        return min(outline, box)
    return outline if outline > 0 else box


def run_zone_area_text(run: dict) -> str:
    """The zone's size as one short phrase, or "" when the run kept no box."""
    area = run_zone_area_m2(run)
    if area <= 0:
        return ""
    if area < _HECTARE_CEILING_M2:
        hectares = area / 10_000.0
        value = f"{hectares:.1f}" if hectares < 10 else str(int(round(hectares)))
        return tr("{n} ha").format(n=value)
    km2 = area / _HECTARE_CEILING_M2
    value = f"{km2:.1f}" if km2 < 10 else str(int(round(km2)))
    return tr("{n} km2").format(n=value)


def run_objects_text(run: dict) -> str:
    """How many shapes the run produced."""
    objects = int(run.get("objects") or 0)
    if objects == 1:
        return tr("1 object")
    return tr("{n} objects").format(n=_fmt_count(objects))


def run_detections_text(run: dict) -> str:
    """What the run cost, in the unit the account is billed in.

    One cloud detection is one charged tile, so printing both numbers says the
    same thing twice; the charged count only appears when it differs.
    """
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
    """What has already been done with this run, or "" when nothing has.

    The server keeps no such flag, so this reads the local marks store: what
    THIS computer restored or exported. A run with no objects says so instead,
    because that is the one outcome a user comes back to the list to check.
    """
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
    """The run's start as a local time struct, or None when unparsable.

    The stored stamp is UTC, so it is read with timegm and handed back to
    localtime: mktime would read it as local and shift every run by the
    machine's own offset.
    """
    stamp = run_started_at(run)
    if not stamp:
        return None
    try:
        return time.localtime(
            calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")))
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def run_time_text(run: dict) -> str:
    """Clock time of the run on the user's own day, or "" when unparsable.

    Short format on purpose: the locale's long one adds seconds and a zone
    name ("21:18:00 CEST"), which is three facts where the card needs one.
    """
    local = _run_local_time(run)
    if local is None:
        return ""
    try:
        return QLocale().toString(
            QTime(local.tm_hour, local.tm_min), _SHORT_FORMAT)
    except (TypeError, ValueError):
        return time.strftime("%H:%M", local)


def run_day_key(run: dict) -> str:
    """The local calendar day a run belongs to, as 'YYYY-MM-DD'.

    Local, not UTC: a run started at 01:00 UTC belongs to the evening the user
    remembers, and grouping it under the next day makes the list unreadable.
    """
    local = _run_local_time(run)
    if local is None:
        return ""
    return time.strftime("%Y-%m-%d", local)


def run_day_label(day_key: str) -> str:
    """The heading over one day's runs: today, yesterday, or the date."""
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
    """(day label, runs) newest day first, keeping the server's order inside.

    The list already arrives newest first, so the days come out in order
    without a second sort, and a run the server could not date lands in its
    own trailing group rather than silently joining today.
    """
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
