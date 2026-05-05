
from __future__ import annotations


def object_overlaps_zone(geom, zone, engine=None) -> bool:







    if zone is None:
        return True
    if geom is None or geom.isEmpty():
        return False
    try:
        if not geom.boundingBox().intersects(zone.boundingBox()):
            return False
        if engine is not None:
            try:
                if engine.contains(geom.constGet()):
                    return True
                return (engine.intersects(geom.constGet())
                        and not engine.touches(geom.constGet()))
            except Exception:  # noqa: BLE001  # nosec B110
                pass
        cut = geom.intersection(zone)
        return cut is not None and not cut.isEmpty() and cut.area() > 0.0
    except Exception:  # noqa: BLE001
        return True


def rows_overlapping_zone(rows: list, zone, engine=None) -> list:

    if zone is None:
        return rows
    return [row for row in rows if object_overlaps_zone(row[1], zone, engine)]
