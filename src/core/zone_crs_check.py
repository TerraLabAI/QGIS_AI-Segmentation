















from __future__ import annotations




_LON_LIMIT = 181.0
_LAT_LIMIT = 91.0


def zone_fits_declared_crs(geom, crs) -> bool:







    try:
        if geom is None or crs is None or not crs.isValid():
            return True
        box = geom.boundingBox()
        if box is None or box.isEmpty():
            return True
        in_lonlat_range = (
            -_LON_LIMIT <= box.xMinimum() <= _LON_LIMIT
            and -_LON_LIMIT <= box.xMaximum() <= _LON_LIMIT
            and -_LAT_LIMIT <= box.yMinimum() <= _LAT_LIMIT
            and -_LAT_LIMIT <= box.yMaximum() <= _LAT_LIMIT
        )



        return in_lonlat_range if crs.isGeographic() else not in_lonlat_range
    except Exception:  # noqa: BLE001
        return True
