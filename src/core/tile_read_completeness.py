






















from __future__ import annotations

import sys

import numpy as np
from qgis.core import Qgis


def _argb_types() -> tuple:



    data_types = getattr(Qgis, "DataType", None)
    if data_types is None:
        return ()
    found = (getattr(data_types, "ARGB32", None),
             getattr(data_types, "ARGB32_Premultiplied", None))
    return tuple(t for t in found if t is not None)


def read_carries_alpha(data_type) -> bool:






    argb_types = _argb_types()
    return bool(argb_types) and data_type in argb_types


def read_alpha_plane(data: bytes, width: int, height: int, data_type):






    if not read_carries_alpha(data_type):
        return None
    if width <= 0 or height <= 0:
        return None
    if not data or len(data) != width * height * 4:
        return None
    alpha_offset = 3 if sys.byteorder == "little" else 0
    return np.frombuffer(data, dtype=np.uint8)[alpha_offset::4]


def online_read_is_complete(data: bytes, width: int, height: int,
                            data_type) -> bool:







    alpha = read_alpha_plane(data, width, height, data_type)
    if alpha is None:
        return False
    return bool(alpha.min() > 0)


def holes_are_the_same(previous: bytes, current: bytes, width: int,
                       height: int, data_type) -> bool:









    before = read_alpha_plane(previous, width, height, data_type)
    after = read_alpha_plane(current, width, height, data_type)
    if before is None or after is None:
        return False
    return bool(np.array_equal(before > 0, after > 0))
