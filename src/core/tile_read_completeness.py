"""Has one read of an online layer got all its pixels, or is a tile missing?

A crop from an online layer is read from the layer's provider, and the read can
come back while some of its tiles are still on the way. The fetch used to
answer that by reading twice half a second apart and demanding a match, which
costs that half second on every click even when nothing was missing.

On the tile path the question has an exact answer in the read itself. The
provider starts the image fully transparent and paints each tile into it as it
lands, so a pixel whose tile never arrived is still fully transparent, and one
whose tile did arrive is opaque. No transparent pixel means no missing tile.

A transparent pixel does NOT mean the opposite. An ocean tile, the edge of the
service's coverage and a transparent overlay are all transparent for good, so
demanding a full alpha plane sends those reads down the whole retry ladder and
buys nothing. The second answer here is the hole PATTERN across two reads: a
tile on its way fills in between them and its hole moves, while a hole that
survives both is the layer having nothing there.

Only that pixel layout gets a verdict here. Any other one (a numeric band from
a coverage service, an unknown type) answers False, so the caller keeps the
slower re-read it has always used.
"""
from __future__ import annotations

import numpy as np
from qgis.core import Qgis


def _argb_types() -> tuple:
    """The two pixel layouts that carry an alpha byte, read off Qgis at call
    time. Import time is too early: a build (or a test double) may not carry
    the enum, and a missing enum must not stop the module from importing."""
    data_types = getattr(Qgis, "DataType", None)
    if data_types is None:
        return ()
    found = (getattr(data_types, "ARGB32", None),
             getattr(data_types, "ARGB32_Premultiplied", None))
    return tuple(t for t in found if t is not None)


def read_carries_alpha(data_type) -> bool:
    """Whether a read of this type can be asked whether its tiles all arrived.

    True only for the two layouts that carry an alpha byte. Everything else
    (a numeric band from a coverage service, an unknown type) has no such
    evidence in the pixels, and the caller settles it by re-reading instead.
    """
    argb_types = _argb_types()
    return bool(argb_types) and data_type in argb_types


def read_alpha_plane(data: bytes, width: int, height: int, data_type):
    """The alpha byte of every pixel of one read, or None when the read carries
    no alpha or does not match its own pixel size.

    A strided view, so a 1024x1024 crop costs about a millisecond and copies
    nothing.
    """
    if not read_carries_alpha(data_type):
        return None
    if width <= 0 or height <= 0:
        return None
    if not data or len(data) != width * height * 4:
        return None
    return np.frombuffer(data, dtype=np.uint8)[3::4]


def online_read_is_complete(data: bytes, width: int, height: int,
                            data_type) -> bool:
    """True when every pixel of this provider read carries an arrived tile.

    Args:
        data: the raw bytes of the read (``block.data()``).
        width, height: the block's pixel size.
        data_type: the block's Qgis.DataType.
    """
    alpha = read_alpha_plane(data, width, height, data_type)
    if alpha is None:
        return False
    return bool(alpha.min() > 0)


def holes_are_the_same(previous: bytes, current: bytes, width: int,
                       height: int, data_type) -> bool:
    """True when two reads of one extent leave the SAME pixels transparent.

    Answers the case ``online_read_is_complete`` cannot: a read that has holes
    on purpose. Compares which pixels carry a tile, not the tile colours, so an
    overlay that redraws itself between the two reads still counts as settled.

    False whenever either read carries no alpha, so a caller without this
    evidence keeps its own re-read.
    """
    before = read_alpha_plane(previous, width, height, data_type)
    after = read_alpha_plane(current, width, height, data_type)
    if before is None or after is None:
        return False
    return bool(np.array_equal(before > 0, after > 0))
