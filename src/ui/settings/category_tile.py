







from __future__ import annotations

import zlib

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import QHBoxLayout, QLabel, QWidget

from ..dock.font_scale import scale_px_length
from ..dock.styles import (
    INK_2,
    RADIUS_ROW,
    TINT_ON,
    category_ink,
    category_tint,
)



NAV_GLYPH_CATEGORIES = {
    "person": "green",
    "gem": "amber",
    "package": "teal",
    "play": "coral",
    "terminal": "leaf",
    "puzzle": "violet",
    "chat_bubble": "sky",
    "warning": "coral",
}


TONE_CATEGORIES = {"info": "sky", "warning": "amber", "success": "green", "error": "coral"}


_AVATAR_CATEGORIES = ("green", "leaf", "amber", "teal", "violet", "sky")

TILE_PX = 30
TILE_GLYPH_PX = 17
TILE_SMALL_PX = 28
TILE_SMALL_GLYPH_PX = 16


def category_icon_tile(glyph: str, category: str | None, parent: QWidget | None = None,
                       tile_px: int = TILE_PX, glyph_px: int = TILE_GLYPH_PX) -> QLabel:





    size = scale_px_length(tile_px)
    tile = QLabel(parent)
    tile.setObjectName("categoryTile")
    tile.setFixedSize(size, size)
    tile.setAlignment(Qt.AlignmentFlag.AlignCenter)


    ground = category_tint(category) if category else TINT_ON
    ink = category_ink(category) if category else INK_2

    tile.setStyleSheet(f"background: {ground}; border: none; border-radius: {RADIUS_ROW}px;")
    try:
        from ..icons import pixmap_for

        tile.setPixmap(pixmap_for(tile, glyph, scale_px_length(glyph_px), QColor(ink)))
    except (ImportError, RuntimeError, AttributeError):
        pass  # nosec B110
    tile.setAccessibleName("")
    return tile


def tile_beside(tile: QLabel, widget: QWidget, spacing: int = 12) -> QHBoxLayout:

    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(spacing)
    row.addWidget(tile, 0, Qt.AlignmentFlag.AlignVCenter)
    row.addWidget(widget, 1)
    return row


def avatar_category(email: str) -> str:

    text = (email or "").strip().lower()
    if not text or text == "-":
        return "green"
    return _AVATAR_CATEGORIES[zlib.crc32(text.encode("utf-8")) % len(_AVATAR_CATEGORIES)]


__all__ = [
    "NAV_GLYPH_CATEGORIES",
    "TONE_CATEGORIES",
    "avatar_category",
    "category_icon_tile",
    "tile_beside",
]
