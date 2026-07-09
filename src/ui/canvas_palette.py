"""Single source of truth for every canvas color the plugin draws.

The color language, shared by every TerraLab plugin that draws on canvas:
BLUE = AI output, editable/pending. AMBER = the one object being edited now.
GREEN = kept/validated this session. RED = committed to a layer.
CHROME_* = zone/grid/badge scaffolding, a distinct blue tone from data,
never filling large areas. Width is a second cue (active=3) so color is
never the only signal (colorblind safety).

Layering: the QGIS renderer STRINGS ("r,g,b,a") that describe a saved or
under-review layer live in core/layer_conventions.py (core must not import
ui). This module (ui) imports and re-exports them so ui code has one place
to reach for every canvas color, and holds every QColor constant itself.
"""
from __future__ import annotations

from qgis.PyQt.QtGui import QColor

from ..core.layer_conventions import (
    _COMMITTED_RED as _COMMITTED_RED,
    _REVIEW_FILL as _REVIEW_FILL,
    _REVIEW_OUTLINE as _REVIEW_OUTLINE,
)

# Data states. Two-color language on the canvas: blue = editable (pending, not
# yet saved), green = validated. The object currently open for editing reuses
# pending-blue and leans on a bolder outline for emphasis, so there is no third
# hue (an earlier amber "active" state read as an unrelated random colour).
PENDING_FILL = QColor(0, 120, 255, 100)
PENDING_STROKE = QColor(0, 80, 200, 255)
PENDING_WIDTH = 2

KEPT_FILL = QColor(0, 200, 100, 60)
KEPT_STROKE = QColor(0, 200, 100, 180)
KEPT_WIDTH = 2

# Renderer strings (QGIS symbol layer properties want "r,g,b,a" strings).
# The three under-review / committed strings are re-exported from core so the
# meaning lives in one place; the outline-display-mode red is a ui-only review
# concern and is defined here.
REVIEW_FILL_STR = _REVIEW_FILL
REVIEW_OUTLINE_STR = _REVIEW_OUTLINE
COMMITTED_OUTLINE_STR = _COMMITTED_RED
OUTLINE_MODE_STROKE_STR = "227,26,28,255"

# Chrome (zone / grid / badge / draw tool)
CHROME_BLUE = QColor(66, 133, 244)
ZONE_FILL = QColor(66, 133, 244, 18)
ZONE_STROKE = QColor(66, 133, 244, 200)
ZONE_DRAW_FILL = QColor(66, 133, 244, 55)
ZONE_DRAW_LINE = QColor(66, 133, 244, 235)
GRID_LINE = QColor(40, 99, 196, 210)
BADGE_FILL = QColor(66, 133, 244)
BADGE_FILL_DISABLED = QColor(66, 133, 244, 115)
BADGE_FILL_HOVER = QColor(211, 47, 47)          # BRAND_RED on hover = deletion affordance
BADGE_X = QColor(255, 255, 255)
CLOSE_DOT_OK = QColor(34, 197, 94)
CLOSE_DOT_WARN = QColor(255, 213, 0)

# Setup-time helpers
EXEMPLAR_FILL = QColor(67, 160, 71, 60)
EXEMPLAR_STROKE = QColor(67, 160, 71)
EXCLUDE_FILL = QColor(229, 57, 53, 60)
EXCLUDE_STROKE = QColor(229, 57, 53)

# Click markers (manual maptool)
MARKER_POSITIVE = QColor(0, 200, 0)
MARKER_NEGATIVE = QColor(220, 0, 0)
