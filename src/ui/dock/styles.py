"""Brand colors and QSS constants for the AI Segmentation dock
(design-system values shared with AI Edit), plus tiny style helpers."""
from __future__ import annotations

from ...core.i18n import tr

# Collapsed height for refine panel title (just enough to show the arrow + label)
_REFINE_COLLAPSED_HEIGHT = 25

# Automatic (Pro) detection confidence + review shape-refine defaults (Simplify /
# Clean / Round corners / Fill holes / Expand) live in core/review_defaults.py,
# imported at the top as the private _AUTO_* aliases so this dock and the plugin
# controller share one source. Confidence range 0.05-0.95 keeps both extremes
# reachable without hitting a degenerate 0/1.

# Post-run review confidence: only the SLIDER is quantized (5% steps) so a mouse
# drag stops on round values instead of firing the heavy client-side re-filter +
# re-merge on every single percentage it passes through. The spinbox stays free
# (1% precision) so the user can still dial an exact cutoff; slider and spinbox
# may therefore diverge, and the snap lives ONLY in the slider drag handler.
# The spinbox also gets a LOWER floor (1%) than the slider (5%): the 5%-step
# slider has no clean sub-5% stop, but the arrows let the user reach right down
# to 1% to surface the very faintest detections. Below 5% the slider just parks
# at its own 5% floor while the spinbox holds the true cutoff (the source of
# truth for the re-filter).
_REVIEW_CONF_STEP = 5
_REVIEW_CONF_MIN = 5
_REVIEW_CONF_MAX = 95
_REVIEW_CONF_SPIN_MIN = 1


def _snap_review_conf(value: int) -> int:
    """Round a review-confidence percent to the nearest slider step, clamped to range."""
    snapped = int(round(value / _REVIEW_CONF_STEP)) * _REVIEW_CONF_STEP
    return max(_REVIEW_CONF_MIN, min(_REVIEW_CONF_MAX, snapped))


# Brand colors (Material Design 2 - shared with AI Edit, same values).
# Primary CTA buttons keep the material green; it reads as THE action color.
# Every other green accent uses the TerraLab leaf green below.
BTN_GREEN = "#43a047"
BTN_GREEN_HOVER = "#2e7d32"
BTN_GREEN_DISABLED = "#c8e6c9"

# Brand accent green = the QGIS green (terralab-website --qgis-green). Lime
# fills use BRAND_GREEN; green text on light backgrounds uses BRAND_GREEN_TEXT.
BRAND_GREEN = "#8bac27"
BRAND_GREEN_TEXT = "#4d7c0f"
BRAND_BLUE = "#1e88e5"
BRAND_BLUE_HOVER = "#1976d2"
BRAND_RED = "#d32f2f"
BRAND_RED_HOVER = "#b71c1c"
BRAND_GRAY = "#757575"
BRAND_GRAY_HOVER = "#616161"
BRAND_DISABLED = "#b0bec5"
DISABLED_TEXT = "#666666"
ERROR_TEXT = "#ef5350"
SUCCESS_TEXT = "#66bb6a"

# Shared slider look for the Automatic detail + review-confidence sliders. The
# default QSlider handle is a tiny sliver that is fiddly to grab on the narrow
# dock; this gives a thicker groove, a filled (sub-page) track in brand blue and
# a large round handle that is easy to drag precisely.
_SLIDER_QSS = (
    "QSlider:horizontal { min-height: 22px; }"
    "QSlider::groove:horizontal { height: 6px; border-radius: 3px;"
    " background: rgba(128,128,128,0.30); }"
    f"QSlider::sub-page:horizontal {{ height: 6px; border-radius: 3px; background: {BRAND_BLUE}; }}"
    f"QSlider::handle:horizontal {{ background: {BRAND_BLUE}; border: 2px solid palette(base);"
    " width: 16px; height: 16px; margin: -7px 0; border-radius: 10px; }"
    f"QSlider::handle:horizontal:hover {{ background: {BRAND_BLUE_HOVER}; }}"
)

# Subtle bordered "card" used to group the Automatic step-2 sections (the
# prompt and the detail/tiles controls) so each reads as a distinct block with
# a clear hierarchy. The #objectName selector keeps the fill and border on the
# card itself and off its child widgets; a bare ``QWidget {}`` rule would
# cascade into every label and combo inside.
_CARD_QSS = (
    "QWidget#{name} {{ background-color: rgba(128, 128, 128, 0.06);"
    " border: 1px solid rgba(128, 128, 128, 0.22); border-radius: 6px; }}"
)

# Manual-session instruction label: the framed card look (default) and the
# compact muted-hint look used during a Refine-in-Manual handoff, where the
# blue banner above already frames the context and a boxed multi-line card
# read as enormous. One place for both so build.py and state.py never diverge.
_INSTRUCTIONS_CARD_QSS = (
    "QLabel {"
    " background-color: rgba(128, 128, 128, 0.12);"
    " border: 1px solid rgba(128, 128, 128, 0.25);"
    " border-radius: 4px;"
    " padding: 8px;"
    " font-size: 12px;"
    " color: palette(text);"
    "}"
)
_INSTRUCTIONS_HINT_QSS = (
    "QLabel {"
    " background: transparent;"
    " border: none;"
    " padding: 2px 0px;"
    " font-size: 11px;"
    " color: rgba(128,128,128,0.95);"
    "}"
)


def _auto_step_header(num: int, title: str, optional: bool = False) -> str:
    """Rich-text 'N - Title' header for an Automatic step-2 card: a brand-blue
    step number, a bold title, and a muted '- optional' suffix when the step
    can be skipped. The three cards then read top to bottom as an ordered
    checklist (describe, then optionally show an example, then set detail),
    which is the 'do one thing at a time' shape the page is built around."""
    opt = ""
    if optional:
        opt = (' <span style="color: rgba(128,128,128,0.85);'
               ' font-weight: normal;">- {}</span>').format(tr("optional"))
    return (
        f'<span style="color: {BRAND_BLUE}; font-weight: 700;">{num}</span>'
        f'&nbsp;&middot;&nbsp;<span style="font-weight: 700;">{title}</span>{opt}'
    )


# Design-system QSS constants, identical to AI Edit (dock_widget.py).
# border: none kills the native frame on dark themes; black text on the
# mid-tone fills keeps AA contrast on both light and dark QGIS themes.
_BTN_GREEN = (
    f"QPushButton {{ background-color: {BTN_GREEN}; color: #000000;"
    f" padding: 8px 16px; border: none; border-radius: 4px; }}"
    f"QPushButton:hover {{ background-color: {BTN_GREEN_HOVER}; color: #000000; }}"
    f"QPushButton:disabled {{ background-color: {BTN_GREEN_DISABLED};"
    f" color: {DISABLED_TEXT}; }}"
)

_BTN_GREEN_AUTH = (
    f"QPushButton {{ background-color: {BTN_GREEN}; color: #000000;"
    f" border: none; border-radius: 4px; }}"
    f"QPushButton:hover {{ background-color: {BTN_GREEN_HOVER}; }}"
    f"QPushButton:disabled {{ background-color: {BRAND_DISABLED};"
    f" color: {DISABLED_TEXT}; }}"
)

_BTN_BLUE = (
    f"QPushButton {{ background-color: {BRAND_BLUE}; color: #000000;"
    f" padding: 6px 12px; border: none; border-radius: 4px; }}"
    f"QPushButton:hover {{ background-color: {BRAND_BLUE_HOVER}; color: #000000; }}"
    f"QPushButton:disabled {{ background-color: {BRAND_DISABLED};"
    f" color: {DISABLED_TEXT}; }}"
)

_BTN_BLUE_AUTH = (
    f"QPushButton {{ background-color: {BRAND_BLUE}; color: #000000;"
    f" border: none; border-radius: 4px; }}"
    f"QPushButton:hover {{ background-color: {BRAND_BLUE_HOVER}; }}"
    f"QPushButton:disabled {{ background-color: {BRAND_DISABLED}; }}"
)

# Primary blue CTA: same heft as _BTN_GREEN (8px 16px) but in the Automatic
# brand blue. Used for the Automatic-mode "Start" so it echoes the blue tab
# underline, the way the green Start echoes the green Interactive underline.
_BTN_BLUE_PRIMARY = (
    f"QPushButton {{ background-color: {BRAND_BLUE}; color: #000000;"
    f" padding: 8px 16px; border: none; border-radius: 4px; }}"
    f"QPushButton:hover {{ background-color: {BRAND_BLUE_HOVER}; color: #000000; }}"
    f"QPushButton:disabled {{ background-color: {BRAND_DISABLED};"
    f" color: {DISABLED_TEXT}; }}"
)

# Ghost / outline button (mirrors AI Edit's _BTN_GHOST): transparent fill with
# a faint border, for a secondary action that sits beside a filled primary
# (e.g. Exit next to Detect).
_BTN_GHOST = (
    "QPushButton { background-color: transparent; color: palette(text);"
    " padding: 8px 16px; border-radius: 4px;"
    " border: 1px solid rgba(128, 128, 128, 0.35); }"
    "QPushButton:hover { background-color: rgba(128, 128, 128, 0.15);"
    " border: 1px solid rgba(128, 128, 128, 0.5); }"
    f"QPushButton:disabled {{ background-color: rgba(128, 128, 128, 0.08);"
    f" border: 1px solid rgba(128, 128, 128, 0.15); color: {DISABLED_TEXT}; }}"
)

_BTN_GRAY = (
    f"QPushButton {{ background-color: {BRAND_GRAY}; color: #000000;"
    f" padding: 4px 8px; border: none; border-radius: 4px; }}"
    f"QPushButton:hover {{ background-color: {BRAND_GRAY_HOVER}; color: #000000; }}"
    f"QPushButton:disabled {{ background-color: {BRAND_DISABLED}; color: {DISABLED_TEXT}; }}"
)

_BTN_RED = (
    f"QPushButton {{ background-color: rgba(211,47,47,0.12); color: {BRAND_RED};"
    f" padding: 6px 12px; border: none; border-radius: 4px; }}"
    f"QPushButton:hover {{ background-color: rgba(211,47,47,0.22); }}"
)

_BTN_EXPORT_READY = (
    f"QPushButton {{ background-color: {BTN_GREEN}; color: #000000;"
    f" padding: 6px 12px; border: none; border-radius: 4px; }}"
    f"QPushButton:hover {{ background-color: {BTN_GREEN_HOVER}; color: #000000; }}"
)

_BTN_EXPORT_DISABLED = (
    f"QPushButton {{ background-color: {BRAND_DISABLED}; color: {DISABLED_TEXT};"
    f" padding: 6px 12px; border: none; border-radius: 4px; }}"
)

# Compact filled buttons for the browser-handoff waiting state. Both carry a
# soft tint (never transparent): neutral for "open again", red for "cancel".
_BTN_PAIR_NEUTRAL = (
    "QPushButton { background-color: rgba(128,128,128,0.16); color: palette(text);"
    " border: none; border-radius: 4px; }"
    "QPushButton:hover { background-color: rgba(128,128,128,0.28); }"
)
_BTN_PAIR_CANCEL = (
    f"QPushButton {{ background-color: rgba(211,47,47,0.12); color: {BRAND_RED};"
    f" border: none; border-radius: 4px; }}"
    f"QPushButton:hover {{ background-color: rgba(211,47,47,0.22); }}"
)

# Footer icon buttons (gear / question mark) - slim toolbuttons that mirror
# AI Edit. Hover state is driven by a dynamic `hover` property rather than
# Qt's :hover pseudo, because with InstantPopup menus Qt fails to fire a
# Leave event once the menu closes, so the button stays tinted until the
# next real mouse move. ``_FooterIconButton.set_hovered(False)`` resets it.
# The TerraLab leaf-green ``[active]`` tint marks "this menu is open".
_FOOTER_ICON_BTN_STYLE = (
    "QToolButton { background: transparent; border: none; padding: 6px 10px;"
    " font-size: 22px; font-weight: 600;"
    " color: palette(text); border-radius: 4px; }"
    'QToolButton[hover="true"] { background: rgba(128,128,128,0.15); }'
    'QToolButton[active="true"] { background: rgba(139, 172, 39, 0.55); }'
    'QToolButton[active="true"][hover="true"] { background: rgba(139, 172, 39, 0.75); }'
    "QToolButton::menu-indicator { image: none; width: 0; }"
)

# Help (question mark) hovers green - the leaf tint invites the user toward
# Tutorial / Report a problem instead of reading as a neutral icon.
_HELP_ICON_BTN_STYLE = (
    "QToolButton { background: transparent; border: none; padding: 6px 10px;"
    " font-size: 22px; font-weight: 600;"
    " color: palette(text); border-radius: 4px; }"
    'QToolButton[hover="true"] { background: rgba(139, 172, 39, 0.35); }'
    'QToolButton[active="true"] { background: rgba(139, 172, 39, 0.55); }'
    'QToolButton[active="true"][hover="true"] { background: rgba(139, 172, 39, 0.75); }'
    "QToolButton::menu-indicator { image: none; width: 0; }"
)

_FOOTER_MENU_STYLE = (
    "QMenu { background: palette(base); border: 1px solid rgba(128,128,128,0.35);"
    " border-radius: 6px; padding: 4px; }"
    "QMenu::item { background: transparent; padding: 6px 14px; border-radius: 4px;"
    " color: palette(text); }"
    "QMenu::item:selected { background: rgba(128,128,128,0.18); }"
)

# Footer cross-promo CTA - same flat/transparent + hover-tint look as the gear
# and help buttons, but sized for a label (11px) instead of a 22px glyph so the
# text reads as a small button rather than dwarfing the icons beside it.
_FOOTER_CTA_BTN_STYLE = (
    "QToolButton { background: transparent; border: none; padding: 6px 10px;"
    " font-size: 11px; font-weight: 600;"
    " color: palette(text); border-radius: 4px; }"
    'QToolButton[hover="true"] { background: rgba(128,128,128,0.15); }'
)
