"""Brand colors and QSS constants for the AI Segmentation dock
(design-system values shared with AI Edit), plus tiny style helpers."""
from __future__ import annotations


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
    # Disabled = fully grey (no brand blue anywhere): the gated Detail slider
    # must read as "not usable yet" at a glance, not as a broken live control.
    "QSlider::sub-page:horizontal:disabled { background: rgba(128,128,128,0.30); }"
    "QSlider::handle:horizontal:disabled { background: rgba(128,128,128,0.45);"
    " border: 2px solid palette(base); }"
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

# Standard inner margins for a card built from _CARD_QSS, so sibling cards
# align to the pixel: (left, top, right, bottom).
_CARD_MARGINS = (10, 8, 10, 10)

# NOTE: no colored edge ornaments. A left "spine" stripe and title ticks were
# tried and rejected (Yvann, 2026-07-11: colored slots on text edges read as
# generic AI-tool design). Cards carry hierarchy through content, never
# through a colored border.

# ---------------------------------------------------------------------------
# Semantic message taxonomy: one hue carries ONE meaning, shared across both
# TerraLab plugins. Every message lives in a styled container (never naked
# text); text stays palette(text) so both QGIS themes read.
#   neutral  = instruction, how to do something (never coloured)
#   info     = guidance in the brand blue (THE only blue)
#   armed    = a map tool is armed and waiting for a draw/click (denser blue)
#   success  = done/measured outcome, in the lime accent (the CTA green
#              means "advance", it never announces success)
#   warning  = caution, translucent amber (readable on dark and light)
#   error    = failure; *_TRANSIENT is the denser variant for toasts that
#              replace content instead of sitting beside it
#   premium  = paid capability, blue family with a distinct treatment
#              (star prefix + underlined action link), never inline in
#              other guidance text
# Fill/border pairs, per kind.
_MSG_TINTS = {
    "neutral": ("rgba(128, 128, 128, 0.12)", "rgba(128, 128, 128, 0.25)"),
    "info": ("rgba(30, 136, 229, 0.08)", "rgba(30, 136, 229, 0.22)"),
    "armed": ("rgba(30, 136, 229, 0.12)", "rgba(30, 136, 229, 0.40)"),
    "success": ("rgba(139, 172, 39, 0.14)", "rgba(139, 172, 39, 0.45)"),
    "warning": ("rgba(245, 166, 35, 0.12)", "rgba(245, 166, 35, 0.45)"),
    "error": ("rgba(229, 72, 77, 0.14)", "rgba(229, 72, 77, 0.45)"),
    "error_transient": ("rgba(229, 72, 77, 0.25)", "rgba(229, 72, 77, 0.60)"),
    "premium": ("rgba(30, 136, 229, 0.12)", "rgba(30, 136, 229, 0.40)"),
}

# Star prefix for premium/upsell copy (D9b treatment).
_PREMIUM_STAR = "★"

# Message-kind glyph prefixes. Statuses (armed/success/warning/error) carry
# quiet monochrome TEXT glyphs, plain characters tinted by the label's own
# text color (U+FE0E forces text presentation on macOS); mass-emoji reads as
# cheap. The ONE exception is info/tips: the lightbulb emoji, warmer than a
# flat i-icon for guidance (Yvann's call 2026-07-11 evening).
_MSG_GLYPHS = {
    "neutral": "",
    "info": "💡",
    "armed": "✎",
    "success": "✓",
    "warning": "⚠︎",
    "error": "✕",
    "error_transient": "✕",
    "premium": _PREMIUM_STAR,
}


def _msg_text(kind: str, text: str) -> str:
    """Prefix a message with its kind's quiet monochrome glyph (two spaces,
    matching the chip convention). Kinds without a glyph pass through."""
    glyph = _MSG_GLYPHS.get(kind, "")
    return f"{glyph}  {text}" if glyph else text


def _msg_label_qss(kind: str) -> str:
    """QSS for a single-QLabel message of the given taxonomy kind."""
    fill, border = _MSG_TINTS[kind]
    text = ERROR_TEXT if kind.startswith("error") else "palette(text)"
    return (
        f"QLabel {{ background-color: {fill}; border: 1px solid {border};"
        f" border-radius: 4px; padding: 8px; font-size: 12px;"
        f" color: {text}; }}"
    )


def _msg_card_qss(name: str, kind: str) -> str:
    """QSS for a message CARD (a named QWidget with child labels) of the
    given taxonomy kind. Child labels stay transparent so the tint lives on
    the card only; remember WA_StyledBackground on the widget."""
    fill, border = _MSG_TINTS[kind]
    text = ERROR_TEXT if kind.startswith("error") else "palette(text)"
    return (
        f"QWidget#{name} {{ background-color: {fill};"
        f" border: 1px solid {border}; border-radius: 6px; }}"
        f"QLabel {{ background: transparent; border: none; color: {text}; }}"
    )


def _micro_header(text: str):
    """Micro section header: a quiet 10px bold label in NORMAL case, THE one
    way to introduce a subsection inside a card (Outline / Selection /
    Detection and friends). Deliberately typographic only: no uppercase, no
    letter-spacing, no colored tick or ornament (Yvann's calls 2026-07-11).
    Returns a QWidget whose ``header_label`` attribute is the QLabel, for
    dynamic call sites."""
    from qgis.PyQt.QtWidgets import QHBoxLayout, QLabel, QWidget

    w = QWidget()
    row = QHBoxLayout(w)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    lbl = QLabel(text)
    lbl.setStyleSheet(
        "font-size: 10px; font-weight: bold;"
        " color: palette(text); background: transparent; border: none;")
    row.addWidget(lbl)
    row.addStretch(1)
    w.header_label = lbl
    return w


def _card_divider():
    """1px full-width separator between the sub-blocks of ONE card (the
    review card's Confidence / View-as / Refine zones). Quiet neutral grey,
    never a colored ornament."""
    from qgis.PyQt.QtWidgets import QFrame

    line = QFrame()
    line.setFrameShape(QFrame.Shape.NoFrame)
    line.setFixedHeight(1)
    line.setStyleSheet("background: rgba(128, 128, 128, 0.16); border: none;")
    return line


def _step_dial(num: int, state: str = "todo"):
    """20px round step dial for ordered page steps: ``todo`` is a grey
    outline number, ``active`` a filled brand-blue number, ``done`` a lime
    outlined check. Returns a fixed-size QLabel."""
    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtWidgets import QLabel

    lbl = QLabel("✓" if state == "done" else str(num))
    lbl.setFixedSize(20, 20)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    if state == "active":
        qss = (f"background: {BRAND_BLUE}; color: #000000; border: none;"
               " border-radius: 10px; font-size: 11px; font-weight: 700;")
    elif state == "done":
        qss = (f"background: transparent; color: {BRAND_GREEN};"
               " border: 1px solid rgba(139, 172, 39, 0.75);"
               " border-radius: 10px; font-size: 11px; font-weight: 700;")
    else:
        qss = ("background: transparent; color: rgba(128, 128, 128, 0.95);"
               " border: 1px solid rgba(128, 128, 128, 0.45);"
               " border-radius: 10px; font-size: 11px; font-weight: 600;")
    lbl.setStyleSheet(qss)
    return lbl


def _sign_badge(symbol: str, color: str):
    """16px circular outline badge for the +/- click legend (extend/trim).
    One helper so every legend renders the same badge."""
    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtWidgets import QLabel

    badge = QLabel(symbol)
    badge.setFixedSize(16, 16)
    badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
    badge.setStyleSheet(
        f"background: transparent; border: 1px solid {color};"
        f" border-radius: 8px; color: {color};"
        " font-weight: bold; font-size: 11px;")
    return badge


# Framed section header-button with a chevron: a full-width control that
# reads as clickable at a glance (collapsible sections: the review's Shape
# settings, the Manual refine panel). Normal-case title (no uppercase, no
# letter-spacing). Hover warms the text/border blue.
_SECTION_TOGGLE_QSS = (
    "QPushButton { font-size: 11px; color: palette(text);"
    " font-weight: bold; background-color: rgba(128, 128, 128, 0.10);"
    " border: 1px solid rgba(128, 128, 128, 0.30); border-radius: 4px;"
    " padding: 8px 10px; text-align: left; }"
    f"QPushButton:hover {{ color: {BRAND_BLUE};"
    " border-color: rgba(30, 136, 229, 0.7); }"
)

# Theme-safe combobox for combos living inside a styled card. A parent card
# stylesheet knocks the child QComboBox off the app palette on the dark QGIS
# theme (selected text painted black on the dark base), so the combo names
# its colors explicitly via palette roles, which follow both themes.
_COMBO_THEME_QSS = (
    "QComboBox { color: palette(text); background-color: palette(base);"
    " border: 1px solid rgba(128, 128, 128, 0.35); border-radius: 3px;"
    " padding: 2px 8px; }"
    "QComboBox QAbstractItemView { color: palette(text);"
    " background-color: palette(base);"
    " selection-background-color: rgba(30, 136, 229, 0.35); }"
)

# Thin determinate progress line (3px) on a faint grey track: progress reads
# as a quiet instrument strip, not a heavy native bar. Call sites must
# setTextVisible(False); the measured status text lives in a label beside it.
_PROGRESS_THIN_QSS = (
    "QProgressBar { background: rgba(128, 128, 128, 0.25); border: none;"
    " border-radius: 2px; max-height: 3px; min-height: 3px; }"
    f"QProgressBar::chunk {{ background: {BRAND_BLUE}; border-radius: 2px; }}"
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

# Outline-blue secondary button: blue is the plugin's "temporary / still
# editing" colour, so an outlined blue action reads as "keep working on this
# result" next to a green "done" primary (e.g. the library picker next to
# the prompt).
_BTN_BLUE_OUTLINE = (
    f"QPushButton {{ background-color: transparent; color: {BRAND_BLUE};"
    f" border: 1px solid {BRAND_BLUE}; border-radius: 4px; font-weight: 600;"
    " padding: 6px 12px; }"
    "QPushButton:hover { background-color: rgba(30, 136, 229, 0.12); }"
    f"QPushButton:disabled {{ color: {DISABLED_TEXT};"
    f" border-color: {DISABLED_TEXT}; }}"
)

# Outline-red destructive secondary: quieter than the soft-fill _BTN_RED, for
# a destructive action that sits in a row NEXT TO a filled primary (the row's
# single loud button stays the primary).
_BTN_RED_OUTLINE = (
    f"QPushButton {{ background-color: transparent; color: {BRAND_RED};"
    " border: 1px solid rgba(211, 47, 47, 0.55); border-radius: 4px;"
    " padding: 6px 12px; }"
    "QPushButton:hover { background-color: rgba(211, 47, 47, 0.12); }"
    f"QPushButton:disabled {{ color: {DISABLED_TEXT};"
    " border-color: rgba(128, 128, 128, 0.35); }"
)

# Quiet text-link buttons: a blue link for a navigational/upsell side action,
# and a muted grey one for a de-emphasized escape hatch (hover warms it red,
# e.g. Cancel detection). Both underline on hover only.
_BTN_LINK = (
    f"QPushButton {{ background: transparent; border: none; color: {BRAND_BLUE};"
    " font-size: 11px; text-align: left; padding: 2px 0px; }"
    "QPushButton:hover { text-decoration: underline; }"
)
_BTN_LINK_MUTED = (
    "QPushButton { background: transparent; border: none;"
    " color: rgba(128, 128, 128, 0.9); font-size: 11px; padding: 4px 8px; }"
    f"QPushButton:hover {{ color: {ERROR_TEXT}; text-decoration: underline; }}"
)

# One-click suggestion chip (blue family): a small tinted action that fills
# in a prompt or arms a tool, e.g. the zero-result rescue chips.
_CHIP_QSS = (
    "QPushButton { background: rgba(30, 136, 229, 0.10);"
    " border: 1px solid rgba(30, 136, 229, 0.35); border-radius: 6px;"
    " color: palette(text); font-size: 12px; text-align: left;"
    " padding: 6px 10px; }"
    "QPushButton:hover { background: rgba(30, 136, 229, 0.20); }"
)

# Neutral outlined chip (the AI Edit prompt-row "Library" look): a quiet grey
# pill at rest, TerraLab-green tint on hover/press. For guided-path side
# buttons that sit NEXT TO an input and must not compete with the primary
# flow (an outlined brand-blue button there read as a competing action).
_BTN_CHIP = (
    "QPushButton { background: rgba(128, 128, 128, 0.08);"
    " border: 1px solid rgba(128, 128, 128, 0.40); border-radius: 6px;"
    " padding: 6px 12px; font-size: 12px; color: palette(text); }"
    "QPushButton:hover { background: rgba(139, 172, 39, 0.18);"
    " border-color: rgba(139, 172, 39, 0.65); }"
    "QPushButton:pressed { background: rgba(139, 172, 39, 0.32);"
    " border-color: rgba(139, 172, 39, 0.85); }"
    "QPushButton:disabled { color: rgba(128, 128, 128, 0.40);"
    " background: transparent; border-color: rgba(128, 128, 128, 0.20); }"
)

# Quiet one-line recap card (neutral grey family) for last-run summaries.
_RECAP_CARD_QSS = (
    "QLabel { font-size: 11px; color: palette(text);"
    " border: 1px solid rgba(128, 128, 128, 0.35);"
    " border-radius: 6px; padding: 6px 8px;"
    " background: rgba(128, 128, 128, 0.09); }"
)


def _btn_toggle_qss(rgb: tuple[int, int, int], text: str, armed_text: str,
                    weight: int = 700, quiet: bool = False) -> str:
    """Armable toggle button (tinted outline at rest, solid fill while the
    ``armed`` dynamic property is true). One generator so every draw-arming
    button (example, exclude) shares the exact same states.

    ``quiet=True`` gives a ghost rest state (neutral border, plain text) that
    only takes the color on hover/armed: for optional-path toggles that must
    not compete with the screen's real primary."""
    r, g, b = rgb
    solid = f"rgb({r}, {g}, {b})"
    if quiet:
        rest = (
            "QPushButton { background: transparent; color: palette(text);"
            " border: 1px solid rgba(128, 128, 128, 0.40); border-radius: 6px;"
            " padding: 6px 12px; font-size: 12px; }"
            f"QPushButton:hover {{ background: rgba({r}, {g}, {b}, 0.14);"
            f" border-color: rgba({r}, {g}, {b}, 0.55); }}"
        )
    else:
        rest = (
            f"QPushButton {{ background: rgba({r}, {g}, {b}, 0.12); color: {text};"
            f" border: 1px solid rgba({r}, {g}, {b}, 0.55); border-radius: 6px;"
            f" padding: 9px 16px; font-size: 12px; font-weight: {weight}; }}"
            f"QPushButton:hover {{ background: rgba({r}, {g}, {b}, 0.22); }}"
        )
    combined = rest
    combined += f'QPushButton[armed="true"] {{ background: {solid}; color: {armed_text};'
    combined += f" border: 1px solid {solid}; }}"
    combined += "QPushButton:disabled { background: transparent;"
    combined += " color: rgba(128, 128, 128, 0.5); border-color: rgba(128, 128, 128, 0.3); }"
    return combined


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
