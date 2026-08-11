"""Brand colors and QSS constants for the AI Segmentation dock
(design-system values shared with AI Edit), plus tiny style helpers."""
from __future__ import annotations

from .font_scale import scale_qss_font_px as _scale_qss_font_px

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
# These two are the DESIGN floors. A run whose noise floor keeps fainter
# detections replaces both with its own (see set_review_conf_floor), so the
# cutoff the review opens at is always reachable on the controls.
_REVIEW_CONF_STEP = 5
_REVIEW_CONF_MIN = 5
_REVIEW_CONF_MAX = 95
_REVIEW_CONF_SPIN_MIN = 1


def _snap_review_conf(value: int, floor: int | None = None) -> int:
    """Round a review-confidence percent to the nearest slider step, clamped to
    range. ``floor`` overrides the design minimum with the run's own one, which
    a run keeping fainter detections than the design floor needs so the slider
    can still reach the cutoff the review opened at. None keeps the design
    minimum."""
    lo = _REVIEW_CONF_MIN if floor is None else max(0, int(floor))
    snapped = int(round(value / _REVIEW_CONF_STEP)) * _REVIEW_CONF_STEP
    return max(lo, min(_REVIEW_CONF_MAX, snapped))


# Brand colors (Material Design 2 - shared with AI Edit, same values).
# Primary CTA buttons keep the material green; it reads as THE action color.
# Every other green accent uses the TerraLab leaf green below.
BTN_GREEN = "#43a047"
BTN_GREEN_HOVER = "#2e7d32"
BTN_GREEN_DISABLED = "#c8e6c9"

# Brand accent green = the QGIS green. Lime fills use BRAND_GREEN; green text
# on light backgrounds uses BRAND_GREEN_TEXT.
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

# The open body of a collapsible head (_SECTION_TOGGLE_OPEN_QSS). The head
# draws the top edge, so this card starts square and borderless there and the
# pair reads as ONE box. Two stacked cards with a gap between them said the
# head and the settings were separate things, when the head is the handle of
# the box it opens.
_CARD_JOINED_QSS = (
    "QWidget#{name} {{ background-color: rgba(128, 128, 128, 0.06);"
    " border: 1px solid rgba(128, 128, 128, 0.30); border-top: none;"
    " border-top-left-radius: 0px; border-top-right-radius: 0px;"
    " border-bottom-left-radius: 6px; border-bottom-right-radius: 6px; }}"
)

# Block nested INSIDE a _CARD_QSS card, for a control that belongs to the card
# but is not part of its current content (the persistent View-as row sitting
# above the step pages). One step more fill than the card it sits on, so it
# reads as its own component without a second hue. Same #objectName rule as
# _CARD_QSS: the fill stays off the child widgets.
_SUBCARD_QSS = (
    "QWidget#{name} {{ background-color: rgba(128, 128, 128, 0.10);"
    " border: 1px solid rgba(128, 128, 128, 0.28); border-radius: 6px; }}"
)
_SUBCARD_MARGINS = (10, 8, 10, 8)

# Border reset for a QPushButton living inside a _CARD_QSS / _msg_card_qss
# card: the button's own constant already sets its fill and hover state, so
# only the native frame needs killing here. Append to a card stylesheet
# (``_msg_card_qss(name, kind) + _CARD_CHILD_BTN_RESET_QSS``) rather than
# re-typing the rule at each call site.
_CARD_CHILD_BTN_RESET_QSS = "QPushButton { border: none; }"

# NOTE: no colored edge ornaments. A left "spine" stripe and title ticks were
# tried and rejected: colored slots on text edges read as generic AI-tool
# design. Cards carry hierarchy through content, never through a colored
# border.

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
#              other guidance text. For an OFFER only: a box that blocks a
#              button is an error, red and starless, or the refusal reads as
#              an invitation and the user keeps clicking a dead button.
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

# The two Semi-Auto engine cards. A picture of each place says "off your
# machine" and "on your machine" before either name is read, which is the whole
# job of that row. The second deliberate exception to the text-glyph rule below
# (the first is the tip lightbulb), and kept as a PAIR: one colour emoji beside
# one monochrome character reads as a mistake, not as a choice.
_CLOUD_EMOJI = "☁️"
_LAPTOP_EMOJI = "💻"

# Message-kind glyph prefixes. Statuses (armed/success/warning/error) carry
# quiet monochrome TEXT glyphs, plain characters tinted by the label's own
# text color (U+FE0E forces text presentation on macOS); mass-emoji reads as
# cheap. The ONE exception is info/tips: the lightbulb emoji, warmer than a
# flat i-icon for guidance.
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


# Sentinel href for the in-banner "Report this problem" action (same value as
# AI Edit's, so both plugins route report links through one recognizable key).
# It is never opened as a URL: a linkActivated handler intercepts it and opens
# the copy-logs/email dialog instead.
_REPORT_HREF = "terralab://report-problem"


def _error_banner_html(message: str, report_link_text: str) -> str:
    """RichText body for an actionable error banner: the error glyph + the
    escaped message (newlines preserved as line breaks), then a persistent
    report link styled in the banner's own error text color (a bare <a> would
    otherwise render in the default hyperlink blue). ``message`` is escaped so
    a server-supplied error string can never inject markup."""
    import html
    safe = html.escape(message or "").replace("\n", "<br>")
    body = _msg_text("error", safe)
    link = (
        f'<a href="{_REPORT_HREF}" style="color: {ERROR_TEXT};">'
        f"{html.escape(report_link_text)}</a>"
    )
    return f"{body}<br>{link}"


def _msg_label_qss(kind: str) -> str:
    """QSS for a single-QLabel message of the given taxonomy kind."""
    from .font_scale import scale_qss_font_px

    fill, border = _MSG_TINTS[kind]
    text = ERROR_TEXT if kind.startswith("error") else "palette(text)"
    return scale_qss_font_px(
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


def _micro_header(text: str, gloss: str | None = None):
    """Micro section header: a quiet 10px bold label in NORMAL case, THE one
    way to introduce a subsection inside a card (Outline / Selection /
    Detection and friends). Deliberately typographic only: no uppercase, no
    letter-spacing, no colored tick or ornament.
    Returns a QWidget whose ``header_label`` attribute is the QLabel, for
    dynamic call sites.

    ``gloss`` (optional) is a muted one-line note shown on the same row right
    after the title, for a subsection that reads clearer when it names its
    purpose (e.g. "Shape - how each outline is styled") without spending a
    second line. Exposed as ``gloss_label`` when present."""
    from qgis.PyQt.QtWidgets import QHBoxLayout, QLabel, QWidget

    w = QWidget()
    row = QHBoxLayout(w)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    from .font_scale import scale_qss_font_px

    lbl = QLabel(text)
    lbl.setStyleSheet(scale_qss_font_px(
        "font-size: 10px; font-weight: bold;"
        " color: palette(text); background: transparent; border: none;"))
    row.addWidget(lbl)
    if gloss:
        gl = QLabel(gloss)
        gl.setStyleSheet(scale_qss_font_px(
            "font-size: 10px; color: rgba(128, 128, 128, 0.95);"
            " background: transparent; border: none;"))
        row.addWidget(gl)
        w.gloss_label = gl
    row.addStretch(1)
    w.header_label = lbl
    return w


def _settings_zone(obj_name: str, title: str, gloss: str, items: list):
    """One group of settings as its own sub-card: the house ``_SUBCARD_QSS``
    recipe with a ``_micro_header`` naming the group and glossing its job.

    THE way a settings panel is cut into groups, so Shape / Outline / Size read
    as separated panels instead of one flat block of rows. Shared by the
    Automatic review's Shapes step and Manual's Refine panel, which is why it
    lives here and not beside either one. ``items`` are QLayout or QWidget rows,
    kept in their original build so their wiring and visibility toggles hold.
    Sibling sub-cards need no divider between them (a divider is for zones
    INSIDE one card)."""
    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtWidgets import QLayout, QVBoxLayout, QWidget

    zone = QWidget()
    zone.setObjectName(obj_name)
    zone.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    zone.setStyleSheet(_SUBCARD_QSS.format(name=obj_name))
    col = QVBoxLayout(zone)
    col.setContentsMargins(*_SUBCARD_MARGINS)
    col.setSpacing(6)
    col.addWidget(_micro_header(title, gloss))
    for item in items:
        if isinstance(item, QLayout):
            col.addLayout(item)
        else:
            col.addWidget(item)
    return zone


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


def _choice_divider(text: str):
    """The word between two sibling cards that are the two halves of ONE
    choice: a hairline, the word, a hairline.

    Not `_card_divider`, which separates the zones inside a single card. Use it
    only where the two cards are alternatives, never as decoration between
    unrelated cards. The caller passes the translated word."""
    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtWidgets import QHBoxLayout, QLabel, QWidget

    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 2, 0, 2)
    lay.setSpacing(8)
    label = QLabel(text)
    label.setStyleSheet(
        "font-size: 11px; color: rgba(128,128,128,0.95);"
        " background: transparent; border: none;")
    # The hairlines carry a fixed 1px height, so they need the alignment: with
    # none they would settle at the top of the row instead of on the word's line.
    lay.addWidget(_card_divider(), 1, Qt.AlignmentFlag.AlignVCenter)
    lay.addWidget(label, 0, Qt.AlignmentFlag.AlignVCenter)
    lay.addWidget(_card_divider(), 1, Qt.AlignmentFlag.AlignVCenter)
    return row


def _step_dial(num: int, state: str = "todo"):
    """20px round step dial for ordered page steps: ``todo`` is a grey
    outline number, ``active`` a filled brand-blue number, ``done`` a lime
    outlined check. Returns a fixed-size QLabel."""
    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtWidgets import QLabel

    from .font_scale import scale_px_length, scale_qss_font_px

    lbl = QLabel("✓" if state == "done" else str(num))
    # The circle grows with its own number, else a larger digit spills out of
    # a dial that stayed 20 wide.
    side = scale_px_length(20)
    radius = side // 2
    lbl.setFixedSize(side, side)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    if state == "active":
        qss = (f"background: {BRAND_BLUE}; color: #000000; border: none;"
               f" border-radius: {radius}px; font-size: 11px; font-weight: 700;")
    elif state == "done":
        qss = (f"background: transparent; color: {BRAND_GREEN};"
               " border: 1px solid rgba(139, 172, 39, 0.75);"
               f" border-radius: {radius}px; font-size: 11px; font-weight: 700;")
    else:
        qss = ("background: transparent; color: rgba(128, 128, 128, 0.95);"
               " border: 1px solid rgba(128, 128, 128, 0.45);"
               f" border-radius: {radius}px; font-size: 11px; font-weight: 600;")
    lbl.setStyleSheet(scale_qss_font_px(qss))
    return lbl


def _sign_badge(symbol: str, color: str):
    """16px circular outline badge for the +/- click legend (extend/trim).
    One helper so every legend renders the same badge."""
    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtWidgets import QLabel

    from .font_scale import scale_px_length, scale_qss_font_px

    badge = QLabel(symbol)
    side = scale_px_length(16)
    badge.setFixedSize(side, side)
    badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
    badge.setStyleSheet(scale_qss_font_px(
        f"background: transparent; border: 1px solid {color};"
        f" border-radius: {side // 2}px; color: {color};"
        " font-weight: bold; font-size: 11px;"))
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

# The same head with its card open under it (_CARD_JOINED_QSS): no bottom edge
# and no bottom corners, so head and body draw as one box. Hover only moves the
# text colour here, because tinting three sides of a box whose fourth side is
# the card below leaves a broken rectangle on the screen.
_SECTION_TOGGLE_OPEN_QSS = (
    "QPushButton { font-size: 11px; color: palette(text);"
    " font-weight: bold; background-color: rgba(128, 128, 128, 0.10);"
    " border: 1px solid rgba(128, 128, 128, 0.30); border-bottom: none;"
    " border-top-left-radius: 6px; border-top-right-radius: 6px;"
    " border-bottom-left-radius: 0px; border-bottom-right-radius: 0px;"
    " padding: 8px 10px; text-align: left; }"
    f"QPushButton:hover {{ color: {BRAND_BLUE}; }}"
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

# Theme-safe line edit for a text input living inside a styled card, mirroring
# _COMBO_THEME_QSS: palette(text) on palette(base) so the input follows both
# the light and dark QGIS theme, with a brand-blue border on focus.
_INPUT_THEME_QSS = (
    "QLineEdit { border: 1px solid rgba(128, 128, 128, 0.35);"
    " border-radius: 6px; padding: 7px 10px; background: palette(base);"
    " color: palette(text); }"
    f"QLineEdit:focus {{ border: 1px solid {BRAND_BLUE}; }}"
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

# Weight of every button label. Windows draws the default UI font much thinner
# than macOS does, and a regular-weight label on a filled button was the first
# thing that stopped being readable there. One value, so the whole button
# family moves together.
_BTN_LABEL_WEIGHT = "font-weight: 600;"

_BTN_GREEN = (
    f"QPushButton {{ background-color: {BTN_GREEN}; color: #000000;"
    f" padding: 8px 16px; border: none; border-radius: 4px;"
    f" {_BTN_LABEL_WEIGHT} }}"
    f"QPushButton:hover {{ background-color: {BTN_GREEN_HOVER}; color: #000000; }}"
    f"QPushButton:disabled {{ background-color: {BTN_GREEN_DISABLED};"
    f" color: {DISABLED_TEXT}; }}"
)

# The green primary of a numbered step in the review ladder. Same fill as
# _BTN_GREEN, a step up: beside it sits an underlined "Re-run the whole zone"
# link, and at the shared default size the link won the eye and users re-ran a
# zone they only meant to move past.
#
# 13px, the design system's title step. It stays two points above that 11px
# link and keeps the fill and the weight the link does not have, so the primary
# still reads first; 15px only made the button shout across a narrow dock.
_BTN_GREEN_STEP = (
    f"QPushButton {{ background-color: {BTN_GREEN}; color: #000000;"
    f" padding: 10px 16px; border: none; border-radius: 4px;"
    f" font-size: 13px; {_BTN_LABEL_WEIGHT} }}"
    f"QPushButton:hover {{ background-color: {BTN_GREEN_HOVER}; color: #000000; }}"
    f"QPushButton:disabled {{ background-color: {BTN_GREEN_DISABLED};"
    f" color: {DISABLED_TEXT}; }}"
)

_BTN_GREEN_AUTH = (
    f"QPushButton {{ background-color: {BTN_GREEN}; color: #000000;"
    f" border: none; border-radius: 4px; {_BTN_LABEL_WEIGHT} }}"
    f"QPushButton:hover {{ background-color: {BTN_GREEN_HOVER}; }}"
    f"QPushButton:disabled {{ background-color: {BRAND_DISABLED};"
    f" color: {DISABLED_TEXT}; }}"
)

_BTN_BLUE = (
    f"QPushButton {{ background-color: {BRAND_BLUE}; color: #000000;"
    f" padding: 6px 12px; border: none; border-radius: 4px;"
    f" {_BTN_LABEL_WEIGHT} }}"
    f"QPushButton:hover {{ background-color: {BRAND_BLUE_HOVER}; color: #000000; }}"
    f"QPushButton:disabled {{ background-color: {BRAND_DISABLED};"
    f" color: {DISABLED_TEXT}; }}"
)

_BTN_BLUE_AUTH = (
    f"QPushButton {{ background-color: {BRAND_BLUE}; color: #000000;"
    f" border: none; border-radius: 4px; {_BTN_LABEL_WEIGHT} }}"
    f"QPushButton:hover {{ background-color: {BRAND_BLUE_HOVER}; }}"
    f"QPushButton:disabled {{ background-color: {BRAND_DISABLED}; }}"
)

# Primary blue CTA: same heft as _BTN_GREEN (8px 16px) but in the Automatic
# brand blue. Used for the Automatic-mode "Start" so it echoes the blue tab
# underline, the way the green Start echoes the green Interactive underline.
_BTN_BLUE_PRIMARY = (
    f"QPushButton {{ background-color: {BRAND_BLUE}; color: #000000;"
    f" padding: 8px 16px; border: none; border-radius: 4px;"
    f" {_BTN_LABEL_WEIGHT} }}"
    f"QPushButton:hover {{ background-color: {BRAND_BLUE_HOVER}; color: #000000; }}"
    f"QPushButton:disabled {{ background-color: {BRAND_DISABLED};"
    f" color: {DISABLED_TEXT}; }}"
)


# The one button that starts a mode. It is alone on its page and it is the only
# thing there is to click, so its label is larger than a button sitting in a row
# with others. A function rather than two more constants, so Manual green and
# Automatic blue can never drift apart.
_BTN_START_FONT_PX = 13


def _btn_start_qss(base: str) -> str:
    """A primary button constant, with the label size a page's Start carries."""
    return base + f"QPushButton {{ font-size: {_BTN_START_FONT_PX}px; }}"


# Ghost / outline button (mirrors AI Edit's _BTN_GHOST): transparent fill with
# a faint border, for a secondary action that sits beside a filled primary
# (e.g. Exit next to Detect).
_BTN_GHOST = (
    "QPushButton { background-color: transparent; color: palette(text);"
    " padding: 8px 16px; border-radius: 4px;"
    f" {_BTN_LABEL_WEIGHT}"
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
    f" {_BTN_LABEL_WEIGHT}"
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
# One step louder than _BTN_LINK_MUTED, same shape: full-contrast text at 11px,
# semi-bold, underlined at rest. For a secondary action that a user must be
# able to FIND (re-run the whole zone), where the muted grey read as a
# subtitle. Still no fill and no border, so the screen's one filled primary
# keeps the hierarchy. It stays UNDER the step primary's size: at 12px next to
# a default-size green button it read as the louder of the two.
_BTN_LINK_STRONG = (
    "QPushButton { background: transparent; border: none;"
    " color: palette(text); font-size: 11px; font-weight: 600;"
    " padding: 4px 8px; text-decoration: underline; }"
    f"QPushButton:hover {{ color: {ERROR_TEXT}; }}"
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
    f" padding: 6px 12px; font-size: 12px; color: palette(text);"
    f" {_BTN_LABEL_WEIGHT} }}"
    "QPushButton:hover { background: rgba(139, 172, 39, 0.18);"
    " border-color: rgba(139, 172, 39, 0.65); }"
    "QPushButton:pressed { background: rgba(139, 172, 39, 0.32);"
    " border-color: rgba(139, 172, 39, 0.85); }"
    "QPushButton:disabled { color: rgba(128, 128, 128, 0.40);"
    " background: transparent; border-color: rgba(128, 128, 128, 0.20); }"
)

# Action tile for a small grid of same-weight actions on ONE object (the
# Correct step's selected-detection menu). Nothing here is filled: the page
# keeps exactly one filled button (the green primary), so the grid reads as a
# menu of equals rather than four competing calls to action. Text is left
# aligned so the leading glyphs line up down each column.
_BTN_TILE = (
    "QPushButton { background: rgba(128, 128, 128, 0.08);"
    " border: 1px solid rgba(128, 128, 128, 0.30); border-radius: 6px;"
    " color: palette(text); font-size: 12px; text-align: left;"
    f" {_BTN_LABEL_WEIGHT}"
    " padding: 9px 10px; }"
    "QPushButton:hover { background: rgba(30, 136, 229, 0.12);"
    " border-color: rgba(30, 136, 229, 0.45); }"
    "QPushButton:disabled { color: rgba(128, 128, 128, 0.40);"
    " background: transparent; border-color: rgba(128, 128, 128, 0.18); }"
)

# The same tile, held open: blue is the "this tool is armed / this panel is
# showing" state, so an open tile matches every other armed control.
_BTN_TILE_ACTIVE = (
    "QPushButton { background: rgba(30, 136, 229, 0.18);"
    f" border: 1px solid {BRAND_BLUE}; border-radius: 6px;"
    " color: palette(text); font-size: 12px; text-align: left;"
    " font-weight: 600; padding: 9px 10px; }"
    "QPushButton:hover { background: rgba(30, 136, 229, 0.26); }"
)

# Segmented two-half switch. The active half is FILLED brand blue, because a
# tint behind text reads as a hover state rather than as the current choice: on
# a dark panel the two halves came out nearly the same weight and the user had
# to hunt for the selected one. Text on the fill is WHITE, not the black the
# filled BUTTON constants use: a switch half is a label, not a call to action,
# and black on blue reads as disabled. Unchecked halves keep a 1px transparent
# border so
# checking one never shifts its text by a pixel. A design-system addition
# mirrored in the web tokens (.seg). No PRO badge on either caller: neither is a
# paid gate.


def _segmented_switch_qss(name: str) -> str:
    """QSS for a two-half segmented control on the given frame objectName.

    One shape for every "pick one of two" on the panel, so a user who has
    learnt the Correct step's switch already knows the Semi-Auto engine one.
    Add a caller here rather than writing a second set of rules.
    """
    return (
        f"QFrame#{name} {{"
        "  background: rgba(128, 128, 128, 0.05);"
        "  border: 1px solid rgba(128, 128, 128, 0.35);"
        "  border-radius: 6px;"
        "}"
        "QPushButton {"
        "  background: transparent;"
        "  border: 1px solid transparent;"
        "  border-radius: 5px;"
        "  padding: 6px 0px;"
        "  font-size: 12px;"
        "  color: palette(text);"
        "}"
        "QPushButton:hover {"
        "  background: rgba(128, 128, 128, 0.12);"
        "}"
        "QPushButton:checked {"
        f"  background: {BRAND_BLUE};"
        "  color: #ffffff;"
        "  font-weight: 700;"
        f"  border: 1px solid {BRAND_BLUE};"
        "}"
        f"QPushButton:checked:hover {{"
        f"  background: {BRAND_BLUE_HOVER};"
        f"  border-color: {BRAND_BLUE_HOVER};"
        f"}}"
    )


_METHOD_SWITCH_QSS = _segmented_switch_qss("methodSwitchFrame")

# The Semi-Auto engine picker: two option cards side by side, each carrying a
# name and the one thing that separates it from the other.
#
# NOT the segmented bar above, and the difference is deliberate. That control
# switches a view, so a transparent resting half is right: it costs nothing to
# try. This one picks where the work runs and what it costs, and the unpicked
# side has to read as a real, pressable thing rather than as empty space, so
# both cards carry a fill and a border at rest.
#
# The picked one is FILLED with the brand blue, like every other picked cell on
# the panel. A translucent blue wash was tried and rejected: it read as a hover
# state, so the two cards looked equally chosen and the answer to "which one am
# I on" cost a second look.
_ENGINE_CARD_QSS = (
    "QPushButton {"
    "  background: rgba(128, 128, 128, 0.10);"
    "  border: 1px solid rgba(128, 128, 128, 0.32);"
    "  border-radius: 6px;"
    "  text-align: left;"
    "}"
    "QPushButton:hover {"
    "  background: rgba(128, 128, 128, 0.18);"
    "  border-color: rgba(128, 128, 128, 0.50);"
    "}"
    "QPushButton:checked {"
    f"  background: {BRAND_BLUE};"
    f"  border: 1px solid {BRAND_BLUE};"
    "}"
    f"QPushButton:checked:hover {{ background: {BRAND_BLUE_HOVER};"
    f" border: 1px solid {BRAND_BLUE_HOVER}; }}"
    "QLabel { background: transparent; border: none; }"
)

# The two lines inside an engine card, picked and unpicked.
#
# They have to be constants the widget re-applies on every toggle, not a
# descendant rule in the QSS above. A QLabel carries its own stylesheet, and a
# stylesheet set on the widget itself always beats one inherited from an
# ancestor, so `QPushButton:checked QLabel { color: ... }` never wins and the
# picked card would ship near-white text on the brand blue.
#
# White on the filled blue, not the black the filled-button constants use.
# Yvann's call, twice: a picked card has to keep reading as a card of text, and
# black on blue read as a disabled control to him. The gloss is the full white
# too, not a dimmed one, because at 11px on a saturated fill any transparency
# turns to mud.
_ENGINE_CARD_TITLE_QSS = (
    "font-size: 12px; font-weight: bold; color: palette(text);")
_ENGINE_CARD_GLOSS_QSS = "font-size: 11px; color: palette(text);"
_ENGINE_CARD_TITLE_ON_QSS = (
    "font-size: 12px; font-weight: bold; color: #ffffff;")
_ENGINE_CARD_GLOSS_ON_QSS = "font-size: 11px; color: #ffffff;"


# Destructive footer row: the one action that ENDS an object, kept quiet and
# apart from its peers. Neutral at rest so a panel of ordinary choices never
# reads as a warning; the red appears on hover, when the pointer is already on
# it. Same family as _BTN_LINK_MUTED, sized for a full row.
_BTN_REMOVE_ROW = (
    "QPushButton { background: transparent; border: none; text-align: left;"
    " color: rgba(128, 128, 128, 0.95); font-size: 12px; padding: 4px 0px; }"
    f"QPushButton:hover {{ color: {ERROR_TEXT}; text-decoration: underline; }}"
    f"QPushButton:disabled {{ color: {DISABLED_TEXT}; }}"
)

# _RECAP_CARD_QSS lived here and went on 2026-08-11 with its last caller. Both
# last-run recap cards, Semi-Auto's and Automatic's, are gone: what a finished
# run produced is in the legend and on the footer credit ring, and a Start page
# is about the next run. Do not rebuild a green summary card for that.


def _btn_toggle_qss(rgb: tuple[int, int, int], text: str, armed_text: str,
                    weight: int = 700, quiet: bool = False,
                    filled: bool = False) -> str:
    """Armable toggle button (tinted outline at rest, solid fill while the
    ``armed`` dynamic property is true). One generator so every draw-arming
    button (example, exclude) shares the exact same states.

    ``quiet=True`` gives a ghost rest state (neutral border, plain text) that
    only takes the color on hover/armed: for optional-path toggles that must
    not compete with the screen's real primary.

    ``filled=True`` gives a solid-fill rest (black text on the colour, the
    _BTN_GREEN convention) that darkens on hover and armed: for a toggle that
    IS the step's action, so it reads as a clear coloured button."""
    r, g, b = rgb
    solid = f"rgb({r}, {g}, {b})"
    dark = f"rgb({int(r * 0.8)}, {int(g * 0.8)}, {int(b * 0.8)})"
    if quiet:
        rest = (
            "QPushButton { background: transparent; color: palette(text);"
            " border: 1px solid rgba(128, 128, 128, 0.40); border-radius: 6px;"
            " padding: 6px 12px; font-size: 12px; }"
            f"QPushButton:hover {{ background: rgba({r}, {g}, {b}, 0.14);"
            f" border-color: rgba({r}, {g}, {b}, 0.55); }}"
        )
    elif filled:
        rest = (
            f"QPushButton {{ background: {solid}; color: #000000;"
            f" border: none; border-radius: 6px; padding: 9px 16px;"
            f" font-size: 12px; font-weight: {weight}; }}"
            f"QPushButton:hover {{ background: {dark}; }}"
        )
    else:
        rest = (
            f"QPushButton {{ background: rgba({r}, {g}, {b}, 0.12); color: {text};"
            f" border: 1px solid rgba({r}, {g}, {b}, 0.55); border-radius: 6px;"
            f" padding: 9px 16px; font-size: 12px; font-weight: {weight}; }}"
            f"QPushButton:hover {{ background: rgba({r}, {g}, {b}, 0.22); }}"
        )
    combined = rest
    if filled:
        # A filled toggle is solid at rest, so "armed" (drawing now) reads as
        # the darker shade with black text, not a fresh fill.
        combined += (f'QPushButton[armed="true"] {{ background: {dark};'
                     f" color: #000000; border: none; }}")
    else:
        combined += (f'QPushButton[armed="true"] {{ background: {solid};'
                     f" color: {armed_text}; border: 1px solid {solid}; }}")
    combined += "QPushButton:disabled { background: transparent;"
    combined += " color: rgba(128, 128, 128, 0.5); border-color: rgba(128, 128, 128, 0.3); }"
    from .font_scale import scale_qss_font_px

    return scale_qss_font_px(combined)


# Small filled action button for a DismissibleHint's optional CTA (e.g. "Open
# the tutorial"), in the hint's own tint color. Black text on the solid fill
# (the same AA-safe convention as _BTN_GREEN / _BTN_BLUE_PRIMARY) rather than
# white-on-color, which fails contrast on lighter tints and drifts from the
# button-family look. A function, not a fixed constant, because the color
# follows whichever tint the hint card uses.
def _btn_hint_action_qss(rgb: tuple[int, int, int]) -> str:
    from .font_scale import scale_qss_font_px

    r, g, b = rgb
    return scale_qss_font_px(
        f"QToolButton {{ background: rgb({r}, {g}, {b}); color: #000000;"
        " border: none; border-radius: 4px; padding: 3px 10px;"
        " font-size: 11px; font-weight: 700; }"
        f"QToolButton:hover {{ background: rgba({r}, {g}, {b}, 0.85); }}"
    )


_BTN_GRAY = (
    f"QPushButton {{ background-color: {BRAND_GRAY}; color: #000000;"
    f" padding: 4px 8px; border: none; border-radius: 4px;"
    f" {_BTN_LABEL_WEIGHT} }}"
    f"QPushButton:hover {{ background-color: {BRAND_GRAY_HOVER}; color: #000000; }}"
    f"QPushButton:disabled {{ background-color: {BRAND_DISABLED}; color: {DISABLED_TEXT}; }}"
)

_BTN_RED = (
    f"QPushButton {{ background-color: rgba(211,47,47,0.12); color: {BRAND_RED};"
    f" padding: 6px 12px; border: none; border-radius: 4px;"
    f" {_BTN_LABEL_WEIGHT} }}"
    f"QPushButton:hover {{ background-color: rgba(211,47,47,0.22); }}"
)

_BTN_EXPORT_READY = (
    f"QPushButton {{ background-color: {BTN_GREEN}; color: #000000;"
    f" padding: 6px 12px; border: none; border-radius: 4px;"
    f" {_BTN_LABEL_WEIGHT} }}"
    f"QPushButton:hover {{ background-color: {BTN_GREEN_HOVER}; color: #000000; }}"
)

_BTN_EXPORT_DISABLED = (
    f"QPushButton {{ background-color: {BRAND_DISABLED}; color: {DISABLED_TEXT};"
    f" padding: 6px 12px; border: none; border-radius: 4px;"
    f" {_BTN_LABEL_WEIGHT} }}"
)

# Compact filled buttons for the browser-handoff waiting state. Both carry a
# soft tint (never transparent): neutral for "open again", red for "cancel".
_BTN_PAIR_NEUTRAL = (
    "QPushButton { background-color: rgba(128,128,128,0.16); color: palette(text);"
    f" border: none; border-radius: 4px; {_BTN_LABEL_WEIGHT} }}"
    "QPushButton:hover { background-color: rgba(128,128,128,0.28); }"
)
_BTN_PAIR_CANCEL = (
    f"QPushButton {{ background-color: rgba(211,47,47,0.12); color: {BRAND_RED};"
    f" border: none; border-radius: 4px; {_BTN_LABEL_WEIGHT} }}"
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


# The panel follows the text size the user set in QGIS (see font_scale). A
# constant applied at build time is caught by the pass over the finished panel,
# but the same constant re-applied on a state change is not, and the widget
# would snap back to the base size mid-session. Growing them here, once, covers
# both. A no-op on the default text size, and outside QGIS.
for _qss_name in (
    "_SECTION_TOGGLE_QSS",
    "_SECTION_TOGGLE_OPEN_QSS",
    "_ENGINE_CARD_TITLE_QSS",
    "_ENGINE_CARD_GLOSS_QSS",
    "_ENGINE_CARD_TITLE_ON_QSS",
    "_ENGINE_CARD_GLOSS_ON_QSS",
    "_INSTRUCTIONS_CARD_QSS",
    "_INSTRUCTIONS_HINT_QSS",
    "_BTN_LINK",
    "_BTN_LINK_MUTED",
    "_BTN_LINK_STRONG",
    "_CHIP_QSS",
    "_BTN_CHIP",
    "_BTN_TILE",
    "_BTN_TILE_ACTIVE",
    "_METHOD_SWITCH_QSS",
    "_BTN_REMOVE_ROW",
    "_FOOTER_ICON_BTN_STYLE",
    "_HELP_ICON_BTN_STYLE",
    "_FOOTER_CTA_BTN_STYLE",
):
    globals()[_qss_name] = _scale_qss_font_px(globals()[_qss_name])
del _qss_name
