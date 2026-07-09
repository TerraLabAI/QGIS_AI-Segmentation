

from __future__ import annotations

from ...core.server_dials import dial_in_range
from .font_scale import scale_qss_font_px as _scale_qss_font_px




















_REVIEW_CONF_STEP = 5
_REVIEW_CONF_MIN = 5
_REVIEW_CONF_MAX = 95
_REVIEW_CONF_SPIN_MIN = 1


def review_conf_step() -> int:

    return dial_in_range("tuning.review.conf_step", _REVIEW_CONF_STEP, 1, 25)


def review_conf_min() -> int:

    return dial_in_range("tuning.review.conf_min", _REVIEW_CONF_MIN, 0, 90)


def review_conf_max() -> int:





    lo = review_conf_min()
    return dial_in_range("tuning.review.conf_max", _REVIEW_CONF_MAX, lo + 1, 99)


def _snap_review_conf(value: int, floor: int | None = None) -> int:





    lo = review_conf_min() if floor is None else max(0, int(floor))
    step = review_conf_step()
    snapped = int(round(value / step)) * step
    return max(lo, min(review_conf_max(), snapped))























FONT_BASE = 13
FONT_BODY = 12
FONT_HINT = 11


FONT_MICRO = 11

FONT_PROSE = 14





RADIUS_CHIP = 6
RADIUS_CONTROL = 8
RADIUS_ROW = RADIUS_CONTROL
RADIUS_CARD = 10
RADIUS_PANEL = 14

BTN_PX = 32
BTN_SMALL_PX = 28
ROW_PX = 28
BTN_PILL_PX = BTN_PX
BTN_PRIMARY_WIDE_PX = 36


BTN_CHIP_PX = 30









RADIUS_PILL = 10
RADIUS_PILL_WIDE = RADIUS_PILL



SPACE_OUTER = 8
SPACE_CARD = 6
SPACE_TIGHT = 4
SPACE_STAGE = 12


def _dark_at_import() -> bool:
    try:
        from qgis.PyQt.QtGui import QPalette
        from qgis.PyQt.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            return False
        return app.palette().color(QPalette.ColorRole.Window).lightness() < 128
    except Exception:  # noqa: BLE001
        return False










_NEUTRALS_LIGHT = {
    "page": "#f7f8fa", "canvas": "#eef0f2", "surface": "#fefefe", "inset": "#f4f6f8",
    "hover": "#eff1f4", "hover_2": "#e2e5ea", "field": "#eef0f3",
    "ink": "#1f2124", "ink_2": "#5b5f66", "ink_3": "#6b7079", "ink_hover": "#33363b",
    "line": "#e2e5eb", "line_strong": "#cdd2d9", "line_soft": "#edf0f3",
    "tooltip_bg": "#25272b", "tooltip_fg": "#f6f7f8",
}
_NEUTRALS_DARK = {
    "page": "#17181a", "canvas": "#1c1d1f", "surface": "#232427", "inset": "#1f2022",
    "hover": "#2a2b2e", "hover_2": "#313236", "field": "#2b2c2f",
    "ink": "#f2f3f4", "ink_2": "#a5a8ad", "ink_3": "#92959b", "ink_hover": "#d7dade",
    "line": "#2e3033", "line_strong": "#3a3c40", "line_soft": "#27282b",
    "tooltip_bg": "#111214", "tooltip_fg": "#f3f4f5",
}



DARK_UI = _dark_at_import()
_N = _NEUTRALS_DARK if DARK_UI else _NEUTRALS_LIGHT
PAGE = _N["page"]
CANVAS = _N["canvas"]
SURFACE = _N["surface"]
INSET = _N["inset"]
HOVER = _N["hover"]
HOVER_ON = _N["hover_2"]
FIELD = _N["field"]
INK = _N["ink"]
INK_2 = _N["ink_2"]
INK_3 = _N["ink_3"]
INK_HOVER = _N["ink_hover"]
LINE = _N["line"]
LINE_STRONG = _N["line_strong"]
LINE_SOFT = _N["line_soft"]
TOOLTIP_BG = _N["tooltip_bg"]
TOOLTIP_FG = _N["tooltip_fg"]


MUTED = INK_2
MUTED_SOFT = INK_3
HAIRLINE = LINE
HAIRLINE_STRONG = LINE_STRONG
TINT = FIELD
TINT_HOVER = HOVER
TINT_ON = HOVER_ON





BTN_GREEN = "#43a047"
BTN_GREEN_HOVER = "#2e7d32"
BTN_GREEN_DISABLED = "#c8e6c9"



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




ACCENT = BTN_GREEN


ACCENT_DARK = "#57b35b"


ON_ACCENT = "#000000"




ACCENT_TINT = "rgba(30, 136, 229, 0.10)"
ACCENT_TINT_ON = "rgba(30, 136, 229, 0.22)"
ACCENT_BORDER_SOFT = "rgba(30, 136, 229, 0.35)"
ACCENT_BORDER = BRAND_BLUE



LINK_INK = "#42a5f5" if DARK_UI else "#1565c0"



LINE_INPUT = "#7a7d83" if DARK_UI else "#80868f"





ACCENT_INK_LIGHT = "#437010"
ACCENT_INK_DARK = "#a3c644"





RED_INK = "#f26b69" if DARK_UI else "#c62828"


GREEN_TEXT = "#66bb6a" if DARK_UI else "#2e7d32"
ORANGE_TEXT = "#f5a623" if DARK_UI else "#b45309"
RED_TEXT = RED_INK
RED_TINT = "rgba(229, 72, 77, 0.10)" if not DARK_UI else "rgba(238, 92, 97, 0.14)"


DISABLED_FILL = FIELD
DISABLED_INK = INK_3


def is_dark() -> bool:


    try:
        from qgis.PyQt.QtGui import QPalette
        from qgis.PyQt.QtWidgets import QApplication

        app = QApplication.instance()
        palette = app.palette() if app is not None else QPalette()
        return palette.color(QPalette.ColorRole.Window).lightness() < 128
    except Exception:  # noqa: BLE001
        return False


def accent_ink() -> str:

    return ACCENT_INK_DARK if is_dark() else ACCENT_INK_LIGHT






_SLIDER_QSS = (
    "QSlider:horizontal { min-height: 22px; }"
    f"QSlider::groove:horizontal {{ height: 6px; border-radius: 3px;"
    f" background: {LINE_STRONG}; }}"
    f"QSlider::sub-page:horizontal {{ height: 6px; border-radius: 3px; background: {BRAND_BLUE}; }}"



    f"QSlider::handle:horizontal {{ background: {SURFACE}; border: 1px solid {LINE_INPUT};"
    " width: 16px; height: 16px; margin: -6px 0; border-radius: 9px; }"
    f"QSlider::handle:horizontal:hover {{ border-color: {BRAND_BLUE}; }}"


    f"QSlider::sub-page:horizontal:disabled {{ background: {LINE_STRONG}; }}"
    f"QSlider::handle:horizontal:disabled {{ background: {FIELD};"
    f" border: 1px solid {LINE}; }}"
)






_SCROLL_AREA_QSS = (
    "QScrollArea { background: transparent; border: none; }"
    "QScrollArea > QWidget > QWidget { background: transparent; }"
    "QScrollBar:vertical { background: transparent; width: 8px;"
    " margin: 2px 2px 2px 0; border: none; }"
    f"QScrollBar::handle:vertical {{ background: {LINE_STRONG};"
    " border-radius: 3px; min-height: 24px; }"
    f"QScrollBar::handle:vertical:hover {{ background: {INK_3}; }}"
    "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {"
    " height: 0px; width: 0px; border: none; background: none; }"
    "QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {"
    " height: 0px; width: 0px; image: none; }"
    "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {"
    " background: transparent; }"
    "QScrollBar:horizontal { background: transparent; height: 8px;"
    " margin: 0 2px 2px 2px; border: none; }"
    f"QScrollBar::handle:horizontal {{ background: {LINE_STRONG};"
    " border-radius: 3px; min-width: 24px; }"
    f"QScrollBar::handle:horizontal:hover {{ background: {INK_3}; }}"
    "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {"
    " height: 0px; width: 0px; border: none; background: none; }"
    "QScrollBar::left-arrow:horizontal, QScrollBar::right-arrow:horizontal {"
    " height: 0px; width: 0px; image: none; }"
    "QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {"
    " background: transparent; }"
)


def apply_quiet_scrollbar(area, extra: str = "") -> None:

    try:
        area.setStyleSheet(_SCROLL_AREA_QSS + extra)
    except (AttributeError, RuntimeError):

        pass







_CARD_QSS = (
    f"QWidget#{{name}} {{{{ background-color: {SURFACE};"
    f" border: 1px solid {LINE}; border-radius: {RADIUS_CARD}px; }}}}"
)



_CARD_MARGINS = (12, 10, 12, 10)






_CARD_JOINED_QSS = (
    f"QWidget#{{name}} {{{{ background-color: {SURFACE};"
    f" border: 1px solid {LINE}; border-top: none;"
    " border-top-left-radius: 0px; border-top-right-radius: 0px;"
    f" border-bottom-left-radius: {RADIUS_CARD}px;"
    f" border-bottom-right-radius: {RADIUS_CARD}px; }}}}"
)






_SUBCARD_QSS = (
    f"QWidget#{{name}} {{{{ background-color: {INSET};"
    f" border: 1px solid {LINE}; border-radius: {RADIUS_CONTROL}px; }}}}"
)
_SUBCARD_MARGINS = (12, 10, 12, 10)






_CARD_CHILD_BTN_RESET_QSS = "QPushButton { border: none; }"





























_MSG_TINTS = {
    "neutral": (SURFACE, LINE),
    "info": (SURFACE, LINE),

    "armed": (SURFACE, ACCENT_BORDER_SOFT),
    "success": (SURFACE, LINE),
    "warning": ("rgba(245, 166, 35, 0.10)", "rgba(245, 166, 35, 0.40)"),
    "error": (RED_TINT, "rgba(229, 72, 77, 0.40)"),
    "error_transient": ("rgba(229, 72, 77, 0.20)", "rgba(229, 72, 77, 0.55)"),
    "premium": (SURFACE, LINE),
}




_PREMIUM_STAR = ""




_CLOUD_EMOJI = ""
_LAPTOP_EMOJI = ""







_MSG_GLYPHS = {
    "neutral": "",
    "info": "",
    "armed": "✎",
    "success": "✓",
    "warning": "⚠︎",
    "error": "✕",
    "error_transient": "✕",
    "premium": _PREMIUM_STAR,
}



_MSG_ICON_NAMES = {
    "neutral": "dash",
    "info": "lightbulb",
    "armed": "pencil",
    "success": "check",
    "warning": "warning",
    "error": "close",
    "error_transient": "close",
    "premium": "spark",
}


def msg_glyph_name(kind: str) -> str:

    return _MSG_ICON_NAMES.get(kind, "dash")


def msg_glyph_colour(kind: str) -> str:


    if kind.startswith("error"):
        return RED_INK
    return {"warning": ORANGE_TEXT, "armed": BRAND_BLUE,
            "success": GREEN_TEXT}.get(kind, INK_2)


def _msg_text(kind: str, text: str) -> str:


    glyph = _MSG_GLYPHS.get(kind, "")
    return f"{glyph}  {text}" if glyph else text






_REPORT_HREF = "terralab://report-problem"


def _error_banner_html(message: str, report_link_text: str) -> str:





    import html
    safe = html.escape(message or "").replace("\n", "<br>")
    body = msg_rich("error", safe, is_html=True)
    link = (
        f'<a href="{_REPORT_HREF}" style="color: {LINK_INK};'
        ' text-decoration: none;">'
        f"{html.escape(report_link_text)}</a>"
    )
    return f"{body}<br>{link}"


def _msg_label_qss(kind: str) -> str:

    from .font_scale import scale_qss_font_px

    fill, border = _MSG_TINTS[kind]
    text = RED_TEXT if kind.startswith("error") else INK
    return scale_qss_font_px(
        f"QLabel {{ background-color: {fill}; border: 1px solid {border};"
        f" border-radius: {RADIUS_CARD}px; padding: 10px 12px;"
        f" font-size: {FONT_BODY}px; color: {text}; }}"
    )


def _msg_card_qss(name: str, kind: str) -> str:



    fill, border = _MSG_TINTS[kind]
    text = RED_TEXT if kind.startswith("error") else INK
    return (
        f"QWidget#{name} {{ background-color: {fill};"
        f" border: 1px solid {border}; border-radius: {RADIUS_CARD}px; }}"
        f"QLabel {{ background: transparent; border: none; color: {text}; }}"
    )


def _micro_header(text: str, gloss: str | None = None):











    from qgis.PyQt.QtWidgets import QHBoxLayout, QLabel, QWidget

    w = QWidget()
    row = QHBoxLayout(w)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    from .font_scale import scale_qss_font_px

    lbl = QLabel(text)
    lbl.setStyleSheet(scale_qss_font_px(
        f"font-size: {FONT_MICRO}px; font-weight: 600; color: {INK_2};"
        " background: transparent; border: none;"))
    row.addWidget(lbl)
    if gloss:
        gl = QLabel(gloss)
        gl.setStyleSheet(scale_qss_font_px(
            f"font-size: {FONT_MICRO}px; color: {INK_3};"
            " background: transparent; border: none;"))
        row.addWidget(gl)
        w.gloss_label = gl
    row.addStretch(1)
    w.header_label = lbl
    return w


def _settings_zone(obj_name: str, title: str, gloss: str, items: list):










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



    from qgis.PyQt.QtWidgets import QFrame

    line = QFrame()
    line.setFrameShape(QFrame.Shape.NoFrame)
    line.setFixedHeight(1)
    line.setStyleSheet(f"background: {LINE}; border: none;")
    return line


def _choice_divider(text: str):






    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtWidgets import QHBoxLayout, QLabel, QWidget

    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 2, 0, 2)
    lay.setSpacing(8)
    label = QLabel(text)
    label.setStyleSheet(_scale_qss_font_px(
        f"font-size: {FONT_HINT}px; color: {INK_3};"
        " background: transparent; border: none;"))


    lay.addWidget(_card_divider(), 1, Qt.AlignmentFlag.AlignVCenter)
    lay.addWidget(label, 0, Qt.AlignmentFlag.AlignVCenter)
    lay.addWidget(_card_divider(), 1, Qt.AlignmentFlag.AlignVCenter)
    return row


def _step_dial(num: int, state: str = "todo"):







    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtWidgets import QLabel

    from .font_scale import scale_px_length, scale_qss_font_px

    lbl = QLabel("✓" if state == "done" else str(num))


    side = scale_px_length(20)
    radius = side // 2
    lbl.setFixedSize(side, side)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    if state == "active":
        qss = (f"background: {BRAND_BLUE}; color: {ON_ACCENT}; border: none;"
               f" border-radius: {radius}px; font-size: {FONT_HINT}px;"
               " font-weight: 700;")
    elif state == "done":
        qss = (f"background: transparent; color: {accent_ink()};"
               f" border: 1px solid {accent_ink()};"
               f" border-radius: {radius}px; font-size: {FONT_HINT}px;"
               " font-weight: 700;")
    else:
        qss = (f"background: transparent; color: {MUTED};"
               f" border: 1px solid {HAIRLINE_STRONG};"
               f" border-radius: {radius}px; font-size: {FONT_HINT}px;"
               " font-weight: 600;")
    lbl.setStyleSheet(scale_qss_font_px(qss))
    return lbl


def _sign_badge(symbol: str, color: str):


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
        f" font-weight: bold; font-size: {FONT_HINT}px;"))
    return badge






_SECTION_TOGGLE_QSS = (
    f"QPushButton {{ font-size: {FONT_BODY}px; color: {INK};"
    f" font-weight: 600; background-color: {SURFACE};"
    f" border: 1px solid {LINE}; border-radius: {RADIUS_CARD}px;"
    " padding: 8px 12px; text-align: left; }"
    f"QPushButton:hover {{ background-color: {HOVER};"
    f" border-color: {LINE_STRONG}; }}"
    f" QPushButton:disabled {{ color: {INK_3};"
    f" background-color: transparent; border-color: {LINE}; }}"
)





_SECTION_TOGGLE_OPEN_QSS = (
    f"QPushButton {{ font-size: {FONT_BODY}px; color: {INK};"
    f" font-weight: 600; background-color: {SURFACE};"
    f" border: 1px solid {LINE}; border-bottom: none;"
    f" border-top-left-radius: {RADIUS_CARD}px;"
    f" border-top-right-radius: {RADIUS_CARD}px;"
    " border-bottom-left-radius: 0px; border-bottom-right-radius: 0px;"
    " padding: 8px 12px; text-align: left; }"
    f"QPushButton:hover {{ background-color: {HOVER}; }}"
    f" QPushButton:disabled {{ color: {INK_3};"
    f" background-color: transparent; border-color: {LINE}; }}"
)




_FIELD_LABEL_QSS = (
    f"QLabel {{ font-size: {FONT_HINT}px; color: {MUTED};"
    " background: transparent; border: none; }"
)






_COMBO_THEME_QSS = (
    f"QComboBox {{ color: {INK}; background-color: {SURFACE};"
    f" border: 1px solid {LINE_INPUT}; border-radius: {RADIUS_CONTROL}px;"
    f" padding: 0 10px; min-height: {BTN_PX - 2}px; font-size: {FONT_BODY}px; }}"
    f"QComboBox:hover {{ border-color: {INK_2}; }}"
    f"QComboBox:focus, QComboBox:on {{ border-color: {ACCENT_BORDER}; }}"
    f"QComboBox:disabled {{ color: {INK_3}; background-color: {INSET};"
    f" border-color: {LINE}; }}"


    f"QComboBox QAbstractItemView {{ color: {INK};"
    f" background-color: {SURFACE}; outline: none;"
    f" border: 1px solid {LINE_STRONG}; border-radius: {RADIUS_CARD}px;"
    " padding: 6px;"
    f" selection-background-color: {ACCENT_TINT_ON}; selection-color: {INK}; }}"
)




_INPUT_THEME_QSS = (
    f"QLineEdit {{ border: 1px solid {LINE_INPUT};"
    f" border-radius: {RADIUS_CONTROL}px; padding: 6px 10px;"
    f" background: {SURFACE}; color: {INK}; }}"
    f"QLineEdit:hover {{ border-color: {INK_2}; }}"
    f"QLineEdit:focus {{ border: 1px solid {ACCENT_BORDER}; }}"
    f"QLineEdit:disabled {{ color: {INK_3}; background: {INSET};"
    f" border-color: {LINE}; }}"
)













_SPIN_FRAME_QSS = (
    f"QAbstractSpinBox {{ color: {INK}; background-color: {SURFACE};"
    f" border: 1px solid {LINE_INPUT}; border-radius: {RADIUS_CONTROL}px;"



    f" padding: 3px 18px 3px 8px; font-size: {FONT_BODY}px; }}"
    f"QAbstractSpinBox:hover {{ border-color: {INK_2}; }}"
    f"QAbstractSpinBox:focus {{ border-color: {ACCENT_BORDER}; }}"
    f"QAbstractSpinBox:disabled {{ color: {INK_3};"
    f" border-color: {LINE}; background: {INSET}; }}"
)



_SPIN_ARROW_INK = "#8c8c8c"
_SPIN_ARROW_BOX = 9

_SPIN_ICON_URLS: tuple[str, str] | None = None


def _spin_arrow_icons() -> tuple[str, str] | None:






    global _SPIN_ICON_URLS
    if _SPIN_ICON_URLS is not None:
        return _SPIN_ICON_URLS
    import os
    import tempfile

    box = _SPIN_ARROW_BOX
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{box}"'
            f' height="{box}" viewBox="0 0 {box} {box}">')
    stroke = (f'fill="none" stroke="{_SPIN_ARROW_INK}" stroke-width="1.4"'
              ' stroke-linecap="round" stroke-linejoin="round"')
    bodies = {
        "spin_up.svg": f'{head}<path d="M2 5.6 L4.5 3.1 L7 5.6" {stroke}/></svg>',
        "spin_down.svg": f'{head}<path d="M2 3.4 L4.5 5.9 L7 3.4" {stroke}/></svg>',
    }
    try:
        icon_dir = tempfile.mkdtemp(prefix="qgis_ai_seg_spin_")
        urls = []
        for name, body in bodies.items():
            path = os.path.join(icon_dir, name).replace("\\", "/")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(body)
            urls.append(path)
        _SPIN_ICON_URLS = (urls[0], urls[1])
    except OSError:
        return None
    return _SPIN_ICON_URLS


def combo_theme_qss() -> str:





    icons = _spin_arrow_icons()
    if icons is None:
        return _COMBO_THEME_QSS
    _, down = icons
    return (
        _COMBO_THEME_QSS
        + "QComboBox::drop-down { subcontrol-origin: border;"
          " subcontrol-position: center right; width: 20px;"
          " background: transparent; border: none; margin-right: 2px; }"
        + f'QComboBox::down-arrow {{ image: url("{down}");'
          f" width: {_SPIN_ARROW_BOX}px; height: {_SPIN_ARROW_BOX}px; }}"
    )


def spin_theme_qss() -> str:

    icons = _spin_arrow_icons()
    if icons is None:
        return _SPIN_FRAME_QSS
    up, down = icons
    return (
        _SPIN_FRAME_QSS
        + "QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {"
          " subcontrol-origin: border; width: 15px; height: 11px;"
          " background: transparent; border: none; margin-right: 3px; }"
          "QAbstractSpinBox::up-button { subcontrol-position: top right;"
          " margin-top: 2px; }"
          "QAbstractSpinBox::down-button { subcontrol-position: bottom right;"
          " margin-bottom: 2px; }"
        + f'QAbstractSpinBox::up-arrow {{ image: url("{up}");'
          f" width: {_SPIN_ARROW_BOX}px; height: {_SPIN_ARROW_BOX}px; }}"
        + f'QAbstractSpinBox::down-arrow {{ image: url("{down}");'
          f" width: {_SPIN_ARROW_BOX}px; height: {_SPIN_ARROW_BOX}px; }}"
        + "QAbstractSpinBox::up-arrow:disabled,"
          " QAbstractSpinBox::down-arrow:disabled { opacity: 80; }"
    )


def apply_input_theme_to_tree(root) -> None:







    try:
        from qgis.PyQt.QtWidgets import (
            QAbstractSpinBox,
            QComboBox,
            QLineEdit,
        )
    except ImportError:  # pragma: no cover
        return
    pairs = ((QAbstractSpinBox, spin_theme_qss()),
             (QComboBox, combo_theme_qss()),
             (QLineEdit, _INPUT_THEME_QSS))
    for cls, qss in pairs:
        try:
            found = root.findChildren(cls)
        except (AttributeError, RuntimeError):
            continue
        for w in found:
            try:


                parent = w.parent()
                if cls is QLineEdit and isinstance(
                        parent, (QAbstractSpinBox, QComboBox)):
                    continue
                if w.styleSheet().strip():
                    continue
                w.setStyleSheet(_scale_qss_font_px(qss))
            except (AttributeError, RuntimeError):
                continue







_PROGRESS_THIN_QSS = (
    f"QProgressBar {{ background: {FIELD}; border: none;"
    " border-radius: 2px; max-height: 3px; min-height: 3px; }"
    f"QProgressBar::chunk {{ background: {BRAND_BLUE}; border-radius: 2px; }}"
)





_INSTRUCTIONS_CARD_QSS = (
    "QLabel {"
    f" background-color: {SURFACE};"
    f" border: 1px solid {LINE};"
    f" border-radius: {RADIUS_CARD}px;"
    " padding: 10px 12px;"
    f" font-size: {FONT_BODY}px;"
    f" color: {INK};"
    "}"
)
_INSTRUCTIONS_HINT_QSS = (
    "QLabel {"
    " background: transparent;"
    " border: none;"
    " padding: 2px 0px;"
    f" font-size: {FONT_HINT}px;"
    f" color: {INK_2};"
    "}"
)

























_BTN_LABEL_WEIGHT = "font-weight: 600;"

_BTN_PRIMARY = (
    f"QPushButton {{ background-color: {ACCENT}; color: {ON_ACCENT};"
    f" padding: 0 16px; min-height: {BTN_PILL_PX}px; border: none;"
    f" border-radius: {RADIUS_PILL}px; font-size: {FONT_BODY}px;"
    f" {_BTN_LABEL_WEIGHT} }}"
    f"QPushButton:hover {{ background-color: {ACCENT_DARK}; color: {ON_ACCENT}; }}"
    f"QPushButton:pressed {{ background-color: {ACCENT_DARK}; }}"


    f"QPushButton:disabled {{ background-color: {DISABLED_FILL};"
    f" color: {DISABLED_INK}; }}"
)
_BTN_GREEN = _BTN_PRIMARY









_BTN_GREEN_STEP = (
    f"QPushButton {{ background-color: {ACCENT}; color: {ON_ACCENT};"
    f" padding: 0 18px; min-height: {BTN_PRIMARY_WIDE_PX}px; border: none;"
    f" border-radius: {RADIUS_PILL_WIDE}px; font-size: {FONT_BASE}px;"
    f" {_BTN_LABEL_WEIGHT} }}"
    f"QPushButton:hover {{ background-color: {ACCENT_DARK}; color: {ON_ACCENT}; }}"
    f"QPushButton:pressed {{ background-color: {ACCENT_DARK}; }}"
    f"QPushButton:disabled {{ background-color: {DISABLED_FILL};"
    f" color: {DISABLED_INK}; }}"
)













BTN_BLUE_FILL = BRAND_BLUE_HOVER
ON_BLUE_FILL = "#ffffff"
_BTN_BLUE_STEP = (
    f"QPushButton {{ background-color: {BTN_BLUE_FILL}; color: {ON_BLUE_FILL};"
    f" padding: 0 18px; min-height: {BTN_PRIMARY_WIDE_PX}px; border: none;"
    f" border-radius: {RADIUS_PILL_WIDE}px; font-size: {FONT_BASE}px;"
    f" {_BTN_LABEL_WEIGHT} }}"
    f"QPushButton:hover {{ background-color: {BRAND_BLUE};"
    f" color: {ON_BLUE_FILL}; }}"
    f"QPushButton:pressed {{ background-color: {BRAND_BLUE}; }}"
    f"QPushButton:disabled {{ background-color: {DISABLED_FILL};"
    f" color: {DISABLED_INK}; }}"
)



_BTN_GREEN_AUTH = (
    f"QPushButton {{ background-color: {ACCENT}; color: {ON_ACCENT};"
    f" border: none; border-radius: {RADIUS_PILL_WIDE}px;"
    f" padding: 0 18px; min-height: {BTN_PRIMARY_WIDE_PX}px;"
    f" font-size: {FONT_BASE}px; {_BTN_LABEL_WEIGHT} }}"
    f"QPushButton:hover {{ background-color: {ACCENT_DARK}; }}"
    f"QPushButton:pressed {{ background-color: {ACCENT_DARK}; }}"
    f"QPushButton:disabled {{ background-color: {DISABLED_FILL};"
    f" color: {DISABLED_INK}; }}"
)






_BTN_BLUE = (
    f"QPushButton {{ background-color: {BTN_BLUE_FILL}; color: {ON_BLUE_FILL};"
    f" padding: 0 16px; min-height: {BTN_PILL_PX}px; border: none;"
    f" border-radius: {RADIUS_PILL}px; font-size: {FONT_BODY}px;"
    f" {_BTN_LABEL_WEIGHT} }}"
    f"QPushButton:hover {{ background-color: {BRAND_BLUE};"
    f" color: {ON_BLUE_FILL}; }}"
    f"QPushButton:pressed {{ background-color: {BRAND_BLUE}; }}"
    f"QPushButton:disabled {{ background-color: {DISABLED_FILL};"
    f" color: {DISABLED_INK}; }}"
)


_BTN_BLUE_AUTH = (
    f"QPushButton {{ background-color: {BTN_BLUE_FILL}; color: {ON_BLUE_FILL};"
    f" border: none; border-radius: {RADIUS_PILL}px;"
    f" font-size: {FONT_BODY}px; {_BTN_LABEL_WEIGHT} }}"
    f"QPushButton:hover {{ background-color: {BRAND_BLUE}; }}"
    f"QPushButton:pressed {{ background-color: {BRAND_BLUE}; }}"
    f"QPushButton:disabled {{ background-color: {DISABLED_FILL};"
    f" color: {DISABLED_INK}; }}"
)




_UPDATE_LATER_STYLE = (
    f"QPushButton {{ background: transparent; color: {MUTED};"
    f" border: none; border-radius: {RADIUS_CONTROL}px;"
    f" font-size: {FONT_HINT}px; padding: 4px 8px; }}"
    f"QPushButton:hover {{ color: {INK}; text-decoration: underline; }}"
)




_BTN_BLUE_PRIMARY = _BTN_BLUE + "QPushButton { padding: 0 18px; }"






_BTN_START_FONT_PX = 14





_BTN_START_WEIGHT = "font-weight: 700;"






_BTN_START_FULL_WIDTH_PAD_PX = 6


def _btn_start_qss(base: str, full_width: bool = False) -> str:





    pad = (f" padding: 0 {_BTN_START_FULL_WIDTH_PAD_PX}px;" if full_width
           else "")
    return base + (f"QPushButton {{ font-size: {_BTN_START_FONT_PX}px;"
                   f" {_BTN_START_WEIGHT}{pad} }}")





_BTN_GHOST = (
    f"QPushButton {{ background-color: {SURFACE}; color: {INK};"
    f" padding: 0 14px; min-height: {BTN_PILL_PX}px;"
    f" border-radius: {RADIUS_PILL}px; font-size: {FONT_BODY}px;"
    f" {_BTN_LABEL_WEIGHT}"
    f" border: 1px solid {LINE_STRONG}; }}"
    f"QPushButton:hover {{ background-color: {HOVER}; }}"
    f"QPushButton:pressed {{ background-color: {HOVER_ON}; }}"
    f"QPushButton:disabled {{ background-color: transparent;"
    f" border: 1px solid {LINE}; color: {INK_3}; }}"
)



_BTN_BLUE_OUTLINE = (
    f"QPushButton {{ background-color: transparent; color: {BRAND_BLUE};"
    f" padding: 0 14px; min-height: {BTN_PILL_PX}px;"
    f" border: 1px solid {BRAND_BLUE}; border-radius: {RADIUS_PILL}px;"
    f" font-size: {FONT_BODY}px; {_BTN_LABEL_WEIGHT} }}"
    f"QPushButton:hover {{ background-color: {ACCENT_TINT}; }}"
    f"QPushButton:pressed {{ background-color: {ACCENT_TINT_ON}; }}"
    f"QPushButton:disabled {{ color: {INK_3};"
    f" border-color: {LINE}; }}"
)




_BTN_RED_OUTLINE = (
    f"QPushButton {{ background-color: {SURFACE}; color: {RED_INK};"
    f" border: 1px solid {LINE_STRONG}; border-radius: {RADIUS_PILL}px;"
    f" font-size: {FONT_BODY}px; {_BTN_LABEL_WEIGHT}"
    f" padding: 0 14px; min-height: {BTN_PILL_PX}px; }}"
    f"QPushButton:hover {{ background-color: {RED_TINT};"
    f" border-color: {RED_INK}; }}"
    f"QPushButton:pressed {{ background-color: {RED_TINT}; }}"
    f"QPushButton:disabled {{ color: {INK_3};"
    f" border-color: {LINE}; }}"
)






_BTN_LINK = (
    f"QPushButton {{ background: transparent; border: none; color: {INK_2};"
    f" font-size: {FONT_HINT}px; font-weight: 500; text-align: left; padding: 2px 0px; }}"
    f"QPushButton:hover {{ color: {INK}; text-decoration: underline; }}"
    f"QPushButton:disabled {{ color: {INK_3}; text-decoration: none; }}"
)
_BTN_LINK_MUTED = (
    "QPushButton { background: transparent; border: none;"
    f" color: {INK_2}; font-size: {FONT_HINT}px; padding: 4px 8px;"
    f" border-radius: {RADIUS_CONTROL}px; }}"
    f"QPushButton:hover {{ color: {RED_INK}; background: {HOVER}; }}"
    f"QPushButton:pressed {{ color: {RED_INK}; background: {HOVER_ON}; }}"
    f"QPushButton:disabled {{ color: {INK_3}; background: transparent; }}"
)




_BTN_LINK_QUIET = (
    "QPushButton { background: transparent; border: none;"
    f" color: {INK_2}; font-size: {FONT_HINT}px; padding: 4px 8px;"
    f" border-radius: {RADIUS_CONTROL}px; }}"
    f"QPushButton:hover {{ color: {INK}; background: {HOVER};"
    " text-decoration: underline; }"
    f"QPushButton:pressed {{ color: {INK}; background: {HOVER_ON}; }}"
    f"QPushButton:disabled {{ color: {INK_3}; background: transparent;"
    " text-decoration: none; }"
)






_BTN_LINK_STRONG = (
    "QPushButton { background: transparent; border: none;"
    f" color: {INK}; font-size: {FONT_HINT}px; font-weight: 600;"
    " padding: 4px 8px; text-decoration: underline; }"
    f"QPushButton:hover {{ color: {RED_INK}; }}"
    f"QPushButton:disabled {{ color: {INK_3}; }}"
)



_CHIP_QSS = (
    f"QPushButton {{ background: {SURFACE};"
    f" border: 1px solid {LINE};"
    f" border-radius: {RADIUS_CONTROL}px;"
    f" color: {INK}; font-size: {FONT_BODY}px; text-align: left;"
    " padding: 7px 12px; }"
    f"QPushButton:hover {{ background: {HOVER}; border-color: {LINE_STRONG}; }}"
    f"QPushButton:pressed {{ background: {HOVER_ON}; }}"
    f"QPushButton:disabled {{ color: {INK_3};"
    f" background: transparent; border-color: {LINE}; }}"
)







_BTN_CHIP = (
    f"QPushButton {{ background: {SURFACE};"
    f" border: 1px solid {LINE}; border-radius: {RADIUS_CONTROL}px;"
    f" padding: 5px 10px; font-size: {FONT_BODY}px; color: {INK};"
    " font-weight: 500; }"
    f"QPushButton:hover {{ background: {HOVER}; border-color: {LINE_STRONG}; }}"
    f"QPushButton:pressed {{ background: {HOVER_ON}; border-color: {LINE_STRONG}; }}"
    f"QPushButton:disabled {{ color: {INK_3};"
    f" background: transparent; border-color: {LINE}; }}"
)






_BTN_TILE = (
    f"QPushButton {{ background: {SURFACE};"
    f" border: 1px solid {LINE}; border-radius: {RADIUS_CARD}px;"
    f" color: {INK}; font-size: {FONT_BODY}px; text-align: left;"
    f" {_BTN_LABEL_WEIGHT}"
    " padding: 10px 12px; }"
    f"QPushButton:hover {{ background: {HOVER};"
    f" border-color: {LINE_STRONG}; }}"
    f"QPushButton:pressed {{ background: {HOVER_ON}; }}"
    f"QPushButton:disabled {{ color: {INK_3};"
    f" background: transparent; border-color: {LINE}; }}"
)



_BTN_TILE_ACTIVE = (
    f"QPushButton {{ background: {ACCENT_TINT};"
    f" border: 1px solid {BRAND_BLUE}; border-radius: {RADIUS_CARD}px;"
    f" color: {INK}; font-size: {FONT_BODY}px; text-align: left;"
    " font-weight: 600; padding: 10px 12px; }"
    f"QPushButton:hover {{ background: {ACCENT_TINT_ON}; }}"
    f"QPushButton:disabled {{ color: {INK_3}; background: transparent;"
    f" border-color: {LINE}; }}"
)













SEGMENT_ON = HOVER_ON if DARK_UI else SURFACE


def _segmented_switch_qss(name: str) -> str:






    return (
        f"QFrame#{name} {{"
        f"  background: {FIELD};"
        "  border: none;"
        f"  border-radius: {RADIUS_CARD}px;"
        "}"
        "QPushButton {"
        "  background: transparent;"
        "  border: 1px solid transparent;"
        f"  border-radius: {RADIUS_CONTROL}px;"
        "  padding: 5px 0px;"
        f"  font-size: {FONT_BODY}px;"
        "  font-weight: 500;"
        f"  color: {INK_2};"
        "}"
        "QPushButton:hover {"
        f"  color: {INK};"
        "}"

        f"QPushButton:pressed:!checked {{ background: {HOVER_ON}; }}"


        "QPushButton:checked {"
        f"  background: {SEGMENT_ON};"
        f"  color: {INK};"
        "  font-weight: 600;"
        f"  border: 1px solid {LINE_STRONG};"
        "}"

        f"QPushButton:focus {{ border: 1px solid {ACCENT_BORDER}; }}"
        f"QPushButton:disabled {{ color: {INK_3}; }}"
    )


_METHOD_SWITCH_QSS = _segmented_switch_qss("methodSwitchFrame")


def _mode_tabs_qss(name: str) -> str:











    return (
        f"QFrame#{name} {{"
        "  background: transparent;"


        f"  border-bottom: 1px solid {LINE_STRONG};"
        "}"
        "QPushButton {"
        "  background: transparent;"
        "  border: none;"
        "  border-bottom: 2px solid transparent;"
        "  padding: 0px;"
        f"  font-size: {FONT_BODY}px;"
        "  font-weight: 500;"
        f"  color: {INK_2};"
        "}"
        f"QPushButton:hover {{ color: {INK}; }}"
        f"QPushButton:pressed {{ color: {INK}; }}"
        "QPushButton:checked { font-weight: 600; }"


        f"QPushButton:focus {{ background: {ACCENT_TINT};"
        f" border-radius: {RADIUS_CONTROL}px; }}"
        f"QPushButton:disabled {{ color: {INK_3}; }}"
    )



















_ENGINE_CARD_QSS = (
    "QPushButton {"
    f"  background: {SURFACE};"
    f"  border: 1px solid {LINE};"
    f"  border-radius: {RADIUS_CARD}px;"
    "  text-align: left;"
    "}"
    "QPushButton:hover {"
    f"  background: {HOVER};"
    f"  border-color: {LINE_STRONG};"
    "}"


    "QPushButton:checked {"
    f"  background: {ACCENT_TINT};"
    f"  border: 1px solid {BRAND_BLUE};"
    "}"
    f"QPushButton:checked:hover {{ background: {ACCENT_TINT_ON}; }}"
    "QLabel { background: transparent; border: none; }"
)













_ENGINE_CARD_TITLE_QSS = (
    f"font-size: {FONT_BODY}px; font-weight: 600; color: {INK};")
_ENGINE_CARD_GLOSS_QSS = f"font-size: {FONT_HINT}px; color: {INK_2};"
_ENGINE_CARD_TITLE_ON_QSS = _ENGINE_CARD_TITLE_QSS
_ENGINE_CARD_GLOSS_ON_QSS = _ENGINE_CARD_GLOSS_QSS






_BTN_REMOVE_ROW = (
    "QPushButton { background: transparent; border: none; text-align: left;"
    f" color: {INK_2}; font-size: {FONT_BODY}px; padding: 4px 0px; }}"
    f"QPushButton:hover {{ color: {RED_INK}; }}"

    f"QPushButton:pressed {{ color: {RED_INK}; text-decoration: underline; }}"
    f"QPushButton:disabled {{ color: {INK_3}; }}"
)







def _btn_toggle_qss(rgb: tuple[int, int, int], text: str, armed_text: str,
                    weight: int = 700, quiet: bool = False,
                    filled: bool = False) -> str:











    r, g, b = rgb
    solid = f"rgb({r}, {g}, {b})"
    dark = f"rgb({int(r * 0.8)}, {int(g * 0.8)}, {int(b * 0.8)})"
    if quiet:
        rest = (
            f"QPushButton {{ background: {SURFACE}; color: {INK};"
            f" border: 1px solid {LINE_STRONG};"
            f" border-radius: {RADIUS_PILL}px;"
            f" padding: 0 14px; min-height: {BTN_PILL_PX}px;"
            f" font-size: {FONT_BODY}px; }}"
            f"QPushButton:hover {{ background: rgba({r}, {g}, {b}, 0.12);"
            f" border-color: rgba({r}, {g}, {b}, 0.45); }}"
            f"QPushButton:pressed {{ background: rgba({r}, {g}, {b}, 0.22); }}"
        )
    elif filled:
        rest = (
            f"QPushButton {{ background: {solid}; color: {ON_ACCENT};"
            f" border: none; border-radius: {RADIUS_PILL}px;"
            f" padding: 0 16px; min-height: {BTN_PILL_PX}px;"
            f" font-size: {FONT_BODY}px; font-weight: {weight}; }}"
            f"QPushButton:hover {{ background: {dark}; }}"
            f"QPushButton:pressed {{ background: {dark}; }}"
        )
    else:
        rest = (
            f"QPushButton {{ background: transparent; color: {text};"
            f" border: 1px solid rgba({r}, {g}, {b}, 0.45);"
            f" border-radius: {RADIUS_PILL}px;"
            f" padding: 0 16px; min-height: {BTN_PILL_PX}px;"
            f" font-size: {FONT_BODY}px; font-weight: {weight}; }}"
            f"QPushButton:hover {{ background: rgba({r}, {g}, {b}, 0.14); }}"
            f"QPushButton:pressed {{ background: rgba({r}, {g}, {b}, 0.24); }}"
        )
    combined = rest
    if filled:


        combined += (f'QPushButton[armed="true"] {{ background: {dark};'
                     f" color: {ON_ACCENT}; border: none; }}")
    else:
        combined += (f'QPushButton[armed="true"] {{ background: {solid};'
                     f" color: {armed_text}; border: 1px solid {solid}; }}")
    combined += "QPushButton:disabled { background: transparent;"
    combined += f" color: {INK_3}; border-color: {LINE}; }}"

    combined += _FOCUS_RING_QSS
    from .font_scale import scale_qss_font_px

    return scale_qss_font_px(combined)












def _btn_hint_action_qss(rgb: tuple[int, int, int]) -> str:
    from .font_scale import scale_qss_font_px

    del rgb


    return scale_qss_font_px(
        f"QToolButton {{ background: {SURFACE}; color: {INK};"
        f" border: 1px solid {LINE_STRONG}; border-radius: {RADIUS_CONTROL}px;"
        f" padding: 0 12px; min-height: {BTN_SMALL_PX - 2}px;"
        f" font-size: {FONT_BODY}px; font-weight: 600; }}"
        f"QToolButton:hover {{ background: {HOVER}; }}"
        f"QToolButton:pressed {{ background: {HOVER_ON}; }}"
        f"QToolButton:focus {{ border: 2px solid {ACCENT_BORDER}; }}"
        f"QToolButton:disabled {{ color: {INK_3}; background: transparent;"
        f" border-color: {LINE}; }}"
    )





_BTN_GRAY = (
    f"QPushButton {{ background-color: {SURFACE}; color: {INK};"
    f" padding: 0 12px; min-height: {BTN_PILL_PX}px;"
    f" border: 1px solid {LINE_STRONG};"
    f" border-radius: {RADIUS_PILL}px; font-size: {FONT_BODY}px;"
    f" {_BTN_LABEL_WEIGHT} }}"
    f"QPushButton:hover {{ background-color: {HOVER}; }}"
    f"QPushButton:pressed {{ background-color: {HOVER_ON}; }}"
    f"QPushButton:disabled {{ background-color: transparent;"
    f" border-color: {LINE}; color: {DISABLED_INK}; }}"
)

_BTN_RED = (
    f"QPushButton {{ background-color: {SURFACE}; color: {RED_INK};"
    f" padding: 0 14px; min-height: {BTN_PILL_PX}px;"
    f" border: 1px solid {LINE_STRONG}; border-radius: {RADIUS_PILL}px;"
    f" font-size: {FONT_BODY}px; {_BTN_LABEL_WEIGHT} }}"
    f"QPushButton:hover {{ background-color: {RED_TINT};"
    f" border-color: {RED_INK}; }}"
    f"QPushButton:pressed {{ background-color: {RED_TINT}; }}"
    f"QPushButton:disabled {{ background-color: transparent;"
    f" border-color: {LINE}; color: {INK_3}; }}"
)

_BTN_EXPORT_READY = _BTN_PRIMARY




_BTN_PAIR_NEUTRAL = _BTN_GHOST
_BTN_PAIR_CANCEL = _BTN_GHOST


_MENU_QSS = (
    f"QMenu {{ background: {SURFACE}; border: 1px solid {LINE_STRONG};"
    f" border-radius: {RADIUS_CARD}px; padding: 6px; }}"
    "QMenu::item { background: transparent; padding: 7px 8px;"
    f" border-radius: {RADIUS_ROW}px; color: {INK_2}; font-size: {FONT_BODY}px; }}"
    f"QMenu::item:selected {{ background: {HOVER}; color: {INK}; }}"
    f"QMenu::item:disabled {{ color: {INK_3}; }}"
    f"QMenu::separator {{ height: 1px; background: {LINE};"
    " margin: 4px 8px; }"
)



_TOOLTIP_QSS = (
    f"QToolTip {{ background: {TOOLTIP_BG}; color: {TOOLTIP_FG};"
    f" border: 1px solid {LINE_STRONG}; border-radius: {RADIUS_CHIP}px;"
    f" padding: 4px 8px; font-size: {FONT_HINT}px; }}"
)










_FOCUS_RING_QSS = f"QPushButton:focus {{ border: 2px solid {ACCENT_BORDER}; }}"
for _ring_name in (
    "_BTN_PRIMARY", "_BTN_GREEN", "_BTN_GREEN_STEP", "_BTN_GREEN_AUTH",
    "_BTN_BLUE", "_BTN_BLUE_STEP", "_BTN_BLUE_AUTH", "_BTN_BLUE_PRIMARY",
    "_BTN_GHOST",
    "_BTN_BLUE_OUTLINE", "_BTN_RED_OUTLINE", "_BTN_LINK", "_BTN_LINK_MUTED",
    "_BTN_LINK_QUIET", "_BTN_LINK_STRONG", "_CHIP_QSS", "_BTN_CHIP", "_BTN_TILE",
    "_BTN_TILE_ACTIVE", "_ENGINE_CARD_QSS", "_BTN_REMOVE_ROW", "_BTN_GRAY",
    "_BTN_RED", "_BTN_EXPORT_READY", "_BTN_PAIR_NEUTRAL", "_BTN_PAIR_CANCEL",
    "_SECTION_TOGGLE_QSS", "_SECTION_TOGGLE_OPEN_QSS", "_UPDATE_LATER_STYLE",
):
    globals()[_ring_name] = globals()[_ring_name] + _FOCUS_RING_QSS
del _ring_name


def apply_keyboard_focus_policy(root) -> None:





    try:
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtWidgets import QAbstractButton

        for button in root.findChildren(QAbstractButton):
            if button.focusPolicy() == Qt.FocusPolicy.StrongFocus:
                button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
    except (AttributeError, RuntimeError, TypeError):
        pass  # nosec B110


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
    "_BTN_LINK_QUIET",
    "_BTN_LINK_STRONG",
    "_CHIP_QSS",
    "_BTN_CHIP",
    "_BTN_TILE",
    "_BTN_TILE_ACTIVE",
    "_BTN_GREEN_STEP",
    "_BTN_PRIMARY",
    "_BTN_GREEN",
    "_BTN_GREEN_AUTH",
    "_BTN_GHOST",
    "_BTN_RED",
    "_BTN_RED_OUTLINE",
    "_COMBO_THEME_QSS",
    "_FIELD_LABEL_QSS",
    "_INPUT_THEME_QSS",
    "_MENU_QSS",
    "_TOOLTIP_QSS",
    "_BTN_GRAY",
    "_BTN_PAIR_NEUTRAL",
    "_BTN_PAIR_CANCEL",
    "_UPDATE_LATER_STYLE",
    "_METHOD_SWITCH_QSS",
    "_BTN_REMOVE_ROW",
):
    globals()[_qss_name] = _scale_qss_font_px(globals()[_qss_name])
del _qss_name




_BTN_CANCEL_DETECTION = _BTN_LINK_MUTED + "QPushButton { padding-left: 0px; }"



__all__ = [
    "ACCENT_BORDER",
    "ACCENT_TINT",
    "BRAND_BLUE",
    "BRAND_BLUE_HOVER",
    "BRAND_GREEN",
    "BRAND_GREEN_TEXT",
    "BRAND_RED",
    "BRAND_RED_HOVER",
    "BTN_GREEN",
    "BTN_GREEN_HOVER",
    "BTN_PILL_PX",
    "ERROR_TEXT",
    "FONT_BASE",
    "FONT_BODY",
    "FONT_HINT",
    "FONT_MICRO",
    "HAIRLINE",
    "HAIRLINE_STRONG",
    "MUTED",
    "RADIUS_CARD",
    "RADIUS_PANEL",
    "RADIUS_PILL",
    "RADIUS_ROW",
    "SUCCESS_TEXT",
    "TINT",
    "TINT_ON",
    "_BTN_BLUE",
    "_BTN_BLUE_AUTH",
    "_BTN_BLUE_OUTLINE",
    "_BTN_BLUE_PRIMARY",
    "_BTN_BLUE_STEP",
    "_BTN_CHIP",
    "_BTN_CANCEL_DETECTION",
    "_BTN_EXPORT_READY",
    "_BTN_GHOST",
    "_BTN_GRAY",
    "_BTN_GREEN",
    "_BTN_GREEN_AUTH",
    "_BTN_GREEN_STEP",
    "_BTN_LINK",
    "_BTN_LINK_MUTED",
    "_BTN_LINK_QUIET",
    "_BTN_LINK_STRONG",
    "_BTN_PAIR_CANCEL",
    "_BTN_PAIR_NEUTRAL",
    "_BTN_RED",
    "_BTN_RED_OUTLINE",
    "_BTN_REMOVE_ROW",
    "_BTN_TILE",
    "_BTN_TILE_ACTIVE",
    "_CARD_CHILD_BTN_RESET_QSS",
    "_CARD_JOINED_QSS",
    "_CARD_MARGINS",
    "_CARD_QSS",
    "_CHIP_QSS",
    "_CLOUD_EMOJI",
    "_ENGINE_CARD_GLOSS_ON_QSS",
    "_ENGINE_CARD_GLOSS_QSS",
    "_ENGINE_CARD_QSS",
    "_ENGINE_CARD_TITLE_ON_QSS",
    "_ENGINE_CARD_TITLE_QSS",
    "_FIELD_LABEL_QSS",
    "_INPUT_THEME_QSS",
    "_INSTRUCTIONS_CARD_QSS",
    "_INSTRUCTIONS_HINT_QSS",
    "_LAPTOP_EMOJI",
    "_MENU_QSS",
    "_TOOLTIP_QSS",
    "RED_INK",
    "msg_glyph_colour",
    "msg_glyph_name",
    "_METHOD_SWITCH_QSS",
    "_MSG_GLYPHS",
    "_PREMIUM_STAR",
    "_PROGRESS_THIN_QSS",
    "_REPORT_HREF",
    "_REVIEW_CONF_MAX",
    "_REVIEW_CONF_MIN",
    "_REVIEW_CONF_SPIN_MIN",
    "_REVIEW_CONF_STEP",
    "_SECTION_TOGGLE_OPEN_QSS",
    "_SECTION_TOGGLE_QSS",
    "_SLIDER_QSS",
    "_SUBCARD_MARGINS",
    "_SUBCARD_QSS",
    "_UPDATE_LATER_STYLE",
    "_btn_hint_action_qss",
    "_btn_start_qss",
    "_btn_toggle_qss",
    "_card_divider",
    "_choice_divider",
    "_error_banner_html",
    "_micro_header",
    "_msg_card_qss",
    "_msg_label_qss",
    "_msg_text",
    "_settings_zone",
    "_snap_review_conf",
    "_step_dial",
    "accent_ink",
    "apply_input_theme_to_tree",
    "apply_quiet_scrollbar",
    "combo_theme_qss",
]










_BTN_SETTINGS_INK = (
    "QPushButton { background: palette(text); color: palette(base); border: none;"
    f" border-radius: {RADIUS_CONTROL}px; padding: 7px 14px;"
    f" font-size: {FONT_BODY}px; {_BTN_LABEL_WEIGHT} }}"
    f"QPushButton:hover {{ background: {INK_HOVER}; }}"
    f"QPushButton:pressed {{ background: {INK_2}; }}"
    f"QPushButton:disabled {{ background: {HOVER_ON}; color: {INK_3}; }}"
)






_BTN_SETTINGS_ACCENT = (
    f"QPushButton {{ background: {ACCENT}; color: {ON_ACCENT}; border: none;"
    f" border-radius: {RADIUS_CONTROL}px; padding: 7px 16px;"
    f" font-size: {FONT_BODY}px; {_BTN_LABEL_WEIGHT} }}"
    f"QPushButton:hover {{ background: {ACCENT_DARK}; }}"
    f"QPushButton:pressed {{ background: {ACCENT_DARK}; }}"


    f"QPushButton:disabled {{ background: {DISABLED_FILL}; color: {DISABLED_INK}; }}"
)

_BTN_SETTINGS_GHOST = (
    f"QPushButton {{ background: transparent; color: {INK};"
    f" border: 1px solid {LINE_STRONG}; border-radius: {RADIUS_CONTROL}px;"
    f" padding: 6px 14px; font-size: {FONT_BODY}px; {_BTN_LABEL_WEIGHT} }}"
    f"QPushButton:hover {{ background: {ACCENT_TINT}; border-color: {ACCENT_BORDER_SOFT}; }}"
    f"QPushButton:pressed {{ background: {ACCENT_TINT_ON}; }}"
    f"QPushButton:disabled {{ color: {INK_3}; border-color: {LINE}; }}"
)


_BTN_SETTINGS_DANGER = (
    f"QPushButton {{ background: transparent; color: {RED_INK};"
    f" border: 1px solid {LINE_STRONG}; border-radius: {RADIUS_CONTROL}px;"
    f" padding: 6px 14px; font-size: {FONT_BODY}px; {_BTN_LABEL_WEIGHT} }}"
    f"QPushButton:hover {{ background: {RED_TINT};"
    f" border-color: {RED_INK}; }}"
    f"QPushButton:pressed {{ background: {RED_TINT}; }}"
    f"QPushButton:disabled {{ color: {INK_3}; border-color: {LINE}; }}"
)

_BTN_SETTINGS_TEXT = (
    f"QPushButton {{ background: transparent; color: {INK_2}; border: none;"
    f" border-radius: {RADIUS_CONTROL}px; padding: 7px 10px; font-size: {FONT_BODY}px; }}"
    f"QPushButton:hover {{ color: {INK}; background: {HOVER}; }}"
    f"QPushButton:pressed {{ background: {HOVER_ON}; }}"
    f"QPushButton:disabled {{ color: {INK_3}; background: transparent; }}"
)


_BTN_SETTINGS_RAIL_CTA = (
    f"QPushButton {{ background: {ACCENT}; color: {ON_ACCENT}; border: none;"
    f" border-radius: {RADIUS_CONTROL}px; margin: 4px 12px 6px 14px; padding: 7px 10px;"
    f" font-size: {FONT_BODY}px; font-weight: 700; }}"
    f"QPushButton:hover {{ background: {ACCENT_DARK}; }}"
    f"QPushButton:pressed {{ background: {ACCENT_DARK}; }}"
    f"QPushButton:disabled {{ background: {DISABLED_FILL}; color: {DISABLED_INK}; }}"
)

for _qss_name in (
    "_BTN_SETTINGS_INK",
    "_BTN_SETTINGS_ACCENT",
    "_BTN_SETTINGS_GHOST",
    "_BTN_SETTINGS_DANGER",
    "_BTN_SETTINGS_TEXT",
    "_BTN_SETTINGS_RAIL_CTA",
):

    globals()[_qss_name] = _scale_qss_font_px(globals()[_qss_name])
del _qss_name

__all__ += [
    "_BTN_SETTINGS_ACCENT",
    "_BTN_SETTINGS_DANGER",
    "_BTN_SETTINGS_GHOST",
    "_BTN_SETTINGS_INK",
    "_BTN_SETTINGS_RAIL_CTA",
    "_BTN_SETTINGS_TEXT",
]







_MSG_GLYPH_URIS: dict = {}


def msg_glyph_html(kind: str, px: int = 12) -> str:








    key = (kind, px)
    if key not in _MSG_GLYPH_URIS:
        uri = ""
        try:
            from qgis.PyQt.QtCore import QBuffer, QByteArray, QIODevice
            from qgis.PyQt.QtGui import QColor

            from ..icons import render_pixmap

            if kind in _MSG_ICON_NAMES and kind != "neutral":
                pixmap = render_pixmap(msg_glyph_name(kind),
                                       QColor(msg_glyph_colour(kind)), px, 2.0)
                data = QByteArray()
                buffer = QBuffer(data)
                buffer.open(QIODevice.OpenModeFlag.WriteOnly)
                pixmap.save(buffer, "PNG")
                uri = "data:image/png;base64," + bytes(data.toBase64()).decode("ascii")
        except Exception:  # noqa: BLE001
            uri = ""
        _MSG_GLYPH_URIS[key] = uri
    uri = _MSG_GLYPH_URIS[key]
    if not uri:
        return ""
    return f'<img src="{uri}" width="{px}" height="{px}">&nbsp;&nbsp;'


def msg_rich(kind: str, text: str, is_html: bool = False, px: int = 16) -> str:








    import html

    body = text if is_html else html.escape(text or "").replace("\n", "<br>")
    glyph = msg_glyph_html(kind, px)
    if not glyph:
        return body
    return ('<table cellspacing="0" cellpadding="0"><tr>'
            f'<td valign="top" style="padding-top: 1px;">{glyph}</td>'
            f'<td valign="top">{body}</td></tr></table>')


def no_break_words(text: str) -> str:





    return (text or "").replace("Semi-Auto", "Semi\u2011Auto")


def _settings_section(title: str, gloss: str, items: list, divider: bool = True):






    from qgis.PyQt.QtWidgets import QLayout, QVBoxLayout, QWidget

    section = QWidget()
    col = QVBoxLayout(section)
    col.setContentsMargins(0, 0, 0, 0)
    col.setSpacing(SPACE_CARD)
    if divider:
        col.addWidget(_card_divider())
        col.addSpacing(2)
    col.addWidget(_micro_header(title, gloss))
    for item in items:
        if isinstance(item, QLayout):
            col.addLayout(item)
        else:
            col.addWidget(item)
    return section





_DISCLOSURE_ROW_QSS = _scale_qss_font_px(
    f"QPushButton {{ font-size: {FONT_BODY}px; color: {INK};"
    " font-weight: 600; background: transparent; border: none;"
    f" border-radius: {RADIUS_CONTROL}px; padding: 6px 4px; text-align: left; }}"
    f"QPushButton:hover {{ background-color: {HOVER}; }}"
    f"QPushButton:disabled {{ color: {INK_3}; }}"
    + _FOCUS_RING_QSS
)


_HINT_LINE_QSS = _scale_qss_font_px(
    f"font-size: {FONT_HINT}px; color: {INK_2}; background: transparent;")




_SECTION_LABEL_QSS = _scale_qss_font_px(
    f"font-size: {FONT_MICRO}px; font-weight: 600;"
    f" color: {INK_3}; background: transparent; border: none;")



FONT_HERO = FONT_BASE + 4
_HERO_TITLE_QSS = _scale_qss_font_px(
    f"font-size: {FONT_HERO}px; font-weight: 600; color: {INK};"
    " background: transparent; border: none;")

__all__ += [
    "GREEN_TEXT",
    "LINE_INPUT",
    "LINK_INK",
    "ORANGE_TEXT",
    "RED_TEXT",
    "FONT_HERO",
    "_HERO_TITLE_QSS",
    "_SECTION_LABEL_QSS",
    "_DISCLOSURE_ROW_QSS",
    "_HINT_LINE_QSS",
    "_settings_section",
    "apply_keyboard_focus_policy",
    "msg_glyph_html",
    "msg_rich",
    "no_break_words",
]














_BTN_SETTINGS_STEP_INK = LINK_INK
_BTN_SETTINGS_STEP = _scale_qss_font_px(
    f"QPushButton {{ background: transparent; color: {_BTN_SETTINGS_STEP_INK};"
    f" border: 1px solid {BRAND_BLUE}; border-radius: {RADIUS_CONTROL}px;"
    f" padding: 6px 14px; font-size: {FONT_BODY}px; font-weight: 600; }}"
    f"QPushButton:hover {{ background: {ACCENT_TINT}; border-color: {BRAND_BLUE}; }}"
    f"QPushButton:pressed {{ background: {ACCENT_TINT_ON}; }}"
    f"QPushButton:disabled {{ color: {INK_3}; border-color: {LINE}; }}"
)

__all__ += ["_BTN_SETTINGS_STEP"]







_CATEGORY_HUES = {

    "green": ("#43a047", "#2e7d32", "#66bb6a", "67, 160, 71"),
    "leaf": ("#8bac27", "#437010", "#a3c644", "139, 172, 39"),
    "amber": ("#e0952b", "#96590c", "#f0b252", "224, 149, 43"),
    "teal": ("#1f9e96", "#0b6e68", "#3cc3ba", "31, 158, 150"),
    "coral": ("#e0603f", "#b23c20", "#f28a6c", "224, 96, 63"),
    "violet": ("#7c6cd0", "#5a48b8", "#a597ee", "124, 108, 208"),
    "sky": ("#3e86d6", "#1f5fa8", "#6ea8ec", "62, 134, 214"),
}
CATEGORY_NAMES = tuple(_CATEGORY_HUES)


def category_fill(name: str) -> str:

    return _CATEGORY_HUES.get(name, _CATEGORY_HUES["green"])[0]


def category_ink(name: str) -> str:

    fill, light, dark, _ = _CATEGORY_HUES.get(name, _CATEGORY_HUES["green"])
    return dark if DARK_UI else light


def category_tint(name: str, strong: bool = False) -> str:

    rgb = _CATEGORY_HUES.get(name, _CATEGORY_HUES["green"])[3]
    alpha = (0.26 if strong else 0.18) if DARK_UI else (0.20 if strong else 0.12)
    return f"rgba({rgb}, {alpha})"


def category_line(name: str) -> str:

    rgb = _CATEGORY_HUES.get(name, _CATEGORY_HUES["green"])[3]
    return f"rgba({rgb}, {0.40 if DARK_UI else 0.35})"


def gauge_category(fraction_left: float) -> str:

    if fraction_left <= 0:
        return "coral"
    return "amber" if fraction_left < 0.2 else "leaf"


__all__ += ["CATEGORY_NAMES", "category_fill", "category_ink", "category_tint",
            "category_line", "gauge_category"]










MODE_HUES = {"interactive": "green", "automatic": "sky"}
HUE_TIP = "leaf"
HUE_TUTORIAL = "coral"
HUE_LOCAL = "teal"
HUE_CLOUD = "sky"
HUE_RESULT = "green"
HUE_RUN = "teal"
HUE_HOWTO = "sky"
HUE_REVIEW = "violet"


CATEGORY_TILE_PX = 28
CATEGORY_TILE_GLYPH_PX = 16


def _blend_over(base_hex: str, rgb: str, alpha: float) -> str:


    base = base_hex.lstrip("#")
    b = [int(base[i:i + 2], 16) for i in (0, 2, 4)]
    f = [int(x) for x in rgb.split(",")]
    mixed = [round(fc * alpha + bc * (1 - alpha)) for fc, bc in zip(f, b)]
    return "#" + "".join(f"{v:02x}" for v in mixed)


def category_wash(name: str) -> str:

    rgb = _CATEGORY_HUES.get(name, _CATEGORY_HUES["green"])[3]
    return _blend_over(SURFACE, rgb, 0.10 if DARK_UI else 0.06)


def category_wash_line(name: str) -> str:

    rgb = _CATEGORY_HUES.get(name, _CATEGORY_HUES["green"])[3]
    return _blend_over(SURFACE, rgb, 0.30 if DARK_UI else 0.26)


def category_tile_bg(name: str, on_wash: bool = False) -> str:

    rgb = _CATEGORY_HUES.get(name, _CATEGORY_HUES["green"])[3]
    alpha = (0.26 if on_wash else 0.20) if DARK_UI else (0.18 if on_wash else 0.13)
    return _blend_over(SURFACE, rgb, alpha)


def category_card_qss(obj_name: str, name: str) -> str:


    return (
        f"QWidget#{obj_name} {{ background-color: {category_wash(name)};"
        f" border: 1px solid {category_wash_line(name)};"
        f" border-radius: {RADIUS_CARD}px; }}"
        f"QWidget#{obj_name} QLabel {{ background: transparent; border: none; color: {INK}; }}"
    )


def category_label_qss(name: str) -> str:

    return _scale_qss_font_px(
        f"QLabel {{ background-color: {category_wash(name)};"
        f" border: 1px solid {category_wash_line(name)};"
        f" border-radius: {RADIUS_CARD}px; padding: 10px 12px;"
        f" font-size: {FONT_BODY}px; color: {INK}; }}"
    )


def category_tile_qss(name: str, side: int, on_wash: bool = False,
                      round_tile: bool = False) -> str:
    radius = side // 2 if round_tile else min(RADIUS_ROW, side // 2)
    return (f"QLabel {{ background: {category_tile_bg(name, on_wash)}; border: none;"
            f" border-radius: {radius}px; }}")


def paint_category_tile(label, glyph: str, name: str,
                        glyph_px: int = CATEGORY_TILE_GLYPH_PX,
                        on_wash: bool = False) -> None:


    try:
        label.setStyleSheet(category_tile_qss(name, label.width(), on_wash))
        from qgis.PyQt.QtGui import QColor

        from ..icons import pixmap_for

        label.setPixmap(pixmap_for(label, glyph, glyph_px, QColor(category_ink(name))))
    except Exception:  # noqa: BLE001
        return


def category_tile(glyph: str, name: str, side: int = CATEGORY_TILE_PX,
                  glyph_px: int = CATEGORY_TILE_GLYPH_PX, on_wash: bool = False):

    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtWidgets import QLabel

    from .font_scale import scale_px_length

    tile = QLabel()
    px = scale_px_length(side)
    tile.setFixedSize(px, px)
    tile.setAlignment(Qt.AlignmentFlag.AlignCenter)
    paint_category_tile(tile, glyph, name, glyph_px, on_wash)
    return tile


_CATEGORY_GLYPH_URIS: dict = {}


def category_glyph_html(glyph: str, name: str, px: int = 14) -> str:


    key = (glyph, name, px, DARK_UI)
    if key not in _CATEGORY_GLYPH_URIS:
        uri = ""
        try:
            from qgis.PyQt.QtCore import QBuffer, QByteArray, QIODevice
            from qgis.PyQt.QtGui import QColor

            from ..icons import render_pixmap

            pixmap = render_pixmap(glyph, QColor(category_ink(name)), px * 2, 1.0)
            data = QByteArray()
            buf = QBuffer(data)
            buf.open(QIODevice.OpenModeFlag.WriteOnly)
            pixmap.save(buf, "PNG")
            buf.close()
            uri = "data:image/png;base64," + bytes(data.toBase64()).decode("ascii")
        except Exception:  # noqa: BLE001
            uri = ""
        _CATEGORY_GLYPH_URIS[key] = uri
    uri = _CATEGORY_GLYPH_URIS[key]
    if not uri:
        return ""
    return (f'<img src="{uri}" width="{px}" height="{px}"'
            ' style="vertical-align: middle;">&nbsp;&nbsp;')


def category_progress_qss(name: str = HUE_RUN) -> str:


    return (
        f"QProgressBar {{ background: {FIELD}; border: none;"
        " border-radius: 2px; max-height: 3px; min-height: 3px; }"
        f"QProgressBar::chunk {{ background: {category_ink(name)}; border-radius: 2px; }}"
    )


__all__ += [
    "CATEGORY_TILE_GLYPH_PX", "CATEGORY_TILE_PX", "HUE_CLOUD", "HUE_HOWTO",
    "HUE_LOCAL", "HUE_RESULT", "HUE_REVIEW", "HUE_RUN", "HUE_TIP", "HUE_TUTORIAL",
    "MODE_HUES", "category_card_qss", "category_glyph_html", "category_label_qss",
    "category_progress_qss", "category_tile", "category_tile_bg", "category_tile_qss",
    "category_wash", "category_wash_line", "paint_category_tile",
]








ENGINE_CARD_HUES = {"cloud": HUE_CLOUD, "local": HUE_LOCAL}


def _engine_card_qss() -> str:
    rules = [
        "QPushButton {"
        f"  background: {SURFACE};"
        f"  border: 1px solid {LINE_STRONG};"
        f"  border-radius: {RADIUS_CARD}px;"
        "  text-align: left;"
        "}",
        f"QPushButton:hover {{ background: {HOVER}; border-color: {LINE_INPUT}; }}",
        f"QPushButton:focus {{ border: 2px solid {ACCENT_BORDER}; }}",
        "QLabel { background: transparent; border: none; }",
    ]
    for key, hue in ENGINE_CARD_HUES.items():
        rules.append(
            f'QPushButton[engine="{key}"]:checked {{'
            f"  background: {category_tile_bg(hue)};"
            f"  border: 2px solid {category_ink(hue)};"
            "}"
            f'QPushButton[engine="{key}"]:checked:hover {{'
            f"  background: {category_tile_bg(hue, on_wash=True)}; }}"
        )
    return "".join(rules)


_ENGINE_CARD_PICK_QSS = _engine_card_qss()


_ENGINE_CARD_TITLE_PICKED_QSS = _scale_qss_font_px(
    f"font-size: {FONT_BODY}px; font-weight: 600; color: {INK};")
_ENGINE_CARD_GLOSS_PICKED_QSS = _scale_qss_font_px(
    f"font-size: {FONT_HINT}px; color: {INK_2};")

__all__ += ["ENGINE_CARD_HUES", "_ENGINE_CARD_PICK_QSS",
            "_ENGINE_CARD_TITLE_PICKED_QSS", "_ENGINE_CARD_GLOSS_PICKED_QSS"]


def locked_combo_qss(mode: str) -> str:






    del mode
    return (
        combo_theme_qss()
        + "QComboBox::drop-down { width: 0px; border: none; }"
        + "QComboBox::down-arrow { image: none; width: 0px; }"
        + f"QComboBox:disabled {{ color: {INK}; background-color: {FIELD};"
        f" border-color: {LINE}; }}"
    )


__all__ += ["locked_combo_qss"]


def tint_section_titles(root, hue: str) -> None:


    try:
        from qgis.PyQt.QtWidgets import QWidget

        ink = category_ink(hue)
        for w in [root, *root.findChildren(QWidget)]:
            label = getattr(w, "header_label", None)
            if label is None:
                continue
            qss = label.styleSheet()
            if f"color: {INK_2};" in qss:
                label.setStyleSheet(qss.replace(f"color: {INK_2};", f"color: {ink};"))
    except Exception:  # noqa: BLE001  # nosec B110
        pass


__all__ += ["tint_section_titles"]










_FOLD_ROW_QSS = (
    "QPushButton { background: transparent; border: none; padding: 0;"
    f" border-radius: {RADIUS_CONTROL}px; text-align: left; }}"
    f"QPushButton:hover {{ background: {HOVER}; }}"
    f"QPushButton:pressed {{ background: {HOVER_ON}; }}"
    + _FOCUS_RING_QSS
)
_FOLD_TITLE_QSS = _scale_qss_font_px(
    f"font-size: {FONT_BASE}px; font-weight: 600; color: {INK};"
    " background: transparent; border: none;")
_FOLD_FACT_QSS = _scale_qss_font_px(
    f"font-size: {FONT_HINT}px; color: {INK_2};"
    " background: transparent; border: none;")



_QUIET_LINK_QSS = _scale_qss_font_px(
    "QPushButton { background: transparent; border: none;"
    f" color: {INK_2}; font-size: {FONT_BODY}px; font-weight: 500;"
    f" padding: 4px 8px; border-radius: {RADIUS_CONTROL}px; }}"
    f"QPushButton:hover {{ color: {INK}; background: {HOVER}; }}"
    f"QPushButton:pressed {{ background: {HOVER_ON}; }}"
    f"QPushButton:disabled {{ color: {INK_3}; }}"
    + _FOCUS_RING_QSS
)

__all__ += ["_FOLD_ROW_QSS", "_FOLD_TITLE_QSS", "_FOLD_FACT_QSS",
            "_QUIET_LINK_QSS"]















_FOLD_FACT_CHIP_QSS = (
    f"background: {HOVER}; border-radius: {RADIUS_CHIP}px; padding: 1px 8px;"
)


def _accent_title_qss(hue: str) -> str:









    return _scale_qss_font_px(
        f"font-size: {FONT_BASE}px; font-weight: 600; color: {category_ink(hue)};"
        " background: transparent; border: none;")


__all__ += ["_FOLD_FACT_CHIP_QSS", "_accent_title_qss"]










_SETUP_STATUS_QSS = _scale_qss_font_px(
    f"font-size: {FONT_BODY}px; font-weight: 600; color: {INK};"
    " background: transparent; border: none;")
_SETUP_STATUS_ERROR_QSS = _scale_qss_font_px(
    f"font-size: {FONT_BODY}px; font-weight: 600; color: {RED_TEXT};"
    " background: transparent; border: none;")



_MUTED_LINE_QSS = _scale_qss_font_px(
    f"font-size: {FONT_BODY}px; color: {INK_2}; background: transparent;"
    " border: none;")



_CARD_TITLE_QSS = _scale_qss_font_px(
    f"font-size: {FONT_BODY}px; font-weight: 600; color: {INK};"
    " background: transparent; border: none;")

__all__ += ["_SETUP_STATUS_QSS", "_SETUP_STATUS_ERROR_QSS", "_MUTED_LINE_QSS",
            "_CARD_TITLE_QSS"]

















def _theme_role_pairs() -> list[tuple[str, str]]:

    pairs = [(_NEUTRALS_LIGHT[k], _NEUTRALS_DARK[k]) for k in _NEUTRALS_LIGHT]
    pairs += [
        ("#1565c0", "#42a5f5"),
        ("#80868f", "#7a7d83"),
        ("#c62828", "#f26b69"),
        ("#b45309", "#f5a623"),
        ("rgba(229, 72, 77, 0.10)", "rgba(238, 92, 97, 0.14)"),
        (ACCENT_INK_LIGHT, ACCENT_INK_DARK),
    ]
    surface_light = _NEUTRALS_LIGHT["surface"]
    surface_dark = _NEUTRALS_DARK["surface"]
    for _fill, ink_light, ink_dark, rgb in _CATEGORY_HUES.values():
        pairs += [
            (ink_light, ink_dark),
            (f"rgba({rgb}, 0.12)", f"rgba({rgb}, 0.18)"),
            (f"rgba({rgb}, 0.2)", f"rgba({rgb}, 0.26)"),
            (f"rgba({rgb}, 0.35)", f"rgba({rgb}, 0.4)"),
            (_blend_over(surface_light, rgb, 0.06), _blend_over(surface_dark, rgb, 0.10)),
            (_blend_over(surface_light, rgb, 0.26), _blend_over(surface_dark, rgb, 0.30)),
            (_blend_over(surface_light, rgb, 0.13), _blend_over(surface_dark, rgb, 0.20)),
            (_blend_over(surface_light, rgb, 0.18), _blend_over(surface_dark, rgb, 0.26)),
        ]
    return pairs


def _theme_swap_map(to_dark: bool) -> dict[str, str]:

    swap: dict[str, str] = {}
    for light, dark in _theme_role_pairs():
        src, dst = (light, dark) if to_dark else (dark, light)
        if src.lower() != dst.lower():
            swap.setdefault(src.lower(), dst)
    return swap


def _theme_swap_text(text: str, swap: dict[str, str], pattern) -> str:
    return pattern.sub(lambda m: swap.get(m.group(0).lower(), m.group(0)), text)


def _theme_swap_pattern(swap: dict[str, str]):
    import re


    keys = sorted(swap, key=len, reverse=True)
    alternatives = "|".join(
        re.escape(k) + (r"(?![0-9a-fA-F])" if k.startswith("#") else "")
        for k in keys)
    return re.compile(alternatives, re.IGNORECASE)



_THEME_FIXED_SUFFIXES = ("_LIGHT", "_DARK")


def _retheme_module_constants(swap: dict[str, str], pattern) -> None:


    import sys



    root = __name__.rsplit(".dock.", 1)[0]
    for mod_name, module in list(sys.modules.items()):
        if module is None or not mod_name.startswith(root):
            continue
        if mod_name.endswith("canvas_palette"):
            continue
        namespace = getattr(module, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        for name, value in list(namespace.items()):
            if not isinstance(value, str) or name.endswith(_THEME_FIXED_SUFFIXES):
                continue
            if "#" not in value and "rgba(" not in value:
                continue
            swapped = _theme_swap_text(value, swap, pattern)
            if swapped != value:
                namespace[name] = swapped


def _retheme_widget_tree(root_widget, swap: dict[str, str], pattern) -> None:
    from qgis.PyQt.QtWidgets import QWidget

    widgets = [root_widget] + list(root_widget.findChildren(QWidget))
    for widget in widgets:
        try:
            sheet = widget.styleSheet()
            if not sheet:
                continue
            swapped = _theme_swap_text(sheet, swap, pattern)
            if swapped != sheet:
                widget.setStyleSheet(swapped)
        except RuntimeError:
            continue






_THEME_REPAINTERS = None


def repaint_on_theme_change(widget, callback) -> None:


    global _THEME_REPAINTERS
    try:
        import weakref

        if _THEME_REPAINTERS is None:
            _THEME_REPAINTERS = weakref.WeakKeyDictionary()
        _THEME_REPAINTERS[widget] = callback
    except TypeError:
        pass  # nosec B110


def _repaint_theme_glyphs() -> None:
    for widget, callback in list((_THEME_REPAINTERS or {}).items()):
        try:
            callback(widget)
        except Exception:  # noqa: BLE001  # nosec B112
            continue


def follow_qgis_theme(root_widget) -> bool:





    global DARK_UI, _N, SEGMENT_ON
    try:
        dark_now = is_dark()
        if dark_now == DARK_UI:
            return False
        swap = _theme_swap_map(to_dark=dark_now)
        pattern = _theme_swap_pattern(swap)
        _retheme_module_constants(swap, pattern)
        DARK_UI = dark_now
        _N = _NEUTRALS_DARK if dark_now else _NEUTRALS_LIGHT
        SEGMENT_ON = HOVER_ON if dark_now else SURFACE
        try:
            root_widget.setUpdatesEnabled(False)
            _retheme_widget_tree(root_widget, swap, pattern)
            _repaint_theme_glyphs()
        finally:
            root_widget.setUpdatesEnabled(True)
            root_widget.update()
        return True
    except Exception:  # noqa: BLE001
        return False


__all__ += ["follow_qgis_theme", "repaint_on_theme_change"]
