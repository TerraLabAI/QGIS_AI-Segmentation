








from __future__ import annotations

from qgis.PyQt.QtCore import QRectF, QSize, Qt
from qgis.PyQt.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from qgis.PyQt.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ...core.i18n import tr
from .auto_flow_look import auto_flow_sheet
from .font_scale import scale_qss_font_px
from .review_card_rows import row_lead_glyph
from .styles import (
    FONT_BODY,
    HOVER,
    INK,
    LINE,
    RADIUS_CARD,
    RADIUS_CONTROL,
    SURFACE,
    combo_theme_qss,
)






_LEGEND_RANDOM_DOTS = ("#e65133", "#b3e633", "#33e67a", "#3389e6")
_DISPLAY_SWATCH = {
    "normal": ("#0078ff",),
    "outline": ("#e31a1c",),
    "confidence": ("#fde725", "#440154"),
    "random": _LEGEND_RANDOM_DOTS,
}




_VIEW_COMBO_QSS = scale_qss_font_px(
    "QComboBox { background-color: transparent; border: 1px solid transparent;"
    f" border-radius: {RADIUS_CONTROL}px; padding: 0 4px 0 8px;"
    f" font-size: {FONT_BODY}px; color: {INK}; }}"
    f"QComboBox:hover, QComboBox:on {{ background-color: {HOVER};"
    " border-color: transparent; }"
)


def _legend_dots(colors, glyph: str = "●") -> str:

    return "".join(
        f'<span style="color:{c};">{glyph}</span>'
        for c in colors
    )


def display_legend_html(mode: str) -> str:




    if mode == "outline":
        return "{d}&nbsp; {t}".format(
            d=_legend_dots(("#e31a1c",), glyph="○"),
            t=tr("Outlines only - check boundaries against the imagery"))
    if mode == "confidence":
        return "{y}&nbsp;{cf}&nbsp; &middot; &nbsp;{p}&nbsp;{un}".format(
            y=_legend_dots(("#fde725",)), cf=tr("confident"),
            p=_legend_dots(("#440154",)), un=tr("uncertain"))
    if mode == "random":
        return "{d}&nbsp; {t}".format(
            d=_legend_dots(_LEGEND_RANDOM_DOTS),
            t=tr("One color per object - check neighbors are separated"))
    return "{d}&nbsp; {t}".format(
        d=_legend_dots(("#0078ff",)),
        t=tr("Detected object"))


def display_legend_shown(mode: str) -> bool:



    return mode == "confidence"


def display_swatch_icon(mode: str) -> QIcon:


    colors = _DISPLAY_SWATCH.get(mode, _DISPLAY_SWATCH["normal"])
    scale = 2
    width, height = 34, 12
    pix = QPixmap(width * scale, height * scale)
    pix.setDevicePixelRatio(scale)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        dot = 7.0
        step = 8.5
        x = 1.0
        for color in colors:
            rect = QRectF(x, (height - dot) / 2.0, dot, dot)
            if mode == "outline":
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(color), 1.8))
                painter.drawEllipse(rect.adjusted(0.9, 0.9, -0.9, -0.9))
            else:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(color))
                painter.drawEllipse(rect)
            x += step
    finally:
        painter.end()
    return QIcon(pix)


def build_display_combo(dock) -> QComboBox:


    combo = QComboBox()


    combo.setStyleSheet(combo_theme_qss() + _VIEW_COMBO_QSS)
    combo.setCursor(Qt.CursorShape.PointingHandCursor)
    combo.setFocusPolicy(Qt.FocusPolicy.TabFocus)
    items = (
        (tr("Normal"), "normal"),
        (tr("Outline"), "outline"),
        (tr("Confidence"), "confidence"),


        (tr("Distinct"), "random"),
    )
    for label, key in items:
        combo.addItem(display_swatch_icon(key), label, key)

        tip = display_legend_html(key).split("&nbsp; ", 1)[-1]
        combo.setItemData(combo.count() - 1, tip.replace("&middot;", "·")
                          .replace("&nbsp;", " "), Qt.ItemDataRole.ToolTipRole)
    combo.setIconSize(QSize(30, 12))



    combo.setCurrentIndex(max(0, combo.findData("random")))
    combo.setToolTip(tr(
        "How detections are coloured on the map (visual only): Normal fill, "
        "Outline, Confidence heatmap, or a distinct colour per object to tell "
        "them apart."))

    def _on_mode_picked(_index: int) -> None:
        mode = combo.currentData() or "random"
        sync_display_legend(dock)
        dock.auto_display_mode_changed.emit(mode)

    combo.currentIndexChanged.connect(_on_mode_picked)
    return combo


def sync_display_legend(dock) -> None:

    combo = getattr(dock, "auto_display_combo", None)
    legend = getattr(dock, "auto_display_legend", None)
    if combo is None or legend is None:
        return
    try:
        legend.setVisible(display_legend_shown(combo.currentData() or "random"))
    except (RuntimeError, AttributeError):
        pass


def build_review_view_block(dock) -> QWidget:



    block = QWidget()
    block.setObjectName("autoReviewViewBlock")
    block.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    block.setStyleSheet(
        f"QWidget#autoReviewViewBlock {{ background: {SURFACE};"
        f" border: 1px solid {LINE}; border-radius: {RADIUS_CARD}px; }}"
        "QLabel { background: transparent; border: none; }"
        + auto_flow_sheet())
    block.setVisible(False)
    col = QVBoxLayout(block)

    col.setContentsMargins(12, 4, 6, 4)
    col.setSpacing(0)
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(8)
    row.addWidget(row_lead_glyph("eye"), 0, Qt.AlignmentFlag.AlignVCenter)


    label = QLabel(tr("View detections as:").rstrip(":  "))
    label.setObjectName("autoFieldLabel")
    row.addWidget(label, 0, Qt.AlignmentFlag.AlignVCenter)
    row.addStretch(1)
    dock.auto_display_combo = build_display_combo(dock)
    row.addWidget(dock.auto_display_combo, 0, Qt.AlignmentFlag.AlignVCenter)
    col.addLayout(row)

    dock.auto_display_legend = QLabel(display_legend_html("random"))
    dock.auto_display_legend.setWordWrap(True)
    dock.auto_display_legend.setTextFormat(Qt.TextFormat.RichText)
    dock.auto_display_legend.setObjectName("autoHint")

    dock.auto_display_legend.setContentsMargins(24, 0, 0, 6)
    col.addWidget(dock.auto_display_legend)
    dock.auto_review_view_row = block
    sync_display_legend(dock)
    return block


__all__ = [
    "build_display_combo",
    "build_review_view_block",
    "display_legend_html",
    "display_legend_shown",
    "display_swatch_icon",
    "sync_display_legend",
]
