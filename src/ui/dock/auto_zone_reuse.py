





















from __future__ import annotations

from qgis.PyQt.QtCore import QPoint, QSize, Qt
from qgis.PyQt.QtGui import QFontMetrics
from qgis.PyQt.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
from .font_scale import scale_px_length, scale_qss_font_px
from .styles import (
    _BTN_BLUE_OUTLINE,
    _CARD_MARGINS,
    _CARD_QSS,
    _FOLD_TITLE_QSS,
    _HINT_LINE_QSS,
    _MENU_QSS,
    ACCENT_BORDER,
    ACCENT_TINT,
    ACCENT_TINT_ON,
    BTN_PILL_PX,
    FONT_BODY,
    INK,
    MUTED,
    RADIUS_CONTROL,
)
from .ui_refresh_credits import format_km2_surface





_ZONE_LINK_QSS = scale_qss_font_px(
    "QPushButton#autoZoneReuseLink { background: transparent;"
    f" color: {MUTED}; border: 2px solid transparent;"
    f" border-radius: {RADIUS_CONTROL}px;"
    f" padding: 2px 10px; font-size: {FONT_BODY}px; }}"
    f"QPushButton#autoZoneReuseLink:hover {{ background: {ACCENT_TINT};"
    f" color: {INK}; }}"
    f"QPushButton#autoZoneReuseLink:pressed {{ background: {ACCENT_TINT_ON}; }}"
    f"QPushButton#autoZoneReuseLink:focus {{ border-color: {ACCENT_BORDER}; }}"
)

_ZONE_LINK_ROW_PX = 28
_ZONE_LINK_GLYPH_PX = 14




_PICK_NAME_PX = 210


_PICK_MENU_MIN_PX = 200


def zone_area_text(km2: float) -> str:

    return tr("{n} km²").format(n=format_km2_surface(km2))


def zone_card_name(label: str, layer_name: str, default_name: str = "") -> str:







    default = str(default_name or "").strip().casefold()
    for candidate in (label, layer_name):
        text = str(candidate or "").strip()
        if text and text.casefold() != default:
            return text
    return ""


def zone_source_row_text(kind: str, label: str, feature_count: int,
                         km2: float) -> str:






    name = str(label or "").strip() or tr("Layer")
    if kind == "selection":
        name = tr("{name}, {n} selected").format(name=name, n=int(feature_count))
    area = zone_area_text(km2) if km2 > 0 else ""
    return f"{name}  ·  {area}" if area else name


def build_zone_ready_card(on_use) -> QWidget:





    card = QWidget()
    card.setObjectName("autoZoneReadyCard")
    card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    card.setStyleSheet(_CARD_QSS.format(name="autoZoneReadyCard"))
    layout = QVBoxLayout(card)
    layout.setContentsMargins(*_CARD_MARGINS)
    layout.setSpacing(6)

    title = QLabel(tr("Zone of interest"))
    title.setStyleSheet(_FOLD_TITLE_QSS)


    title.setWordWrap(True)
    layout.addWidget(title)

    detail = QLabel("")
    detail.setStyleSheet(_HINT_LINE_QSS)
    detail.setWordWrap(True)
    layout.addWidget(detail)

    button = QPushButton(tr("Use this zone"))
    button.setObjectName("autoZoneReadyUse")


    button.setStyleSheet(_BTN_BLUE_OUTLINE)
    button.setMinimumHeight(BTN_PILL_PX)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setAutoDefault(False)
    button.setAccessibleName(tr("Use this zone"))
    button.clicked.connect(on_use)
    layout.addWidget(button)

    card.title_label = title
    card.detail_label = detail
    card.use_button = button
    card.setVisible(False)
    return card


def build_zone_reuse_link(on_click) -> QWidget:



    row = QWidget()
    row.setObjectName("autoZoneReuseRow")
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    button = QPushButton(tr("Or use an existing zone"), row)
    button.setObjectName("autoZoneReuseLink")
    button.setStyleSheet(_ZONE_LINK_QSS)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setAutoDefault(False)
    button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
    button.setToolTip(tr("Take the zone from a layer or a selection"))
    button.setAccessibleName(tr("Or use an existing zone"))
    button.setMinimumHeight(_ZONE_LINK_ROW_PX)
    try:
        from qgis.PyQt.QtGui import QColor

        from ..icons import icon_for

        glyph = _ZONE_LINK_GLYPH_PX
        button.setIcon(icon_for(button, "layers", glyph, QColor(MUTED)))
        button.setIconSize(QSize(glyph, glyph))
    except (RuntimeError, AttributeError, TypeError):
        pass  # nosec B110
    button.clicked.connect(on_click)
    layout.addStretch(1)
    layout.addWidget(button)
    layout.addStretch(1)
    row.link_button = button
    return row


def _elided(widget, text: str) -> str:

    try:
        metrics = QFontMetrics(widget.font())
        return metrics.elidedText(
            text, Qt.TextElideMode.ElideMiddle, scale_px_length(_PICK_NAME_PX))
    except (RuntimeError, AttributeError, TypeError):
        return text


class DockAutoZoneReuseMixin:


    def refresh_auto_zone_ready_card(self) -> None:





        card = getattr(self, "auto_zone_ready_card", None)
        if card is None:
            return
        found = self._read_shared_zone_for_card()
        try:
            if found is None:
                card.setVisible(False)
                return
            name, km2 = found
            area = zone_area_text(km2)
            card.detail_label.setText(f"{name}  ·  {area}" if name else area)
            card.detail_label.setToolTip(name)
            card.setVisible(True)
        except (RuntimeError, AttributeError):
            self.auto_zone_ready_card = None

    def _read_shared_zone_for_card(self):






        try:
            from qgis.core import QgsProject

            from ...core import zone_of_interest as zoi

            project = QgsProject.instance()
            zone = zoi.read_zone(project)
            if zone is None:
                return None
            layer = project.mapLayer(zone.layer_id) if zone.layer_id else None
            layer_name = layer.name() if layer is not None else ""
            name = zone_card_name(zone.label, layer_name, zoi.ZONE_LAYER_NAME)
            return name, zone.area_km2()
        except Exception:  # noqa: BLE001
            return None

    def _on_auto_zone_ready_use(self) -> None:

        self.auto_zone_source_picked.emit("zone", "")

    def _on_auto_zone_reuse_link(self) -> None:






        row = getattr(self, "auto_zone_reuse_row", None)
        if row is None:
            return
        try:
            anchor = row.link_button
        except (RuntimeError, AttributeError):
            return
        sources = self._read_zone_sources_for_picker()
        menu = QMenu(anchor)
        menu.setStyleSheet(scale_qss_font_px(_MENU_QSS))
        menu.setMinimumWidth(scale_px_length(_PICK_MENU_MIN_PX))


        menu.setToolTipsVisible(True)
        if not sources:
            empty = menu.addAction(tr("No polygon layer in this project"))
            empty.setEnabled(False)
        previous_kind = ""
        for source in sources:
            if previous_kind and source.kind != previous_kind:
                menu.addSeparator()
            previous_kind = source.kind
            full = zone_source_row_text(
                source.kind, source.label, source.feature_count,
                source.area_km2)
            action = menu.addAction(_elided(menu, full))
            action.setToolTip(full)
            action.setData((source.kind, source.layer_id))




        picked: dict = {}
        menu.triggered.connect(lambda action: picked.update(row=action))
        try:
            below = QPoint(0, anchor.height() + scale_px_length(4))
            menu.exec(anchor.mapToGlobal(below))
        finally:
            menu.deleteLater()
        chosen = picked.get("row")
        if chosen is None:
            return
        data = chosen.data()
        if not data:
            return
        kind, layer_id = data
        self.auto_zone_source_picked.emit(str(kind), str(layer_id))

    def _read_zone_sources_for_picker(self) -> list:



        try:
            from ...core import zone_of_interest as zoi

            return list(zoi.zone_sources())
        except Exception:  # noqa: BLE001
            return []
