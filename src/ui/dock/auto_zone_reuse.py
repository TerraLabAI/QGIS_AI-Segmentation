






















from __future__ import annotations

from qgis.PyQt.QtCore import QPoint, QSize, Qt
from qgis.PyQt.QtGui import QFontMetrics, QIcon
from qgis.PyQt.QtWidgets import (
    QHBoxLayout,
    QMenu,
    QPushButton,
    QWidget,
)

from ...core.i18n import tr
from .font_scale import scale_px_length, scale_qss_font_px
from .styles import (
    _MENU_QSS,
    ACCENT_BORDER,
    ACCENT_TINT,
    ACCENT_TINT_ON,
    FONT_BODY,
    INK,
    MUTED,
    RADIUS_CONTROL,
)
from .ui_refresh_credits import format_km2_surface


def _zone_link_qss(name: str, padding: str) -> str:




    return scale_qss_font_px(
        f"QPushButton#{name} {{ background: transparent;"
        f" color: {MUTED}; border: 2px solid transparent;"
        f" border-radius: {RADIUS_CONTROL}px;"
        f" padding: {padding}; font-size: {FONT_BODY}px; }}"
        f"QPushButton#{name}:hover {{ background: {ACCENT_TINT};"
        f" color: {INK}; }}"
        f"QPushButton#{name}:pressed {{ background: {ACCENT_TINT_ON}; }}"
        f"QPushButton#{name}:focus {{ border-color: {ACCENT_BORDER}; }}"
    )


_ZONE_LINK_QSS = _zone_link_qss("autoZoneReuseLink", "2px 10px")


_ZONE_PICK_QSS = _zone_link_qss("autoZonePickButton", "2px 4px")

_ZONE_LINK_ROW_PX = 28
_ZONE_LINK_GLYPH_PX = 14




_PICK_NAME_PX = 210


_PICK_MENU_MIN_PX = 200


_HELD_LINK_TEXT_PX = 190


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


def build_zone_reuse_link(on_click, on_pick) -> QWidget:




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
    pick = QPushButton("", row)
    pick.setObjectName("autoZonePickButton")
    pick.setStyleSheet(_ZONE_PICK_QSS)
    pick.setCursor(Qt.CursorShape.PointingHandCursor)
    pick.setAutoDefault(False)
    pick.setFocusPolicy(Qt.FocusPolicy.TabFocus)
    pick.setToolTip(tr("Take the zone from a layer or a selection"))
    pick.setAccessibleName(tr("Take the zone from a layer or a selection"))
    pick.setMinimumHeight(_ZONE_LINK_ROW_PX)
    try:
        from qgis.PyQt.QtGui import QColor

        from ..icons import icon_for

        glyph = _ZONE_LINK_GLYPH_PX
        pick.setIcon(icon_for(pick, "layers", glyph, QColor(MUTED)))
        pick.setIconSize(QSize(glyph, glyph))
    except (RuntimeError, AttributeError, TypeError):
        pick.setText("...")
    pick.clicked.connect(on_pick)


    pick.setVisible(False)
    layout.addStretch(1)
    layout.addWidget(button)
    layout.addWidget(pick)
    layout.addStretch(1)
    row.link_button = button
    row.pick_button = pick


    row.link_icon = button.icon()
    return row


def _elided(widget, text: str, width_px: int = _PICK_NAME_PX) -> str:

    try:
        metrics = QFontMetrics(widget.font())
        return metrics.elidedText(
            text, Qt.TextElideMode.ElideMiddle, scale_px_length(width_px))
    except (RuntimeError, AttributeError, TypeError):
        return text


class DockAutoZoneReuseMixin:


    def refresh_auto_zone_reuse_link(self) -> None:





        row = getattr(self, "auto_zone_reuse_row", None)
        if row is None:
            return
        found = self._read_shared_zone_for_link()
        try:
            link = row.link_button
            if found is None:
                self._auto_zone_held = False
                text = tr("Or use an existing zone")
                link.setText(text)
                link.setToolTip(tr("Take the zone from a layer or a selection"))
                link.setAccessibleName(text)
                link.setIcon(row.link_icon)
                row.pick_button.setVisible(False)
                return
            name, km2 = found
            full = tr("Or use {name} · {area}").format(
                name=name, area=zone_area_text(km2))
            self._auto_zone_held = True
            link.setText(_elided(link, full, _HELD_LINK_TEXT_PX))
            link.setToolTip(full)
            link.setAccessibleName(full)
            link.setIcon(QIcon())
            row.pick_button.setVisible(True)
        except (RuntimeError, AttributeError):
            self.auto_zone_reuse_row = None

    def _read_shared_zone_for_link(self):






        try:
            from qgis.core import QgsProject

            from ...core import zone_of_interest as zoi

            project = QgsProject.instance()
            zone = zoi.read_zone(project)
            if zone is None:
                return None
            layer = project.mapLayer(zone.layer_id) if zone.layer_id else None
            layer_name = layer.name() if layer is not None else ""
            if zoi.is_default_zone_name(layer_name):
                layer_name = ""
            default_name = zoi.zone_layer_name()
            name = zone_card_name(zone.label, layer_name, default_name)
            return name or default_name, zone.area_km2()
        except Exception:  # noqa: BLE001
            return None

    def _on_auto_zone_reuse_link(self) -> None:


        if getattr(self, "_auto_zone_held", False):
            self.auto_zone_source_picked.emit("zone", "")
            return
        self._open_auto_zone_picker()

    def _open_auto_zone_picker(self) -> None:






        row = getattr(self, "auto_zone_reuse_row", None)
        if row is None:
            return
        try:
            anchor = row.link_button
            if row.pick_button.isVisible():
                anchor = row.pick_button
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
