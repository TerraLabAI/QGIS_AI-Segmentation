

















from __future__ import annotations

from qgis.PyQt.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ....core import qt_compat as QtC
from ....core.i18n import tr
from ....core.presets.segmentation_presets import pick_label
from ...dock.font_scale import scale_px_length
from .common import (
    _RAIL_FAVORITES_TARGET,
    _RAIL_GROUP,
    _RAIL_HISTORY_VIEWS,
    _RAIL_ITEM_COUNT,
    _RAIL_PANEL,
    _RAIL_POPULAR_TARGET,
    _RAIL_RECENT_TARGET,
    _rail_item_style,
    _rail_label_style,
)

_RAIL_WIDTH = 210


def _rail_widget_alive(widget) -> bool:


    if widget is None:
        return False
    try:
        from qgis.PyQt import sip

        return not sip.isdeleted(widget)
    except (ImportError, TypeError, RuntimeError):
        return True


class LibraryRailMixin:




    def _build_library_rail(self) -> QWidget:


        self._rail_items: dict[str, QPushButton] = {}
        self._rail_labels: dict[str, QLabel] = {}
        self._rail_counts: dict[str, QLabel] = {}
        self._rail_active: str | None = None

        panel = QFrame()
        panel.setObjectName("librail")
        panel.setStyleSheet(_RAIL_PANEL)
        panel.setFixedWidth(scale_px_length(_RAIL_WIDTH))
        box = QVBoxLayout(panel)
        box.setContentsMargins(4, 6, 10, 8)
        box.setSpacing(2)

        self._add_rail_group(box, tr("Featured"), first=True)
        box.addWidget(self._make_rail_item(
            _RAIL_POPULAR_TARGET, tr("Popular"),
            self._rail_count_label(_RAIL_POPULAR_TARGET)))

        rows = self._category_rail_rows()
        if rows:
            self._add_rail_group(box, tr("Categories"))
            for target, label, count in rows:
                box.addWidget(self._make_rail_item(
                    target, label, self._rail_count_text(count)))



        self._add_rail_group(box, tr("My work"))
        box.addWidget(self._make_rail_item(
            _RAIL_RECENT_TARGET, tr("Recent"),
            self._rail_count_label(_RAIL_RECENT_TARGET)))
        box.addWidget(self._make_rail_item(
            _RAIL_FAVORITES_TARGET, tr("Favorites"),
            self._rail_count_label(_RAIL_FAVORITES_TARGET)))

        box.addStretch()
        return panel

    def _category_rail_rows(self) -> list[tuple[str, str, int]]:



        rows: list[tuple[str, str, int]] = []
        for cat in self._categories:
            presets = cat.get("presets", [])
            if not presets:
                continue
            rows.append((
                cat["key"],
                pick_label(cat.get("label"), cat.get("key", "")),
                len(presets),
            ))
        return rows

    @staticmethod
    def _add_rail_group(box: QVBoxLayout, label: str, first: bool = False) -> None:
        lbl = QLabel(label)
        lbl.setStyleSheet(_RAIL_GROUP)
        lbl.setContentsMargins(9, 2 if first else 12, 8, 4)
        box.addWidget(lbl)

    def _make_rail_item(self, target: str, label: str,
                        count_text: str) -> QPushButton:
        btn = QPushButton()
        btn.setObjectName("railitem")




        btn.setAutoDefault(False)
        btn.setCursor(QtC.PointingHandCursor)
        btn.setStyleSheet(_rail_item_style(False))
        btn.setSizePolicy(QtC.SizePolicyExpanding, QtC.SizePolicyFixed)
        btn.clicked.connect(lambda _c=False, t=target: self._on_rail_click(t))



        row = QHBoxLayout(btn)
        row.setContentsMargins(12, 0, 4, 0)
        row.setSpacing(10)
        text = QLabel(label)
        text.setStyleSheet(_rail_label_style(False))
        text.setAttribute(QtC.WA_TransparentForMouseEvents)
        row.addWidget(text, 1)
        count_lbl = QLabel(count_text)
        count_lbl.setStyleSheet(_RAIL_ITEM_COUNT)
        count_lbl.setAttribute(QtC.WA_TransparentForMouseEvents)
        row.addWidget(count_lbl)

        self._rail_items[target] = btn
        self._rail_labels[target] = text
        self._rail_counts[target] = count_lbl
        return btn



    @staticmethod
    def _rail_count_text(count: int) -> str:


        return str(count) if count > 0 else ""

    def _rail_count(self, target: str) -> int:







        if target == _RAIL_POPULAR_TARGET:
            return sum(1 for i in self._top_picks if i in self._by_id)
        if target == _RAIL_RECENT_TARGET:
            if not self._auth:
                return len(self._local_recent_entries())
            return len(self._hist_runs.get("all") or [])
        if target == _RAIL_FAVORITES_TARGET:
            server = (len(self._hist_runs.get("favorites") or [])
                      if self._auth else 0)
            return server + len(self._favorite_template_presets())
        for cat in self._categories:
            if cat.get("key") == target:
                return len(cat.get("presets", []))
        return 0

    def _rail_count_label(self, target: str) -> str:






        view = _RAIL_HISTORY_VIEWS.get(target)
        if view is not None and self._history_view_loading(view):
            return ""
        text = self._rail_count_text(self._rail_count(target))
        if text and view is not None and self._hist_has_more.get(view):
            text += "+"
        return text

    def _refresh_rail_counts(self) -> None:


        for target in (_RAIL_RECENT_TARGET, _RAIL_FAVORITES_TARGET):
            lbl = self._rail_counts.get(target)
            if _rail_widget_alive(lbl):
                lbl.setText(self._rail_count_label(target))



    def _on_rail_click(self, target: str) -> None:

        if self._search.text().strip():
            self._search.blockSignals(True)
            self._search.clear()
            self._search.blockSignals(False)
            self._query = ""
        self._select_tab(target)

    def _restyle_rail_item(self, target: str, active: bool) -> None:
        btn = self._rail_items.get(target)
        if not _rail_widget_alive(btn):
            return
        btn.setStyleSheet(_rail_item_style(active))
        lbl = self._rail_labels.get(target)
        if _rail_widget_alive(lbl):
            lbl.setStyleSheet(_rail_label_style(active))

    def _set_rail_active(self, target: str | None) -> None:






        if target == self._rail_active:
            return
        previous = self._rail_active
        self._rail_active = target
        if previous is not None:
            self._restyle_rail_item(previous, False)
        if target is not None:
            self._restyle_rail_item(target, True)
