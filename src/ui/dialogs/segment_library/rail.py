"""Persistent navigation rail for the Segment library.

Left of the card grid, always visible. It NAVIGATES between the library's
views and marks exactly one as active ("you are here"), unlike a filter that
only narrows a list. Three groups: Featured (the curated Popular picks),
Categories (one row per catalogue family) and My work (the user's own
detections).

Rows are deliberately plain, label plus a muted count, with no glyph column
and no per-row hue. The tinted Unicode glyph each category used to carry made
the rail the loudest surface in the dialog; the quiet index is AI Edit's rail,
which this mirrors. The only color is the lime bar on the active row.

Mixed into SegmentLibraryDialog. It reads `_categories`, `_top_picks`,
`_by_id`, `_hist_runs` and `_active_key` off the host, and calls the host's
`_select_tab`.
"""
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
    _RAIL_ITEM_COUNT,
    _RAIL_PANEL,
    _RAIL_POPULAR_TARGET,
    _RAIL_RECENT_TARGET,
    _rail_item_style,
    _rail_label_style,
)

_RAIL_WIDTH = 210


def _rail_widget_alive(widget) -> bool:
    """True while the row's underlying C++ object is still there. A history
    reply can land after the dialog closed, and touching a dead label raises."""
    if widget is None:
        return False
    try:
        from qgis.PyQt import sip

        return not sip.isdeleted(widget)
    except (ImportError, TypeError, RuntimeError):
        return True


class LibraryRailMixin:
    """Builds and drives the library's navigation rail."""

    # ---- build -----------------------------------------------------------

    def _build_library_rail(self) -> QWidget:
        # target -> its button, its text label and its count label, so one row
        # can be restyled or recounted without walking the whole rail.
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
            _RAIL_POPULAR_TARGET, tr("Popular"), self._rail_count(_RAIL_POPULAR_TARGET)))

        rows = self._category_rail_rows()
        if rows:
            self._add_rail_group(box, tr("Categories"))
            for target, label, count in rows:
                box.addWidget(self._make_rail_item(target, label, count))

        # The user's own work closes the rail, the way AI Edit's does: a first
        # run has nothing there, so the curated content has to lead.
        self._add_rail_group(box, tr("My work"))
        box.addWidget(self._make_rail_item(
            _RAIL_RECENT_TARGET, tr("Recent"), self._rail_count(_RAIL_RECENT_TARGET)))
        box.addWidget(self._make_rail_item(
            _RAIL_FAVORITES_TARGET, tr("Favorites"),
            self._rail_count(_RAIL_FAVORITES_TARGET)))

        box.addStretch()
        return panel

    def _category_rail_rows(self) -> list[tuple[str, str, int]]:
        """One (target, label, count) per non-empty catalogue category, in
        catalogue order. An empty category is skipped so the rail never shows
        a row that opens on nothing."""
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

    def _make_rail_item(self, target: str, label: str, count: int) -> QPushButton:
        btn = QPushButton()
        btn.setObjectName("railitem")
        btn.setCursor(QtC.PointingHandCursor)
        btn.setStyleSheet(_rail_item_style(False))
        btn.setSizePolicy(QtC.SizePolicyExpanding, QtC.SizePolicyFixed)
        btn.clicked.connect(lambda _c=False, t=target: self._on_rail_click(t))

        # Left margin clears the 3px active bar so the text never touches it;
        # identical on inactive rows, which reserve the same 3px transparent.
        row = QHBoxLayout(btn)
        row.setContentsMargins(12, 0, 4, 0)
        row.setSpacing(10)
        text = QLabel(label)
        text.setStyleSheet(_rail_label_style(False))
        text.setAttribute(QtC.WA_TransparentForMouseEvents)
        row.addWidget(text, 1)
        count_lbl = QLabel(self._rail_count_text(count))
        count_lbl.setStyleSheet(_RAIL_ITEM_COUNT)
        count_lbl.setAttribute(QtC.WA_TransparentForMouseEvents)
        row.addWidget(count_lbl)

        self._rail_items[target] = btn
        self._rail_labels[target] = text
        self._rail_counts[target] = count_lbl
        return btn

    # ---- counts ----------------------------------------------------------

    @staticmethod
    def _rail_count_text(count: int) -> str:
        """A zero count prints nothing: an empty Recent row should read as
        quiet, not as a row advertising that it holds nothing."""
        return str(count) if count > 0 else ""

    def _rail_count(self, target: str) -> int:
        """How many entries the row's view holds right now."""
        if target == _RAIL_POPULAR_TARGET:
            return sum(1 for i in self._top_picks if i in self._by_id)
        if target == _RAIL_RECENT_TARGET:
            runs = self._hist_runs.get("all") or []
            return len(runs) if runs else len(self._local_recent_entries())
        if target == _RAIL_FAVORITES_TARGET:
            return (len(self._hist_runs.get("favorites") or [])
                    + len(self._favorite_template_presets()))
        for cat in self._categories:
            if cat.get("key") == target:
                return len(cat.get("presets", []))
        return 0

    def _refresh_rail_counts(self) -> None:
        """Update the counts that move with the user's own work. Category and
        Popular counts are catalogue-fixed and never change here."""
        for target in (_RAIL_RECENT_TARGET, _RAIL_FAVORITES_TARGET):
            lbl = self._rail_counts.get(target)
            if _rail_widget_alive(lbl):
                lbl.setText(self._rail_count_text(self._rail_count(target)))

    # ---- navigation ------------------------------------------------------

    def _on_rail_click(self, target: str) -> None:
        # A rail click is an explicit "leave search": clear the box quietly.
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
        """Mark one row active and clear the previous one. An unchanged target
        does nothing at all, and a changed one touches only the two rows that
        actually move: setStyleSheet on a row costs a restyle plus a repaint,
        and every search keystroke reaches here. The invariant this rests on is
        that every row except `_rail_active` carries the inactive sheet, which
        `_make_rail_item` establishes and only this method changes."""
        if target == self._rail_active:
            return
        previous = self._rail_active
        self._rail_active = target
        if previous is not None:
            self._restyle_rail_item(previous, False)
        if target is not None:
            self._restyle_rail_item(target, True)
