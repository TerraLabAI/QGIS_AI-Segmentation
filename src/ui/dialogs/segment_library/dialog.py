































from __future__ import annotations

from qgis.PyQt.QtCore import QEvent, QPoint, QTimer
from qgis.PyQt.QtGui import QGuiApplication
from qgis.PyQt.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ....core import detection_history
from ....core import qt_compat as QtC
from ....core.i18n import tr
from ....core.presets import run_history_cache, segment_history
from ....core.presets.segmentation_presets import pick_label, preset_query_rank
from ....core.presets.segmentation_presets_client import (
    base_url,
    cached_or_offline_catalog,
)
from ....core.presets.template_favorites import (
    is_favorite_template,
    toggle_favorite_template,
)
from ....core.qt_compat import safe_disconnect
from ....core.server_dials import dial_in_range
from ...dock.styles import apply_quiet_scrollbar
from ...template_demo_loader import TemplateDemoLoader
from .cards import _PresetTile
from .common import (
    _EMPTY_GLYPH,
    _EMPTY_MSG,
    _GHOST_BTN_QSS,
    _RAIL_FAVORITES_TARGET,
    _RAIL_HISTORY_VIEWS,
    _RAIL_HOME_TARGET,
    _RAIL_POPULAR_TARGET,
    _RAIL_RECENT_TARGET,
    _apply_library_ground,
    _category_glyph,
    _empty_glyph_pixmap,
    _fmt_count,
)
from .detail import _PresetDetailDialog
from .history import LibraryHistoryMixin
from .home import LibraryPageHeader, LibraryShelf, style_search_pill
from .rail import LibraryRailMixin
from .recent_local import restore_recent_on_map
from .run_actions import LibraryRunActionsMixin
from .run_card import _RunCard





_CARD_MIN_W = 270
_GRID_SPACING = 12
_GRID_COLS_MIN = 1
_GRID_COLS_MAX = 5
_GRID_COLS_DEFAULT = 3


_CATALOG_COLS = 3
_TILE_MIN_W = 170


_EVENT_RESIZE = QtC.resolve_qt_enum(QEvent, "Type", "Resize")


class SegmentLibraryDialog(LibraryRailMixin, LibraryHistoryMixin,
                           LibraryRunActionsMixin, QDialog):






    def __init__(self, parent=None, *, recent: list[dict] | None = None,
                 plugin=None, view_only: bool = False, source: str = "chip",
                 no_imagery: bool = False):
        super().__init__(parent)


        self._no_imagery = bool(no_imagery)


        self._open_source = source




        self._view_only = bool(view_only)
        self.setWindowTitle(
            tr("Segment library (view only)") if self._view_only
            else tr("Segment library"))
        _apply_library_ground(self)
        self.setSizeGripEnabled(True)
        self._apply_open_size()
        self._selected_prompt: str | None = None
        self._selected_raster: dict | None = None
        self._detail_open = False
        self._base = base_url()


        self._categories, self._top_picks = cached_or_offline_catalog()
        self._index_catalog()
        self._recent_local = (list(recent) if recent is not None
                              else segment_history.get_recent())


        self._history_local = detection_history.get_entries()
        self._active_key = _RAIL_HOME_TARGET
        self._query = ""


        self._cards_by_id: dict[str, list] = {}



        self._cols = _GRID_COLS_DEFAULT
        self._grid_widgets: list = []


        self._grid_sections: list | None = None
        self._grid_span_all = False


        self._grid_signature: tuple | None = None
        self._run_cards: list[tuple[dict, _RunCard]] = []

        self._loader = TemplateDemoLoader(self)
        self._loader.loaded.connect(self._on_demo_loaded)
        self._loader.failed.connect(self._on_demo_failed)




        self._plugin = plugin
        self._auth: dict = {}
        self._auth_revision = None
        try:
            from ....core.activation_manager import auth_revision, get_auth_header
            self._auth_revision = auth_revision()
            self._auth = get_auth_header() or {}
        except Exception:  # noqa: BLE001
            self._auth = {}
        self._client = None
        self._hist_runs: dict[str, list[dict]] = {
            "all": run_history_cache.get_runs(), "favorites": []}
        self._hist_has_more = {"all": False, "favorites": False}
        self._hist_synced: set = set()
        self._hist_inflight: set = set()
        self._hist_fail_logged: set = set()



        self._fav_overrides: dict[str, bool] = {}


        self._hist_failed: set = set()
        self._hist_pages_loaded = 0


        self._hist_cards: dict[str, _RunCard] = {}
        self._thumb_cards: dict[str, _RunCard] = {}
        self._hist_busy = False






        self._fetch_worker = None
        self._fetch_progress = None




        self._live_workers: list[tuple] = []
        self._pending_action: tuple | None = None

        self._detail_dlg = None
        self._tabs_tracked: set = set()
        self._hist_loader = TemplateDemoLoader(self)
        self._hist_loader.loaded.connect(self._on_thumb_loaded)
        self._hist_loader.failed.connect(self._on_thumb_failed)

        self._build_ui()
        from ...dock.font_scale import apply_font_scale_to_tree

        apply_font_scale_to_tree(self)
        self._select_tab(_RAIL_HOME_TARGET)
        self._track_tab_opened("detect")



    def _index_catalog(self) -> None:
        self._by_id = {
            p["id"]: p for cat in self._categories for p in cat.get("presets", [])}


        self._by_token = {
            p.get("prompt", ""): p for cat in self._categories
            for p in cat.get("presets", []) if p.get("prompt")}
        self._cat_label_by_id: dict[str, str] = {}
        for cat in self._categories:
            cat_label = pick_label(cat.get("label"), cat.get("key", ""))
            for p in cat.get("presets", []):
                self._cat_label_by_id[p.get("id", "")] = cat_label
        self._cat_key_by_id: dict[str, str] = {
            p.get("id", ""): str(cat.get("key") or "")
            for cat in self._categories for p in cat.get("presets", [])}

    def refresh_catalog(self) -> None:



        cats, tops = cached_or_offline_catalog()
        if (cats, list(tops)) == (self._categories, list(self._top_picks)):
            return
        self._categories, self._top_picks = cats, tops
        self._index_catalog()
        det = getattr(self, "_open_preset_detail", None)
        if det is not None:
            fresh = self._by_id.get(det._preset.get("id", ""))
            if fresh is not None:
                det.refresh_preset(fresh)
        if _RAIL_HISTORY_VIEWS.get(self._active_key) is not None:
            return
        self._repaint_grid()

    def _end_detail(self) -> None:



        self._detail_open = False
        if not getattr(self, "_catalog_rebuild_pending", False):
            return
        from qgis.PyQt.QtCore import QTimer

        def _later(dlg=self):
            try:
                if dlg._catalog_rebuild_pending and not dlg._detail_open:
                    dlg._repaint_grid()
            except RuntimeError:
                pass
        QTimer.singleShot(0, _later)

    def _repaint_grid(self) -> None:

        self._catalog_rebuild_pending = False
        if self._selected_prompt is not None or not self.isVisible():
            return
        bar = self._scroll.verticalScrollBar()
        pos = bar.value()


        kept = {}
        for pid, cards in self._cards_by_id.items():
            for card in cards:
                try:
                    kept[pid] = (card.slider._before, card.slider._after)
                    break
                except RuntimeError:
                    pass
        self.setUpdatesEnabled(False)
        try:
            self._rebuild_current_grid()
            for pid, (before, after) in kept.items():
                for card in self._cards_by_id.get(pid, []):
                    if before is not None:
                        card.set_image("before", before)
                    if after is not None:
                        card.set_image("after", after)
        finally:
            self.setUpdatesEnabled(True)
        bar.setValue(pos)

    def _apply_open_size(self) -> None:



        target_w, target_h = 1220, 880
        floor_w, floor_h = 640, 480







        try:
            screen = self.screen() or QGuiApplication.primaryScreen()
        except (AttributeError, RuntimeError):
            screen = QGuiApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            target_w = min(target_w, int(avail.width() * 0.96))
            target_h = min(target_h, int(avail.height() * 0.92))
            floor_w = min(floor_w, target_w)
            floor_h = min(floor_h, target_h)
        self.setMinimumSize(floor_w, floor_h)
        self.resize(target_w, target_h)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        self._search = QLineEdit()
        self._search.setPlaceholderText(tr("Search objects"))
        self._search.setClearButtonEnabled(True)
        style_search_pill(self._search)
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(
            dial_in_range("tuning.library.search_debounce_ms", 180, 50, 1000))
        self._search_timer.timeout.connect(self._apply_search)
        self._search.textChanged.connect(lambda _t: self._search_timer.start())



        self._page_header = LibraryPageHeader(self._search)

        body = QHBoxLayout()
        body.setSpacing(8)


        body.addWidget(self._build_library_rail())





        content = QVBoxLayout()
        content.setContentsMargins(16, 8, 4, 0)
        content.setSpacing(16)
        content.addWidget(self._page_header)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtC.FrameNoFrame)
        apply_quiet_scrollbar(self._scroll)
        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 2, 12, 2)
        self._grid.setHorizontalSpacing(_GRID_SPACING)
        self._grid.setVerticalSpacing(_GRID_SPACING)
        for c in range(self._cols):
            self._grid.setColumnStretch(c, 1)
        self._scroll.setWidget(self._grid_host)
        content.addWidget(self._scroll, 1)




        self._hist_older_btn = QPushButton(tr("Load older runs"))
        self._hist_older_btn.setStyleSheet(_GHOST_BTN_QSS)
        self._hist_older_btn.setCursor(QtC.PointingHandCursor)




        self._hist_older_btn.setAutoDefault(False)
        self._hist_older_btn.setVisible(False)
        self._hist_older_btn.clicked.connect(self._load_older_runs)
        content.addWidget(self._hist_older_btn, 0, QtC.AlignCenter)
        body.addLayout(content, 1)

        root.addLayout(body, 1)


        self._lazy_timer = QTimer(self)
        self._lazy_timer.setSingleShot(True)
        self._lazy_timer.setInterval(
            dial_in_range("tuning.library.lazy_scroll_debounce_ms", 50, 10, 500))
        self._lazy_timer.timeout.connect(self._load_visible_cards)
        self._scroll.verticalScrollBar().valueChanged.connect(
            lambda _v: self._lazy_timer.start())
        self._scroll.viewport().installEventFilter(self)
        self._cols = self._column_count()

    def _refresh_library_chrome(self) -> None:




        self._refresh_rail_counts()
        view = _RAIL_HISTORY_VIEWS.get(self._active_key)
        self._hist_older_btn.setVisible(
            view is not None and bool(self._hist_has_more.get(view)))
        loading = view is not None and view in self._hist_inflight
        self._hist_older_btn.setEnabled(not loading)
        self._hist_older_btn.setText(
            tr("Loading...") if loading else tr("Load older runs"))

    def _track_tab_opened(self, tab: str) -> None:
        if tab in self._tabs_tracked:
            return
        self._tabs_tracked.add(tab)
        try:
            from ....core import telemetry_session_events
            telemetry_session_events.track_library_opened(
                tab, source=getattr(self, "_open_source", "chip"))
        except Exception:
            pass  # nosec B110



    def _select_tab(self, key: str) -> None:
        self._active_key = key
        self._set_rail_active(key)



        view = _RAIL_HISTORY_VIEWS.get(key)
        if view is not None:
            self._track_tab_opened("history")
            self._sync_history_view(view)
        self._rebuild_current_grid()
        self._refresh_library_chrome()

    def _paint_page_header(self) -> None:

        header = getattr(self, "_page_header", None)
        if header is None:
            return
        key = self._active_key
        view = _RAIL_HISTORY_VIEWS.get(key)
        if view is not None:
            title = tr("Recent") if key == _RAIL_RECENT_TARGET else tr("Favorites")
            header.set_text(title, "", _category_glyph(key))
        elif self._query or key == _RAIL_HOME_TARGET:
            header.set_text(tr("Examples"), self._home_line())
        else:
            count = len(self._presets_for_tab(key))
            line = (tr("1 object") if count == 1
                    else tr("{n} objects").format(n=_fmt_count(count)))
            header.set_text(self._tab_label(key), line, _category_glyph(key))

    @staticmethod
    def _home_line() -> str:
        return tr("Objects the AI can find on your imagery. Open one to see it.")

    def _tab_label(self, key: str) -> str:
        if key == _RAIL_POPULAR_TARGET:
            return tr("Popular")
        for cat in self._categories:
            if cat.get("key") == key:
                return pick_label(cat.get("label"), cat.get("key", ""))
        return tr("Examples")

    def _rebuild_current_grid(self) -> None:










        if self._detail_open:
            self._catalog_rebuild_pending = True
            return
        self._paint_page_header()


        cols = self._column_count()
        cols_changed = cols != self._cols
        self._cols = cols
        view = _RAIL_HISTORY_VIEWS.get(self._active_key)
        if view is not None:
            self._rebuild_history_grid(view)
            if cols_changed and self._grid_widgets:
                self._apply_grid_positions()
            return
        if self._query:
            self._rebuild_grid(
                self._search_matches(self._query),
                tr("No object matches that search."), "search", counted=True)
            return
        if self._active_key == _RAIL_HOME_TARGET:
            self._rebuild_home_shelves()
            return
        presets = self._presets_for_tab(self._active_key)
        if not presets and self._active_key != _RAIL_POPULAR_TARGET:




            self._select_tab(_RAIL_HOME_TARGET)
            return
        self._rebuild_grid(presets, tr("Nothing in this category yet."))

    def _home_shelf_rows(self) -> list[tuple[str, str, list[dict]]]:


        rows = []
        popular = self._presets_for_tab(_RAIL_POPULAR_TARGET)
        if popular:
            rows.append((_RAIL_POPULAR_TARGET, tr("Popular"), popular))
        for cat in self._categories:
            presets = list(cat.get("presets", []))
            if presets:
                rows.append((str(cat.get("key") or ""),
                             pick_label(cat.get("label"), cat.get("key", "")),
                             presets))
        return rows

    def _rebuild_home_shelves(self) -> None:

        self._clear_grid()
        rows = self._home_shelf_rows()
        if not rows:
            self._empty_label(tr("Nothing in this category yet."))
            return
        sections = []
        for key, title, presets in rows:
            tiles = self._build_preset_cards(presets)
            shelf = LibraryShelf(
                key, title, tiles,
                lambda k=key: self._on_rail_click(k),
                self._lazy_timer.start, self._grid_host)
            sections.append((shelf, []))
        self._place_grouped_grid(sections)
        self._schedule_visible_cards_load()

    def _apply_search(self) -> None:
        self._query = self._search.text().strip().lower()



        in_history = _RAIL_HISTORY_VIEWS.get(self._active_key) is not None
        self._set_rail_active(
            self._active_key if in_history or not self._query else None)
        self._rebuild_current_grid()

    def _search_matches(self, query: str) -> list[dict]:





        ranked = []
        for p in self._by_id.values():
            rank = preset_query_rank(
                p, query, self._cat_label_by_id.get(p.get("id", ""), ""))
            if rank is not None:
                ranked.append((rank, p))
        return [p for _rank, p in sorted(ranked, key=lambda item: item[0])]

    def _presets_for_tab(self, key: str) -> list[dict]:
        if key == _RAIL_POPULAR_TARGET:
            return [self._by_id[i] for i in self._top_picks if i in self._by_id]
        for cat in self._categories:
            if cat["key"] == key:
                return list(cat.get("presets", []))
        return []



    def _clear_grid(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._cards_by_id.clear()
        self._hist_cards.clear()
        self._thumb_cards.clear()
        self._run_cards.clear()
        self._grid_widgets = []
        self._grid_sections = None
        self._grid_span_all = False
        self._grid_signature = None

    def _column_count(self) -> int:






        try:
            width = self._scroll.viewport().width()
        except RuntimeError:
            return _GRID_COLS_DEFAULT
        if width <= 0:
            return _GRID_COLS_DEFAULT


        from ...dock.font_scale import scale_px_length
        if _RAIL_HISTORY_VIEWS.get(self._active_key) is None:
            step = scale_px_length(_TILE_MIN_W) + _GRID_SPACING
            return max(_GRID_COLS_MIN,
                       min(_CATALOG_COLS, (width + _GRID_SPACING) // step))
        step = scale_px_length(_CARD_MIN_W) + _GRID_SPACING
        return max(_GRID_COLS_MIN, min(_GRID_COLS_MAX, (width + _GRID_SPACING) // step))

    def _place_grid(self, widgets: list, span_all: bool = False) -> None:

        self._grid_widgets = list(widgets)
        self._grid_sections = None
        self._grid_span_all = bool(span_all)
        self._settle_grid_widgets()
        self._apply_grid_positions()
        self._update_count_label(0 if span_all else len(self._grid_widgets))

    def _place_grouped_grid(self, sections: list) -> None:







        self._grid_sections = [(header, list(cards)) for header, cards in sections]
        self._grid_widgets = []
        for header, cards in self._grid_sections:
            self._grid_widgets.append(header)
            self._grid_widgets.extend(cards)
        self._grid_span_all = False
        self._settle_grid_widgets()
        self._apply_grid_positions()
        self._update_count_label(
            sum(len(cards) for _h, cards in self._grid_sections))

    def _settle_grid_widgets(self) -> None:






        from ...dock.font_scale import apply_font_scale_to_tree

        for widget in self._grid_widgets:
            apply_font_scale_to_tree(widget)
            settle = getattr(widget, "on_font_scale_applied", None)
            if settle is not None:
                settle()

    def _update_count_label(self, count: int) -> None:


        header = getattr(self, "_page_header", None)
        if header is None:
            return
        in_history = _RAIL_HISTORY_VIEWS.get(self._active_key) is not None
        if not (self._query or in_history):
            return
        if count <= 0:
            header.set_subtitle(self._home_line() if not in_history else "")
        elif count == 1:
            header.set_subtitle(tr("1 result"))
        else:
            header.set_subtitle(tr("{n} results").format(n=_fmt_count(count)))

    def _apply_grid_positions(self) -> None:
        cols = self._cols
        while self._grid.count():
            self._grid.takeAt(0)
        for c in range(max(self._grid.columnCount(), _GRID_COLS_MAX)):
            self._grid.setColumnStretch(c, 0)





        for r in range(self._grid.rowCount()):
            self._grid.setRowStretch(r, 0)
        if self._grid_span_all:
            if self._grid_widgets:
                self._grid.addWidget(self._grid_widgets[0], 0, 0, 1, cols)


                self._grid.setRowStretch(0, 1)
        elif self._grid_sections is not None:
            row = 0
            for header, cards in self._grid_sections:
                self._grid.addWidget(header, row, 0, 1, cols)
                row += 1
                for idx, card in enumerate(cards):
                    self._grid.addWidget(card, row + idx // cols, idx % cols)
                row += (len(cards) + cols - 1) // cols
            self._grid.setRowStretch(row, 1)
        else:
            for idx, widget in enumerate(self._grid_widgets):
                self._grid.addWidget(widget, idx // cols, idx % cols)
            rows = (len(self._grid_widgets) + cols - 1) // cols
            self._grid.setRowStretch(rows, 1)
        for c in range(cols):
            self._grid.setColumnStretch(c, 1)

    def _reflow_if_needed(self) -> None:
        cols = self._column_count()
        if cols != self._cols:
            self._cols = cols
            self._apply_grid_positions()
            self._lazy_timer.start()

    def resizeEvent(self, ev):  # noqa: N802
        super().resizeEvent(ev)
        self._reflow_if_needed()

    def eventFilter(self, obj, event):  # noqa: N802






        try:
            if obj is self._scroll.viewport() and event.type() == _EVENT_RESIZE:
                self._reflow_if_needed()
        except (RuntimeError, AttributeError):
            pass  # nosec B110
        try:
            return super().eventFilter(obj, event)
        except RuntimeError:
            return False

    def _empty_label(self, text: str, glyph: str = "layers",
                     action: tuple | None = None) -> None:





        host = QWidget(self._grid_host)
        outer = QHBoxLayout(host)
        outer.setContentsMargins(20, 40, 20, 40)
        outer.addStretch()
        inner_host = QWidget(host)
        from ...dock.font_scale import scale_px_length

        inner_host.setMaximumWidth(scale_px_length(360))
        inner = QVBoxLayout(inner_host)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(10)



        inner.addStretch()

        mark = QLabel()
        mark.setAlignment(QtC.AlignCenter)
        mark.setStyleSheet(_EMPTY_GLYPH)
        mark.setPixmap(_empty_glyph_pixmap(mark, glyph))
        inner.addWidget(mark)
        msg = QLabel(text)
        msg.setWordWrap(True)
        msg.setAlignment(QtC.AlignCenter)
        msg.setStyleSheet(_EMPTY_MSG)
        inner.addWidget(msg)
        if action is not None:
            label, callback = action
            btn = QPushButton(label, inner_host)
            btn.setStyleSheet(_GHOST_BTN_QSS)
            btn.setAutoDefault(False)
            btn.setCursor(QtC.PointingHandCursor)
            btn.clicked.connect(callback)
            row = QHBoxLayout()
            row.addStretch()
            row.addWidget(btn)
            row.addStretch()
            inner.addLayout(row)
        inner.addStretch()



        outer.addWidget(inner_host, 0, QtC.AlignCenter)
        outer.addStretch()
        self._place_grid([host], span_all=True)

    def _build_preset_cards(self, presets: list[dict]) -> list:
        cards = []
        for preset in presets:
            pid = str(preset.get("id") or "")
            card = _PresetTile(
                preset, self._grid_host,
                category_key=self._cat_key_by_id.get(pid, ""),
                category_label=self._cat_label_by_id.get(pid, ""))
            card.activated.connect(self._open_detail)
            card.star_toggled.connect(self._toggle_template_favorite)
            card.set_favorite(is_favorite_template(pid))
            cards.append(card)
            if pid:
                self._cards_by_id.setdefault(pid, []).append(card)
        return cards

    def _rebuild_grid(self, presets: list[dict], empty_text: str,
                      empty_glyph: str = "layers", counted: bool = False) -> None:






        self._clear_grid()
        if not presets:
            self._empty_label(empty_text, empty_glyph)
            self._update_count_label(0)
            return
        self._place_grid(self._build_preset_cards(presets))
        if counted:
            self._update_count_label(len(presets))



        self._schedule_visible_cards_load()

    def _schedule_visible_cards_load(self) -> None:

        QtC.safe_single_shot(0, self, self._load_visible_cards)
        settle_ms = dial_in_range("tuning.library.layout_settle_ms", 80, 20, 500)
        QtC.safe_single_shot(settle_ms, self, self._load_visible_cards)



    def _load_visible_cards(self) -> None:





        if not self._cards_by_id and not self._run_cards:
            return
        try:
            viewport = self._scroll.viewport()
            vp_h = viewport.height()
        except RuntimeError:
            return
        margin = max(vp_h, 1)

        vp_w = max(viewport.width(), 1)

        def near_viewport(card) -> bool:
            try:
                at = card.mapTo(viewport, QPoint(0, 0))
            except RuntimeError:
                return False


            return (at.y() + card.height() >= -margin and at.y() <= vp_h + margin
                    and at.x() + card.width() >= -vp_w and at.x() <= 2 * vp_w)

        for cards in list(self._cards_by_id.values()):
            for card in cards:
                if near_viewport(card):
                    card.request_demos(self._loader, self._base)
        for run, card in list(self._run_cards):
            if near_viewport(card):
                self._request_run_thumb(run, card)

    def _on_demo_loaded(self, pid: str, which: str, pixmap) -> None:
        if which not in ("before", "after"):
            return
        for card in self._cards_by_id.get(pid, []):
            try:
                card.set_image(which, pixmap)
            except RuntimeError:
                pass  # nosec B110

    def _on_demo_failed(self, pid: str, which: str) -> None:
        if which not in ("before", "after"):
            return
        for card in self._cards_by_id.get(pid, []):
            try:
                card.mark_missing(which)
            except RuntimeError:
                pass  # nosec B110

    def _toggle_template_favorite(self, preset: dict, _checked: bool) -> None:


        pid = str(preset.get("id") or "")
        if not pid:
            return
        toggle_favorite_template(pid)

        starred = is_favorite_template(pid)
        for card in self._cards_by_id.get(pid, []):
            try:
                card.set_favorite(starred)
            except RuntimeError:
                pass  # nosec B110
        self._refresh_library_chrome()
        if self._active_key == _RAIL_FAVORITES_TARGET and not self._query:


            QtC.safe_single_shot(0, self, self._rebuild_current_grid)



    def _open_detail(self, preset: dict) -> None:



        if self._detail_open or self._selected_prompt is not None:
            return
        self._detail_open = True
        try:
            dlg = _PresetDetailDialog(
                preset, self._base, self,
                category_label=self._cat_label_by_id.get(preset.get("id", ""), ""))
            self._open_preset_detail = dlg
            try:
                dlg.exec()
            finally:
                self._open_preset_detail = None
            if dlg.chosen and not self._view_only:
                if dlg.tried_raster_path:
                    self._selected_raster = {
                        "path": dlg.tried_raster_path,
                        "label": pick_label(preset.get("label"), preset.get("prompt", "")),
                        "preset_id": preset.get("id", ""),
                    }
                else:
                    self._track_example_picked(preset)
                self._select(preset)
        finally:
            self._end_detail()

    def _on_recent_activated(self, entry: dict) -> None:





        if self._view_only:
            return
        restore_recent_on_map(self._plugin, entry)
        self._select(entry)

    def _on_recent_rerun(self, entry: dict) -> None:




        if self._view_only:
            return
        dock = self._dock_widget()
        if dock is None:
            return
        self.reject()
        dock.history_rerun_requested.emit(dict(entry))

    def _on_recent_reuse_prompt(self, entry: dict) -> None:


        if self._view_only:
            return
        dock = self._dock_widget()
        if dock is None:
            return
        prompt = (entry.get("prompt") or "").strip()
        self.reject()
        dock.history_reuse_prompt_requested.emit(prompt)

    def _dock_widget(self):



        dock = getattr(self._plugin, "dock_widget", None)
        if dock is not None:
            return dock
        parent = self.parent()
        return parent if hasattr(parent, "history_rerun_requested") else None

    def _select(self, preset: dict) -> None:
        self._selected_prompt = preset.get("prompt", "")
        self.accept()

    def get_selected_prompt(self) -> str | None:
        return self._selected_prompt

    def get_selected_raster(self) -> dict | None:




        return self._selected_raster

    def _track_example_picked(self, preset: dict) -> None:
        key = str(self._active_key or "")
        if self._query:
            tab = "search"
        elif key == _RAIL_HOME_TARGET:
            tab = "home"
        elif key == _RAIL_POPULAR_TARGET:
            tab = "popular"
        else:
            tab = _RAIL_HISTORY_VIEWS.get(key) or key
        try:
            from ....core import telemetry_session_events
            telemetry_session_events.track_library_example_picked(
                str(preset.get("id", "")), tab)
        except Exception:
            pass  # nosec B110



    def _track_live_worker(self, worker, *signal_names: str) -> None:

        self._live_workers.append((worker, signal_names))

    def done(self, result):  # noqa: N802







        workers = self._live_workers
        self._live_workers = []
        self._fetch_worker = None
        for worker, signal_names in workers:
            for signal_name in signal_names:
                safe_disconnect(worker, signal_name)
            try:
                worker.requestInterruption()
            except (RuntimeError, TypeError, AttributeError):
                pass  # nosec B110
        self._close_fetch_progress()
        super().done(result)
