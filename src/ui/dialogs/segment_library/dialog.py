"""Segment library: a visual gallery of cloud-model object prompts with before/after
previews. Mirrors AI Edit's prompt-library dialog (one sidebar: the user's own
detections on top, curated templates below, searchable card grid + detail
popup), trimmed to what segmentation needs.

Picking a card returns the preset's **English token** (the literal cloud-model
prompt), which the dock drops into the prompt box. Labels are localized;
tokens are not.

Performance: the catalogue is read from a non-blocking cache (the network
prefetch is the plugin's job), and demo images load lazily per visible card so
the first paint never waits on the whole grid.

The old top-level [ Detect | History ] switch is gone: there is ONE place for
past work. "Recent" lists every cloud run (server-side, warm-started from a
local cache, falling back to the signed-out local recents), with one-click
prompt reuse plus restore-to-review, direct export and favorites. Every
detection the user has made is kept here (there is no delete); runs are the
user's own segmentation history to reuse, restore or export at any time.
The local fallback renders from core/detection_history.py
(zone thumbnail + extent + exported layer name recorded at Finish): clicking
a recent card reuses the prompt AND restores the map (zoom back to the zone,
re-activate the exported layer when it is still in the project).
All history network calls run on QThread workers; when the
history endpoints are not deployed yet the tabs degrade to their empty states
(no error spam). The dialog tolerates plugin=None: history actions that need
the plugin (Restore / Export) are disabled with a tooltip.
"""
from __future__ import annotations

from qgis.core import Qgis
from qgis.PyQt.QtCore import QEvent, QPoint, QTimer
from qgis.PyQt.QtGui import QGuiApplication
from qgis.PyQt.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ....core import detection_history
from ....core import qt_compat as QtC
from ....core.i18n import tr
from ....core.logging_utils import log
from ....core.presets import run_history_cache, segment_history
from ....core.presets.segmentation_presets import pick_label, preset_matches_query
from ....core.presets.segmentation_presets_client import (
    base_url,
    cached_or_offline_catalog,
)
from ....core.presets.template_favorites import (
    favorite_template_ids,
    is_favorite_template,
    toggle_favorite_template,
)
from ....core.qt_compat import safe_disconnect
from ...dock.font_scale import scale_px_length
from ...plugin.shared import park_orphaned_worker
from ...template_demo_loader import TemplateDemoLoader
from .cards import _PresetCard, _RecentCard, _RunCard
from .common import (
    _EMPTY_GLYPH,
    _EMPTY_MSG,
    _GHOST_BTN_QSS,
    _META_QSS,
    _SEARCH_QSS,
    _SECTION_HEADER,
    _SIDEBAR_ITEM,
    _SIDEBAR_ITEM_ACTIVE,
    _fmt_count,
    _run_key,
    _sidebar_icon_html,
    _SidebarButton,
    _tab_label_html,
)
from .detail import (
    _ExportRunDialog,
    _PresetDetailDialog,
    _RunDetailDialog,
    _RunProgressDialog,
)
from .recent_local import merge_local_recents, recent_view, restore_recent_on_map
from .workers import (
    _HistoryFetchWorker,
    _RunFavoriteWorker,
    _RunFetchWorker,
    _RunZoneFetchWorker,
)

# Sidebar keys for the synthetic (non-category) tabs. History views map onto
# the server-side view names via _HISTORY_VIEWS.
_RECENT_KEY = "__recent__"
_FAVORITES_KEY = "__favorites__"
_TOP_KEY = "__top__"
_HISTORY_VIEWS = {
    _RECENT_KEY: "all",
    _FAVORITES_KEY: "favorites",
}
# The grid follows the dialog width instead of pinning a column count: three
# columns squeeze on a narrow dialog and over-stretch on a wide one. The bounds
# keep a preview big enough to read and stop the cards turning into a contact
# sheet on a very wide screen.
_CARD_MIN_W = 270
_GRID_SPACING = 12
_GRID_COLS_MIN = 1
_GRID_COLS_MAX = 5
_GRID_COLS_DEFAULT = 3
# Resolved through qt_compat so the scoped/flat enum split stays out of the
# Qt6 static check.
_EVENT_RESIZE = QtC.resolve_qt_enum(QEvent, "Type", "Resize")


class SegmentLibraryDialog(QDialog):
    """The gallery. ``get_selected_prompt()`` returns the chosen English token.

    The catalogue and the recent list are read non-blocking (cache / QSettings):
    nothing here touches the network, so the dialog always opens instantly.
    """

    def __init__(self, parent=None, *, recent: list[dict] | None = None,
                 plugin=None, view_only: bool = False):
        super().__init__(parent)
        # view_only: opened while a detection run / review is in flight. Browsing
        # (scroll, search, inspect, favorites) stays fully live, but every action
        # that would pick a prompt or start a run is inert, and the re-run buttons
        # on Recent cards grey out. Mirrors AI Edit's browse-only library.
        self._view_only = bool(view_only)
        self.setWindowTitle(
            tr("Segment library (view only)") if self._view_only
            else tr("Segment library"))
        self.setSizeGripEnabled(True)
        self._apply_open_size()
        self._selected_prompt: str | None = None
        self._detail_open = False
        self._base = base_url()
        # Non-blocking: cached server catalogue (ignoring TTL) or the bundled
        # offline one. The plugin's background prefetch keeps the cache warm.
        self._categories, self._top_picks = cached_or_offline_catalog()
        self._by_id = {
            p["id"]: p for cat in self._categories for p in cat.get("presets", [])}
        # token -> preset, so a recent object that matches a catalogue entry can
        # borrow its localized label; id -> category label for the detail badge.
        self._by_token = {
            p.get("prompt", ""): p for cat in self._categories
            for p in cat.get("presets", []) if p.get("prompt")}
        self._cat_label_by_id: dict[str, str] = {}
        for cat in self._categories:
            cat_label = pick_label(cat.get("label"), cat.get("key", ""))
            for p in cat.get("presets", []):
                self._cat_label_by_id[p.get("id", "")] = cat_label
        self._recent_local = (list(recent) if recent is not None
                              else segment_history.get_recent())
        # Rich local run history (zone extent + exported layer + thumbnail),
        # recorded at Finish. Read is fail-safe ([] on any problem).
        self._history_local = detection_history.get_entries()
        self._active_key = _TOP_KEY
        self._query = ""
        self._cards_by_id: dict[str, _PresetCard] = {}
        # Grid ownership: the widgets currently on show, and how many columns
        # they are spread over. Kept apart from the layout so a width change
        # re-places the same cards instead of rebuilding (and refetching) them.
        self._cols = _GRID_COLS_DEFAULT
        self._grid_widgets: list = []
        self._grid_span_all = False
        # What the grid currently paints, when it paints run history. Compared
        # against a fresh sync so an unchanged page costs nothing.
        self._grid_signature: tuple | None = None
        self._run_cards: list[tuple[dict, _RunCard]] = []

        self._loader = TemplateDemoLoader(self)
        self._loader.loaded.connect(self._on_demo_loaded)
        self._loader.failed.connect(self._on_demo_failed)

        # ---- run history state -------------------------------------------
        # plugin=None keeps template picking fully working; history actions
        # that need the plugin (Restore / Export) are disabled with a tooltip.
        self._plugin = plugin
        self._auth: dict = {}
        try:
            from ....core.activation_manager import get_auth_header
            self._auth = get_auth_header() or {}
        except Exception:  # noqa: BLE001 -- unsigned-in is a normal state
            self._auth = {}
        self._client = None  # lazy TerraLabClient, built on first history use
        self._hist_runs: dict[str, list[dict]] = {
            "all": run_history_cache.get_runs(), "favorites": []}
        self._hist_has_more = {"all": False, "favorites": False}
        self._hist_synced: set = set()
        self._hist_inflight: set = set()
        self._hist_fail_logged: set = set()
        self._hist_pages_loaded = 0
        # Two registries: run key -> card follows favorite toggles, archived
        # tile id -> card routes the loaded images back to the right preview.
        self._hist_cards: dict[str, _RunCard] = {}
        self._thumb_cards: dict[str, _RunCard] = {}
        self._hist_busy = False
        self._pending_action: tuple | None = None
        self._detail_dlg: _RunDetailDialog | None = None
        self._tabs_tracked: set = set()
        self._hist_loader = TemplateDemoLoader(self)
        self._hist_loader.loaded.connect(self._on_thumb_loaded)
        self._hist_loader.failed.connect(self._on_thumb_failed)

        self._build_ui()
        from ...dock.font_scale import apply_font_scale_to_tree

        apply_font_scale_to_tree(self)
        self._select_tab(_TOP_KEY)
        self._track_tab_opened("detect")

    # ---- UI scaffold -----------------------------------------------------

    def _apply_open_size(self) -> None:
        """Open large: hug the 220px sidebar + a 3-column card grid, grown
        toward the screen so the previews read big. Clamped to the available
        screen so it never spills offscreen (AI Edit's open-size rule)."""
        target_w, target_h = 1220, 880
        floor_w, floor_h = 640, 480
        # This dialog's own screen, not the primary one. A Windows desk is
        # commonly a scaled laptop panel next to an external monitor, and QGIS
        # sits on the second as often as on the first, so the primary screen's
        # geometry is the wrong ruler. The floor is clamped by the same read as
        # the target: a 1366x768 laptop at 175% text scaling has about 400
        # units of height, so a flat 480 floor was taller than the desktop and
        # nothing could bring the window back.
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
        self._search.setPlaceholderText(
            tr("Search objects... e.g. building, solar panel"))
        self._search.setClearButtonEnabled(True)
        self._search.setStyleSheet(_SEARCH_QSS)
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(180)
        self._search_timer.timeout.connect(self._apply_search)
        self._search.textChanged.connect(lambda _t: self._search_timer.start())
        search_row = QHBoxLayout()
        search_row.setContentsMargins(0, 0, 0, 0)
        search_row.setSpacing(10)
        search_row.addWidget(self._search, 1)
        # Trailing count, so an empty grid reads as a filter outcome rather than
        # a failure, and a full one says how much there is without counting.
        self._count_label = QLabel("")
        self._count_label.setStyleSheet(_META_QSS)
        search_row.addWidget(self._count_label)
        root.addLayout(search_row)

        body = QHBoxLayout()
        body.setSpacing(8)

        # Sidebar: the user's own detections first, curated templates below
        # (AI Edit's "Your prompts / Templates" grouping).
        sidebar_host = QWidget()
        sidebar_host.setFixedWidth(scale_px_length(220))
        self._sidebar = QVBoxLayout(sidebar_host)
        self._sidebar.setContentsMargins(0, 0, 0, 0)
        self._sidebar.setSpacing(2)
        self._tab_buttons: dict[str, _SidebarButton] = {}
        self._add_section(tr("Your detections"))
        self._add_tab(_RECENT_KEY, tr("Recent"))
        self._add_tab(_FAVORITES_KEY, tr("Favorites"))
        sep_wrap = QWidget()
        sep_wrap.setFixedHeight(13)
        sep_inner = QVBoxLayout(sep_wrap)
        sep_inner.setContentsMargins(12, 6, 12, 6)
        sep_line = QFrame()
        sep_line.setFixedHeight(1)
        sep_line.setStyleSheet("background: rgba(128,128,128,0.3); border: none;")
        sep_inner.addWidget(sep_line)
        self._sidebar.addWidget(sep_wrap)
        self._add_section(tr("Templates"))
        self._add_tab(_TOP_KEY, tr("Popular"))
        for cat in self._categories:
            self._add_tab(cat["key"], pick_label(cat.get("label"), cat.get("key", "")))
        self._sidebar.addStretch()
        self._hist_older_btn = QPushButton(tr("Load older runs"))
        self._hist_older_btn.setStyleSheet(_GHOST_BTN_QSS)
        self._hist_older_btn.setCursor(QtC.PointingHandCursor)
        self._hist_older_btn.setVisible(False)
        self._hist_older_btn.clicked.connect(self._load_older_runs)
        self._sidebar.addWidget(self._hist_older_btn)
        body.addWidget(sidebar_host)

        vsep = QFrame()
        vsep.setFrameShape(QtC.FrameVLine)
        vsep.setFrameShadow(QtC.FrameSunken)
        body.addWidget(vsep)

        # Card grid in a scroll area (shared by every tab + search).
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtC.FrameNoFrame)
        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(2, 2, 2, 2)
        self._grid.setHorizontalSpacing(_GRID_SPACING)
        self._grid.setVerticalSpacing(_GRID_SPACING)
        for c in range(self._cols):
            self._grid.setColumnStretch(c, 1)
        self._scroll.setWidget(self._grid_host)
        body.addWidget(self._scroll, 1)

        root.addLayout(body, 1)

        # Lazy demo loading: only fetch images for cards near the viewport.
        self._lazy_timer = QTimer(self)
        self._lazy_timer.setSingleShot(True)
        self._lazy_timer.setInterval(50)
        self._lazy_timer.timeout.connect(self._load_visible_cards)
        self._scroll.verticalScrollBar().valueChanged.connect(
            lambda _v: self._lazy_timer.start())
        self._scroll.viewport().installEventFilter(self)
        self._cols = self._column_count()

    def _add_section(self, label: str) -> None:
        # Sentence case: the design system bans uppercase across the plugin.
        lbl = QLabel(label)
        lbl.setStyleSheet(_SECTION_HEADER)
        self._sidebar.addWidget(lbl)

    def _add_tab(self, key: str, label: str) -> None:
        btn = _SidebarButton(
            _sidebar_icon_html(key),
            _tab_label_html(label, self._tab_count(key)))
        btn.setStyleSheet(_SIDEBAR_ITEM)
        btn.setCursor(QtC.PointingHandCursor)
        btn.setSizePolicy(QtC.SizePolicyExpanding, QtC.SizePolicyFixed)
        btn.clicked.connect(lambda _c=False, k=key: self._on_sidebar_click(k))
        self._sidebar.addWidget(btn)
        self._tab_buttons[key] = btn

    def _tab_count(self, key: str) -> int | None:
        """Muted '(N)' badge: only the personal tabs carry a count."""
        if key == _RECENT_KEY:
            runs = self._hist_runs.get("all") or []
            return len(runs) if runs else len(self._local_recent_entries())
        if key == _FAVORITES_KEY:
            return (len(self._hist_runs.get("favorites") or []) + len(self._favorite_template_presets()))
        return None

    def _tab_labels(self) -> dict[str, str]:
        labels = {
            _RECENT_KEY: tr("Recent"),
            _FAVORITES_KEY: tr("Favorites"),
            _TOP_KEY: tr("Popular"),
        }
        for cat in self._categories:
            labels[cat["key"]] = pick_label(cat.get("label"), cat.get("key", ""))
        return labels

    def _refresh_sidebar_counts(self) -> None:
        labels = self._tab_labels()
        for key in (_RECENT_KEY, _FAVORITES_KEY):
            btn = self._tab_buttons.get(key)
            if btn is not None:
                btn.set_label_html(
                    _sidebar_icon_html(key),
                    _tab_label_html(labels[key], self._tab_count(key)))
        self._hist_older_btn.setVisible(
            self._active_key in _HISTORY_VIEWS and bool(self._hist_has_more.get(_HISTORY_VIEWS[self._active_key])))

    def _track_tab_opened(self, tab: str) -> None:
        if tab in self._tabs_tracked:
            return
        self._tabs_tracked.add(tab)
        try:
            from ....core import telemetry_session_events
            telemetry_session_events.track_library_opened(tab)
        except Exception:
            pass  # nosec B110

    # ---- navigation ------------------------------------------------------

    def _on_sidebar_click(self, key: str) -> None:
        # Sidebar click is an explicit "leave search": clear the box quietly.
        if self._search.text().strip():
            self._search.blockSignals(True)
            self._search.clear()
            self._search.blockSignals(False)
            self._query = ""
        self._select_tab(key)

    def _select_tab(self, key: str) -> None:
        self._active_key = key
        for k, btn in self._tab_buttons.items():
            btn.setStyleSheet(_SIDEBAR_ITEM_ACTIVE if k == key else _SIDEBAR_ITEM)
        self._rebuild_current_grid()
        view = _HISTORY_VIEWS.get(key)
        if view is not None:
            self._track_tab_opened("history")
            self._sync_history_view(view)
        self._refresh_sidebar_counts()

    def _rebuild_current_grid(self) -> None:
        if self._query:
            self._rebuild_grid(self._search_matches(self._query))
            return
        view = _HISTORY_VIEWS.get(self._active_key)
        if view is not None:
            self._rebuild_history_grid(view)
        else:
            self._rebuild_grid(self._presets_for_tab(self._active_key))

    def _apply_search(self) -> None:
        self._query = self._search.text().strip().lower()
        self._rebuild_current_grid()

    def _search_matches(self, query: str) -> list[dict]:
        """Match over the token, the localized label, the category and any
        search terms the catalogue carries, accent-folded on both sides so
        "eolienne" reaches the same card as "éolienne"."""
        return [p for p in self._by_id.values()
                if preset_matches_query(
                    p, query, self._cat_label_by_id.get(p.get("id", ""), ""))]

    def _presets_for_tab(self, key: str) -> list[dict]:
        if key == _TOP_KEY:
            return [self._by_id[i] for i in self._top_picks if i in self._by_id]
        for cat in self._categories:
            if cat["key"] == key:
                return list(cat.get("presets", []))
        return []

    # ---- grid ------------------------------------------------------------

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
        self._grid_span_all = False
        self._grid_signature = None

    def _column_count(self) -> int:
        """How many cards fit across the viewport at the card's minimum width.

        A fixed column count either squeezes the cards on a narrow dialog or
        stretches three of them across a wide one; the grid follows the width
        instead, within bounds that keep a preview readable.
        """
        try:
            width = self._scroll.viewport().width()
        except RuntimeError:
            return _GRID_COLS_DEFAULT
        if width <= 0:
            return _GRID_COLS_DEFAULT
        step = _CARD_MIN_W + _GRID_SPACING
        return max(_GRID_COLS_MIN, min(_GRID_COLS_MAX, (width + _GRID_SPACING) // step))

    def _place_grid(self, widgets: list, span_all: bool = False) -> None:
        """Own the grid's contents, then lay them out at the current width."""
        self._grid_widgets = list(widgets)
        self._grid_span_all = bool(span_all)
        # Cards are built when their tab is picked, long after the window was,
        # so the pass the window made over itself never saw them.
        from ...dock.font_scale import apply_font_scale_to_tree

        for widget in self._grid_widgets:
            apply_font_scale_to_tree(widget)
        self._apply_grid_positions()
        self._update_count_label(0 if span_all else len(self._grid_widgets))

    def _update_count_label(self, count: int) -> None:
        label = getattr(self, "_count_label", None)
        if label is None:
            return
        if count <= 0:
            label.setText("")
        elif count == 1:
            label.setText(tr("1 result"))
        else:
            label.setText(tr("{n} results").format(n=_fmt_count(count)))

    def _apply_grid_positions(self) -> None:
        cols = self._cols
        while self._grid.count():
            self._grid.takeAt(0)  # detaches the item; the widget keeps its parent
        for c in range(max(self._grid.columnCount(), _GRID_COLS_MAX)):
            self._grid.setColumnStretch(c, 0)
        # The scroll area resizes the host to the viewport, so a grid shorter
        # than one screen has spare height to give away. Rows must not take it:
        # two rows of cards would drift apart and a lone card would float in
        # the middle. Every row of cards stays at 0 and a trailing spacer row
        # absorbs the slack, which pins the block to the top left.
        for r in range(self._grid.rowCount()):
            self._grid.setRowStretch(r, 0)
        if self._grid_span_all:
            if self._grid_widgets:
                self._grid.addWidget(self._grid_widgets[0], 0, 0, 1, cols)
                # The hero empty state is the one thing that owns the height:
                # it centres itself inside its own row.
                self._grid.setRowStretch(0, 1)
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

    def resizeEvent(self, ev):  # noqa: N802 - Qt signature
        super().resizeEvent(ev)
        self._reflow_if_needed()

    def eventFilter(self, obj, event):  # noqa: N802 - Qt signature
        # The column count depends on the viewport, not on the dialog: the
        # viewport keeps resizing after the dialog has settled (scrollbar
        # appearing, first layout pass), and watching the dialog alone leaves
        # the grid stuck on whatever width it guessed before any of that.
        try:
            is_viewport = obj is self._scroll.viewport()
        except RuntimeError:
            is_viewport = False
        if is_viewport and event.type() == _EVENT_RESIZE:
            self._reflow_if_needed()
        return super().eventFilter(obj, event)

    def _empty_label(self, text: str, glyph: str = "◇") -> None:
        """Hero empty state: one glyph, one sentence, centered, nothing else."""
        host = QWidget(self._grid_host)
        outer = QHBoxLayout(host)
        outer.setContentsMargins(20, 40, 20, 40)
        outer.addStretch()
        inner_host = QWidget(host)
        inner_host.setMaximumWidth(360)
        inner = QVBoxLayout(inner_host)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(10)
        # Stretches inside, not just an alignment outside: the empty grid cell
        # is the full height of the viewport, and without them the layout hands
        # that height to the two labels and drives them to opposite edges.
        inner.addStretch()
        mark = QLabel(glyph)
        mark.setAlignment(QtC.AlignCenter)
        mark.setStyleSheet(_EMPTY_GLYPH)
        inner.addWidget(mark)
        msg = QLabel(text)
        msg.setWordWrap(True)
        msg.setAlignment(QtC.AlignCenter)
        msg.setStyleSheet(_EMPTY_MSG)
        inner.addWidget(msg)
        inner.addStretch()
        # Centered, not stretched: an HBox grows its child to the full cell
        # height unless an alignment is set, which pushes the glyph to the top
        # and the sentence to the bottom of an otherwise empty grid.
        outer.addWidget(inner_host, 0, QtC.AlignCenter)
        outer.addStretch()
        self._place_grid([host], span_all=True)

    def _build_preset_cards(self, presets: list[dict]) -> list:
        cards = []
        for preset in presets:
            card = _PresetCard(preset, self._grid_host)
            card.activated.connect(self._open_detail)
            card.star_toggled.connect(self._toggle_template_favorite)
            card.set_favorite(is_favorite_template(preset.get("id", "")))
            cards.append(card)
            self._cards_by_id[preset["id"]] = card
        return cards

    def _build_run_cards(self, runs: list[dict], view: str) -> list:
        cards = []
        for run in runs:
            card = _RunCard(run, view, parent=self._grid_host)
            card.opened.connect(self._open_run_detail)
            card.star_toggled.connect(self._toggle_favorite)
            cards.append(card)
            self._run_cards.append((run, card))
            # Registered up front, not when the images are requested: an
            # off-screen card still has to follow a favorite toggle.
            self._hist_cards[_run_key(run)] = card
        return cards

    def _rebuild_grid(self, presets: list[dict]) -> None:
        self._clear_grid()
        if not presets:
            self._empty_label(tr("No object matches that search."), "⌕")
            return
        self._place_grid(self._build_preset_cards(presets))
        # Kick lazy loading for whatever is visible now + once layout settles.
        QTimer.singleShot(0, self._load_visible_cards)
        QTimer.singleShot(80, self._load_visible_cards)

    def _history_grid_signature(self, view: str) -> tuple:
        """Everything a history card reads. Two equal signatures paint the
        same grid, so the rebuild that follows a sync can be skipped."""
        runs = self._hist_runs.get(view) or []
        presets = (tuple(p.get("id", "") for p in self._favorite_template_presets())
                   if view == "favorites" else ())
        return (view, presets, tuple(
            (_run_key(r), bool(r.get("is_favorite")), r.get("objects"),
             r.get("tiles"), r.get("preview_request_id")) for r in runs))

    def _rebuild_history_grid(self, view: str) -> None:
        # The dialog opens on the cached page, then a background sync answers
        # with what is almost always the same page. Rebuilding then throws away
        # a dozen live cards and their decoded images only to build the same
        # ones back, which on a slow machine is the longest stall of the open.
        signature = self._history_grid_signature(view)
        if signature == self._grid_signature and self._grid_widgets:
            # Same runs, kept cards. Hand them the fresh payload so nothing on
            # the card keeps reading the copy that came off the disk cache.
            fresh = {_run_key(r): r for r in (self._hist_runs.get(view) or [])}
            self._run_cards = [
                (fresh.get(_run_key(run), run), card)
                for run, card in self._run_cards]
            for run, card in self._run_cards:
                card.adopt_run(run)
            return
        self._clear_grid()
        self._grid_signature = signature
        runs = self._hist_runs.get(view) or []
        cards: list = []
        if view == "favorites":
            # One tab, both kinds of star: the objects the user keeps around,
            # then the detections they kept. Objects come first because they are
            # what a new run starts from.
            cards.extend(self._build_preset_cards(self._favorite_template_presets()))
        if not runs and not cards:
            if view == "all" and self._local_recent_entries():
                # Signed-out / endpoint-less fallback: the local run history
                # still gives one-click restore of past detections.
                self._rebuild_recent_local_grid()
                return
            if view == "favorites":
                self._empty_label(
                    tr("Star a detection or an object to keep it here."), "★")
            else:
                self._empty_label(
                    tr("Nothing here yet. Your automatic detections will "
                       "land here, ready to reuse, restore or export."))
            return
        cards.extend(self._build_run_cards(runs, view))
        self._place_grid(cards)
        QTimer.singleShot(0, self._load_visible_cards)
        QTimer.singleShot(80, self._load_visible_cards)

    def _favorite_template_presets(self) -> list[dict]:
        """Starred templates, most recently starred first."""
        return [self._by_id[i] for i in favorite_template_ids() if i in self._by_id]

    def _local_recent_entries(self) -> list[dict]:
        """The Recent tab's local feed (see recent_local.merge_local_recents)."""
        return merge_local_recents(self._history_local, self._recent_local)

    def _rebuild_recent_local_grid(self) -> None:
        cards = []
        for entry in self._local_recent_entries():
            card = _RecentCard(recent_view(entry, self._by_token), self._grid_host,
                               view_only=self._view_only)
            card.activated.connect(self._on_recent_activated)
            card.rerun_requested.connect(self._on_recent_rerun)
            card.reuse_prompt_requested.connect(self._on_recent_reuse_prompt)
            cards.append(card)
        self._place_grid(cards)

    # ---- demo image routing ---------------------------------------------

    def _load_visible_cards(self) -> None:
        """Request images only for cards in or near the viewport (one screen of
        lookahead), so opening a big category never fires every fetch at once.

        Run cards matter most here: each one pulls two full archived tiles.
        """
        if not self._cards_by_id and not self._run_cards:
            return
        try:
            viewport = self._scroll.viewport()
            vp_h = viewport.height()
        except RuntimeError:
            return
        margin = max(vp_h, 1)  # one screen of lookahead either way

        def near_viewport(card) -> bool:
            try:
                top = card.mapTo(viewport, QPoint(0, 0)).y()
            except RuntimeError:
                return False  # card already torn down
            return top + card.height() >= -margin and top <= vp_h + margin

        for card in list(self._cards_by_id.values()):
            if near_viewport(card):
                card.request_demos(self._loader, self._base)
        for run, card in list(self._run_cards):
            if near_viewport(card):
                self._request_run_thumb(run, card)

    def _on_demo_loaded(self, pid: str, which: str, pixmap) -> None:
        card = self._cards_by_id.get(pid)
        if card is not None and which in ("before", "after"):
            card.set_image(which, pixmap)

    def _on_demo_failed(self, pid: str, which: str) -> None:
        card = self._cards_by_id.get(pid)
        if card is not None and which in ("before", "after"):
            card.mark_missing(which)

    # ---- history sync ------------------------------------------------------

    def _history_client(self):
        if self._client is None:
            from ....api.terralab_client import TerraLabClient
            self._client = TerraLabClient()
        return self._client

    def _sync_history_view(self, view: str, before: str | None = None) -> None:
        """Refresh one view from the server, off the GUI thread. Silently a
        no-op when not signed in or when a sync is already in flight."""
        if not self._auth:
            return
        if before is None and view in self._hist_synced:
            return
        if view in self._hist_inflight:
            return
        self._hist_inflight.add(view)
        worker = _HistoryFetchWorker(
            self._history_client(), self._auth, view, before)
        worker.page_fetched.connect(self._on_history_page)
        worker.failed.connect(self._on_history_failed)
        park_orphaned_worker(worker)
        worker.start()

    def _load_older_runs(self) -> None:
        view = _HISTORY_VIEWS.get(self._active_key)
        if view is None:
            return
        runs = self._hist_runs.get(view) or []
        if not runs:
            return
        oldest = runs[-1].get("started_at") or runs[-1].get("created_at")
        if not oldest:
            return
        self._sync_history_view(view, before=str(oldest))

    def _displayed_view(self) -> str | None:
        return _HISTORY_VIEWS.get(self._active_key) if not self._query else None

    def _on_history_page(self, view: str, runs: list, has_more: bool,
                         first: bool) -> None:
        self._hist_inflight.discard(view)
        self._hist_synced.add(view)
        if first:
            self._hist_runs[view] = runs
            if view == "all":
                run_history_cache.save_runs(runs)
                try:
                    from ....core import telemetry_session_events
                    telemetry_session_events.track_history_synced(len(runs))
                except Exception:
                    pass  # nosec B110
        else:
            known = {_run_key(r) for r in self._hist_runs[view]}
            self._hist_runs[view].extend(
                r for r in runs if _run_key(r) not in known)
            self._hist_pages_loaded += 1
            try:
                from ....core import telemetry_session_events
                telemetry_session_events.track_history_page_loaded(self._hist_pages_loaded)
            except Exception:
                pass  # nosec B110
        self._hist_has_more[view] = has_more
        if view == self._displayed_view():
            self._rebuild_history_grid(view)
        self._refresh_sidebar_counts()

    def _on_history_failed(self, view: str, code: str) -> None:
        """A failed sync (including the endpoints not deployed yet) degrades to
        the cached/empty state - one quiet log line per view, no error spam."""
        self._hist_inflight.discard(view)
        self._hist_synced.add(view)
        if view not in self._hist_fail_logged:
            self._hist_fail_logged.add(view)
            log(f"Run history unavailable ({view}): {code}",
                Qgis.MessageLevel.Info)
        if view == self._displayed_view():
            self._rebuild_history_grid(view)

    # ---- thumbnails --------------------------------------------------------

    def _card_thumb_width(self) -> int:
        """Width to ask the server for, in real pixels of the preview band.

        The stored tile is a full 1024 px capture and the band it lands in is
        about 320 px wide, so the card would spend a megabyte and a half to
        paint a thumbnail. One width for every screen: a card scaled down from
        512 reads sharp everywhere, and picking a smaller step off the device
        pixel ratio only made the same run look softer on some displays. 512 is
        a width the image route accepts; anything else it snaps, so this cannot
        ask for a size that does not exist.
        """
        return 512

    def _artifact_url(self, request_id: str, which: str,
                      width: int | None = None) -> str:
        """Authorized artifact URL for one stored tile (streamed, no redirect,
        so the auth header never leaves our server). The id comes from the
        server, so it is percent-encoded: a raw "?", "#" or "&" in it would
        rewrite the query and fetch the wrong artifact.

        ``width`` asks for a downscaled copy. An older server ignores the
        parameter and answers with the full artifact, which still paints.
        """
        from urllib.parse import quote

        url = "{}/api/ai-segmentation/image/{}?type={}&stream=1".format(
            self._base, quote(str(request_id), safe=""), quote(str(which), safe=""))
        if width:
            url += f"&w={int(width)}"
        return url

    def _request_run_thumb(self, run: dict, card: _RunCard) -> None:
        """Fetch both halves of a run's comparison: the imagery as it was sent
        and the same tile with the masks painted on.

        The archived input is strictly more available than the overlay: a tile
        that found nothing has an input and no preview, so the input is what
        keeps the card readable.
        """
        rid = str(run.get("preview_request_id") or "")
        if not rid:
            card.mark_missing("input")
            card.mark_missing("preview")
            return
        width = self._card_thumb_width()
        urls: dict[str, tuple[str, dict | None]] = {
            "input": (self._artifact_url(rid, "input", width), self._auth or None),
        }
        signed = run.get("preview_url") or ""
        if self._auth:
            # Our own route can be asked for a card-sized copy; the signed URL
            # the payload carries can only ever hand back the full tile.
            urls["preview"] = (
                self._artifact_url(rid, "preview", width), self._auth)
        elif signed and signed.startswith(("http://", "https://")):
            # Signed-URL fallback: the signature is the auth, no headers.
            urls["preview"] = (signed, None)
        else:
            urls["preview"] = (self._artifact_url(rid, "preview", width), None)
        self._thumb_cards[rid] = card
        card.request_artifacts(self._hist_loader, urls, variant=str(width))

    def _on_thumb_loaded(self, pid: str, which: str, pixmap) -> None:
        if which not in ("input", "preview"):
            return
        card = self._thumb_cards.get(pid)
        if card is not None:
            try:
                card.set_image(which, pixmap)
            except RuntimeError:
                pass  # card torn down while the fetch was in flight

    def _on_thumb_failed(self, pid: str, which: str) -> None:
        if which not in ("input", "preview"):
            return
        card = self._thumb_cards.get(pid)
        if card is not None:
            try:
                card.mark_missing(which)
            except RuntimeError:
                pass

    # ---- run actions -------------------------------------------------------

    def _open_run_detail(self, run: dict) -> None:
        if self._detail_open or self._hist_busy:
            return
        self._detail_open = True
        try:
            dlg = _RunDetailDialog(run, self)
            self._detail_dlg = dlg
            dlg.exec()
        finally:
            self._detail_dlg = None
            self._detail_open = False

    def _toggle_favorite(self, run: dict, is_favorite: bool) -> None:
        """Optimistic star: flip locally at once, sync in the background,
        revert on error (AI Edit's _GenerationFavoriteWorker pattern)."""
        run_id = run.get("run_id")
        if not run_id or not self._auth:
            return
        run["is_favorite"] = is_favorite
        self._apply_favorite_ui(run, is_favorite)
        worker = _RunFavoriteWorker(
            self._history_client(), self._auth, str(run_id), is_favorite)
        worker.done.connect(self._on_favorite_done)
        park_orphaned_worker(worker)
        worker.start()

    def _toggle_template_favorite(self, preset: dict, _checked: bool) -> None:
        """Star a template. Local only: templates are a client-side catalogue,
        so there is no server row to flip and nothing to sync."""
        pid = str(preset.get("id") or "")
        if not pid:
            return
        toggle_favorite_template(pid)
        self._refresh_sidebar_counts()
        if self._active_key == _FAVORITES_KEY and not self._query:
            # Same Qt6 rule as the run star: never destroy the emitting card
            # from inside its own signal.
            QtC.safe_single_shot(0, self, self._rebuild_current_grid)

    def _apply_favorite_ui(self, run: dict, is_favorite: bool) -> None:
        key = _run_key(run)
        favs = self._hist_runs.get("favorites")
        if favs is not None and "favorites" in self._hist_synced:
            if is_favorite and all(_run_key(r) != key for r in favs):
                favs.insert(0, run)
            elif not is_favorite:
                self._hist_runs["favorites"] = [
                    r for r in favs if _run_key(r) != key]
        for view_runs in self._hist_runs.values():
            for r in view_runs:
                if _run_key(r) == key:
                    r["is_favorite"] = is_favorite
        if self._displayed_view() == "favorites":
            # Deferred: this runs from the star's own click handler, and the
            # rebuild destroys the card that emitted it. Tearing a widget down
            # inside its own signal aborts QGIS on Qt6.
            QtC.safe_single_shot(
                0, self, lambda: self._rebuild_history_grid("favorites"))
        else:
            card = self._hist_cards.get(key)
            if card is not None:
                try:
                    card.set_favorite(is_favorite)
                except RuntimeError:
                    pass
        if self._detail_dlg is not None:
            try:
                self._detail_dlg.set_favorite(is_favorite)
            except RuntimeError:
                pass
        self._refresh_sidebar_counts()

    def _on_favorite_done(self, run_id: str, is_favorite: bool, ok: bool) -> None:
        if ok:
            try:
                from ....core import telemetry_session_events
                telemetry_session_events.track_history_favorite_toggled(run_id, is_favorite)
            except Exception:
                pass  # nosec B110
            run_history_cache.save_runs(self._hist_runs.get("all") or [])
            return
        # Revert the optimistic flip.
        for view_runs in self._hist_runs.values():
            for r in view_runs:
                if str(r.get("run_id") or "") == run_id:
                    self._apply_favorite_ui(r, not is_favorite)
                    return

    # ---- restore / export ---------------------------------------------------

    def _start_run_fetch(self, run: dict, action: tuple) -> None:
        """Everything a restore or an export needs, on one background thread.

        The tiles, the stored masks, the decode into geometry and (for an
        export) the file write all happen there: done in the click handler they
        froze QGIS for a minute or more on a big run, with nothing on screen.
        """
        if self._hist_busy or not self._auth:
            return
        self._hist_busy = True
        self._pending_action = action
        if self._detail_dlg is not None:
            try:
                self._detail_dlg.set_busy(True)
            except RuntimeError:
                pass
        from ...plugin.run_restore import run_merge_separate

        # (driver, confidence, path) for an export, None for a restore.
        export = tuple(action[1:4]) if action and action[0] == "export" else None
        # The merge policy is read HERE, on the GUI thread, and handed over as
        # a plain bool: the worker must touch neither the plugin nor its caches.
        worker = _RunFetchWorker(
            self._history_client(), self._auth, run,
            run_merge_separate(self._plugin, run), export)
        worker.fetched.connect(self._on_run_fetched)
        worker.failed.connect(self._on_run_fetch_failed)
        worker.cancelled.connect(self._on_run_fetch_cancelled)
        worker.progress.connect(self._on_run_fetch_progress)
        self._fetch_worker = worker
        self._show_fetch_progress()
        park_orphaned_worker(worker)
        worker.start()

    def _end_run_fetch(self) -> None:
        self._hist_busy = False
        self._pending_action = None
        if self._detail_dlg is not None:
            try:
                self._detail_dlg.set_busy(False)
            except RuntimeError:
                pass

    # ---- the wait window ----------------------------------------------------

    def _show_fetch_progress(self) -> None:
        """Arm the wait window, shown after a beat so a short run does not
        flash a dialog on screen and take it away again."""
        dlg = _RunProgressDialog(self._detail_dlg or self)
        dlg.cancelled.connect(self._on_fetch_cancel_requested)
        self._fetch_progress = dlg
        QtC.safe_single_shot(350, self, self._reveal_fetch_progress)

    def _reveal_fetch_progress(self) -> None:
        dlg = self._fetch_progress
        if dlg is None or not self._hist_busy:
            return
        try:
            dlg.show()
        except RuntimeError:
            pass

    def _close_fetch_progress(self) -> None:
        """Take the wait window down because the work ended.

        Never call this from its own cancelled signal: Qt routes a programmatic
        close through reject(), and tearing a widget down inside its own signal
        aborts QGIS on Qt6.
        """
        dlg = self._fetch_progress
        self._fetch_progress = None
        if dlg is None:
            return
        try:
            dlg.finish()
            dlg.deleteLater()
        except RuntimeError:
            pass

    def _on_run_fetch_progress(self, phase: str, done: int, total: int) -> None:
        dlg = self._fetch_progress
        if dlg is None:
            return
        if phase == "decode":
            text = tr("Rebuilding shapes ({done} of {total})").format(
                done=done, total=total)
        elif phase == "write":
            text = tr("Writing the file...")
            done, total = 0, 0
        else:
            text = tr("Loading stored detections ({done} of {total})").format(
                done=done, total=total)
        try:
            dlg.set_step(text, done, total)
        except RuntimeError:
            pass

    def _on_fetch_cancel_requested(self) -> None:
        """Stop waiting for this run, now.

        The thread is usually inside a blocking network call, so it takes up to
        one call to notice. Its signals are cut here rather than left to land
        in a dialog that has moved on, and the user gets the buttons back at
        once; park_orphaned_worker owns what is left of the thread's life.
        """
        worker = self._fetch_worker
        self._fetch_worker = None
        # The window is closing itself (this runs from its own signal), so only
        # drop the handle to it.
        dlg = self._fetch_progress
        self._fetch_progress = None
        if dlg is not None:
            try:
                dlg.deleteLater()
            except RuntimeError:
                pass
        if worker is not None:
            try:
                worker.requestInterruption()
            except (RuntimeError, TypeError):
                pass
            # One guard per signal: batched with the interrupt above, a thread
            # that had already finished raised on the first call and left all
            # three handlers connected.
            for signal_name in ("fetched", "failed", "progress"):
                safe_disconnect(worker, signal_name)
        self._end_run_fetch()

    def _on_run_fetch_cancelled(self) -> None:
        """The thread noticed the stop and wound down. The dialog freed itself
        the moment Cancel was clicked, so there is nothing to undo here."""
        log("Run history fetch stopped by the user", Qgis.MessageLevel.Info)

    def _request_restore(self, run: dict, _detail_dlg=None) -> None:
        if self._plugin is None or self._view_only:
            return
        self._start_run_fetch(run, ("restore",))

    def _request_rerun(self, run: dict, _detail_dlg=None) -> None:
        """Point the Automatic flow back at this run: same ground, same object,
        same number of tiles, stopped one click short of spending anything.

        Only the tile rows are fetched. The stored detections are what Restore
        is for, and pointing at a zone does not need them.
        """
        if self._plugin is None or self._view_only or self._hist_busy:
            return
        if not self._auth:
            return
        self._hist_busy = True
        if self._detail_dlg is not None:
            try:
                self._detail_dlg.set_busy(True)
            except RuntimeError:
                pass
        worker = _RunZoneFetchWorker(self._history_client(), self._auth, run)
        worker.fetched.connect(self._on_rerun_zone_fetched)
        worker.failed.connect(self._on_run_fetch_failed)
        park_orphaned_worker(worker)
        worker.start()

    def _on_rerun_zone_fetched(self, run: dict, tiles: list) -> None:
        from ...plugin.run_restore import zone_extent_from_tiles
        self._end_run_fetch()
        zone = zone_extent_from_tiles(tiles)
        if zone is None:
            QMessageBox.warning(
                self, tr("Segment library"),
                tr("This run did not keep where it looked, so it cannot be "
                   "pointed at the same place. Draw the zone again."))
            return
        extent, authid = zone
        dock = self._dock_widget()
        if dock is None:
            return
        if self._detail_dlg is not None:
            try:
                self._detail_dlg.reject()
            except RuntimeError:
                pass
        self.reject()  # close first; the plugin work is deferred a tick
        dock.history_rerun_requested.emit({
            "prompt": run.get("prompt") or "",
            "extent": list(extent),
            "crs": authid,
            "tiles": int(run.get("tiles") or len(tiles)),
        })

    def _request_export(self, run: dict, _detail_dlg=None) -> None:
        if self._plugin is None or self._view_only:
            return
        from ...plugin.run_restore import snap_confidence
        default_conf = snap_confidence(run.get("threshold"), 0.30)
        if default_conf <= 0.15:
            default_conf = 0.30
        dlg = _ExportRunDialog(run, default_conf, self._detail_dlg or self)
        if not dlg.exec() or not dlg.path():
            return
        self._start_run_fetch(
            run, ("export", dlg.driver(), dlg.confidence(), dlg.path()))

    def _on_run_fetch_failed(self, code: str) -> None:
        self._fetch_worker = None
        self._close_fetch_progress()
        self._end_run_fetch()
        log(f"Run history fetch failed: {code}", Qgis.MessageLevel.Warning)
        QMessageBox.warning(
            self, tr("Segment library"),
            tr("Could not load this run's stored detections. Try again later."))

    def _on_run_fetched(self, run: dict, tiles: list, outcome: dict) -> None:
        action = self._pending_action or ("restore",)
        self._fetch_worker = None
        self._close_fetch_progress()
        self._end_run_fetch()
        if "export" in outcome:
            self._finish_export(run, outcome, action[1], action[3])
            return
        self._finish_restore(run, tiles, outcome)

    def _missing_tiles_note(self, outcome: dict) -> str:
        """One sentence when the fetch ran out of its wall-clock budget, so a
        short result is never passed off as the whole run. Empty otherwise."""
        skipped = int(outcome.get("tiles_skipped") or 0)
        if skipped <= 0:
            return ""
        return tr("{n} tile(s) took too long to load and are missing from "
                  "this result.").format(n=skipped)

    def _finish_restore(self, run: dict, tiles: list, decoded: dict) -> None:
        from qgis.PyQt.QtCore import Qt

        from ...plugin import run_restore
        # Building the review is still GUI work (the same tail a live run runs
        # through _complete_auto_finalize); the decode that used to dominate it
        # is already done on the thread.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ok = run_restore.restore_run(self._plugin, run, tiles, decoded)
        finally:
            QApplication.restoreOverrideCursor()
        if not ok:
            QMessageBox.warning(
                self, tr("Segment library"),
                tr("Could not load this run's stored detections. Try again later."))
            return
        note = self._missing_tiles_note(decoded)
        if note:
            try:
                self._plugin.iface.messageBar().pushWarning(
                    "AI Segmentation", note)
            except (RuntimeError, AttributeError):
                pass
        if self._detail_dlg is not None:
            try:
                self._detail_dlg.accept()
            except RuntimeError:
                pass
        self.reject()  # no prompt chosen; the review is now open on the map

    def _finish_export(self, run: dict, outcome: dict, driver: str,
                       path: str) -> None:
        """The file is already written (the fetch thread did it); what is left
        is putting it on the map and saying how it went."""
        from qgis.core import QgsProject

        from ...plugin.run_restore import load_exported_layer

        summary = outcome.get("export") or {}
        count = int(summary.get("count") or 0)
        layer = load_exported_layer(path, driver) if summary.get("written") else None
        if not count or layer is None:
            QMessageBox.warning(
                self, tr("Export..."),
                tr("Nothing to export at this confidence. Lower it and try again.")
                if not count else
                tr("The export failed. Check the file path and try again."))
            return
        QgsProject.instance().addMapLayer(layer)
        try:
            from ....core import telemetry_session_events
            telemetry_session_events.track_history_exported(driver, count, run_id=_run_key(run))
        except Exception:
            pass  # nosec B110
        note = self._missing_tiles_note(outcome)
        QMessageBox.information(
            self, tr("Export..."),
            tr("Exported {n} polygon(s).").format(n=count) + (f"\n\n{note}" if note else ""))

    # ---- selection -------------------------------------------------------

    def _open_detail(self, preset: dict) -> None:
        # Re-entrancy guard: a single physical click can deliver two activations
        # (slider click + propagated card release). The first opens the modal;
        # any second one while it is open - or after a selection - is ignored.
        if self._detail_open or self._selected_prompt is not None:
            return
        self._detail_open = True
        try:
            dlg = _PresetDetailDialog(
                preset, self._base, self,
                category_label=self._cat_label_by_id.get(preset.get("id", ""), ""))
            dlg.exec()
            if dlg.chosen and not self._view_only:
                self._select(preset)
        finally:
            self._detail_open = False

    def _on_recent_activated(self, entry: dict) -> None:
        """A recent card is one-click "take me back": restore the map first
        (zoom to the stored zone, re-activate the exported layer so it can be
        inspected/exported at once), then reuse the object like any Use flow
        (accept + drop the token in the prompt box). Every restore step is
        best-effort; the prompt reuse always happens."""
        if self._view_only:
            return
        restore_recent_on_map(self._plugin, entry)
        self._select(entry)

    def _on_recent_rerun(self, entry: dict) -> None:
        """"Run again here": close the library and hand the stored run (zone
        extent + CRS + prompt) to the dock, which rebuilds the exact zone and
        lands the user on step 2 ready to Detect. Relayed through the dock's
        signal so the plugin owns the orchestration (see auto_zone.py)."""
        if self._view_only:
            return
        dock = self._dock_widget()
        if dock is None:
            return
        self.reject()  # close first; the plugin work is deferred a tick
        dock.history_rerun_requested.emit(dict(entry))

    def _on_recent_reuse_prompt(self, entry: dict) -> None:
        """"Same object, new zone": close the library and hand only the prompt
        token to the dock; the plugin starts the flow on the draw-zone step."""
        if self._view_only:
            return
        dock = self._dock_widget()
        if dock is None:
            return
        prompt = (entry.get("prompt") or "").strip()
        self.reject()
        dock.history_reuse_prompt_requested.emit(prompt)

    def _dock_widget(self):
        """The dock that owns the re-run relay signals, or None. The dialog is
        parented to the dock, but go through the plugin when present so the
        wiring matches the rest of the history actions."""
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
