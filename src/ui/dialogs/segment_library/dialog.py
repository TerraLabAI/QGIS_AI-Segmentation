"""Segment library: the window that holds both halves of "what do I detect".

Left of the grid, a rail: the curated object catalogue (Popular, then one row
per category) and, at the foot, the user's own work (Recent, Favorites).
Picking a template card returns its **English token** (the literal cloud-model
prompt), which the dock drops into the prompt box. Labels are localized;
tokens are not.

This file owns the window and the catalogue side: the size, the search box,
the card grid and how it reflows, and the template detail popup. The two other
halves are mixins:

- ``rail.py`` (LibraryRailMixin): the navigation rail and its counts.
- ``history.py`` (LibraryHistoryMixin): the user's runs, their states, their
  thumbnails, their stars and their removal.
- ``run_actions.py`` (LibraryRunActionsMixin): restore, re-run and export for
  one stored run, each on a background thread behind one wait window.

Performance: the catalogue is read from a non-blocking cache (the network
prefetch is the plugin's job), and demo images load lazily per visible card so
the first paint never waits on the whole grid. Nothing in ``__init__`` touches
the network, so the window always opens instantly.

The dialog tolerates plugin=None and a signed-out account: the template side
stays fully usable, and the history side says which of those two it is instead
of showing an empty list.
"""
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
from ....core.presets.segmentation_presets import pick_label, preset_matches_query
from ....core.presets.segmentation_presets_client import (
    base_url,
    cached_or_offline_catalog,
)
from ....core.presets.template_favorites import (
    is_favorite_template,
    toggle_favorite_template,
)
from ....core.qt_compat import safe_disconnect
from ...template_demo_loader import TemplateDemoLoader
from .cards import _PresetCard
from .common import (
    _EMPTY_GLYPH,
    _EMPTY_MSG,
    _GHOST_BTN_QSS,
    _META_QSS,
    _RAIL_FAVORITES_TARGET,
    _RAIL_HISTORY_VIEWS,
    _RAIL_POPULAR_TARGET,
    _SEARCH_QSS,
    _fmt_count,
)
from .detail import _PresetDetailDialog
from .history import LibraryHistoryMixin
from .rail import LibraryRailMixin
from .recent_local import restore_recent_on_map
from .run_actions import LibraryRunActionsMixin
from .run_card import _RunCard

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


class SegmentLibraryDialog(LibraryRailMixin, LibraryHistoryMixin,
                           LibraryRunActionsMixin, QDialog):
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
        self._active_key = _RAIL_POPULAR_TARGET
        self._query = ""
        self._cards_by_id: dict[str, _PresetCard] = {}
        # Grid ownership: the widgets currently on show, and how many columns
        # they are spread over. Kept apart from the layout so a width change
        # re-places the same cards instead of rebuilding (and refetching) them.
        self._cols = _GRID_COLS_DEFAULT
        self._grid_widgets: list = []
        # Set when the grid is laid out as (day header, cards) blocks, so
        # a width change re-places the same widgets under the same headers.
        self._grid_sections: list | None = None
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
        # Stars flipped in this window, run key -> wanted state. A page that
        # lands after a flip carries the server's older answer, and replacing
        # the list with it un-starred a run the user had just starred.
        self._fav_overrides: dict[str, bool] = {}
        # Views whose last sync did not come back. Kept apart from the log-once
        # set so the empty state can tell a failed read from a new account.
        self._hist_failed: set = set()
        self._hist_pages_loaded = 0
        # Two registries: run key -> card follows favorite toggles, archived
        # tile id -> card routes the loaded images back to the right preview.
        self._hist_cards: dict[str, _RunCard] = {}
        self._thumb_cards: dict[str, _RunCard] = {}
        self._hist_busy = False
        # Declared here, not only where the fetch starts. The failure handler
        # closes the progress window before it clears _hist_busy, so on a fetch
        # that fails before one was ever opened (a run with no stored tiles, or
        # offline) the missing attribute raised, the busy flag stayed set, and
        # every later Restore, Export and Rerun in that library session did
        # nothing at all.
        self._fetch_worker = None
        self._fetch_progress = None
        # Every background worker started from this window, with the signals it
        # emits back into it. They outlive the dialog (park_orphaned_worker owns
        # the thread), so closing mid-fetch has to cut them loose; the list is
        # as long as the actions one visit takes, a handful.
        self._live_workers: list[tuple] = []
        self._pending_action: tuple | None = None
        # The run detail popup while it is open (history.py owns it).
        self._detail_dlg = None
        self._tabs_tracked: set = set()
        self._hist_loader = TemplateDemoLoader(self)
        self._hist_loader.loaded.connect(self._on_thumb_loaded)
        self._hist_loader.failed.connect(self._on_thumb_failed)

        self._build_ui()
        from ...dock.font_scale import apply_font_scale_to_tree

        apply_font_scale_to_tree(self)
        self._select_tab(_RAIL_POPULAR_TARGET)
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
        # The rail carries its own right border, so there is no separate
        # separator line beside it.
        body.addWidget(self._build_library_rail())

        # Card grid in a scroll area (shared by every tab + search), with the
        # history paging button under it. The button used to sit at the foot of
        # the sidebar, a column it has nothing to do with; it belongs at the end
        # of the list it extends.
        content = QVBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(8)
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
        content.addWidget(self._scroll, 1)

        # Added straight to the column, not wrapped in a row: a QVBoxLayout
        # skips a hidden widget's spacing, but never a nested layout's, so a
        # wrapper would leave a permanent gap under the grid.
        self._hist_older_btn = QPushButton(tr("Load older runs"))
        self._hist_older_btn.setStyleSheet(_GHOST_BTN_QSS)
        self._hist_older_btn.setCursor(QtC.PointingHandCursor)
        # A QPushButton inside a QDialog volunteers as the default button, and
        # the search box has the focus: without this, Return over a search
        # result fires whichever button was built first instead of doing
        # nothing. No button in this window is the one Return should press.
        self._hist_older_btn.setAutoDefault(False)
        self._hist_older_btn.setVisible(False)
        self._hist_older_btn.clicked.connect(self._load_older_runs)
        content.addWidget(self._hist_older_btn, 0, QtC.AlignCenter)
        body.addLayout(content, 1)

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

    def _refresh_library_chrome(self) -> None:
        """Re-read everything the rail and the paging button show. Called after
        any change to the history lists or the favorites, and after a sync ends
        either way: the paging button reads the in-flight state, so a failed
        page that skipped this left it saying it was still loading."""
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
            telemetry_session_events.track_library_opened(tab)
        except Exception:
            pass  # nosec B110

    # ---- navigation ------------------------------------------------------

    def _select_tab(self, key: str) -> None:
        self._active_key = key
        self._set_rail_active(key)
        # The sync starts BEFORE the first paint: the grid reads the in-flight
        # set to tell "still loading" from "nothing here yet", and the worker's
        # answer is queued, so it cannot land before the grid is built.
        view = _RAIL_HISTORY_VIEWS.get(key)
        if view is not None:
            self._track_tab_opened("history")
            self._sync_history_view(view)
        self._rebuild_current_grid()
        self._refresh_library_chrome()

    def _rebuild_current_grid(self) -> None:
        """Paint whichever view the rail is on, searched or not.

        A history view keeps its own search: the box filters the runs by their
        prompt, and the view still owns its loading, failed and signed-out
        states, so a search never turns one of them into a bare empty grid.
        """
        view = _RAIL_HISTORY_VIEWS.get(self._active_key)
        if view is not None:
            self._rebuild_history_grid(view)
            return
        if self._query:
            self._rebuild_grid(
                self._search_matches(self._query),
                tr("No object matches that search."), "⌕")
            return
        presets = self._presets_for_tab(self._active_key)
        if not presets and self._active_key != _RAIL_POPULAR_TARGET:
            # An unknown key (a catalogue that lost a category between two
            # opens) used to paint "Nothing in this category yet." under an
            # unlit rail, which reads as the window losing its place. Go back
            # to the row that always has something on it.
            self._select_tab(_RAIL_POPULAR_TARGET)
            return
        self._rebuild_grid(presets, tr("Nothing in this category yet."))

    def _apply_search(self) -> None:
        self._query = self._search.text().strip().lower()
        # A search over the catalogue is its own view, so no rail row is "you
        # are here" while one is showing. A search inside a history view is a
        # filter ON that view, so the row it belongs to stays lit.
        in_history = _RAIL_HISTORY_VIEWS.get(self._active_key) is not None
        self._set_rail_active(
            self._active_key if in_history or not self._query else None)
        self._rebuild_current_grid()

    def _search_matches(self, query: str) -> list[dict]:
        """Match over the token, the localized label, the category and any
        search terms the catalogue carries, accent-folded on both sides so
        "eolienne" reaches the same card as "éolienne"."""
        return [p for p in self._by_id.values()
                if preset_matches_query(
                    p, query, self._cat_label_by_id.get(p.get("id", ""), ""))]

    def _presets_for_tab(self, key: str) -> list[dict]:
        if key == _RAIL_POPULAR_TARGET:
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
        self._grid_sections = None
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
        # The card floor grows with the UI font, so the column count has to
        # read the scaled width or the grid packs cards tighter than they are.
        from ...dock.font_scale import scale_px_length
        step = scale_px_length(_CARD_MIN_W) + _GRID_SPACING
        return max(_GRID_COLS_MIN, min(_GRID_COLS_MAX, (width + _GRID_SPACING) // step))

    def _place_grid(self, widgets: list, span_all: bool = False) -> None:
        """Own the grid's contents, then lay them out at the current width."""
        self._grid_widgets = list(widgets)
        self._grid_sections = None
        self._grid_span_all = bool(span_all)
        self._settle_grid_widgets()
        self._apply_grid_positions()
        self._update_count_label(0 if span_all else len(self._grid_widgets))

    def _place_grouped_grid(self, sections: list) -> None:
        """Lay the grid out as (header, cards) blocks instead of one flat run.

        The run history reads as a diary, not as a wall: a full-width header
        opens each day and its cards flow under it. The sections are kept, not
        just their positions, so a width change re-places the same widgets
        rather than rebuilding cards and refetching their imagery.
        """
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
        """Grow the new widgets to the user's text size and let each one
        re-measure whatever it pins by hand.

        Cards are built when their tab is picked, long after the window was, so
        the pass the window made over itself never saw them.
        """
        from ...dock.font_scale import apply_font_scale_to_tree

        for widget in self._grid_widgets:
            apply_font_scale_to_tree(widget)
            settle = getattr(widget, "on_font_scale_applied", None)
            if settle is not None:
                settle()

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

    def _empty_label(self, text: str, glyph: str = "◇",
                     action: tuple | None = None) -> None:
        """Hero empty state: one glyph, one sentence, centered.

        ``action`` is an optional (label, callback) pair, for the states where
        the sentence names something the user can do from here.
        """
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

    def _rebuild_grid(self, presets: list[dict], empty_text: str,
                      empty_glyph: str = "◇") -> None:
        """Paint the card grid, or the empty state the CALLER names.

        The sentence has to come from outside: a tab with nothing in it is not
        a failed search, and telling the user their search matched nothing when
        they typed nothing reads as blame for something they never did.
        """
        self._clear_grid()
        if not presets:
            self._empty_label(empty_text, empty_glyph)
            return
        self._place_grid(self._build_preset_cards(presets))
        # Kick lazy loading for whatever is visible now + once layout settles.
        QTimer.singleShot(0, self._load_visible_cards)
        QTimer.singleShot(80, self._load_visible_cards)

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

    def _toggle_template_favorite(self, preset: dict, _checked: bool) -> None:
        """Star a template. Local only: templates are a client-side catalogue,
        so there is no server row to flip and nothing to sync."""
        pid = str(preset.get("id") or "")
        if not pid:
            return
        toggle_favorite_template(pid)
        self._refresh_library_chrome()
        if self._active_key == _RAIL_FAVORITES_TARGET and not self._query:
            # Same Qt6 rule as the run star: never destroy the emitting card
            # from inside its own signal.
            QtC.safe_single_shot(0, self, self._rebuild_current_grid)

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

    # ---- teardown ----------------------------------------------------------

    def _track_live_worker(self, worker, *signal_names: str) -> None:
        """Remember a worker and the signals it fires back into this window."""
        self._live_workers.append((worker, signal_names))

    def done(self, result):  # noqa: N802 - Qt signature
        """Cut every background worker loose before the window goes.

        The threads outlive the dialog, so a history page, a star or a run
        fetch still in flight kept firing into a dismissed window. Their
        signals go first, then each one is asked to stop; the thread itself
        winds down under park_orphaned_worker, and is never terminated.
        """
        workers = self._live_workers
        self._live_workers = []
        self._fetch_worker = None
        for worker, signal_names in workers:
            for signal_name in signal_names:
                safe_disconnect(worker, signal_name)
            try:
                worker.requestInterruption()
            except (RuntimeError, TypeError, AttributeError):
                pass  # nosec B110 - already finished, or never started
        self._close_fetch_progress()
        super().done(result)
