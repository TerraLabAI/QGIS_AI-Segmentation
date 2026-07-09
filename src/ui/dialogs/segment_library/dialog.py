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
from qgis.PyQt.QtCore import QPoint, QTimer
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
from ....core.presets.segmentation_presets import pick_label
from ....core.presets.segmentation_presets_client import (
    base_url,
    cached_or_offline_catalog,
)
from ...template_demo_loader import TemplateDemoLoader
from .cards import _PresetCard, _RecentCard, _RunCard
from .common import (
    _EMPTY_MSG,
    _GHOST_BTN_QSS,
    _run_key,
    _SEARCH_QSS,
    _SECTION_HEADER,
    _SIDEBAR_ITEM,
    _SIDEBAR_ITEM_ACTIVE,
    _sidebar_icon_html,
    _SidebarButton,
    _tab_label_html,
)
from .detail import _ExportRunDialog, _PresetDetailDialog, _RunDetailDialog
from .recent_local import merge_local_recents, recent_view, restore_recent_on_map
from .workers import (
    _detach_worker,
    _HistoryFetchWorker,
    _RunFavoriteWorker,
    _RunFetchWorker,
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
_GRID_COLS = 3


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
        self.setMinimumSize(640, 480)
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
        self._hist_cards: dict[str, _RunCard] = {}
        self._hist_busy = False
        self._pending_action: tuple | None = None
        self._detail_dlg: _RunDetailDialog | None = None
        self._tabs_tracked: set = set()
        self._hist_loader = TemplateDemoLoader(self)
        self._hist_loader.loaded.connect(self._on_thumb_loaded)
        self._hist_loader.failed.connect(self._on_thumb_failed)

        self._build_ui()
        self._select_tab(_TOP_KEY)
        self._track_tab_opened("detect")

    # ---- UI scaffold -----------------------------------------------------

    def _apply_open_size(self) -> None:
        """Open large: hug the 220px sidebar + a 3-column card grid, grown
        toward the screen so the previews read big. Clamped to the available
        screen so it never spills offscreen (AI Edit's open-size rule)."""
        target_w, target_h = 1220, 880
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            target_w = min(target_w, int(avail.width() * 0.96))
            target_h = min(target_h, int(avail.height() * 0.92))
        self.resize(max(target_w, 640), max(target_h, 480))

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
        root.addWidget(self._search)

        body = QHBoxLayout()
        body.setSpacing(8)

        # Sidebar: the user's own detections first, curated templates below
        # (AI Edit's "Your prompts / Templates" grouping).
        sidebar_host = QWidget()
        sidebar_host.setFixedWidth(220)
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
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(12)
        for c in range(_GRID_COLS):
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

    def _add_section(self, label: str) -> None:
        lbl = QLabel(label.upper())
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
            return len(self._hist_runs.get("favorites") or [])
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
            from ....core import telemetry
            from ....core import telemetry_events as tev
            telemetry.track(tev.LIBRARY_OPENED, {"tab": tab})
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
        matches = []
        for p in self._by_id.values():
            hay = f"{p.get('prompt', '')} {pick_label(p.get('label'), '')}".lower()
            if query in hay:
                matches.append(p)
        return matches

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

    def _empty_label(self, text: str) -> None:
        empty = QLabel(text)
        empty.setWordWrap(True)
        empty.setStyleSheet(_EMPTY_MSG)
        self._grid.addWidget(empty, 0, 0, 1, _GRID_COLS)

    def _rebuild_grid(self, presets: list[dict]) -> None:
        self._clear_grid()
        if not presets:
            self._empty_label(tr("No matching objects."))
            return
        for idx, preset in enumerate(presets):
            card = _PresetCard(preset, self._grid_host)
            card.activated.connect(self._open_detail)
            self._grid.addWidget(card, idx // _GRID_COLS, idx % _GRID_COLS)
            self._cards_by_id[preset["id"]] = card
        # Kick lazy loading for whatever is visible now + once layout settles.
        QTimer.singleShot(0, self._load_visible_cards)
        QTimer.singleShot(80, self._load_visible_cards)

    def _rebuild_history_grid(self, view: str) -> None:
        self._clear_grid()
        runs = self._hist_runs.get(view) or []
        if not runs:
            if view == "all" and self._local_recent_entries():
                # Signed-out / endpoint-less fallback: the local run history
                # still gives one-click restore of past detections.
                self._rebuild_recent_local_grid()
                return
            self._empty_label({
                "all": tr("Nothing here yet. Your automatic detections will "
                          "land here, ready to reuse, restore or export."),
                "favorites": tr("Star a detection to keep it here."),
            }[view])
            return
        for idx, run in enumerate(runs):
            card = _RunCard(run, view, parent=self._grid_host)
            card.opened.connect(self._open_run_detail)
            card.star_toggled.connect(self._toggle_favorite)
            self._grid.addWidget(card, idx // _GRID_COLS, idx % _GRID_COLS)
            self._request_run_thumb(run, card)

    def _local_recent_entries(self) -> list[dict]:
        """The Recent tab's local feed (see recent_local.merge_local_recents)."""
        return merge_local_recents(self._history_local, self._recent_local)

    def _rebuild_recent_local_grid(self) -> None:
        for idx, entry in enumerate(self._local_recent_entries()):
            card = _RecentCard(recent_view(entry, self._by_token), self._grid_host,
                               view_only=self._view_only)
            card.activated.connect(self._on_recent_activated)
            card.rerun_requested.connect(self._on_recent_rerun)
            card.reuse_prompt_requested.connect(self._on_recent_reuse_prompt)
            self._grid.addWidget(card, idx // _GRID_COLS, idx % _GRID_COLS)

    # ---- demo image routing ---------------------------------------------

    def _load_visible_cards(self) -> None:
        """Request demos only for preset cards in/near the viewport (one screen
        of lookahead), so opening a big category never fires every fetch."""
        if not self._cards_by_id:
            return
        viewport = self._scroll.viewport()
        vp_h = viewport.height()
        margin = max(vp_h, 1)  # one screen of lookahead either way
        for card in list(self._cards_by_id.values()):
            try:
                top = card.mapTo(viewport, QPoint(0, 0)).y()
            except RuntimeError:
                continue  # card already torn down
            if top + card.height() >= -margin and top <= vp_h + margin:
                card.request_demos(self._loader, self._base)

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
        _detach_worker(worker)
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
                    from ....core import telemetry
                    from ....core import telemetry_events as tev
                    telemetry.track(tev.HISTORY_SYNCED, {"runs": len(runs)})
                except Exception:
                    pass  # nosec B110
        else:
            known = {_run_key(r) for r in self._hist_runs[view]}
            self._hist_runs[view].extend(
                r for r in runs if _run_key(r) not in known)
            self._hist_pages_loaded += 1
            try:
                from ....core import telemetry
                from ....core import telemetry_events as tev
                telemetry.track(tev.HISTORY_PAGE_LOADED,
                                {"page": self._hist_pages_loaded})
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

    def _artifact_url(self, request_id: str, which: str) -> str:
        """Authorized artifact URL for one stored tile (streamed, no redirect,
        so the auth header never leaves our server)."""
        return "{}/api/ai-segmentation/image/{}?type={}&stream=1".format(
            self._base, request_id, which)

    def _request_run_thumb(self, run: dict, card: _RunCard) -> None:
        rid = run.get("preview_request_id") or ""
        url = run.get("preview_url") or ""
        headers = None
        if url and url.startswith(("http://", "https://")):
            # Directly-signed URL: the signature is the auth, no headers.
            key = rid or _run_key(run)
        elif rid:
            url = self._artifact_url(rid, "preview")
            headers = self._auth or None
            key = rid
        else:
            card.mark_thumb_missing()
            return
        if not key:
            card.mark_thumb_missing()
            return
        self._hist_cards[key] = card
        self._hist_loader.request(key, "thumb", url, headers=headers)

    def _on_thumb_loaded(self, pid: str, which: str, pixmap) -> None:
        if which != "thumb":
            return
        card = self._hist_cards.get(pid)
        if card is not None:
            try:
                card.set_thumb(pixmap)
            except RuntimeError:
                pass  # card torn down while the fetch was in flight

    def _on_thumb_failed(self, pid: str, which: str) -> None:
        if which != "thumb":
            return
        card = self._hist_cards.get(pid)
        if card is not None:
            try:
                card.mark_thumb_missing()
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
            if dlg.use_requested and not self._view_only:
                self._select(run)
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
        _detach_worker(worker)
        worker.start()

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
            self._rebuild_history_grid("favorites")
        else:
            rid = run.get("preview_request_id") or key
            card = self._hist_cards.get(rid)
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
                from ....core import telemetry
                from ....core import telemetry_events as tev
                telemetry.track(tev.HISTORY_FAVORITE_TOGGLED, {
                    "run_id": run_id, "is_favorite": is_favorite})
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
        if self._hist_busy or not self._auth:
            return
        self._hist_busy = True
        self._pending_action = action
        if self._detail_dlg is not None:
            try:
                self._detail_dlg.set_busy(True)
            except RuntimeError:
                pass
        worker = _RunFetchWorker(self._history_client(), self._auth, run)
        worker.fetched.connect(self._on_run_fetched)
        worker.failed.connect(self._on_run_fetch_failed)
        _detach_worker(worker)
        worker.start()

    def _end_run_fetch(self) -> None:
        self._hist_busy = False
        self._pending_action = None
        if self._detail_dlg is not None:
            try:
                self._detail_dlg.set_busy(False)
            except RuntimeError:
                pass

    def _request_restore(self, run: dict, _detail_dlg=None) -> None:
        if self._plugin is None or self._view_only:
            return
        self._start_run_fetch(run, ("restore",))

    def _request_export(self, run: dict, _detail_dlg=None) -> None:
        if self._plugin is None or self._view_only:
            return
        from ....core.run_restore import snap_confidence
        default_conf = snap_confidence(run.get("threshold"), 0.30)
        if default_conf <= 0.15:
            default_conf = 0.30
        dlg = _ExportRunDialog(run, default_conf, self._detail_dlg or self)
        if not dlg.exec() or not dlg.path():
            return
        self._start_run_fetch(
            run, ("export", dlg.driver(), dlg.confidence(), dlg.path()))

    def _on_run_fetch_failed(self, code: str) -> None:
        self._end_run_fetch()
        log(f"Run history fetch failed: {code}", Qgis.MessageLevel.Warning)
        QMessageBox.warning(
            self, tr("Segment library"),
            tr("Could not load this run's stored detections. Try again later."))

    def _on_run_fetched(self, run: dict, tiles: list, masks_per_tile: dict) -> None:
        action = self._pending_action or ("restore",)
        self._end_run_fetch()
        if action[0] == "export":
            self._finish_export(run, tiles, masks_per_tile,
                                action[1], action[2], action[3])
            return
        self._finish_restore(run, tiles, masks_per_tile)

    def _finish_restore(self, run: dict, tiles: list, masks_per_tile: dict) -> None:
        from qgis.PyQt.QtCore import Qt

        from ....core import run_restore
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ok = run_restore.restore_run(self._plugin, run, tiles, masks_per_tile)
        finally:
            QApplication.restoreOverrideCursor()
        if not ok:
            QMessageBox.warning(
                self, tr("Segment library"),
                tr("Could not load this run's stored detections. Try again later."))
            return
        if self._detail_dlg is not None:
            try:
                self._detail_dlg.accept()
            except RuntimeError:
                pass
        self.reject()  # no prompt chosen; the review is now open on the map

    def _finish_export(self, run: dict, tiles: list, masks_per_tile: dict,
                       driver: str, confidence: float, path: str) -> None:
        from qgis.core import QgsCoordinateReferenceSystem, QgsProject
        from qgis.PyQt.QtCore import Qt

        from ....core import run_restore
        from ....core.polygon_exporter import export_geometries_to_file

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            geoms, crs_authid = run_restore.build_run_geometries(
                self._plugin, run, tiles, masks_per_tile, confidence)
            layer = None
            if geoms:
                layer = export_geometries_to_file(
                    geoms, QgsCoordinateReferenceSystem(crs_authid), path,
                    driver=driver)
        finally:
            QApplication.restoreOverrideCursor()
        if not geoms or layer is None:
            QMessageBox.warning(
                self, tr("Export..."),
                tr("Nothing to export at this confidence. Lower it and try again.")
                if not geoms else
                tr("The export failed. Check the file path and try again."))
            return
        QgsProject.instance().addMapLayer(layer)
        try:
            from ....core import telemetry
            from ....core import telemetry_events as tev
            telemetry.track(tev.HISTORY_EXPORTED, {
                "run_id": _run_key(run),
                "format": driver,
                "objects": len(geoms),
            })
        except Exception:
            pass  # nosec B110
        QMessageBox.information(
            self, tr("Export..."),
            tr("Exported {n} polygon(s).").format(n=len(geoms)))

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
