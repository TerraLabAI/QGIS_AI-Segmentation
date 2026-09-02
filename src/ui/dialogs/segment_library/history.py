"""The library's "My work" side: the user's own runs, read from the account.

Two views share one pipeline, Recent (every run) and Favorites (the starred
ones). Both are paged from the server on a background thread, painted as one
card per run grouped by the day it happened, and kept honest about which state
they are in: loading, empty, unreadable, or signed out. Nothing here guesses.
A view with nothing in it says which of those four it is, and offers the one
thing the user can do about it.

The rules this file exists to keep:

- The list never shows one thing while the rail counts another. The rail reads
  the same numbers the grid paints, and prints none while a view is loading.
- The local prompt-only recents are a FALLBACK, never the face of Recent. They
  appear when the account cannot answer (signed out, or a failed read), under
  a header that says so, and never on top of a good server answer.
- Every action a card offers is one the window can actually run right now.
  While a restore, an export or a re-run is in flight, the action rows go
  away rather than taking a click and answering nothing.

Mixed into SegmentLibraryDialog beside LibraryRailMixin and
LibraryRunActionsMixin. Never define a method name that one of those defines.
"""
from __future__ import annotations

from qgis.core import Qgis
from qgis.PyQt.QtCore import QTimer
from qgis.PyQt.QtWidgets import QMessageBox

from ....core import qt_compat as QtC
from ....core.i18n import tr
from ....core.logging_utils import log
from ....core.presets import run_history_cache
from ....core.presets.template_favorites import favorite_template_ids
from ...plugin.shared import park_orphaned_worker
from .cards import _RecentCard
from .common import _RAIL_HISTORY_VIEWS, _run_key
from .detail import _RunDetailDialog
from .recent_local import merge_local_recents, recent_view
from .run_card import _DayHeader, _RunCard
from .run_marks import forget_run_mark, set_run_mark
from .run_summary import group_runs_by_day
from .workers import _HistoryFetchWorker, _RunDeleteWorker, _RunFavoriteWorker

# Rows asked for before anything is measured: three columns at the widest the
# window opens, so two rows are on screen on every desk. Past that the viewport
# pass decides, because it can see where the cards landed.
_EAGER_THUMB_CARDS = 6


class LibraryHistoryMixin:
    """Reads, paints and edits the user's own run history."""

    # ---- sync ------------------------------------------------------------

    def showEvent(self, event):  # noqa: N802 - Qt signature
        """Start reading the history the moment the window appears.

        The window opens on the object catalogue, and the history used to be
        read only when the user reached for it: the wait for the list, and then
        for every picture on it, started at that click. Nothing here paints.
        The answer is stored, its first pictures are pulled into the cache, and
        the tab finds both already there.
        """
        super().showEvent(event)
        if getattr(self, "_hist_prefetched", False):
            return
        self._hist_prefetched = True
        self._sync_history_view("all")

    def _history_client(self):
        if self._client is None:
            from ....api.terralab_client import TerraLabClient
            self._client = TerraLabClient()
        return self._client

    def _sync_history_view(self, view: str, before: str | None = None) -> bool:
        """Refresh one view from the server, off the GUI thread.

        Returns whether a fetch was actually started: a no-op here used to
        leave the paging button saying it was loading a page nobody asked for.
        """
        if not self._auth:
            return False
        if before is None and view in self._hist_synced:
            return False
        if view in self._hist_inflight:
            return False
        self._hist_inflight.add(view)
        worker = _HistoryFetchWorker(
            self._history_client(), self._auth, view, before)
        worker.page_fetched.connect(self._on_history_page)
        worker.failed.connect(self._on_history_failed)
        self._track_live_worker(worker, "page_fetched", "failed")
        park_orphaned_worker(worker)
        worker.start()
        return True

    def _load_older_runs(self) -> None:
        view = _RAIL_HISTORY_VIEWS.get(self._active_key)
        if view is None:
            return
        runs = self._hist_runs.get(view) or []
        if not runs:
            return
        oldest = runs[-1].get("started_at") or runs[-1].get("created_at")
        if not oldest:
            return
        if not self._sync_history_view(view, before=str(oldest)):
            return
        # The page takes a network round trip. Say so on the button that asked
        # for it, rather than leaving a live control that answers nothing.
        self._refresh_library_chrome()

    def _displayed_view(self) -> str | None:
        """The history view the grid is painting right now, or None."""
        return _RAIL_HISTORY_VIEWS.get(self._active_key)

    def _on_history_page(self, view: str, runs: list, has_more: bool,
                         first: bool) -> None:
        self._hist_inflight.discard(view)
        self._hist_synced.add(view)
        self._hist_failed.discard(view)
        if first:
            self._hist_runs[view] = self._apply_favorite_overrides(runs)
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
                r for r in self._apply_favorite_overrides(runs)
                if _run_key(r) not in known)
            self._hist_pages_loaded += 1
            try:
                from ....core import telemetry_session_events
                telemetry_session_events.track_history_page_loaded(self._hist_pages_loaded)
            except Exception:
                pass  # nosec B110
        self._hist_has_more[view] = has_more
        if view == self._displayed_view():
            self._rebuild_current_grid()
        elif first:
            # The page landed while the user is on another tab. Pull the first
            # rows' pictures now, so the tab paints from the disk cache.
            self._prefetch_run_thumbs(self._hist_runs[view])
        self._refresh_library_chrome()

    def _on_history_failed(self, view: str, code: str) -> None:
        """A failed sync (including the endpoints not deployed yet) degrades to
        the cached/empty state - one quiet log line per view, no error spam."""
        self._hist_inflight.discard(view)
        self._hist_synced.add(view)
        self._hist_failed.add(view)
        if view not in self._hist_fail_logged:
            self._hist_fail_logged.add(view)
            log(f"Run history unavailable ({view}): {code}",
                Qgis.MessageLevel.Info)
        if view == self._displayed_view():
            self._rebuild_current_grid()
        self._refresh_library_chrome()

    def _retry_history_view(self, view: str) -> None:
        """Ask the server for this view again after a failed read."""
        self._hist_failed.discard(view)
        self._hist_synced.discard(view)
        self._sync_history_view(view)
        self._rebuild_current_grid()
        self._refresh_library_chrome()

    def _request_sign_in(self) -> None:
        """Close the library and put the dock on its sign-in page."""
        plugin = self._plugin
        self.reject()
        show = getattr(plugin, "_show_sign_in_page", None)
        if show is None:
            return
        try:
            show()
        except (RuntimeError, AttributeError):
            pass  # nosec B110 - the dock went with the window

    def _apply_favorite_overrides(self, runs: list) -> list:
        """Put this window's own star flips back on a freshly read page.

        A page already out when the user starred a run comes back saying the
        run is not starred, and it used to win: the row the user had just
        starred went back to grey while Favorites listed it.
        """
        if not self._fav_overrides:
            return runs
        for run in runs:
            wanted = self._fav_overrides.get(_run_key(run))
            if wanted is not None:
                run["is_favorite"] = wanted
        return runs

    def _history_view_loading(self, view: str) -> bool:
        """True while the first read of this view is still out."""
        return view in self._hist_inflight and view not in self._hist_synced

    # ---- the grid --------------------------------------------------------

    def _favorite_template_presets(self) -> list[dict]:
        """Starred templates, most recently starred first."""
        return [self._by_id[i] for i in favorite_template_ids() if i in self._by_id]

    def _local_recent_entries(self) -> list[dict]:
        """The local fallback feed (see recent_local.merge_local_recents)."""
        return merge_local_recents(self._history_local, self._recent_local)

    def _history_grid_signature(self, view: str) -> tuple:
        """Everything a history card reads. Two equal signatures paint the
        same grid, so the rebuild that follows a sync can be skipped."""
        runs = self._hist_runs.get(view) or []
        presets = (tuple(p.get("id", "") for p in self._favorite_template_presets())
                   if view == "favorites" else ())
        return (view, self._query, presets, self._run_actions_available(), tuple(
            (_run_key(r), bool(r.get("is_favorite")), r.get("objects"),
             r.get("tiles"), r.get("preview_request_id"),
             r.get("input_request_id")) for r in runs))

    def _history_runs_for_grid(self, view: str) -> list[dict]:
        """The runs this view paints, filtered by the search box.

        The box used to filter the object catalogue only, so typing anything
        while a history view was open emptied the grid.
        """
        runs = list(self._hist_runs.get(view) or [])
        if not self._query:
            return runs
        return [r for r in runs
                if self._query in (r.get("prompt") or "").lower()]

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
        runs = self._history_runs_for_grid(view)
        presets = (self._favorite_template_presets()
                   if view == "favorites" and not self._query else [])
        if not runs and not presets:
            self._paint_history_empty_state(view)
            return
        sections: list[tuple] = []
        if presets:
            # One tab, both kinds of star: the objects the user keeps around,
            # then the detections they kept. Objects come first because they are
            # what a new run starts from.
            sections.append((_DayHeader(tr("Starred objects"), self._grid_host),
                             self._build_preset_cards(presets)))
        for label, day_runs in group_runs_by_day(runs):
            sections.append((_DayHeader(label, self._grid_host),
                             self._build_run_cards(day_runs, view)))
        self._place_grouped_grid(sections)
        # The first rows are on screen whatever the layout settles to, so their
        # pictures are asked for here rather than one turn of the event loop
        # later: the round trip is already under way by the time the grid is
        # painted. The passes below still cover everything further down.
        for eager_run, eager_card in self._run_cards[:_EAGER_THUMB_CARDS]:
            self._request_run_thumb(eager_run, eager_card)
        QTimer.singleShot(0, self._load_visible_cards)
        QTimer.singleShot(80, self._load_visible_cards)

    def _paint_history_empty_state(self, view: str) -> None:
        """The one thing this view has to say, and the one thing to do here.

        Order matters: a signed-out account, a read that never came back and a
        genuinely empty history are three different states, and telling a new
        user their history is empty when the read simply failed sends them
        looking for runs they never lost.
        """
        if self._query:
            # The box searches what is loaded, and the account holds more than
            # one page. Saying so beats a flat "nothing", which would read as
            # "you never ran this" when the run is one page further back.
            if self._hist_has_more.get(view):
                self._empty_label(
                    tr("No run matches that search. Load older runs to look "
                       "further back."), "⌕")
            else:
                self._empty_label(tr("No run matches that search."), "⌕")
            return
        if not self._auth:
            sign_in = (tr("Sign in"), self._request_sign_in)
            if self._paint_local_recents_grid(
                    tr("Saved on this computer"),
                    tr("Sign in to see every run on your account."), sign_in):
                return
            if view == "favorites":
                self._empty_label(
                    tr("Sign in to keep detections here."), "★", sign_in)
            else:
                self._empty_label(
                    tr("Sign in to see your past runs."), action=sign_in)
            return
        if self._history_view_loading(view):
            self._empty_label(tr("Loading your runs..."), "◌")
            return
        if view in self._hist_failed:
            retry = (tr("Retry"), lambda: self._retry_history_view(view))
            if view == "all" and self._paint_local_recents_grid(
                    tr("Saved on this computer"),
                    tr("Could not reach TerraLab. Your runs are still there."),
                    retry):
                return
            self._empty_label(
                tr("Could not reach TerraLab. Your saved runs are still "
                   "there."), "⚠", retry)
            return
        if view == "favorites":
            self._empty_label(
                tr("Star a run or an object to keep it here."), "★")
            return
        self._empty_label(
            tr("No runs yet. Your Automatic runs appear here, ready to "
               "reuse, restore or export."))

    def _paint_local_recents_grid(self, title: str, note: str,
                                  action: tuple) -> bool:
        """Paint the local prompt-only feed under a header that names the
        state it stands in for. False when there is nothing local to show.

        Never the face of Recent: this is what the window falls back to when
        the account cannot answer, and the header says exactly that so the
        list is not mistaken for the run history itself.
        """
        entries = self._local_recent_entries()
        if not entries:
            return False
        cards = []
        for entry in entries:
            card = _RecentCard(recent_view(entry, self._by_token), self._grid_host,
                               view_only=self._view_only)
            card.activated.connect(self._on_recent_activated)
            card.rerun_requested.connect(self._on_recent_rerun)
            card.reuse_prompt_requested.connect(self._on_recent_reuse_prompt)
            cards.append(card)
        header = _DayHeader(title, self._grid_host, note=note, action=action)
        self._place_grouped_grid([(header, cards)])
        return True

    def _run_actions_available(self) -> bool:
        """Whether a card's own buttons can do anything right now.

        Four things have to hold: an account to read the run back from, a
        plugin to put it on the map, no run already in flight, and a window
        that is not in its browse-only face.
        """
        return bool(self._auth) and self._plugin is not None \
            and not self._hist_busy and not self._view_only

    def _build_run_cards(self, runs: list[dict], view: str) -> list:
        cards = []
        can_act = self._run_actions_available()
        for run in runs:
            card = _RunCard(run, view, parent=self._grid_host,
                            can_star=bool(self._auth), can_act=can_act)
            card.opened.connect(self._open_run_detail)
            card.star_toggled.connect(self._toggle_favorite)
            card.restore_requested.connect(self._request_restore)
            card.rerun_requested.connect(self._request_rerun)
            card.export_requested.connect(self._request_export)
            card.delete_requested.connect(self._request_delete)
            cards.append(card)
            self._run_cards.append((run, card))
            # Registered up front, not when the images are requested: an
            # off-screen card still has to follow a favorite toggle.
            self._hist_cards[_run_key(run)] = card
        return cards

    def _set_run_actions_enabled(self, usable: bool) -> None:
        """Show or hide every card's action row in one pass, so a fetch that
        started from one card does not leave the others live."""
        for _run, card in list(self._run_cards):
            setter = getattr(card, "set_actions_enabled", None)
            if setter is not None:
                setter(usable)

    # ---- thumbnails ------------------------------------------------------

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

    def _run_thumb_urls(self, run: dict) -> dict[str, tuple]:
        """Where to fetch each half of one run's comparison, first and second.

        The payload carries a ready-to-use address for the card-sized copy of
        each half: signed, so no header, and one hop shorter than asking us for
        it. That is the first address. Our own route is the second, and it is
        what answers on a network that blocks storage, on a run archived before
        the server built these copies, and for any older server that sends no
        such address at all.

        Both addresses hand back the same picture, and the card stores it under
        one key, so whichever one answers, the next open reads it off the disk.
        """
        width = self._card_thumb_width()
        preview_id = str(run.get("preview_request_id") or "")
        input_id = str(run.get("input_request_id") or "") or preview_id
        urls: dict[str, tuple] = {}
        for which, rid, direct in (
            ("input", input_id, run.get("input_thumb_url")),
            # preview_url is the full-size address older servers already send.
            # It is worse than a card-sized copy and better than nothing.
            ("preview", preview_id,
             run.get("preview_thumb_url") or run.get("preview_url")),
        ):
            route = (self._artifact_url(rid, which, width),
                     self._auth or None) if rid else None
            first = str(direct or "")
            if first.startswith(("http://", "https://")):
                urls[which] = (first, None) + (route or ())
            elif route is not None:
                urls[which] = route
        return urls

    def _request_run_thumb(self, run: dict, card: _RunCard) -> None:
        """Fetch both halves of a run's comparison: the imagery as it was sent
        and the same tile with the masks painted on.

        The archived input is strictly more available than the overlay: a tile
        that found nothing has an input and no preview, so the input is what
        keeps the card readable.
        """
        key = card.artifact_key()
        if not key:
            card.mark_missing("input")
            card.mark_missing("preview")
            return
        urls = self._run_thumb_urls(run)
        self._thumb_cards[key] = card
        card.request_artifacts(self._hist_loader, urls,
                               variant=str(self._card_thumb_width()))

    def _prefetch_run_thumbs(self, runs: list) -> None:
        """Pull the first rows' pictures while the user is on another tab.

        No card exists yet, so nothing is painted: the bytes land in the disk
        cache and the grid reads them off the disk when the tab opens. Bounded
        to the rows that will be on screen, so a visit that never reaches the
        history costs a handful of small images and no more.
        """
        loader = getattr(self, "_hist_loader", None)
        if loader is None:
            return
        width = str(self._card_thumb_width())
        for run in list(runs)[:_EAGER_THUMB_CARDS]:
            key = str(run.get("preview_request_id")
                      or run.get("input_request_id") or "")
            if not key:
                continue
            for which, entry in self._run_thumb_urls(run).items():
                loader.request(key, which, entry[0],
                               headers=entry[1] if len(entry) > 1 else None,
                               variant=width, immutable=True,
                               fallback_url=entry[2] if len(entry) > 2 else None,
                               fallback_headers=entry[3] if len(entry) > 3 else None)

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

    # ---- one run's detail popup ------------------------------------------

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

    # ---- favorites -------------------------------------------------------

    def _toggle_favorite(self, run: dict, is_favorite: bool) -> None:
        """Optimistic star: flip locally at once, sync in the background,
        revert on error (AI Edit's _GenerationFavoriteWorker pattern)."""
        run_id = run.get("run_id")
        if not run_id or not self._auth:
            return
        run["is_favorite"] = is_favorite
        self._fav_overrides[_run_key(run)] = is_favorite
        self._apply_favorite_ui(run, is_favorite)
        worker = _RunFavoriteWorker(
            self._history_client(), self._auth, str(run_id), is_favorite)
        worker.done.connect(self._on_favorite_done)
        self._track_live_worker(worker, "done")
        park_orphaned_worker(worker)
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
            # Deferred: this runs from the star's own click handler, and the
            # rebuild destroys the card that emitted it. Tearing a widget down
            # inside its own signal aborts QGIS on Qt6.
            QtC.safe_single_shot(0, self, self._rebuild_current_grid)
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
        self._refresh_library_chrome()

    def _on_favorite_done(self, run_id: str, is_favorite: bool, ok: bool) -> None:
        if ok:
            try:
                from ....core import telemetry_session_events
                telemetry_session_events.track_history_favorite_toggled(run_id, is_favorite)
            except Exception:
                pass  # nosec B110
            run_history_cache.save_runs(self._hist_runs.get("all") or [])
            return
        # Revert the optimistic flip, and stop forcing it onto later pages:
        # the server refused it, so its answer is the right one from here.
        self._fav_overrides.pop(str(run_id), None)
        for view_runs in self._hist_runs.values():
            for r in view_runs:
                if str(r.get("run_id") or "") == run_id:
                    self._apply_favorite_ui(r, not is_favorite)
                    return

    # ---- delete ----------------------------------------------------------

    def _request_delete(self, run: dict, _detail_dlg=None) -> None:
        """Take one run out of the history, after asking.

        The server soft-deletes, so nothing about the run is destroyed: what
        goes is its place in this list. The question is still asked, because
        from here the user has no way back to it.
        """
        run_id = str(run.get("run_id") or "")
        if not run_id or not self._auth:
            return
        answer = QMessageBox.question(
            self._detail_dlg or self, tr("Delete run"),
            tr("Remove this run from your history? Its detections stay stored, "
               "but it will not be listed here any more."),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._drop_run_locally(run_id)
        if self._detail_dlg is not None:
            try:
                self._detail_dlg.accept()
            except RuntimeError:
                pass
        worker = _RunDeleteWorker(
            self._history_client(), self._auth, run_id, True)
        worker.done.connect(self._on_delete_done)
        self._track_live_worker(worker, "done")
        park_orphaned_worker(worker)
        worker.start()

    def _drop_run_locally(self, run_id: str) -> None:
        """Remove one run from every view and repaint, before the server has
        answered: a delete that waits a round trip reads as a dead button."""
        for view, view_runs in list(self._hist_runs.items()):
            self._hist_runs[view] = [
                r for r in view_runs if str(r.get("run_id") or "") != run_id]
        run_history_cache.save_runs(self._hist_runs.get("all") or [])
        forget_run_mark(run_id)
        self._rebuild_current_grid()
        self._refresh_library_chrome()

    def _on_delete_done(self, run_id: str, deleted: bool, ok: bool) -> None:
        if ok:
            return
        # The row is still on the account, so the list is now lying. Say so and
        # read the view again rather than leaving a run the user cannot see.
        log(f"Run history delete refused: {run_id[:8]}",
            Qgis.MessageLevel.Warning)
        QMessageBox.warning(
            self, tr("Segment library"),
            tr("Could not remove this run. Try again later."))
        for view in ("all", "favorites"):
            self._hist_synced.discard(view)
        view = self._displayed_view()
        if view is not None:
            self._sync_history_view(view)

    # ---- what a finished action leaves behind ----------------------------

    def _mark_run_done(self, run: dict, kind: str) -> None:
        """Record that this computer restored or exported a run, and repaint
        the card so the list says so without a reload."""
        key = _run_key(run)
        if not key:
            return
        set_run_mark(key, kind)
        card = self._hist_cards.get(key)
        if card is not None:
            try:
                card.refresh_texts()
            except RuntimeError:
                pass  # nosec B110 - the card went with the grid
