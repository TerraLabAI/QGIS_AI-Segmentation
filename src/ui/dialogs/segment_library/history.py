






















from __future__ import annotations

from qgis.core import Qgis

from ....core import qt_compat as QtC
from ....core.i18n import tr
from ....core.logging_utils import log
from ....core.presets import run_history_cache
from ....core.presets.template_favorites import favorite_template_ids
from ....core.server_dials import dial_in_range
from ...plugin.shared import park_orphaned_worker
from ..confirm_dialog import question, warning_box
from .cards import _RecentCard
from .common import _RAIL_HISTORY_VIEWS, _run_key
from .detail import _RunDetailDialog
from .recent_local import merge_local_recents, recent_view
from .run_card import _DayHeader, _RunCard
from .run_marks import forget_run_mark, set_run_mark
from .run_summary import group_runs_by_day
from .workers import _HistoryFetchWorker, _RunDeleteWorker, _RunFavoriteWorker




_EAGER_THUMB_CARDS = 6



_LOCAL_RECENTS_SHOWN = 60


def _eager_thumb_cards() -> int:
    return dial_in_range("tuning.library.eager_thumb_cards", _EAGER_THUMB_CARDS, 0, 24)


class LibraryHistoryMixin:




    def showEvent(self, event):  # noqa: N802








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


        self._refresh_library_chrome()

    def _displayed_view(self) -> str | None:

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


            self._prefetch_run_thumbs(self._hist_runs[view])
        self._refresh_library_chrome()

    def _on_history_failed(self, view: str, code: str) -> None:


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

        self._hist_failed.discard(view)
        self._hist_synced.discard(view)
        self._sync_history_view(view)
        self._rebuild_current_grid()
        self._refresh_library_chrome()

    def _request_sign_in(self) -> None:

        plugin = self._plugin
        self.reject()
        show = getattr(plugin, "_show_sign_in_page", None)
        if show is None:
            return
        try:
            show()
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _apply_favorite_overrides(self, runs: list) -> list:






        if not self._fav_overrides:
            return runs
        for run in runs:
            wanted = self._fav_overrides.get(_run_key(run))
            if wanted is not None:
                run["is_favorite"] = wanted
        return runs

    def _history_view_loading(self, view: str) -> bool:

        return view in self._hist_inflight and view not in self._hist_synced



    def _favorite_template_presets(self) -> list[dict]:

        return [self._by_id[i] for i in favorite_template_ids() if i in self._by_id]

    def _local_recent_entries(self) -> list[dict]:

        return merge_local_recents(self._history_local, self._recent_local)

    def _history_grid_signature(self, view: str) -> tuple:


        runs = self._hist_runs.get(view) or []
        presets = (tuple(p.get("id", "") for p in self._favorite_template_presets())
                   if view == "favorites" else ())
        return (view, self._query, presets, self._run_actions_available(), tuple(
            (_run_key(r), bool(r.get("is_favorite")), r.get("objects"),
             r.get("tiles"), r.get("preview_request_id"),
             r.get("input_request_id")) for r in runs))

    def _history_runs_for_grid(self, view: str) -> list[dict]:





        runs = list(self._hist_runs.get(view) or [])
        if not self._query:
            return runs
        return [r for r in runs
                if self._query in (r.get("prompt") or "").lower()]

    def _rebuild_history_grid(self, view: str) -> None:




        signature = self._history_grid_signature(view)
        if signature == self._grid_signature and self._grid_widgets:


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



            sections.append((_DayHeader(tr("Starred objects"), self._grid_host),
                             self._build_preset_cards(presets)))
        for label, day_runs in group_runs_by_day(runs):
            sections.append((_DayHeader(label, self._grid_host),
                             self._build_run_cards(day_runs, view)))
        self._place_grouped_grid(sections)




        for eager_run, eager_card in self._run_cards[:_eager_thumb_cards()]:
            self._request_run_thumb(eager_run, eager_card)
        self._schedule_visible_cards_load()

    def _paint_history_empty_state(self, view: str) -> None:







        if self._query:



            if self._hist_has_more.get(view):
                self._empty_label(
                    tr("No run matches that search. Load older runs to look "
                       "further back."), "search")
            else:
                self._empty_label(tr("No run matches that search."), "search")
            return
        if not self._auth:
            sign_in = (tr("Sign in"), self._request_sign_in)
            if self._paint_local_recents_grid(
                    tr("Saved on this computer"),
                    tr("Sign in to see every run on your account."), sign_in):
                return
            if view == "favorites":
                self._empty_label(
                    tr("Sign in to keep detections here."), "star", sign_in)
            else:
                self._empty_label(
                    tr("Sign in to see your past runs."), action=sign_in)
            return
        if self._history_view_loading(view):
            self._empty_label(tr("Loading your runs..."), "clock")
            return
        if view in self._hist_failed:
            retry = (tr("Retry"), lambda: self._retry_history_view(view))
            if view == "all" and self._paint_local_recents_grid(
                    tr("Saved on this computer"),
                    tr("Could not reach TerraLab. Your runs are still there."),
                    retry):
                return


            self._empty_label(
                tr("Could not reach TerraLab. Your runs are still there."),
                "warning", retry)
            return
        if view == "favorites":
            self._empty_label(
                tr("Star a run or an object to keep it here."), "star")
            return
        self._empty_label(
            tr("No runs yet. Your Automatic runs appear here, ready to "
               "reuse, restore or export."))

    def _paint_local_recents_grid(self, title: str, note: str,
                                  action: tuple) -> bool:







        entries = self._local_recent_entries()
        if not entries:
            return False


        shown = int(dial_in_range(
            "library.local_recents_max", _LOCAL_RECENTS_SHOWN, 10, 500))
        cards = []
        for entry in entries[:shown]:
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


            self._hist_cards[_run_key(run)] = card
        return cards

    def _set_run_actions_enabled(self, usable: bool) -> None:


        for _run, card in list(self._run_cards):
            setter = getattr(card, "set_actions_enabled", None)
            if setter is not None:
                setter(usable)



    def _card_thumb_width(self) -> int:










        return dial_in_range("tuning.library.card_thumb_width_px", 512, 128, 1024)

    def _artifact_url(self, request_id: str, which: str,
                      width: int | None = None) -> str:








        from urllib.parse import quote

        url = "{}/api/ai-segmentation/image/{}?type={}&stream=1".format(
            self._base, quote(str(request_id), safe=""), quote(str(which), safe=""))
        if width:
            url += f"&w={int(width)}"
        return url

    def _run_thumb_urls(self, run: dict) -> dict[str, tuple]:












        width = self._card_thumb_width()
        preview_id = str(run.get("preview_request_id") or "")
        input_id = str(run.get("input_request_id") or "") or preview_id
        urls: dict[str, tuple] = {}
        for which, rid, direct in (
            ("input", input_id, run.get("input_thumb_url")),


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







        loader = getattr(self, "_hist_loader", None)
        if loader is None:
            return
        width = str(self._card_thumb_width())
        for run in list(runs)[:_eager_thumb_cards()]:
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
                pass

    def _on_thumb_failed(self, pid: str, which: str) -> None:
        if which not in ("input", "preview"):
            return
        card = self._thumb_cards.get(pid)
        if card is not None:
            try:
                card.mark_missing(which)
            except RuntimeError:
                pass



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


        self._fav_overrides.pop(str(run_id), None)
        for view_runs in self._hist_runs.values():
            for r in view_runs:
                if str(r.get("run_id") or "") == run_id:
                    self._apply_favorite_ui(r, not is_favorite)
                    return



    def _request_delete(self, run: dict, _detail_dlg=None) -> None:






        run_id = str(run.get("run_id") or "")
        if not run_id or not self._auth:
            return

        if not question(
                self._detail_dlg or self, tr("Delete this run?"),
                tr("It leaves your history. Its detections stay stored."),
                default_yes=False, destructive=True, yes_label=tr("Delete")):
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


        log(f"Run history delete refused: {run_id[:8]}",
            Qgis.MessageLevel.Warning)


        warning_box(self, tr("Could not remove this run. Try again later."))
        for view in ("all", "favorites"):
            self._hist_synced.discard(view)
        view = self._displayed_view()
        if view is not None:
            self._sync_history_view(view)



    def _mark_run_done(self, run: dict, kind: str) -> None:


        key = _run_key(run)
        if not key:
            return
        set_run_mark(key, kind)
        card = self._hist_cards.get(key)
        if card is not None:
            try:
                card.refresh_texts()
            except RuntimeError:
                pass  # nosec B110
