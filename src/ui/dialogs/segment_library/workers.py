"""Background QThread workers for the Segment library's run history.

All network runs OFF the GUI thread (AI Edit's proven pattern), and so does
every heavy thing the fetched data feeds: decoding a past run's stored masks
into geometry, and writing its direct export. Both used to run in the click
handler and froze QGIS for a minute or more on a big run.

Every worker is handed to ``park_orphaned_worker`` by its call site, so closing
the dialog or quitting QGIS mid-fetch joins the thread instead of destroying it
under itself (which aborts the QGIS process).
"""
from __future__ import annotations

import time

from qgis.PyQt.QtCore import QThread, pyqtSignal

from .common import _history_error

# Wall clock the whole mask loop may spend. Each tile costs one archive fetch
# (30 s timeout) plus a status fallback (15 s), so on a dead link a big run
# would hold the thread for hours. Cancel is the user's way out; this is the
# backstop for the user who walked away. Generous per tile so a big run on a
# healthy link always finishes, hard ceiling so a bad one cannot run forever.
_MASK_BUDGET_BASE_S = 30.0
_MASK_BUDGET_PER_TILE_S = 2.0
_MASK_BUDGET_MAX_S = 600.0


class _HistoryFetchWorker(QThread):
    """Background fetch of one page of the run history for one view
    ('all' / 'favorites'). ``before=None`` is the first page
    (the initial sync); a cursor fetches an older page (Load older runs)."""

    page_fetched = pyqtSignal(str, list, bool, bool)  # view, runs, has_more, first
    failed = pyqtSignal(str, str)                     # view, code

    def __init__(self, client, auth: dict, view: str,
                 before: str | None = None, parent=None):
        super().__init__(parent)
        self._client = client
        self._auth = auth
        self._view = view
        self._before = before

    def run(self):
        try:
            resp = self._client.get_seg_history(
                self._auth,
                limit=12,
                before=self._before,
                favorites_only=self._view == "favorites",
                deleted=False,
            )
        except Exception as err:  # noqa: BLE001
            self.failed.emit(self._view, f"exception: {err}")
            return
        code = _history_error(resp)
        if code is not None:
            self.failed.emit(self._view, code)
            return
        runs = [r for r in (resp.get("runs") or []) if isinstance(r, dict)]
        self.page_fetched.emit(
            self._view, runs, bool(resp.get("has_more", False)),
            self._before is None)


class _RunFavoriteWorker(QThread):
    """Star/unstar one run; reports back so the optimistic UI can revert."""

    done = pyqtSignal(str, bool, bool)  # run_id, is_favorite, ok

    def __init__(self, client, auth: dict, run_id: str,
                 is_favorite: bool, parent=None):
        super().__init__(parent)
        self._client = client
        self._auth = auth
        self._run_id = run_id
        self._fav = is_favorite

    def run(self):
        ok = False
        try:
            resp = self._client.set_seg_run_favorite(
                self._auth, self._run_id, self._fav)
            ok = _history_error(resp) is None
        except Exception:  # noqa: BLE001
            ok = False
        self.done.emit(self._run_id, self._fav, ok)


class _RunZoneFetchWorker(QThread):
    """Fetch just the per-tile rows of one run, for "run this zone again".

    Deliberately not _RunFetchWorker: that one also pulls every tile's stored
    masks, which is the expensive part and which pointing at a zone does not
    need."""

    fetched = pyqtSignal(dict, list)  # run, tiles
    failed = pyqtSignal(str)

    def __init__(self, client, auth: dict, run: dict, parent=None):
        super().__init__(parent)
        self._client = client
        self._auth = auth
        self._run = dict(run)

    def run(self):
        try:
            detail = self._client.get_seg_run_detail(
                self._auth,
                run_id=self._run.get("run_id"),
                group_key=self._run.get("group_key"),
            )
        except Exception as err:  # noqa: BLE001
            self.failed.emit(f"detail exception: {err}")
            return
        code = _history_error(detail)
        if code is not None:
            self.failed.emit(f"detail: {code}")
            return
        tiles = [t for t in (detail.get("tiles") or []) if isinstance(t, dict)]
        if not tiles:
            self.failed.emit("detail: no tiles")
            return
        self.fetched.emit(self._run, tiles)


class _RunFetchWorker(QThread):
    """Everything a restore or a direct export of one run needs, end to end on
    one background thread: the per-tile detail rows, each tile's stored masks
    (archive first, DB copy fallback), the decode + merge into geometry, and,
    when an export was asked for, the file write itself.

    Stoppable: ``requestInterruption()`` is polled between tiles and between
    masks, so Cancel lands within one network call instead of at the end of the
    run. The mask loop also carries a wall-clock budget.
    """

    #: run, tiles, outcome. The outcome is the decode bundle
    #: (run_restore.decode_run_masks) plus "tiles_skipped", and "export" when
    #: this worker was given a file to write.
    fetched = pyqtSignal(dict, list, dict)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    #: phase key ("masks", "decode", "write"), done, total.
    progress = pyqtSignal(str, int, int)

    def __init__(self, client, auth: dict, run: dict, merge_separate: bool,
                 export: tuple | None = None, parent=None):
        super().__init__(parent)
        self._client = client
        self._auth = auth
        self._run = dict(run)
        self._merge_separate = bool(merge_separate)
        # (driver, confidence, path) for a direct export, None for a restore.
        self._export = export

    def run(self):
        tiles = self._fetch_tiles()
        if tiles is None:
            return
        masks_per_tile, skipped = self._fetch_masks(tiles)
        if masks_per_tile is None:
            return
        if not masks_per_tile:
            self.failed.emit("masks: none available")
            return
        outcome = self._decode(tiles, masks_per_tile)
        if outcome is None:
            return
        outcome["tiles_skipped"] = skipped
        self.fetched.emit(self._run, tiles, outcome)

    def _fetch_tiles(self) -> list | None:
        """The run's per-tile rows, or None once the failure was reported."""
        try:
            detail = self._client.get_seg_run_detail(
                self._auth,
                run_id=self._run.get("run_id"),
                group_key=self._run.get("group_key"),
            )
        except Exception as err:  # noqa: BLE001
            self.failed.emit(f"detail exception: {err}")
            return None
        if self.isInterruptionRequested():
            self.cancelled.emit()
            return None
        code = _history_error(detail)
        if code is not None:
            self.failed.emit(f"detail: {code}")
            return None
        tiles = [t for t in (detail.get("tiles") or []) if isinstance(t, dict)]
        if not tiles:
            self.failed.emit("detail: no tiles")
            return None
        # Run-level fields the list payload may lack (threshold and friends).
        for key in ("threshold", "mask_threshold", "crs_authid", "pixel_size_m"):
            if self._run.get(key) is None and detail.get(key) is not None:
                self._run[key] = detail.get(key)
        return tiles

    def _fetch_masks(self, tiles: list) -> tuple:
        """(masks by request id, tiles given up on), or (None, 0) when stopped.

        Running out of budget is not a failure: what came back still restores,
        and the caller tells the user how many tiles are missing from it.
        """
        masks_per_tile: dict = {}
        total = len(tiles)
        budget = min(_MASK_BUDGET_MAX_S,
                     _MASK_BUDGET_BASE_S + _MASK_BUDGET_PER_TILE_S * total)
        started = time.monotonic()
        skipped = 0
        for index, tile in enumerate(tiles):
            if self.isInterruptionRequested():
                self.cancelled.emit()
                return None, 0
            if time.monotonic() - started > budget:
                skipped = total - index
                break
            self.progress.emit("masks", index, total)
            rid = tile.get("request_id")
            if not rid:
                continue
            payload = None
            if tile.get("has_archive", True):
                try:
                    resp = self._client.fetch_run_masks(self._auth, rid)
                    # The archive is the raw masks JSON: either a bare list or
                    # a {"masks": [...]} wrapper (never an error-shaped dict).
                    if isinstance(resp, list) or _history_error(resp) is None:
                        payload = resp
                except Exception:  # noqa: BLE001
                    payload = None
            if payload is None:
                # Fallback: the DB copy via the existing status route (may be
                # truncated for very dense tiles, still better than nothing).
                try:
                    resp = self._client.get_detection_status(rid, self._auth)
                    if _history_error(resp) is None and resp.get("masks"):
                        payload = resp
                except Exception:  # noqa: BLE001
                    payload = None
            if payload is not None:
                masks_per_tile[rid] = payload
        self.progress.emit("masks", total - skipped, total)
        return masks_per_tile, skipped

    def _decode(self, tiles: list, masks_per_tile: dict) -> dict | None:
        """Masks to geometry, then the file write for an export. None once the
        stop or the failure was reported."""
        from ...plugin.run_restore import decode_run_masks, export_decoded_run

        total = len(tiles)
        try:
            decoded = decode_run_masks(
                self._run, tiles, masks_per_tile, self._merge_separate,
                on_tile=lambda done, _total: self.progress.emit(
                    "decode", done, total),
                is_cancelled=self.isInterruptionRequested)
        except Exception as err:  # noqa: BLE001
            self.failed.emit(f"decode exception: {err}")
            return None
        if decoded is None:
            self.cancelled.emit()
            return None
        if self._export is None:
            return decoded
        driver, confidence, path = self._export
        self.progress.emit("write", 0, 0)
        try:
            decoded["export"] = export_decoded_run(
                decoded, float(confidence), path, driver)
        except Exception as err:  # noqa: BLE001
            self.failed.emit(f"export exception: {err}")
            return None
        # The file is written and the GUI only needs the summary, so the
        # geometries are released here rather than travelling to it.
        decoded["objects"] = []
        return decoded
