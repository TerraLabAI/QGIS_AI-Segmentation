"""Background QThread workers for the Segment library's run history.

All network runs OFF the GUI thread (AI Edit's proven pattern). Workers are
detached into a module-level set so closing the dialog mid-fetch never
garbage-collects a running QThread (which would abort the QGIS process).
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QThread, pyqtSignal

from .common import _history_error

# In-flight background workers, held independently of any dialog. A running
# QThread that loses its last Python reference can be garbage-collected and
# destroyed mid-run, which aborts the QGIS process. Keeping the worker here
# until it emits finished lets the dialog be closed at any time while the
# blocking fetch is still going, without crashing.
_INFLIGHT_WORKERS: set = set()


def _detach_worker(worker: QThread) -> None:
    _INFLIGHT_WORKERS.add(worker)
    worker.finished.connect(lambda: _INFLIGHT_WORKERS.discard(worker))
    worker.finished.connect(worker.deleteLater)


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


class _RunFetchWorker(QThread):
    """Fetch everything a restore/export needs for one run: the per-tile
    detail rows, then each tile's stored masks (archive first, DB copy
    fallback). One background thread for the whole batch."""

    fetched = pyqtSignal(dict, list, dict)  # run, tiles, masks_per_tile
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
        # Run-level fields the list payload may lack (threshold and friends).
        for key in ("threshold", "mask_threshold", "crs_authid", "pixel_size_m"):
            if self._run.get(key) is None and detail.get(key) is not None:
                self._run[key] = detail.get(key)
        masks_per_tile: dict = {}
        for tile in tiles:
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
        if not masks_per_tile:
            self.failed.emit("masks: none available")
            return
        self.fetched.emit(self._run, tiles, masks_per_tile)
