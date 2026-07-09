










from __future__ import annotations

import time

from qgis.PyQt.QtCore import QThread, pyqtSignal

from ....core.server_dials import dial_in_range
from ....core.surface_dials import (
    library_mask_budget_base_s,
    library_mask_budget_max_s,
    library_mask_budget_per_tile_s,
)
from .common import _history_error


_HISTORY_PAGE_SIZE = 12







_MASK_BUDGET_BASE_S = 30.0
_MASK_BUDGET_PER_TILE_S = 2.0
_MASK_BUDGET_MAX_S = 600.0


class _HistoryFetchWorker(QThread):




    page_fetched = pyqtSignal(str, list, bool, bool)
    failed = pyqtSignal(str, str)

    def __init__(self, client, auth: dict, view: str,
                 before: str | None = None, parent=None):
        super().__init__(parent)
        self._client = client
        self._auth = auth
        self._view = view
        self._before = before

    def run(self):


        if self.isInterruptionRequested():
            return
        try:
            resp = self._client.get_seg_history(
                self._auth,
                limit=dial_in_range(
                    "tuning.library.history_page_size", _HISTORY_PAGE_SIZE, 4, 100),
                before=self._before,
                favorites_only=self._view == "favorites",
                deleted=False,
            )
        except Exception as err:  # noqa: BLE001
            if not self.isInterruptionRequested():
                self.failed.emit(self._view, f"exception: {err}")
            return
        if self.isInterruptionRequested():
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


    done = pyqtSignal(str, bool, bool)

    def __init__(self, client, auth: dict, run_id: str,
                 is_favorite: bool, parent=None):
        super().__init__(parent)
        self._client = client
        self._auth = auth
        self._run_id = run_id
        self._fav = is_favorite

    def run(self):


        if self.isInterruptionRequested():
            return
        ok = False
        try:
            resp = self._client.set_seg_run_favorite(
                self._auth, self._run_id, self._fav)
            ok = _history_error(resp) is None
        except Exception:  # noqa: BLE001
            ok = False
        if self.isInterruptionRequested():
            return
        self.done.emit(self._run_id, self._fav, ok)


class _RunDeleteWorker(QThread):






    done = pyqtSignal(str, bool, bool)

    def __init__(self, client, auth: dict, run_id: str,
                 deleted: bool = True, parent=None):
        super().__init__(parent)
        self._client = client
        self._auth = auth
        self._run_id = run_id
        self._deleted = deleted

    def run(self):


        if self.isInterruptionRequested():
            return
        ok = False
        try:
            if self._deleted:
                resp = self._client.delete_seg_run(self._auth, self._run_id)
            else:
                resp = self._client.undelete_seg_run(self._auth, self._run_id)
            ok = _history_error(resp) is None
        except Exception:  # noqa: BLE001
            ok = False
        if self.isInterruptionRequested():
            return
        self.done.emit(self._run_id, self._deleted, ok)


class _RunZoneFetchWorker(QThread):






    fetched = pyqtSignal(dict, list)
    failed = pyqtSignal(str)

    def __init__(self, client, auth: dict, run: dict, parent=None):
        super().__init__(parent)
        self._client = client
        self._auth = auth
        self._run = dict(run)

    def run(self):


        if self.isInterruptionRequested():
            return
        try:
            detail = self._client.get_seg_run_detail(
                self._auth,
                run_id=self._run.get("run_id"),
                group_key=self._run.get("group_key"),
            )
        except Exception as err:  # noqa: BLE001
            if not self.isInterruptionRequested():
                self.failed.emit(f"detail exception: {err}")
            return
        if self.isInterruptionRequested():
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













    fetched = pyqtSignal(dict, list, dict)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    progress = pyqtSignal(str, int, int)

    def __init__(self, client, auth: dict, run: dict, merge_separate: bool,
                 export: tuple | None = None, parent=None):
        super().__init__(parent)
        self._client = client
        self._auth = auth
        self._run = dict(run)
        self._merge_separate = bool(merge_separate)

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

        for key in ("threshold", "mask_threshold", "crs_authid", "pixel_size_m"):
            if self._run.get(key) is None and detail.get(key) is not None:
                self._run[key] = detail.get(key)
        return tiles

    def _fetch_masks(self, tiles: list) -> tuple:





        masks_per_tile: dict = {}
        total = len(tiles)
        budget = min(library_mask_budget_max_s(_MASK_BUDGET_MAX_S),
                     library_mask_budget_base_s(_MASK_BUDGET_BASE_S)
                     + library_mask_budget_per_tile_s(_MASK_BUDGET_PER_TILE_S) * total)
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


                    if isinstance(resp, list) or _history_error(resp) is None:
                        payload = resp
                except Exception:  # noqa: BLE001
                    payload = None
            if payload is None:


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
        if self.isInterruptionRequested():


            self.cancelled.emit()
            return None
        driver, confidence, path = self._export
        self.progress.emit("write", 0, 0)
        try:
            decoded["export"] = export_decoded_run(
                decoded, float(confidence), path, driver)
        except Exception as err:  # noqa: BLE001
            self.failed.emit(f"export exception: {err}")
            return None


        decoded["objects"] = []
        return decoded
