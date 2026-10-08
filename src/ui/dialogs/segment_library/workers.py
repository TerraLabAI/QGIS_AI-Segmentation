










from __future__ import annotations

import sys
import time

from qgis.PyQt.QtCore import QThread, pyqtSignal

from ....core.server_dials import dial_in_range
from ....core.surface_dials import (
    library_mask_budget_base_s,
    library_mask_budget_max_s,
    library_mask_budget_per_tile_s,
)
from ....core.tile_filter_answer import stored_model_mask_count
from .common import _history_error


_HISTORY_PAGE_SIZE = 12







_MASK_BUDGET_BASE_S = 30.0
_MASK_BUDGET_PER_TILE_S = 2.0
_MASK_BUDGET_MAX_S = 600.0



_MASK_FETCH_BATCH = 6


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
        from ....core.activation_manager import auth_revision

        self._client = client
        self._auth = dict(auth)
        self._auth_revision = auth_revision()
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
        if skipped and self._export is not None:


            self.failed.emit(f"masks: {skipped} tile(s) unavailable for export")
            return
        if not masks_per_tile:
            self.failed.emit("masks: none available")
            return
        outcome = self._decode(tiles, masks_per_tile)
        if outcome is None:
            return
        if self._fetch_cancelled():
            self.cancelled.emit()
            return
        outcome["tiles_skipped"] = skipped
        self.fetched.emit(self._run, tiles, outcome)

    def _fetch_tiles(self) -> list | None:

        if self._fetch_cancelled():
            self.cancelled.emit()
            return None
        try:
            detail = self._client.get_seg_run_detail(
                self._auth,
                run_id=self._run.get("run_id"),
                group_key=self._run.get("group_key"),
                prompt=str(self._run.get("prompt") or "").strip(),
            )
        except Exception as err:  # noqa: BLE001
            if self._fetch_cancelled():
                self.cancelled.emit()
            else:
                self.failed.emit(f"detail exception: {err}")
            return None
        if self._fetch_cancelled():
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





        for key in ("threshold", "mask_threshold", "crs_authid", "pixel_size_m", "decisions",
                    "zone_keep_margin_m", "run_policy", "resolved"):
            if self._run.get(key) is None and detail.get(key) is not None:
                self._run[key] = detail.get(key)
        self._fetch_restore_plan()
        return tiles

    def _fetch_restore_plan(self) -> None:




        prompt = str(self._run.get("prompt") or "").strip()
        if self._export is not None or not prompt or self._run.get("restore_plan") is not None:
            return
        try:
            plan = self._client.get_seg_run_plan(prompt, None, None, auth=self._auth)
        except Exception:  # noqa: BLE001
            return
        if isinstance(plan, dict) and not plan.get("error"):
            self._run["restore_plan"] = plan

    def _fetch_masks(self, tiles: list) -> tuple:





        masks_per_tile: dict = {}
        complete_tiles: set = set()
        total = len(tiles)
        budget = min(library_mask_budget_max_s(_MASK_BUDGET_MAX_S),
                     library_mask_budget_base_s(_MASK_BUDGET_BASE_S)
                     + library_mask_budget_per_tile_s(_MASK_BUDGET_PER_TILE_S) * total)
        batch = int(dial_in_range(
            "tuning.library.mask_fetch_batch", _MASK_FETCH_BATCH, 1, 12))
        started = time.monotonic()
        for first in range(0, total, batch):
            if self._fetch_cancelled():
                self.cancelled.emit()
                return None, 0
            if time.monotonic() - started > budget:
                break
            self.progress.emit("masks", first, total)
            chunk = [t for t in tiles[first:first + batch] if t.get("request_id")]
            group = [t["request_id"] for t in chunk]
            archived = [t["request_id"] for t in chunk if t.get("has_archive", True)]
            answers: list = []
            if archived:
                try:
                    answers = self._client.fetch_run_masks_many(
                        self._auth, archived,
                        should_abort=self._fetch_cancelled)
                except Exception:  # noqa: BLE001
                    answers = []
            tiles_by_id = {tile["request_id"]: tile for tile in chunk}
            for rid, resp in zip(archived, answers):
                masks = self._validated_masks(resp, tiles_by_id[rid])
                if masks is not None:
                    masks_per_tile[rid] = masks


                    record = resp.get("tile_filters") if isinstance(resp, dict) else None
                    if isinstance(record, dict) and not isinstance(
                            tiles_by_id[rid].get("tile_filters"), dict):
                        tiles_by_id[rid]["tile_filters"] = record


                    if self._mask_count_complete(
                            stored_model_mask_count(masks), tiles_by_id[rid]):
                        complete_tiles.add(rid)
            if self._fetch_cancelled():
                self.cancelled.emit()
                return None, 0
            for rid in group:
                if self._fetch_cancelled():
                    self.cancelled.emit()
                    return None, 0
                if rid in complete_tiles:
                    continue


                try:
                    resp = self._client.get_detection_status(rid, self._auth)
                    masks = self._validated_masks(resp, tiles_by_id[rid])
                    if masks is not None:
                        complete = self._mask_count_complete(
                            stored_model_mask_count(masks), tiles_by_id[rid])
                        if (complete or rid not in masks_per_tile
                                or len(masks) > len(masks_per_tile[rid])):
                            masks_per_tile[rid] = masks
                        if complete:
                            complete_tiles.add(rid)
                except Exception:  # noqa: BLE001
                    pass  # nosec B110
        if self._fetch_cancelled():
            self.cancelled.emit()
            return None, 0
        skipped = sum(1 for tile in tiles
                      if not tile.get("request_id") or tile["request_id"] not in complete_tiles)
        self.progress.emit("masks", total - skipped, total)
        return masks_per_tile, skipped

    def _validated_masks(self, payload, tile: dict) -> list | None:







        from ....core.mask_crops import iter_detection_crops
        from ....core.tile_manager import TILE_SIZE

        if isinstance(payload, list):
            masks = payload
        elif isinstance(payload, dict) and _history_error(payload) is None:
            masks = payload.get("masks")
        else:
            return None
        try:
            response = {"masks": masks, "width": tile.get("output_width"),
                        "height": tile.get("output_height")}
            for _crop, _score, _box in iter_detection_crops(
                    response, TILE_SIZE, TILE_SIZE, -sys.float_info.max, strict=True):
                if self._fetch_cancelled():
                    return None
        except (TypeError, ValueError, OverflowError):
            return None
        return masks

    def _fetch_cancelled(self) -> bool:

        from ....core.activation_manager import auth_revision

        return self.isInterruptionRequested() or self._auth_revision != auth_revision()

    @staticmethod
    def _mask_count_complete(count: int, tile: dict) -> bool:





        expected = tile.get("mask_count")
        return expected is None or (
            isinstance(expected, int) and not isinstance(expected, bool)
            and expected >= 0 and count == expected)

    def _decode(self, tiles: list, masks_per_tile: dict) -> dict | None:


        from ...plugin.run_restore import decode_run_masks, export_decoded_run

        if self._fetch_cancelled():
            self.cancelled.emit()
            return None
        from ....core.detection_policy_core import policy_scope
        from ...plugin.run_restore import restore_run_policy

        total = len(tiles)
        try:
            with policy_scope(restore_run_policy(self._run)[0]):
                decoded = decode_run_masks(
                    self._run, tiles, masks_per_tile, self._merge_separate,
                    on_tile=lambda done, _total: self.progress.emit(
                        "decode", done, total),
                    is_cancelled=self._fetch_cancelled)
        except Exception as err:  # noqa: BLE001
            if self._fetch_cancelled():
                self.cancelled.emit()
            else:
                self.failed.emit(f"decode exception: {err}")
            return None
        if decoded is None:
            self.cancelled.emit()
            return None
        if self._export is None:
            return decoded
        if self._fetch_cancelled():


            self.cancelled.emit()
            return None
        driver, confidence, path = self._export
        self.progress.emit("write", 0, 0)
        try:
            decoded["export"] = export_decoded_run(
                decoded, float(confidence), path, driver)
        except Exception as err:  # noqa: BLE001
            if self._fetch_cancelled():
                self.cancelled.emit()
            else:
                self.failed.emit(f"export exception: {err}")
            return None


        decoded["objects"] = []
        return decoded
