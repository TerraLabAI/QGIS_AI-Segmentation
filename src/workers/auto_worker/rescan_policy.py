










from __future__ import annotations

import logging
import time
from collections import deque

__all__ = [
    "AutoRescanPolicyMixin",
    "_RESPLIT_TIME_RATIO",
    "_SUBDIV_MAX_DEPTH",
    "logger",
]

logger = logging.getLogger(__name__)





















_SUBDIV_MAX_DEPTH = 0







_RESPLIT_TIME_RATIO = 1.0


class AutoRescanPolicyMixin:


    def _maybe_subdivide(self, tile_idx: int) -> bool:










        from ...core.tile_manager import subdivide_quadrants

        if self._stop_requested or self._stamps:
            return False
        if self._resplit_time_spent():
            return False
        depth = self._tile_depth.get(tile_idx, 0)
        if depth >= self._subdiv_max_depth or self._subdivide_budget < 4:
            return False
        try:
            tx, ty, tw, th = self._tiles[tile_idx]
        except (IndexError, ValueError):
            return False
        quads = subdivide_quadrants(
            tx, ty, tw, th,
            overlap_fraction=self._subdiv_overlap,
            min_parent_px=self._subdiv_min_parent_px,
        )
        if not quads:
            return False
        quads = [q for q in quads if self._quad_intersects_zone(q)]
        if not quads or len(quads) > self._subdivide_budget:
            return False
        self._subdivide_budget -= len(quads)
        self.tiles_subdivided += 1
        for spec in quads:
            self._pending_subtiles.append((spec, depth + 1, tile_idx))
        self._mark_rescanning(tile_idx, len(quads))
        logger.debug(
            "AutoDetectionWorker: tile %d saturated, re-split into %d "
            "quadrant(s) at depth %d", tile_idx, len(quads), depth + 1,
        )
        return True

    def _quad_intersects_zone(self, spec) -> bool:



        if self._clip_geom is None:
            return True
        try:
            from qgis.core import QgsGeometry, QgsRectangle

            bbox = self._make_tile_transform(*spec)["bbox_native"]
            rect = QgsGeometry.fromRect(
                QgsRectangle(bbox[0], bbox[1], bbox[2], bbox[3]))
            return bool(self._clip_geom.intersects(rect))
        except Exception:  # noqa: BLE001
            return True

    def _mark_rescanning(self, tile_idx: int, quads: int) -> None:






        root = self._billed_ancestor_of(tile_idx)
        if root is None:
            root = tile_idx
        first = root not in self._rescanning
        self._rescanning[root] = self._rescanning.get(root, 0) + quads
        if not first:
            return
        try:
            tx, ty, tw, th = self._tiles[root]
            bbox = self._make_tile_transform(tx, ty, tw, th)["bbox_native"]
            self.rescan_state.emit(root, bbox, True)
        except (IndexError, ValueError, KeyError, RuntimeError):
            self._rescanning.pop(root, None)

    def _settle_rescanning(self, tile_idx: int) -> None:




        self._settle_rescanning_root(self._billed_ancestor_of(tile_idx))

    def _settle_rescanning_root(self, root: int | None) -> None:



        if root is None or root not in self._rescanning:
            return
        left = self._rescanning[root] - 1
        if left > 0:
            self._rescanning[root] = left
            return
        del self._rescanning[root]
        try:
            self.rescan_state.emit(root, None, False)
        except RuntimeError:
            pass

    def _billed_ancestor_of(self, tile_idx: int) -> int | None:




        parent = self._parent_of.get(tile_idx)
        while parent is not None and self._parent_of.get(parent) is not None:
            parent = self._parent_of.get(parent)
        return parent

    def _drop_unsent_quadrants(self, pending: deque) -> int:

















        if not pending:
            return 0
        kept: deque = deque()
        dropped = 0
        while pending:
            item = pending.popleft()
            if self._billed_ancestor_of(item[0]) is None:
                kept.append(item)
                continue
            if self._parent_of.get(item[0]) in self._parents_with_child_results:
                kept.append(item)
                continue
            dropped += 1
            self._settle_rescanning(item[0])
        pending.extend(kept)
        if dropped:
            self._resplit_dropped += dropped
            logger.debug(
                "AutoDetectionWorker: re-split time budget spent, dropped %d "
                "unsent quadrant(s)", dropped)
        return dropped

    def _resplit_time_spent(self) -> bool:







        deadline = self._resplit_deadline
        return bool(deadline) and time.monotonic() > deadline

    def _drain_subtiles(self, pending: deque) -> int:







        if self._resplit_time_spent():


            if self._pending_subtiles:
                self._resplit_dropped += len(self._pending_subtiles)
                logger.debug(
                    "AutoDetectionWorker: re-split time budget spent, dropped %d "
                    "queued quadrant(s)", len(self._pending_subtiles))




                for _spec, _depth, parent_idx in self._pending_subtiles:
                    root = self._billed_ancestor_of(parent_idx)
                    self._settle_rescanning_root(
                        parent_idx if root is None else root)
                self._pending_subtiles.clear()






            return -self._drop_unsent_quadrants(pending)
        from ...core.tile_manager import TILE_SIZE

        added = 0
        while self._pending_subtiles:
            spec, depth, parent_idx = self._pending_subtiles.pop()
            idx = len(self._tiles)
            self._tiles.append(spec)
            self._tile_depth[idx] = depth
            self._parent_of[idx] = parent_idx
            _tx, _ty, tw, th = spec








            self._tile_outsize[idx] = (min(tw * 2, TILE_SIZE),
                                       min(th * 2, TILE_SIZE))
            pending.append((idx, spec))
            added += 1
        return added

    def _flush_withheld(self) -> None:










        withheld, self._withheld = self._withheld, {}
        for parent_idx, dets in withheld.items():
            if parent_idx in self._parents_with_child_results or not dets:
                continue
            try:
                self.tile_completed.emit(parent_idx, dets)
            except RuntimeError:
                return

    def _clear_rescan_marks(self) -> None:





        if not self._rescanning:
            return
        self._rescanning.clear()
        try:
            self.rescan_state.emit(-1, None, False)
        except RuntimeError:
            pass  # nosec B110
