"""Exemplar-only runs: the count-vs-map grouping decision and the client-side
re-merge of the retained raw fragments it needs.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog


class RawFragmentRemerge:
    """The exemplar-only count re-merge, one cursor over the raw fragments.

    Folding tens of thousands of retained fragments into a fresh merger is
    GEOS-heavy and used to run in one blocking call between the last tile and
    the review. Stepping it lets the finalize pump yield to the event loop the
    same way the redundancy sweep does. ``step`` advances its cursor before
    touching a fragment, so a bad one is skipped rather than repeated.
    """

    def __init__(self, frags: list, merger, tile_area: float,
                 hard_cov: float, max_cov: float, min_fill: float,
                 merge_separate: bool) -> None:
        self._frags = frags
        self._merger = merger
        self._tile_area = float(tile_area or 0.0)
        self._hard_cov = float(hard_cov)
        self._max_cov = float(max_cov)
        self._min_fill = float(min_fill)
        self._separate = bool(merge_separate)
        self._cursor = 0
        self.gated = 0
        self.total = len(frags)

    def step(self, count: int = 64) -> bool:
        """Fold up to ``count`` more fragments. True when every one is in."""
        from qgis.core import QgsGeometry

        from ...workers.auto_detection_worker import AutoDetectionWorker

        end = min(self._cursor + max(1, int(count)), self.total)
        while self._cursor < end:
            wkb, score = self._frags[self._cursor]
            self._cursor += 1
            geom = QgsGeometry()
            geom.fromWkb(wkb)
            if geom.isEmpty():
                continue
            if self._separate and self._tile_area > 0:
                cov = geom.area() / self._tile_area
                if cov > self._hard_cov:
                    self.gated += 1
                    continue
                if cov > self._max_cov and not AutoDetectionWorker._is_compact_shape(
                        geom, self._min_fill):
                    self.gated += 1
                    continue
            self._merger.add(geom, float(score))
        return self._cursor >= self.total

    def result(self) -> list:
        """The merged (fid, geom, score) set. Call once every fragment is in."""
        # Every fragment is folded in: the same single legal moment as the live
        # path (see _resolve_exemplar_finalize_ided).
        self._merger.restore_absorbed_partitions()
        return self._merger.result_scored_ided()


class AutoExemplarGroupingMixin:
    """Distinct-objects vs continuous-cover grouping for an exemplar-only run."""

    # ---- Exemplar-only count-vs-map auto decision + override ----------------

    def _resolve_exemplar_finalize_ided(self) -> list:
        """The (fid, geom, score) merged set finalize should build objects from.

        For a prompted run (or any run that is not exemplar-only) this is simply
        the live merger's result. For an exemplar-only run it applies the
        automatic count-vs-map decision from the run's own masks: the live merger
        already streamed MAP, so a MAP decision keeps it, while a SEPARATE
        decision re-merges the retained raw fragments client-side (gates + a
        fresh SEPARATE merger). The chosen mode is stored BEFORE the review
        opens, since downstream seam logic reads it."""
        if self._auto_merger is not None:
            # Every tile is in, which is the ONE legal moment to give back the
            # objects a coarse reading swallowed: mid-run the parts of a
            # complex have not all arrived, so an earlier call would judge an
            # incomplete set and keep the blob. A no-op unless the run's class
            # armed it (see IncrementalMerger.restore_absorbed_partitions).
            try:
                # A restore is the one thing that can move an object under a
                # fid AFTER the stitch thread has stopped, so the shapes that
                # thread built no longer describe what it touched. It touches
                # a handful of objects out of the whole run, so record THOSE
                # fids and leave the rest seedable (see
                # _seed_review_refine_cache).
                self._auto_merger.restore_absorbed_partitions()
                self._note_stitch_shapes_dirty(
                    getattr(self._auto_merger, "restored_fids", None))
                self._auto_stitch_shapes_stale = False
            except (AttributeError, RuntimeError):
                self._auto_stitch_shapes_stale = True
        live = (
            self._auto_merger.result_scored_ided()
            if self._auto_merger is not None else []
        )
        if not getattr(self, "_auto_is_exemplar_only", False):
            return live
        want_separate = self._decide_exemplar_merge_separate()
        self._auto_merge_mode_source = "signal"
        frags = getattr(self, "_auto_raw_fragments", None)
        if want_separate and frags:
            self._auto_merge_separate = True
            # A fresh merger mints its own fids, so nothing the stitch thread
            # shaped can be matched to this set.
            self._auto_stitch_shapes_stale = True
            return self._remerge_raw_fragments(True)
        if want_separate:
            # The read was count, and the fragments a count re-merge needs are
            # gone: the run passed the retain ceiling and the stitcher freed
            # them. Without this line the run comes back as one continuous
            # shape instead of N counted objects, and nothing says why.
            QgsMessageLog.logMessage(
                "Auto detection: count grouping was chosen but the retained "
                "fragments are unavailable (retain ceiling reached); grouping "
                "as continuous cover instead.",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
        # MAP (or SEPARATE wanted but fragments overflowed / empty): the live
        # merger is already MAP, so keep it and record MAP as the mode.
        self._auto_merge_separate = False
        return live

    def _decide_exemplar_merge_separate(self) -> bool:
        """True = keep SEPARATE (count), False = MAP (continuous cover), decided
        from the run's own masks.

        The signal is the area-weighted mean tile coverage of the fragments
        (sum(cov^2)/sum(cov), cov = fragment ground area / tile ground area,
        failure blobs above the hard cap excluded): the tile fraction a typical
        unit of detected ground belongs to. Continuous cover (many medium
        fragments tiling the zone) scores high; small countable objects (each a
        tiny fraction of a tile) score near zero, and excluding the failure blobs
        keeps a handful from faking cover. Compared against the server-tunable
        map_likeness_min_share; when no fragment was measured the counting-safe
        policy default stands."""
        from ...core import detection_policy
        cov_sum = float(getattr(self, "_auto_raw_cov_sum", 0.0) or 0.0)
        if cov_sum <= 0.0:
            return detection_policy.exemplar_only_merge_separate()
        cov_sq_sum = float(getattr(self, "_auto_raw_cov_sq_sum", 0.0) or 0.0)
        map_likeness = cov_sq_sum / cov_sum
        threshold = detection_policy.map_likeness_min_share()
        is_map = map_likeness >= threshold
        QgsMessageLog.logMessage(
            "Auto detection: exemplar-only map-likeness {:.3f} vs threshold "
            "{:.3f} -> {}".format(
                map_likeness, threshold,
                "continuous cover" if is_map else "distinct objects"),
            "AI Segmentation", level=Qgis.MessageLevel.Info,
        )
        return not is_map

    def _begin_exemplar_finalize_merge(self) -> tuple:
        """``(rows, remerge)`` for the finalize: exactly one of them is set.

        The cooperative twin of ``_resolve_exemplar_finalize_ided``. A prompted
        run, or an exemplar-only run the signal read as continuous cover, has
        its rows already and returns ``(rows, None)``. An exemplar-only run
        read as distinct objects returns ``(None, RawFragmentRemerge)``, which
        the finalize pump steps so the re-merge yields to the event loop
        instead of freezing the map for the length of the fold.
        """
        if self._auto_merger is not None:
            # Every tile is in, which is the ONE legal moment to give back the
            # objects a coarse reading swallowed (see
            # _resolve_exemplar_finalize_ided for why it may not run earlier).
            try:
                # Only the objects the restore moved lose their shape; see
                # _resolve_exemplar_finalize_ided.
                self._auto_merger.restore_absorbed_partitions()
                self._note_stitch_shapes_dirty(
                    getattr(self._auto_merger, "restored_fids", None))
                self._auto_stitch_shapes_stale = False
            except (AttributeError, RuntimeError):
                self._auto_stitch_shapes_stale = True
        live = (
            self._auto_merger.result_scored_ided()
            if self._auto_merger is not None else []
        )
        if not getattr(self, "_auto_is_exemplar_only", False):
            return live, None
        want_separate = self._decide_exemplar_merge_separate()
        self._auto_merge_mode_source = "signal"
        frags = getattr(self, "_auto_raw_fragments", None)
        if want_separate and frags:
            self._auto_merge_separate = True
            # A fresh merger mints its own fids, so nothing the stitch thread
            # shaped can be matched to this set.
            self._auto_stitch_shapes_stale = True
            return None, self._raw_fragment_remerge(True)
        if want_separate:
            # The read was count, and the fragments a count re-merge needs are
            # gone: the run passed the retain ceiling and the stitcher freed
            # them. Without this line the run comes back as one continuous
            # shape instead of N counted objects, and nothing says why.
            QgsMessageLog.logMessage(
                "Auto detection: count grouping was chosen but the retained "
                "fragments are unavailable (retain ceiling reached); grouping "
                "as continuous cover instead.",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
        self._auto_merge_separate = False
        return live, None

    def _log_raw_fragment_remerge(self, remerge, took_ms: int) -> None:
        """One line per re-merge: how many fragments, which grouping, what it
        cost. The only observability this pass has."""
        QgsMessageLog.logMessage(
            "Auto detection: re-merged {} raw fragment(s) as {} ({} gated) in "
            "{} ms".format(
                remerge.total,
                "distinct objects" if self._auto_merge_separate
                else "continuous cover",
                remerge.gated, int(took_ms)),
            "AI Segmentation", level=Qgis.MessageLevel.Info,
        )

    def _remerge_raw_fragments(self, merge_separate: bool) -> list:
        """Re-merge the retained exemplar raw fragments the given way, to the
        end, and return the (fid, geom, score) merged set.

        The blocking form, for a caller with no GUI to protect (the headless
        run). The interactive finalize steps the same pass instead, through
        ``_begin_exemplar_finalize_merge``."""
        import time as _t

        remerge = self._raw_fragment_remerge(merge_separate)
        t0 = _t.monotonic()
        while not remerge.step(512):
            pass
        out = remerge.result()
        self._log_raw_fragment_remerge(remerge, (_t.monotonic() - t0) * 1000)
        return out

    def _raw_fragment_remerge(self, merge_separate: bool) -> RawFragmentRemerge:
        """The stepper that re-merges the retained exemplar raw fragments.

        SEPARATE applies the worker's coverage gates client-side (hard cap drop
        + a compactness check above the soft cap), then folds survivors into a
        fresh IncrementalMerger built with the exact kwargs the run merger
        uses. The cover sweep runs downstream, in finalize."""
        from ...core.polygon_exporter import IncrementalMerger
        from ...workers.auto_detection_worker import (
            _COMPACT_MIN_FILL,
            _HARD_TILE_COVERAGE,
            _MAX_TILE_COVERAGE,
        )

        frags = getattr(self, "_auto_raw_fragments", None) or []
        # _auto_seam_min_dim reads _auto_merge_separate, so set it first.
        self._auto_merge_separate = merge_separate
        from ...core.detection_policy import merge_scalar_kwargs, merge_scalars
        # The run's own resolved scalars when there are any, else a fresh read,
        # so a re-merge outside a live run still follows the server policy
        # instead of the constructor defaults. Picked off the merger's own
        # signature, exactly like the run merger, so the two cannot diverge.
        ms = self._auto_merge_scalars or merge_scalars()
        merger = IncrementalMerger(
            seam_min_dim=self._auto_seam_min_dim(),
            select_duplicates=merge_separate,
            gsd=self._auto_gsd,
            # The run's own restore decision, resolved once at run start. An
            # exemplar-only run reaches SEPARATE through HERE, not through the
            # live merger, so leaving it out would silently drop the restore on
            # exactly the path that re-groups a finished run.
            restore_partitions=(
                merge_separate and bool(getattr(self, "_auto_restore_partitions", False))),
            **merge_scalar_kwargs(IncrementalMerger, ms),
        )
        tile_area = float(getattr(self, "_auto_tile_ground_area", 0.0) or 0.0)
        # Same server-overridable coverage gates the run worker resolved. The
        # compactness floor is one of them, so it is passed in rather than left
        # to the helper's constant default.
        from ...core.detection_policy import (
            compact_min_fill,
            hard_tile_coverage,
            max_tile_coverage,
        )
        hard_cov = hard_tile_coverage(_HARD_TILE_COVERAGE)
        max_cov = max_tile_coverage(_MAX_TILE_COVERAGE)
        min_fill = compact_min_fill(_COMPACT_MIN_FILL)
        return RawFragmentRemerge(
            list(frags), merger, tile_area, hard_cov, max_cov, min_fill,
            merge_separate)

    def _clear_auto_raw_fragments(self) -> None:
        """Drop the raw-fragment retention + count-vs-map counters (a review
        end, a new run, a teardown). Cheap and idempotent."""
        self._auto_is_exemplar_only = False
        self._auto_retain_raw = False
        self._auto_collect_raw = False
        self._auto_raw_fragments = None
        self._auto_raw_n_total = 0
        self._auto_raw_cov_sum = 0.0
        self._auto_raw_cov_sq_sum = 0.0
        self._auto_tile_ground_area = 0.0
