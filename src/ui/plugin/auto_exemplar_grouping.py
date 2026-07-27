"""Exemplar-only runs: the count-vs-map grouping decision and the client-side
re-merge of the retained raw fragments it needs.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog


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
                # Counted, because it is the one thing that can move an object
                # under a fid AFTER the stitch thread has stopped: any restore
                # invalidates the shapes that thread built, so the finalize must
                # not seed its cache from them (see _seed_review_refine_cache).
                self._auto_stitch_shapes_stale = bool(
                    self._auto_merger.restore_absorbed_partitions())
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

    def _remerge_raw_fragments(self, merge_separate: bool) -> list:
        """Re-merge the retained exemplar raw fragments the given way and return
        the (fid, geom, score) merged set. SEPARATE first applies the worker's
        coverage gates client-side (hard cap drop + a compactness check above the
        soft cap), then folds survivors into a fresh IncrementalMerger built with
        the exact kwargs the run merger uses. Bounded synchronous work (the
        SEPARATE branch only runs on runs the signal judged NOT map-like, so the
        fragment count is modest); logged for observability. The cover sweep runs
        downstream, in finalize."""
        import time as _t

        from qgis.core import QgsGeometry

        from ...core.polygon_exporter import IncrementalMerger
        from ...workers.auto_detection_worker import (
            _COMPACT_MIN_FILL,
            _HARD_TILE_COVERAGE,
            _MAX_TILE_COVERAGE,
            AutoDetectionWorker,
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
        t0 = _t.monotonic()
        gated = 0
        for wkb, score in frags:
            geom = QgsGeometry()
            geom.fromWkb(wkb)
            if geom.isEmpty():
                continue
            if merge_separate and tile_area > 0:
                cov = geom.area() / tile_area
                if cov > hard_cov:
                    gated += 1
                    continue
                if cov > max_cov and not AutoDetectionWorker._is_compact_shape(
                        geom, min_fill):
                    gated += 1
                    continue
            merger.add(geom, float(score))
        # Every fragment is folded in: same single legal moment as the live
        # path (see _resolve_exemplar_finalize_ided).
        merger.restore_absorbed_partitions()
        out = merger.result_scored_ided()
        QgsMessageLog.logMessage(
            "Auto detection: re-merged {} raw fragment(s) as {} ({} gated) in "
            "{} ms".format(
                len(frags), "distinct objects" if merge_separate else "continuous cover",
                gated, int((_t.monotonic() - t0) * 1000)),
            "AI Segmentation", level=Qgis.MessageLevel.Info,
        )
        return out

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
