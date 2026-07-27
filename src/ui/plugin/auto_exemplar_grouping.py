






from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog


class RawFragmentRemerge:









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



        self._merger.restore_absorbed_partitions()
        return self._merger.result_scored_ided()


class AutoExemplarGroupingMixin:




    def _resolve_exemplar_finalize_ided(self) -> list:









        if self._auto_merger is not None:





            try:






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


            self._auto_stitch_shapes_stale = True
            return self._remerge_raw_fragments(True)
        if want_separate:




            QgsMessageLog.logMessage(
                "Auto detection: count grouping was chosen but the retained "
                "fragments are unavailable (retain ceiling reached); grouping "
                "as continuous cover instead.",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )


        self._auto_merge_separate = False
        return live

    def _decide_exemplar_merge_separate(self) -> bool:












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









        if self._auto_merger is not None:



            try:


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


            self._auto_stitch_shapes_stale = True
            return None, self._raw_fragment_remerge(True)
        if want_separate:




            QgsMessageLog.logMessage(
                "Auto detection: count grouping was chosen but the retained "
                "fragments are unavailable (retain ceiling reached); grouping "
                "as continuous cover instead.",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
        self._auto_merge_separate = False
        return live, None

    def _log_raw_fragment_remerge(self, remerge, took_ms: int) -> None:


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






        import time as _t

        remerge = self._raw_fragment_remerge(merge_separate)
        t0 = _t.monotonic()
        while not remerge.step(512):
            pass
        out = remerge.result()
        self._log_raw_fragment_remerge(remerge, (_t.monotonic() - t0) * 1000)
        return out

    def _raw_fragment_remerge(self, merge_separate: bool) -> RawFragmentRemerge:






        from ...core.polygon_exporter import IncrementalMerger
        from ...workers.auto_detection_worker import (
            _COMPACT_MIN_FILL,
            _HARD_TILE_COVERAGE,
            _MAX_TILE_COVERAGE,
        )

        frags = getattr(self, "_auto_raw_fragments", None) or []

        self._auto_merge_separate = merge_separate
        from ...core.detection_policy import merge_scalar_kwargs, merge_scalars




        ms = self._auto_merge_scalars or merge_scalars()
        merger = IncrementalMerger(
            seam_min_dim=self._auto_seam_min_dim(),
            select_duplicates=merge_separate,
            gsd=self._auto_gsd,




            restore_partitions=(
                merge_separate and bool(getattr(self, "_auto_restore_partitions", False))),
            **merge_scalar_kwargs(IncrementalMerger, ms),
        )
        tile_area = float(getattr(self, "_auto_tile_ground_area", 0.0) or 0.0)



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


        self._auto_is_exemplar_only = False
        self._auto_retain_raw = False
        self._auto_collect_raw = False
        self._auto_raw_fragments = None
        self._auto_raw_n_total = 0
        self._auto_raw_cov_sum = 0.0
        self._auto_raw_cov_sq_sum = 0.0
        self._auto_tile_ground_area = 0.0
