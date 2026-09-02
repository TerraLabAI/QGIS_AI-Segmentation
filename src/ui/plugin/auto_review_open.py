"""Opening the post-run review: the headless export branch, the review state
the dock is seeded from, and the default export archived at that moment.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsGeometry,
    QgsMessageLog,
)


class AutoReviewOpenMixin:
    """Hand a finished run to the review, or export it headless."""

    def _run_export_crs(self, source_layer) -> QgsCoordinateReferenceSystem:
        """The CRS the finished run's geometry is in.

        The run's captured authid whenever it names a valid system. A custom
        projection comes back with an EMPTY authid, and EPSG:4326 is not a safe
        stand-in there: the coordinates are metres, so the saved layer lands far
        from the imagery with nothing said. The source layer's own CRS answers
        that case, and the last-resort substitute is logged rather than taken in
        silence.
        """
        authid = str(getattr(self, "_auto_crs_authid", None) or "").strip()
        if authid:
            crs = QgsCoordinateReferenceSystem(authid)
            if crs.isValid():
                return crs
        layer_crs = None
        try:
            if source_layer is not None:
                layer_crs = source_layer.crs()
        except (RuntimeError, AttributeError):
            layer_crs = None
        if layer_crs is not None and layer_crs.isValid():
            QgsMessageLog.logMessage(
                "Auto review: the run carries no usable CRS id; the export "
                "takes the source layer's CRS.",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return layer_crs
        QgsMessageLog.logMessage(
            "Auto review: no CRS could be resolved for the run; the export "
            "falls back to EPSG:4326 and may be misplaced.",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return QgsCoordinateReferenceSystem("EPSG:4326")

    def _complete_auto_finalize(self, visible: list, tiles_succeeded: int,
                                scores: list | None = None,
                                ids: list | None = None) -> None:
        """Finish a run: ``visible`` is the current VISIBLE object set (already
        whole-object filtered + shape-refined). ``scores`` / ``ids`` are the
        parallel per-object score and canonical-identity lists (same order as
        ``visible``) that feed the review heatmap and the stable Random colours.
        Headless exports it straight to a layer; interactive enters the post-run
        review."""
        # Determine source layer name.
        source_layer = self._get_active_raster_layer()
        source_layer_name = ""
        try:
            if source_layer is not None:
                source_layer_name = source_layer.name()
        except (RuntimeError, AttributeError):
            pass
        crs = self._run_export_crs(source_layer)

        # Get prompt text for output filename.
        prompt_text = ""
        try:
            if self.dock_widget:
                prompt_text = self.dock_widget.auto_prompt_input.text().strip()
        except (RuntimeError, AttributeError):
            pass

        if self._auto_headless_run:
            # MCP/headless: no human to review, so export the default-filtered
            # visible set (keeps the stable API contract: a saved layer name).
            if not visible:
                # Nothing kept: drop the live layer so a zero-result agent run
                # never leaves the raw streamed masks on the canvas (there is no
                # review teardown to remove it, unlike the interactive path).
                self._remove_auto_selection_layer()
                self._record_auto_zero_result(tiles_succeeded)
                return
            exported_layer_name = self._export_auto_detections(
                visible, crs, source_layer_name, prompt_text, scores=scores)
            self._remove_auto_selection_layer()
            result = {
                "status": "completed",
                "instances": len(visible),
                "tiles_processed": tiles_succeeded,
                "layer_name": exported_layer_name,
            }
            prior = self._last_auto_result
            if isinstance(prior, dict) and prior.get("status") == "credits_exhausted":
                # The exhausted terminal recorded its result before finalize:
                # keep BOTH facts (the saved layer and the quota) so the
                # headless caller neither orphans paid work nor misses the
                # exhaustion signal.
                result["status"] = "credits_exhausted"
                result["credits_remaining"] = prior.get("credits_remaining", 0)
            self._last_auto_result = result
            QgsMessageLog.logMessage(
                f"Auto detection: exported {len(visible)} polygon(s)",
                "AI Segmentation", level=Qgis.MessageLevel.Info,
            )
            return

        # Interactive: enter the review even if the fresh confidence cutoff hides
        # every object (as long as the run DID find objects), so the confidence
        # slider can bring them back. Only a truly empty run is a zero result.
        if not self._auto_objects:
            self._record_auto_zero_result(tiles_succeeded)
            return

        pixel_size = self._auto_refine_pixel_size()
        self._auto_review = {
            "geoms": visible,
            "scores": scores or [],
            "ids": ids or [],
            "crs": crs,
            "source_layer_name": source_layer_name,
            "prompt": prompt_text,
            "pixel_size": pixel_size,
            # Provenance stamp for the incremental review push: the visible set
            # is the cache-normalized refine output under this shape key.
            "stamp": ("acc", (self._auto_reslice_cache or {}).get("key")),
        }
        # Fresh correction round for the linear review ladder (journal, sets,
        # queue, dials back on Keep).
        try:
            self._reset_auto_corrections()
        except (RuntimeError, AttributeError):
            pass
        # Record result so MCP get_status stays consistent.
        self._last_auto_result = {
            "status": "completed",
            "instances": len(visible),
            "tiles_processed": tiles_succeeded,
            "layer_name": None,
        }
        # A tile still at the cloud model's per-inference mask ceiling AFTER the
        # re-split ladder means the scene was denser than the run could resolve.
        # Internal log only (this runs once per run); the review UI stays quiet.
        if getattr(self, "_auto_dense_tiles", 0):
            QgsMessageLog.logMessage(
                f"Auto detection: {self._auto_dense_tiles} tile(s) still at the max masks per "
                "inference after re-split; denser tiling (higher Detail) may "
                "catch more objects.",
                "AI Segmentation", level=Qgis.MessageLevel.Info,
            )
        # Seed the refine controls (confidence + size + shape) with this run's
        # smart preset so a value left over from a previous review cannot
        # wrongly filter this fresh result. The visible set was already computed
        # with the same preset (_fresh_review_params), so the widgets and the
        # geometry agree.
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_review_active(
                    True, count=len(visible), preset=self._auto_review_preset())
                # Shared borders: offered on a land-cover run only, and only
                # when the result is small enough for one pass. Off by
                # default either way.
                self.dock_widget.set_boundary_snap_offered(
                    self._boundary_snap_offered())
                # Drop the blue zone fill so the detections are not washed out by
                # the overlay during review (the outline stays for context).
                self._set_zone_band_fill_visible(False)
            except (RuntimeError, AttributeError):
                pass
        # The review is open on the AI fix method until the user says otherwise,
        # so start the local model here. The load then runs while the user reads
        # the Keep step, instead of under the first pick on Correct. Silent, and
        # it stops itself on a machine with no local environment.
        self._warm_local_ai_for_review()
        # Swap the live-run blue outline for the review's default Random colours
        # (one colour per object, seeded fresh for every NEW review; the combo
        # follows signal-free so control and renderer agree).
        self._seed_review_display_mode()
        if self._auto_selection_layer is not None:
            self._apply_review_display_mode(self._auto_selection_layer)
        self._refresh_auto_review_preview()
        self._review_conf_moves = 0  # fresh confidence-move counter for this review
        # Telemetry: the run's terminal event (completed only; cancel/exhaust
        # already emitted theirs) plus the review-opened funnel step.
        try:
            from ...core import telemetry_run_events
            ctx = self._auto_run_ctx or {}
            total = ctx.get("total", tiles_succeeded)
            instances_found = len(self._auto_objects)
            visible_n = len(visible)
            start_pct = int(round((self._auto_confidence or 0.0) * 100))
            if self._auto_tel_stop_reason in (None, "completed"):
                from .auto_client_profile import client_profile_props
                blob_armed, blob_dropped, tile_m = self._auto_blob_guard_stats()
                telemetry_run_events.track_auto_detect_completed(
                    run_id=self._auto_run_id or "",
                    duration_ms=self._auto_duration_ms(),
                    tiles_done=tiles_succeeded,
                    tiles_failed=max(0, total - tiles_succeeded),
                    instances_found=instances_found,
                    instances_visible_at_default=visible_n,
                    zero_at_default=visible_n == 0,
                    stop_reason="completed",
                    warming_ms=self._auto_warming_wait_ms(),
                    merge_mode_final="separate" if self._auto_merge_separate else "map",
                    blob_armed=blob_armed,
                    blob_dropped=blob_dropped,
                    tile_ground_m=tile_m,
                    client_profile=client_profile_props(self),
                )
            telemetry_run_events.track_review_opened(
                run_id=self._auto_run_id or "",
                instances_found=instances_found,
                visible_at_start=visible_n,
                start_confidence=start_pct,
                auto_lowered=start_pct < int(round(self._effective_confidence_default() * 100)),
            )
        except Exception:
            pass  # nosec B110
        # Fresh per-review engagement flags for the abandonment telemetry
        # (one review_abandoned max per review, refined/confidence split; the
        # conf-move counter is reset above with the other fresh-review state).
        self._review_tel_refined = False
        self._review_tel_conf_changed = False
        self._review_abandon_tracked = False
        self._offer_closed_canopy_advice(tiles_succeeded)
        self._remember_auto_run_pace(tiles_succeeded)
        QgsMessageLog.logMessage(
            f"Auto detection: {len(visible)} object(s) ready for review",
            "AI Segmentation", level=Qgis.MessageLevel.Info,
        )

    def _remember_auto_run_pace(self, tiles_succeeded: int) -> None:
        """Feed this run's wall clock, Detect click to review open, to the
        quote's memory, and requote the dock from it right away.

        Never raises and never logs a figure: a missing start stamp (a run
        restored from disk, a headless run) just teaches nothing.
        """
        started = getattr(self, "_auto_run_started_mono", None)
        self._auto_run_started_mono = None
        if started is None:
            return
        try:
            import time as _time

            from ...core.run_pace_memory import own_machine_pace, remember_run
            remember_run(int(tiles_succeeded or 0), _time.monotonic() - started)
            pace = own_machine_pace()
            if pace is not None and self.dock_widget is not None:
                self.dock_widget.set_auto_own_pace(pace)
        except Exception:  # noqa: BLE001 -- the memory is a convenience
            pass  # nosec B110

    def _offer_closed_canopy_advice(self, tiles_succeeded: int) -> None:
        """Show the closed-forest card when this run has its signature: a
        tree word, separate objects, few raw masks per tile and whole tiles
        handed back as one block. Every other run hides the card."""
        if not self.dock_widget:
            return
        try:
            from ...core.detection_policy import closed_canopy_signature, prompt_suggests_canopy
            prompt = str((self._auto_run_ctx or {}).get("prompt") or "")
            # A localized word ("arbre") is judged by its English token, the
            # way every other prompt lookup works.
            try:
                prompt = str(self._resolve_object_token(prompt) or prompt)
            except (RuntimeError, AttributeError, TypeError):
                pass  # nosec B110 - the typed word stands
            span_dropped = int(getattr(self, "_auto_blob_split", (0, 0, 0))[1])
            on = (
                bool(getattr(self, "_auto_merge_separate", True))
                and prompt_suggests_canopy(prompt)
                and closed_canopy_signature(
                    int(getattr(self, "_auto_raw_count", 0) or 0),
                    int(tiles_succeeded or 0), span_dropped)
            )
            if on:
                QgsMessageLog.logMessage(
                    f"Auto detection: closed-canopy signature ({span_dropped} "
                    f"spanning mask(s) dropped over {tiles_succeeded} tile(s)); "
                    "review shows the forest advice",
                    "AI Segmentation", level=Qgis.MessageLevel.Info,
                )
            self.dock_widget.set_closed_canopy_advice(on)
        except Exception:  # noqa: BLE001 - advice never blocks a review
            pass  # nosec B110

    def _archive_auto_default_export(self) -> None:
        """Archive the run's clean export the moment the review opens, using the
        auto-default confidence + refine settings the review starts with (the
        set the user sees before touching anything). This makes every run's
        clean geometry reach the service even when the user closes without
        Finish. Finish still uploads the reviewed set afterwards under the same
        run_id (latest wins), so a hand-edited result overwrites this default.

        Reuses the SAME hidden background QgsTask the Finish upload uses (never
        blocks the GUI) and keeps its exact format (precision + native CRS in
        run_export_upload._feature_json). Fires at most once per run and is
        best-effort end to end: any failure is swallowed and never touches the
        review or the local export."""
        review = getattr(self, "_auto_review", None)
        if not review or getattr(self, "_auto_headless_run", False):
            return
        run_id = getattr(self, "_auto_run_id", None)
        if not run_id:
            return
        # One default archive per run: never re-fire for a run already archived
        # (guards against any future extra call into this completion path).
        if getattr(self, "_auto_default_export_run_id", None) == run_id:
            return
        self._auto_default_export_run_id = run_id
        try:
            geoms = review.get("geoms") or []
            scores = review.get("scores")
            if scores is not None and len(scores) != len(geoms):
                scores = None
            refined, refined_scores = [], []
            for index, g in enumerate(geoms):
                if g is None or g.isEmpty():
                    continue
                refined.append(QgsGeometry(g))
                refined_scores.append(scores[index] if scores else None)
            if not refined:
                return
            from .run_export_upload import queue_run_export_upload
            # This leg fires the moment review opens, before the user could
            # have touched the confidence slider, so the gate that actually
            # filtered `refined` is the review's START confidence, not the
            # "finish" default this call used to take silently (every
            # abandoned run was then recorded as a Finish).
            try:
                default_confidence = float(self._review_start_confidence())
            except Exception:  # noqa: BLE001
                default_confidence = None
            queue_run_export_upload(
                self, review, refined, refined_scores,
                export_path="review_open", confidence_applied=default_confidence)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
