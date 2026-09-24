













from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsFeature,
    QgsField,
    QgsMessageLog,
    QgsProject,
    QgsVectorLayer,
)

from ...core.i18n import tr
from ...workers.live_stitch_thread import (
    DELTA_REMOVE,
    STITCH_JOIN_TIMEOUT_MS,
    LiveStitchThread,
)
from .shared import (
    _FIELD_TYPE_DOUBLE,
    _FIELD_TYPE_INT,
    _FIELD_TYPE_STRING,
    _add_features_with_ids,
    _apply_fast_render,
    _notify_provider_write,
    auto_live_repaint_settings,
    park_orphaned_worker,
)




_STITCH_WAIT_SLICE_MS = 25




_STITCH_DRAIN_BUDGET_S = 30.0


def _stitch_drain_budget_s() -> float:



    from ...core.server_dials import dial_in_range
    return dial_in_range(
        "tuning.auto.stitch_drain_budget_s", _STITCH_DRAIN_BUDGET_S, 5.0, 300.0)


def _stitch_wait_slice_ms() -> int:


    from ...core.server_dials import dial_in_range
    return dial_in_range(
        "tuning.auto.stitch_wait_slice_ms", _STITCH_WAIT_SLICE_MS, 5, 250)


def _stitch_join_timeout_ms() -> int:



    from ...core.server_dials import dial_in_range
    return dial_in_range(
        "tuning.auto.stitch_join_timeout_ms", STITCH_JOIN_TIMEOUT_MS, 2000, 30000)


def _diff_live_fid_map(old_map: dict, current: list):






















    adds = []
    geom_changes = {}
    attr_changes = {}
    kept_map = {}
    seen = set()
    for fid, stamp, is_full, score in current:
        seen.add(fid)
        rec = old_map.get(fid)
        if rec is None:
            adds.append(fid)
            continue
        prov_fid, old_stamp, old_is_full, old_score = rec
        if old_stamp != stamp or old_is_full != is_full:
            geom_changes[prov_fid] = fid
        if old_score != score:
            attr_changes[prov_fid] = fid
        kept_map[fid] = (prov_fid, stamp, is_full, score)
    deletes = [rec[0] for fid, rec in old_map.items() if fid not in seen]
    return adds, geom_changes, attr_changes, deletes, kept_map


def _latest_delta_per_object(deltas: list) -> tuple[list, dict]:







    order: list = []
    latest: dict = {}
    for kind, fid, geom, score in deltas:
        if fid not in latest:
            order.append(fid)
        latest[fid] = (kind, geom, score)
    return order, latest


class AutoResultsMixin:


    def _create_auto_selection_layer(self, source_layer) -> QgsVectorLayer | None:

        try:





            crs_authid = self._auto_crs_authid or source_layer.crs().authid()
            layer = QgsVectorLayer(
                f"MultiPolygon?crs={crs_authid}",
                tr("Auto detection (live)"),
                "memory",
            )
            if not layer.isValid():
                return None
            pr = layer.dataProvider()
            pr.addAttributes([
                QgsField("label", _FIELD_TYPE_STRING),
                QgsField("score", _FIELD_TYPE_DOUBLE),






                QgsField("det_id", _FIELD_TYPE_INT),
            ])
            layer.updateFields()







            self._apply_review_display_mode(layer)


            _apply_fast_render(layer)




            from ...core.output_store import drop_from_snapping, mark_temp_layer
            mark_temp_layer(layer)
            QgsProject.instance().addMapLayer(layer, False)


            drop_from_snapping(layer)
            root = QgsProject.instance().layerTreeRoot()
            root.insertLayer(0, layer)


            self._review_fid_map = {}
            return layer
        except (RuntimeError, AttributeError):
            return None



    def _on_auto_rescan_state(self, tile_idx: int, bbox, active: bool) -> None:









        rects = getattr(self, "_auto_rescan_rects", None)
        if rects is None:
            rects = self._auto_rescan_rects = {}
        if tile_idx < 0:
            rects.clear()
        elif not active:
            rects.pop(tile_idx, None)
        elif bbox:
            try:
                rects[tile_idx] = (float(bbox[0]), float(bbox[1]),
                                   float(bbox[2]), float(bbox[3]))
            except (TypeError, ValueError, IndexError):
                return
        self._repaint_auto_rescan_band()

    def _auto_rescan_band_allowed(self) -> bool:





        from qgis.PyQt.QtCore import QSettings
        try:
            return bool(QSettings().value(
                "TerraLab/auto_debug_tiles", False, type=bool))
        except (RuntimeError, TypeError, ValueError):
            return False

    def _repaint_auto_rescan_band(self) -> None:





        from qgis.core import QgsGeometry, QgsRectangle
        from qgis.gui import QgsRubberBand

        from ...core.qt_compat import PolygonGeometry
        from ..canvas_palette import RESCAN_FILL, RESCAN_STROKE

        rects = getattr(self, "_auto_rescan_rects", None) or {}
        band = getattr(self, "_auto_rescan_band", None)
        layer = self._auto_selection_layer
        if not rects or layer is None or not self._auto_rescan_band_allowed():
            if band is not None:
                self._safe_remove_rubber_band(band)
                self._auto_rescan_band = None
            return
        try:
            if band is None:
                band = QgsRubberBand(self.iface.mapCanvas(), PolygonGeometry)
                band.setFillColor(RESCAN_FILL)
                band.setStrokeColor(RESCAN_STROKE)
                band.setWidth(1)
                self._auto_rescan_band = band
            parts = [QgsGeometry.fromRect(QgsRectangle(*r))
                     for r in rects.values()]



            band.setToGeometry(QgsGeometry.collectGeometry(parts), layer.crs())
        except (RuntimeError, AttributeError, TypeError):
            self._clear_auto_rescan_band()

    def _clear_auto_rescan_band(self) -> None:

        self._auto_rescan_rects = {}
        band = getattr(self, "_auto_rescan_band", None)
        self._auto_rescan_band = None
        if band is not None:
            self._safe_remove_rubber_band(band)

    def _remove_auto_selection_layer(self) -> None:










        try:
            self._abort_qgis_edit_bridge_if_active()
        except (RuntimeError, AttributeError):
            pass
        try:
            self._disarm_shape_tool()
        except (RuntimeError, AttributeError):
            pass







        try:
            self._disconnect_live_repaint_pacer()
            self._clear_auto_rescan_band()
        except Exception:  # nosec B110
            pass
        layer = self._auto_selection_layer
        self._auto_selection_layer = None
        self._review_fid_map = {}
        if layer is None:
            return
        try:
            lid = layer.id()
            QgsProject.instance().removeMapLayer(lid)
        except (RuntimeError, AttributeError):
            pass



    def _start_auto_stitcher(self, geo_transform: dict | None = None) -> None:











        self._abort_auto_stitcher()
        if self._auto_merger is None:
            return
        params = self._fresh_review_params()
        factor = self._auto_run_metres_per_unit(geo_transform)
        hard_cov = 0.0
        if self._auto_is_exemplar_only:
            from ...core.detection_policy import hard_tile_coverage
            from ...workers.auto_detection_worker import _HARD_TILE_COVERAGE
            hard_cov = hard_tile_coverage(_HARD_TILE_COVERAGE)



        self._auto_stitch_shapes = None
        self._auto_stitch_shape_px = 0.0
        self._auto_stitch_shape_mpu = 0.0
        self._auto_stitch_shapes_stale = True
        self._auto_stitch_dirty_fids = set()
        stitcher = LiveStitchThread(
            self._auto_merger,
            params=params,
            pixel_size=self._auto_refine_pixel_size(),
            metres_per_unit=factor,
            crs_authid=self._auto_crs_authid or "EPSG:4326",
            unit_aspect=self._auto_run_unit_aspect(geo_transform),
            retain_fragments=bool(getattr(self, "_auto_retain_raw", False)),
            retain_coverage=bool(self._auto_is_exemplar_only),
            tile_ground_area=float(getattr(self, "_auto_tile_ground_area", 0.0)),
            hard_coverage=hard_cov,
        )
        from qgis.PyQt.QtCore import Qt
        stitcher.batch_ready.connect(
            self._on_auto_stitch_batch, Qt.ConnectionType.QueuedConnection)
        self._arm_server_finalize_record(stitcher)
        self._auto_stitcher = stitcher
        stitcher.start()

    def _auto_run_centre_xy(self, geo_transform: dict | None):



        bbox = (geo_transform or {}).get("bbox")
        if bbox and len(bbox) == 4:
            return ((float(bbox[0]) + float(bbox[2])) / 2.0,
                    (float(bbox[1]) + float(bbox[3])) / 2.0)
        if self._auto_clip_polygon is not None:
            try:
                point = self._auto_clip_polygon.boundingBox().center()
                return (point.x(), point.y())
            except (RuntimeError, AttributeError):
                return None
        return None

    def _auto_run_metres_per_unit(self, geo_transform: dict | None) -> float:





        centre = self._auto_run_centre_xy(geo_transform)
        if centre is None:
            return 1.0
        try:
            return self._auto_crs_metres_per_unit(centre[0], centre[1])
        except (RuntimeError, AttributeError, TypeError):
            return 1.0

    def _auto_run_unit_aspect(self, geo_transform: dict | None) -> float:




        centre = self._auto_run_centre_xy(geo_transform)
        if centre is None:
            return 1.0
        try:
            return self._auto_crs_unit_aspect(centre[0], centre[1])
        except (RuntimeError, AttributeError, TypeError):
            return 1.0

    def _finish_auto_stitcher(self, timeout_ms: int = STITCH_JOIN_TIMEOUT_MS) -> bool:






        stitcher = self._auto_stitcher
        if stitcher is None:
            return True
        try:
            stitcher.finish()
            if not stitcher.join_run(timeout_ms):
                return False
        except RuntimeError:


            self._auto_stitcher = None
            return True
        self._auto_stitcher = None
        self._harvest_auto_stitch_counters(stitcher)
        return True

    def _abort_auto_stitch_queue(self) -> None:







        stitcher = self._auto_stitcher
        if stitcher is None:
            return
        try:
            stitcher.abort()
        except RuntimeError:
            pass

    def _abort_auto_stitcher(self) -> None:







        stitcher = self._auto_stitcher
        self._auto_stitcher = None
        if stitcher is None:
            return


        try:
            stitcher.batch_ready.disconnect()
        except (RuntimeError, TypeError):
            pass
        try:
            stitcher.abort()
            if not stitcher.join_run(_stitch_join_timeout_ms()):
                park_orphaned_worker(stitcher)
        except RuntimeError:
            pass

    def _harvest_auto_stitch_counters(self, stitcher) -> None:





        if stitcher.raw_count > self._auto_raw_count:
            self._auto_raw_count = stitcher.raw_count
        if self._auto_is_exemplar_only:
            self._auto_raw_n_total = stitcher.raw_n_total
            self._auto_raw_cov_sum = stitcher.raw_coverage_sum
            self._auto_raw_cov_sq_sum = stitcher.raw_coverage_sq_sum
        if getattr(self, "_auto_retain_raw", False):
            self._auto_raw_fragments = stitcher.raw_fragments




        self._take_server_finalize_record(stitcher)
        self._auto_stitch_shapes = stitcher.shaped_geoms
        self._auto_stitch_shape_px = stitcher.shape_pixel_size
        self._auto_stitch_shape_mpu = stitcher.shape_metres_per_unit



    def _on_auto_tile_completed(self, tile_idx: int, tagged_detections: list) -> None:










        self._note_auto_progress()
        stitcher = self._auto_stitcher
        if stitcher is None or not tagged_detections:
            return
        stitcher.submit(tagged_detections, self._auto_refine_pixel_size(), tile_idx)

    def _on_auto_stitch_batch(self) -> None:


        self._note_auto_progress()
        self._push_auto_assemble_progress()
        self._request_auto_live_repaint()

    def _push_auto_assemble_progress(self) -> None:


        dock = self.dock_widget
        if dock is None:
            return
        folded, queued = self._auto_stitch_backlog()




        converting = int(getattr(
            getattr(self, "_auto_worker", None), "tiles_awaiting_conversion", 0) or 0)
        try:
            dock.set_auto_assemble_tiles(folded, folded + queued + converting)
        except (RuntimeError, AttributeError):
            pass

    def _auto_stitch_backlog(self) -> tuple[int, int]:


        stitcher = self._auto_stitcher
        if stitcher is None:
            return 0, 0
        try:
            return int(stitcher.tiles_folded), int(stitcher.pending())
        except RuntimeError:
            return 0, 0

    def _drain_auto_tiles_now(self) -> None:









        import time as _t




        budget = _stitch_drain_budget_s()
        wait_slice_ms = _stitch_wait_slice_ms()
        folded_seen, _queued = self._auto_stitch_backlog()
        deadline = _t.monotonic() + budget
        hard_deadline = None
        while not self._finish_auto_stitcher(timeout_ms=wait_slice_ms):
            now = _t.monotonic()
            folded, queued = self._auto_stitch_backlog()
            if folded != folded_seen:
                folded_seen = folded
                deadline = now + budget
            if hard_deadline is None:
                if now < deadline:
                    continue
                QgsMessageLog.logMessage(
                    "Auto detection: live stitcher folded no tile for "
                    f"{int(budget)}s with {queued} still queued; "
                    "finalizing what it folded",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
                self._abort_auto_stitch_queue()
                hard_deadline = now + _stitch_join_timeout_ms() / 1000.0
            elif now >= hard_deadline:



                QgsMessageLog.logMessage(
                    "Auto detection: live stitcher would not stop; keeping "
                    "results as they stand", "AI Segmentation",
                    level=Qgis.MessageLevel.Warning)
                self._abort_auto_stitcher()
                return

    def _apply_auto_live_deltas(self) -> None:







        layer = self._auto_selection_layer
        stitcher = self._auto_stitcher
        if layer is None or stitcher is None:
            return
        deltas = stitcher.take_deltas()
        if not deltas:
            return
        import time as _time
        _draw_t0 = _time.monotonic()
        try:
            if not layer.isValid():
                return
            pr = layer.dataProvider()
            fields = layer.fields()
            score_idx = fields.indexOf("score")
            fid_map = self._auto_live_fid_map
            order, latest = _latest_delta_per_object(deltas)

            deletes: list = []
            adds: list = []
            geom_changes: dict = {}
            attr_changes: dict = {}
            for fid in order:
                kind, geom, score = latest[fid]
                if kind == DELTA_REMOVE:
                    rec = fid_map.pop(fid, None)
                    if rec is not None:
                        deletes.append(rec[0])
                    continue
                rec = fid_map.get(fid)
                if rec is None:
                    feat = QgsFeature(fields)
                    feat.setGeometry(geom)


                    feat.setAttributes(["", score, fid])
                    adds.append((fid, feat))
                    continue
                prov_fid, old_score = rec
                geom_changes[prov_fid] = geom
                if old_score != score:
                    attr_changes[prov_fid] = score
                fid_map[fid] = (prov_fid, score)

            changed = False
            if deletes:
                pr.deleteFeatures(deletes)
                changed = True
            if adds:




                ok, added = _add_features_with_ids(
                    pr, [feat for _, feat in adds])
                if ok and len(added) == len(adds):
                    for (fid, _feat), out in zip(adds, added):
                        pfid = out.id()
                        if pfid is not None and pfid >= 0:
                            fid_map[fid] = (pfid, latest[fid][2])
                changed = True
            if geom_changes:
                pr.changeGeometryValues(geom_changes)
                changed = True
            if attr_changes and score_idx >= 0:
                pr.changeAttributeValues(
                    {pfid: {score_idx: sc} for pfid, sc in attr_changes.items()})
                changed = True
            if not changed:
                return



            if self.dock_widget is not None:
                try:




                    self.dock_widget.set_auto_run_found_count(
                        (self._auto_run_ctx or {}).get("prompt") or "",
                        len(fid_map))
                except (RuntimeError, AttributeError):
                    pass





            _notify_provider_write(layer)
            self._repaint_live_layer(layer)
        except (RuntimeError, AttributeError):


            pass
        finally:




            self._auto_live_draw_ms = getattr(
                self, "_auto_live_draw_ms", 0.0) + (
                    _time.monotonic() - _draw_t0) * 1000.0
            self._auto_live_draw_ticks = getattr(
                self, "_auto_live_draw_ticks", 0) + 1

    def _repaint_live_layer(self, layer) -> None:


















        canvas = None
        try:
            canvas = self.iface.mapCanvas()
        except (RuntimeError, AttributeError):
            canvas = None
        if canvas is None:
            layer.triggerRepaint()
            return
        self._connect_live_repaint_pacer(canvas)
        try:
            drawing = bool(canvas.isDrawing())
        except (RuntimeError, AttributeError):
            drawing = False
        if drawing or not self._live_repaint_allowed_now():



            self._auto_live_repaint_pending = True
            try:
                layer.triggerRepaint(True)
            except (RuntimeError, AttributeError, TypeError):
                pass
            self._arm_live_repaint_cooldown()
            return
        self._auto_live_repaint_pending = False
        self._note_live_repaint_sent()
        layer.triggerRepaint()



    def _live_repaint_allowed_now(self) -> bool:

        import time
        not_before = getattr(self, "_auto_live_repaint_not_before", 0.0)
        return time.monotonic() >= not_before

    def _note_live_repaint_sent(self) -> None:






        import time
        frame_s = getattr(self, "_auto_live_frame_s", 0.0)
        _, cost_ratio, max_ms = auto_live_repaint_settings()
        cool = min(max_ms / 1000.0, frame_s * cost_ratio)
        self._auto_live_repaint_not_before = time.monotonic() + cool
        self._auto_live_frame_started = time.monotonic()

    def _arm_live_repaint_cooldown(self) -> None:



        import time

        from qgis.PyQt.QtCore import QTimer
        if getattr(self, "_auto_live_cooldown_timer", None) is None:
            timer = QTimer(self.iface.mainWindow())
            timer.setSingleShot(True)
            timer.timeout.connect(self._on_live_repaint_cooldown)
            self._auto_live_cooldown_timer = timer
        if self._auto_live_cooldown_timer.isActive():
            return
        wait_ms = int(max(0.0, getattr(self, "_auto_live_repaint_not_before", 0.0) - time.monotonic()) * 1000) + 20
        self._auto_live_cooldown_timer.start(wait_ms)

    def _on_live_repaint_cooldown(self) -> None:

        if not getattr(self, "_auto_live_repaint_pending", False):
            return
        layer = getattr(self, "_auto_selection_layer", None)
        if layer is None:
            return
        try:
            if not layer.isValid():
                return
            canvas = self.iface.mapCanvas()
            if canvas is not None and canvas.isDrawing():
                return
            self._auto_live_repaint_pending = False
            self._note_live_repaint_sent()
            layer.triggerRepaint()
        except (RuntimeError, AttributeError):
            pass

    def _connect_live_repaint_pacer(self, canvas) -> None:



        if self._auto_live_pacer_canvas is canvas:
            return
        try:
            from ...core.server_dials import feature_enabled
            enabled = feature_enabled("live_repaint_pacer")
        except Exception:  # noqa: BLE001
            enabled = True
        if not enabled:
            return
        self._disconnect_live_repaint_pacer()




        try:
            from .canvas_redraw_handover import hold_map_picture_for_live_run
            hold_map_picture_for_live_run(canvas)
        except (RuntimeError, AttributeError, ImportError):  # nosec B110
            pass
        try:
            canvas.mapCanvasRefreshed.connect(self._on_live_canvas_refreshed)


            try:
                canvas.renderStarting.connect(self._on_live_canvas_render_started)
            except (RuntimeError, AttributeError, TypeError):
                pass
            self._auto_live_pacer_canvas = canvas
        except (RuntimeError, AttributeError, TypeError):
            self._auto_live_pacer_canvas = None

    def _disconnect_live_repaint_pacer(self) -> None:


        canvas = self._auto_live_pacer_canvas
        self._auto_live_pacer_canvas = None
        self._auto_live_repaint_pending = False
        if canvas is None:
            return
        try:
            from .canvas_redraw_handover import release_live_run_picture_hold
            release_live_run_picture_hold(canvas)
        except (RuntimeError, AttributeError, ImportError):  # nosec B110
            pass
        try:
            canvas.mapCanvasRefreshed.disconnect(self._on_live_canvas_refreshed)
        except (RuntimeError, AttributeError, TypeError):
            pass
        try:
            canvas.renderStarting.disconnect(self._on_live_canvas_render_started)
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _on_live_canvas_render_started(self) -> None:

        import time
        self._auto_live_frame_started = time.monotonic()

    def _on_live_canvas_refreshed(self) -> None:




        import time
        started = getattr(self, "_auto_live_frame_started", 0.0)
        if started:
            measured = max(0.0, time.monotonic() - started)
            prev = getattr(self, "_auto_live_frame_s", 0.0)



            from ...core.server_dials import dial_in_range
            decay = dial_in_range(
                "tuning.auto.live_frame_decay", 0.3, 0.05, 0.95)
            self._auto_live_frame_s = (
                measured if measured > prev else prev * (1.0 - decay) + measured * decay)
            self._auto_live_frame_started = 0.0
        if not self._auto_live_repaint_pending:
            return
        if not self._live_repaint_allowed_now():
            self._arm_live_repaint_cooldown()
            return
        self._auto_live_repaint_pending = False
        layer = getattr(self, "_auto_selection_layer", None)
        if layer is None:
            return
        try:
            if layer.isValid():
                self._note_live_repaint_sent()
                layer.triggerRepaint()
        except (RuntimeError, AttributeError):
            pass

    def _request_auto_live_repaint(self) -> None:






        from qgis.PyQt.QtCore import QTimer
        if self._auto_repaint_timer is None:
            self._auto_repaint_timer = QTimer(self.iface.mainWindow())
            self._auto_repaint_timer.setSingleShot(True)
            self._auto_repaint_timer.timeout.connect(self._apply_auto_live_deltas)
        if not self._auto_repaint_timer.isActive():
            from ...core.server_dials import dial_in_range
            n = len(self._auto_live_fid_map)
            interval, _, _ = auto_live_repaint_settings()
            ease_threshold = dial_in_range(
                "tuning.auto.live_repaint_ease_threshold", 1500, 100, 100000)
            if n >= ease_threshold:
                interval_cap = dial_in_range(
                    "tuning.auto.live_repaint_interval_cap_ms", 1200, 200, 10000)
                ease_step = dial_in_range(
                    "tuning.auto.live_repaint_ease_step_ms", 200, 0, 2000)
                interval = min(
                    interval_cap, interval + (n // ease_threshold) * ease_step)
            self._auto_repaint_timer.start(interval)

    def _pause_preview_jobs(self) -> None:





        if getattr(self, "_preview_jobs_paused", None) is not None:
            return
        try:
            from ...core.server_dials import feature_enabled
            enabled = feature_enabled("pause_preview_jobs")
        except Exception:  # noqa: BLE001
            enabled = True
        if not enabled:
            return
        try:
            canvas = self.iface.mapCanvas()
            prev = bool(canvas.previewJobsEnabled())
            if prev:
                canvas.setPreviewJobsEnabled(False)
            self._preview_jobs_paused = prev
        except (RuntimeError, AttributeError):
            self._preview_jobs_paused = None

    def _resume_preview_jobs(self) -> None:


        prev = getattr(self, "_preview_jobs_paused", None)
        self._preview_jobs_paused = None
        if prev:
            try:
                self.iface.mapCanvas().setPreviewJobsEnabled(True)
            except (RuntimeError, AttributeError):
                pass

    def _stop_auto_live_pump(self) -> None:




        self._resume_preview_jobs()
        self._disconnect_live_repaint_pacer()
        self._abort_auto_stitcher()



        self._auto_live_fid_map = {}




        for attr in ("_auto_repaint_timer", "_auto_live_cooldown_timer"):
            timer = getattr(self, attr, None)
            setattr(self, attr, None)
            if timer is not None:
                try:
                    timer.stop()
                    timer.timeout.disconnect()
                    timer.deleteLater()
                except (RuntimeError, AttributeError, TypeError):
                    pass


        self._auto_live_frame_s = 0.0
        self._auto_live_frame_started = 0.0
        self._auto_live_repaint_not_before = 0.0

    def _reset_auto_live_pipeline(self) -> None:





        self._stop_auto_live_pump()


        from .auto_client_profile import stop_gui_gap_watch
        stop_gui_gap_watch(self)

        self._auto_finalize_gen += 1
        self._auto_finalize_state = None




        self._stop_review_refine_thread()





        self._finish_billed_autosave()

        self._auto_preview_build_gen += 1
        self._auto_preview_build_state = None

    def _drop_auto_tile_bridge(self) -> None:






        bridge = self._auto_tile_bridge
        self._auto_tile_bridge = None
        if bridge is not None:
            try:
                bridge.cancel()
            except (RuntimeError, AttributeError):
                pass



            try:
                bridge.release_render_clone()
            except (RuntimeError, AttributeError):
                pass
