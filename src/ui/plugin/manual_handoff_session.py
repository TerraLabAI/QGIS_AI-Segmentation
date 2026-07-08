








from __future__ import annotations

from qgis.core import Qgis, QgsGeometry, QgsMessageLog, QgsProject

from ...core.i18n import tr
from ...core.review_defaults import REFINE_SMOOTH_ITERATIONS


class ManualHandoffSessionMixin:










    def _resolve_auto_source_layer(self):


        ctx = self._auto_run_ctx or {}
        lid = ctx.get("layer_id")
        if lid:
            lyr = QgsProject.instance().mapLayer(lid)
            if lyr is not None:
                return lyr
        name = (self._auto_review or {}).get("source_layer_name")
        if name:
            for lyr in QgsProject.instance().mapLayersByName(name):
                return lyr
        return None

    def _manual_env_ready(self) -> bool:












        if self._ensure_cloud_correct_predictor():
            return True
        if self.predictor is not None:
            return True








        in_flight = [self.download_worker, self._predictor_worker,
                     self._startup_check_worker]
        if getattr(self, "_install_includes_local_model", True):
            in_flight += [self.deps_install_worker, self._verify_worker]
        for w in in_flight:
            try:
                if w is not None and w.isRunning():
                    return True
            except RuntimeError:
                continue






        try:
            from ...core.venv_manager import local_model_ready
            model_ok, _why = local_model_ready()
            if not model_ok:
                self._env_ready = False
                return False
        except Exception:  # nosec B110
            pass

        try:
            from ...core.checkpoint_manager import checkpoint_exists
            if not checkpoint_exists():
                self._env_ready = False
                return False
        except Exception:  # nosec B110
            pass


        try:
            from ...core.venv_manager import get_venv_status





            ready, _msg = get_venv_status(allow_subprocess_probe=False)
            self._env_ready = bool(ready)
            return bool(ready)
        except Exception:
            return True

    def _on_reshape_ai_requested(self) -> None:















        review = self._auto_review
        if not review or not self.dock_widget:
            return






        if getattr(self, "_refine_handoff_active", False) or getattr(
                self, "_qgis_bridge_active", False):
            return
        idx = getattr(self, "_correct_selected_idx", None)
        objects = getattr(self, "_auto_objects", None) or []
        if idx is None or idx < 0 or idx >= len(objects):
            return
        layer = self._resolve_auto_source_layer()
        if layer is None:
            return


        geom = objects[idx][0]
        anchor = geom.pointOnSurface() if geom is not None else None
        if anchor is not None and not anchor.isEmpty():
            pt = anchor.asPoint()
            self._reshape_open_anchor = (pt.x(), pt.y())
        else:
            self._reshape_open_anchor = None



        if not self._manual_env_ready():
            if self._local_ai_install_pending():
                return





            if (getattr(self, "_local_ai_load_failed", False)
                    and getattr(self, "_local_ai_install_attempted", False)):
                self._warn_local_ai_unavailable_once()
                return
            from ..dialogs.confirm_dialog import (
                PRIMARY,
                SECONDARY,
                ChoiceButton,
                ask_choice,
            )
            if ask_choice(
                self.iface.mainWindow(), tr("Fixing needs a one-time setup"),
                tr(
                    "Fixing a polygon uses the free on-device AI, which is not "
                    "installed yet. Install it now? It runs once and takes a few "
                    "minutes. The review waits for it, then opens this polygon "
                    "for you."),
                [ChoiceButton("cancel", tr("Cancel"), SECONDARY),
                 ChoiceButton("install", tr("Install now"), PRIMARY)],
                default="install", escape="cancel",
            ) != "install":
                return



            self._begin_local_ai_install("reshape")
            return

        self._disarm_shape_tool()
        self._handoff_source_layer = layer
        self._pending_refine_import = False
        self._refine_handoff_active = True
        self._auto_refined_in_manual = True
        try:
            import time as _time

            from ...core import telemetry_run_events
            self._refine_handoff_t0 = _time.monotonic()
            telemetry_run_events.track_refine_in_manual_entered(
                run_id=self._auto_run_id or "",
                instances=len(review.get("geoms", [])),
            )
        except Exception:
            pass  # nosec B110


        self._remove_auto_selection_layer()
        self._set_exemplar_bands_visible(False)
        self._set_auto_zone_overlays_visible(False)





        self._correct_selected_idx = idx
        try:
            self.dock_widget.set_correct_selection(1)
        except (RuntimeError, AttributeError):
            pass
        _push = getattr(self, "_push_shape_only_state", None)
        if _push is not None:
            _push()
        try:
            self.dock_widget.enter_ai_reshape_state()
        except (RuntimeError, AttributeError):
            pass







        if not self._cloud_correct_predictor_active():
            self._ensure_interactive_setup()
        self._enter_manual_refine_session()
        self._open_reshape_target()

    def _open_reshape_target(self) -> None:



        if self._pending_refine_import:
            return
        anchor = getattr(self, "_reshape_open_anchor", None)
        self._reshape_open_anchor = None
        if anchor is None or not self.saved_polygons:
            return
        from qgis.core import QgsPointXY
        pt = QgsPointXY(anchor[0], anchor[1])



        pair = getattr(self, "_handoff_crs_pair", None)
        if pair:
            xform = self._handoff_crs_xform(pair[0], pair[1])
            if xform is not None:
                try:
                    pt = xform.transform(pt)
                except Exception:  # noqa: BLE001  # nosec B110
                    pass
        idx = self._hit_test_saved_polygon(pt)
        if idx is not None:
            self._open_saved_polygon_for_edit(idx, pt)

    def _clear_refine_install_pending(self) -> None:




        if not getattr(self, "_refine_install_pending", False):
            return
        self._refine_install_pending = False
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_review_installing(False)
            except (RuntimeError, AttributeError):
                pass

    def _enter_manual_refine_session(self) -> None:




        review = self._auto_review
        layer = getattr(self, "_handoff_source_layer", None)
        if not review or layer is None:
            return




        if self.predictor is None:
            self._pending_refine_import = True







            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(0, self._retry_predictor_load_for_handoff)
            return
        self._pending_refine_import = False


        try:
            combo = self.dock_widget.layer_combo
            combo.blockSignals(True)
            try:
                combo.setLayer(layer)
            finally:


                combo.blockSignals(False)
        except (RuntimeError, AttributeError):
            pass

        self._on_start_segmentation(layer)



        self._seed_refine_from_review()
        self._import_review_geoms_as_saved(review)

    def _seed_refine_from_review(self) -> None:














        try:
            params = self._widget_review_params()
            self._refine_simplify = max(
                0.0, float(params.get("simplify_px") or 0.0))
            self._refine_points_pct = max(
                1, min(100, int(params.get("points_pct") or 100)))
            self._refine_smooth = (
                REFINE_SMOOTH_ITERATIONS if params.get("smooth") else 0)
            self._refine_clean = max(0.0, float(params.get("open_px") or 0.0))
            self._refine_fill_holes = bool(params.get("fill_holes"))
            self._refine_fill_holes_max_m2 = max(
                0.0, float(params.get("fill_max_m2") or 0.0))
            self._refine_ortho = bool(params.get("ortho"))
            self._refine_min_size_m2 = max(0.0, float(params.get("min_a") or 0.0))
            self._refine_max_size_m2 = 0.0
            self.dock_widget.set_refine_values(
                self._refine_simplify, self._refine_smooth,
                self._refine_expand, self._refine_fill_holes,
                right_angles=self._refine_ortho,
                fill_holes_max_m2=self._refine_fill_holes_max_m2,
                clean=self._refine_clean,
                points_pct=self._refine_points_pct)
            self.dock_widget.set_size_filter_values(
                self._refine_min_size_m2, self._refine_max_size_m2)
        except (RuntimeError, AttributeError):
            pass

    def _retry_predictor_load_for_handoff(self) -> None:










        if not (self._pending_refine_import and self._refine_handoff_active):
            return
        if self.predictor is not None:
            return
        for w in (self.deps_install_worker, self._verify_worker,
                  self.download_worker, self._predictor_worker,
                  self._startup_check_worker):
            try:
                if w is not None and w.isRunning():
                    return
            except RuntimeError:
                continue
        self._load_predictor()









    def _handoff_crs_xform(self, src_authid, dst_authid):




        if not src_authid or not dst_authid or src_authid == dst_authid:
            return None
        try:
            from qgis.core import (
                QgsCoordinateReferenceSystem,
                QgsCoordinateTransform,
            )
            src = QgsCoordinateReferenceSystem(src_authid)
            dst = QgsCoordinateReferenceSystem(dst_authid)
            if not src.isValid() or not dst.isValid():
                return None
            xform = QgsCoordinateTransform(src, dst, QgsProject.instance())
            return xform if xform.isValid() else None
        except (RuntimeError, AttributeError, TypeError):
            return None

    @staticmethod
    def _handoff_reproject(geom, xform):



        try:
            from ...core.qt_compat import geometry_op_succeeded

            out = QgsGeometry(geom)
            if not geometry_op_succeeded(out.transform(xform)):
                return None
            return None if out.isEmpty() else out
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _handoff_layer_authid(layer):

        try:
            if layer is None:
                return None
            crs = layer.crs()
            return crs.authid() if crs is not None and crs.isValid() else None
        except (RuntimeError, AttributeError):
            return None

    def _handoff_entries_to_run_crs(self, entries: list) -> list:








        pair = getattr(self, "_handoff_crs_pair", None)
        self._handoff_crs_pair = None
        if not pair or not entries:
            return entries
        run_authid, raster_authid = pair
        xform = self._handoff_crs_xform(raster_authid, run_authid)
        if xform is None:
            QgsMessageLog.logMessage(
                "Refine handoff: no transform back to the run CRS; "
                f"{len(entries)} shape(s) kept in {raster_authid}.",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return entries
        out, failed = [], 0
        for g, det_id, score, touched in entries:
            moved = self._handoff_reproject(g, xform)
            if moved is None:
                failed += 1
                moved = g
            out.append((moved, det_id, score, touched))
        if failed:
            QgsMessageLog.logMessage(
                f"Refine handoff: {failed} shape(s) kept in {raster_authid}, "
                "the transform back refused them.",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return out

    def _import_review_geoms_as_saved(self, review) -> None:













        geoms = review.get("geoms") or []



        scores = review.get("scores") or []
        ids = review.get("ids") or []
        crs = review.get("crs")
        run_authid = crs.authid() if crs is not None and crs.isValid() else None
        raster_authid = self._handoff_layer_authid(
            getattr(self, "_handoff_source_layer", None))



        self._handoff_crs_pair = None
        xform = self._handoff_crs_xform(run_authid, raster_authid)
        if xform is not None:
            self._handoff_crs_pair = (run_authid, raster_authid)




        authid = raster_authid or run_authid
        if authid:
            self._ensure_handoff_layers(authid)
        else:



            QgsMessageLog.logMessage(
                "Refine handoff: neither the run nor the raster has a valid "
                "CRS; no seed layers were created.",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)








        max_id = max((int(i) for i in ids if i is not None), default=-1)
        for fid in (getattr(self, "_auto_object_fids", None) or ()):
            if isinstance(fid, int) and fid > max_id:
                max_id = fid
        self._handoff_det_id_seq = max_id + 1



        exempt_ids = set(getattr(self, "_auto_manual_object_ids", None) or ())




        imported_ids: set[int] = set()
        failed = 0
        for n, g in enumerate(geoms):
            if g is None or g.isEmpty():
                continue
            if xform is not None:
                moved = self._handoff_reproject(g, xform)
                if moved is None:
                    failed += 1
                else:
                    g = moved
            det_id = ids[n] if n < len(ids) and ids[n] is not None else None
            if det_id is None:
                det_id = self._next_handoff_det_id()
            imported_ids.add(int(det_id))
            score = scores[n] if n < len(scores) and scores[n] is not None else None
            self.saved_polygons.append({
                "geometry_wkt": g.asWkt(),


                "geom_obj": g,
                "transform_info": {"crs": authid} if authid else None,
                "raw_mask": None,
                "points_positive": [],
                "points_negative": [],
                "refine_simplify": self._refine_simplify,
                "refine_points_pct": self._refine_points_pct,
                "refine_smooth": self._refine_smooth,
                "refine_clean": self._refine_clean,
                "refine_expand": self._refine_expand,
                "refine_fill_holes": self._refine_fill_holes,
                "refine_fill_holes_max_m2": self._refine_fill_holes_max_m2,
                "refine_ortho": self._refine_ortho,
                "refine_min_area": self._refine_min_area,
                "refine_min_size_m2": self._refine_min_size_m2,


                "refine_max_size_m2": 0.0,
                "manual_touched": det_id in exempt_ids,

                "validated": False,




                "det_id": int(det_id),
                "score": float(score) if score is not None else None,
            })


            self.saved_rubber_bands.append(None)
        if failed:
            QgsMessageLog.logMessage(
                f"Refine handoff: {failed} shape(s) kept in {run_authid}, "
                f"the transform to {raster_authid} refused them.",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        self._handoff_imported_det_ids = imported_ids
        self._rebuild_handoff_layers()
        if self.dock_widget:
            try:
                self.dock_widget.set_saved_polygon_count(len(self.saved_polygons))
            except (RuntimeError, AttributeError):
                pass

    def _on_reshape_done(self) -> None:







        if getattr(self, "_qgis_bridge_active", False):
            self.finish_qgis_edit_bridge()
            return
        if not self._refine_handoff_active:





            self._sweep_stale_refine_canvas()
            return

        if getattr(self, "_refine_add_mode_active", False):
            try:
                self._exit_ai_add_mode()
            except (RuntimeError, AttributeError):
                pass
        self._collect_manual_refine_into_review()
        try:
            import time as _time

            from ...core import telemetry_run_events
            t0 = getattr(self, "_refine_handoff_t0", None)
            telemetry_run_events.track_refine_in_manual_back(
                run_id=self._auto_run_id or "",
                validated_count=len((self._auto_review or {}).get("geoms", [])),
                duration_ms=int((_time.monotonic() - t0) * 1000) if t0 else None,
            )
        except Exception:
            pass  # nosec B110


        self._restore_auto_review_after_handoff()
        try:
            self.dock_widget.leave_ai_reshape_state()
        except (RuntimeError, AttributeError):
            pass
        self._arm_correct_select()

    def _sweep_stale_refine_canvas(self) -> None:









        if getattr(self, "_auto_review", None) is None:
            return



        self._end_correct_focus()
        tool = getattr(self, "map_tool", None)
        if tool is not None:
            try:
                tool.clear_markers()
            except (RuntimeError, AttributeError):
                pass
            try:
                canvas = self.iface.mapCanvas()
                if canvas.mapTool() is tool:
                    canvas.unsetMapTool(tool)
            except (RuntimeError, AttributeError):
                pass
        try:
            self.dock_widget.leave_ai_reshape_state()
        except (RuntimeError, AttributeError):
            pass


        self._arm_correct_select()

    def _abandon_fix_session_for_discard(self) -> None:













        if getattr(self, "_qgis_bridge_active", False):
            try:
                self._abort_qgis_edit_bridge_if_active()
            except (RuntimeError, AttributeError):
                pass
        if getattr(self, "_refine_add_mode_active", False):
            try:
                self._exit_ai_add_mode()
            except (RuntimeError, AttributeError):
                pass
        was_active = bool(getattr(self, "_refine_handoff_active", False))
        had_seeds = (getattr(self, "_handoff_pending_layer", None) is not None
                     or getattr(self, "_handoff_kept_layer", None) is not None)
        self._refine_handoff_active = False
        self._handoff_source_layer = None
        self._pending_refine_import = False
        self._handoff_crs_pair = None



        if was_active or had_seeds:
            try:
                self._teardown_manual_session()
            except Exception:
                pass  # nosec B110


            self._remove_handoff_layers()



        try:
            self._sweep_stale_refine_canvas()
        except Exception:  # nosec B110
            pass

    def _restore_auto_review_after_handoff(self) -> None:




        review = self._auto_review
        layer = getattr(self, "_handoff_source_layer", None)
        self._refine_handoff_active = False
        self._handoff_source_layer = None



        try:
            self.dock_widget.reset_refine_sliders()
        except (RuntimeError, AttributeError):
            pass  # nosec B110


        self._drop_cloud_correct_predictor()

        self._set_exemplar_bands_visible(True)
        self._set_auto_zone_overlays_visible(True)
        if layer is not None:
            self._remove_auto_selection_layer()
            self._auto_selection_layer = self._create_auto_selection_layer(layer)
        if self.dock_widget and review is not None:
            try:
                self.dock_widget.set_auto_review_active(
                    True, count=len(review.get("geoms") or []),
                    reset_controls=False)
            except (RuntimeError, AttributeError):
                pass



        self._refresh_auto_review_preview()
        self._start_auto_reslice()


        self._end_correct_focus()
