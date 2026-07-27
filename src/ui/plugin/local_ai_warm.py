











































from __future__ import annotations

import os

from qgis.core import (
    Qgis,
    QgsMessageLog,
    QgsPointXY,
)

from .shared import _debounce_timer





CORRECT_HOVER_WARM_MS = 200



MANUAL_HOVER_WARM_MS = 300




MANUAL_LOCAL_HOVER_WARM_MS = 120



_MAX_HOVER_WARM_MS = 5_000


def local_ai_warmup_enabled() -> bool:







    try:
        from ...core.server_dials import feature_enabled

        return feature_enabled("local_ai_warmup")
    except Exception:  # noqa: BLE001  # nosec B110
        return True






_MIN_HOVER_WARM_MS = 50


def correct_hover_warm_ms() -> int:





    try:
        from ...core.server_dials import dial_in_range

        return int(dial_in_range("ui.warm_hover_ms.correct", CORRECT_HOVER_WARM_MS,
                                 _MIN_HOVER_WARM_MS, _MAX_HOVER_WARM_MS))
    except Exception:  # noqa: BLE001  # nosec B110
        return CORRECT_HOVER_WARM_MS


def manual_hover_warm_ms() -> int:

    try:
        from ...core.server_dials import dial_in_range

        return int(dial_in_range("ui.warm_hover_ms.manual", MANUAL_HOVER_WARM_MS,
                                 _MIN_HOVER_WARM_MS, _MAX_HOVER_WARM_MS))
    except Exception:  # noqa: BLE001  # nosec B110
        return MANUAL_HOVER_WARM_MS


def manual_local_hover_warm_ms() -> int:






    try:
        from ...core.server_dials import dial_in_range

        return int(dial_in_range("ui.warm_hover_ms.manual_local",
                                 MANUAL_LOCAL_HOVER_WARM_MS,
                                 _MIN_HOVER_WARM_MS, _MAX_HOVER_WARM_MS))
    except Exception:  # noqa: BLE001  # nosec B110
        return MANUAL_LOCAL_HOVER_WARM_MS


class LocalAiWarmMixin:


    def _correct_step_resting_on_ai(self) -> bool:





        if getattr(self, "_headless", False):
            return False
        if getattr(self, "_correct_method", "ai") != "ai":
            return False
        if self._auto_review is None or getattr(self, "_auto_review_step", 0) != 1:
            return False

        if getattr(self, "_refine_handoff_active", False):
            return False
        if getattr(self, "_qgis_bridge_active", False):
            return False
        return getattr(self, "_auto_worker", None) is None

    def _correct_ai_warm_allowed(self) -> bool:

        if not local_ai_warmup_enabled():
            return False


        if self._correct_ai_route_is_remote():
            return False
        return self._correct_step_resting_on_ai()

    def _correct_crop_warm_allowed(self) -> bool:








        if not local_ai_warmup_enabled():
            return False
        if not self._correct_step_resting_on_ai():
            return False
        if self._correct_ai_route_is_remote():
            return True
        return self._correct_ai_warm_allowed()

    def _review_ai_warm_allowed(self) -> bool:








        if not local_ai_warmup_enabled():
            return False


        if self._correct_ai_route_is_remote():
            return False
        if getattr(self, "_headless", False):
            return False
        if getattr(self, "_correct_method", "ai") != "ai":
            return False
        if self._auto_review is None:
            return False

        if getattr(self, "_refine_handoff_active", False):
            return False
        if getattr(self, "_qgis_bridge_active", False):
            return False
        return getattr(self, "_auto_worker", None) is None

    def _start_or_warm_local_ai(self) -> None:












        predictor = getattr(self, "predictor", None)
        if predictor is None:
            self._start_local_ai_load_for_correct()
            return
        if getattr(self, "_encoding_in_progress", False):
            return
        try:
            predictor.warm_up()
        except Exception:  # noqa: BLE001
            pass  # nosec B110

    def _warm_local_ai_for_review(self) -> None:






        if self._review_ai_warm_allowed():
            self._start_or_warm_local_ai()

    def _warm_local_ai_for_correct(self) -> None:





        if self._correct_ai_warm_allowed():
            self._start_or_warm_local_ai()

    def _start_local_ai_load_for_correct(self) -> None:








        for attr in ("deps_install_worker", "_verify_worker", "download_worker",
                     "_predictor_worker", "_startup_check_worker"):
            worker = getattr(self, attr, None)
            try:
                if worker is not None and worker.isRunning():
                    return
            except RuntimeError:
                continue
        try:
            from ...core.checkpoint_manager import checkpoint_exists
            from ...core.venv_manager import get_venv_status
            if not checkpoint_exists():
                return



            ready, _msg = get_venv_status(allow_subprocess_probe=False)
            if not ready:
                return
        except Exception:  # noqa: BLE001
            return
        QgsMessageLog.logMessage(
            "Review armed on the AI fix method: loading the local model "
            "ahead of the pick",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        self._warm_predictor_on_ready = True
        self._load_predictor()

    def _warm_local_ai_for_manual(self) -> None:















        if getattr(self, "_headless", False):
            return
        predictor = getattr(self, "predictor", None)
        if predictor is None:
            self._warm_predictor_on_ready = True
            return
        if getattr(self, "_encoding_in_progress", False):
            return
        try:
            predictor.warm_up()
        except Exception:  # noqa: BLE001
            pass  # nosec B110





    def _schedule_correct_hover_warm(self, idx) -> None:


        if idx is None or not self._correct_crop_warm_allowed():
            return
        if self.dock_widget is None:
            return





        self._correct_hover_warm_idx = idx
        if (getattr(self, "predictor", None) is None
                and not self._correct_ai_route_is_remote()):


            return
        _debounce_timer(self, "_correct_hover_warm_timer", self.dock_widget,
                        correct_hover_warm_ms(), self._warm_hovered_correct_crop)

    def _warm_hovered_correct_crop(self) -> None:





        idx = getattr(self, "_correct_hover_warm_idx", None)
        if idx is None or not self._correct_crop_warm_allowed():
            return
        if getattr(self, "_encoding_in_progress", False):
            return
        objects = getattr(self, "_auto_objects", None) or []
        if idx < 0 or idx >= len(objects):
            return
        geom = objects[idx][0]
        if geom is None or geom.isEmpty():
            return
        if self._correct_ai_route_is_remote():




            if not self._ensure_cloud_correct_predictor():
                return
        elif getattr(self, "predictor", None) is None:
            return
        if not self._bind_correct_crop_context():
            return
        from ...core.crop_window import crop_window_key, neighborhood_crop_window
        bb = geom.boundingBox()





        cx, cy, scale = neighborhood_crop_window(
            (bb.xMinimum(), bb.yMinimum(), bb.xMaximum(), bb.yMaximum()),
            self._get_native_pixel_size())
        if crop_window_key(cx, cy, scale) in (
                getattr(self, "_encoded_crop_window", None),
                getattr(self, "_inflight_crop_window", None)):
            return
        self._extract_and_encode_crop(
            QgsPointXY(cx, cy), mupp_override=scale, show_busy=False, quiet=True)





    def _manual_warm_allowed(self) -> bool:



















        if not local_ai_warmup_enabled():
            return False
        if getattr(self, "_headless", False):
            return False
        if getattr(self, "_refine_handoff_active", False):
            return False
        if getattr(self, "predictor", None) is None:
            return False
        if self._current_layer is None or not self._current_raster_path:
            return False
        if getattr(self, "_is_online_layer", False) and not self._online_warm_layer_ready():
            return False
        if getattr(self, "_encoding_in_progress", False):
            return False
        if getattr(self, "_pending_manual_click", None) is not None:
            return False
        if self.current_mask is not None:
            return False
        if getattr(self, "_is_refining_saved_object", False):
            return False




        if getattr(self, "_unfrozen_display_polygon", None) is not None:
            return False
        if self._frozen_sessions or self._active_crop_points_positive:
            return False
        if self._active_crop_points_negative:
            return False


        prompts = getattr(self, "prompts", None)
        if prompts is None:
            return True
        try:
            return sum(prompts.point_count) == 0
        except (AttributeError, TypeError):
            return True

    def _online_warm_layer_ready(self) -> bool:





        from ...core.online_layer_twin import online_layer_twin, online_prewarm_enabled

        if not online_prewarm_enabled():
            return False
        return online_layer_twin(self._current_layer) is not None

    def _begin_speculative_manual_crop(self, center_point, scale) -> None:




        self._speculative_manual_crop = bool(self._extract_and_encode_crop(
            center_point, mupp_override=scale, show_busy=False, quiet=True))

    def _abandon_speculative_manual_crop(self, click_raster_pt=None) -> bool:













        if not getattr(self, "_speculative_manual_crop", False):
            return False
        if not getattr(self, "_encoding_in_progress", False):
            self._speculative_manual_crop = False
            return False
        if getattr(self, "_encode_cursor_set", True):


            self._speculative_manual_crop = False
            return False
        if getattr(self, "_pending_manual_click", None) is not None:



            return False
        if click_raster_pt is not None and self._warm_serves_click(click_raster_pt):


            self._speculative_manual_crop = False
            return False
        self._speculative_manual_crop = False
        self._invalidate_manual_encode()








        self._current_crop_info = None
        self._encoded_crop_window = None
        QgsMessageLog.logMessage(
            "Dropped a warm-up crop: the user asked for something else",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        return True

    def _schedule_manual_hover_warm(self, canvas_point) -> None:


        if self.dock_widget is None:
            return




        self._manual_warm_canvas_point = canvas_point
        if not self._manual_warm_allowed():
            return
        rest_ms = (manual_local_hover_warm_ms() if self._manual_warm_reads_a_file()
                   else manual_hover_warm_ms())
        _debounce_timer(self, "_manual_hover_warm_timer", self.dock_widget,
                        rest_ms, self._warm_hovered_manual_crop)

    def _on_manual_view_changed(self) -> None:







        tool = getattr(self, "map_tool", None)
        if self.dock_widget is None or tool is None or getattr(self, "_headless", False):
            return
        try:
            if not tool.isActive():
                return
        except RuntimeError:
            return
        if not self._manual_warm_allowed():
            return
        rest_ms = (manual_local_hover_warm_ms() if self._manual_warm_reads_a_file()
                   else manual_hover_warm_ms())
        _debounce_timer(self, "_manual_view_warm_timer", self.dock_widget,
                        rest_ms, self._warm_crop_under_still_cursor)

    def _still_cursor_map_point(self):



        from qgis.PyQt.QtGui import QCursor

        try:
            canvas = self.iface.mapCanvas()
            if not canvas.underMouse():
                return None
            pixel = canvas.mapFromGlobal(QCursor.pos())
            if not canvas.rect().contains(pixel):
                return None
            return QgsPointXY(canvas.getCoordinateTransform().toMapCoordinates(
                pixel.x(), pixel.y()))
        except (RuntimeError, AttributeError):
            return None

    def _warm_crop_under_still_cursor(self) -> None:




        point = self._still_cursor_map_point()
        if point is None:
            return
        self._manual_warm_canvas_point = point
        self._warm_hovered_manual_crop()

    def _manual_warm_reads_a_file(self) -> bool:






        if getattr(self, "_is_online_layer", False):
            return False
        source = getattr(self, "_current_raster_path", None)
        if not source:
            return False



        memo = getattr(self, "_manual_warm_file_memo", None)
        if memo is not None and memo[0] == source:
            return memo[1]
        try:
            provider = self._current_layer.dataProvider()
            if provider is None or provider.name() != "gdal":
                reads_a_file = False
            else:
                reads_a_file = os.path.isfile(source)
        except (RuntimeError, AttributeError, OSError, ValueError):
            return False
        self._manual_warm_file_memo = (source, reads_a_file)
        return reads_a_file

    def _replay_hover_warm_when_ready(self) -> None:






        if getattr(self, "_headless", False):
            return
        if getattr(self, "predictor", None) is None:
            return
        if getattr(self, "_correct_hover_warm_idx", None) is not None:
            self._warm_hovered_correct_crop()
        if getattr(self, "_manual_warm_canvas_point", None) is not None:
            self._warm_hovered_manual_crop()

    def _warm_hovered_manual_crop(self) -> None:













        point = getattr(self, "_manual_warm_canvas_point", None)
        if point is None or not self._manual_warm_allowed():
            return
        try:
            raster_pt = self._transform_to_raster_crs(point)
        except (RuntimeError, AttributeError):
            return
        if raster_pt is None or not self._is_point_in_raster_extent(raster_pt):
            return


        if self._check_crop_status(raster_pt) == "ok":
            return
        from ...core.crop_window import crop_window_key
        window = self._warm_window_for(raster_pt)
        if window is None:
            return
        center, scale = window
        if crop_window_key(center.x(), center.y(), scale or 1.0) in (
                getattr(self, "_encoded_crop_window", None),
                getattr(self, "_inflight_crop_window", None)):
            return
        QgsMessageLog.logMessage(
            "Manual: warming the crop under the resting cursor",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        self._begin_speculative_manual_crop(center, scale)

    def _warm_window_for(self, raster_pt):


        from ...core.crop_window import snap_center_to_grid
        if getattr(self, "_is_online_layer", False):
            return self._online_grid_window(raster_pt)
        scale = self._compute_initial_scale_factor()
        cx, cy = snap_center_to_grid(
            raster_pt.x(), raster_pt.y(), scale or 1.0,
            self._get_native_pixel_size())
        return QgsPointXY(cx, cy), scale

    def _warm_serves_click(self, raster_pt) -> bool:


        from ...core.crop_window import crop_window_key
        try:
            window = self._warm_window_for(raster_pt)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return False
        if window is None:
            return False
        center, scale = window
        inflight = getattr(self, "_inflight_crop_window", None)
        return inflight is not None and inflight == crop_window_key(
            center.x(), center.y(), scale or 1.0)

    def _bind_correct_crop_context(self) -> bool:











        layer = self._resolve_auto_source_layer()
        if layer is None:
            return False
        try:
            if not self._is_layer_valid(layer):
                return False
            if self._needs_canvas_render(layer):
                return False
            source = layer.source()
        except (RuntimeError, AttributeError):
            return False
        if self._current_layer is layer and self._current_raster_path:
            return True
        held = self._current_layer is not None and self._current_layer is not layer
        session_owns_it = (getattr(self, "_manual_session_parked", False)
                           or bool(getattr(self, "saved_polygons", None)))
        if held and session_owns_it:



            return False
        self._current_layer = layer
        self._current_layer_name = layer.name().replace(" ", "_")
        self._current_raster_path = source
        self._is_online_layer = False
        self._is_non_georeferenced_mode = not self._is_layer_georeferenced(layer)
        if held:


            self._current_crop_info = None
            self._encoded_crop_window = None
            self._inflight_crop_window = None
        self._bind_correct_crop_transforms(layer)
        return True

    def _bind_correct_crop_transforms(self, layer=None) -> None:






        self._rebuild_manual_crs_transforms()
        self._start_canvas_crs_watch()
