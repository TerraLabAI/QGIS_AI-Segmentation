







from __future__ import annotations

import time
from pathlib import Path

from qgis.core import Qgis, QgsCoordinateTransform, QgsGeometry, QgsMessageLog, QgsProject, QgsRectangle
from qgis.gui import QgisInterface, QgsRubberBand
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QMenu

from ..core.log_scrub import stop_log_collector
from ..core.prompt_manager import FrozenCropSession, PromptManager
from ..core.qt_compat import QAction
from ..core.review_defaults import AUTO_DEFAULT_CONFIDENCE as _AUTO_DEFAULT_CONFIDENCE
from ..core.review_defaults import (
    REFINE_CLEAN_DEFAULT,
    REFINE_EXPAND_DEFAULT,
    REFINE_FILL_HOLES_DEFAULT,
    REFINE_FILL_HOLES_MAX_M2_DEFAULT,
    REFINE_MIN_SIZE_M2_DEFAULT,
    REFINE_ORTHO_DEFAULT,
    REFINE_POINTS_PCT_DEFAULT,
    REFINE_SIMPLIFY_DEFAULT,
    REFINE_SMOOTH_DEFAULT,
)
from .ai_segmentation_dockwidget import AISegmentationDockWidget
from .ai_segmentation_maptool import AISegmentationMapTool
from .plugin.auto_autosave_offload import AutoAutosaveOffloadMixin
from .plugin.auto_correct import AutoCorrectMixin
from .plugin.auto_density_probe import AutoDensityProbeMixin
from .plugin.auto_detail_window import AutoDetailWindowMixin
from .plugin.auto_exemplar_grouping import AutoExemplarGroupingMixin
from .plugin.auto_export_offload import AutoExportOffloadMixin
from .plugin.auto_finalize_steps import AutoFinalizeStepsMixin
from .plugin.auto_flow import AutoFlowMixin
from .plugin.auto_grid_fill import AutoGridFillMixin
from .plugin.auto_imagery_guard import AutoImageryGuardMixin
from .plugin.auto_lifecycle import AutoLifecycleMixin
from .plugin.auto_object_build import AutoObjectBuildMixin
from .plugin.auto_results import AutoResultsMixin
from .plugin.auto_review import AutoReviewMixin
from .plugin.auto_review_display import AutoReviewDisplayMixin
from .plugin.auto_review_geometry import AutoReviewGeometryMixin
from .plugin.auto_review_offload import AutoReviewOffloadMixin
from .plugin.auto_review_open import AutoReviewOpenMixin
from .plugin.auto_review_params import AutoReviewParamsMixin
from .plugin.auto_run import AutoRunMixin
from .plugin.auto_run_terminal import AutoRunTerminalMixin
from .plugin.auto_server_finalize import AutoServerFinalizeMixin
from .plugin.auto_shape_edit import AutoShapeEditMixin
from .plugin.auto_shape_overrides import AutoShapeOverridesMixin
from .plugin.auto_tile_plan import AutoTilePlanMixin
from .plugin.auto_zone import AutoZoneMixin
from .plugin.bridge_isolation import BridgeIsolationMixin
from .plugin.correct_ai_route import CorrectAiRouteMixin
from .plugin.correct_focus import CorrectFocusMixin
from .plugin.credits_watch import AutoCreditsWatchMixin
from .plugin.demo_scene import DemoSceneMixin
from .plugin.env_setup import EnvSetupMixin
from .plugin.exemplars import ExemplarsMixin
from .plugin.handoff_seed_layers import HandoffSeedLayersMixin
from .plugin.handoff_shape import HandoffShapeMixin
from .plugin.local_ai_install_lock import LocalAiInstallLockMixin
from .plugin.local_ai_warm import LocalAiWarmMixin
from .plugin.manual_add import ManualAddMixin
from .plugin.manual_cloud_predictor import ManualCloudPredictorMixin
from .plugin.manual_crop_window import ManualCropWindowMixin
from .plugin.manual_crops import ManualCropsMixin
from .plugin.manual_handoff import ManualHandoffMixin
from .plugin.manual_hover_preview import ManualHoverPreviewMixin
from .plugin.manual_object_billing import ManualObjectBillingMixin
from .plugin.manual_predict import ManualPredictMixin
from .plugin.manual_shape_cache import ManualShapeCacheMixin
from .plugin.manual_workflow import ManualWorkflowMixin
from .plugin.plugin_canvas_state import CanvasStateMixin
from .plugin.plugin_gui_shell import GuiShellMixin
from .plugin.plugin_host_events import HostEventsMixin
from .plugin.plugin_server_prefetch import ServerPrefetchMixin
from .plugin.qgis_edit_bridge import QgisEditBridgeMixin
from .plugin.qgis_edit_tool_messages import QgisEditToolMessagesMixin
from .plugin.shared import detach_widget_from_main_window, join_orphaned_workers, park_orphaned_worker


class AISegmentationPlugin(
    AutoFlowMixin,
    AutoCreditsWatchMixin,
    AutoDetailWindowMixin,
    AutoGridFillMixin,
    AutoTilePlanMixin,
    AutoDensityProbeMixin,
    AutoCorrectMixin,
    LocalAiWarmMixin,
    LocalAiInstallLockMixin,
    CorrectAiRouteMixin,
    ManualCloudPredictorMixin,
    AutoShapeEditMixin,
    AutoShapeOverridesMixin,
    CorrectFocusMixin,
    QgisEditBridgeMixin,
    BridgeIsolationMixin,
    QgisEditToolMessagesMixin,
    AutoRunMixin,
    AutoImageryGuardMixin,
    AutoResultsMixin,
    AutoReviewDisplayMixin,
    HandoffSeedLayersMixin,
    AutoRunTerminalMixin,
    AutoServerFinalizeMixin,
    AutoExemplarGroupingMixin,
    AutoAutosaveOffloadMixin,
    AutoExportOffloadMixin,
    AutoObjectBuildMixin,
    AutoReviewParamsMixin,
    AutoReviewGeometryMixin,
    AutoReviewOffloadMixin,
    AutoFinalizeStepsMixin,
    AutoReviewOpenMixin,
    AutoReviewMixin,
    ManualHandoffMixin,
    ManualHoverPreviewMixin,
    ManualAddMixin,
    HandoffShapeMixin,
    ExemplarsMixin,
    AutoLifecycleMixin,
    AutoZoneMixin,
    DemoSceneMixin,
    EnvSetupMixin,
    ManualObjectBillingMixin,
    ManualWorkflowMixin,
    ManualCropsMixin,
    ManualCropWindowMixin,
    ManualShapeCacheMixin,
    ManualPredictMixin,
    CanvasStateMixin,
    GuiShellMixin,
    ServerPrefetchMixin,
    HostEventsMixin,
):




    def __init__(self, iface: QgisInterface):
        self.iface = iface
        self.plugin_dir = Path(__file__).parent.parent.parent

        self.dock_widget: AISegmentationDockWidget | None = None
        self._dock_created = False
        self.map_tool: AISegmentationMapTool | None = None
        self.action: QAction | None = None
        self.terralab_menu: QMenu | None = None
        self.terralab_toolbar = None

        self.predictor = None


        self._local_predictor_held = None



        self._manual_credit_ledger = None


        self._manual_charge_tasks = []
        self.prompts = PromptManager()

        self.current_mask = None
        self.current_score = 0.0
        self.current_transform_info = None





        self._mask_preview_memo = None
        self._manual_outline_memo = None
        self.current_low_res_mask = None
        self.saved_polygons = []



        self._refine_handoff_active = False


        self._is_refining_saved_object = False


        self._active_refine_origin_entry = None


        self._deleted_objects_stack: list = []


        self._handoff_selected_entries: list = []
        self._handoff_selection_band = None
        self._handoff_hover_band = None
        self._handoff_hover_entry = None




        self._handoff_hit_index = None
        self._handoff_tok2entry: dict = {}
        self._handoff_hit_tok_seq: int = 0



        self._handoff_prewarm_timer = None
        self._correct_hover_warm_timer = None


        self._handoff_det_id_seq = None


        self._handoff_imported_det_ids: set[int] = set()



        self._handoff_source_layer = None
        self._pending_refine_import = False




        self._refine_install_pending = False







        self._handoff_pending_layer = None
        self._handoff_kept_layer = None
        self._mask_state_history: list = []
        self._frozen_sessions: list[FrozenCropSession] = []
        self._active_crop_points_positive: list[tuple[float, float]] = []
        self._active_crop_points_negative: list[tuple[float, float]] = []


        self._unfrozen_display_polygon: QgsGeometry | None = None

        self._initialized = False
        self._setup_done = False
        self._current_layer = None
        self._current_layer_name = ""


        self._headless = False
        self._headless_error = None




        self._refine_simplify = REFINE_SIMPLIFY_DEFAULT


        self._refine_points_pct = REFINE_POINTS_PCT_DEFAULT
        self._refine_smooth = REFINE_SMOOTH_DEFAULT



        self._refine_clean = REFINE_CLEAN_DEFAULT
        self._refine_expand = REFINE_EXPAND_DEFAULT
        self._refine_fill_holes = REFINE_FILL_HOLES_DEFAULT


        self._refine_fill_holes_max_m2 = REFINE_FILL_HOLES_MAX_M2_DEFAULT
        self._refine_ortho = REFINE_ORTHO_DEFAULT
        self._refine_min_area = 200


        self._refine_min_size_m2 = REFINE_MIN_SIZE_M2_DEFAULT

        self._refine_max_size_m2 = 0.0

        self._is_non_georeferenced_mode = False
        self._is_online_layer = False
        self._disjoint_warning_shown = False


        self._unsure_warning_shown = False


        self._rasterio_repair_attempted = False



        self._crop_errors_reported = set()


        self._current_crop_info = None
        self._current_raster_path = None
        self._encoding_in_progress = False
        self._shortcut_filter = None
        self._current_crop_canvas_mupp = None
        self._current_crop_actual_mupp = None
        self._current_crop_scale_factor = None
        self.deps_install_worker = None
        self.download_worker = None
        self._verify_worker = None
        self._predictor_worker = None
        self._startup_check_worker = None
        self._device_info_worker = None




        self._install_includes_local_model = True



        self._env_ready = False



        self._key_revalidate_pending = False
        self._config_prefetch_task = None



        self._config_refresh_timer = None



        self._config_last_fetch_unix: float = 0.0



        self._credits_watch_timer = None
        self._credits_activation_relay = None
        self._credits_last_read_unix: float = 0.0


        self._credits_stable_reads: int = 0
        self._credits_last_fingerprint = None
        self._plan_upgrade_announced = False


        self._catalog_prefetch_task = None

        self._last_key_validation_unix: float = 0.0

        self._last_conn_notice_monotonic: float = 0.0
        self._pairing_worker = None
        self._pairing_cancel_task = None

        self.mask_rubber_band: QgsRubberBand | None = None




        self.saved_rubber_bands: list[QgsRubberBand | None] = []

        self._previous_map_tool = None
        self._stopping_segmentation = False
        self._exporting_in_progress = False



        self._canvas_to_raster_xform: QgsCoordinateTransform | None = None
        self._raster_to_canvas_xform: QgsCoordinateTransform | None = None


        self._auto_zone: QgsRectangle | None = None



        self._auto_zone_polygon = None


        self._auto_free_zone_fit = None



        self._auto_clip_polygon = None
        self._auto_clip_engine = None
        self._zone_selection_tool = None


        self._history_thumb_job = None

        self._init_auto_correct_state()

        self._init_qgis_bridge_state()


        self._init_bridge_isolation_state()





        self._auto_detail_user_locked = False
        self._auto_detail_lock_prompt = ""



        self._auto_detail_seeded: int | None = None



        self._auto_exemplar_seed_m: float | None = None
        self._zone_rubber_band: QgsRubberBand | None = None
        self._zone_delete_badge = None
        self._zone_badge_filter = None
        self._zone_escape_filter = None
        self._zone_grid_rubber_band = None




        self._zone_grid_geom_cache = {}


        self._auto_rescan_band = None
        self._auto_rescan_rects: dict = {}





        self._auto_grid_suppressed = False
        self._tile_manager = None




        from ..core.exemplar_store import ExemplarStore
        self._auto_exemplar_store = ExemplarStore()
        self._exemplar_maptool = None
        self._exemplar_bands: dict = {}
        self._maptool_before_exemplar = None
        self._pending_exemplar_label = 1




        self._maptool_before_zone = None


        self._auto_worker = None


        self._auto_tile_bridge = None




        self._auto_merger = None
        self._auto_crs_authid: str | None = None
        self._auto_gsd: float = 0.0
        self._auto_gsd_m: float = 0.0



        self._auto_mask_gsd: float = 0.0



        self._auto_merge_separate: bool = True



        self._auto_merge_mode_source: str = "prompt"







        self._auto_is_exemplar_only: bool = False
        self._auto_raw_fragments: list | None = None
        self._auto_raw_n_total: int = 0
        self._auto_raw_cov_sum: float = 0.0
        self._auto_raw_cov_sq_sum: float = 0.0
        self._auto_tile_ground_area: float = 0.0



        self._auto_retain_raw: bool = False



        self._auto_collect_raw: bool = False
        self._auto_selection_layer = None




        self._auto_display_mode = "random"
        self._auto_run_id: str | None = None



        self._auto_default_export_run_id: str | None = None
        self._auto_run_ctx: dict | None = None


        self._auto_cancelled_slot = None


        self._auto_quota_stop_banner: str | None = None
        self._last_usage: dict = {}
        self._usage_fetch_task = None


        self._billing_warning_shown = False

        self._warmup_task = None
        self._last_warmup_monotonic: float = 0.0
        self._session_end_task = None




        self._auto_run_plan: dict | None = None
        self._auto_run_plan_task = None

        self._auto_run_plan_task_prompt = ""


        self._auto_plan_detect_wait: dict | None = None
        self._auto_plan_detect_generation = 0
        self._auto_plan_detect_resumed = False




        self._auto_attribute_filters: list[dict[str, str]] = []




        self._auto_token_cache: dict[str, str] = {}
        self._auto_token_task = None

        self._last_auto_result: dict | None = None



        self._auto_render_ms: int = 0
        self._auto_detect_t0: float = 0.0


        self._auto_tel_stop_reason: str | None = None
        self._auto_skipped_tiles: int = 0
        self._auto_timeout_tiles: int = 0

        self._auto_review: dict | None = None

        self._auto_headless_run: bool = False



        self._auto_review_preset_overrides: dict | None = None




        self._auto_confidence: float = _AUTO_DEFAULT_CONFIDENCE




        self._auto_raw_count: int = 0



        self._auto_dense_tiles: int = 0







        self._auto_objects: list = []



        self._review_push_err_logged: bool = False







        self._auto_preview_geoms: list = []
        self._auto_preview_build_state: dict | None = None
        self._auto_preview_build_gen: int = 0




        self._auto_stitcher = None





        self._review_refine_thread = None

        self._review_refine_inflight: dict = {}
        self._review_refine_stamp = None


        self._auto_autosave_thread = None


        self._auto_export_job = None



        self._auto_imagery_probe = None
        self._auto_imagery_resume = None
        self._auto_repaint_timer = None





        self._auto_live_repaint_pending = False
        self._auto_live_pacer_canvas = None




        self._auto_live_frame_s = 0.0
        self._auto_live_frame_started = 0.0
        self._auto_live_repaint_not_before = 0.0
        self._auto_live_cooldown_timer = None




        self._auto_live_fid_map: dict = {}




        self._auto_finalize_state: dict | None = None
        self._auto_finalize_gen: int = 0







        self._auto_reslice_cache: dict = {"key": None, "geoms": {}}





        self._review_fid_map: dict = {}



        self._review_live_refiners: dict = {}

    def initGui(self):








        try:
            self._build_gui()
        except Exception:
            try:
                self.unload()
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            raise

    def unload(self):





        if getattr(self, "_auto_headless_run", False) or getattr(self, "_headless", False):
            self._unload_deferred = True
            QgsMessageLog.logMessage(
                "Unload deferred: a detection started from the API is still "
                "running", "AI Segmentation", level=Qgis.MessageLevel.Info)
            return
        self._unload_deferred = False



        if self.dock_widget is not None:
            try:
                self.dock_widget.stop_dock_content_build()
            except (RuntimeError, AttributeError):
                pass

        try:
            from ..core import sibling_sign_in
            sibling_sign_in.cancel("ai-segmentation")
        except Exception:  # noqa: BLE001
            pass  # nosec B110



        try:
            self._finish_auto_review_export_offload()
        except Exception:  # noqa: BLE001
            pass  # nosec B110


        try:
            self._abandon_imagery_probe()
        except Exception:  # noqa: BLE001
            pass  # nosec B110



        parked_drop = getattr(self, "_pending_autosave_drop", None)
        if parked_drop is not None:
            try:
                parked_drop()
            except Exception:  # noqa: BLE001
                pass  # nosec B110


        try:
            self._unregister_processing_provider()
        except Exception:  # noqa: BLE001
            pass  # nosec B110

        try:
            from ..agent_bridge import unregister_product
            unregister_product("segmentation")
        except Exception:  # noqa: BLE001
            pass  # nosec B110



        try:
            from ..core.telemetry import flush as _telemetry_flush
            _telemetry_flush()
        except Exception:
            pass  # nosec B110
        try:
            from ..core.telemetry import stop_flush_timer as _telemetry_stop_timer
            _telemetry_stop_timer()
        except Exception:
            pass  # nosec B110


        try:
            self._signal_gpu_session_end("unload")
        except Exception:  # noqa: BLE001
            pass  # nosec B110




        try:
            stop_log_collector()
        except Exception:  # noqa: BLE001
            pass  # nosec B110


        try:
            from .plugin.run_export_upload import cancel_inflight_uploads
            cancel_inflight_uploads()
        except Exception:  # noqa: BLE001
            pass  # nosec B110


        _crop_read_worker = None
        try:
            _read = getattr(self, "_crop_read", None)
            if isinstance(_read, dict):
                _crop_read_worker = _read.get("worker")
        except Exception:  # noqa: BLE001
            pass  # nosec B110




        try:
            self._invalidate_manual_encode()
        except Exception:  # noqa: BLE001
            pass  # nosec B110


        try:
            from ..core.online_layer_twin import release_online_layer_twin
            release_online_layer_twin()
        except Exception:  # noqa: BLE001
            pass  # nosec B110



        try:
            from ..core.xyz_tile_fetch import forget_direct_tile_fetch_failures
            forget_direct_tile_fetch_failures()
        except Exception:  # noqa: BLE001
            pass  # nosec B110



        try:
            from ..core.raster_dataset_cache import release_raster_datasets
            release_raster_datasets()
        except Exception:  # noqa: BLE001
            pass  # nosec B110



        self._abort_qgis_edit_bridge_if_active()




        try:
            self._clear_bridge_isolation()
        except Exception:  # noqa: BLE001
            pass  # nosec B110


        try:
            self._end_correct_wait()
        except Exception:  # noqa: BLE001
            pass  # nosec B110



        try:
            if getattr(self, "_refine_handoff_active", False) and self.saved_polygons:
                self._collect_manual_refine_into_review()
        except Exception:
            pass  # nosec B110



        self._refine_handoff_active = False
        self._pending_refine_import = False
        self._handoff_source_layer = None


        self._refine_add_mode_active = False
        self._ai_add_install_pending = False

        try:
            QgsProject.instance().layersWillBeRemoved.disconnect(
                self._on_layers_will_be_removed)
        except (TypeError, RuntimeError):
            pass
        try:
            QgsProject.instance().readProject.disconnect(
                self._on_project_read_sweep_temp)
        except (TypeError, RuntimeError):
            pass





        try:
            self._stop_canvas_crs_watch()
        except (RuntimeError, AttributeError):
            pass



        try:
            from .plugin.canvas_redraw_handover import (
                release_live_run_picture_hold,
                release_map_picture_hold,
            )
            release_live_run_picture_hold(self.iface.mapCanvas())
            release_map_picture_hold(self.iface.mapCanvas())
        except (RuntimeError, AttributeError, ImportError):
            pass  # nosec B110

        try:
            if self._shortcut_filter is not None:




                for target in (
                    lambda: self.iface.mainWindow(),
                    lambda: self.iface.mapCanvas().viewport(),
                    lambda: self.iface.mapCanvas(),
                ):
                    try:
                        target().removeEventFilter(self._shortcut_filter)
                    except (RuntimeError, AttributeError):
                        pass
                self._shortcut_filter = None
        except (RuntimeError, AttributeError):
            pass


        if self.dock_widget:
            try:
                self.dock_widget.cleanup_signals()
            except (TypeError, RuntimeError, AttributeError):
                pass









            try:
                _dock_signals = [
                    (self.dock_widget.manual_engine_changed, self._on_manual_engine_changed),
                    (self.dock_widget.install_requested, self._on_install_requested),
                    (self.dock_widget.cancel_install_requested, self._on_cancel_install),
                    (self.dock_widget.start_segmentation_requested, self._on_start_segmentation),
                    (self.dock_widget.save_polygon_requested, self._on_save_polygon),
                    (self.dock_widget.export_layer_requested, self._on_export_layer),
                    (self.dock_widget.undo_requested, self._on_undo),
                    (self.dock_widget.stop_segmentation_requested, self._on_stop_segmentation),
                    (self.dock_widget.clear_selection_requested, self._on_clear_selection),
                    (self.dock_widget.refine_settings_changed, self._on_refine_settings_changed),
                    (self.dock_widget.size_filter_changed, self._on_size_filter_changed),
                    (self.dock_widget.fill_holes_size_changed,
                     self._on_fill_holes_size_changed),
                    (self.dock_widget.clean_edges_changed, self._on_clean_edges_changed),
                    (self.dock_widget.outline_budget_changed,
                     self._on_outline_budget_changed),
                    (self.dock_widget.settings_clicked, self._on_settings_clicked),
                    (self.dock_widget.pairing_requested, self._on_pairing_requested),
                    (self.dock_widget.pairing_cancel_requested, self._on_cancel_pairing),
                    (self.dock_widget.visibilityChanged, self._on_dock_visibility_changed),
                    (self.dock_widget.mode_changed, self._on_mode_changed),
                    (self.dock_widget.auto_detect_requested, self._on_auto_detect_requested),
                    (self.dock_widget.history_rerun_requested, self._on_history_rerun_requested),
                    (self.dock_widget.history_reuse_prompt_requested,
                     self._on_history_reuse_prompt_requested),
                    (self.dock_widget.zone_draw_requested, self._on_zone_draw_requested),
                    (self.dock_widget.auto_zone_source_picked,
                     self._on_auto_zone_source_picked),
                    (self.dock_widget.auto_step_changed, self._on_auto_step_changed),
                    (self.dock_widget.auto_detail_changed, self._on_auto_detail_changed),
                    (self.dock_widget.auto_advanced_toggled, self._on_auto_advanced_toggled),
                    (self.dock_widget.auto_prompt_committed, self._reseed_auto_detail_for_object),
                    (self.dock_widget.auto_refine_changed, self._on_auto_refine_changed_debounced),
                    (self.dock_widget.auto_export_requested, self._on_auto_export_clicked),
                    (self.dock_widget.auto_retry_requested, self._on_auto_retry_guarded),
                    (self.dock_widget.auto_review_exit_requested, self._on_auto_review_exit_clicked),
                    (self.dock_widget.auto_display_mode_changed, self._on_auto_display_mode_changed),
                    (self.dock_widget.auto_library_requested, self._on_auto_library_clicked),
                    (self.dock_widget.auto_demo_requested, self._on_auto_demo_requested),
                    (self.dock_widget.auto_reshape_ai_requested, self._on_reshape_ai_requested),
                    (self.dock_widget.auto_reshape_done_requested, self._on_reshape_done),
                    (self.dock_widget.auto_correct_method_changed, self._on_correct_method_changed),
                    (self.dock_widget.auto_ai_add_requested, self._on_ai_add_requested),
                    (self.dock_widget.auto_ai_add_keep_requested, self._route_save_add_mode),
                    (self.dock_widget.auto_review_install_cancel_requested,
                     self._on_review_install_cancel_requested),
                    (self.dock_widget.auto_exit_requested, self._on_auto_exit_clicked),
                    (self.dock_widget.auto_add_exemplar_requested, self._on_add_exemplar_requested),
                    (self.dock_widget.auto_exemplar_remove_requested, self._on_exemplar_remove_requested),
                    (self.dock_widget.auto_zero_assist_clicked, self._on_auto_zero_assist_clicked),
                    (self.dock_widget.auto_escape_pressed, self._on_auto_escape_shortcut),
                    (self.dock_widget.auto_enter_pressed, self._on_auto_enter_pressed),
                    (self.dock_widget.auto_review_confidence_changed, self._on_auto_review_confidence_changed),
                    (self.dock_widget.auto_review_confidence_preview, self._on_auto_review_confidence_preview),
                    (self.dock_widget.auto_show_tiles_changed, self._on_auto_show_tiles_toggled),
                    (self.dock_widget.auto_edit_in_qgis_requested, self.enter_qgis_edit_bridge),
                    (self.dock_widget.auto_add_polygon_requested, self._on_add_polygon_requested),
                    (self.dock_widget.auto_qgis_bridge_done_requested,
                     self.finish_qgis_edit_bridge),
                    (self.dock_widget.auto_qgis_bridge_tool_requested,
                     self.activate_qgis_bridge_tool),
                    (self.dock_widget.auto_qgis_bridge_undo_requested,
                     self.undo_qgis_bridge_edit),
                    (self.dock_widget.auto_qgis_bridge_gesture_requested,
                     self._on_bridge_gesture_requested),
                    (self.dock_widget.auto_qgis_bridge_points_changed,
                     self._on_bridge_points_changed),
                    (self.dock_widget.auto_qgis_bridge_delete_requested,
                     self.delete_bridge_target_polygon),
                    (self.dock_widget._auto_review_debounce_timer.timeout,
                     self._on_auto_review_refine_debounced),

                    (self.dock_widget.auto_correction_undo_requested,
                     self._on_auto_correction_undo_requested),
                    (self.dock_widget.auto_correction_clear_requested,
                     self._on_auto_correction_clear_requested),
                    (self.dock_widget.auto_review_step_requested,
                     self._on_auto_review_step_requested),
                    (self.dock_widget.auto_correct_status_action_requested,
                     self._on_correct_status_action_requested),

                    (self.dock_widget.auto_shape_edit_requested,
                     self._on_auto_shape_edit_requested),
                    (self.dock_widget.auto_remove_requested, self._on_remove_requested),
                    (self.dock_widget.auto_shape_only_changed, self._on_shape_only_changed),
                    (self.dock_widget.auto_shape_only_reset_requested,
                     self._on_shape_only_reset),
                ]


                if self.dock_widget.dock_content_built:
                    _dock_signals += [
                        (self.dock_widget.layer_combo.layerChanged, self._on_layer_combo_changed),
                        (self.dock_widget.auto_layer_combo.layerChanged,
                         self._on_auto_layer_combo_changed),
                        (self.dock_widget.auto_cancel_btn.clicked, self._on_auto_cancel_clicked),
                        (self.dock_widget.auto_correct_undo_shortcut.activated,
                         self._on_auto_undo_pressed),
                    ]
            except (TypeError, RuntimeError, AttributeError):
                _dock_signals = []
            for sig, slot in _dock_signals:
                try:
                    sig.disconnect(slot)
                except (TypeError, RuntimeError, AttributeError):
                    pass



            for _timer_name in (
                "_progress_timer",
                "_refine_debounce_timer",
                "_auto_review_debounce_timer",
                "_auto_prompt_debounce_timer",
                "_visibility_debounce_timer",
                "_auto_progress_ease_timer",
            ):
                try:
                    _timer = getattr(self.dock_widget, _timer_name, None)
                    if _timer is not None:
                        _timer.stop()
                except (AttributeError, RuntimeError):
                    pass
        if self.map_tool:


            for sig_name, slot in (
                ("positive_click", self._on_positive_click),
                ("negative_click", self._on_negative_click),
                ("double_click", self._on_canvas_double_click),
                ("cursor_moved", self._on_handoff_cursor_moved),
                ("cursor_moved", self._on_hover_cursor_moved),
                ("tool_deactivated", self._on_tool_deactivated),
            ):
                try:
                    getattr(self.map_tool, sig_name).disconnect(slot)
                except (TypeError, RuntimeError, AttributeError):
                    pass
        try:
            self.iface.mapCanvas().extentsChanged.disconnect(self._on_manual_view_changed)
        except (TypeError, RuntimeError, AttributeError):
            pass





        self._teardown_hover_preview()





        try:
            self._drop_cloud_correct_predictor()
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        if self.predictor:
            import threading
            pred = self.predictor
            self.predictor = None
            t = threading.Thread(target=lambda: pred.cleanup(), daemon=True)
            t.start()
            t.join(timeout=8)
            if t.is_alive():
                QgsMessageLog.logMessage(
                    "Predictor cleanup did not finish within 8s",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Warning
                )



        self._cancel_pairing_worker()
        self._pairing_worker = None
        self._pairing_cancel_task = None


        if self._config_refresh_timer is not None:
            try:
                self._config_refresh_timer.stop()
            except RuntimeError:
                pass
            self._config_refresh_timer = None
        self._disarm_credits_watch()
        self._cancel_task("_config_prefetch_task")
        self._cancel_task("_catalog_prefetch_task")
        self._cancel_task("_usage_fetch_task")
        self._cancel_task("_warmup_task")
        self._auto_plan_detect_wait = None
        self._cancel_task("_auto_run_plan_task")
        self._cancel_task("_auto_token_task")
        self._cancel_manual_charge_tasks()







        _qthread_workers = [
            self.deps_install_worker, self.download_worker, self._verify_worker,
            getattr(self, "_predictor_worker", None),
            getattr(self, "_startup_check_worker", None),
            getattr(self, "_device_info_worker", None),
            getattr(self, "_manual_encode_worker", None),
            _crop_read_worker,
            getattr(self, "_remove_data_worker", None),
        ]
        for worker in _qthread_workers:
            if worker:
                try:
                    if hasattr(worker, "progress"):
                        worker.progress.disconnect()
                except (TypeError, RuntimeError):
                    pass
                try:



                    if hasattr(worker, "done"):
                        worker.done.disconnect()
                except (TypeError, RuntimeError):
                    pass














        try:
            from ..core.checkpoint_manager import request_download_cancel
            request_download_cancel()
        except Exception:
            pass  # nosec B110
        for worker in _qthread_workers:






            try:
                if worker and worker.isRunning() and hasattr(worker, "cancel"):
                    worker.cancel()
            except (RuntimeError, AttributeError):
                pass





        deadline = time.monotonic() + 3.0
        for worker in _qthread_workers:



            try:
                if not (worker and worker.isRunning()):
                    continue
                left_ms = int(max(0.0, deadline - time.monotonic()) * 1000)
                if left_ms > 0 and worker.wait(left_ms):
                    continue
                park_orphaned_worker(worker)
            except RuntimeError:
                pass
        self.deps_install_worker = None
        self.download_worker = None
        self._verify_worker = None
        self._predictor_worker = None
        self._startup_check_worker = None
        self._device_info_worker = None
        self._manual_encode_worker = None
        self._remove_data_worker = None


        try:
            self.action.triggered.disconnect(self.toggle_dock_widget)
        except (TypeError, RuntimeError, AttributeError):
            pass

        from .terralab_menu import remove_from_plugins_menu, remove_plugin_from_menu
        try:
            remove_from_plugins_menu(self.iface, self.action)
        except (RuntimeError, AttributeError):
            pass
        ai_edit_action = getattr(self, "ai_edit_action", None)
        if ai_edit_action is not None:
            try:
                remove_from_plugins_menu(self.iface, ai_edit_action)
            except (RuntimeError, AttributeError):
                pass
        if self.terralab_menu:
            try:
                remove_plugin_from_menu(
                    self.terralab_menu, self.action, self.iface.mainWindow())
            except (RuntimeError, AttributeError):
                pass
            ai_edit_action = getattr(self, "ai_edit_action", None)
            if ai_edit_action is not None:
                try:
                    remove_plugin_from_menu(
                        self.terralab_menu, ai_edit_action, self.iface.mainWindow())
                except (RuntimeError, AttributeError):
                    pass
            self.terralab_menu = None

        from .terralab_toolbar import remove_action_from_toolbar
        if self.terralab_toolbar:
            try:
                remove_action_from_toolbar(
                    self.terralab_toolbar, self.action, self.iface.mainWindow())
            except (RuntimeError, AttributeError):
                pass
            ai_edit_action = getattr(self, "ai_edit_action", None)
            if ai_edit_action is not None:
                try:
                    remove_action_from_toolbar(
                        self.terralab_toolbar, ai_edit_action, self.iface.mainWindow())
                except (RuntimeError, AttributeError):
                    pass
            self.terralab_toolbar = None
        self.ai_edit_action = None






        try:
            self._autosave_pending_auto_review(exit_path="unload")
        except Exception:  # noqa: BLE001
            pass  # nosec B110



        parked_drop = getattr(self, "_pending_autosave_drop", None)
        if parked_drop is not None:
            try:
                parked_drop()
            except Exception:  # noqa: BLE001
                pass  # nosec B110





        try:
            self._autosave_manual_saved_polygons()
        except Exception:  # noqa: BLE001
            pass  # nosec B110













        if self.dock_widget:
            try:
                self.iface.removeDockWidget(self.dock_widget)
            except (RuntimeError, AttributeError):
                pass
            detach_widget_from_main_window(self.dock_widget)
            self.dock_widget = None

        try:
            from .dock.temp_icon_dirs import remove_icon_dirs
            remove_icon_dirs()
        except Exception:  # noqa: BLE001
            pass  # nosec B110





        account_dialog = getattr(self, "_account_dialog", None)
        if account_dialog is not None:
            try:
                account_dialog.reject()
            except (RuntimeError, AttributeError):
                pass
            detach_widget_from_main_window(account_dialog)
            self._account_dialog = None







        _dead_map_tools = [self.map_tool]
        if self.map_tool:
            try:
                self.map_tool.clear_markers()
            except (RuntimeError, AttributeError):
                pass
            try:
                if self.iface.mapCanvas().mapTool() == self.map_tool:
                    self.iface.mapCanvas().unsetMapTool(self.map_tool)



                    self._restore_previous_map_tool()
            except RuntimeError:
                pass
            self.map_tool = None


        self._safe_remove_rubber_band(self.mask_rubber_band)
        self.mask_rubber_band = None

        for rb in self.saved_rubber_bands:
            self._safe_remove_rubber_band(rb)
        self.saved_rubber_bands = []
        self._remove_handoff_layers()




        self._cancel_history_thumbnail()






        self._stop_auto_detection()
        auto_worker = self._auto_worker
        if auto_worker is not None:


            for slot in (getattr(self, "_auto_cancelled_slot", None),
                         self._on_auto_cancelled):
                if slot is None:
                    continue
                try:
                    auto_worker.cancelled.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
            self._auto_cancelled_slot = None
            try:
                still_running = auto_worker.isRunning() and not auto_worker.wait(5000)
            except RuntimeError:
                still_running = False
            if still_running:




                park_orphaned_worker(auto_worker)
            self._auto_worker = None
        self._drop_auto_tile_bridge()
        _dead_map_tools += [
            getattr(self, "_zone_selection_tool", None),
            getattr(self, "_exemplar_maptool", None),
            getattr(self, "_shape_maptool", None),
        ]
        self._teardown_auto_mode()







        for tool in _dead_map_tools:
            if tool is None:
                continue
            try:
                if self.iface.mapCanvas().mapTool() is tool:
                    self.iface.mapCanvas().unsetMapTool(tool)
            except (RuntimeError, AttributeError):
                pass
            try:
                tool.deleteLater()
            except (RuntimeError, AttributeError):
                pass
        self._zone_selection_tool = None
        self._exemplar_maptool = None
        self._shape_maptool = None





        for attr in ("_auto_repaint_timer", "_auto_live_cooldown_timer"):
            timer = getattr(self, attr, None)
            if timer is None:
                continue
            try:
                timer.stop()
                timer.timeout.disconnect()
                timer.deleteLater()
            except (TypeError, RuntimeError, AttributeError):
                pass
            setattr(self, attr, None)









        join_orphaned_workers(5.0)

    def _ensure_dock_widget(self):

        if self._dock_created:
            return
        self._dock_created = True


        try:
            from ..core.telemetry import new_session
            new_session()
        except Exception:
            pass  # nosec B110

        self.dock_widget = AISegmentationDockWidget(self.iface.mainWindow())





        def _wire_dock_children():
            self.dock_widget.layer_combo.layerChanged.connect(self._on_layer_combo_changed)
            self.dock_widget.auto_layer_combo.layerChanged.connect(
                self._on_auto_layer_combo_changed)

            self.dock_widget.auto_cancel_btn.clicked.connect(self._on_auto_cancel_clicked)




            self.dock_widget.auto_correct_undo_shortcut.activated.connect(
                self._on_auto_undo_pressed)

        self.dock_widget.manual_engine_changed.connect(
            self._on_manual_engine_changed)
        self.dock_widget.install_requested.connect(self._on_install_requested)
        self.dock_widget.cancel_install_requested.connect(self._on_cancel_install)
        self.dock_widget.start_segmentation_requested.connect(self._on_start_segmentation)
        self.dock_widget.save_polygon_requested.connect(self._on_save_polygon)
        self.dock_widget.export_layer_requested.connect(self._on_export_layer)
        self.dock_widget.undo_requested.connect(self._on_undo)
        self.dock_widget.stop_segmentation_requested.connect(self._on_stop_segmentation)
        self.dock_widget.clear_selection_requested.connect(self._on_clear_selection)
        self.dock_widget.refine_settings_changed.connect(self._on_refine_settings_changed)
        self.dock_widget.size_filter_changed.connect(self._on_size_filter_changed)
        self.dock_widget.fill_holes_size_changed.connect(
            self._on_fill_holes_size_changed)
        self.dock_widget.clean_edges_changed.connect(
            self._on_clean_edges_changed)
        self.dock_widget.outline_budget_changed.connect(
            self._on_outline_budget_changed)
        self.dock_widget.settings_clicked.connect(self._on_settings_clicked)
        self.dock_widget.pairing_requested.connect(self._on_pairing_requested)
        self.dock_widget.pairing_cancel_requested.connect(self._on_cancel_pairing)
        self.dock_widget.mode_changed.connect(self._on_mode_changed)
        self.dock_widget.auto_detect_requested.connect(self._on_auto_detect_requested)
        self.dock_widget.auto_library_requested.connect(self._on_auto_library_clicked)
        self.dock_widget.auto_demo_requested.connect(self._on_auto_demo_requested)
        self.dock_widget.history_rerun_requested.connect(self._on_history_rerun_requested)
        self.dock_widget.history_reuse_prompt_requested.connect(
            self._on_history_reuse_prompt_requested)
        self.dock_widget.zone_draw_requested.connect(self._on_zone_draw_requested)

        self.dock_widget.auto_zone_source_picked.connect(
            self._on_auto_zone_source_picked)
        self.dock_widget.auto_step_changed.connect(self._on_auto_step_changed)
        self.dock_widget.auto_detail_changed.connect(self._on_auto_detail_changed)
        self.dock_widget.auto_advanced_toggled.connect(self._on_auto_advanced_toggled)
        self.dock_widget.auto_prompt_committed.connect(self._reseed_auto_detail_for_object)

        self.dock_widget.auto_refine_changed.connect(self._on_auto_refine_changed_debounced)
        self.dock_widget.auto_export_requested.connect(self._on_auto_export_clicked)
        self.dock_widget.auto_retry_requested.connect(self._on_auto_retry_guarded)
        self.dock_widget.auto_review_exit_requested.connect(self._on_auto_review_exit_clicked)
        self.dock_widget.auto_display_mode_changed.connect(self._on_auto_display_mode_changed)

        self.dock_widget.auto_reshape_ai_requested.connect(
            self._on_reshape_ai_requested)
        self.dock_widget.auto_reshape_done_requested.connect(self._on_reshape_done)

        self.dock_widget.auto_correct_method_changed.connect(
            self._on_correct_method_changed)
        self.dock_widget.auto_ai_add_requested.connect(self._on_ai_add_requested)
        self.dock_widget.auto_ai_add_keep_requested.connect(
            self._route_save_add_mode)


        self.dock_widget.auto_review_install_cancel_requested.connect(
            self._on_review_install_cancel_requested)
        self.dock_widget.auto_exit_requested.connect(self._on_auto_exit_clicked)

        self.dock_widget.auto_add_exemplar_requested.connect(self._on_add_exemplar_requested)
        self.dock_widget.auto_exemplar_remove_requested.connect(self._on_exemplar_remove_requested)
        self.dock_widget.auto_zero_assist_clicked.connect(self._on_auto_zero_assist_clicked)
        self.dock_widget.auto_escape_pressed.connect(self._on_auto_escape_shortcut)
        self.dock_widget.auto_enter_pressed.connect(self._on_auto_enter_pressed)
        self.dock_widget.auto_review_confidence_changed.connect(
            self._on_auto_review_confidence_changed)
        self.dock_widget.auto_review_confidence_preview.connect(
            self._on_auto_review_confidence_preview)
        self.dock_widget.auto_show_tiles_changed.connect(self._on_auto_show_tiles_toggled)
        self.dock_widget._auto_review_debounce_timer.timeout.connect(
            self._on_auto_review_refine_debounced)

        self._connect_auto_correct_signals()





        try:
            self.dock_widget.auto_edit_in_qgis_requested.connect(
                self.enter_qgis_edit_bridge)
            self.dock_widget.auto_add_polygon_requested.connect(
                self._on_add_polygon_requested)
            self.dock_widget.auto_qgis_bridge_done_requested.connect(
                self.finish_qgis_edit_bridge)
            self.dock_widget.auto_qgis_bridge_tool_requested.connect(
                self.activate_qgis_bridge_tool)
            self.dock_widget.auto_qgis_bridge_undo_requested.connect(
                self.undo_qgis_bridge_edit)
            self.dock_widget.auto_qgis_bridge_gesture_requested.connect(
                self._on_bridge_gesture_requested)
            self.dock_widget.auto_qgis_bridge_points_changed.connect(
                self._on_bridge_points_changed)
            self.dock_widget.auto_qgis_bridge_delete_requested.connect(
                self.delete_bridge_target_polygon)
        except (AttributeError, RuntimeError):
            pass
        self.dock_widget.when_dock_content_built(_wire_dock_children)






        self._first_time_setup_done = False




        self._interactive_setup_done = False
        self.dock_widget.visibilityChanged.connect(self._on_dock_visibility_changed)
        self.iface.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_widget)
        self._initialized = True
        self._setup_done = True



        if self.dock_widget.isVisible() and not self._first_time_setup_done:
            self._on_dock_visibility_changed(True)


__all__ = [
    "AISegmentationPlugin",
]
