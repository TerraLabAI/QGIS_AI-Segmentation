from __future__ import annotations

from qgis.core import QgsProject
from qgis.PyQt.QtCore import Qt, QTimer, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFrame,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.activation_manager import (
    is_plugin_activated,
)
from ..core.i18n import tr
from .dock.about import DockAboutMixin
from .dock.activation_state import DockActivationMixin
from .dock.auto_build import DockAutoBuildMixin
from .dock.auto_correct_build import DockAutoCorrectBuildMixin
from .dock.auto_credits import DockAutoCreditsMixin
from .dock.auto_detail_level import DockAutoDetailLevelMixin
from .dock.auto_flow_steps import DockAutoFlowStepsMixin
from .dock.auto_prompt_box import DockAutoPromptBoxMixin
from .dock.auto_prompt_gate import DockAutoPromptGateMixin
from .dock.auto_prompt_suggest import DockAutoPromptSuggestMixin
from .dock.auto_review_build import DockAutoReviewBuildMixin
from .dock.auto_review_correct import DockAutoReviewCorrectMixin
from .dock.auto_review_panel import DockAutoReviewPanelMixin
from .dock.auto_run_block import DockAutoRunBlockMixin
from .dock.auto_run_lifecycle import DockAutoRunLifecycleMixin
from .dock.auto_run_status import DockAutoRunStatusMixin
from .dock.build import DockBuildMixin
from .dock.dock_sizing import dock_minimum_height
from .dock.exemplar_upsell import DockExemplarUpsellMixin
from .dock.install_lock import DockInstallLockMixin
from .dock.manual_credit_gate import DockManualCreditGateMixin
from .dock.manual_engine import DockManualEngineMixin
from .dock.manual_local_install import DockManualLocalInstallMixin
from .dock.manual_notice import DockManualNoticeMixin
from .dock.pro_ceiling_contact import DockProCeilingContactMixin
from .dock.pro_nudges import DockProNudgesMixin
from .dock.qgis_bridge import DockQgisBridgeMixin
from .dock.refine import DockRefineMixin
from .dock.server_switches import DockServerSwitchesMixin
from .dock.styles import (
    apply_input_theme_to_tree,
    apply_keyboard_focus_policy,
    apply_quiet_scrollbar,  # noqa: F401
)
from .dock.ui_refresh import DockStateMixin
from .dock.widgets import Mode, _WheelGuard


class AISegmentationDockWidget(
    DockBuildMixin,
    DockAutoBuildMixin,
    DockAutoReviewBuildMixin,
    DockAutoCorrectBuildMixin,
    DockRefineMixin,
    DockQgisBridgeMixin,
    DockAboutMixin,
    DockServerSwitchesMixin,
    DockManualEngineMixin,
    DockManualLocalInstallMixin,
    DockManualNoticeMixin,
    DockManualCreditGateMixin,
    DockActivationMixin,
    DockAutoPromptBoxMixin,
    DockAutoPromptGateMixin,
    DockAutoPromptSuggestMixin,
    DockAutoDetailLevelMixin,
    DockAutoCreditsMixin,
    DockProCeilingContactMixin,
    DockProNudgesMixin,
    DockAutoFlowStepsMixin,
    DockAutoRunBlockMixin,
    DockAutoRunLifecycleMixin,
    DockAutoRunStatusMixin,
    DockExemplarUpsellMixin,
    DockAutoReviewPanelMixin,
    DockAutoReviewCorrectMixin,
    DockInstallLockMixin,
    DockStateMixin,
    QDockWidget,
):

    install_requested = pyqtSignal()
    cancel_install_requested = pyqtSignal()

    auto_review_install_cancel_requested = pyqtSignal()
    start_segmentation_requested = pyqtSignal(object)
    undo_requested = pyqtSignal()
    save_polygon_requested = pyqtSignal()
    settings_clicked = pyqtSignal()
    export_layer_requested = pyqtSignal()
    stop_segmentation_requested = pyqtSignal()
    clear_selection_requested = pyqtSignal()
    pairing_requested = pyqtSignal(str)
    pairing_cancel_requested = pyqtSignal(str)


    refine_settings_changed = pyqtSignal(int, int, int, bool, bool)


    size_filter_changed = pyqtSignal(float, float)







    outline_budget_changed = pyqtSignal(float, int)



    fill_holes_size_changed = pyqtSignal(float)




    clean_edges_changed = pyqtSignal(float)
    mode_changed = pyqtSignal(object)
    manual_engine_changed = pyqtSignal(bool)
    auto_detect_requested = pyqtSignal()
    auto_library_requested = pyqtSignal()
    auto_demo_requested = pyqtSignal()
    history_rerun_requested = pyqtSignal(dict)
    history_reuse_prompt_requested = pyqtSignal(str)
    zone_draw_requested = pyqtSignal()
    auto_detail_changed = pyqtSignal(int)
    auto_advanced_toggled = pyqtSignal(bool)
    auto_prompt_committed = pyqtSignal(str)
    auto_step_changed = pyqtSignal(int)
    auto_refine_changed = pyqtSignal()
    auto_review_confidence_changed = pyqtSignal(int)
    auto_review_confidence_preview = pyqtSignal(int)
    auto_display_mode_changed = pyqtSignal(str)
    auto_show_tiles_changed = pyqtSignal(bool)
    auto_export_requested = pyqtSignal()
    auto_retry_requested = pyqtSignal()
    auto_review_exit_requested = pyqtSignal()
    auto_exit_requested = pyqtSignal()
    auto_add_exemplar_requested = pyqtSignal(int)
    auto_exemplar_remove_requested = pyqtSignal(str)
    auto_zero_assist_clicked = pyqtSignal(str, str)
    auto_escape_pressed = pyqtSignal()
    auto_enter_pressed = pyqtSignal()

    auto_correction_undo_requested = pyqtSignal()
    auto_correction_clear_requested = pyqtSignal()
    auto_review_step_requested = pyqtSignal(int)
    auto_correct_status_action_requested = pyqtSignal()
    auto_shape_edit_requested = pyqtSignal(str)


    auto_edit_in_qgis_requested = pyqtSignal()
    auto_add_polygon_requested = pyqtSignal()
    auto_qgis_bridge_done_requested = pyqtSignal()
    auto_qgis_bridge_tool_requested = pyqtSignal(str)
    auto_qgis_bridge_undo_requested = pyqtSignal()
    auto_qgis_bridge_gesture_requested = pyqtSignal(str)
    auto_qgis_bridge_points_changed = pyqtSignal(int)
    auto_qgis_bridge_delete_requested = pyqtSignal()

    auto_reshape_ai_requested = pyqtSignal()
    auto_reshape_done_requested = pyqtSignal()
    auto_remove_requested = pyqtSignal()


    auto_correct_method_changed = pyqtSignal(str)
    auto_ai_add_requested = pyqtSignal()
    auto_ai_add_keep_requested = pyqtSignal()






    auto_shape_only_changed = pyqtSignal(dict)
    auto_shape_only_reset_requested = pyqtSignal()











    _EXEMPLARS_SHIPPED_ENABLED = True

    @property
    def _EXEMPLARS_ENABLED(self) -> bool:



        if not self._EXEMPLARS_SHIPPED_ENABLED:
            return False
        from ..core.server_dials import feature_enabled

        return feature_enabled("exemplars")

    def __init__(self, parent=None):
        super().__init__(tr("AI Segmentation by TerraLab"), parent)


        self.setObjectName("AISegmentationDockWidget")

        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.setMinimumWidth(260)







        self.setMinimumHeight(dock_minimum_height(self))

        self._setup_title_bar()





        self._mode: Mode = Mode.AUTOMATIC
        self._auto_credits: int | None = None
        self._auto_credits_total: int | None = None
        self._auto_free_left: int | None = None
        self._auto_is_subscriber: bool = False



        self._auto_reset_date: str = ""
        self._auto_reset_display: str = ""
        self._auto_run_active: bool = False
        self._auto_zone_too_large: bool = False
        self._auto_zone_is_set: bool = False
        self._auto_review_active: bool = False



        self._auto_cancelling: bool = False







        self._auto_warmup_timer = None
        self._auto_warming_since: float | None = None
        self._auto_queue_position: int = 0
        self._auto_queue_eta: int = 0



        self._auto_est_credits: int | None = None


        self._refine_handoff: bool = False


        self._auto_positive_exemplars: int = 0


        self._auto_exemplar_items: list = []



        self._auto_started: bool = False

        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setSpacing(8)
        self.main_layout.setContentsMargins(8, 8, 8, 8)

        self._setup_ui()



        from .dock.font_scale import apply_font_scale_to_tree

        apply_font_scale_to_tree(self.main_widget)




        apply_input_theme_to_tree(self.main_widget)


        apply_keyboard_focus_policy(self.main_widget)
        apply_keyboard_focus_policy(self._custom_title_bar)

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.main_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)



        apply_quiet_scrollbar(scroll_area)


        body_holder = QWidget()
        body_col = QVBoxLayout(body_holder)
        body_col.setContentsMargins(0, 0, 0, 0)
        body_col.setSpacing(0)
        body_col.addWidget(scroll_area, 1)
        body_col.addWidget(self.update_gate_page, 1)
        apply_keyboard_focus_policy(self.update_gate_page)
        self.setWidget(body_holder)


        self._dock_scroll_area = scroll_area




        self._wheel_guard = _WheelGuard(scroll_area.viewport(), self)



        for _w in self.main_widget.findChildren((QComboBox, QSpinBox, QDoubleSpinBox, QSlider)):
            _w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            _w.installEventFilter(self._wheel_guard)

        self._dependencies_ok = False
        self._checkpoint_ok = False





        self._setup_section_wanted = False



        self._manual_install_dialog = None




        self._manual_install_failed = False
        self._segmentation_active = False
        self._has_mask = False
        self._saved_polygon_count = 0
        self._positive_count = 0
        self._negative_count = 0
        self._plugin_activated = is_plugin_activated()
        self._activation_popup_shown = False
        self._segmentation_layer_id = None


        self._progress_timer = QTimer(self)
        self._progress_timer.timeout.connect(self._on_progress_tick)
        self._current_progress = 0
        self._target_progress = 0
        self._install_start_time = None
        self._last_percent = 0
        self._last_percent_time = None
        self._creep_counter = 0

        self._refine_debounce_timer = QTimer(self)
        self._refine_debounce_timer.setSingleShot(True)
        self._refine_debounce_timer.timeout.connect(self._emit_refine_changed)

        self._auto_review_debounce_timer = QTimer(self)
        self._auto_review_debounce_timer.setSingleShot(True)



        self._auto_conf_debounce_timer = QTimer(self)
        self._auto_conf_debounce_timer.setSingleShot(True)
        self._auto_conf_debounce_timer.timeout.connect(self._emit_auto_confidence_changed)




        self._auto_conf_preview_timer = QTimer(self)
        self._auto_conf_preview_timer.setSingleShot(True)
        self._auto_conf_preview_timer.timeout.connect(self._emit_auto_confidence_preview)





        self._auto_prompt_debounce_timer = QTimer(self)
        self._auto_prompt_debounce_timer.setSingleShot(True)
        self._auto_prompt_debounce_timer.timeout.connect(self._emit_auto_prompt_committed)

        self._auto_progress_ratio = 0.0




        self._auto_progress_ease_timer = None
        self._auto_progress_target = 0
        self._auto_progress_shown = 0
        self._auto_progress_dirty = False


        self._auto_billed_tile_total = 0
        self._auto_progress_phase = "grid"


        self._visibility_debounce_timer = QTimer(self)
        self._visibility_debounce_timer.setSingleShot(True)
        self._visibility_debounce_timer.timeout.connect(self._update_ui_state)


        QgsProject.instance().layersAdded.connect(self._on_layers_added)
        QgsProject.instance().layersRemoved.connect(self._on_layers_removed)

        QgsProject.instance().layerTreeRoot().visibilityChanged.connect(
            self._on_layer_visibility_changed)



        self.visibilityChanged.connect(self._on_dock_hidden_reset_engine)


        self._update_full_ui()
