from __future__ import annotations


from qgis.core import QgsProject
from qgis.PyQt.QtCore import Qt, QTimer, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFrame,
    QScrollArea,
    QDoubleSpinBox,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


from ..core.activation_manager import (
    is_plugin_activated,
)
from ..core.i18n import tr
from .dock.build import DockBuildMixin
from .dock.auto_build import DockAutoBuildMixin
from .dock.auto_review_build import DockAutoReviewBuildMixin
from .dock.refine import DockRefineMixin
from .dock.handoff import DockHandoffMixin
from .dock.about import DockAboutMixin
from .dock.activation_state import DockActivationMixin
from .dock.auto_state import DockAutoStateMixin
from .dock.state import DockStateMixin
from .dock.styles import (  # noqa: F401 - re-exported for other modules
    BRAND_BLUE,
    BRAND_BLUE_HOVER,
    BRAND_DISABLED,
    BRAND_GRAY,
    BRAND_GRAY_HOVER,
    BRAND_GREEN,
    BRAND_GREEN_TEXT,
    BRAND_RED,
    BRAND_RED_HOVER,
    BTN_GREEN,
    BTN_GREEN_DISABLED,
    BTN_GREEN_HOVER,
    DISABLED_TEXT,
    ERROR_TEXT,
    SUCCESS_TEXT,
    _BTN_BLUE,
    _BTN_BLUE_AUTH,
    _BTN_BLUE_PRIMARY,
    _BTN_EXPORT_DISABLED,
    _BTN_EXPORT_READY,
    _BTN_GHOST,
    _BTN_GRAY,
    _BTN_GREEN,
    _BTN_GREEN_AUTH,
    _BTN_PAIR_CANCEL,
    _BTN_PAIR_NEUTRAL,
    _BTN_RED,
    _CARD_QSS,
    _FOOTER_CTA_BTN_STYLE,
    _FOOTER_ICON_BTN_STYLE,
    _FOOTER_MENU_STYLE,
    _HELP_ICON_BTN_STYLE,
    _REFINE_COLLAPSED_HEIGHT,
    _REVIEW_CONF_MAX,
    _REVIEW_CONF_MIN,
    _REVIEW_CONF_SPIN_MIN,
    _REVIEW_CONF_STEP,
    _SLIDER_QSS,
    _snap_review_conf,
)
from .dock.widgets import (  # noqa: F401 - re-exported for other modules
    Mode,
    _FooterIconButton,
    _ModeSwitch,
    _Spinner,
    _WheelGuard,
    _ZoneGestureGlyph,
)


class AISegmentationDockWidget(
    DockBuildMixin,
    DockAutoBuildMixin,
    DockAutoReviewBuildMixin,
    DockRefineMixin,
    DockHandoffMixin,
    DockAboutMixin,
    DockActivationMixin,
    DockAutoStateMixin,
    DockStateMixin,
    QDockWidget,
):

    install_requested = pyqtSignal()
    cancel_install_requested = pyqtSignal()
    start_segmentation_requested = pyqtSignal(object)
    undo_requested = pyqtSignal()
    save_polygon_requested = pyqtSignal()
    settings_clicked = pyqtSignal()
    export_layer_requested = pyqtSignal()
    stop_segmentation_requested = pyqtSignal()
    pairing_requested = pyqtSignal(str)        # one-click connect: emits the minted pairing code
    pairing_cancel_requested = pyqtSignal(str)  # user cancelled the browser handoff (emits the code)
    # simplify, smooth, expand, fill_holes, right_angles
    # (min_area is auto-computed server-side and no longer in the UI)
    refine_settings_changed = pyqtSignal(int, int, int, bool, bool)
    # Min/Max size window in ground m2 (0 = off). Emitted right BEFORE
    # refine_settings_changed on the same debounce tick (store-only handler).
    size_filter_changed = pyqtSignal(float, float)
    mode_changed = pyqtSignal(object)          # emits Mode value
    auto_detect_requested = pyqtSignal()       # user clicked Detect in Automatic mode
    auto_library_requested = pyqtSignal()      # user clicked Library (open prompt gallery)
    auto_demo_requested = pyqtSignal()         # first-run hero: load the demo basemap and select it (user runs flow)
    history_rerun_requested = pyqtSignal(dict)  # Recent card "Run again here" (stored run entry)
    history_reuse_prompt_requested = pyqtSignal(str)  # Recent card "Same object, new zone" (prompt token)
    zone_draw_requested = pyqtSignal()         # zone drawing should (re)start on the canvas
    auto_detail_changed = pyqtSignal(int)      # detail slider moved (value = grid side n)
    auto_prompt_committed = pyqtSignal(str)    # object class settled (debounced) -> re-seed detail
    auto_step_changed = pyqtSignal(int)        # Automatic flow switched to this step (0-2)
    auto_refine_changed = pyqtSignal()         # any auto refine control changed
    auto_review_confidence_changed = pyqtSignal(int)  # review confidence slider released (percent)
    auto_review_confidence_preview = pyqtSignal(int)   # live, while dragging (percent) - fast preview
    auto_display_mode_changed = pyqtSignal(str)        # review colours: 'normal' / 'confidence' / 'random'
    auto_show_tiles_changed = pyqtSignal(bool)  # review "Show tiles" debug checkbox toggled
    auto_export_requested = pyqtSignal()       # user clicked Export to layer in review panel
    auto_retry_requested = pyqtSignal()        # user clicked Adjust & run again in review (keep inputs)
    auto_review_exit_requested = pyqtSignal()  # user clicked Exit in review (Save/Discard dialog)
    auto_refine_in_manual_requested = pyqtSignal()  # hand the reviewed detections to Manual mode
    back_to_review_requested = pyqtSignal()    # return from a Manual refine handoff to Auto review
    handoff_edit_requested = pyqtSignal()      # state card Edit shape (single selected detection)
    handoff_delete_requested = pyqtSignal()    # state card Remove (selection or the open edit)
    auto_exit_requested = pyqtSignal()         # user clicked Exit on the prompt step
    auto_add_exemplar_requested = pyqtSignal(int)   # draw an example (1 = positive, 0 = exclude)
    auto_exemplar_retry_requested = pyqtSignal()    # exemplar nudge: retry then arm the example draw
    auto_exemplar_remove_requested = pyqtSignal(str)  # user clicked x on an exemplar chip (id)
    auto_merge_override_requested = pyqtSignal()  # review: re-group the auto count-vs-map decision
    auto_zero_assist_clicked = pyqtSignal(str, str)  # zero-result rescue chip (kind, to_prompt)
    auto_escape_pressed = pyqtSignal()         # Escape in the Automatic flow (exit / cancel draw)
    auto_enter_pressed = pyqtSignal()          # Enter in the Automatic flow (detect / export review)

    # Visual exemplars ("draw one example, find all" + exclude boxes) are the
    # PRIMARY input now: a drawn example is the cloud model's biggest quality lever, so the
    # reference panel leads step 2 and a text/gallery prompt is the secondary
    # option (Detect enables on either). Note the open caveat: the cloud model's box exemplars
    # force single-image mode, which starves resolution on LARGE zones (a small
    # example shrinks below the model's useful size). Phase 1 is scoped to zones
    # small enough to stay sharp (see _refresh_reference_guard); the
    # composite-per-tile path (stamp the crop into each tile) lifts that limit.
    _EXEMPLARS_ENABLED = True

    def __init__(self, parent=None):
        super().__init__(tr("AI Segmentation by TerraLab"), parent)
        # A stable objectName lets QGIS persist and restore the dock's
        # open/closed state and position across sessions (same as AI Edit).
        self.setObjectName("AISegmentationDockWidget")

        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.setMinimumWidth(260)

        self._setup_title_bar()

        # Initialize state variables that are needed during UI setup
        self._refine_expanded = True  # Refine panel expanded by default

        # --- Pro / mode state ---
        # The dock ALWAYS opens on Automatic, for everyone,
        # every time (no persisted last-mode restore). The internal mode
        # values stay "interactive"/"automatic" (MCP API stability).
        self._mode: Mode = Mode.AUTOMATIC
        self._auto_credits: int | None = None
        self._auto_credits_total: int | None = None
        self._auto_free_left: int | None = None
        self._auto_is_subscriber: bool = False
        # Subscription period end (ISO date) from the last usage fetch.
        self._auto_reset_date: str = ""
        self._auto_run_active: bool = False
        self._auto_zone_too_large: bool = False
        self._auto_zone_is_set: bool = False
        self._auto_review_active: bool = False
        # True from the instant Cancel is pressed until the run winds down: keeps
        # the "keeping what's found" note on the progress card while the worker
        # drains its in-flight tiles, instead of the send/queue line.
        self._auto_cancelling: bool = False
        # Live "waking up the AI" feedback for the pre-first-tile window: a cold
        # GPU can take up to ~a minute to answer, and with a static 0% bar that
        # reads as a hang. A 1s QTimer (created lazily) animates the bar
        # (indeterminate) and evolves the label while no tile has landed yet.
        # _auto_warming_since is the monotonic run-start used for the elapsed
        # readout; _auto_queue_position mirrors the last server queue answer
        # (0 = flowing, -1 = busy/warming with no place known, >=1 = real spot).
        self._auto_warmup_timer = None
        self._auto_warming_since: float | None = None
        self._auto_queue_position: int = 0
        self._auto_queue_eta: int = 0
        # Last per-zone credit estimate (tile count) and whether it exceeds the
        # known balance. When it does, Detect is hard-blocked: a run may never
        # launch under-funded. None = no estimate yet.
        self._auto_est_credits: int | None = None
        self._auto_insufficient_credits: bool = False
        # True while a Manual session is refining the Automatic results: the
        # manual Export-to-layer / Stop buttons are hidden so "Back to review"
        # (then Finish) stays the single, unambiguous commit path.
        self._refine_handoff: bool = False
        # Total detections seeded into a Refine-in-Manual handoff, and the kept
        # (validated) count. Both drive the compact instructions hint ("N of M
        # detections kept - click a blue detection to edit it").
        self._handoff_seed_total: int = 0
        self._handoff_kept: int = 0
        # How many detections are currently SELECTED in the handoff (selection-
        # first review): drives the "N selected" guidance card state.
        self._handoff_selected: int = 0
        # True while a detection is OPEN for editing in the handoff (the state
        # card swaps to the editing actions).
        self._handoff_editing: bool = False
        # True while the local model is still loading after a Refine click:
        # the handoff header says so and the state card stays hidden.
        self._handoff_preparing: bool = False
        # Count of positive ("find similar") visual exemplars currently set, so
        # Detect can enable on exemplars alone (no text prompt required).
        self._auto_positive_exemplars: int = 0
        # True once the user clicks "Start Automatic Segmentation": the layer
        # is locked and the flow moves to the draw-zone step. Reset to False
        # only by Exit (back to the Start step with the layer editable again).
        self._auto_started: bool = False

        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setSpacing(8)
        self.main_layout.setContentsMargins(8, 8, 8, 8)

        self._setup_ui()

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.main_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidget(scroll_area)

        # Stop the mouse wheel from changing combo/spin values while the user is
        # just scrolling the panel (the "what to detect" text used to flip). One
        # guard, swept across every value widget in the panel.
        self._wheel_guard = _WheelGuard(scroll_area.viewport(), self)
        # QSlider included: hovering the confidence/detail slider while scrolling
        # the panel was silently dragging its value. With the guard, the wheel
        # only moves a slider once it is focused (clicked), else it scrolls the panel.
        for _w in self.main_widget.findChildren((QComboBox, QSpinBox, QDoubleSpinBox, QSlider)):
            _w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            _w.installEventFilter(self._wheel_guard)

        self._dependencies_ok = False
        self._checkpoint_ok = False
        # True while the Manual install/download UI is the interactive content
        # (install needed, running, failed, or model download pending). Drives
        # the setup section's visibility from real state instead of the sticky
        # setup_group.isVisible(), which an Automatic round trip would wipe -
        # leaving the Manual page empty over a still-running background install.
        self._setup_section_wanted = False
        self._segmentation_active = False
        self._has_mask = False
        self._saved_polygon_count = 0
        # True once a Manual session has candidate layers to append to, so the
        # export destination row is shown; when False the Manual export flow
        # looks exactly like before (single "Export to a layer" button).
        self._has_export_candidates = False
        self._positive_count = 0
        self._negative_count = 0
        self._plugin_activated = is_plugin_activated()
        self._activation_popup_shown = False  # Track if popup was shown
        self._segmentation_layer_id = None  # Track which layer we're segmenting
        # Note: _refine_expanded is initialized before _setup_ui() call

        # Smooth progress animation timer for long-running installs
        self._progress_timer = QTimer(self)
        self._progress_timer.timeout.connect(self._on_progress_tick)
        self._current_progress = 0
        self._target_progress = 0
        self._install_start_time = None
        self._last_percent = 0
        self._last_percent_time = None
        self._creep_counter = 0
        # Debounce timer for refinement sliders
        self._refine_debounce_timer = QTimer(self)
        self._refine_debounce_timer.setSingleShot(True)
        self._refine_debounce_timer.timeout.connect(self._emit_refine_changed)
        # Debounce timer for auto review refine controls (150 ms, same pattern).
        self._auto_review_debounce_timer = QTimer(self)
        self._auto_review_debounce_timer.setSingleShot(True)
        # Debounce timer for the review confidence slider/spinbox. The re-filter
        # re-merges every stored detection, so coalesce rapid drags/keystrokes
        # into one rebuild when the user settles instead of one per change.
        self._auto_conf_debounce_timer = QTimer(self)
        self._auto_conf_debounce_timer.setSingleShot(True)
        self._auto_conf_debounce_timer.timeout.connect(self._emit_auto_confidence_changed)
        # Live preview WHILE dragging the confidence slider: a very short debounce
        # that re-shows the filtered detections cheaply (no heavy re-merge/refine)
        # so the map tracks the handle in real time. The accurate rebuild still
        # runs on release via _auto_conf_debounce_timer.
        self._auto_conf_preview_timer = QTimer(self)
        self._auto_conf_preview_timer.setSingleShot(True)
        self._auto_conf_preview_timer.timeout.connect(self._emit_auto_confidence_preview)

        # Debounce timer for the object-class prompt: when the typed object
        # settles (~500 ms after the last keystroke) emit auto_prompt_committed
        # so the plugin can re-seed the object-aware detail default without the
        # slider jittering on every character.
        self._auto_prompt_debounce_timer = QTimer(self)
        self._auto_prompt_debounce_timer.setSingleShot(True)
        self._auto_prompt_debounce_timer.timeout.connect(self._emit_auto_prompt_committed)

        self._auto_progress_ratio = 0.0

        # Debounce timer for layer visibility changes (fires per-node in groups)
        self._visibility_debounce_timer = QTimer(self)
        self._visibility_debounce_timer.setSingleShot(True)
        self._visibility_debounce_timer.timeout.connect(self._update_ui_state)

        # Connect to project layer signals for dynamic updates
        QgsProject.instance().layersAdded.connect(self._on_layers_added)
        QgsProject.instance().layersRemoved.connect(self._on_layers_removed)
        # Refresh layer dropdown when layer visibility is toggled (debounced)
        QgsProject.instance().layerTreeRoot().visibilityChanged.connect(
            self._on_layer_visibility_changed)

        # Update UI state
        self._update_full_ui()
