






from __future__ import annotations

REGISTRY_VERSION = 50


PLUGIN_FIRST_OPEN = "plugin_first_open"
PLUGIN_OPENED = "plugin_opened"
PLUGIN_ACTIVATED = "plugin_activated"
MODE_SWITCHED = "mode_switched"
INSTALL_STARTED = "install_started"
INSTALL_COMPLETED = "install_completed"
INSTALL_FAILED = "install_failed"
INSTALL_CANCELLED = "install_cancelled"
MODEL_DOWNLOAD_COMPLETED = "model_download_completed"

PAIRING_STARTED = "pairing_started"
PAIRING_FAILED = "pairing_failed"
PAIRING_CANCELLED = "pairing_cancelled"


FIRST_GENERATION_MILESTONE = "first_generation_milestone"


AUTO_START_CLICKED = "auto_start_clicked"
ZONE_DRAWN = "zone_drawn"
AUTO_ZONE_TOO_LARGE = "auto_zone_too_large"

AUTO_ZONE_FREE_CLIPPED = "auto_zone_free_clipped"

AUTO_ZONE_FREE_CLIP_CHOICE = "auto_zone_free_clip_choice"
AUTO_PROMPT_COMMITTED = "auto_prompt_committed"
AUTO_PROMPT_STEERED = "auto_prompt_steered"

AUTO_PROMPT_REWRITTEN = "auto_prompt_rewritten"

AUTO_PROMPT_HINT_SHOWN = "auto_prompt_hint_shown"
EXEMPLAR_ADDED = "exemplar_added"
EXEMPLAR_REMOVED = "exemplar_removed"
DETAIL_CHANGED = "detail_changed"
AUTO_DETECT_STARTED = "auto_detect_started"
AUTO_DETECT_COMPLETED = "auto_detect_completed"
AUTO_DETECT_FAILED = "auto_detect_failed"
AUTO_DETECT_CANCELLED = "auto_detect_cancelled"
CREDITS_EXHAUSTED = "credits_exhausted"
AUTO_TILES_DEGRADED = "auto_tiles_degraded"
AUTO_ZERO_RESULT = "auto_zero_result"


AUTO_GATE_SCAN = "auto_gate_scan"

AUTO_RUN_SLOW_NOTICE = "auto_run_slow_notice"

AUTO_RUN_LOG = "auto_run_log"


REVIEW_OPENED = "review_opened"
REVIEW_CONFIDENCE_FINAL = "review_confidence_final"


REVIEW_ABANDONED = "review_abandoned"
REVIEW_DISPLAY_MODE = "review_display_mode"
REVIEW_SHAPE_ADJUSTED = "review_shape_adjusted"
REFINE_IN_MANUAL_ENTERED = "refine_in_manual_entered"
REFINE_IN_MANUAL_BACK = "refine_in_manual_back"
AUTO_EXPORT_DONE = "auto_export_done"
AUTO_RETRY_CLICKED = "auto_retry_clicked"
AUTO_EXIT_CLICKED = "auto_exit_clicked"

ZERO_ASSIST_CLICKED = "zero_assist_clicked"
EXEMPLAR_NUDGE_SHOWN = "exemplar_nudge_shown"
EXEMPLAR_NUDGE_CLICKED = "exemplar_nudge_clicked"

TUTORIAL_OPENED = "tutorial_opened"

REVIEW_CORRECT_BOX = "review_correct_box"
REVIEW_CORRECT_UNDO = "review_correct_undo"
REVIEW_STEP = "review_step"


AUTO_EDIT_IN_QGIS = "auto_edit_in_qgis"


SEGMENTATION_RUN = "segmentation_run"
MANUAL_EXPORT_DONE = "manual_export_done"
MANUAL_SESSION_SUMMARY = "manual_session_summary"

MANUAL_ABANDONED = "manual_abandoned"



MANUAL_ENGINE_CHOSEN = "manual_engine_chosen"
MANUAL_CLOUD_CONSENT = "manual_cloud_consent"


MANUAL_CLICK_ANSWERED = "manual_click_answered"

MANUAL_OBJECT_CHARGED = "manual_object_charged"

MANUAL_OBJECTS_WALL_HIT = "manual_objects_wall_hit"


PRO_UPSELL_VIEWED = "pro_upsell_viewed"
PRO_UPSELL_CLICKED = "pro_upsell_clicked"
FREE_TASTE_CONSUMED = "free_taste_consumed"
LOW_CREDIT_BANNER_VIEWED = "low_credit_banner_viewed"
DETECT_BLOCKED = "detect_blocked"



PLUGIN_UPDATE_PROMPT_SHOWN = "plugin_update_prompt_shown"
PLUGIN_UPDATE_PROMPT_CLICKED = "plugin_update_prompt_clicked"
PLUGIN_UPDATE_PROMPT_SUPPRESSED = "plugin_update_prompt_suppressed"






ACCOUNT_SIGNED_OUT = "account_signed_out"
ACCOUNT_DASHBOARD_OPENED = "account_dashboard_opened"
TELEMETRY_OPT_CHANGED = "telemetry_opt_changed"


LIBRARY_OPENED = "library_opened"
HISTORY_SYNCED = "history_synced"
HISTORY_RESTORED = "history_restored"
HISTORY_EXPORTED = "history_exported"
HISTORY_FAVORITE_TOGGLED = "history_favorite_toggled"
HISTORY_PAGE_LOADED = "history_page_loaded"

HISTORY_RERUN = "history_rerun"


PLUGIN_ERROR = "plugin_error"



FLUSH_NOW = frozenset({
    AUTO_DETECT_STARTED, AUTO_DETECT_COMPLETED, AUTO_DETECT_FAILED, AUTO_DETECT_CANCELLED,
    CREDITS_EXHAUSTED, AUTO_ZERO_RESULT, AUTO_TILES_DEGRADED, AUTO_EXPORT_DONE,
    AUTO_RUN_LOG,
    MANUAL_SESSION_SUMMARY, PLUGIN_ERROR, INSTALL_FAILED,
    HISTORY_RESTORED, HISTORY_EXPORTED,


    INSTALL_CANCELLED, PAIRING_FAILED, PAIRING_CANCELLED, FIRST_GENERATION_MILESTONE,


    REVIEW_ABANDONED,


    TELEMETRY_OPT_CHANGED,
})





NO_CONSENT_EVENTS = frozenset({
    PLUGIN_FIRST_OPEN,
    PLUGIN_OPENED,
    PLUGIN_ACTIVATED,
    SEGMENTATION_RUN,
})



ALL_EVENTS = frozenset({
    PLUGIN_FIRST_OPEN,
    PLUGIN_OPENED,
    PLUGIN_ACTIVATED,
    MODE_SWITCHED,
    INSTALL_STARTED,
    INSTALL_COMPLETED,
    INSTALL_FAILED,
    INSTALL_CANCELLED,
    MODEL_DOWNLOAD_COMPLETED,
    PAIRING_STARTED,
    PAIRING_FAILED,
    PAIRING_CANCELLED,
    FIRST_GENERATION_MILESTONE,
    AUTO_START_CLICKED,
    ZONE_DRAWN,
    AUTO_ZONE_TOO_LARGE,
    AUTO_ZONE_FREE_CLIPPED,
    AUTO_ZONE_FREE_CLIP_CHOICE,
    AUTO_PROMPT_COMMITTED,
    AUTO_PROMPT_STEERED,
    AUTO_PROMPT_REWRITTEN,
    AUTO_PROMPT_HINT_SHOWN,
    EXEMPLAR_ADDED,
    EXEMPLAR_REMOVED,
    DETAIL_CHANGED,
    AUTO_DETECT_STARTED,
    AUTO_DETECT_COMPLETED,
    AUTO_DETECT_FAILED,
    AUTO_DETECT_CANCELLED,
    CREDITS_EXHAUSTED,
    AUTO_TILES_DEGRADED,
    AUTO_ZERO_RESULT,
    AUTO_GATE_SCAN,
    AUTO_RUN_SLOW_NOTICE,
    AUTO_RUN_LOG,
    REVIEW_OPENED,
    REVIEW_CONFIDENCE_FINAL,
    REVIEW_ABANDONED,
    REVIEW_DISPLAY_MODE,
    REVIEW_SHAPE_ADJUSTED,
    REFINE_IN_MANUAL_ENTERED,
    REFINE_IN_MANUAL_BACK,
    AUTO_EXPORT_DONE,
    AUTO_RETRY_CLICKED,
    AUTO_EXIT_CLICKED,
    ZERO_ASSIST_CLICKED,
    EXEMPLAR_NUDGE_SHOWN,
    EXEMPLAR_NUDGE_CLICKED,
    TUTORIAL_OPENED,
    REVIEW_CORRECT_BOX,
    REVIEW_CORRECT_UNDO,
    REVIEW_STEP,
    AUTO_EDIT_IN_QGIS,
    SEGMENTATION_RUN,
    MANUAL_EXPORT_DONE,
    MANUAL_SESSION_SUMMARY,
    MANUAL_ABANDONED,
    MANUAL_ENGINE_CHOSEN,
    MANUAL_CLOUD_CONSENT,
    MANUAL_CLICK_ANSWERED,
    MANUAL_OBJECT_CHARGED,
    MANUAL_OBJECTS_WALL_HIT,
    PRO_UPSELL_VIEWED,
    PRO_UPSELL_CLICKED,
    FREE_TASTE_CONSUMED,
    LOW_CREDIT_BANNER_VIEWED,
    DETECT_BLOCKED,
    PLUGIN_UPDATE_PROMPT_SHOWN,
    PLUGIN_UPDATE_PROMPT_CLICKED,
    PLUGIN_UPDATE_PROMPT_SUPPRESSED,
    ACCOUNT_SIGNED_OUT,
    ACCOUNT_DASHBOARD_OPENED,
    TELEMETRY_OPT_CHANGED,
    LIBRARY_OPENED,
    HISTORY_SYNCED,
    HISTORY_RESTORED,
    HISTORY_EXPORTED,
    HISTORY_FAVORITE_TOGGLED,
    HISTORY_PAGE_LOADED,
    HISTORY_RERUN,
    PLUGIN_ERROR,
})











REQUIRED_PROPS: dict[str, tuple[str, ...]] = {

    PLUGIN_FIRST_OPEN: (),
    PLUGIN_OPENED: (),
    PLUGIN_ACTIVATED: (),
    MODE_SWITCHED: ("to_mode",),
    INSTALL_STARTED: (),
    INSTALL_COMPLETED: (),
    INSTALL_FAILED: (),
    INSTALL_CANCELLED: (),
    MODEL_DOWNLOAD_COMPLETED: (),
    PAIRING_STARTED: (),
    PAIRING_FAILED: ("error_code",),
    PAIRING_CANCELLED: (),
    FIRST_GENERATION_MILESTONE: (),

    AUTO_START_CLICKED: (),
    ZONE_DRAWN: (),
    AUTO_ZONE_TOO_LARGE: ("area_km2",),
    AUTO_ZONE_FREE_CLIPPED: ("km2_requested", "km2_processed"),
    AUTO_ZONE_FREE_CLIP_CHOICE: ("choice", "km2_requested"),
    AUTO_PROMPT_COMMITTED: (),
    AUTO_PROMPT_STEERED: (),
    AUTO_PROMPT_REWRITTEN: ("kind",),
    AUTO_PROMPT_HINT_SHOWN: ("kind",),
    EXEMPLAR_ADDED: (),
    EXEMPLAR_REMOVED: (),
    DETAIL_CHANGED: (),
    AUTO_DETECT_STARTED: ("run_id",),
    AUTO_DETECT_COMPLETED: ("run_id",),
    AUTO_DETECT_FAILED: ("run_id",),
    AUTO_DETECT_CANCELLED: ("run_id",),
    CREDITS_EXHAUSTED: ("run_id",),
    AUTO_TILES_DEGRADED: ("run_id",),
    AUTO_ZERO_RESULT: ("run_id",),
    AUTO_GATE_SCAN: ("run_id", "scans", "tiles_skipped"),
    AUTO_RUN_SLOW_NOTICE: ("run_id", "phase"),
    AUTO_RUN_LOG: ("run_id",),

    REVIEW_OPENED: (),
    REVIEW_CONFIDENCE_FINAL: ("run_id",),
    REVIEW_ABANDONED: ("run_id",),
    REVIEW_DISPLAY_MODE: (),
    REVIEW_SHAPE_ADJUSTED: (),
    REFINE_IN_MANUAL_ENTERED: ("run_id",),
    REFINE_IN_MANUAL_BACK: ("run_id",),
    AUTO_EXPORT_DONE: ("run_id",),
    AUTO_RETRY_CLICKED: ("run_id",),
    AUTO_EXIT_CLICKED: (),
    ZERO_ASSIST_CLICKED: (),
    EXEMPLAR_NUDGE_SHOWN: (),
    EXEMPLAR_NUDGE_CLICKED: (),
    TUTORIAL_OPENED: (),
    REVIEW_CORRECT_BOX: ("run_id", "label", "outcome", "objects"),
    REVIEW_CORRECT_UNDO: ("run_id", "kind"),
    REVIEW_STEP: ("run_id", "step"),
    AUTO_EDIT_IN_QGIS: ("run_id", "outcome"),



    SEGMENTATION_RUN: ("success", "sample_rate"),
    MANUAL_EXPORT_DONE: (),
    MANUAL_SESSION_SUMMARY: (),
    MANUAL_ABANDONED: ("context",),
    MANUAL_ENGINE_CHOSEN: ("engine",),
    MANUAL_CLOUD_CONSENT: ("accepted",),

    MANUAL_CLICK_ANSWERED: ("engine", "sample_rate"),
    MANUAL_OBJECT_CHARGED: ("outcome",),
    MANUAL_OBJECTS_WALL_HIT: ("is_subscriber", "in_session"),

    PRO_UPSELL_VIEWED: (),
    PRO_UPSELL_CLICKED: (),
    FREE_TASTE_CONSUMED: (),
    LOW_CREDIT_BANNER_VIEWED: (),
    DETECT_BLOCKED: (),

    PLUGIN_UPDATE_PROMPT_SHOWN: ("offered_version", "trigger"),
    PLUGIN_UPDATE_PROMPT_CLICKED: ("offered_version", "action"),
    PLUGIN_UPDATE_PROMPT_SUPPRESSED: ("served_version", "reason"),

    ACCOUNT_SIGNED_OUT: (),
    ACCOUNT_DASHBOARD_OPENED: (),
    TELEMETRY_OPT_CHANGED: ("enabled",),

    LIBRARY_OPENED: (),
    HISTORY_SYNCED: (),
    HISTORY_RESTORED: ("run_id",),
    HISTORY_EXPORTED: (),
    HISTORY_FAVORITE_TOGGLED: ("run_id",),
    HISTORY_PAGE_LOADED: (),
    HISTORY_RERUN: ("kind",),




    PLUGIN_ERROR: ("stage", "error_code"),
}
