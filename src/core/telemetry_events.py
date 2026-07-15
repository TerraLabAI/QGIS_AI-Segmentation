"""Event-name constants + required-props registry (mirror of the website registry).

Single source of truth lives in the website repo; the ai-segmentation subset is
vendored here (and in analytics_events.json) the same way prompt presets are
mirrored. Never pass a raw string to telemetry.track(); use a constant below.
Run scripts/check_telemetry.py to verify this module and the website registry
stay in sync. Keep in sync with terralab-website analytics_events.json; bump
REGISTRY_VERSION together.
"""
from __future__ import annotations

REGISTRY_VERSION = 10

# --- Lifecycle ------------------------------------------------------------
PLUGIN_FIRST_OPEN = "plugin_first_open"
PLUGIN_OPENED = "plugin_opened"
PLUGIN_ACTIVATED = "plugin_activated"
MODE_SWITCHED = "mode_switched"
INSTALL_STARTED = "install_started"
INSTALL_COMPLETED = "install_completed"
INSTALL_FAILED = "install_failed"
INSTALL_CANCELLED = "install_cancelled"
MODEL_DOWNLOAD_COMPLETED = "model_download_completed"
# Browser sign-in (pairing) lifecycle; success is plugin_activated.
PAIRING_STARTED = "pairing_started"
PAIRING_FAILED = "pairing_failed"
PAIRING_CANCELLED = "pairing_cancelled"
# First successful export ever on this machine (one-shot, mode = auto|manual).
# Shares the cross-product first-value event name used by the ecosystem.
FIRST_GENERATION_MILESTONE = "first_generation_milestone"

# --- Automatic funnel -----------------------------------------------------
AUTO_START_CLICKED = "auto_start_clicked"
ZONE_DRAWN = "zone_drawn"
AUTO_ZONE_TOO_LARGE = "auto_zone_too_large"
AUTO_PROMPT_COMMITTED = "auto_prompt_committed"
AUTO_PROMPT_STEERED = "auto_prompt_steered"
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

# --- Review / refine ------------------------------------------------------
REVIEW_OPENED = "review_opened"
REVIEW_CONFIDENCE_FINAL = "review_confidence_final"
# Left the review without clicking Finish (exit_path says how); passive
# leaves still autosave (auto_export_done carries autosave=true).
REVIEW_ABANDONED = "review_abandoned"
REVIEW_DISPLAY_MODE = "review_display_mode"
REVIEW_SHAPE_ADJUSTED = "review_shape_adjusted"
REFINE_IN_MANUAL_ENTERED = "refine_in_manual_entered"
REFINE_IN_MANUAL_BACK = "refine_in_manual_back"
AUTO_EXPORT_DONE = "auto_export_done"
AUTO_RETRY_CLICKED = "auto_retry_clicked"
AUTO_EXIT_CLICKED = "auto_exit_clicked"
# Zero-result assist + exemplar nudge.
ZERO_ASSIST_CLICKED = "zero_assist_clicked"
EXEMPLAR_NUDGE_SHOWN = "exemplar_nudge_shown"
EXEMPLAR_NUDGE_CLICKED = "exemplar_nudge_clicked"
# Tutorial-discovery opens; source = touchpoint id.
TUTORIAL_OPENED = "tutorial_opened"

# --- Manual ---------------------------------------------------------------
SEGMENTATION_RUN = "segmentation_run"
MANUAL_EXPORT_DONE = "manual_export_done"
MANUAL_SESSION_SUMMARY = "manual_session_summary"
# Confirmed discard of unsaved manual work; context = change_layer | stop.
MANUAL_ABANDONED = "manual_abandoned"

# --- Monetization ---------------------------------------------------------
PRO_UPSELL_VIEWED = "pro_upsell_viewed"
PRO_UPSELL_CLICKED = "pro_upsell_clicked"
FREE_TASTE_CONSUMED = "free_taste_consumed"
LOW_CREDIT_BANNER_VIEWED = "low_credit_banner_viewed"
DETECT_BLOCKED = "detect_blocked"

# --- Library / run history --------------------------------------------------
LIBRARY_OPENED = "library_opened"
HISTORY_SYNCED = "history_synced"
HISTORY_RESTORED = "history_restored"
HISTORY_EXPORTED = "history_exported"
HISTORY_DELETED = "history_deleted"
HISTORY_UNDELETED = "history_undeleted"
HISTORY_FAVORITE_TOGGLED = "history_favorite_toggled"
HISTORY_PAGE_LOADED = "history_page_loaded"
# One-click re-run from a Recent card: kind = "same_zone" | "new_zone".
HISTORY_RERUN = "history_rerun"

# --- Errors ---------------------------------------------------------------
PLUGIN_ERROR = "plugin_error"

# Events flushed immediately (paid-funnel milestones + failures): they must not
# wait in the batch, so a crash right after does not lose the record.
FLUSH_NOW = frozenset({
    AUTO_DETECT_STARTED, AUTO_DETECT_COMPLETED, AUTO_DETECT_FAILED, AUTO_DETECT_CANCELLED,
    CREDITS_EXHAUSTED, AUTO_ZERO_RESULT, AUTO_TILES_DEGRADED, AUTO_EXPORT_DONE,
    MANUAL_SESSION_SUMMARY, PLUGIN_ERROR, INSTALL_FAILED,
    HISTORY_RESTORED, HISTORY_EXPORTED,
    # The session often ends right after these (quit after cancelling, browser
    # handoff after pairing): the batch would die with it.
    INSTALL_CANCELLED, PAIRING_FAILED, PAIRING_CANCELLED, FIRST_GENERATION_MILESTONE,
    # Leaving the review (Discard && exit, unload) is often the last act of the
    # session; without an immediate flush the abandonment signal dies with it.
    REVIEW_ABANDONED,
})

# Lifecycle events with no user-generated content; they ship as long as the
# plugin is activated (no ToS gate beyond the global opt-out). Mirrors the
# server relay allow-list.
NO_CONSENT_EVENTS = frozenset({
    PLUGIN_FIRST_OPEN,
    PLUGIN_OPENED,
    PLUGIN_ACTIVATED,
    SEGMENTATION_RUN,
})

# Every event this plugin may emit. check_telemetry.py verifies this matches the
# vendored website registry subset exactly.
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
    AUTO_PROMPT_COMMITTED,
    AUTO_PROMPT_STEERED,
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
    SEGMENTATION_RUN,
    MANUAL_EXPORT_DONE,
    MANUAL_SESSION_SUMMARY,
    MANUAL_ABANDONED,
    PRO_UPSELL_VIEWED,
    PRO_UPSELL_CLICKED,
    FREE_TASTE_CONSUMED,
    LOW_CREDIT_BANNER_VIEWED,
    DETECT_BLOCKED,
    LIBRARY_OPENED,
    HISTORY_SYNCED,
    HISTORY_RESTORED,
    HISTORY_EXPORTED,
    HISTORY_DELETED,
    HISTORY_UNDELETED,
    HISTORY_FAVORITE_TOGGLED,
    HISTORY_PAGE_LOADED,
    HISTORY_RERUN,
    PLUGIN_ERROR,
})

# Required non-session properties per event (session/universal props such as
# product_id, plugin_version, os, device_hash are added by session props).
# Mirrors the "required: true" fields of the vendored website registry subset.
REQUIRED_PROPS: dict[str, tuple[str, ...]] = {
    PLUGIN_FIRST_OPEN: (),
    PLUGIN_OPENED: (),
    PLUGIN_ACTIVATED: (),
    MODE_SWITCHED: ("to_mode",),
    INSTALL_STARTED: (),
    INSTALL_COMPLETED: ("duration_ms",),
    INSTALL_FAILED: ("error_class",),
    INSTALL_CANCELLED: (),
    MODEL_DOWNLOAD_COMPLETED: ("model",),
    PAIRING_STARTED: (),
    PAIRING_FAILED: ("error_code",),
    PAIRING_CANCELLED: (),
    FIRST_GENERATION_MILESTONE: ("mode",),
    AUTO_START_CLICKED: ("layer_kind",),
    ZONE_DRAWN: ("vertices", "area_km2"),
    AUTO_ZONE_TOO_LARGE: ("area_km2",),
    AUTO_PROMPT_COMMITTED: ("prompt",),
    AUTO_PROMPT_STEERED: (),
    EXEMPLAR_ADDED: ("count_after",),
    EXEMPLAR_REMOVED: ("count_after",),
    DETAIL_CHANGED: ("detail", "tiles", "source"),
    AUTO_DETECT_STARTED: (
        "run_id", "tiles", "object_class", "detail", "est_credits", "is_free_tier",
    ),
    AUTO_DETECT_COMPLETED: (
        "run_id", "duration_ms", "tiles_done", "instances_found", "zero_at_default",
    ),
    AUTO_DETECT_FAILED: ("run_id", "error_class"),
    AUTO_DETECT_CANCELLED: ("run_id", "tiles_done"),
    CREDITS_EXHAUSTED: ("run_id",),
    AUTO_TILES_DEGRADED: ("run_id",),
    AUTO_ZERO_RESULT: ("run_id", "object_class"),
    REVIEW_OPENED: ("run_id", "instances_found"),
    REVIEW_CONFIDENCE_FINAL: ("run_id", "final_pct"),
    REVIEW_ABANDONED: ("run_id",),
    REVIEW_DISPLAY_MODE: ("mode",),
    REVIEW_SHAPE_ADJUSTED: ("control",),
    REFINE_IN_MANUAL_ENTERED: ("run_id",),
    REFINE_IN_MANUAL_BACK: ("run_id",),
    AUTO_EXPORT_DONE: ("run_id", "exported_count"),
    AUTO_RETRY_CLICKED: ("run_id",),
    AUTO_EXIT_CLICKED: ("from_step",),
    ZERO_ASSIST_CLICKED: ("kind",),
    EXEMPLAR_NUDGE_SHOWN: ("run_id", "object_class"),
    EXEMPLAR_NUDGE_CLICKED: ("run_id", "object_class"),
    TUTORIAL_OPENED: ("source",),
    SEGMENTATION_RUN: ("success",),
    MANUAL_EXPORT_DONE: ("polygon_count",),
    MANUAL_SESSION_SUMMARY: ("saves",),
    MANUAL_ABANDONED: ("context",),
    PRO_UPSELL_VIEWED: ("trigger",),
    PRO_UPSELL_CLICKED: ("source",),
    FREE_TASTE_CONSUMED: ("remaining",),
    LOW_CREDIT_BANNER_VIEWED: ("remaining",),
    DETECT_BLOCKED: ("reason",),
    LIBRARY_OPENED: ("tab",),
    HISTORY_SYNCED: ("runs",),
    HISTORY_RESTORED: ("run_id", "tiles", "objects"),
    HISTORY_EXPORTED: ("format", "objects"),
    HISTORY_DELETED: ("run_id",),
    HISTORY_UNDELETED: ("run_id",),
    HISTORY_FAVORITE_TOGGLED: ("run_id", "is_favorite"),
    HISTORY_PAGE_LOADED: ("page",),
    HISTORY_RERUN: ("kind",),
    # error_code is the stable English exception class name (never a localized
    # dialog title). Additive OPTIONAL props (present when the error-capture
    # helper produced them, absent otherwise; server + registry treat them as
    # optional): traceback_hash (short sha of the path-scrubbed traceback, for
    # grouping recurrences of the same crash) and module (the source module the
    # exception was caught in).
    PLUGIN_ERROR: ("stage", "error_code"),
}
