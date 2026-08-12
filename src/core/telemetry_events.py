"""Event-name constants + required-props registry.

Mirror of the server-side event registry: the server owns the definitions and
this module vendors the subset the plugin emits. Never pass a raw string to
telemetry.track(); use a constant below. Add or change an event here and bump
REGISTRY_VERSION together.
"""
from __future__ import annotations

REGISTRY_VERSION = 22

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
# Prompt silently swapped for its English run token (localized / plural / alias).
AUTO_PROMPT_REWRITTEN = "auto_prompt_rewritten"
# Non-blocking prompt-guidance hint shown under the box (exemplar_boost / plan_hint).
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
# Empty-tile scan gate outcome for one run (emitted only when the server
# policy armed the gate; fallback says why an armed gate stood down).
AUTO_GATE_SCAN = "auto_gate_scan"

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
# Post-run review: the free hand edits, their undo, and step navigation.
REVIEW_CORRECT_BOX = "review_correct_box"
REVIEW_CORRECT_UNDO = "review_correct_undo"
REVIEW_STEP = "review_step"
# QGIS digitizing bridge lifecycle (Correct step -> native QGIS editing).
# outcome = opened | committed | rolled_back.
AUTO_EDIT_IN_QGIS = "auto_edit_in_qgis"

# --- Manual ---------------------------------------------------------------
SEGMENTATION_RUN = "segmentation_run"
MANUAL_EXPORT_DONE = "manual_export_done"
MANUAL_SESSION_SUMMARY = "manual_session_summary"
# Confirmed discard of unsaved manual work; context = change_layer | stop.
MANUAL_ABANDONED = "manual_abandoned"
# Which AI answers a Semi-Auto click, and the answer to the notice that stands
# before the first cloud one. Together they are whether the cloud engine is
# being adopted, and how many people refuse it once they read what it sends.
MANUAL_ENGINE_CHOSEN = "manual_engine_chosen"
MANUAL_CLOUD_CONSENT = "manual_cloud_consent"
# One click and its answer: which AI gave it, and how long the user waited.
# Sampled, because a Semi-Auto session is hundreds of clicks.
MANUAL_CLICK_ANSWERED = "manual_click_answered"
# What the account did with the credit for one saved object.
MANUAL_OBJECT_CHARGED = "manual_object_charged"

# --- Monetization ---------------------------------------------------------
PRO_UPSELL_VIEWED = "pro_upsell_viewed"
PRO_UPSELL_CLICKED = "pro_upsell_clicked"
FREE_TASTE_CONSUMED = "free_taste_consumed"
LOW_CREDIT_BANNER_VIEWED = "low_credit_banner_viewed"
DETECT_BLOCKED = "detect_blocked"

# --- Account dialog ---------------------------------------------------------
# Sign out is the clearest churn signal the plugin can send. The dashboard open
# marks the handoff to the browser, where checkout lives. telemetry_opt_changed
# is the only event that may fire while the user is turning telemetry OFF: it
# goes out just before the flag flips, and nothing follows it.
ACCOUNT_SIGNED_OUT = "account_signed_out"
ACCOUNT_DASHBOARD_OPENED = "account_dashboard_opened"
TELEMETRY_OPT_CHANGED = "telemetry_opt_changed"

# --- Library / run history --------------------------------------------------
LIBRARY_OPENED = "library_opened"
HISTORY_SYNCED = "history_synced"
HISTORY_RESTORED = "history_restored"
HISTORY_EXPORTED = "history_exported"
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
    # The telemetry flag flips to off immediately after this one. A batched
    # send would be dropped by the very opt-out it reports.
    TELEMETRY_OPT_CHANGED,
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
    PRO_UPSELL_VIEWED,
    PRO_UPSELL_CLICKED,
    FREE_TASTE_CONSUMED,
    LOW_CREDIT_BANNER_VIEWED,
    DETECT_BLOCKED,
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

# Required non-session properties per event: exactly the props the registry
# marks "required: true", minus the universal ones (product_id, source) and the
# session props (plugin_version, os, device_hash and friends) that every event
# carries anyway.
#
# This is a MIRROR, not a wish list. A prop the plugin always sends but the
# registry marks optional does NOT belong here: listing it makes the table
# disagree with the contract the relay validates against, and check_telemetry.py
# fails the push in both directions. Every event in ALL_EVENTS must have an
# entry, so a new event cannot ship with its properties unchecked.
REQUIRED_PROPS: dict[str, tuple[str, ...]] = {
    # --- Lifecycle --------------------------------------------------------
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
    # --- Automatic funnel -------------------------------------------------
    AUTO_START_CLICKED: (),
    ZONE_DRAWN: (),
    AUTO_ZONE_TOO_LARGE: ("area_km2",),
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
    # --- Review / refine --------------------------------------------------
    REVIEW_OPENED: ("run_id",),
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
    # --- Manual -----------------------------------------------------------
    # sample_rate carries the 1-in-N factor every count has to be multiplied by,
    # so a batch without it cannot be read at all.
    SEGMENTATION_RUN: ("success", "sample_rate"),
    MANUAL_EXPORT_DONE: (),
    MANUAL_SESSION_SUMMARY: (),
    MANUAL_ABANDONED: ("context",),
    MANUAL_ENGINE_CHOSEN: ("engine",),
    MANUAL_CLOUD_CONSENT: ("accepted",),
    # sample_rate carries the 1-in-N factor, same rule as SEGMENTATION_RUN.
    MANUAL_CLICK_ANSWERED: ("engine", "sample_rate"),
    MANUAL_OBJECT_CHARGED: ("outcome",),
    # --- Monetization -----------------------------------------------------
    PRO_UPSELL_VIEWED: (),
    PRO_UPSELL_CLICKED: (),
    FREE_TASTE_CONSUMED: (),
    LOW_CREDIT_BANNER_VIEWED: (),
    DETECT_BLOCKED: (),
    # --- Account dialog ---------------------------------------------------
    ACCOUNT_SIGNED_OUT: (),
    ACCOUNT_DASHBOARD_OPENED: (),
    TELEMETRY_OPT_CHANGED: ("enabled",),
    # --- Library / run history --------------------------------------------
    LIBRARY_OPENED: (),
    HISTORY_SYNCED: (),
    HISTORY_RESTORED: ("run_id",),
    HISTORY_EXPORTED: (),
    HISTORY_FAVORITE_TOGGLED: ("run_id",),
    HISTORY_PAGE_LOADED: (),
    HISTORY_RERUN: ("kind",),
    # --- Errors -----------------------------------------------------------
    # error_code is the stable English exception class name (never a localized
    # dialog title). traceback_hash and module ride along when the error-capture
    # helper produced them, and the registry keeps both optional.
    PLUGIN_ERROR: ("stage", "error_code"),
}
