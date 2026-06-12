"""Vendored telemetry event constants for AI Segmentation.

Single source of truth for event NAMES used by telemetry.py. The website
event registry is the canonical spec; analytics_events.json (next to this
file) is the vendored subset for this product. scripts/check_telemetry.py
fails the build if a raw event-name string is passed to the track function,
or if these constants drift from analytics_events.json.

Rules (mirror the website registry):
- Never invent an event name inline. Add it here AND to the website registry.
- duration_ms is integer milliseconds. Counts use a _count suffix.
- Booleans are real booleans and always present (explicit false).
- Error shape: stage + error_code + error_message (scrubbed, no paths/coords).
- Sampled events carry sample_rate (int N meaning "1 in N kept", 1 = unsampled).
"""

from __future__ import annotations

# Bump when the event set or required-props map below changes. Mirrors the
# "version" field of analytics_events.json.
REGISTRY_VERSION = 1

# --- Event name constants -------------------------------------------------

PLUGIN_OPENED = "plugin_opened"
PLUGIN_ACTIVATED = "plugin_activated"
ACTIVATION_ATTEMPTED = "activation_attempted"
DEPENDENCIES_INSTALLED = "dependencies_installed"
MODEL_DOWNLOAD_COMPLETED = "model_download_completed"
SEGMENTATION_RUN = "segmentation_run"
SESSION_SUMMARY = "session_summary"
EXPORT_COMPLETED = "export_completed"
PRO_PANEL_VIEWED = "pro_panel_viewed"
SUBSCRIBE_LINK_CLICKED = "subscribe_link_clicked"
TRIAL_EXHAUSTED_VIEWED = "trial_exhausted_viewed"
PLUGIN_ERROR = "plugin_error"

# Every event this plugin can emit. check_telemetry.py verifies each is
# present in analytics_events.json.
ALL_EVENTS = frozenset({
    PLUGIN_OPENED,
    PLUGIN_ACTIVATED,
    ACTIVATION_ATTEMPTED,
    DEPENDENCIES_INSTALLED,
    MODEL_DOWNLOAD_COMPLETED,
    SEGMENTATION_RUN,
    SESSION_SUMMARY,
    EXPORT_COMPLETED,
    PRO_PANEL_VIEWED,
    SUBSCRIBE_LINK_CLICKED,
    TRIAL_EXHAUSTED_VIEWED,
    PLUGIN_ERROR,
})

# Required properties beyond the universal/session props every payload carries
# (product_id, plugin_version, os, qgis_version, ...). Keep in sync with the
# "required: true" flags in analytics_events.json; check_telemetry.py asserts
# this map does not contradict the vendored registry.
REQUIRED_PROPS: dict[str, tuple[str, ...]] = {
    PLUGIN_OPENED: (),
    PLUGIN_ACTIVATED: (),
    ACTIVATION_ATTEMPTED: ("success",),
    DEPENDENCIES_INSTALLED: ("success",),
    MODEL_DOWNLOAD_COMPLETED: (),
    SEGMENTATION_RUN: ("success", "sample_rate"),
    SESSION_SUMMARY: (),
    EXPORT_COMPLETED: (),
    PRO_PANEL_VIEWED: (),
    SUBSCRIBE_LINK_CLICKED: (),
    TRIAL_EXHAUSTED_VIEWED: ("is_free_tier",),
    PLUGIN_ERROR: ("stage", "error_code"),
}
