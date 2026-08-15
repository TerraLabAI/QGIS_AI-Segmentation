"""Telemetry for everything outside an Automatic run.

Opening the plugin, installing, pairing, switching mode, the account
dialog, Manual mode, the Pro upsells and the Library.

Wrappers only: each one names its event's properties in one place so call sites
stay readable. The transport (consent, batching, scrubbing, the POST) is in
telemetry.py, and every function here ends in its track() call.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QSettings

from . import telemetry_events as ev
from .telemetry import scrub_payload_value, track

_FIRST_OPEN_KEY = "AI_Segmentation/first_open_sent"


def track_plugin_first_open() -> None:
    """Fire exactly once, ever, the first time the dock is opened on this machine.

    Guarded by a persistent QSettings flag so an install -> first-open ->
    activation funnel has a clean entry marker. No consent needed (a lifecycle
    ping with no user content); parks pre-auth like plugin_opened."""
    try:
        settings = QSettings()
        if bool(settings.value(_FIRST_OPEN_KEY, False, type=bool)):
            return
        settings.setValue(_FIRST_OPEN_KEY, True)
    except Exception:  # nosec B110 - never break the open path on a settings error
        return
    track(ev.PLUGIN_FIRST_OPEN)


def track_plugin_opened() -> None:
    """Fire once per dock-open."""
    track(ev.PLUGIN_OPENED)


def track_plugin_activated(duration_ms: int | None = None) -> None:
    """Fire when the activation key is validated. duration_ms is the elapsed
    browser sign-in wait when the key came from the pairing flow."""
    track(ev.PLUGIN_ACTIVATED, {"duration_ms": duration_ms})


def track_pairing_started() -> None:
    """Fire when the browser sign-in poll starts (success = plugin_activated)."""
    track(ev.PAIRING_STARTED)


def track_pairing_failed(error_code: str, duration_ms: int | None = None) -> None:
    """error_code is the server rejection code, or "timeout" when the poll
    window expired (a stable machine string, never a localized message)."""
    track(ev.PAIRING_FAILED, {"error_code": error_code, "duration_ms": duration_ms})


def track_pairing_cancelled(duration_ms: int | None = None) -> None:
    track(ev.PAIRING_CANCELLED, {"duration_ms": duration_ms})


def track_mode_switched(to_mode: str, had_unsaved_manual: bool = False,
                        auto_step: int | None = None) -> None:
    """Fire when the user changes the Manual / Automatic toggle."""
    track(ev.MODE_SWITCHED, {
        "to_mode": to_mode,
        "had_unsaved_manual": bool(had_unsaved_manual),
        "auto_step": auto_step,
    })


# entry: "semi_auto_modal" when the install came through the window that holds
# QGIS until it ends, "background" for the lighter one Automatic asks for.
def track_install_started(entry: str = "background") -> None:
    track(ev.INSTALL_STARTED, {"entry": entry})


def track_install_completed(duration_ms: int | None = None,
                            python_minor: int | None = None,
                            retry_count: int | None = None,
                            entry: str = "background") -> None:
    track(ev.INSTALL_COMPLETED, {
        "duration_ms": duration_ms,
        "python_minor": python_minor,
        "retry_count": retry_count,
        "entry": entry,
    })


def track_install_failed(error_class: str, duration_ms: int | None = None,
                         python_minor: int | None = None,
                         retry_count: int | None = None,
                         detail: str | None = None,
                         entry: str = "background") -> None:
    """detail: what the installer actually said, scrubbed and truncated.

    Without it an error_class of "installation_failed" means only that no
    classifier matched, which is the case that most needs a reason: an
    unmatched failure carries nothing to act on, and it is not the rare one.
    Scrubbed like every other string leaving the machine, then cut to 300: the
    useful part of a pip or venv error is its first line, and the server
    truncates to 200 anyway.
    """
    props = {
        "error_class": error_class,
        "duration_ms": duration_ms,
        "python_minor": python_minor,
        "retry_count": retry_count,
        "entry": entry,
    }
    if detail:
        props["error_detail"] = scrub_payload_value(detail[:300])
    track(ev.INSTALL_FAILED, props)


def track_install_cancelled(duration_ms: int | None = None,
                            entry: str = "background") -> None:
    """Fire when the user cancels the dependency install mid-way."""
    track(ev.INSTALL_CANCELLED, {"duration_ms": duration_ms, "entry": entry})


def track_model_download_completed(model: str, duration_ms: int | None = None) -> None:
    """model is "sam1" or "sam2" ONLY (never a checkpoint URL or file name)."""
    track(ev.MODEL_DOWNLOAD_COMPLETED, {"model": model, "duration_ms": duration_ms})


_segmentation_run_sent_this_session = False


def track_segmentation_run(success: bool, duration_ms: int | None = None) -> None:
    """Fire when a manual segmentation run completes (or fails).

    Success runs are sampled 1-in-10 (power users click hundreds of times per
    session); the first run per session is always sent (sample_rate 1). Failures
    are ALWAYS sent unsampled (sample_rate 1) so the failure rate is real."""
    global _segmentation_run_sent_this_session
    import random

    if not success:
        # Failures are never sampled: the failure signal must be complete.
        track(ev.SEGMENTATION_RUN, {
            "success": False, "duration_ms": duration_ms, "sample_rate": 1,
        })
        return

    if _segmentation_run_sent_this_session:
        if random.random() >= 0.1:  # nosec B311 - sampling, not crypto
            return
        sample_rate = 10
    else:
        _segmentation_run_sent_this_session = True
        sample_rate = 1
    track(ev.SEGMENTATION_RUN, {
        "success": True, "duration_ms": duration_ms, "sample_rate": sample_rate,
    })


def track_manual_export_done(
    polygon_count: int, refine_used: bool, destination: str = "new"
) -> None:
    # destination: "new" (a fresh layer) | "append" (added to an existing
    # layer). Additive optional property; older servers ignore it.
    track(ev.MANUAL_EXPORT_DONE, {
        "polygon_count": polygon_count,
        "refine_used": bool(refine_used),
        "destination": destination,
    })


def track_manual_session_summary(saves: int, undos: int,
                                 duration_ms: int | None = None) -> None:
    track(ev.MANUAL_SESSION_SUMMARY, {
        "saves": saves,
        "undos": undos,
        "duration_ms": duration_ms,
    })


def track_manual_engine_chosen(engine: str, local_installed: bool = False,
                               from_install_gate: bool = False) -> None:
    """Which AI a Semi-Auto user picked. engine: "cloud" | "offline".

    Unsampled and rare by nature: the card is answered once per install, plus
    whenever somebody reopens it with Switch.
    """
    track(ev.MANUAL_ENGINE_CHOSEN, {
        "engine": engine,
        "local_installed": bool(local_installed),
        "from_install_gate": bool(from_install_gate),
    })


def track_manual_cloud_consent(accepted: bool) -> None:
    """The answer to the notice shown before any imagery leaves the machine.

    DORMANT (2026-08-15): nothing calls it. The notice is a line in the engine
    card now, not a modal, so there is no answer to count. Kept with its event
    constant so the registry and the historical series stay readable.
    """
    track(ev.MANUAL_CLOUD_CONSENT, {"accepted": bool(accepted)})


_click_answered_sent_this_session = False


def track_manual_click_answered(engine: str, duration_ms: int | None = None,
                                used_fallback: bool = False,
                                is_correct: bool = False) -> None:
    """One Semi-Auto click came back with a shape. engine: "cloud" | "local".

    Sampled the same way as track_segmentation_run, and for the same reason: a
    session is hundreds of clicks. The first click of a session always goes out
    (sample_rate 1), the rest 1-in-10.

    Safe from any thread: it only tracks, it never flushes.
    """
    global _click_answered_sent_this_session
    import random

    if _click_answered_sent_this_session:
        if random.random() >= 0.1:  # nosec B311 - sampling, not crypto
            return
        sample_rate = 10
    else:
        _click_answered_sent_this_session = True
        sample_rate = 1
    track(ev.MANUAL_CLICK_ANSWERED, {
        "engine": engine,
        "duration_ms": duration_ms,
        "used_fallback": bool(used_fallback),
        "is_correct": bool(is_correct),
        "sample_rate": sample_rate,
    })


def track_manual_object_charged(outcome: str, objects_charged: int | None = None,
                                error_code: str = "") -> None:
    """What the account did with the credit for one saved object.

    outcome: "charged" | "refused" | "exhausted". Unsampled: a saved object is
    rare next to a click, and this is the money line.
    """
    track(ev.MANUAL_OBJECT_CHARGED, {
        "outcome": outcome,
        "objects_charged": objects_charged,
        "error_code": error_code or None,
    })


def track_manual_abandoned(context: str, polygon_count: int) -> None:
    """Fire when the user CONFIRMS discarding unsaved manual work.
    context: "change_layer" | "stop"."""
    track(ev.MANUAL_ABANDONED, {
        "context": context,
        "polygon_count": polygon_count,
    })


_FIRST_SUCCESS_KEY = "AI_Segmentation/first_success_sent"


def track_first_generation_milestone(mode: str) -> None:
    """One-shot per machine: the user's first successful export ever (their
    first real value moment, in either mode). mode: "auto" | "manual"."""
    try:
        settings = QSettings()
        if bool(settings.value(_FIRST_SUCCESS_KEY, False, type=bool)):
            return
        settings.setValue(_FIRST_SUCCESS_KEY, True)
    except Exception:  # nosec B110 - never break the export path
        return
    track(ev.FIRST_GENERATION_MILESTONE, {"mode": mode})


_upsell_viewed_triggers: set[str] = set()
_low_credit_banner_viewed_this_session = False


def track_pro_upsell_viewed(trigger: str = "free_exhausted") -> None:
    """Fire at most once per session PER TRIGGER when an upsell first renders.

    Deduplicated by trigger, not by process. A single flag for every trigger
    made the surfaces compete for one slot: whichever fired first in a QGIS
    session silenced the others for its whole lifetime, so the view count came
    out below the click count, which is impossible and made every view-to-click
    ratio unusable.
    """
    if trigger in _upsell_viewed_triggers:
        return
    _upsell_viewed_triggers.add(trigger)
    track(ev.PRO_UPSELL_VIEWED, {"trigger": trigger})


def track_pro_upsell_clicked(source: str = "upsell_card") -> None:
    """source: upsell_card / subscribe_pill / low_credit_banner /
    exhausted_status / credit_gauge (the footer balance, which opens the
    dashboard rather than the checkout)."""
    track(ev.PRO_UPSELL_CLICKED, {"source": source})


def track_free_taste_consumed(remaining: int, run_id: str = "") -> None:
    """remaining = free detections left after this one. run_id joins the taste
    to the run that consumed it (best effort: "" when unknown)."""
    track(ev.FREE_TASTE_CONSUMED, {"remaining": remaining, "run_id": run_id})


def track_low_credit_banner_viewed(remaining: int, total: int) -> None:
    """Fire at most once per session when the low-credit banner first shows."""
    global _low_credit_banner_viewed_this_session
    if _low_credit_banner_viewed_this_session:
        return
    _low_credit_banner_viewed_this_session = True
    track(ev.LOW_CREDIT_BANNER_VIEWED, {"remaining": remaining, "total": total})


def track_detect_blocked(reason: str) -> None:
    """reason: credits / zone_too_large / cost_over_balance / worker_busy /
    no_layer / raster_shape / not_activated / kill_switch / no_auth /
    prompt_<guard>."""
    track(ev.DETECT_BLOCKED, {"reason": reason})


def track_account_signed_out(source: str = "account_card") -> None:
    """source: account_card (the Sign out button) or error_card (the one shown
    when the account fails to load)."""
    track(ev.ACCOUNT_SIGNED_OUT, {"source": source})


def track_account_dashboard_opened(source: str = "account_card") -> None:
    """The user left QGIS for the web dashboard. source names the link."""
    track(ev.ACCOUNT_DASHBOARD_OPENED, {"source": source})


def track_telemetry_opt_changed(enabled: bool) -> None:
    """The Privacy checkbox moved. MUST be called BEFORE the flag is written on
    an opt-out, because track() short-circuits on a disabled flag and the event
    would never leave. It is in FLUSH_NOW, so it ships on the spot rather than
    waiting in a batch the opt-out will silence."""
    track(ev.TELEMETRY_OPT_CHANGED, {"enabled": bool(enabled)})


def track_library_opened(tab: str) -> None:
    """tab is the pane the dialog landed on (recent / history / favorites)."""
    track(ev.LIBRARY_OPENED, {"tab": tab})


def track_history_synced(runs: int) -> None:
    """runs is how many server-side runs the sync brought back."""
    track(ev.HISTORY_SYNCED, {"runs": runs})


def track_history_page_loaded(page: int) -> None:
    """page is 0-based: page 1 is the first Load older runs click."""
    track(ev.HISTORY_PAGE_LOADED, {"page": int(page)})


def track_history_favorite_toggled(run_id: str, is_favorite: bool) -> None:
    """is_favorite is the state AFTER the toggle."""
    track(ev.HISTORY_FAVORITE_TOGGLED, {
        "run_id": run_id,
        "is_favorite": bool(is_favorite),
    })


def track_history_restored(run_id: str, tiles: int, objects: int,
                           age_days: int | None = None) -> None:
    """A past run was rebuilt on the canvas from the archive, for zero credits.
    age_days says how stale the run the user came back to was."""
    track(ev.HISTORY_RESTORED, {
        "run_id": run_id,
        "tiles": int(tiles),
        "objects": int(objects),
        "age_days": age_days,
    })


def track_history_exported(fmt: str, objects: int, run_id: str = "") -> None:
    """fmt is the OGR driver the file was written with (GPKG, GeoJSON,
    ESRI Shapefile, KML), not a display name."""
    props = {"format": fmt, "objects": int(objects)}
    if run_id:
        props["run_id"] = run_id
    track(ev.HISTORY_EXPORTED, props)


def track_history_rerun(kind: str) -> None:
    """One-click re-run from a Recent card. kind: same_zone or new_zone."""
    track(ev.HISTORY_RERUN, {"kind": kind})
