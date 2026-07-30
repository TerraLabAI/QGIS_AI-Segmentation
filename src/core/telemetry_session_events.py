








from __future__ import annotations

from qgis.PyQt.QtCore import QSettings

from . import telemetry_events as ev
from .telemetry import (
    drop_queued_events,
    is_telemetry_enabled,
    scrub_payload_value,
    track,
)

_FIRST_OPEN_KEY = "AI_Segmentation/first_open_sent"


def track_plugin_first_open() -> None:









    if not is_telemetry_enabled():
        return
    try:
        settings = QSettings()
        if bool(settings.value(_FIRST_OPEN_KEY, False, type=bool)):
            return
        settings.setValue(_FIRST_OPEN_KEY, True)
    except Exception:  # nosec B110
        return
    track(ev.PLUGIN_FIRST_OPEN)


def track_plugin_opened() -> None:

    track(ev.PLUGIN_OPENED)


def track_plugin_activated(duration_ms: int | None = None) -> None:


    track(ev.PLUGIN_ACTIVATED, {"duration_ms": duration_ms})


def track_pairing_started() -> None:

    track(ev.PAIRING_STARTED)


def track_pairing_failed(error_code: str, duration_ms: int | None = None) -> None:


    track(ev.PAIRING_FAILED, {"error_code": error_code, "duration_ms": duration_ms})


def track_pairing_cancelled(duration_ms: int | None = None) -> None:
    track(ev.PAIRING_CANCELLED, {"duration_ms": duration_ms})


def track_mode_switched(to_mode: str, had_unsaved_manual: bool = False,
                        auto_step: int | None = None) -> None:

    track(ev.MODE_SWITCHED, {
        "to_mode": to_mode,
        "had_unsaved_manual": bool(had_unsaved_manual),
        "auto_step": auto_step,
    })




def track_install_started(entry: str = "background") -> None:
    track(ev.INSTALL_STARTED, {"entry": entry})


def track_install_completed(duration_ms: int | None = None,
                            python_minor: int | None = None,
                            retry_count: int | None = None,
                            entry: str = "background",
                            local_model_ready: bool | None = None) -> None:






    track(ev.INSTALL_COMPLETED, {
        "duration_ms": duration_ms,
        "python_minor": python_minor,
        "retry_count": retry_count,
        "entry": entry,
        "local_model_ready": None if local_model_ready is None else bool(local_model_ready),
    })


def track_install_failed(error_class: str, duration_ms: int | None = None,
                         python_minor: int | None = None,
                         retry_count: int | None = None,
                         detail: str | None = None,
                         entry: str = "background") -> None:









    props = {
        "error_class": error_class,
        "duration_ms": duration_ms,
        "python_minor": python_minor,
        "retry_count": retry_count,
        "entry": entry,
    }
    if detail:
        props["error_detail"] = scrub_payload_value(detail)[:300]
    track(ev.INSTALL_FAILED, props)


def track_install_cancelled(duration_ms: int | None = None,
                            entry: str = "background") -> None:

    track(ev.INSTALL_CANCELLED, {"duration_ms": duration_ms, "entry": entry})


def track_model_download_completed(model: str, duration_ms: int | None = None) -> None:

    track(ev.MODEL_DOWNLOAD_COMPLETED, {"model": model, "duration_ms": duration_ms})



_sent_this_session: set[str] = set()


def track_segmentation_run(success: bool, duration_ms: int | None = None) -> None:





    import random

    if not success:

        track(ev.SEGMENTATION_RUN, {
            "success": False, "duration_ms": duration_ms, "sample_rate": 1,
        })
        return

    if "segmentation_run" in _sent_this_session:
        if random.random() >= 0.1:  # nosec B311
            return
        sample_rate = 10
    else:
        _sent_this_session.add("segmentation_run")
        sample_rate = 1
    track(ev.SEGMENTATION_RUN, {
        "success": True, "duration_ms": duration_ms, "sample_rate": sample_rate,
    })


def track_manual_export_done(
    polygon_count: int, refine_used: bool, destination: str = "new"
) -> None:


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





    track(ev.MANUAL_ENGINE_CHOSEN, {
        "engine": engine,
        "local_installed": bool(local_installed),
        "from_install_gate": bool(from_install_gate),
    })


def track_manual_cloud_consent(accepted: bool) -> None:






    track(ev.MANUAL_CLOUD_CONSENT, {"accepted": bool(accepted)})


def track_manual_click_answered(engine: str, duration_ms: int | None = None,
                                used_fallback: bool = False,
                                is_correct: bool = False) -> None:








    import random

    if "click_answered" in _sent_this_session:
        if random.random() >= 0.1:  # nosec B311
            return
        sample_rate = 10
    else:
        _sent_this_session.add("click_answered")
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





    track(ev.MANUAL_OBJECT_CHARGED, {
        "outcome": outcome,
        "objects_charged": objects_charged,
        "error_code": error_code or None,
    })


def track_manual_objects_wall_hit(is_subscriber: bool, in_session: bool,
                                  objects_cap: int | None = None,
                                  objects_used: int | None = None) -> None:


    track(ev.MANUAL_OBJECTS_WALL_HIT, {
        "is_subscriber": bool(is_subscriber),
        "in_session": bool(in_session),
        "objects_cap": objects_cap,
        "objects_used": objects_used,
    })


def track_manual_abandoned(context: str, polygon_count: int) -> None:


    track(ev.MANUAL_ABANDONED, {
        "context": context,
        "polygon_count": polygon_count,
    })


_FIRST_SUCCESS_KEY = "AI_Segmentation/first_success_sent"


def track_first_generation_milestone(mode: str) -> None:





    if not is_telemetry_enabled():
        return
    try:
        settings = QSettings()
        if bool(settings.value(_FIRST_SUCCESS_KEY, False, type=bool)):
            return
        settings.setValue(_FIRST_SUCCESS_KEY, True)
    except Exception:  # nosec B110
        return
    track(ev.FIRST_GENERATION_MILESTONE, {"mode": mode})


_upsell_viewed_triggers: set[str] = set()


def track_pro_upsell_viewed(trigger: str = "free_exhausted",
                            cta_source: str | None = None) -> None:



















    if trigger in _upsell_viewed_triggers:
        return
    _upsell_viewed_triggers.add(trigger)
    track(ev.PRO_UPSELL_VIEWED,
          {"trigger": trigger, "cta_source": cta_source or trigger})


def track_pro_upsell_clicked(
    source: str = "upsell_card", checkout_link: str | None = None
) -> None:









    props = {"source": source}
    if checkout_link:
        props["checkout_link"] = checkout_link
    track(ev.PRO_UPSELL_CLICKED, props)


def track_free_taste_consumed(remaining: int, run_id: str = "") -> None:


    track(ev.FREE_TASTE_CONSUMED, {"remaining": remaining, "run_id": run_id})


def track_low_credit_banner_viewed(remaining: int, total: int) -> None:

    if "low_credit_banner_viewed" in _sent_this_session:
        return
    _sent_this_session.add("low_credit_banner_viewed")
    track(ev.LOW_CREDIT_BANNER_VIEWED, {"remaining": remaining, "total": total})


def track_detect_blocked(reason: str) -> None:




    track(ev.DETECT_BLOCKED, {"reason": reason})


def track_account_signed_out(source: str = "account_card") -> None:


    track(ev.ACCOUNT_SIGNED_OUT, {"source": source})


def track_account_dashboard_opened(source: str = "account_card") -> None:

    track(ev.ACCOUNT_DASHBOARD_OPENED, {"source": source})


def track_telemetry_opt_changed(enabled: bool) -> None:









    if not enabled:
        drop_queued_events()
    track(ev.TELEMETRY_OPT_CHANGED, {"enabled": bool(enabled)})


def track_library_opened(tab: str) -> None:

    track(ev.LIBRARY_OPENED, {"tab": tab})


def track_history_synced(runs: int) -> None:

    track(ev.HISTORY_SYNCED, {"runs": runs})


def track_history_page_loaded(page: int) -> None:

    track(ev.HISTORY_PAGE_LOADED, {"page": int(page)})


def track_history_favorite_toggled(run_id: str, is_favorite: bool) -> None:

    track(ev.HISTORY_FAVORITE_TOGGLED, {
        "run_id": run_id,
        "is_favorite": bool(is_favorite),
    })


def track_history_restored(run_id: str, tiles: int, objects: int,
                           age_days: int | None = None) -> None:


    track(ev.HISTORY_RESTORED, {
        "run_id": run_id,
        "tiles": int(tiles),
        "objects": int(objects),
        "age_days": age_days,
    })


def track_history_exported(fmt: str, objects: int, run_id: str = "") -> None:


    props = {"format": fmt, "objects": int(objects)}
    if run_id:
        props["run_id"] = run_id
    track(ev.HISTORY_EXPORTED, props)


def track_history_rerun(kind: str) -> None:

    track(ev.HISTORY_RERUN, {"kind": kind})
