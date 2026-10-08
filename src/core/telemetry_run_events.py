








from __future__ import annotations

from . import telemetry_events as ev
from .telemetry import scrub_payload_value, track
from .telemetry_run_context import (
    note_review_opened,
    note_run_ended,
    note_run_started,
    review_elapsed_props,
    run_attempt_props,
    run_failure_stage,
    run_headless_props,
)
from .telemetry_run_profile import client_props, review_pass_props
from .telemetry_session_events import _sent_this_session





_REVIEW_ITEM_SAMPLE_RATE = 10


def track_auto_start_clicked(layer_kind: str, has_credits_known: bool = False) -> None:


    track(ev.AUTO_START_CLICKED, {
        "layer_kind": layer_kind,
        "has_credits_known": bool(has_credits_known),
        **run_attempt_props(),
    })


def track_zone_drawn(vertices: int, area_km2: float, zone_kind: str = "polygon") -> None:
    track(ev.ZONE_DRAWN, {
        "vertices": vertices,
        "area_km2": round(area_km2, 1),
        "zone_kind": zone_kind,
        **run_attempt_props(),
    })


def track_auto_zone_too_large(area_km2: float) -> None:


    track(ev.AUTO_ZONE_TOO_LARGE, {"area_km2": round(area_km2, 1)})


def track_auto_zone_free_clipped(km2_requested: float, km2_processed: float,
                                 km2_left: float | None = None) -> None:


    props = {
        "km2_requested": round(float(km2_requested), 3),
        "km2_processed": round(float(km2_processed), 3),
    }
    if km2_left is not None:
        props["km2_left"] = round(float(km2_left), 3)
    track(ev.AUTO_ZONE_FREE_CLIPPED, props)


def track_auto_zone_free_clip_choice(choice: str, km2_requested: float,
                                     km2_processed: float | None = None,
                                     km2_left: float | None = None) -> None:


    props = {"choice": choice, "km2_requested": round(float(km2_requested), 3)}
    if km2_processed is not None:
        props["km2_processed"] = round(float(km2_processed), 3)
    if km2_left is not None:
        props["km2_left"] = round(float(km2_left), 3)
    track(ev.AUTO_ZONE_FREE_CLIP_CHOICE, props)


def track_auto_prompt_committed(prompt: str, from_library: bool = False) -> None:















    track(ev.AUTO_PROMPT_COMMITTED,
          {"prompt": scrub_payload_value(prompt),
           "from_library": bool(from_library),
           **run_attempt_props()})


def track_auto_prompt_steered(prompt: str, suggestion: str = "") -> None:







    track(ev.AUTO_PROMPT_STEERED, {
        "prompt": scrub_payload_value(prompt or ""),
        "suggestion": suggestion or "",
    })


def track_auto_prompt_rewritten(kind: str, prompt: str = "") -> None:






    track(ev.AUTO_PROMPT_REWRITTEN, {
        "kind": kind,
        "prompt": scrub_payload_value(prompt or ""),
    })


def track_auto_prompt_hint_shown(kind: str, prompt: str = "") -> None:







    track(ev.AUTO_PROMPT_HINT_SHOWN, {
        "kind": kind,
        "prompt": scrub_payload_value(prompt or ""),
    })


def track_tutorial_opened(source: str) -> None:


    track(ev.TUTORIAL_OPENED, {"source": source})


def track_exemplar_added(count_after: int, label: str = "") -> None:


    track(ev.EXEMPLAR_ADDED, {"count_after": count_after, "label": label})


def track_exemplar_removed(count_after: int) -> None:
    track(ev.EXEMPLAR_REMOVED, {"count_after": count_after})


def track_detail_changed(detail: int, tiles: int, source: str,
                         band_lo: int = 0, band_hi: int = 0,
                         object_bound: bool = False) -> None:










    track(ev.DETAIL_CHANGED, {
        "detail": detail, "tiles": tiles, "source": source,
        "band_lo": band_lo, "band_hi": band_hi,
        "at_fine_end": bool(band_hi) and detail >= band_hi,
        "object_bound": bool(object_bound),
    })


def track_auto_detect_started(run_id: str, tiles: int, zone_km2: float,
                              object_class: str, detail: int, exemplar_count: int,
                              est_credits: int, credits_before: int | None,
                              is_free_tier: bool,
                              merge_mode: str = "separate",
                              merge_mode_source: str = "prompt",
                              detail_seeded: int | None = None,
                              tile_props: dict | None = None,
                              plan_hold_ms: int = 0,
                              headless: bool = False) -> None:















    if detail_seeded is None:
        detail_source = "unknown"
    elif int(detail_seeded) == int(detail):
        detail_source = "seed"
    else:
        detail_source = "user"
    props = {
        "run_id": run_id,
        "tiles": tiles,
        "zone_km2": round(zone_km2, 2),


        "object_class": scrub_payload_value(object_class or ""),
        "detail": detail,
        "detail_source": detail_source,
        "exemplar_count": exemplar_count,
        "est_credits": est_credits,
        "credits_before": credits_before,
        "is_free_tier": bool(is_free_tier),
        "merge_mode": merge_mode,
        "merge_mode_source": merge_mode_source,
        "plan_hold_ms": max(0, int(plan_hold_ms or 0)),
    }
    from .telemetry_config_props import config_provenance_props

    props.update(config_provenance_props())
    if detail_seeded is not None:
        props["detail_seeded"] = int(detail_seeded)


    note_run_started(run_id, tiles, headless=headless)
    props.update(run_attempt_props(run_id))


    for key in ("tile_plan", "tile_ground_m", "tile_prior_m", "tile_reasons",
                "tile_warning", "grid_reach_levels"):
        if tile_props and key in tile_props:
            props[key] = tile_props[key]
    track(ev.AUTO_DETECT_STARTED, props)


def track_auto_detect_completed(run_id: str, duration_ms: int, tiles_done: int,
                                tiles_failed: int, instances_found: int,
                                instances_visible_at_default: int, zero_at_default: bool,
                                stop_reason: str = "completed",
                                warming_ms: int = 0,
                                merge_mode_final: str = "separate",
                                blob_armed: int = 0,
                                blob_dropped: int = 0,
                                tile_ground_m: int = 0,
                                client_profile: dict | None = None) -> None:




















    props = {
        "run_id": run_id,
        "duration_ms": duration_ms,
        "tiles_done": tiles_done,
        "tiles_failed": tiles_failed,
        "instances_found": instances_found,
        "instances_visible_at_default": instances_visible_at_default,
        "zero_at_default": bool(zero_at_default),
        "stop_reason": stop_reason,
        "warming_ms": warming_ms,
        "merge_mode_final": merge_mode_final,
        "blob_armed": int(blob_armed),
        "blob_dropped": int(blob_dropped),
        "tile_ground_m": int(tile_ground_m),
    }
    props.update(client_props(client_profile))
    props.update(run_attempt_props(run_id))
    props.update(run_headless_props(run_id))
    note_run_ended(run_id)
    track(ev.AUTO_DETECT_COMPLETED, props)


def track_auto_gate_scan(run_id: str, tiles: int, group: int, scans: int,
                         blocks: int, tiles_skipped: int, tiles_prepaid: int,
                         tiles_unscanned: int, fallback: str,
                         scan_ms: int, tiles_prefiltered: int = 0) -> None:







    track(ev.AUTO_GATE_SCAN, {
        "run_id": run_id,
        "tiles": tiles,
        "group": group,
        "scans": scans,
        "blocks": blocks,
        "tiles_skipped": tiles_skipped,
        "tiles_prepaid": tiles_prepaid,
        "tiles_unscanned": tiles_unscanned,
        "tiles_prefiltered": tiles_prefiltered,
        "fallback": fallback,
        "scan_ms": scan_ms,
    })


def track_auto_detect_failed(run_id: str, error_class: str, tiles_done: int,
                             duration_ms: int | None = None,
                             warming_ms: int = 0,
                             client_profile: dict | None = None,
                             error_code: str = "",
                             stage: str = "") -> None:







    props = {
        "run_id": run_id,
        "error_class": error_class,
        "error_code": error_code or f"auto_detect_{(error_class or 'unknown').lower()}",
        "stage": stage or run_failure_stage(),
        "tiles_done": tiles_done,
        "duration_ms": duration_ms,
        "warming_ms": warming_ms,
    }
    props.update(client_props(client_profile))
    props.update(run_attempt_props(run_id))
    props.update(run_headless_props(run_id))
    note_run_ended(run_id)
    track(ev.AUTO_DETECT_FAILED, props)


def track_auto_detect_cancelled(run_id: str, tiles_done: int, tiles_total: int,
                                salvaged_to_review: bool,
                                duration_ms: int | None = None,
                                warming_ms: int = 0,
                                backend_stalled: bool = False,
                                submit_retries: int = 0,
                                client_profile: dict | None = None) -> None:









    props = {
        "run_id": run_id,
        "tiles_done": tiles_done,
        "tiles_total": tiles_total,
        "salvaged_to_review": bool(salvaged_to_review),
        "duration_ms": duration_ms,
        "warming_ms": warming_ms,
        "backend_stalled": bool(backend_stalled),
        "submit_retries": int(submit_retries),
    }
    props.update(client_props(client_profile))
    props.update(run_attempt_props(run_id))
    props.update(run_headless_props(run_id))
    note_run_ended(run_id)
    track(ev.AUTO_DETECT_CANCELLED, props)


def track_credits_exhausted(run_id: str, tiles_done: int, tiles_total: int,
                            is_free_tier: bool) -> None:
    note_run_ended(run_id)
    track(ev.CREDITS_EXHAUSTED, {
        "run_id": run_id,
        "tiles_done": tiles_done,
        "tiles_total": tiles_total,
        "is_free_tier": bool(is_free_tier),
    })


def track_auto_tiles_degraded(run_id: str, skipped_tiles: int, timeout_tiles: int,
                              blank_tiles: int = 0,
                              render_failed_tiles: int = 0,
                              prefiltered_tiles: int = 0) -> None:
    track(ev.AUTO_TILES_DEGRADED, {
        "run_id": run_id,
        "skipped_tiles": skipped_tiles,
        "timeout_tiles": timeout_tiles,



        "blank_tiles": blank_tiles,
        "render_failed_tiles": render_failed_tiles,


        "prefiltered_tiles": prefiltered_tiles,
    })


def track_auto_zero_result(run_id: str, tiles: int, object_class: str,
                           had_exemplar: bool) -> None:
    track(ev.AUTO_ZERO_RESULT, {
        "run_id": run_id,
        "tiles": tiles,

        "object_class": scrub_payload_value(object_class or ""),
        "had_exemplar": bool(had_exemplar),
    })


def track_zero_assist_clicked(kind: str, from_prompt: str,
                              to_prompt: str = "") -> None:
    track(ev.ZERO_ASSIST_CLICKED, {
        "kind": kind,



        "from_prompt": scrub_payload_value(from_prompt),
        "to_prompt": scrub_payload_value(to_prompt or ""),
    })


def track_review_opened(run_id: str, instances_found: int, visible_at_start: int,
                        start_confidence: int, auto_lowered: bool) -> None:
    note_review_opened(run_id)
    track(ev.REVIEW_OPENED, {
        "run_id": run_id,
        "instances_found": instances_found,
        "visible_at_start": visible_at_start,
        "start_confidence": start_confidence,
        "auto_lowered": bool(auto_lowered),
    })


def track_review_confidence_final(run_id: str, final_pct: int, visible_count: int,
                                  moves: int, reveal_clicks: int = 0) -> None:

    track(ev.REVIEW_CONFIDENCE_FINAL, {
        "run_id": run_id,
        "final_pct": final_pct,
        "visible_count": visible_count,
        "moves": moves,
        "reveal_clicks": int(reveal_clicks),
    })


def track_review_display_mode(mode: str, run_id: str = "") -> None:



    track(ev.REVIEW_DISPLAY_MODE, {"mode": mode, "run_id": run_id})


def track_review_shape_adjusted(control: str, value, run_id: str = "") -> None:



    track(ev.REVIEW_SHAPE_ADJUSTED, {
        "control": control,
        "value": "" if value is None else str(value),
        "run_id": run_id,
    })


def track_refine_in_manual_entered(run_id: str, instances: int) -> None:
    track(ev.REFINE_IN_MANUAL_ENTERED, {"run_id": run_id, "instances": instances})


def track_refine_in_manual_back(run_id: str, validated_count: int,
                                duration_ms: int | None = None) -> None:
    track(ev.REFINE_IN_MANUAL_BACK, {
        "run_id": run_id,
        "validated_count": validated_count,
        "duration_ms": duration_ms,
    })


def track_auto_export_done(run_id: str, exported_count: int, visible_pct_of_found: int,
                           final_confidence: int, display_mode: str,
                           refined_in_manual: bool, autosave: bool = False,
                           pass_profile: dict | None = None,
                           land_cover: bool = False,
                           export_ms: int | None = None) -> None:






    props = {
        "run_id": run_id,
        "exported_count": exported_count,
        "visible_pct_of_found": visible_pct_of_found,
        "final_confidence": final_confidence,
        "display_mode": display_mode,
        "refined_in_manual": bool(refined_in_manual),
        "autosave": bool(autosave),
        "land_cover": bool(land_cover),
    }
    if export_ms is not None:
        props["export_ms"] = max(0, int(export_ms))
    props.update(review_pass_props(pass_profile))
    props.update(review_elapsed_props(run_id))
    track(ev.AUTO_EXPORT_DONE, props)


def land_cover_zone_bucket(km2: float) -> str:

    if km2 < 0.1:
        return "<0.1"
    if km2 < 1:
        return "0.1-1"
    if km2 < 10:
        return "1-10"
    return ">=10"


def track_land_cover_shown(run_id: str, class_count: int, zone_km2: float) -> None:
    track(ev.AUTO_LAND_COVER_SHOWN, {
        "run_id": run_id, "class_count": int(class_count),
        "zone_km2_bucket": land_cover_zone_bucket(float(zone_km2 or 0.0))})


def track_auto_target_switched(to: str, source: str) -> None:


    track(ev.AUTO_TARGET_SWITCHED, {"to": str(to), "source": str(source)})


def track_my_classes(outcome: str, run_id: str, class_count: int, billed_km2: float,
                     replayed: bool = False, stage: str = "", error_code: str = "",
                     refunded: bool = False) -> None:

    props = {"run_id": run_id, "class_count": int(class_count),
             "billed_km2": round(float(billed_km2 or 0.0), 4)}
    if outcome == "started":
        track(ev.AUTO_MY_CLASSES_STARTED, props)
    elif outcome == "succeeded":
        track(ev.AUTO_MY_CLASSES_SUCCEEDED, dict(props, replayed=bool(replayed)))
    else:
        track(ev.AUTO_MY_CLASSES_FAILED, dict(
            props, stage=str(stage or "request"), error_code=str(error_code or "UNKNOWN")[:64],
            refunded=bool(refunded)))


def track_land_cover_table_copied(run_id: str, class_count: int) -> None:
    track(ev.AUTO_LAND_COVER_TABLE_COPIED, {
        "run_id": run_id, "class_count": int(class_count)})


def track_land_cover_min_patch_changed(run_id: str, min_patch_m2: float,
                                       from_default: bool) -> None:
    track(ev.AUTO_LAND_COVER_MIN_PATCH_CHANGED, {
        "run_id": run_id, "min_patch_m2": float(min_patch_m2),
        "from_default": bool(from_default)})


def track_review_abandoned(run_id: str, instances_at_exit: int, refined: bool,
                           confidence_changed: bool, exit_path: str,
                           pass_profile: dict | None = None) -> None:




    props = {
        "run_id": run_id,
        "instances_at_exit": int(instances_at_exit),
        "refined": bool(refined),
        "confidence_changed": bool(confidence_changed),
        "exit_path": exit_path,
    }
    props.update(review_pass_props(pass_profile))
    props.update(review_elapsed_props(run_id))
    track(ev.REVIEW_ABANDONED, props)


def track_auto_retry_clicked(run_id: str, discarded_count: int, confirmed: bool) -> None:
    track(ev.AUTO_RETRY_CLICKED, {
        "run_id": run_id,
        "discarded_count": discarded_count,
        "confirmed": bool(confirmed),
    })


def track_auto_resume(offered: bool, run_id: str, tiles_missing: int,
                      cache_kept: bool) -> None:



    track(ev.AUTO_RESUME_OFFERED if offered else ev.AUTO_RESUME_CLICKED, {
        "run_id": run_id,
        "tiles_missing": int(tiles_missing),
        "cache_kept": bool(cache_kept),
    })


def track_review_correct_box(run_id: str, label: int, outcome: str,
                             objects: int, gesture: str = "box") -> None:














    import random

    if "review_correct_box" in _sent_this_session:
        if random.random() >= 1 / _REVIEW_ITEM_SAMPLE_RATE:  # nosec B311
            return
        sample_rate = _REVIEW_ITEM_SAMPLE_RATE
    else:
        _sent_this_session.add("review_correct_box")
        sample_rate = 1
    track(ev.REVIEW_CORRECT_BOX, {
        "run_id": run_id,
        "label": int(label),
        "outcome": outcome,
        "objects": int(objects),
        "gesture": gesture,
        "sample_rate": sample_rate,
    })


def track_review_correct_undo(run_id: str, kind: str) -> None:


    track(ev.REVIEW_CORRECT_UNDO, {"run_id": run_id, "kind": kind})


def track_review_step(run_id: str, step: int) -> None:





    import random

    if "review_step" in _sent_this_session:
        if random.random() >= 1 / _REVIEW_ITEM_SAMPLE_RATE:  # nosec B311
            return
        sample_rate = _REVIEW_ITEM_SAMPLE_RATE
    else:
        _sent_this_session.add("review_step")
        sample_rate = 1
    track(ev.REVIEW_STEP, {"run_id": run_id, "step": int(step), "sample_rate": sample_rate})


def track_qgis_edit_bridge(run_id: str, outcome: str,
                           duration_ms: int | None = None,
                           features: int | None = None) -> None:







    props: dict = {"run_id": run_id, "outcome": outcome}
    if duration_ms is not None:
        props["duration_ms"] = int(duration_ms)
    if features is not None:
        props["features"] = int(features)
    track(ev.AUTO_EDIT_IN_QGIS, props)


def track_auto_imagery_notice(kind: str, run_anyway: bool, prompt: str = "",
                              source_m_per_px: float = 0.0) -> None:



    props = {
        "kind": kind,
        "answer": "run_anyway" if run_anyway else "cancel",
        "prompt": scrub_payload_value(prompt or ""),
    }
    if source_m_per_px and source_m_per_px > 0:
        props["source_m_per_px"] = round(float(source_m_per_px), 2)
    track(ev.AUTO_IMAGERY_NOTICE, props)
