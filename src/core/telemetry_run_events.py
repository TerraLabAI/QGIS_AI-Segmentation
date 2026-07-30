








from __future__ import annotations

from . import telemetry_events as ev
from .telemetry import scrub_payload_value, track
from .telemetry_run_profile import client_props, review_pass_props


def track_auto_start_clicked(layer_kind: str, has_credits_known: bool = False) -> None:
    track(ev.AUTO_START_CLICKED, {
        "layer_kind": layer_kind,
        "has_credits_known": bool(has_credits_known),
    })


def track_zone_drawn(vertices: int, area_km2: float, zone_kind: str = "polygon") -> None:
    track(ev.ZONE_DRAWN, {
        "vertices": vertices,
        "area_km2": round(area_km2, 1),
        "zone_kind": zone_kind,
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
           "from_library": bool(from_library)})


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
                              detail_seeded: int | None = None) -> None:











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
    }
    if detail_seeded is not None:
        props["detail_seeded"] = int(detail_seeded)
    track(ev.AUTO_DETECT_STARTED, props)


def track_auto_detect_completed(run_id: str, duration_ms: int, tiles_done: int,
                                tiles_failed: int, instances_found: int,
                                instances_visible_at_default: int, zero_at_default: bool,
                                p50_tile_ms: int | None = None,
                                p95_tile_ms: int | None = None,
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
        "p50_tile_ms": p50_tile_ms,
        "p95_tile_ms": p95_tile_ms,
        "stop_reason": stop_reason,
        "warming_ms": warming_ms,
        "merge_mode_final": merge_mode_final,
        "blob_armed": int(blob_armed),
        "blob_dropped": int(blob_dropped),
        "tile_ground_m": int(tile_ground_m),
    }
    props.update(client_props(client_profile))
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
                             client_profile: dict | None = None) -> None:


    props = {
        "run_id": run_id,
        "error_class": error_class,
        "tiles_done": tiles_done,
        "duration_ms": duration_ms,
        "warming_ms": warming_ms,
    }
    props.update(client_props(client_profile))
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
    track(ev.AUTO_DETECT_CANCELLED, props)


def track_credits_exhausted(run_id: str, tiles_done: int, tiles_total: int,
                            is_free_tier: bool) -> None:
    track(ev.CREDITS_EXHAUSTED, {
        "run_id": run_id,
        "tiles_done": tiles_done,
        "tiles_total": tiles_total,
        "is_free_tier": bool(is_free_tier),
    })


def track_auto_tiles_degraded(run_id: str, skipped_tiles: int, timeout_tiles: int,
                              blank_tiles: int = 0,
                              render_failed_tiles: int = 0) -> None:
    track(ev.AUTO_TILES_DEGRADED, {
        "run_id": run_id,
        "skipped_tiles": skipped_tiles,
        "timeout_tiles": timeout_tiles,



        "blank_tiles": blank_tiles,
        "render_failed_tiles": render_failed_tiles,
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
    track(ev.REVIEW_OPENED, {
        "run_id": run_id,
        "instances_found": instances_found,
        "visible_at_start": visible_at_start,
        "start_confidence": start_confidence,
        "auto_lowered": bool(auto_lowered),
    })


def track_review_confidence_final(run_id: str, final_pct: int, visible_count: int,
                                  moves: int) -> None:
    track(ev.REVIEW_CONFIDENCE_FINAL, {
        "run_id": run_id,
        "final_pct": final_pct,
        "visible_count": visible_count,
        "moves": moves,
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
                           pass_profile: dict | None = None) -> None:



    props = {
        "run_id": run_id,
        "exported_count": exported_count,
        "visible_pct_of_found": visible_pct_of_found,
        "final_confidence": final_confidence,
        "display_mode": display_mode,
        "refined_in_manual": bool(refined_in_manual),
        "autosave": bool(autosave),
    }
    props.update(review_pass_props(pass_profile))
    track(ev.AUTO_EXPORT_DONE, props)


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
    track(ev.REVIEW_ABANDONED, props)


def track_auto_retry_clicked(run_id: str, discarded_count: int, confirmed: bool) -> None:
    track(ev.AUTO_RETRY_CLICKED, {
        "run_id": run_id,
        "discarded_count": discarded_count,
        "confirmed": bool(confirmed),
    })


def track_auto_exit_clicked(from_step: int, autosaved_count: int) -> None:
    track(ev.AUTO_EXIT_CLICKED, {
        "from_step": from_step,
        "autosaved_count": autosaved_count,
    })


def track_review_correct_box(run_id: str, label: int, outcome: str,
                             objects: int, gesture: str = "box") -> None:









    track(ev.REVIEW_CORRECT_BOX, {
        "run_id": run_id,
        "label": int(label),
        "outcome": outcome,
        "objects": int(objects),
        "gesture": gesture,
    })


def track_review_correct_undo(run_id: str, kind: str) -> None:


    track(ev.REVIEW_CORRECT_UNDO, {"run_id": run_id, "kind": kind})


def track_review_step(run_id: str, step: int) -> None:

    track(ev.REVIEW_STEP, {"run_id": run_id, "step": int(step)})


def track_qgis_edit_bridge(run_id: str, outcome: str,
                           duration_ms: int | None = None,
                           features: int | None = None) -> None:







    props: dict = {"run_id": run_id, "outcome": outcome}
    if duration_ms is not None:
        props["duration_ms"] = int(duration_ms)
    if features is not None:
        props["features"] = int(features)
    track(ev.AUTO_EDIT_IN_QGIS, props)
