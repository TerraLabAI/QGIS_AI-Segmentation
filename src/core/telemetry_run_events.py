"""Telemetry for one Automatic run, start to finish.

The zone and the prompt, the run itself, the post-run review and the
corrections made in it.

Wrappers only: each one names its event's properties in one place so call sites
stay readable. The transport (consent, batching, scrubbing, the POST) is in
telemetry.py, and every function here ends in its track() call.
"""
from __future__ import annotations

from . import telemetry_events as ev
from .telemetry import scrub_payload_value, track


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
    """Free-trial zone cap hit: the zone exceeds the free-tier limit. Only
    the rounded area is sent, never coordinates."""
    track(ev.AUTO_ZONE_TOO_LARGE, {"area_km2": round(area_km2, 1)})


def track_auto_prompt_committed(prompt: str, from_library: bool = False) -> None:
    """prompt is what the user typed, verbatim.

    NOT the validated object class, whatever this docstring used to claim. The
    commit fires on a debounce and on focus loss, both of which run before the
    Detect-time guard, so free-form text reaches this call unchanged. That is
    deliberate: the demand mining behind the object catalogue and the pricing
    work reads these rows, and a cleaned value would answer a different
    question.

    Verbatim, but not raw. The scrubber takes out coordinate tuples, URLs and
    email addresses and leaves ordinary words alone, so "solar panel" arrives
    as itself and a pasted address or contact does not arrive at all. That is
    what makes the row answer the demand question without carrying the one
    class of content the telemetry contract refuses.
    """
    track(ev.AUTO_PROMPT_COMMITTED,
          {"prompt": scrub_payload_value(prompt),
           "from_library": bool(from_library)})


def track_auto_prompt_steered(prompt: str, suggestion: str = "") -> None:
    """prompt is the weak 1-2 word object the user typed; suggestion is the term
    steered toward ("" = pointed at the Library). No PII by construction."""
    track(ev.AUTO_PROMPT_STEERED, {"prompt": prompt, "suggestion": suggestion or ""})


def track_auto_prompt_rewritten(kind: str, prompt: str = "") -> None:
    """A committed prompt was swapped for a cleaner run phrase. kind is one of
    "translated" / "plural" / "alias" (the commit-time guard) or
    "server_rewrite" (a server-side language-model rewrite from the run plan);
    prompt is the 1-2 word English token/phrase that will run (no PII by
    construction)."""
    track(ev.AUTO_PROMPT_REWRITTEN, {"kind": kind, "prompt": prompt or ""})


def track_auto_prompt_hint_shown(kind: str, prompt: str = "") -> None:
    """A non-blocking pre-Detect guidance note was shown. kind is
    "exemplar_boost" (a curated high-value-exemplar object with no example
    drawn), "unknown_object" (a word the model does not know well), "plan_hint"
    (a server run-plan hint under the prompt box), or "identical_rerun" (the
    next Detect would repeat the last run exactly); prompt is the 1-2 word
    object class (no PII by construction)."""
    track(ev.AUTO_PROMPT_HINT_SHOWN, {"kind": kind, "prompt": prompt or ""})


def track_tutorial_opened(source: str) -> None:
    """A tutorial/guide open. source is the touchpoint id (footer_tutorial,
    post_signin, zero_results); no PII by construction."""
    track(ev.TUTORIAL_OPENED, {"source": source})


def track_exemplar_added(count_after: int, label: str = "") -> None:
    """label is "include" or "exclude" (additive property; count_after stays
    the registry's only listed one)."""
    track(ev.EXEMPLAR_ADDED, {"count_after": count_after, "label": label})


def track_exemplar_removed(count_after: int) -> None:
    track(ev.EXEMPLAR_REMOVED, {"count_after": count_after})


def track_detail_changed(detail: int, tiles: int, source: str,
                         band_lo: int = 0, band_hi: int = 0,
                         object_bound: bool = False) -> None:
    """source: "auto_seeded" or "user".

    band_lo/band_hi are the levels the slider offered when this landed, and
    object_bound says the fine end came from the named object rather than from
    the zone or the source resolution. Together they answer the only question
    the shipped band cannot answer on its own: whether users pile up against
    the fine end (the wall is too tight) or never reach it (it is not the
    binding limit). All three are additive; the registry's listed properties
    stay detail, tiles and source.
    """
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
                              merge_mode_source: str = "prompt") -> None:
    """merge_mode is the count-vs-map policy the run picked ("separate"/"map");
    merge_mode_source says how it was decided: "prompt" (object token) or
    "signal" (exemplar-only, read from the run's own masks)."""
    track(ev.AUTO_DETECT_STARTED, {
        "run_id": run_id,
        "tiles": tiles,
        "zone_km2": round(zone_km2, 2),
        "object_class": object_class,
        "detail": detail,
        "exemplar_count": exemplar_count,
        "est_credits": est_credits,
        "credits_before": credits_before,
        "is_free_tier": bool(is_free_tier),
        "merge_mode": merge_mode,
        "merge_mode_source": merge_mode_source,
    })


def track_auto_detect_completed(run_id: str, duration_ms: int, tiles_done: int,
                                tiles_failed: int, instances_found: int,
                                instances_visible_at_default: int, zero_at_default: bool,
                                p50_tile_ms: int | None = None,
                                p95_tile_ms: int | None = None,
                                stop_reason: str = "completed",
                                warming_ms: int = 0,
                                merge_mode_final: str = "separate") -> None:
    """warming_ms is the wall time the run spent in the server waiting room
    (cold start / queue) as perceived by the user; 0 = the run never waited.
    Per-tile latency lives server-side keyed by run_id, so no client percentiles.
    merge_mode_final is the count-vs-map grouping the run finished on
    ("separate"/"map")."""
    track(ev.AUTO_DETECT_COMPLETED, {
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
    })


def track_auto_gate_scan(run_id: str, tiles: int, group: int, scans: int,
                         blocks: int, tiles_skipped: int, tiles_prepaid: int,
                         tiles_unscanned: int, fallback: str,
                         scan_ms: int, tiles_prefiltered: int = 0) -> None:
    """Empty-tile scan gate outcome for one run. Emitted only when the server
    policy armed the gate for the run; fallback carries the stand-down reason
    (resolution / not_text_run / bad_config / no_blocks / offline) or "" when
    the scan phase actually ran. tiles_prefiltered counts the tiles the
    client degenerate prefilter settled during the scan phase (no request).
    Safe from the worker thread (track() only queues; the next main-thread
    flush ships it)."""
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
                             warming_ms: int = 0) -> None:
    """error_class: NETWORK/AUTH/CREDITS_EXHAUSTED/SERVER/CANCELLED/TIMEOUT/UNKNOWN."""
    track(ev.AUTO_DETECT_FAILED, {
        "run_id": run_id,
        "error_class": error_class,
        "tiles_done": tiles_done,
        "duration_ms": duration_ms,
        "warming_ms": warming_ms,
    })


def track_auto_detect_cancelled(run_id: str, tiles_done: int, tiles_total: int,
                                salvaged_to_review: bool,
                                duration_ms: int | None = None,
                                warming_ms: int = 0,
                                backend_stalled: bool = False,
                                submit_retries: int = 0) -> None:
    """duration_ms separates a reflex cancel from a gave-up-after-minutes one;
    warming_ms says how much of that wait was the server waiting room (this is
    the busy-time signal; no separate seconds field, so the *_ms convention
    holds). backend_stalled is True when the run billed ZERO tiles AND the
    service was unresponsive (waiting-room time and/or submit retries): the user
    cancelled a sick backend, not a healthy run. submit_retries is the run's
    total transient submit-retry count. Together they keep a backend outage from
    reading as a user-initiated cancel in analytics."""
    track(ev.AUTO_DETECT_CANCELLED, {
        "run_id": run_id,
        "tiles_done": tiles_done,
        "tiles_total": tiles_total,
        "salvaged_to_review": bool(salvaged_to_review),
        "duration_ms": duration_ms,
        "warming_ms": warming_ms,
        "backend_stalled": bool(backend_stalled),
        "submit_retries": int(submit_retries),
    })


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
        # Pre-submit, uncharged drops: blank/nodata skips (credits saved) and
        # render/provider holes (possible coverage gap). Additive; older keys
        # unchanged.
        "blank_tiles": blank_tiles,
        "render_failed_tiles": render_failed_tiles,
    })


def track_auto_zero_result(run_id: str, tiles: int, object_class: str,
                           had_exemplar: bool) -> None:
    track(ev.AUTO_ZERO_RESULT, {
        "run_id": run_id,
        "tiles": tiles,
        "object_class": object_class,
        "had_exemplar": bool(had_exemplar),
    })


def track_zero_assist_clicked(kind: str, from_prompt: str,
                              to_prompt: str = "") -> None:
    track(ev.ZERO_ASSIST_CLICKED, {
        "kind": kind,
        # The user's own typed words. Scrubbed like every other prompt that
        # leaves the machine: the object word is the signal, a pasted address
        # or contact is not.
        "from_prompt": scrub_payload_value(from_prompt),
        "to_prompt": to_prompt,
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
    """run_id joins the display choice to the run it was made on. It was absent
    for the first year of this event, so the server keeps it optional and older
    rows have none."""
    track(ev.REVIEW_DISPLAY_MODE, {"mode": mode, "run_id": run_id})


def track_review_shape_adjusted(control: str, value, run_id: str = "") -> None:
    """value is stringified: these controls mix numbers and checkboxes, and a
    single column cannot hold both. run_id is optional for the same reason as
    track_review_display_mode."""
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
                           refined_in_manual: bool, autosave: bool = False) -> None:
    """autosave marks the passive leave-safety export (mode switch, new run,
    unload), as opposed to an explicit Finish / Save && exit."""
    track(ev.AUTO_EXPORT_DONE, {
        "run_id": run_id,
        "exported_count": exported_count,
        "visible_pct_of_found": visible_pct_of_found,
        "final_confidence": final_confidence,
        "display_mode": display_mode,
        "refined_in_manual": bool(refined_in_manual),
        "autosave": bool(autosave),
    })


def track_review_abandoned(run_id: str, instances_at_exit: int, refined: bool,
                           confidence_changed: bool, exit_path: str) -> None:
    """The user left the Automatic review without clicking Finish. exit_path is
    one of exit_button | new_run | mode_switch | zone_redraw | raster_removed |
    unload | other. No PII by construction."""
    track(ev.REVIEW_ABANDONED, {
        "run_id": run_id,
        "instances_at_exit": int(instances_at_exit),
        "refined": bool(refined),
        "confidence_changed": bool(confidence_changed),
        "exit_path": exit_path,
    })


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
    """One completed correction gesture.

    gesture is box (the drawn add/remove rectangle) | merge | split, so the two
    free hand edits report through the same event instead of adding one.
    outcome is revealed | removed | queued | empty for a box, merged | split for
    a hand edit; objects counts the objects the gesture touched. label keeps the
    box convention (1 include, 0 exclude) and a hand edit reports 1, since it
    reshapes rather than drops.
    """
    track(ev.REVIEW_CORRECT_BOX, {
        "run_id": run_id,
        "label": int(label),
        "outcome": outcome,
        "objects": int(objects),
        "gesture": gesture,
    })


def track_review_correct_undo(run_id: str, kind: str) -> None:
    """kind is the undone journal entry's kind (merge | split), or
    clear_all for the whole-round clear."""
    track(ev.REVIEW_CORRECT_UNDO, {"run_id": run_id, "kind": kind})


def track_review_step(run_id: str, step: int) -> None:
    """Step navigation within the linear review; callers dedupe per run+step."""
    track(ev.REVIEW_STEP, {"run_id": run_id, "step": int(step)})


def track_qgis_edit_bridge(run_id: str, outcome: str,
                           duration_ms: int | None = None,
                           features: int | None = None) -> None:
    """QGIS digitizing bridge lifecycle from the review's Correct step.

    outcome is opened (armed native editing) | committed | rolled_back.
    duration_ms and features are absent on 'opened' and carried on the two
    terminal outcomes (how long the bridge stayed open, and how many features
    folded back).
    """
    props: dict = {"run_id": run_id, "outcome": outcome}
    if duration_ms is not None:
        props["duration_ms"] = int(duration_ms)
    if features is not None:
        props["features"] = int(features)
    track(ev.AUTO_EDIT_IN_QGIS, props)


def track_exemplar_nudge_shown(object_class: str, run_id: str = "",
                               median_score: float = -1.0) -> None:
    """The draw-an-example tip was offered. median_score is the run's median
    detection score, and -1 is the sentinel for the pre-run nudge under the
    prompt box, which has no run to score yet."""
    track(ev.EXEMPLAR_NUDGE_SHOWN, {
        "run_id": run_id,
        "object_class": object_class,
        "median_score": round(float(median_score), 3),
    })


def track_exemplar_nudge_clicked(object_class: str, run_id: str = "",
                                 median_score: float = -1.0) -> None:
    """The nudge link was followed. Same -1 sentinel as the shown event."""
    track(ev.EXEMPLAR_NUDGE_CLICKED, {
        "run_id": run_id,
        "object_class": object_class,
        "median_score": round(float(median_score), 3),
    })
