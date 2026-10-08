









from __future__ import annotations




def agent_workflow_steps() -> list[dict]:











    from .core.server_dials import dial_text

    steps = [
        {
            "step": 1,
            "call": "get_status",
            "why": "Readiness first; mode=automatic checks the cloud path.",
            "optional": False,
        },
        {
            "step": 2,
            "call": "load_model",
            "why": "Only when get_status reports MODEL_NOT_LOADED.",
            "optional": True,
        },
        {
            "step": 3,
            "call": "set_mode",
            "why": "'interactive' for one object at a time, 'automatic' for a zone.",
            "optional": True,
        },
        {
            "step": 4,
            "call": "detect_points",
            "why": "Interactive route: outline one object from points.",
            "optional": True,
        },
        {
            "step": 5,
            "call": "list_object_classes",
            "why": "The words detect_auto accepts as object_class.",
            "optional": True,
        },
        {
            "step": 6,
            "call": "detect_auto",
            "why": "Automatic route: a class and a zone.",
            "optional": True,
        },
        {
            "step": 7,
            "call": "auto_detect_status",
            "why": "Follow a zone run until it ends.",
            "optional": True,
        },
        {
            "step": 8,
            "call": "review_status",
            "why": "Only for a run a person left open in the panel.",
            "optional": True,
        },
    ]
    for step in steps:
        served_why = dial_text("tuning.agent.step_why", step["call"], 400)
        if served_why:
            step["why"] = served_why
    return steps




def agent_method_notes() -> dict[str, dict]:











    from .core.server_dials import dial_text

    fast_free = {"spends": False, "slow": False, "needs_raster": False}
    notes = {
        "capabilities": dict(fast_free, summary="What this build can do."),
        "guide": dict(fast_free, summary="How to get good results, in prose."),
        "get_status": dict(fast_free, summary="Readiness by mode and what to fix."),
        "prepare_interactive": dict(
            fast_free, summary="Open the native panel to draw a zone or examples, without running detection."),
        "get_interactive_state": dict(
            fast_free, summary="Read exact drawn inputs and their source CRS to resume a chat workflow."),
        "install_status": dict(
            fast_free, summary="What is on disk. Read only, installs nothing."),
        "load_model": {
            "spends": False, "slow": True, "needs_raster": False,
            "summary": "Loads the on-device model. Returns a status, never hangs.",
        },
        "set_mode": dict(fast_free, summary="Switch interactive or automatic."),
        "list_object_classes": dict(
            fast_free,
            summary="The validated words detect_auto accepts as object_class."),
        "describe_object_class": dict(
            fast_free, summary="Everything known about one object class."),
        "detect": {
            "spends": True, "slow": False, "needs_raster": True,
            "summary": "One point, one object, saved straight to a GeoPackage.",
        },
        "detect_points": {
            "spends": True, "slow": False, "needs_raster": True,
            "summary": "Several points, one object. Negative points cut parts off.",
        },
        "detect_auto": {
            "spends": True, "slow": True, "needs_raster": True,
            "summary": "Sweeps a zone for every instance of a class. Minutes.",
        },
        "set_auto_zone": dict(
            fast_free, needs_raster=True,
            summary="Set the zone and quote its km2, rough duration and allowance."),
        "auto_detect_status": dict(
            fast_free, summary="Poll a zone run, or wait up to 50 s for its end."),
        "cancel_auto": dict(
            fast_free, summary="Stop a run, keeping what it already produced."),
        "export_polygon": dict(
            fast_free, summary="Write one polygon to a GeoPackage layer."),
        "export_recipe": dict(fast_free, summary="Pack a run's intent into a token."),
        "run_from_recipe": {
            "spends": True, "slow": True, "needs_raster": True,
            "summary": "Replay a run from its token.",
        },
        "refine_settings": dict(
            fast_free, summary="Read the current shape-cleanup settings."),
        "apply_refine": dict(
            fast_free, summary="Reshape the objects of an open review."),
        "review_status": dict(fast_free, summary="What an open review holds."),
        "review_objects": dict(
            fast_free,
            summary="List an open review's objects with their indexes, a page at a time."),
        "review_filter": dict(
            fast_free, summary="Re-filter an open review by confidence and size."),
        "set_display_mode": dict(fast_free, summary="Recolour an open review."),
        "review_remove_object": dict(
            fast_free, summary="Drop one object from an open review."),
        "review_merge_objects": dict(
            fast_free, summary="Join several objects into one. Free."),
        "review_undo_last": dict(fast_free, summary="Take back the last correction."),
        "review_clear_corrections": dict(
            fast_free, summary="Take back every correction of this review."),
        "undo_last_point": dict(
            fast_free, summary="Take back the last click of a panel session."),
    }
    for method, entry in notes.items():
        served_summary = dial_text("tuning.agent.method_summaries", method, 400)
        if served_summary:
            entry["summary"] = served_summary
    return notes


_GUIDE_TEXT = (
    "AI Segmentation turns imagery into vector polygons: one object from a point,"
    " or every instance of a class in a zone.\n"
    "Call the methods in the order agent_workflow_steps() gives, and read"
    " agent_method_notes() for what each one spends and how long it blocks.\n"
    "The full guide loads once the plugin has fetched its configuration over the network.\n"
)




def agent_guide_text() -> str:












    from .core.server_dials import dial_text

    served = dial_text("tuning.agent", "guide_text", 16000)
    return served or _GUIDE_TEXT
