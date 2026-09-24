













from __future__ import annotations

import difflib
import functools
import os

from qgis.core import QgsProject, QgsRasterLayer

from .mcp_api_auto import SegmentationAutoMixin
from .mcp_api_export import SegmentationExportMixin
from .mcp_api_guide import agent_guide_text, agent_method_notes, agent_workflow_steps
from .mcp_api_lifecycle import SegmentationLifecycleMixin
from .mcp_api_manual import SegmentationManualMixin
from .mcp_api_presets import SegmentationPresetsMixin
from .mcp_api_recipe import SegmentationRecipeMixin
from .mcp_api_refine import SegmentationRefineMixin
from .mcp_api_review import SegmentationReviewMixin




_PLUGIN_FOLDER = os.path.basename(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


AISEG_KEYS = list(dict.fromkeys([_PLUGIN_FOLDER, "AI_Segmentation", "QGIS_AI-Segmentation"]))
AISEG_REGISTER_URL = "https://terra-lab.ai/ai-segmentation?utm_source=qgis&utm_medium=mcp&utm_campaign=ai-agent"









API_VERSION = 4

PUBLIC_METHODS = [
    "apply_refine",
    "auto_detect_status",
    "cancel_auto",
    "capabilities",
    "describe_object_class",
    "detect",
    "detect_auto",
    "detect_points",
    "export_polygon",
    "export_recipe",
    "get_status",
    "guide",
    "install_status",
    "list_object_classes",
    "load_model",
    "refine_settings",
    "review_clear_corrections",
    "review_filter",
    "review_merge_objects",
    "review_objects",
    "review_remove_object",
    "review_status",
    "review_undo_last",
    "run_from_recipe",
    "set_auto_zone",
    "set_display_mode",
    "set_mode",
    "undo_last_point",
]





LAYER_NAME_ARGUMENT_NOTE = (
    "layer_name is the imagery layer to read, not a name for the output layer."
)






def not_found_error(
    kind: str,
    given: str,
    available: list[str],
    note: str | None = None,
    valid_range: tuple[int, int] | None = None,
) -> dict:






















    text = str(given)
    if valid_range is not None:
        low, high = valid_range
        if high < low:
            message = f"{kind.capitalize()} {text} is out of range: there is nothing here to address yet."
        else:
            message = (
                f"{kind.capitalize()} {text} is out of range. Valid values run "
                f"from {low} to {high}."
            )
        return {"_error": message if note is None else f"{message} {note}"}

    names = [str(name) for name in (available or [])]
    message = f"No {kind} called '{text}'."
    if note:
        message += f" {note}"



    from .core.server_dials import dial_in_range

    suggestion_n = dial_in_range("tuning.agent.suggestion_count", 3, 1, 8)
    suggestion_cutoff = dial_in_range("tuning.agent.suggestion_cutoff", 0.5, 0.3, 0.9)
    names_cap = dial_in_range("tuning.agent.available_names_cap", 8, 3, 30)
    folded = {name.casefold(): name for name in reversed(names)}
    matched = difflib.get_close_matches(
        text.casefold(), list(folded), n=suggestion_n, cutoff=suggestion_cutoff)
    suggestions = [folded[key] for key in matched]
    if suggestions:
        listed = ", ".join(f"'{name}'" for name in suggestions)
        return {"_error": f"{message} Did you mean: {listed}?",
                "_suggestions": suggestions}
    if names:
        listed = ", ".join(f"'{name}'" for name in names[:names_cap])
        if len(names) > names_cap:
            listed += f" (+{len(names) - names_cap} more)"
        return {"_error": f"{message} Available: {listed}."}
    return {"_error": f"{message} There is no {kind} here to choose from."}


def coerce_bool_param(name: str, value) -> tuple[bool | None, dict | None]:








    if isinstance(value, bool):
        return value, None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True, None
        if lowered in ("false", "0", "no"):
            return False, None
    return None, {"_error": f"{name} must be a boolean (true/false), got {value!r}."}


def raster_layer_names() -> list[str]:





    try:
        return [
            layer.name()
            for layer in QgsProject.instance().mapLayers().values()
            if isinstance(layer, QgsRasterLayer)
        ]
    except (RuntimeError, AttributeError):
        return []






def raster_layer_by_id_or_name(text):

    wanted = str(text or "").strip()
    if not wanted:
        return None, {"_error": (
            f"layer_name must be a non-empty string. {LAYER_NAME_ARGUMENT_NOTE}")}
    try:
        project = QgsProject.instance()
        by_id = project.mapLayer(wanted)
        if isinstance(by_id, QgsRasterLayer):
            return by_id, None
        matches = [
            layer for layer in project.mapLayers().values()
            if isinstance(layer, QgsRasterLayer) and layer.name() == wanted
        ]
    except (RuntimeError, AttributeError):
        matches = []
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        listed = ", ".join(f"'{layer.id()}'" for layer in matches)
        return None, {"_error": (
            f"{len(matches)} raster layers in this project are called "
            f"'{wanted}', so the name does not say which one to read. Pass one "
            f"of these layer ids instead: {listed}.")}
    return None, not_found_error(
        "raster layer", wanted, raster_layer_names(),
        note=LAYER_NAME_ARGUMENT_NOTE,
    )


def _find_plugin():
    import qgis.utils
    for key in AISEG_KEYS:
        plugin = qgis.utils.plugins.get(key)
        if plugin is not None:
            return plugin
    return None






def _never_raises(func):
    @functools.wraps(func)
    def _wrapped(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Exception as err:  # noqa: BLE001


            detail = str(err).strip() or f"{type(err).__name__} (no message)"
            return {"_error": f"{func.__name__} failed: {detail}"}



        if isinstance(result, dict) and "_error" in result:
            if not str(result["_error"] or "").strip():
                result["_error"] = f"{func.__name__} failed and said nothing about why."
        return result
    return _wrapped


class SegmentationMCPAPI(
    SegmentationManualMixin,
    SegmentationAutoMixin,
    SegmentationExportMixin,
    SegmentationPresetsMixin,
    SegmentationRecipeMixin,
    SegmentationRefineMixin,
    SegmentationReviewMixin,
    SegmentationLifecycleMixin,
):


    def __init__(self, plugin):
        self._plugin = plugin

    def capabilities(self) -> dict:

















        return {
            "api_version": API_VERSION,
            "methods": self._methods_this_build_carries(),
            "workflow": agent_workflow_steps(),
            "method_notes": agent_method_notes(),
            "guide": (
                "Call guide() for the plain-text manual on getting good "
                "results: how to choose the word a zone run searches for, when "
                "drawn examples beat a word, and how detail should match the "
                "size of the objects."
            ),
        }

    def guide(self) -> str:










        return agent_guide_text()

    def get_status(self) -> dict:







        plugin = self._plugin

        status = {"installed": True, "api_version": API_VERSION}


        model_downloaded = False
        try:
            from .core.checkpoint_manager import checkpoint_exists
            model_downloaded = checkpoint_exists()
        except Exception:
            pass  # nosec B110

        try:
            predictor_loaded = plugin.predictor is not None
        except (RuntimeError, AttributeError):
            predictor_loaded = False

        if not model_downloaded and not predictor_loaded:
            from .core.server_dials import dial_url

            status.update({
                "ready": False,
                "state": "MODEL_NOT_DOWNLOADED",
                "action_required": (
                    "The AI model is not installed yet. Open the AI Segmentation"
                    " panel and click Install."
                ),
                "register_url": dial_url("tuning.links.agent_register", AISEG_REGISTER_URL),
            })
            return status

        if not predictor_loaded:
            status.update({
                "ready": False,
                "state": "MODEL_NOT_LOADED",



                "action_required": (
                    "Call load_model() to load it, which costs nothing. A"
                    " person does the same thing by opening the AI Segmentation"
                    " panel and clicking 'Start Semi-Auto AI Segmentation'."
                ),
            })
            return status

        status["model_loaded"] = True


        raster_layer = getattr(plugin, "_current_layer", None)
        if raster_layer is None:
            try:
                dock = getattr(plugin, "dock_widget", None)
                if dock and hasattr(dock, "layer_combo"):
                    raster_layer = dock.layer_combo.currentLayer()
            except (RuntimeError, AttributeError):
                raster_layer = None

        if raster_layer is None:
            available = raster_layer_names()
            if available:
                status.update({
                    "ready": False,
                    "state": "NO_RASTER_LAYER",
                    "model_loaded": True,
                    "action_required": (
                        f"No raster layer selected. Available: {', '.join(available)}."
                        " Pass layer_name to detect(), detect_points() or"
                        " detect_auto(), or select one in the panel."
                    ),
                    "available_raster_layers": available,
                })
            else:
                status.update({
                    "ready": False,
                    "state": "NO_RASTER_LAYER",
                    "model_loaded": True,
                    "action_required": "No raster layer in the project. The user needs to load one first.",
                    "available_raster_layers": [],
                })
            return status



        try:
            extent = raster_layer.extent()
            status.update({
                "ready": True,
                "state": "READY",
                "raster_layer": raster_layer.name(),



                "available_raster_layers": raster_layer_names(),
                "raster_extent": {
                    "xmin": extent.xMinimum(),
                    "ymin": extent.yMinimum(),
                    "xmax": extent.xMaximum(),
                    "ymax": extent.yMaximum(),
                },
                "raster_crs": raster_layer.crs().authid(),
            })
        except (RuntimeError, AttributeError):
            status.update({
                "ready": False,
                "state": "NO_RASTER_LAYER",
                "model_loaded": True,
                "action_required": (
                    "The selected raster layer has been removed from the "
                    "project. Select another one in the panel, or pass "
                    "layer_name."
                ),
                "available_raster_layers": raster_layer_names(),
            })
            return status


        try:
            from .core.activation_manager import is_plugin_activated
            if is_plugin_activated():
                dock = getattr(plugin, "dock_widget", None)
                status["mode"] = "interactive"
                if dock and hasattr(dock, "_mode"):
                    status["mode"] = dock._mode.value
                if dock and hasattr(dock, "_auto_credits") and dock._auto_credits is not None:
                    status["auto_credits_remaining"] = dock._auto_credits
                if dock and hasattr(dock, "_auto_is_subscriber"):
                    status["auto_is_subscriber"] = dock._auto_is_subscriber
        except Exception:
            pass  # nosec B110

        return status

    def _methods_this_build_carries(self) -> list[str]:






        return [name for name in PUBLIC_METHODS
                if callable(getattr(self, name, None))]

    def _resolve_raster_layer(self, layer_name: str | None):

        if layer_name:
            layer, _err = raster_layer_by_id_or_name(layer_name)
            return layer
        try:
            return self._plugin._get_active_raster_layer()
        except (RuntimeError, AttributeError):
            return None






for _name in PUBLIC_METHODS:
    _method = getattr(SegmentationMCPAPI, _name, None)
    if _name != "guide" and callable(_method):
        setattr(SegmentationMCPAPI, _name, _never_raises(_method))
del _name, _method
