"""Programmatic API for MCP integration. No UI, no dialogs, no cursors.

This module provides a stable public interface for AI agents to control
AI Segmentation without touching the human UI.

The contract, in three lines. Every public method takes plain JSON-serialisable
arguments and returns a plain dict. No method ever raises: a failure comes back
under the key ``_error``. Nothing is ever removed or renamed, only added, and
``API_VERSION`` says which set a build carries.

The class is assembled here from the mixins in the sibling ``mcp_api_*``
modules, one per concern. Everything stays importable from this module, which
is where external callers and our own Processing algorithms look for it.
"""
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

# QGIS registers a plugin under its install folder name. The released folder is
# "AI_Segmentation"; a checkout installed under its repository folder registers
# under that name instead, so this module's own folder is tried first.
_PLUGIN_FOLDER = os.path.basename(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
AISEG_KEYS = [_PLUGIN_FOLDER, "AI_Segmentation", "QGIS_AI-Segmentation"]
AISEG_REGISTER_URL = "https://terra-lab.ai/ai-segmentation?utm_source=qgis&utm_medium=mcp&utm_campaign=ai-agent"

# Bumped when a method is added. A caller reads it from get_status() or
# capabilities() and negotiates, instead of sniffing signatures. Additive only:
# a method that ships here never leaves.
# 2: multi-point prompting, refine, review corrections, model lifecycle, guide.
# 3: the object-class catalogue (list_object_classes, describe_object_class).
API_VERSION = 3

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
    "review_remove_object",
    "review_status",
    "review_undo_last",
    "run_from_recipe",
    "set_auto_zone",
    "set_display_mode",
    "set_mode",
    "undo_last_point",
]


# The one sentence that stops a caller reading layer_name as "name the layer I
# want created". Passed to not_found_error as its note at every site that
# resolves that argument, so all of them say it the same way.
LAYER_NAME_ARGUMENT_NOTE = (
    "layer_name is the imagery layer to read, not a name for the output layer."
)


# Every "not found" answer in this API is built here, so a caller reads one
# shape whatever it looked for. Three states, and they are not the same: a near
# match is a typo the caller can fix, a full collection is a choice it can make,
# and an empty collection is a different problem entirely.
def not_found_error(
    kind: str,
    given: str,
    available: list[str],
    note: str | None = None,
    valid_range: tuple[int, int] | None = None,
) -> dict:
    """Build the API's ``_error`` for something the caller named and we cannot find.

    Parameters
    ----------
    kind : str
        What was looked for, in lower case: "raster layer", "display mode".
    given : str
        What the caller passed. Shown back quoted.
    available : list[str]
        Every name the caller could have passed. Empty is a valid state and
        gets its own sentence.
    note : str | None
        One sentence about the argument itself, for one a caller misreads.
        Placed right after the opening sentence.
    valid_range : tuple[int, int] | None
        For an integer index, the lowest and highest value that would work.
        The message then states a range and no names are listed.

    Returns
    -------
    dict with ``_error``, plus ``_suggestions`` when there were near matches.
    """
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
    # Folded on both sides: a caller that types Building where the catalogue
    # holds building has made a typo we can name, and a case-sensitive match
    # answered "no such thing" and listed the word it had just refused.
    folded = {name.casefold(): name for name in reversed(names)}
    matched = difflib.get_close_matches(text.casefold(), list(folded), n=3, cutoff=0.5)
    suggestions = [folded[key] for key in matched]
    if suggestions:
        listed = ", ".join(f"'{name}'" for name in suggestions)
        return {"_error": f"{message} Did you mean: {listed}?",
                "_suggestions": suggestions}
    if names:
        listed = ", ".join(f"'{name}'" for name in names[:8])
        if len(names) > 8:
            listed += f" (+{len(names) - 8} more)"
        return {"_error": f"{message} Available: {listed}."}
    return {"_error": f"{message} There is no {kind} here to choose from."}


def raster_layer_names() -> list[str]:
    """Every raster layer name in the project, in project order.

    Names are not unique in QGIS, so this can repeat a name. It is what a
    caller passes as layer_name, so it is reported as the project holds it.
    """
    try:
        return [
            layer.name()
            for layer in QgsProject.instance().mapLayers().values()
            if isinstance(layer, QgsRasterLayer)
        ]
    except (RuntimeError, AttributeError):
        return []


# A caller may hold a layer id, which is unique and stable, or a layer name,
# which is neither. The id is tried first, and two layers sharing a name are
# refused rather than resolved: picking the first match reads whichever imagery
# happens to sit earlier in the project, and the answer looks perfectly normal.
def raster_layer_by_id_or_name(text):
    """Resolve one raster layer from an id or a name, as (layer, error_or_None)."""
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


# The contract says a public method never raises. Hand-written guards inside a
# method cover what its author thought of; this covers the rest. functools.wraps
# keeps __name__, __doc__ and the signature, which agent_schema.py reads at
# runtime to build the machine-readable tool definitions.
def _never_raises(func):
    @functools.wraps(func)
    def _wrapped(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Exception as err:  # noqa: BLE001 - the contract is: never raise.
            # A raised object with no message of its own still has to name
            # something, or the caller is handed "detect failed: ".
            detail = str(err).strip() or f"{type(err).__name__} (no message)"
            return {"_error": f"{func.__name__} failed: {detail}"}
        # A blank _error is worse than a wrong one: it reads as a failure to
        # anything checking for the key and as nothing at all to a person, so
        # the caller cannot tell what refused it. Name the method instead.
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
    """Public API for MCP/headless access to AI Segmentation."""

    def __init__(self, plugin):
        self._plugin = plugin

    def capabilities(self) -> dict:
        """Say what this build can do and in what order to call it.

        Returns
        -------
        dict with keys:
            "api_version"  -- int, bumped when methods are added.
            "methods"      -- list[str], every public method this build has.
            "workflow"     -- list[dict], the normal order of calls, each with
                              ``step``, ``call``, ``why`` and ``optional``.
            "method_notes" -- dict, per method: ``spends`` (can use up the
                              account's monthly allowance), ``slow`` (can block
                              for minutes), ``needs_raster`` and ``summary``.
            "guide"        -- str, one line pointing at guide() for the prose.

        An external tool reads this once instead of probing signatures. Costs
        nothing.
        """
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
        """The plain-text manual on getting GOOD results, not merely valid ones.

        Covers the order of calls, how to pick the word a zone run searches
        for, when drawn examples beat a word, how framing changes what a run
        finds, how detail should match the size of the objects, and what this
        API deliberately will not do.

        Returns one string. Safe to print, or to hand straight to a language
        model as context. Costs nothing.
        """
        return agent_guide_text()

    def get_status(self) -> dict:
        """Check plugin readiness without touching UI.

        Returns ``installed``, ``api_version`` and either ``ready: True`` with
        the raster in use, or ``ready: False`` with a ``state`` and one
        ``action_required`` sentence naming what the person at this computer
        has to do. Costs nothing and calls no server.
        """
        plugin = self._plugin

        status = {"installed": True, "api_version": API_VERSION}

        # Check model downloaded
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
            status.update({
                "ready": False,
                "state": "MODEL_NOT_DOWNLOADED",
                "action_required": (
                    "The AI model is not installed yet. Open the AI Segmentation"
                    " panel and click Install."
                ),
                "register_url": AISEG_REGISTER_URL,
            })
            return status

        if not predictor_loaded:
            status.update({
                "ready": False,
                "state": "MODEL_NOT_LOADED",
                # A caller reading this used to be sent to the panel for
                # something it can do itself. The click stays here for a
                # person; load_model() is the same step without one.
                "action_required": (
                    "Call load_model() to load it, which costs nothing. A"
                    " person does the same thing by opening the AI Segmentation"
                    " panel and clicking 'Start Semi-Auto AI Segmentation'."
                ),
            })
            return status

        status["model_loaded"] = True

        # Check raster layer
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
                        " Pass layer_name to ai_segment_detect or select one in the panel."
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

        # A layer object whose C++ side is already gone answers every call with
        # a RuntimeError, and this used to escape as one.
        try:
            extent = raster_layer.extent()
            status.update({
                "ready": True,
                "state": "READY",
                "raster_layer": raster_layer.name(),
                # Named on every state, not only when nothing is selected: a
                # caller that wants to switch layers has to know what is there,
                # and a ready plugin is exactly when it asks.
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

        # Additive fields: mode + credits. Never remove or rename existing keys.
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
            pass  # nosec B110 -- additive fields; never break existing behavior

        return status

    def _methods_this_build_carries(self) -> list[str]:
        """The published names this object really answers to.

        PUBLIC_METHODS is the catalogue, not the inventory. A mixin that fails
        to import leaves its names in the catalogue and nothing behind them, so
        a caller was told about calls that raise AttributeError.
        """
        return [name for name in PUBLIC_METHODS
                if callable(getattr(self, name, None))]

    def _resolve_raster_layer(self, layer_name: str | None):
        """Return the named raster layer, or the plugin's active one, or None."""
        if layer_name:
            layer, _err = raster_layer_by_id_or_name(layer_name)
            return layer
        try:
            return self._plugin._get_active_raster_layer()
        except (RuntimeError, AttributeError):
            return None


# Wrap every published method once, on the assembled class, so a method added
# to any mixin is covered without its author remembering to decorate it. guide()
# returns a string by contract, so it is left alone: a dict there would break
# the one caller that prints it.
for _name in PUBLIC_METHODS:
    _method = getattr(SegmentationMCPAPI, _name, None)
    if _name != "guide" and callable(_method):
        setattr(SegmentationMCPAPI, _name, _never_raises(_method))
del _name, _method
