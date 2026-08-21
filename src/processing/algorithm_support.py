"""Shared pieces every AI Segmentation Processing algorithm needs.

The algorithms are thin adapters onto ``src/mcp_api.py``. Everything that both
of them need to agree on lives here: how the facade is found, how a facade
error becomes a Processing error, and the flag that keeps the run on the main
thread.
"""
from __future__ import annotations

import re

from qgis.core import (
    Qgis,
    QgsApplication,
    QgsProcessingAlgorithm,
    QgsProcessingException,
)

from ..mcp_api import AISEG_KEYS, _find_plugin

# Same shape as AISEG_REGISTER_URL in mcp_api, tagged for this surface so a
# visit that started in the Processing Toolbox can be told apart.
PROCESSING_PRICING_URL = (
    "https://terra-lab.ai/pricing?utm_source=qgis&utm_medium=processing&utm_campaign=toolbox"
)

# The published algorithm ids. Callers hardcode them, so they never change,
# and every help text that names one reads it from here rather than quoting a
# label, which does change.
STATUS_ALGORITHM_ID = "terralab:segmentationstatus"
ZONE_ALGORITHM_ID = "terralab:segmentzone"
POINT_ALGORITHM_ID = "terralab:segmentpoint"

# The one plan line every help text ends on. Kept in one place so the wording
# cannot drift between algorithms. It states a limit of the algorithm, which a
# caller has to know before it spends anything. It is deliberately not a pitch:
# a description read by an AI assistant should say what the tool does and what
# stops it, and nothing about how the assistant ought to behave.
PLAN_HELP_LINE = (
    "Free accounts are capped by area each month, and a run stops when the cap "
    f"is reached. Run '{STATUS_ALGORITHM_ID}' to read what is left before "
    f"anything is spent. Plan limits: {PROCESSING_PRICING_URL}"
)

# How long the facade lets one blocking zone run go before it gives up on it.
# The facade owns the ceiling; this is the value the algorithm asks for, so the
# number named in the help text is the number in force.
_ZONE_RUN_TIMEOUT_FALLBACK_S = 280


class _ServedZoneRunTimeout:
    """Int-like stand-in for the timeout, resolved at every use.

    The zone algorithm imports the name once, at plugin load, so a plain int
    would freeze the value for the whole session. Resolving on ``int()`` and
    on formatting keeps the number the help text names and the number the run
    asks for the same, the read-through shape ``server_dials.ServerDialSet``
    already uses for sets.
    """

    def _seconds(self) -> int:
        try:
            from ..core.server_dials import dial_in_range

            return int(dial_in_range(
                "processing.zone_run_timeout_s", _ZONE_RUN_TIMEOUT_FALLBACK_S, 30, 3600))
        except Exception:  # noqa: BLE001 - the shipped ceiling always works  # nosec B110
            return _ZONE_RUN_TIMEOUT_FALLBACK_S

    def __int__(self) -> int:
        return self._seconds()

    def __index__(self) -> int:
        return self._seconds()

    def __str__(self) -> str:
        return str(self._seconds())

    def __repr__(self) -> str:
        return repr(self._seconds())

    def __format__(self, spec: str) -> str:
        return format(self._seconds(), spec)


ZONE_RUN_TIMEOUT_SECONDS = _ServedZoneRunTimeout()

# Shown when the build exposes neither name for the "run me on the main thread"
# flag. The algorithms drive Qt widgets, so a background run is a crash, not a
# slowdown: refusing is the safe answer.
MAIN_THREAD_FLAG_MISSING_MESSAGE = (
    "This QGIS build exposes no way to keep the run on the main thread. These algorithms "
    "drive the AI Segmentation panel, and driving it from a background thread would take "
    "QGIS down, so the run is refused. Update QGIS, or use the panel itself."
)

# Shown when the run is already on a background thread. The flag above asks
# Processing to keep it off one; a caller that builds its own context and runs
# the algorithm itself never went through Processing to be asked.
BACKGROUND_THREAD_MESSAGE = (
    "This run started on a background thread. These algorithms drive the AI Segmentation "
    "panel, and driving Qt widgets from a background thread would take QGIS down, so the "
    "run is refused. Run it from the Processing Toolbox, or call it from the main thread."
)


class _ServedSearchTags(list):
    """Read-through list: ``+`` also appends the served extra tags.

    The algorithms build their tags as ``SEGMENTATION_SEARCH_TAGS + [...]``
    and import the name once, so the union has to happen inside the object at
    use time. Union-only: a deploy can teach the fleet a new search word and
    can never take a shipped one away. Plain iteration sees the shipped
    entries only, like ``ServerDialSet``.
    """

    def _with_served_extras(self) -> list[str]:
        shipped = list(self)
        try:
            from ..core.server_dials import dial_list

            served = dial_list("processing.search_tags_extra", (), normalize=str.lower)
            lowered = {tag.lower() for tag in shipped}
            shipped += sorted(tag for tag in served if tag not in lowered)
        except Exception:  # noqa: BLE001 - the shipped words always work  # nosec B110
            pass
        return shipped

    def __add__(self, other):
        return self._with_served_extras() + list(other)


# The words a person types in the Toolbox filter, and the words a model reads
# when it lists algorithms through a generic MCP server.
SEGMENTATION_SEARCH_TAGS = _ServedSearchTags([
    "segmentation",
    "ai",
    "sam",
    "detect",
    "detection",
    "buildings",
    "trees",
    "digitize",
    "vectorize",
    "extract",
    "footprint",
    "machine learning",
    "raster to vector",
    "terralab",
])

# The message shown when the plugin object itself is missing, so the provider
# and the algorithms report the same cause with the same words.
FACADE_MISSING_MESSAGE = (
    "The AI Segmentation plugin is not loaded. Enable it in Plugins > Manage and Install Plugins, "
    f"then reopen this algorithm. (Looked for: {', '.join(AISEG_KEYS)})"
)


def segmentation_facade():
    """Return the plugin's stable facade object, or None when it is absent.

    Reuses mcp_api's own lookup: QGIS registers a plugin under its install
    folder name, which differs between a release install and a checkout.
    """
    plugin = _find_plugin()
    if plugin is None:
        return None
    return getattr(plugin, "mcp_api", None)


def main_thread_only_flag():
    """The "run me on the main thread" flag, or None on a build carrying neither name.

    The flag moved from QgsProcessingAlgorithm.Flag to Qgis.ProcessingAlgorithmFlag,
    so resolve it by name. Never a bare int: QGIS 4 rejects arithmetic on one.
    """
    scope = getattr(Qgis, "ProcessingAlgorithmFlag", None)
    flag = getattr(scope, "NoThreading", None) if scope is not None else None
    if flag is None:
        flag = getattr(QgsProcessingAlgorithm, "FlagNoThreading", None)
    return flag


def no_threading_algorithm_flags(base_flags):
    """Add the "run me on the main thread" flag to an algorithm's own flags.

    The facade drives Qt widgets and blocks on an event loop, so a background
    thread is not an option. When the flag is missing the base flags come back
    unchanged, and ``main_thread_run_refusal`` is what stops the run instead:
    raising here would break plugin load on a build nobody has tested yet.
    """
    flag = main_thread_only_flag()
    if flag is None:
        return base_flags
    return base_flags | flag


def main_thread_run_refusal() -> tuple[bool, str]:
    """Processing's canExecute answer: False when the main-thread flag is missing."""
    if main_thread_only_flag() is None:
        return False, MAIN_THREAD_FLAG_MISSING_MESSAGE
    return True, ""


def on_the_gui_thread() -> bool:
    """True when the caller is on the thread QGIS builds its widgets on.

    Only a proven answer opens the way. An unanswerable question is read as no,
    because what follows drives Qt widgets, and doing that from the wrong
    thread takes QGIS down with the user's project in it. Every QGIS that can
    load this plugin answers, so the closed door costs nothing.
    """
    try:
        from qgis.PyQt.QtCore import QThread

        app = QgsApplication.instance()
        if app is None:
            return False
        return QThread.currentThread() is app.thread()
    except Exception:  # noqa: BLE001 - an unanswered question is read as "not the GUI thread"
        return False


def refuse_when_threading_unsafe(feedback) -> None:
    """Stop a run that could land on a background thread.

    Two walls, because the flag only asks Processing to keep the run on the
    main thread. canExecute refuses in the Toolbox when the flag is missing,
    and a caller reaching processAlgorithm another way is already running
    wherever it chose to, so where it actually landed is checked as well.
    """
    if main_thread_only_flag() is None:
        feedback.reportError(MAIN_THREAD_FLAG_MISSING_MESSAGE, fatalError=True)
        raise QgsProcessingException(MAIN_THREAD_FLAG_MISSING_MESSAGE)
    if not on_the_gui_thread():
        feedback.reportError(BACKGROUND_THREAD_MESSAGE, fatalError=True)
        raise QgsProcessingException(BACKGROUND_THREAD_MESSAGE)


def zone_algorithm_label() -> str:
    """The zone algorithm's live label, read from the class so no help text quotes a stale one."""
    from .algorithm_segment_zone import SegmentZoneAlgorithm
    return SegmentZoneAlgorithm().displayName()


def point_algorithm_label() -> str:
    """The point algorithm's live label, read from the class so no help text quotes a stale one."""
    from .algorithm_segment_point import SegmentPointAlgorithm
    return SegmentPointAlgorithm().displayName()


def status_algorithm_label() -> str:
    """The status algorithm's live label, read from the class so no help text quotes a stale one."""
    from .algorithm_segmentation_status import SegmentationStatusAlgorithm
    return SegmentationStatusAlgorithm().displayName()


def ready_segmentation_facade(feedback):
    """Return the facade only when the plugin says it can work right now.

    Raises rather than starting a run the panel would refuse, and repeats the
    facade's own action_required text so the user reads one instruction.
    """
    refuse_when_threading_unsafe(feedback)
    api = segmentation_facade()
    if api is None:
        feedback.reportError(FACADE_MISSING_MESSAGE, fatalError=True)
        raise QgsProcessingException(FACADE_MISSING_MESSAGE)

    status = api.get_status()
    if not status.get("installed") or not status.get("ready") or status.get("state") != "READY":
        message = status.get("action_required") or (
            "AI Segmentation is not ready. Open the AI Segmentation panel and finish the setup."
        )
        state = status.get("state")
        if state:
            message = f"{message} (state: {state})"
        feedback.reportError(message, fatalError=True)
        raise QgsProcessingException(message)
    return api


# How long a point run waits for the model to come up before it answers "still
# loading" instead of holding the Toolbox. The shipped number is the fallback;
# the one in force is served, so it can be retuned without a release.
_POINT_MODEL_LOAD_FALLBACK_S = 20

# What the wait reports when the model is not there yet, whether the cap ran
# out or the user stopped waiting. The caller says which of the two it was.
STILL_LOADING_MESSAGE = "The model is still loading."


def point_model_load_timeout_seconds() -> int:
    """Seconds a point run gives the model to load, resolved at every call."""
    try:
        from ..core.server_dials import dial_in_range

        return int(dial_in_range(
            "processing.point_model_load_timeout_s",
            _POINT_MODEL_LOAD_FALLBACK_S, 1, 120))
    except Exception:  # noqa: BLE001 - the shipped ceiling always works
        return _POINT_MODEL_LOAD_FALLBACK_S


def load_model_briefly(api, feedback) -> str:
    """Bring the model up under a short cap. Empty string when it is ready.

    An algorithm that lets the facade load the model inside the detection call
    inherits the facade's own long cap, and the Toolbox is frozen for the whole
    of it with nothing to read and no way out. So the load is asked for here,
    under a cap short enough to answer, and a model still coming up is reported
    back rather than waited on.

    The wait is cut into one-second slices so Cancel is read while the model
    comes up, instead of once the whole cap has gone by. Asking again while a
    load runs changes nothing: the loader keeps the worker it already has.
    """
    status = api.get_status()
    if not isinstance(status, dict) or status.get("state") != "MODEL_NOT_LOADED":
        return ""
    if feedback.isCanceled():
        return STILL_LOADING_MESSAGE
    seconds = point_model_load_timeout_seconds()
    feedback.pushInfo(
        f"The model is not loaded yet. Waiting up to {seconds} seconds for it.")
    detail = ""
    for _ in range(max(1, int(seconds))):
        if feedback.isCanceled():
            return STILL_LOADING_MESSAGE
        outcome = api.load_model(timeout_s=1)
        if not isinstance(outcome, dict):
            break
        if outcome.get("loaded"):
            return ""
        detail = str(outcome.get("_error") or "")
        # A slice that comes back before its second is up has nothing left to
        # wait for: the load either refused to start or has already stopped.
        # Asking again would start it over, once per slice.
        if float(outcome.get("waited_s") or 0.0) < 0.9:
            break
    return detail or STILL_LOADING_MESSAGE


def raise_on_facade_error(feedback, result: dict, action: str) -> dict:
    """Turn the facade's ``_error`` key into a Processing failure.

    Facade calls never raise: a failure comes back as a plain dict. Nothing
    downstream would notice, so every call site passes its result through here.
    """
    if not isinstance(result, dict):
        message = f"{action} returned nothing usable."
        feedback.reportError(message, fatalError=True)
        raise QgsProcessingException(message)
    error = result.get("_error")
    if error:
        message = f"{action} failed: {error}"
        feedback.reportError(message, fatalError=True)
        raise QgsProcessingException(message)
    return result


def charged_timeout_seconds(result) -> int | None:
    """Seconds named by the facade's own timeout failure, or None for any other error.

    A facade failure comes back as a plain dict, never as an exception. The
    timeout is the one failure that leaves the account already charged, so the
    call site has to be able to tell it apart and say so.
    """
    error = str((result or {}).get("_error") or "") if isinstance(result, dict) else ""
    if "timed out after" not in error.lower():
        return None
    digits = re.search(r"(\d+)", error)
    return int(digits.group(1)) if digits else int(ZONE_RUN_TIMEOUT_SECONDS)


def project_layer_ids() -> set[str]:
    """Every layer id in the project right now, for a before-and-after comparison."""
    from qgis.core import QgsProject
    return set(QgsProject.instance().mapLayers().keys())


def layer_created_since(before: set[str], preferred_name: str | None = None):
    """The layer this run added to the project, or None when it cannot be told.

    The id difference is the lookup, never the name: QGIS layer names are not
    unique, so two runs producing the same friendly name would otherwise report
    on the first one. ``preferred_name`` only picks between several additions.
    """
    from qgis.core import QgsProject
    project = QgsProject.instance()
    added = [project.mapLayer(layer_id) for layer_id in project.mapLayers()
             if layer_id not in before]
    added = [layer for layer in added if layer is not None]
    if len(added) == 1:
        return added[0]
    for layer in added:
        if preferred_name and layer.name() == preferred_name:
            return layer
    return None


def layer_file_path(layer) -> str:
    """The file a layer reads from, without the OGR layer suffix. Empty when it has none."""
    if layer is None:
        return ""
    source = str(layer.source() or "")
    return source.split("|", 1)[0]


def integer_parameter_type():
    """The "whole number" flavour of QgsProcessingParameterNumber, on QGIS 3 and 4.

    Never None. A None reaches the parameter constructor as "no type given",
    which builds a decimal box where a whole number belongs, so a build
    carrying neither name falls back to the value both flavours agree on.
    """
    from qgis.core import QgsProcessingParameterNumber
    scope = getattr(Qgis, "ProcessingNumberParameterType", None)
    value = getattr(scope, "Integer", None) if scope is not None else None
    if value is None:
        value = getattr(QgsProcessingParameterNumber, "Integer", None)
    if value is None:
        # The enum has held this value since it was introduced, and a bare int
        # is only ever read here, never combined with another flag.
        value = 1
    return value


def parameter_left_unset(parameters, name: str) -> bool:
    """True when the caller gave this optional parameter no usable value.

    Three things mean "unset" and only one of them is None. An empty box in the
    Toolbox arrives as an empty string, and a value read out of a table arrives
    as a QGIS NULL, which is neither None nor empty. Read as a number, both
    become zero, and zero is a real value every numeric parameter here refuses.
    """
    if name not in parameters:
        return True
    value = parameters.get(name)
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    # QGIS NULL is a null QVariant, which is neither None nor empty. Asked by
    # name so nothing is imported that a given build may not carry.
    is_null = getattr(value, "isNull", None)
    if callable(is_null):
        try:
            return bool(is_null())
        except (RuntimeError, TypeError):
            return False
    return False


def add_produced_layer_output(algorithm, name: str, description: str) -> None:
    """Declare the output carrying the layer this run added to the project.

    A modeler chains one algorithm onto the next through a layer output, and a
    string output cannot be chained. The class is guarded because a build
    without it must still register its algorithms; a result key for an output
    that was never declared is simply ignored.
    """
    try:
        from qgis.core import QgsProcessingOutputVectorLayer
    except ImportError:
        return
    algorithm.addOutput(QgsProcessingOutputVectorLayer(name, description))
