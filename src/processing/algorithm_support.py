






from __future__ import annotations

import os
import re

from qgis.core import (
    Qgis,
    QgsApplication,
    QgsProcessingAlgorithm,
    QgsProcessingException,
)

from ..core.i18n import tr
from ..mcp_api import AISEG_KEYS, _find_plugin



PROCESSING_PRICING_URL = (
    "https://terra-lab.ai/pricing?utm_source=qgis&utm_medium=processing&utm_campaign=toolbox"
)


def processing_pricing_url() -> str:

    from ..core.server_dials import dial_url

    return dial_url("tuning.links.processing_pricing", PROCESSING_PRICING_URL)





STATUS_ALGORITHM_ID = "terralab:segmentationstatus"
ZONE_ALGORITHM_ID = "terralab:segmentzone"
POINT_ALGORITHM_ID = "terralab:segmentpoint"








PLAN_HELP_LINE = (
    "Free accounts are capped by area each month, and a run stops when the cap "
    f"is reached. Run '{STATUS_ALGORITHM_ID}' to read what is left before "
    f"anything is spent. Plan limits: {PROCESSING_PRICING_URL}"
)


def plan_help_line() -> str:






    from ..core.server_dials import dial_copy

    fallback = (
        "Free accounts are capped by area each month, and a run stops when the cap "
        f"is reached. Run '{STATUS_ALGORITHM_ID}' to read what is left before "
        f"anything is spent. Plan limits: {processing_pricing_url()}"
    )
    return dial_copy("copy.processing.plan_help_line", fallback)





_ZONE_RUN_TIMEOUT_FALLBACK_S = 280


class _ServedZoneRunTimeout:









    def _seconds(self) -> int:
        try:
            from ..core.server_dials import dial_in_range

            return int(dial_in_range(
                "processing.zone_run_timeout_s", _ZONE_RUN_TIMEOUT_FALLBACK_S, 30, 3600))
        except Exception:  # noqa: BLE001  # nosec B110
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




MAIN_THREAD_FLAG_MISSING_MESSAGE = (
    "This QGIS build exposes no way to keep the run on the main thread. These algorithms "
    "drive the AI Segmentation panel, and driving it from a background thread would take "
    "QGIS down, so the run is refused. Update QGIS, or use the panel itself."
)




BACKGROUND_THREAD_MESSAGE = (
    "This run started on a background thread. These algorithms drive the AI Segmentation "
    "panel, and driving Qt widgets from a background thread would take QGIS down, so the "
    "run is refused. Run it from the Processing Toolbox, or call it from the main thread."
)


class _ServedSearchTags(list):









    def _with_served_extras(self) -> list[str]:
        shipped = list(self)
        try:
            from ..core.server_dials import dial_list

            served = dial_list("processing.search_tags_extra", (), normalize=str.lower)
            lowered = {tag.lower() for tag in shipped}
            shipped += sorted(tag for tag in served if tag not in lowered)
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return shipped

    def __add__(self, other):
        return self._with_served_extras() + list(other)




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



FACADE_MISSING_MESSAGE = (
    "The AI Segmentation plugin is not loaded. Enable it in Plugins > Manage and Install Plugins, "
    f"then reopen this algorithm. (Looked for: {', '.join(AISEG_KEYS)})"
)


def segmentation_facade():





    plugin = _find_plugin()
    if plugin is None:
        return None
    return getattr(plugin, "mcp_api", None)


def main_thread_only_flag():





    scope = getattr(Qgis, "ProcessingAlgorithmFlag", None)
    flag = getattr(scope, "NoThreading", None) if scope is not None else None
    if flag is None:
        flag = getattr(QgsProcessingAlgorithm, "FlagNoThreading", None)
    return flag


def no_threading_algorithm_flags(base_flags):







    flag = main_thread_only_flag()
    if flag is None:
        return base_flags
    return base_flags | flag


def main_thread_run_refusal() -> tuple[bool, str]:

    if main_thread_only_flag() is None:
        return False, MAIN_THREAD_FLAG_MISSING_MESSAGE
    return True, ""


def on_the_gui_thread() -> bool:







    try:
        from qgis.PyQt.QtCore import QThread

        app = QgsApplication.instance()
        if app is None:
            return False
        return QThread.currentThread() is app.thread()
    except Exception:  # noqa: BLE001
        return False


def refuse_when_threading_unsafe(feedback) -> None:







    if main_thread_only_flag() is None:
        feedback.reportError(MAIN_THREAD_FLAG_MISSING_MESSAGE, fatalError=True)
        raise QgsProcessingException(MAIN_THREAD_FLAG_MISSING_MESSAGE)
    if not on_the_gui_thread():
        feedback.reportError(BACKGROUND_THREAD_MESSAGE, fatalError=True)
        raise QgsProcessingException(BACKGROUND_THREAD_MESSAGE)



_PROCESSING_HELP_URL = (
    "https://terra-lab.ai/ai-segmentation?utm_source=qgis&utm_medium=processing&utm_campaign=toolbox"
)


def processing_help_url() -> str:

    from ..core.server_dials import dial_url

    return dial_url("tuning.links.processing_help", _PROCESSING_HELP_URL)


def zone_algorithm_label() -> str:

    from .algorithm_segment_zone import SegmentZoneAlgorithm
    return SegmentZoneAlgorithm().displayName()


def point_algorithm_label() -> str:

    from .algorithm_segment_point import SegmentPointAlgorithm
    return SegmentPointAlgorithm().displayName()


def status_algorithm_label() -> str:

    from .algorithm_segmentation_status import SegmentationStatusAlgorithm
    return SegmentationStatusAlgorithm().displayName()


def ready_segmentation_facade(feedback):





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





_POINT_MODEL_LOAD_FALLBACK_S = 20



STILL_LOADING_MESSAGE = "The model is still loading."


def point_model_load_timeout_seconds() -> int:

    try:
        from ..core.server_dials import dial_in_range

        return int(dial_in_range(
            "processing.point_model_load_timeout_s",
            _POINT_MODEL_LOAD_FALLBACK_S, 1, 120))
    except Exception:  # noqa: BLE001
        return _POINT_MODEL_LOAD_FALLBACK_S


def load_model_briefly(api, feedback) -> str:












    status = api.get_status()
    if not isinstance(status, dict) or status.get("state") != "MODEL_NOT_LOADED":
        return ""
    if feedback.isCanceled():
        return STILL_LOADING_MESSAGE
    seconds = point_model_load_timeout_seconds()
    feedback.pushInfo(
        tr("The model is not loaded yet. Waiting up to {0} seconds for it.").format(seconds))
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



        if float(outcome.get("waited_s") or 0.0) < 0.9:
            break
    return detail or STILL_LOADING_MESSAGE


def raise_on_facade_error(feedback, result: dict, action: str) -> dict:





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






    error = str((result or {}).get("_error") or "") if isinstance(result, dict) else ""
    if "timed out after" not in error.lower():
        return None
    digits = re.search(r"(\d+)", error)
    return int(digits.group(1)) if digits else int(ZONE_RUN_TIMEOUT_SECONDS)


def project_layer_ids() -> set[str]:

    from qgis.core import QgsProject
    return set(QgsProject.instance().mapLayers().keys())


def layer_created_since(before: set[str], preferred_name: str | None = None):






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

    if layer is None:
        return ""
    source = str(layer.source() or "")
    path = source.split("|", 1)[0]



    if path and os.path.isfile(path):
        return os.path.normpath(path)
    return path


def integer_parameter_type():






    from qgis.core import QgsProcessingParameterNumber
    scope = getattr(Qgis, "ProcessingNumberParameterType", None)
    value = getattr(scope, "Integer", None) if scope is not None else None
    if value is None:
        value = getattr(QgsProcessingParameterNumber, "Integer", None)
    if value is None:


        value = 1
    return value


def parameter_left_unset(parameters, name: str) -> bool:







    if name not in parameters:
        return True
    value = parameters.get(name)
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True


    is_null = getattr(value, "isNull", None)
    if callable(is_null):
        try:
            return bool(is_null())
        except (RuntimeError, TypeError):
            return False
    return False


def add_produced_layer_output(algorithm, name: str, description: str) -> None:







    try:
        from qgis.core import QgsProcessingOutputVectorLayer
    except ImportError:
        return
    algorithm.addOutput(QgsProcessingOutputVectorLayer(name, description))
