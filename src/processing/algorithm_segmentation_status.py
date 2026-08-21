"""Processing algorithm: report whether AI segmentation can run right now.

The cheap one. It answers in milliseconds, never touches the network, never
spends anything, and names the two algorithms that do the work. A caller that
cannot list algorithms still learns their ids from here.
"""
from __future__ import annotations

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputBoolean,
    QgsProcessingOutputNumber,
    QgsProcessingOutputString,
    QgsProcessingParameterRasterLayer,
)

from .algorithm_support import (
    FACADE_MISSING_MESSAGE,
    PLAN_HELP_LINE,
    POINT_ALGORITHM_ID,
    SEGMENTATION_SEARCH_TAGS,
    ZONE_ALGORITHM_ID,
    main_thread_run_refusal,
    no_threading_algorithm_flags,
    point_algorithm_label,
    refuse_when_threading_unsafe,
    segmentation_facade,
    zone_algorithm_label,
)


class SegmentationStatusAlgorithm(QgsProcessingAlgorithm):
    """Readiness check: what is installed, what is missing, what to run next."""

    INPUT = "INPUT"
    INSTALLED = "INSTALLED"
    READY = "READY"
    STATE = "STATE"
    ACTION_REQUIRED = "ACTION_REQUIRED"
    CREDITS_REMAINING = "CREDITS_REMAINING"
    RASTER_LAYER = "RASTER_LAYER"
    AVAILABLE_RASTER_LAYERS = "AVAILABLE_RASTER_LAYERS"
    NEXT_ALGORITHMS = "NEXT_ALGORITHMS"

    def createInstance(self):
        return SegmentationStatusAlgorithm()

    # Published interface. Callers hardcode "terralab:segmentationstatus", so
    # this string never changes.
    def name(self):
        return "segmentationstatus"

    # A generic MCP server searches algorithms by plain substring over the id
    # and this label, and nothing else. So the words a person actually types
    # have to be in here, not only in tags or in the help text.
    def displayName(self):
        return "Check AI segmentation and object detection status"

    def shortDescription(self):
        return "Report whether AI segmentation is installed, signed in and ready, and what to run next."

    def group(self):
        return "AI segmentation"

    def groupId(self):
        return "aisegmentation"

    def tags(self):
        return SEGMENTATION_SEARCH_TAGS + [
            "status", "ready", "check", "credits", "setup", "installed", "health",
        ]

    # Reads the live plugin panel, so it belongs on the main thread like the
    # two algorithms it introduces.
    def flags(self):
        return no_threading_algorithm_flags(super().flags())

    # Refuse rather than run on a build where that flag does not exist.
    def canExecute(self):
        return main_thread_run_refusal()

    def shortHelpString(self):
        return (
            "Run this first. It says whether AI segmentation can work right now, and it "
            "answers in milliseconds without spending anything.\n\n"
            "TerraLab AI adds AI segmentation to QGIS: you point at imagery, and it returns "
            "vector polygons around the objects it sees. Buildings, trees, swimming pools, "
            "solar panels, cars, boats, roads, fields. It replaces tracing shapes by hand.\n\n"
            "Two algorithms do the work, and you can run either by id:\n"
            f"  {ZONE_ALGORITHM_ID} - '{zone_algorithm_label()}'. Give it an "
            "imagery layer, a rectangle and a word such as building. It returns one polygon "
            "per object found in that rectangle. It runs on the AI service and can take "
            "several minutes, during which QGIS stays busy. Never start it twice: a second "
            "run costs the user money.\n"
            f"  {POINT_ALGORITHM_ID} - '{point_algorithm_label()}'. Give it an "
            "imagery layer and a point on one object. It returns that object's outline, "
            "usually in a few seconds.\n\n"
            "Neither writes a file output: both let the plugin save the result and add that "
            "one layer to the project, and both report its name and the GeoPackage path.\n\n"
            "What this one returns: INSTALLED and READY (true or false), STATE (one of "
            "MODEL_NOT_DOWNLOADED, MODEL_NOT_LOADED, NO_RASTER_LAYER, READY), ACTION_REQUIRED "
            "(what a person has to do when READY is false), CREDITS_REMAINING, RASTER_LAYER "
            "(the imagery the plugin is set to), AVAILABLE_RASTER_LAYERS (one name per line) "
            "and NEXT_ALGORITHMS. "
            "CREDITS_REMAINING is -1 when the plugin did not report a number, which usually "
            "means nobody is signed in yet.\n\n"
            "When READY is false, read ACTION_REQUIRED out to the user and stop. The setup "
            "steps happen in the AI Segmentation panel in QGIS, not from here, and no "
            "algorithm can install anything.\n\n"
            "Imagery needed by the other two: an aerial or satellite raster, or a web map "
            "layer, zoomed in far enough that the objects are visible.\n\n"
            + PLAN_HELP_LINE
        )

    def helpUrl(self):
        return "https://terra-lab.ai/ai-segmentation?utm_source=qgis&utm_medium=processing&utm_campaign=toolbox"

    def initAlgorithm(self, config=None):
        # Optional: naming a layer here only makes the answer say whether that
        # layer is the one the plugin would use. Nothing is read from it.
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT, "Imagery layer to ask about (optional)", optional=True))

        self.addOutput(QgsProcessingOutputBoolean(
            self.INSTALLED, "Plugin installed"))
        self.addOutput(QgsProcessingOutputBoolean(
            self.READY, "Ready to run"))
        self.addOutput(QgsProcessingOutputString(
            self.STATE, "State"))
        self.addOutput(QgsProcessingOutputString(
            self.ACTION_REQUIRED, "What the user has to do"))
        self.addOutput(QgsProcessingOutputNumber(
            self.CREDITS_REMAINING, "Objects left on the plan this month"))
        self.addOutput(QgsProcessingOutputString(
            self.RASTER_LAYER, "Imagery layer the plugin is set to"))
        self.addOutput(QgsProcessingOutputString(
            self.AVAILABLE_RASTER_LAYERS, "Raster layers in the project"))
        self.addOutput(QgsProcessingOutputString(
            self.NEXT_ALGORITHMS, "Algorithms that do the segmentation"))

    def processAlgorithm(self, parameters, context, feedback):
        refuse_when_threading_unsafe(feedback)
        api = segmentation_facade()
        if api is None:
            feedback.reportError(FACADE_MISSING_MESSAGE, fatalError=True)
            raise QgsProcessingException(FACADE_MISSING_MESSAGE)

        status = api.get_status()
        if not isinstance(status, dict):
            message = "The status call returned nothing usable."
            feedback.reportError(message, fatalError=True)
            raise QgsProcessingException(message)

        state = str(status.get("state") or "")
        ready = bool(status.get("ready"))
        action = str(status.get("action_required") or "")

        raster = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        selected = str(status.get("raster_layer") or "")
        if raster is not None and selected and raster.name() != selected:
            action = (
                f"{action} " if action else ""
            ) + (
                f"You asked about '{raster.name()}' but the plugin is set to '{selected}'. "
                "Pass the layer name to the segmentation algorithm to override it."
            )

        available = status.get("available_raster_layers")
        if not available:
            # An older build names them only when nothing is selected, and a
            # caller reading this on a READY plugin then sees no layer at all.
            available = self._project_raster_names() or (
                [selected] if selected else [])

        feedback.pushInfo(f"State: {state or 'unknown'}. Ready: {ready}.")
        if action:
            feedback.pushInfo(action)

        return {
            self.INSTALLED: bool(status.get("installed")),
            self.READY: ready,
            self.STATE: state,
            self.ACTION_REQUIRED: action,
            self.CREDITS_REMAINING: self._credits_number(status),
            self.RASTER_LAYER: selected,
            # One name per line: a QGIS layer name may hold a comma, and a
            # comma-joined list cannot be split back into names.
            self.AVAILABLE_RASTER_LAYERS: "\n".join(
                str(name) for name in (available or [])),
            self.NEXT_ALGORITHMS: f"{ZONE_ALGORITHM_ID}, {POINT_ALGORITHM_ID}",
        }

    @staticmethod
    def _project_raster_names() -> list[str]:
        """Every raster layer name in the project, read here rather than asked for."""
        try:
            from qgis.core import QgsProject, QgsRasterLayer
            return [
                layer.name()
                for layer in QgsProject.instance().mapLayers().values()
                if isinstance(layer, QgsRasterLayer)
            ]
        except (RuntimeError, AttributeError, ImportError):
            return []

    @staticmethod
    def _credits_number(status: dict) -> int:
        """Objects left this month as a plain number; -1 when the plugin did not report one."""
        raw = status.get("auto_credits_remaining")
        if raw is None:
            return -1
        try:
            return int(raw)
        except (TypeError, ValueError):
            return -1
