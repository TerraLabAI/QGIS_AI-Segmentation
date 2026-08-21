"""Processing algorithm: find every object of one kind inside a zone.

A thin adapter onto SegmentationMCPAPI.detect_auto. No segmentation logic
lives here, and none should.

The plugin writes the result itself, as a styled GeoPackage table added to the
project. This algorithm declares no vector destination, so the run produces
that one layer and nothing else: a Processing sink copy of it would be a second
layer of the same geometry, without the style, the column aliases or the layer
metadata the plugin put on the first one.
"""
from __future__ import annotations

from qgis.core import (
    QgsGeometry,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputNumber,
    QgsProcessingOutputString,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterExtent,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
)

from .algorithm_support import (
    PLAN_HELP_LINE,
    SEGMENTATION_SEARCH_TAGS,
    STATUS_ALGORITHM_ID,
    ZONE_RUN_TIMEOUT_SECONDS,
    add_produced_layer_output,
    charged_timeout_seconds,
    integer_parameter_type,
    layer_created_since,
    layer_file_path,
    main_thread_run_refusal,
    no_threading_algorithm_flags,
    parameter_left_unset,
    point_algorithm_label,
    project_layer_ids,
    raise_on_facade_error,
    ready_segmentation_facade,
    status_algorithm_label,
)


class SegmentZoneAlgorithm(QgsProcessingAlgorithm):
    """Every object of one class inside a rectangle, as polygons."""

    INPUT = "INPUT"
    EXTENT = "EXTENT"
    CLASS = "CLASS"
    DETAIL = "DETAIL"
    INSTANCE_COLORS = "INSTANCE_COLORS"
    INSTANCE_COUNT = "INSTANCE_COUNT"
    # Tiles the service answered, never a charge: a run is charged for the
    # surface of its zone whatever it finds, and this side never learns the
    # final figure, so no output here may claim one.
    TILES_PROCESSED = "TILES_PROCESSED"
    STATUS = "STATUS"
    LAYER_NAME = "LAYER_NAME"
    OUTPUT_LAYER = "OUTPUT_LAYER"
    SAVED_FILE = "SAVED_FILE"

    def createInstance(self):
        return SegmentZoneAlgorithm()

    # Published interface. Callers hardcode "terralab:segmentzone", so this
    # string never changes.
    def name(self):
        return "segmentzone"

    # A generic MCP server searches algorithms by plain substring over the id
    # and this label, and nothing else. So the words a person actually types
    # have to be in here, not only in tags or in the help text.
    def displayName(self):
        return "Detect and extract building footprints, trees and vegetation in a zone (AI)"

    def shortDescription(self):
        return "Detect and outline every building, tree or other object inside a zone, using AI."

    def group(self):
        return "AI segmentation"

    def groupId(self):
        return "aisegmentation"

    def tags(self):
        return SEGMENTATION_SEARCH_TAGS + ["zone", "batch", "objects", "count", "area"]

    # The facade drives the plugin panel and blocks on an event loop, so this
    # algorithm can only run on the main thread.
    def flags(self):
        return no_threading_algorithm_flags(super().flags())

    # Refuse rather than run on a build where that flag does not exist.
    def canExecute(self):
        return main_thread_run_refusal()

    def shortHelpString(self):
        return (
            f"This run can take up to {ZONE_RUN_TIMEOUT_SECONDS} seconds, and QGIS stays busy "
            "until it ends. Wait for it. Never start it again while it is running: a second run "
            "costs the user money.\n\n"
            "Finds every object of one kind inside a zone and returns them as polygons.\n\n"
            "You give it an imagery layer and a rectangle, and you type what to look for. "
            "The AI reads the picture and returns one polygon per object it finds, with the "
            "class you asked for and a score.\n\n"
            "What to type in 'What to detect': one plain word or short phrase, for example "
            "building, tree, swimming pool, solar panel, car, boat, road.\n\n"
            "Imagery it needs: an aerial or satellite raster, or a web map layer, where the "
            "objects are visible. Coarse imagery gives poor results, so zoom in first.\n\n"
            "Cancel: pressing Cancel before the run starts stops it and spends nothing. "
            "Pressing it during the run stops the run within a second or so, and the objects "
            "already found are kept: they are written to a layer the same way a "
            "finished run writes one, and this algorithm reports them under INSTANCE_COUNT, "
            "TILES_PROCESSED and LAYER_NAME with STATUS 'cancelled'. The zone is charged when "
            "the run starts, so a cancel keeps what the service had already found without "
            "lowering the bill. The panel's own Cancel button does the same thing.\n\n"
            f"If the run passes {ZONE_RUN_TIMEOUT_SECONDS} seconds it stops with a timeout. The "
            "account has still been charged for the whole zone, and the objects may "
            "already be in the AI Segmentation panel, so look there before running it again. "
            f"Run '{STATUS_ALGORITHM_ID}' to see what is left on the plan.\n\n"
            "What it returns: no file output. The plugin writes the results itself, as one "
            "styled layer added to the project, so this algorithm reports OUTPUT_LAYER (that "
            "layer, to chain into the next algorithm), LAYER_NAME and SAVED_FILE, the "
            "GeoPackage holding it, alongside INSTANCE_COUNT, TILES_PROCESSED "
            "and STATUS. TILES_PROCESSED counts the imagery tiles the AI answered; it is not "
            "the cost, because the run is charged for the surface of its zone whatever it "
            f"finds. Run '{STATUS_ALGORITHM_ID}' after the run to read the real balance. "
            "Open SAVED_FILE to read the polygons from disk.\n\n"
            "Prefer this over drawing polygons by hand whenever you need many objects over an "
            "area: all building footprints in a district, every tree in a park, every pool in a "
            f"suburb. Use '{point_algorithm_label()}' when you only want one object.\n\n"
            f"Run '{status_algorithm_label()}' first. It answers in milliseconds and tells "
            "you whether this one can run at all.\n\n"
            "This runs on the AI service, so it needs an internet connection and a signed-in "
            "TerraLab account. Open the AI Segmentation panel once to install and sign in.\n\n"
            + PLAN_HELP_LINE
        )

    def helpUrl(self):
        return "https://terra-lab.ai/ai-segmentation?utm_source=qgis&utm_medium=processing&utm_campaign=toolbox"

    def initAlgorithm(self, config=None):
        from ..core.tile_manager import MAX_DETAIL_LEVEL

        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT, "Imagery layer"))
        self.addParameter(QgsProcessingParameterExtent(
            self.EXTENT, "Zone to scan"))
        self.addParameter(QgsProcessingParameterString(
            self.CLASS,
            "What to detect (e.g. building, tree, swimming pool, solar panel)",
            defaultValue="building"))
        # Left empty on purpose: the service picks a grid that suits the object
        # and the zone. A number here overrides that pick.
        self.addParameter(QgsProcessingParameterNumber(
            self.DETAIL,
            "Detail level (leave empty to let the AI choose)",
            type=integer_parameter_type(),
            minValue=1,
            maxValue=MAX_DETAIL_LEVEL,
            defaultValue=None,
            optional=True))
        # Off by default: it costs the canvas one symbol per object, and a run
        # of a few objects reads fine in one colour.
        self.addParameter(QgsProcessingParameterBoolean(
            self.INSTANCE_COLORS,
            "One colour per object (so objects that touch read apart)",
            defaultValue=False))

        # Scalar outputs, so a caller that only reads strings still learns what
        # the run did and where the polygons went.
        self.addOutput(QgsProcessingOutputNumber(
            self.INSTANCE_COUNT, "Number of objects found"))
        self.addOutput(QgsProcessingOutputNumber(
            self.TILES_PROCESSED, "Imagery tiles the AI answered (not the cost)"))
        self.addOutput(QgsProcessingOutputString(
            self.STATUS, "Status"))
        self.addOutput(QgsProcessingOutputString(
            self.LAYER_NAME, "Name of the layer added to the project"))
        add_produced_layer_output(
            self, self.OUTPUT_LAYER, "The layer added to the project")
        self.addOutput(QgsProcessingOutputString(
            self.SAVED_FILE, "GeoPackage the plugin wrote"))

    def processAlgorithm(self, parameters, context, feedback):
        api = ready_segmentation_facade(feedback)

        raster = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if raster is None:
            raise QgsProcessingException(self.invalidRasterError(parameters, self.INPUT))

        object_class = (self.parameterAsString(parameters, self.CLASS, context) or "").strip()
        if not object_class:
            message = "Type what to detect, for example building or tree."
            feedback.reportError(message, fatalError=True)
            raise QgsProcessingException(message)

        # The facade reads the zone in the raster layer's CRS, so ask
        # Processing for the extent already expressed in that CRS.
        extent = self.parameterAsExtent(parameters, self.EXTENT, context, raster.crs())
        if extent is None or extent.isEmpty():
            message = "The zone is empty. Draw a rectangle over the imagery."
            feedback.reportError(message, fatalError=True)
            raise QgsProcessingException(message)
        zone_wkt = QgsGeometry.fromRect(extent).asWkt()

        # An empty box and a NULL read as zero, and zero tiles is not a grid.
        # Left unset, the run picks a grid that suits the object and the zone.
        detail = None
        if not parameter_left_unset(parameters, self.DETAIL):
            detail = self.parameterAsInt(parameters, self.DETAIL, context)
            if detail < 1:
                message = (
                    f"Detail level must be 1 or more, got {detail}. Leave it empty "
                    "to let the AI choose a grid that suits the object and the zone.")
                feedback.reportError(message, fatalError=True)
                raise QgsProcessingException(message)

        instance_colors = self.parameterAsBoolean(parameters, self.INSTANCE_COLORS, context)

        feedback.pushInfo(f"Looking for '{object_class}' on {raster.name()}.")
        feedback.pushInfo(
            "The AI service answers this in one go, so the progress bar stays still and QGIS "
            "stays busy. This can take several minutes. Do not start it again. The zone is "
            "charged when the run starts; Cancel stops the run and keeps what was found.")

        # The last moment cancelling is free. Past this line the call blocks
        # until the service answers, and the whole zone has been charged.
        if feedback.isCanceled():
            # A deliberate stop, so report it rather than raise: an error here
            # reads as a run that broke, and this one never started.
            feedback.pushInfo("Cancelled before the zone was sent. Nothing was spent.")
            return {
                self.INSTANCE_COUNT: 0,
                self.TILES_PROCESSED: 0,
                self.STATUS: "cancelled before the run started",
                self.LAYER_NAME: "",
                self.OUTPUT_LAYER: "",
                self.SAVED_FILE: "",
            }

        before = project_layer_ids()
        # The layer id, not the name: Processing already resolved the exact
        # layer, and a run must not be refused because a second raster in the
        # project happens to carry the same name.
        result = api.detect_auto(
            zone_wkt=zone_wkt,
            object_class=object_class,
            layer_name=raster.id(),
            detail=detail,
            timeout_s=ZONE_RUN_TIMEOUT_SECONDS,
            should_cancel=feedback.isCanceled,
            instance_colors=instance_colors,
        )
        # Say when the colours were asked for and not given, rather than
        # handing back a layer that quietly looks like the option was ignored.
        if isinstance(result, dict) and result.get("instance_colors") is False:
            note = result.get("instance_colors_note")
            if note:
                feedback.pushInfo(note)
        if isinstance(result, dict) and result.get("cancelled"):
            return self._finish_a_cancelled_run(feedback, result, before)
        self._report_a_charged_timeout(feedback, result)
        raise_on_facade_error(feedback, result, "Zone segmentation")

        instances = int(result.get("instances") or 0)
        tiles_processed = int(result.get("tiles_processed") or 0)
        feedback.pushInfo(
            f"Found {instances} object(s) across {tiles_processed} processed tile(s). "
            f"Run '{STATUS_ALGORITHM_ID}' to read what is left on the plan: the run is "
            "charged for the surface of its zone, so the tile count is not the cost.")

        produced = layer_created_since(before, result.get("layer_name"))
        if produced is None:
            feedback.pushWarning(
                "The run finished but added no layer to the project. Look in the AI "
                "Segmentation panel: results waiting for review live there.")
            return {
                self.INSTANCE_COUNT: instances,
                self.TILES_PROCESSED: tiles_processed,
                self.STATUS: "no layer added",
                self.LAYER_NAME: "",
                self.OUTPUT_LAYER: "",
                self.SAVED_FILE: "",
            }

        feedback.setProgress(100)
        feedback.pushInfo(f"Added to the project: {produced.name()}.")
        return {
            self.INSTANCE_COUNT: instances,
            self.TILES_PROCESSED: tiles_processed,
            self.STATUS: f"completed, {produced.featureCount()} feature(s) in one layer",
            self.LAYER_NAME: produced.name(),
            self.OUTPUT_LAYER: produced.id(),
            self.SAVED_FILE: layer_file_path(produced),
        }

    def _finish_a_cancelled_run(self, feedback, result: dict, before: set[str]) -> dict:
        """End a run the user stopped, reporting what it kept.

        Cancelling is a deliberate act, not a failure, so the algorithm reports
        instead of raising. The zone was charged when the run started, so a
        cancel lowers nothing; the objects already found are written exactly as
        a finished run's are.
        """
        instances = int(result.get("instances") or 0)
        tiles_processed = int(result.get("tiles_processed") or 0)
        produced = layer_created_since(before, result.get("layer_name"))
        if produced is None:
            feedback.pushInfo(
                "Cancelled. The AI service had processed "
                f"{tiles_processed} tile(s) before the stop, and nothing was added to the "
                "project. Open the AI Segmentation panel and look for a run waiting for "
                f"review before starting another one. Run '{STATUS_ALGORITHM_ID}' to see "
                "what is left on the plan.")
            return {
                self.INSTANCE_COUNT: instances,
                self.TILES_PROCESSED: tiles_processed,
                self.STATUS: "cancelled, no layer added",
                self.LAYER_NAME: "",
                self.OUTPUT_LAYER: "",
                self.SAVED_FILE: "",
            }
        feedback.pushInfo(
            f"Cancelled. Kept the {instances} object(s) already found, from the "
            f"{tiles_processed} tile(s) processed before the stop. The zone was "
            "charged when the run started, so the stop does not lower the bill. "
            f"Added to the project: {produced.name()}.")
        return {
            self.INSTANCE_COUNT: instances,
            self.TILES_PROCESSED: tiles_processed,
            self.STATUS: f"cancelled, {produced.featureCount()} feature(s) in one layer",
            self.LAYER_NAME: produced.name(),
            self.OUTPUT_LAYER: produced.id(),
            self.SAVED_FILE: layer_file_path(produced),
        }

    @staticmethod
    def _report_a_charged_timeout(feedback, result) -> None:
        """Say that a timed-out run was paid for, before the error ends the run.

        The facade tears the run down on timeout and reports a plain failure.
        Without this the user reads "failed" and never learns that the account
        was charged, nor that the objects may still be waiting in the panel.
        """
        seconds = charged_timeout_seconds(result)
        if seconds is None:
            return
        feedback.reportError(
            f"The run passed {seconds} seconds and was stopped. The account HAS been charged "
            "for the whole zone, and this is not a free retry. The objects it "
            "found may still be recoverable: open the AI Segmentation panel and look for a "
            f"run waiting for review before starting another one. Run '{STATUS_ALGORITHM_ID}' "
            "to see what is left on the plan.")
