"""Processing algorithm: outline the one object under a point.

A thin adapter onto SegmentationMCPAPI.detect. No segmentation logic lives
here, and none should.

The facade saves what it finds and adds that layer to the project itself, so
this algorithm declares no vector destination: a Processing sink copy of the
same outline would be a second layer of the same geometry, without the style
and the metadata the plugin put on the first one.
"""
from __future__ import annotations

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputNumber,
    QgsProcessingOutputString,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterPoint,
    QgsProcessingParameterRasterLayer,
    QgsProject,
)

from .algorithm_support import (
    FACADE_MISSING_MESSAGE,
    PLAN_HELP_LINE,
    SEGMENTATION_SEARCH_TAGS,
    add_produced_layer_output,
    layer_created_since,
    load_model_briefly,
    main_thread_run_refusal,
    no_threading_algorithm_flags,
    project_layer_ids,
    raise_on_facade_error,
    ready_segmentation_facade,
    refuse_when_threading_unsafe,
    segmentation_facade,
    status_algorithm_label,
    zone_algorithm_label,
)


class SegmentPointAlgorithm(QgsProcessingAlgorithm):
    """One point, one outline: the object sitting under a map point, as a polygon."""

    INPUT = "INPUT"
    POINT = "POINT"
    DISCARD_UNSAVED = "DISCARD_UNSAVED"
    INSTANCE_COUNT = "INSTANCE_COUNT"
    SCORE = "SCORE"
    STATUS = "STATUS"
    LAYER_NAME = "LAYER_NAME"
    OUTPUT_LAYER = "OUTPUT_LAYER"
    SAVED_FILE = "SAVED_FILE"
    POLYGON_WKT = "POLYGON_WKT"
    POLYGON_CRS = "POLYGON_CRS"

    def createInstance(self):
        return SegmentPointAlgorithm()

    # Published interface. Callers hardcode "terralab:segmentpoint", so this
    # string never changes.
    def name(self):
        return "segmentpoint"

    # A generic MCP server searches algorithms by plain substring over the id
    # and this label, and nothing else. So the words a person actually types
    # have to be in here, not only in tags or in the help text.
    def displayName(self):
        return "Vectorize one object: outline a building or feature under a point (AI)"

    def shortDescription(self):
        return "Trace the outline of the single object sitting under a map point, using AI."

    def group(self):
        return "AI segmentation"

    def groupId(self):
        return "aisegmentation"

    def tags(self):
        return SEGMENTATION_SEARCH_TAGS + ["point", "click", "outline", "trace", "one object"]

    # The facade drives the plugin panel and blocks on an event loop, so this
    # algorithm can only run on the main thread.
    def flags(self):
        return no_threading_algorithm_flags(super().flags())

    # Refuse rather than run on a build where that flag does not exist.
    def canExecute(self):
        return main_thread_run_refusal()

    def shortHelpString(self):
        return (
            "Traces the outline of the one object under a point, and returns it as a polygon.\n\n"
            "You give it an imagery layer and a point on the object. The AI reads the picture "
            "around that point and returns the shape of what it finds there, with a score "
            "between 0 and 1.\n\n"
            "The point coordinates are read in the project CRS. The point has to fall on the "
            "object itself, not beside it.\n\n"
            "Imagery it needs: an aerial or satellite raster, or a web map layer, where the "
            "object is visible.\n\n"
            "This one saves the object it finds, and saving is what counts against the plan "
            "while the AI runs on TerraLab's servers. A session running on this computer costs "
            "nothing.\n\n"
            "What it returns: no file output. The plugin saves the outline itself and adds that "
            "layer to the project, so this algorithm reports OUTPUT_LAYER (that layer, to chain "
            "into the next algorithm), LAYER_NAME and SAVED_FILE, the GeoPackage holding it, "
            "alongside SCORE (how sure the AI is), INSTANCE_COUNT, STATUS, POLYGON_WKT, the "
            "outline as text for a caller that wants the geometry without opening a file, and "
            "POLYGON_CRS, the CRS that text is in.\n\n"
            "Prefer this over drawing the shape by hand when you want one building, one field, "
            f"one pond or one roof traced accurately. Use '{zone_algorithm_label()}' "
            "when you want all objects of a kind over an area.\n\n"
            f"This one usually answers in a few seconds. Run '{status_algorithm_label()}' "
            "first: it tells you whether the model is installed and which imagery layer is "
            "selected.\n\n"
            "Open the AI Segmentation panel once to install the model and sign in.\n\n"
            + PLAN_HELP_LINE
        )

    def helpUrl(self):
        return "https://terra-lab.ai/ai-segmentation?utm_source=qgis&utm_medium=processing&utm_campaign=toolbox"

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT, "Imagery layer"))
        self.addParameter(QgsProcessingParameterPoint(
            self.POINT, "Point on the object"))
        # Off by default, the same answer the panel gives when it has nobody to
        # ask: outlining on another layer restarts the session and throws away
        # every polygon saved in it but not yet exported.
        self.addParameter(QgsProcessingParameterBoolean(
            self.DISCARD_UNSAVED,
            "Allow restarting on another imagery layer, dropping outlines not yet exported",
            defaultValue=False))

        self.addOutput(QgsProcessingOutputNumber(
            self.INSTANCE_COUNT, "Number of objects found"))
        self.addOutput(QgsProcessingOutputNumber(
            self.SCORE, "Score of the outline"))
        self.addOutput(QgsProcessingOutputString(
            self.STATUS, "Status"))
        self.addOutput(QgsProcessingOutputString(
            self.LAYER_NAME, "Name of the layer added to the project"))
        add_produced_layer_output(
            self, self.OUTPUT_LAYER, "The layer added to the project")
        self.addOutput(QgsProcessingOutputString(
            self.SAVED_FILE, "GeoPackage the plugin wrote"))
        self.addOutput(QgsProcessingOutputString(
            self.POLYGON_WKT, "The outline as well-known text"))
        # WKT alone carries no CRS, so an outline read straight out of
        # POLYGON_WKT used to land wherever the reader assumed.
        self.addOutput(QgsProcessingOutputString(
            self.POLYGON_CRS, "CRS the outline is expressed in"))

    def processAlgorithm(self, parameters, context, feedback):
        refuse_when_threading_unsafe(feedback)
        api = segmentation_facade()
        if api is None:
            feedback.reportError(FACADE_MISSING_MESSAGE, fatalError=True)
            raise QgsProcessingException(FACADE_MISSING_MESSAGE)

        # Cancelling while the model comes up has to end the run, not carry on
        # into a call that spends a credit. The wait below reads Cancel too, so
        # a stop lands within a second of the click.
        if feedback.isCanceled():
            feedback.pushInfo("Cancelled before the model was asked for. Nothing was spent.")
            return self._empty_result("cancelled before the run started")

        # A model still coming up is an answer, not a reason to hold the
        # Toolbox: the detection call would wait on the facade's own long cap.
        still_loading = load_model_briefly(api, feedback)
        if feedback.isCanceled():
            feedback.pushInfo("Cancelled while the model was loading. Nothing was spent.")
            return self._empty_result("cancelled before the run started")
        if still_loading:
            return self._answer_while_the_model_loads(feedback, still_loading)

        api = ready_segmentation_facade(feedback)

        raster = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if raster is None:
            raise QgsProcessingException(self.invalidRasterError(parameters, self.INPUT))

        # The facade reads the point in the canvas CRS and reprojects it to the
        # raster itself, so hand it canvas coordinates and nothing else.
        canvas_crs = self._canvas_crs()
        point = self.parameterAsPoint(parameters, self.POINT, context, canvas_crs)

        feedback.pushInfo(f"Looking at ({point.x():.2f}, {point.y():.2f}) on {raster.name()}.")

        # The last moment cancelling is free: the call below saves what it finds,
        # and saving is what the account pays for. A deliberate stop is reported
        # rather than raised, the same as the zone algorithm: an error here reads
        # as a run that broke, and this one never started.
        if feedback.isCanceled():
            feedback.pushInfo("Cancelled before the point was sent. Nothing was spent.")
            return self._empty_result("cancelled before the run started")

        discard_unsaved = self.parameterAsBoolean(
            parameters, self.DISCARD_UNSAVED, context)

        before = project_layer_ids()
        # The layer id, not the name: Processing already resolved the exact
        # layer, and two rasters in a project may carry the same name.
        result = api.detect(
            point.x(), point.y(), layer_name=raster.id(),
            discard_unsaved=discard_unsaved)
        raise_on_facade_error(feedback, result, "Point segmentation")

        if not result.get("detected") or not result.get("polygon_wkt"):
            message = result.get("message") or "No object was found under that point."
            feedback.reportError(message, fatalError=True)
            raise QgsProcessingException(message)

        export_error = str(result.get("export_error") or "")
        if export_error:
            feedback.pushWarning(
                f"The object was outlined but saving it failed: {export_error}. "
                "The outline is still returned as POLYGON_WKT.")

        produced = layer_created_since(before, result.get("exported_layer"))
        score = float(result.get("score") or 0.0)
        feedback.setProgress(100)
        feedback.pushInfo(f"Outlined one object, score {score:.3f}.")

        return {
            self.INSTANCE_COUNT: 1,
            self.SCORE: score,
            self.STATUS: "completed" if produced is not None else "outlined but not saved",
            self.LAYER_NAME: produced.name() if produced is not None else "",
            self.OUTPUT_LAYER: produced.id() if produced is not None else "",
            self.SAVED_FILE: str(result.get("exported_file") or ""),
            self.POLYGON_WKT: str(result.get("polygon_wkt") or ""),
            self.POLYGON_CRS: str(result.get("crs") or ""),
        }

    def _empty_result(self, status: str) -> dict:
        """Every output at its empty value, with one sentence saying why.

        A run that stops before it sends anything still has to answer on every
        key it declared: a caller reading LAYER_NAME off a missing key gets a
        KeyError instead of the reason the run did nothing.
        """
        return {
            self.INSTANCE_COUNT: 0,
            self.SCORE: 0.0,
            self.STATUS: status,
            self.LAYER_NAME: "",
            self.OUTPUT_LAYER: "",
            self.SAVED_FILE: "",
            self.POLYGON_WKT: "",
            self.POLYGON_CRS: "",
        }

    def _answer_while_the_model_loads(self, feedback, detail: str) -> dict:
        """Report a model still coming up, instead of holding the Toolbox on it."""
        message = (
            f"{detail} Nothing was sent and nothing was spent. Run this "
            "algorithm again in a moment: the model keeps loading in the "
            f"background. Run '{status_algorithm_label()}' to see when it is "
            "ready."
        )
        feedback.pushWarning(message)
        return self._empty_result("still loading the model")

    @staticmethod
    def _canvas_crs() -> QgsCoordinateReferenceSystem:
        """The CRS the facade expects a click in: the canvas one, project CRS otherwise."""
        try:
            import qgis.utils
            iface = getattr(qgis.utils, "iface", None)
            if iface is not None:
                crs = iface.mapCanvas().mapSettings().destinationCrs()
                if crs is not None and crs.isValid():
                    return crs
        except (RuntimeError, AttributeError):
            pass
        return QgsProject.instance().crs()
