"""One polygon at a time inside the Manual (native QGIS) correction session.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); a mixin like its
siblings, so state lives on the plugin instance. Every method here is named
``_bridge_*`` or ``_*_bridge_isolation`` and none of those names exists in
another mixin.

QGIS scopes its digitizing tools to the ACTIVE LAYER, not to a selection, so a
neighbour that is still on that layer stays grabbable: the user can drag its
vertices while another polygon is open for editing. The only lever PyQGIS gives
us is the layer's subset, so the session puts one on and the other detections
leave the editable layer for as long as it runs.

They come back as a read-only context layer in the same dim grey the AI method
uses, because a map that loses every neighbour the moment an edit opens gives
the user nothing to judge the new shape against. Snapping is project-wide
during a session, so their borders still catch the cursor; nothing on that
layer can be edited.

Three constraints shape this file:

- ``QgsVectorLayer.setSubsetString`` does nothing on a layer already in edit
  mode. The subset goes on BEFORE ``startEditing()`` and comes off after the
  commit.
- A feature born inside the session (a hand-drawn polygon, the second half of a
  split) takes a det_id above every id the session opened with, so the subset
  keeps everything above that ceiling and everything with no id yet. Without
  those two clauses the fold, which reads the layer THROUGH the subset, would
  never see the new shape and it would be lost at Save.
- The subset and the context layer are cleared on every exit path. A subset
  left behind shows the review a single polygon for the rest of the session.
"""
from __future__ import annotations

from ...core.i18n import tr
from ..canvas_palette import SESSION_DIM_FILL_STR, SESSION_DIM_STROKE_STR


def bridge_isolation_expression(det_id: int, id_ceiling: int) -> str:
    """The subset leaving ONE polygon reachable on the review layer.

    Three clauses, one per shape the session has to keep: the polygon the user
    picked, a shape drawn or split off during the session (its det_id is minted
    above every id the session opened with), and one still waiting for the
    deferred write that gives it that id.
    """
    return (
        f'"det_id" = {int(det_id):d}'
        f' OR "det_id" > {int(id_ceiling):d}'
        ' OR "det_id" IS NULL'
    )


class BridgeIsolationMixin:
    """Scopes a Manual session to its own polygon, and paints the rest as
    read-only context."""

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _init_bridge_isolation_state(self) -> None:
        """Fresh isolation state; called once from the plugin __init__."""
        # The layer carrying our subset, so the clear targets the same one even
        # if the review layer has been swapped meanwhile.
        self._bridge_isolated_layer = None
        # The subset the layer had before us. Restored rather than blanked: the
        # review never sets one today, and a future one is not ours to drop.
        self._bridge_prior_subset = None
        self._bridge_context_layer = None

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def _isolate_bridge_target(self, layer, det_id) -> bool:
        """Take every other detection off the editable layer and redraw them as
        read-only context. Returns True when the subset went on.

        Call BEFORE ``startEditing()``: a layer already in edit mode refuses a
        subset. A False return leaves the layer whole, which is what the bridge
        did before this existed, never a half-applied state.
        """
        if layer is None or det_id is None:
            return False
        try:
            supports = getattr(layer.dataProvider(), "supportsSubsetString", None)
            if callable(supports) and not supports():
                return False
        except (RuntimeError, AttributeError):
            return False
        try:
            expression = bridge_isolation_expression(
                det_id, self._bridge_det_id_ceiling(layer))
        except (TypeError, ValueError):
            return False
        # Built while the layer still holds every detection: this is the one
        # moment the neighbours can be read off it.
        self._build_bridge_context_layer(layer, det_id)
        try:
            self._bridge_prior_subset = layer.subsetString()
        except (RuntimeError, AttributeError):
            self._bridge_prior_subset = None
        try:
            applied = bool(layer.setSubsetString(expression))
        except (RuntimeError, AttributeError, TypeError):
            applied = False
        if not applied:
            self._bridge_prior_subset = None
            self._remove_bridge_context_layer()
            return False
        self._bridge_isolated_layer = layer
        return True

    def _bridge_det_id_ceiling(self, layer) -> int:
        """The highest det_id in play when the session opens.

        Every id minted during the session lands above it, so the subset can
        keep the new shapes without naming them one by one. Reads the ids the
        bridge already treats as taken, which covers the objects the review
        filters are hiding as well as the ones on the layer.
        """
        try:
            return max(self._bridge_used_det_ids(layer), default=-1)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return -1

    # ------------------------------------------------------------------
    # Clear (runs on every exit path)
    # ------------------------------------------------------------------

    def _clear_bridge_isolation(self) -> None:
        """Put every detection back on the review layer and drop the context.

        Idempotent and never raises, because every bridge exit calls it,
        including the emergency one. It must run AFTER the fold has read the
        layer: the fold tells an edited shape from an untouched one against a
        snapshot taken under the subset, so a layer handed back whole first
        would re-base every visible object.

        The layer and the prior subset are dropped ONLY once the subset has come
        off. Kept otherwise, so the next exit path can try again on the same
        layer: nulled here, the unload net would have nothing to put back and
        the review would show one polygon for the rest of the QGIS session.
        """
        layer = getattr(self, "_bridge_isolated_layer", None)
        prior = getattr(self, "_bridge_prior_subset", None) or ""
        if layer is None or self._restore_bridge_subset(layer, prior):
            self._bridge_isolated_layer = None
            self._bridge_prior_subset = None
        else:
            self._bridge_warn_subset_stuck()
        self._remove_bridge_context_layer()

    def _restore_bridge_subset(self, layer, prior: str) -> bool:
        """Give the layer its own subset back. True once it is off.

        A layer still in edit mode refuses one, and that happens when the commit
        failed. Its buffer is dropped then: the fold has already taken the
        geometry, the reslice republishes the layer a moment later, and a subset
        left on would show the review one polygon for the rest of the session.
        """
        try:
            if layer.setSubsetString(prior):
                self._bridge_notify_subset_change(layer)
                return True
        except (RuntimeError, AttributeError, TypeError):
            return False
        try:
            if layer.isEditable():
                layer.rollBack()
                if layer.setSubsetString(prior):
                    self._bridge_notify_subset_change(layer)
                    return True
        except (RuntimeError, AttributeError, TypeError):
            pass
        return False

    def _bridge_notify_subset_change(self, layer) -> None:
        """Tell QGIS the feature set changed once the subset comes off.

        Taking a subset off leaves the snapping index and the vertex tool's
        geometry cache describing the filtered set, so the locator keeps
        answering with one polygon. Entry emits the same notification, for the
        same reason. Never raises."""
        try:
            from .shared import _notify_provider_write
        except ImportError:
            return
        _notify_provider_write(layer)

    def _bridge_warn_subset_stuck(self) -> None:
        """One log line when the filter would not come off, so the retry on the
        next exit path is readable. Never raises."""
        try:
            from qgis.core import Qgis, QgsMessageLog

            QgsMessageLog.logMessage(
                "Manual edit: the review filter did not come off; the next "
                "exit tries again",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        except Exception:  # noqa: BLE001 -- a log failure must not break teardown
            pass  # nosec B110

    # ------------------------------------------------------------------
    # The read-only context layer
    # ------------------------------------------------------------------

    def _build_bridge_context_layer(self, layer, det_id) -> None:
        """Redraw the detections this session is NOT editing, read-only.

        Geometry only: nothing reads an attribute off this layer, and the copy
        runs on a map click, so it stays as small as it can be. A feature with
        no det_id is left out, because the subset keeps it on the editable
        layer and drawing it twice would read as a duplicate.

        Best-effort at every step: no context is worse than no session.
        """
        self._remove_bridge_context_layer()
        try:
            from qgis.core import QgsFeature, QgsProject, QgsVectorLayer

            from ...core.output_store import drop_from_snapping, mark_temp_layer
            from .shared import _add_features_fast, _apply_fast_render
        except ImportError:
            return
        try:
            context = QgsVectorLayer(
                f"MultiPolygon?crs={layer.crs().authid()}",
                tr("Other detections"), "memory")
            if not context.isValid():
                return
            target = int(det_id)
            others = []
            for feature in layer.getFeatures():
                try:
                    value = feature["det_id"]
                except (KeyError, TypeError):
                    continue
                if not isinstance(value, int) or int(value) == target:
                    continue
                geom = feature.geometry()
                if geom is None or geom.isEmpty():
                    continue
                copy = QgsFeature()
                copy.setGeometry(geom)
                others.append(copy)
            if others:
                _add_features_fast(context.dataProvider(), others)
            self._apply_bridge_context_style(context)
            _apply_fast_render(context)
            # Read-only on top of "never the active layer": two independent
            # reasons no digitizing tool can write to it.
            context.setReadOnly(True)
            mark_temp_layer(context)
            QgsProject.instance().addMapLayer(context, False)
            # Held from here on, so a failure below still leaves something for
            # the clear to remove.
            self._bridge_context_layer = context
            # A scratch layer left in the snapping config dangles once it is
            # gone and crashes the next project save. Snapping runs on every
            # visible layer during a session, so this costs nothing.
            drop_from_snapping(context)
            self._insert_bridge_context_node(context, layer)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass

    def _apply_bridge_context_style(self, context) -> None:
        """The review's dim grey for a polygon no session is editing, so one
        polygon at a time looks the same whichever method drew it."""
        try:
            from qgis.core import QgsFillSymbol, QgsSingleSymbolRenderer
            symbol = QgsFillSymbol.createSimple({
                "color": SESSION_DIM_FILL_STR,
                "outline_color": SESSION_DIM_STROKE_STR,
                # The dim fill alone cannot hold a small polygon on dark
                # imagery, so the outline carries it.
                "outline_width": "0.4",
            })
            context.setRenderer(QgsSingleSymbolRenderer(symbol))
        except (RuntimeError, AttributeError, TypeError, ImportError):
            pass

    def _insert_bridge_context_node(self, context, layer) -> None:
        """Put the context UNDER the polygon being edited, never over it."""
        from qgis.core import QgsProject
        root = QgsProject.instance().layerTreeRoot()
        parent, index = root, 0
        try:
            node = root.findLayer(layer.id())
            if node is not None and node.parent() is not None:
                parent = node.parent()
                index = list(parent.children()).index(node) + 1
        except (RuntimeError, AttributeError, TypeError, ValueError):
            parent, index = root, 0
        parent.insertLayer(index, context)

    def _remove_bridge_context_layer(self) -> None:
        """Drop the context layer from the project. Idempotent."""
        context = getattr(self, "_bridge_context_layer", None)
        self._bridge_context_layer = None
        if context is None:
            return
        try:
            from qgis.core import QgsProject
            QgsProject.instance().removeMapLayer(context.id())
        except (RuntimeError, AttributeError, ImportError):
            pass
