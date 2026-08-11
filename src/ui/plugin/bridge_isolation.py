































from __future__ import annotations

from ...core.i18n import tr
from ..canvas_palette import SESSION_DIM_FILL_STR, SESSION_DIM_STROKE_STR


def bridge_isolation_expression(det_id: int, id_ceiling: int) -> str:







    return (
        f'"det_id" = {int(det_id):d}'
        f' OR "det_id" > {int(id_ceiling):d}'
        ' OR "det_id" IS NULL'
    )


class BridgeIsolationMixin:







    def _init_bridge_isolation_state(self) -> None:



        self._bridge_isolated_layer = None


        self._bridge_prior_subset = None
        self._bridge_context_layer = None





    def _isolate_bridge_target(self, layer, det_id) -> bool:







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







        try:
            return max(self._bridge_used_det_ids(layer), default=-1)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return -1





    def _clear_bridge_isolation(self) -> None:













        layer = getattr(self, "_bridge_isolated_layer", None)
        prior = getattr(self, "_bridge_prior_subset", None) or ""
        if layer is None or self._restore_bridge_subset(layer, prior):
            self._bridge_isolated_layer = None
            self._bridge_prior_subset = None
        else:
            self._bridge_warn_subset_stuck()
        self._remove_bridge_context_layer()

    def _restore_bridge_subset(self, layer, prior: str) -> bool:







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






        try:
            from .shared import _notify_provider_write
        except ImportError:
            return
        _notify_provider_write(layer)

    def _bridge_warn_subset_stuck(self) -> None:


        try:
            from qgis.core import Qgis, QgsMessageLog

            QgsMessageLog.logMessage(
                "Manual edit: the review filter did not come off; the next "
                "exit tries again",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        except Exception:  # noqa: BLE001
            pass  # nosec B110





    def _build_bridge_context_layer(self, layer, det_id) -> None:









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


            context.setReadOnly(True)
            mark_temp_layer(context)
            QgsProject.instance().addMapLayer(context, False)


            self._bridge_context_layer = context



            drop_from_snapping(context)
            self._insert_bridge_context_node(context, layer)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass

    def _apply_bridge_context_style(self, context) -> None:


        try:
            from qgis.core import QgsFillSymbol, QgsSingleSymbolRenderer
            symbol = QgsFillSymbol.createSimple({
                "color": SESSION_DIM_FILL_STR,
                "outline_color": SESSION_DIM_STROKE_STR,


                "outline_width": "0.4",
            })
            context.setRenderer(QgsSingleSymbolRenderer(symbol))
        except (RuntimeError, AttributeError, TypeError, ImportError):
            pass

    def _insert_bridge_context_node(self, context, layer) -> None:

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

        context = getattr(self, "_bridge_context_layer", None)
        self._bridge_context_layer = None
        if context is None:
            return
        try:
            from qgis.core import QgsProject
            QgsProject.instance().removeMapLayer(context.id())
        except (RuntimeError, AttributeError, ImportError):
            pass
