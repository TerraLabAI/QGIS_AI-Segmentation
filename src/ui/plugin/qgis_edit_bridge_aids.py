








from __future__ import annotations

from ...core.i18n import tr
from .bridge_capture_state import report_editing_refused



_SNAP_TOLERANCE_PX = 12


def _snap_mode_all_layers():





    from qgis.core import QgsSnappingConfig
    scope = getattr(QgsSnappingConfig, "SnappingMode", None)
    if scope is not None:
        val = getattr(scope, "AllLayers", None)
        if val is not None:
            return val
    return getattr(QgsSnappingConfig, "AllLayers", None)


def _snap_type_flags():






    from qgis.core import QgsSnappingConfig
    scope = getattr(QgsSnappingConfig, "SnappingTypes", None)
    if scope is not None:
        vtx = getattr(scope, "VertexFlag", None)
        seg = getattr(scope, "SegmentFlag", None)
        if vtx is not None and seg is not None:
            return vtx | seg
    vtx = getattr(QgsSnappingConfig, "Vertex", None)
    seg = getattr(QgsSnappingConfig, "Segment", None)
    if vtx is not None and seg is not None:
        return vtx | seg
    return None


def _tolerance_pixels_unit():




    from qgis.core import QgsTolerance
    scope = getattr(QgsTolerance, "UnitType", None)
    if scope is not None:
        val = getattr(scope, "Pixels", None)
        if val is not None:
            return val
    return getattr(QgsTolerance, "Pixels", None)


def _avoid_mode(name: str):




    from qgis.core import Qgis
    scope = getattr(Qgis, "AvoidIntersectionsMode", None)
    if scope is None:
        return None
    return getattr(scope, name, None)


class EditBridgeAidsMixin:





    def _report_bridge_editing_refused(self, reason: str) -> None:


        report_editing_refused(self, tr, reason)

    def _show_bridge_unavailable(self) -> None:

        message = tr(
            "QGIS could not activate the temporary review layer. Close any "
            "other editing session, then try again.")
        try:
            from qgis.core import Qgis

            from ...core.server_dials import dial_in_range
            self.iface.messageBar().pushMessage(
                "AI Segmentation", message,
                level=Qgis.MessageLevel.Warning,
                duration=dial_in_range("tuning.agent.bridge_unavailable_notice_s", 7, 4, 10))
        except (RuntimeError, AttributeError):
            pass

    def _show_bridge_isolation_failed(self) -> None:






        message = tr(
            "Editing by hand could not open on this polygon on its own. Try "
            "again, or fix it with the AI.")
        try:
            from qgis.core import Qgis, QgsMessageLog

            QgsMessageLog.logMessage(
                "Manual edit refused: the session could not be held to one "
                "polygon", "AI Segmentation", level=Qgis.MessageLevel.Warning)
            from ...core.server_dials import dial_in_range
            self.iface.messageBar().pushMessage(
                "AI Segmentation", message,
                level=Qgis.MessageLevel.Warning,
                duration=dial_in_range("tuning.agent.bridge_isolation_notice_s", 7, 4, 10))
        except (RuntimeError, AttributeError):
            pass





    def _save_bridge_editing_aids(self) -> None:












        from qgis.core import QgsProject
        proj = QgsProject.instance()
        aids: dict = {}
        try:
            cfg = proj.snappingConfig()
            getter = (getattr(cfg, "typeFlag", None) or getattr(cfg, "type", None))
            aids["snap"] = {
                "enabled": bool(cfg.enabled()),
                "mode": cfg.mode(),
                "type": getter() if getter is not None else None,
                "tolerance": cfg.tolerance(),
                "units": cfg.units(),
            }
        except (RuntimeError, AttributeError, TypeError):
            aids["snap"] = None
        try:
            aids["topo"] = bool(proj.topologicalEditing())
        except (RuntimeError, AttributeError):
            aids["topo"] = False
        try:
            aids["avoid_mode"] = proj.avoidIntersectionsMode()
        except (RuntimeError, AttributeError):
            aids["avoid_mode"] = None
        try:
            aids["avoid_layers"] = list(proj.avoidIntersectionsLayers() or [])
        except (RuntimeError, AttributeError):
            aids["avoid_layers"] = []
        self._qgis_bridge_saved_aids = aids

    def _apply_bridge_editing_config(self, layer) -> bool:




        from qgis.core import QgsProject
        proj = QgsProject.instance()










        try:
            layer.dataChanged.emit()
        except (RuntimeError, AttributeError):
            pass
        try:
            if not layer.isEditable() and not layer.startEditing():
                self._report_bridge_editing_refused("refused")
                return False
        except (RuntimeError, AttributeError) as err:
            self._report_bridge_editing_refused(type(err).__name__)
            return False
        self._suppress_bridge_attribute_form(layer)
        try:
            cfg = proj.snappingConfig()
            cfg.setEnabled(True)
            mode = _snap_mode_all_layers()
            if mode is not None:
                cfg.setMode(mode)
            flags = _snap_type_flags()
            if flags is not None:

                setter = (getattr(cfg, "setTypeFlag", None) or getattr(cfg, "setType", None))
                if setter is not None:
                    setter(flags)
            from ...core.server_dials import dial_in_range
            cfg.setTolerance(
                dial_in_range("tuning.agent.snap_tolerance_px", _SNAP_TOLERANCE_PX, 4, 40))
            unit = _tolerance_pixels_unit()
            if unit is not None:
                cfg.setUnits(unit)
            proj.setSnappingConfig(cfg)
        except (RuntimeError, AttributeError, TypeError):
            pass






        try:
            proj.setTopologicalEditing(False)
        except (RuntimeError, AttributeError):
            pass






        try:
            off_mode = _avoid_mode("AllowIntersections")
            if off_mode is not None and hasattr(proj, "setAvoidIntersectionsMode"):
                proj.setAvoidIntersectionsMode(off_mode)
        except (RuntimeError, AttributeError, TypeError):
            pass
        self._apply_bridge_vertex_search_radius()
        self._apply_bridge_selection_colour()
        return True

    def _suppress_bridge_attribute_form(self, layer) -> None:





        try:
            cfg = layer.editFormConfig()
            self._qgis_bridge_saved_form_suppress = cfg.suppress()
            suppress_on = None



            enum = getattr(
                type(cfg), "FeatureFormSuppress", None)
            if enum is not None:
                suppress_on = getattr(enum, "SuppressOn", None)
            if suppress_on is None:
                suppress_on = getattr(type(cfg), "SuppressOn", None)
            if suppress_on is not None:
                cfg.setSuppress(suppress_on)
                layer.setEditFormConfig(cfg)
        except (RuntimeError, AttributeError, TypeError):
            self._qgis_bridge_saved_form_suppress = None

    def _restore_bridge_attribute_form(self, layer) -> None:

        saved = getattr(self, "_qgis_bridge_saved_form_suppress", None)
        self._qgis_bridge_saved_form_suppress = None
        if saved is None or layer is None:
            return
        try:
            cfg = layer.editFormConfig()
            cfg.setSuppress(saved)
            layer.setEditFormConfig(cfg)
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _apply_bridge_selection_colour(self) -> None:










        canvas = self.iface.mapCanvas()
        getter = getattr(canvas, "selectionColor", None)
        setter = getattr(canvas, "setSelectionColor", None)


        if setter is None or getter is None:
            return
        try:
            from qgis.PyQt.QtGui import QColor
            saved = QColor(getter())
        except (RuntimeError, AttributeError, TypeError, ImportError):
            self._qgis_bridge_saved_selection_colour = None
            return


        self._qgis_bridge_saved_selection_colour = saved
        try:
            setter(QColor(255, 255, 0, 60))
            canvas.refresh()
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _restore_bridge_selection_colour(self) -> None:

        saved = getattr(self, "_qgis_bridge_saved_selection_colour", None)
        if saved is None:
            return
        self._qgis_bridge_saved_selection_colour = None
        try:
            canvas = self.iface.mapCanvas()
            setter = getattr(canvas, "setSelectionColor", None)
            if setter is not None:
                setter(saved)
                canvas.refresh()
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _apply_bridge_vertex_search_radius(self) -> None:




        try:
            from qgis.core import QgsSettings
            settings = QgsSettings()
            key = "qgis/digitizing/search_radius_vertex_edit"
            unit_key = "qgis/digitizing/search_radius_vertex_edit_unit"
            prior = settings.value(key, None)
            prior_unit = settings.value(unit_key, None)
            self._qgis_bridge_saved_search_radius = (prior, prior_unit)
            try:
                current = float(prior) if prior is not None else 0.0
            except (TypeError, ValueError):
                current = 0.0
            if current <= 0.0:

                from ...core.server_dials import dial_in_range
                settings.setValue(
                    key, dial_in_range("tuning.agent.vertex_search_radius_px", 10, 4, 40))


                settings.setValue(unit_key, "Pixels")
        except (RuntimeError, AttributeError, TypeError, ValueError, ImportError):
            self._qgis_bridge_saved_search_radius = None

    def _restore_bridge_vertex_search_radius(self) -> None:

        saved = getattr(self, "_qgis_bridge_saved_search_radius", None)
        if saved is None:
            return
        prior, prior_unit = saved
        self._qgis_bridge_saved_search_radius = None
        key = "qgis/digitizing/search_radius_vertex_edit"
        unit_key = "qgis/digitizing/search_radius_vertex_edit_unit"
        try:
            from qgis.core import QgsSettings
            settings = QgsSettings()
            if prior is None:
                settings.remove(key)
            else:
                settings.setValue(key, prior)
            if prior_unit is None:
                settings.remove(unit_key)
            else:
                settings.setValue(unit_key, prior_unit)
        except (RuntimeError, AttributeError, TypeError, ImportError):

            pass

    def _restore_bridge_setting(self, restore, *args) -> None:





        try:
            restore(*args)
        except Exception as exc:  # noqa: BLE001
            self._log_bridge_failure("qgis_bridge_restore", exc)

    def _restore_bridge_editing_aids(self) -> None:



        saved = getattr(self, "_qgis_bridge_saved_aids", None)
        if not saved:
            return
        from qgis.core import QgsProject
        proj = QgsProject.instance()
        try:
            proj.setTopologicalEditing(bool(saved.get("topo")))
        except (RuntimeError, AttributeError):
            pass
        try:
            avoid_mode = saved.get("avoid_mode")
            if avoid_mode is not None and hasattr(proj, "setAvoidIntersectionsMode"):
                proj.setAvoidIntersectionsMode(avoid_mode)
        except (RuntimeError, AttributeError, TypeError):
            pass
        try:
            layers = [ly for ly in (saved.get("avoid_layers") or [])
                      if self._is_layer_valid(ly)]
            if hasattr(proj, "setAvoidIntersectionsLayers"):
                proj.setAvoidIntersectionsLayers(layers)
        except (RuntimeError, AttributeError, TypeError):
            pass
        try:
            snap = saved.get("snap")
            if snap:



                cfg = proj.snappingConfig()
                cfg.setEnabled(snap["enabled"])
                cfg.setMode(snap["mode"])
                if snap["type"] is not None:
                    setter = (getattr(cfg, "setTypeFlag", None)
                              or getattr(cfg, "setType", None))
                    if setter is not None:
                        setter(snap["type"])
                cfg.setTolerance(snap["tolerance"])
                cfg.setUnits(snap["units"])
                proj.setSnappingConfig(cfg)
        except (RuntimeError, AttributeError, TypeError, KeyError):
            pass
