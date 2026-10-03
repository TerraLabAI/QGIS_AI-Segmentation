












from __future__ import annotations

import math

from qgis.core import Qgis, QgsMessageLog

from ...core.i18n import tr


class AutoMyClassesMixin:


    def _my_classes_wanted(self) -> bool:
        dock = self.dock_widget
        try:
            return (dock is not None and dock.auto_target_mode() == "land_cover"
                    and dock.my_classes_active())
        except (RuntimeError, AttributeError):
            return False

    def _my_classes_error(self, message: str) -> None:
        self._headless_error = message
        try:
            self.dock_widget.set_auto_status("error", message)
        except (RuntimeError, AttributeError):  # nosec B110
            pass

    def _start_my_classes_run(self) -> None:

        if getattr(self, "_mc_task", None) is not None:
            return
        from ...core import my_classes as mc
        from ..dock.auto_my_classes import everything_else_name

        dock = self.dock_widget
        rows = dock.my_classes_rows()
        problem = mc.validation_error(rows, everything_else_name())
        if problem:
            dock.set_my_classes_note(problem)
            self._my_classes_error(problem)
            return
        if not self._start_step_worker_not_busy():
            return
        if self._start_step_release_ui_tools() is False:
            return
        layer = self._start_step_pick_layer()
        if layer is None:
            return
        auth = self._start_step_sign_in()
        if not auth:
            return
        grid = self._start_step_pixel_grid(layer)
        if grid is None:
            return
        zone_km2 = float(self._auto_zone_area_km2() or 0.0)
        zone_wkt = self._auto_zone_wkt_wgs84()
        if zone_km2 <= 0 or not zone_wkt:
            self._my_classes_error(tr("Draw a zone first."))
            return
        cap = mc.max_zone_km2()
        if zone_km2 > cap:
            self._my_classes_error(
                tr("This zone is {zone} km2. My classes maps up to {cap} km2 per run: "
                   "draw a smaller zone.").format(zone=f"{zone_km2:.2f}", cap=f"{cap:g}"))
            return

        from qgis.core import QgsCoordinateReferenceSystem, QgsRectangle

        crs_authid = grid.get("crs") or layer.crs().authid()
        bbox = grid["bbox"]
        extent = QgsRectangle(bbox[0], bbox[1], bbox[2], bbox[3])
        width, height = mc.image_size_for(extent.width(), extent.height())
        try:
            m_per_px = float(self._mupp_to_meters(layer, extent, extent.width() / width))
        except Exception:  # noqa: BLE001
            m_per_px = 0.0
        if math.isfinite(m_per_px) and m_per_px > mc.warn_m_per_px():
            from ..dialogs.confirm_dialog import PRIMARY, SECONDARY, ChoiceButton, ask_choice
            clicked = ask_choice(
                self.iface.mainWindow(), tr("Large zone"),
                tr("Large zone: about {m} m per pixel. Smaller zones give finer classes.").format(
                    m=f"{m_per_px:.1f}"),
                [ChoiceButton("cancel", tr("Cancel"), SECONDARY),
                 ChoiceButton("go", tr("Map anyway"), PRIMARY)],
                default="go", escape="cancel")
            if clicked != "go":
                return

        from ...core.tile_images import render_zone_to_image
        try:
            dock.set_auto_status("info", tr("Preparing your zone..."))
        except (RuntimeError, AttributeError):  # nosec B110
            pass
        image, actual = render_zone_to_image(
            layer, extent, width, height, resample_local=True,
            render_crs=QgsCoordinateReferenceSystem(crs_authid))
        if image is None or actual is None:
            self._my_classes_error(tr("The zone could not be drawn. Check the layer shows, then try again."))
            return
        png_b64 = _png_base64(image)
        if not png_b64:
            self._my_classes_error(tr("The zone could not be drawn. Check the layer shows, then try again."))
            return

        import uuid
        run_id = str(uuid.uuid4())
        payload = {
            "run_id": run_id,
            "image": png_b64,
            "width": width,
            "height": height,
            "zone_wkt": zone_wkt,
            "zone_km2": round(zone_km2, 6),
            "classes": [{"name": mc.clean_name(n), "color": c} for n, c in rows],
        }
        ctx = {
            "run_id": run_id, "layer": layer, "crs": crs_authid,
            "bbox": (actual.xMinimum(), actual.yMinimum(), actual.xMaximum(), actual.yMaximum()),
            "width": width, "height": height, "zone_km2": zone_km2,
        }
        from qgis.core import QgsApplication

        from ...api.terralab_client import TerraLabClient
        from ...workers.generic_request_task import GenericRequestTask

        client = TerraLabClient()
        task = GenericRequestTask(
            tr("Mapping your classes"), lambda: client.post_my_classes(payload, auth=auth))
        task.succeeded.connect(lambda answer, c=ctx: self._on_my_classes_answer(answer, c))
        task.failed.connect(lambda code, msg, c=ctx: self._on_my_classes_failed(code, msg, c))
        self._mc_task = task
        ctx["class_count"] = len(rows)
        ctx["billed_km2"] = mc.billed_km2(zone_km2)
        self._my_classes_track("started", ctx)
        try:
            dock.set_auto_status(
                "info", tr("Mapping your classes (billed {km2} km2)...").format(
                    km2=f"{mc.billed_km2(zone_km2):.2f}"))
        except (RuntimeError, AttributeError):  # nosec B110
            pass
        QgsMessageLog.logMessage(
            f"My classes: run sent, {len(rows)} class(es), {width}x{height}px",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        QgsApplication.taskManager().addTask(task)

    def _my_classes_track(self, outcome: str, ctx: dict, **extra) -> None:
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_my_classes(
                outcome, ctx.get("run_id", ""), ctx.get("class_count", 0),
                ctx.get("billed_km2", 0.0), **extra)
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _on_my_classes_failed(self, code: str, message: str, ctx: dict | None = None) -> None:
        self._mc_task = None
        from ...core.error_policy import LINK_OR_TIMEOUT_CODES
        code = str(code or "UNKNOWN")


        self._my_classes_track("failed", ctx or {}, stage="request", error_code=code,
                               refunded=code not in LINK_OR_TIMEOUT_CODES)
        QgsMessageLog.logMessage(
            f"My classes: run failed ({code})", "AI Segmentation", level=Qgis.MessageLevel.Warning)
        self._my_classes_error(str(message or tr("The class map could not be made. Try again.")))

    def _on_my_classes_answer(self, answer: dict, ctx: dict) -> None:

        self._mc_task = None
        from ...core.land_cover import LandCoverMosaic, normalize_legend

        try:
            from ...core.my_classes import decode_class_raster
            grid = decode_class_raster(answer.get("class_raster_png") or "", ctx["width"], ctx["height"])
        except Exception as exc:  # noqa: BLE001
            QgsMessageLog.logMessage(
                f"My classes: unreadable answer ({type(exc).__name__})",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            self._my_classes_track("failed", ctx, stage="answer", error_code="UNREADABLE_RASTER")
            self._my_classes_error(tr("The class map came back unreadable. Try again."))
            return
        legend = normalize_legend(answer.get("class_legend"))
        if not legend:
            self._my_classes_track("failed", ctx, stage="answer", error_code="NO_LEGEND")
            self._my_classes_error(tr("The class map came back unreadable. Try again."))
            return
        if answer.get("billed_km2") is not None:
            ctx["billed_km2"] = answer.get("billed_km2")
        self._my_classes_track("succeeded", ctx, replayed=bool(answer.get("replayed")))
        if self._land_cover_autosave_pending("new_run") is False:
            return
        self._auto_run_id = ctx["run_id"]
        self._auto_crs_authid = ctx["crs"]
        try:
            self._auto_clip_polygon = self._polygon_in_run_crs(ctx["layer"])
        except Exception:  # noqa: BLE001
            self._auto_clip_polygon = None
        h, w = int(grid.shape[0]), int(grid.shape[1])
        store = LandCoverMosaic([(0, 0, w, h)], h, w)
        store.add_tile((0, 0, w, h), grid)
        self._auto_land_cover = {"min_patch_m2": 0.0, "class_legend": legend}
        self._lc_store = store
        self._lc_mosaic = None
        self._lc_geo = {"bbox": tuple(ctx["bbox"]), "img_shape": (h, w)}
        self._lc_served_legend = list(legend)
        self._lc_answer_legend = None
        self._lc_tiles_succeeded = 1
        self._land_cover_capture_zone()

        try:
            from ...core.land_cover import parse_land_cover_plan
            plan = parse_land_cover_plan((getattr(self, "_auto_run_plan", None) or {}).get("plan"))
            floor = float((plan or {}).get("min_patch_m2") or 0.0)
        except Exception:  # noqa: BLE001
            floor = 0.0
        self._auto_land_cover["min_patch_m2"] = floor
        self._lc_min_patch = floor
        self._lc_default_patch = floor
        try:
            self.dock_widget.set_auto_status("info", "")
        except (RuntimeError, AttributeError):  # nosec B110
            pass
        QgsMessageLog.logMessage(
            f"My classes: raster {w}x{h}, billed {answer.get('billed_km2')} km2",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        self._land_cover_start_partition(first=True)


def _png_base64(image) -> str:
    import base64

    from qgis.PyQt.QtCore import QBuffer, QByteArray, QIODevice

    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)

    ok = image.save(buf, "JPG", 90)
    buf.close()
    if not ok:
        return ""
    return base64.b64encode(bytes(data)).decode("ascii")
