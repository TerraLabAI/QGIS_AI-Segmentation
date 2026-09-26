








from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsMessageLog,
    QgsProject,
    QgsVectorLayer,
)

from ...core.i18n import tr
from ...core.land_cover import polygon_parts


_LAND_COVER_WORDS = ("land cover", "landcover")


class AutoLandCoverMixin:




    def _land_cover_plan_for(self, prompt: str) -> dict | None:

        from ...core.land_cover import parse_land_cover_plan

        return parse_land_cover_plan(self._active_run_plan(prompt))

    def _start_step_land_cover_plan(self) -> bool:






        try:
            prompt = self.dock_widget.auto_prompt_input.text().strip()
        except (RuntimeError, AttributeError):
            return True
        if prompt.lower() not in _LAND_COVER_WORDS:
            return True
        if self._land_cover_plan_for(prompt) is not None:
            return True
        plan = None
        try:
            from ...api.terralab_client import TerraLabClient
            from ...core.activation_manager import get_auth_header

            auth = get_auth_header()
            if auth:
                zone_area_m2, native_mupp = self._auto_run_plan_inputs()
                plan = TerraLabClient().get_seg_run_plan(
                    prompt, zone_area_m2, native_mupp, auth=auth)
        except Exception:  # noqa: BLE001
            plan = None
        from ...core.land_cover import parse_land_cover_plan

        if isinstance(plan, dict) and not plan.get("error") and parse_land_cover_plan(plan):
            from ...core.run_decisions import neutral_run_decisions, parse_run_decisions, remember_plan

            remember_plan(prompt, plan)
            self._auto_run_plan = {"prompt": prompt, "plan": plan, "exemplar_size_m": None}

            self._late_plan_clear()
            self._auto_run_decisions = parse_run_decisions(plan) or neutral_run_decisions()
            try:
                from ...core.detection_policy_core import capture_run_policy
                capture_run_policy(plan)
            except Exception:  # noqa: BLE001  # nosec B110
                pass
            self._sync_land_cover_ready()
            return True
        message = tr("Couldn't load land cover settings. Check your connection and try again.")
        self._tel_detect_blocked("settings_not_loaded")
        self._headless_error = message
        self._headless_error_code = "land_cover_settings_missing"
        try:
            from ...core.detection_policy_core import release_run_policy
            release_run_policy()
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        if not getattr(self, "_auto_headless_run", False):
            try:
                self.iface.messageBar().pushWarning("AI Segmentation", message)
            except (RuntimeError, AttributeError):  # nosec B110
                pass
        QgsMessageLog.logMessage(
            "Auto detection: land cover plan missing at start; nothing sent",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return False

    def _land_cover_prefetch_legend(self, prompt: str = "") -> None:



        if (prompt or "").strip().lower() not in _LAND_COVER_WORDS:
            return
        if getattr(self, "_lc_legend_task", None) is not None:
            return
        try:
            from ...core.land_cover import parse_land_cover_plan
            if (self.dock_widget is None or (parse_land_cover_plan(
                    (getattr(self, "_auto_run_plan", None) or {}).get("plan")) or {}).get("class_legend")):
                return
            from qgis.core import QgsApplication

            from ...api.terralab_client import TerraLabClient
            from ...core.activation_manager import get_auth_header, is_plugin_activated
            from ...workers.generic_request_task import GenericRequestTask
            if not is_plugin_activated():
                return
            auth = get_auth_header()
            if not auth:
                return
            client = TerraLabClient()
            task = GenericRequestTask(
                tr("Loading land cover classes"),
                lambda: client.get_seg_run_plan(prompt.strip(), None, None, auth=auth),
                hidden=True)

            def _done(plan):
                self._lc_legend_task = None
                block = parse_land_cover_plan(plan)
                legend = (block or {}).get("class_legend") or []
                if legend and self.dock_widget is not None:
                    try:
                        self.dock_widget.set_auto_land_cover_legend(legend)
                    except (RuntimeError, AttributeError):  # nosec B110
                        pass

            task.succeeded.connect(_done)
            task.failed.connect(lambda *_a: setattr(self, "_lc_legend_task", None))
            self._lc_legend_task = task
            QgsApplication.taskManager().addTask(task)
        except Exception:  # noqa: BLE001  # nosec B110
            self._lc_legend_task = None

    def _sync_land_cover_ready(self) -> None:

        dock = self.dock_widget
        if dock is None:
            return
        rp = getattr(self, "_auto_run_plan", None)
        plan = rp.get("plan") if isinstance(rp, dict) else None
        from ...core.land_cover import parse_land_cover_plan
        try:
            block = parse_land_cover_plan(plan)
            dock.set_auto_land_cover_ready(
                block is not None, (block or {}).get("class_legend") or [])
        except (RuntimeError, AttributeError):
            pass



    def _land_cover_begin_run(self, plan: dict | None, tiles: list,
                              geo_transform: dict) -> bool:


        if self._land_cover_autosave_pending("new_run") is False:
            return False
        self._auto_land_cover = plan
        try:
            self.dock_widget._auto_land_cover_run = False
        except (RuntimeError, AttributeError):  # nosec B110
            pass
        if plan is None:
            return True
        from ...core.land_cover import LandCoverMosaic

        img_h, img_w = geo_transform.get("img_shape", (1, 1))
        self._lc_store = LandCoverMosaic(
            tiles, int(img_h), int(img_w), path=_land_cover_grid_path())
        self._lc_mosaic = self._lc_store
        self._lc_geo = {"bbox": tuple(geo_transform["bbox"]),
                        "img_shape": (int(img_h), int(img_w))}

        self._land_cover_capture_zone()
        self._lc_served_legend = list(plan.get("class_legend") or [])
        try:
            self.dock_widget._auto_land_cover_run = True
        except (RuntimeError, AttributeError):  # nosec B110
            pass

        return True

    def _land_cover_take_tile(self, detections: list) -> bool:


        mosaic = getattr(self, "_lc_mosaic", None)
        if mosaic is None:
            return False
        for item in detections or ():
            if not isinstance(item, dict) or "land_cover_rect" not in item:
                continue
            try:
                mosaic.add_tile(item["land_cover_rect"], item["labels"])
            except (ValueError, IndexError, TypeError) as exc:
                QgsMessageLog.logMessage(
                    f"Land cover: a tile could not be placed ({type(exc).__name__})",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
            legend = item.get("class_legend")
            if legend and not getattr(self, "_lc_answer_legend", None):
                self._lc_answer_legend = list(legend)
            self._land_cover_draw_preview(
                item.get("preview") or (), mosaic.core_map_rect(
                    item["land_cover_rect"], self._lc_geo))
        return True

    def _land_cover_draw_preview(self, polygons, core=None) -> None:


        if not polygons:
            return
        try:
            layer = getattr(self, "_lc_preview_layer", None)
            if layer is None or QgsProject.instance().mapLayer(layer.id()) is None:
                from ...core.land_cover import legend_rows
                from ...core.layer_conventions import make_legend_categorized_renderer
                from ...core.output_store import mark_temp_layer
                crs = str(getattr(self, "_auto_crs_authid", "") or "EPSG:4326")
                layer = QgsVectorLayer(f"Polygon?crs={crs}&field=cid:integer",
                                       tr("Land cover (in progress)"), "memory")
                legend = (getattr(self, "_lc_served_legend", None)
                          or getattr(self, "_lc_answer_legend", None) or [])
                rows = legend_rows([], legend, tr("Class {n}"))
                renderer = make_legend_categorized_renderer(
                    [(str(r["id"]), r["color"]) for r in rows], field="cid")
                if renderer is not None:

                    from qgis.PyQt.QtCore import Qt as _Qt
                    for i, cat in enumerate(renderer.categories()):
                        symbol = cat.symbol().clone()
                        symbol.symbolLayer(0).setStrokeStyle(_Qt.PenStyle.NoPen)
                        renderer.updateCategorySymbol(i, symbol)
                    layer.setRenderer(renderer)
                try:
                    mark_temp_layer(layer)
                except Exception:  # noqa: BLE001  # nosec B110
                    pass
                QgsProject.instance().addMapLayer(layer)
                self._lc_preview_layer = layer
            feats = []
            box = QgsGeometry.fromRect(core) if core is not None else None
            zone = getattr(self, "_auto_clip_polygon", None)
            if zone is not None and not zone.isEmpty():

                box = zone if box is None else box.intersection(zone)
            for cid, wkb in polygons:
                geom = QgsGeometry()
                geom.fromWkb(wkb)
                if box is not None:
                    geom = geom.intersection(box)
                    if geom is None or geom.isEmpty():
                        continue


                for part in polygon_parts(geom):
                    feat = QgsFeature(layer.fields())
                    feat.setGeometry(part)
                    feat.setAttributes([int(cid)])
                    feats.append(feat)
            if feats:
                layer.dataProvider().addFeatures(feats)
            layer.triggerRepaint()
        except (RuntimeError, AttributeError):  # nosec B110
            pass

    def _land_cover_drop_preview(self) -> None:
        layer = getattr(self, "_lc_preview_layer", None)
        self._lc_preview_layer = None
        if layer is None:
            return
        try:
            if QgsProject.instance().mapLayer(layer.id()) is not None:
                QgsProject.instance().removeMapLayer(layer.id())
        except RuntimeError:  # nosec B110
            pass

    def _land_cover_area_scale(self, zone: QgsGeometry | None) -> tuple[float, float]:

        from ...core.layer_conventions import make_area_measurer

        crs = QgsCoordinateReferenceSystem(str(getattr(self, "_auto_crs_authid", "") or ""))
        if zone is None or zone.isEmpty():
            minx, miny, maxx, maxy = self._lc_geo["bbox"]
            from qgis.core import QgsRectangle
            zone = QgsGeometry.fromRect(QgsRectangle(minx, miny, maxx, maxy))
        try:
            measured = float(make_area_measurer(crs).measureArea(zone))
        except Exception:  # noqa: BLE001
            measured = 0.0
        planar = float(zone.area() or 0.0)
        if measured > 0 and planar > 0:
            return measured / planar, measured
        return 1.0, planar

    def _land_cover_capture_zone(self) -> None:

        zone = getattr(self, "_auto_clip_polygon", None)
        zone_wkb = None
        if zone is not None:
            try:
                zone_wkb = bytes(zone.asWkb())
            except (RuntimeError, AttributeError):
                zone_wkb = None
        scale, zone_m2 = self._land_cover_area_scale(zone)
        self._lc_zone_wkb = zone_wkb
        self._lc_area_scale = scale
        self._lc_zone_m2 = zone_m2

    def _land_cover_finalize(self, tiles_succeeded: int, headless: bool = False) -> None:


        try:
            self._drain_auto_tiles_now()
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        self._reset_auto_live_pipeline()
        self._auto_merger = None
        self._remove_auto_selection_layer()
        try:
            self.dock_widget._auto_land_cover_run = False
        except (RuntimeError, AttributeError):  # nosec B110
            pass
        self._land_cover_capture_zone()
        self._lc_tiles_succeeded = tiles_succeeded
        mosaic = self._lc_mosaic
        if mosaic is None or mosaic.tiles_added == 0:
            self._land_cover_teardown()
            self._record_auto_zero_result(tiles_succeeded)
            return
        if headless:
            self._land_cover_finish_headless(tiles_succeeded)
            return


        self._lc_mosaic = None
        plan = self._auto_land_cover or {}
        self._lc_min_patch = float(plan.get("min_patch_m2") or 0.0)
        self._lc_default_patch = self._lc_min_patch
        self._land_cover_start_partition(first=True)

    def _land_cover_run_task(self, fn, on_done) -> None:


        from qgis.core import QgsApplication

        from ...workers.generic_request_task import GenericRequestTask

        self._lc_gen = getattr(self, "_lc_gen", 0) + 1
        gen = self._lc_gen
        task = GenericRequestTask(tr("Building the land cover map"), fn, hidden=True)
        task.succeeded.connect(
            lambda result, g=gen: on_done(result) if g == self._lc_gen else None)
        task.failed.connect(
            lambda _code, msg, g=gen: self._on_land_cover_failed(msg) if g == self._lc_gen else None)
        self._lc_task = task
        QgsApplication.taskManager().addTask(task)

    def _on_land_cover_failed(self, message: str) -> None:
        QgsMessageLog.logMessage(
            f"Land cover: building the map failed ({str(message)[:120]})",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        if getattr(self, "_lc_result", None) is not None:
            self._land_cover_push_card(busy=False)
            return
        self._land_cover_show_error()

    def _land_cover_show_error(self) -> None:


        dock = self.dock_widget
        if dock is None:
            return
        dock._lc_actions = dict(getattr(dock, "_lc_actions", None) or {},
                                retry=self._on_land_cover_retry,
                                exit=self._on_land_cover_exit)
        try:
            dock.set_land_cover_error(True)
        except Exception as exc:  # noqa: BLE001
            QgsMessageLog.logMessage(
                f"Land cover: error card failed ({type(exc).__name__})",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            if self._land_cover_autosave_pending("error") is False:
                return
            self._reset_auto_for_new_run()

    def _on_land_cover_retry(self) -> None:

        if getattr(self, "_lc_store", None) is None:
            return
        try:
            self.dock_widget.set_land_cover_error(False)
        except (RuntimeError, AttributeError):  # nosec B110
            pass
        self._land_cover_start_partition(first=True)

    def _land_cover_start_partition(self, first: bool = False) -> None:
        store, geo = self._lc_store, self._lc_geo
        zone_wkb, scale = self._lc_zone_wkb, self._lc_area_scale
        min_patch = self._lc_min_patch


        grid = store.acquire() if store is not None else None
        if grid is None:
            return
        released = []

        def _release_once(*_args):


            if not released:
                released.append(True)
                store.release()

        def _build():
            from ...core.land_cover_blocks import build_partition_blocked
            try:
                return build_partition_blocked(grid, geo, zone_wkb, min_patch, scale)
            finally:
                _release_once()

        if not first:
            self._land_cover_push_card(busy=True)
        self._land_cover_run_task(
            _build, lambda r, f=first: self._on_land_cover_partition(r, f))
        self._lc_task.taskTerminated.connect(_release_once)

    def _on_land_cover_partition(self, result: dict, first: bool) -> None:
        self._lc_result = result
        self._land_cover_fill_layer()
        if first:
            self._land_cover_open_screen()
        else:
            self._land_cover_push_card(busy=False)



    def _land_cover_legend(self) -> list[dict]:


        from ...core.land_cover import legend_rows



        legend = (getattr(self, "_lc_served_legend", None)
                  or getattr(self, "_lc_answer_legend", None) or [])
        present = (self._lc_result or {}).get("areas", {}).keys()
        return legend_rows(present, legend, tr("Class {n}"))

    def _land_cover_total_m2(self) -> float:


        zone = float(getattr(self, "_lc_zone_m2", 0.0) or 0.0)
        return zone if zone > 0 else float(
            sum((self._lc_result or {}).get("areas", {}).values()))

    def _land_cover_rows(self) -> list[dict]:
        areas = (self._lc_result or {}).get("areas", {})
        hidden = getattr(self, "_lc_hidden", set())
        return [dict(row, area_m2=float(areas.get(row["id"], 0.0)),
                     shown=row["id"] not in hidden)
                for row in self._land_cover_legend()]



    def _land_cover_fill_layer(self) -> None:

        self._land_cover_drop_preview()
        from ...core.output_store import mark_temp_layer

        rows = self._land_cover_legend()
        names = {r["id"]: r["name"] for r in rows}
        layer = getattr(self, "_lc_layer", None)
        if layer is None or not self._land_cover_layer_alive():
            crs = str(getattr(self, "_auto_crs_authid", "") or "EPSG:4326")
            layer = QgsVectorLayer(f"MultiPolygon?crs={crs}", tr("Land cover"), "memory")
            pr = layer.dataProvider()
            from ...core.qt_compat import field_type_string
            pr.addAttributes([QgsField("class", field_type_string())])
            layer.updateFields()
            try:
                mark_temp_layer(layer)
            except Exception:  # noqa: BLE001  # nosec B110
                pass
            QgsProject.instance().addMapLayer(layer)
            self._lc_layer = layer
        pr = layer.dataProvider()
        pr.truncate()
        feats = []
        from ...core.layer_conventions import to_multipolygon

        for cid, wkb in self._lc_result.get("patches", []):
            geom = QgsGeometry()
            geom.fromWkb(wkb)
            for part in polygon_parts(geom):
                multi = to_multipolygon(part)
                if multi is None or multi.isEmpty():
                    continue
                feat = QgsFeature(layer.fields())
                feat.setGeometry(multi)
                feat.setAttributes([names.get(cid, str(cid))])
                feats.append(feat)
        if feats:
            pr.addFeatures(feats)
        layer.updateExtents()
        self._land_cover_apply_renderer()

    def _land_cover_layer_alive(self) -> bool:
        layer = getattr(self, "_lc_layer", None)
        if layer is None:
            return False
        try:
            return QgsProject.instance().mapLayer(layer.id()) is not None
        except RuntimeError:
            return False

    def _land_cover_category_index(self, class_id: int) -> int:
        for i, row in enumerate(self._land_cover_legend()):
            if row["id"] == class_id:
                return i
        return -1

    def _land_cover_apply_renderer(self, focus_id=None) -> None:



        from ...core.layer_conventions import make_legend_categorized_renderer

        if not self._land_cover_layer_alive():
            return
        rows = self._land_cover_legend()
        renderer = make_legend_categorized_renderer(
            [(r["name"], r["color"]) for r in rows])
        if renderer is None:
            return
        hidden = getattr(self, "_lc_hidden", set())
        cats = renderer.categories()
        for i, row in enumerate(rows):
            renderer.updateCategoryRenderState(i, row["id"] not in hidden)
            if focus_id is not None and row["id"] != focus_id:
                symbol = cats[i].symbol().clone()
                symbol.setOpacity(0.18)
                renderer.updateCategorySymbol(i, symbol)
        self._lc_layer.setRenderer(renderer)
        self._lc_layer.triggerRepaint()

    def _on_land_cover_hover(self, class_id) -> None:

        self._land_cover_apply_renderer(class_id)

    def _on_land_cover_toggle(self, class_id) -> None:

        if class_id is None or class_id < 0:
            return
        hidden = getattr(self, "_lc_hidden", set())
        if class_id in hidden:
            hidden.discard(class_id)
        else:
            hidden.add(class_id)
        self._lc_hidden = hidden
        self._land_cover_apply_renderer()
        self._land_cover_push_card(busy=False)



    def _land_cover_open_screen(self) -> None:
        dock = self.dock_widget
        self._lc_hidden = set()
        if dock is None:
            return
        dock._lc_actions = {
            "hover": self._on_land_cover_hover,
            "toggle": self._on_land_cover_toggle,
            "min_patch": self._on_land_cover_min_patch,
            "export": self._on_land_cover_export,
            "copy": self._on_land_cover_copy,
            "rerun": self._on_land_cover_rerun,
            "exit": self._on_land_cover_exit,
        }
        dock._lc_actions["retry"] = self._on_land_cover_retry
        try:
            dock.show_land_cover_result(True)
            self._set_zone_band_fill_visible(False)
            self._land_cover_push_card(busy=False, min_patch=self._lc_min_patch, strict=True)
        except Exception as exc:  # noqa: BLE001
            import traceback
            QgsMessageLog.logMessage(
                "Land cover: showing the result failed: "
                + "".join(traceback.format_exception_only(type(exc), exc)).strip()[:300],
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            self._lc_result = None
            self._land_cover_show_error()
            return
        self._last_auto_result = {
            "status": "completed", "instances": len(self._lc_result.get("patches", [])),
            "tiles_processed": getattr(self, "_lc_tiles_succeeded", 0), "layer_name": None}
        try:
            from ...core import telemetry_run_events
            from .auto_client_profile import client_profile_props
            ctx = self._auto_run_ctx or {}
            tiles = int(getattr(self, "_lc_tiles_succeeded", 0) or 0)
            patches = len(self._lc_result.get("patches", []))
            if self._auto_tel_stop_reason in (None, "completed"):
                telemetry_run_events.track_auto_detect_completed(
                    run_id=self._auto_run_id or "",
                    duration_ms=self._auto_duration_ms(),
                    tiles_done=tiles,
                    tiles_failed=max(0, int(ctx.get("total", tiles) or tiles) - tiles),
                    instances_found=patches,
                    instances_visible_at_default=patches,
                    zero_at_default=patches == 0,
                    stop_reason="completed",
                    warming_ms=self._auto_warming_wait_ms(),
                    merge_mode_final="map",
                    blob_armed=0, blob_dropped=0, tile_ground_m=0.0,
                    client_profile=client_profile_props(self),
                )
            self._remember_auto_run_pace(tiles)
            telemetry_run_events.track_land_cover_shown(
                run_id=self._auto_run_id or "",
                class_count=sum(1 for v in self._lc_result.get("areas", {}).values() if v > 0),
                zone_km2=float(getattr(self, "_lc_zone_m2", 0.0)) / 1e6)
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _land_cover_push_card(self, busy: bool, min_patch: float | None = None,
                              strict: bool = False) -> None:
        dock = self.dock_widget
        if dock is None or getattr(self, "_lc_result", None) is None:
            return
        try:
            dock.set_land_cover_result(
                self._land_cover_rows(), self._land_cover_total_m2(),
                min_patch_m2=min_patch, busy=busy)
        except (RuntimeError, AttributeError):
            if strict:
                raise

    def _on_land_cover_min_patch(self, value: float) -> None:
        value = max(0.0, float(value))
        if abs(value - (getattr(self, "_lc_min_patch", None) or 0.0)) < 1e-9:
            return
        previous = self._lc_min_patch
        self._lc_min_patch = value
        self._land_cover_start_partition(first=False)
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_land_cover_min_patch_changed(
                run_id=self._auto_run_id or "", min_patch_m2=value,
                from_default=abs(previous - getattr(self, "_lc_default_patch", 0.0)) < 1e-9)
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _land_cover_table_text(self) -> str:
        from ...core.land_cover import table_tsv

        rows = sorted(self._land_cover_rows(), key=lambda r: -r["area_m2"])
        return table_tsv(rows, self._land_cover_total_m2(),
                         (tr("Class"), tr("Area (m²)"), tr("Share (%)")), tr("Total"))

    def _on_land_cover_copy(self) -> None:
        from qgis.PyQt.QtWidgets import QApplication

        QApplication.clipboard().setText(self._land_cover_table_text())
        try:
            self.iface.messageBar().pushInfo(
                "AI Segmentation", tr("Land cover table copied"))
        except (RuntimeError, AttributeError):  # nosec B110
            pass
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_land_cover_table_copied(
                run_id=self._auto_run_id or "",
                class_count=sum(1 for r in self._land_cover_rows() if r["area_m2"] > 0))
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _land_cover_write_layer(self, autosave: bool = False) -> str | None:




        result = getattr(self, "_lc_result", None)
        if not result:
            return None
        rows = self._land_cover_legend()
        names = {r["id"]: r["name"] for r in rows}
        hidden = set() if autosave else getattr(self, "_lc_hidden", set())
        patches = result.get("patches", [])
        geoms, classes, ids, codes = [], [], [], []
        for i, (cid, wkb) in enumerate(patches):
            if cid in hidden:
                continue
            geom = QgsGeometry()
            geom.fromWkb(wkb)
            geoms.append(geom)
            classes.append(names.get(cid, str(cid)))
            ids.append(f"lc-{i + 1}")

            codes.append(int(cid) + 1)
        selected_ids = {code - 1 for code in codes}
        source_layer = self._get_active_raster_layer()
        source_name = ""
        try:
            source_name = source_layer.name() if source_layer is not None else ""
        except (RuntimeError, AttributeError):
            source_name = ""
        crs = self._run_export_crs(source_layer)
        export = self._prepare_auto_export(
            geoms, crs, source_name, tr("Land cover"), scores=None, det_ids=ids,
            row_classes=classes,
            class_colors=[(r["name"], r["color"]) for r in rows if r["id"] in selected_ids],
            land_cover={"class_codes": codes, "run_id": self._auto_run_id or "",


                        "min_area_m2": float(result.get("min_patch_m2") or 0.0)})
        if export is None:
            return None
        from ...core.run_export_job import run_export_job

        layer_name = self._adopt_auto_export(export, run_export_job(export["job"]))
        if not layer_name:
            return None
        exported_ids = {code - 1 for code in export["job"]["land_cover"]["class_codes"]}
        self._lc_last_areas = {cid: area for cid, area in result.get("areas", {}).items()
                               if cid in exported_ids}
        count = int(getattr(self, "_auto_export_feature_count", 0) or 0)
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_auto_export_done(
                run_id=self._auto_run_id or "", exported_count=count,
                visible_pct_of_found=int(round(100 * len(geoms) / max(1, len(patches)))),
                final_confidence=0,
                display_mode="land_cover", refined_in_manual=False,
                autosave=autosave, land_cover=True)
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return layer_name

    def _on_land_cover_export(self) -> None:

        layer_name = self._land_cover_write_layer()
        if not layer_name:
            try:
                message = (tr("Show a class on the map to export it.")
                           if getattr(self, "_auto_export_failure", "") == "nothing_visible"
                           else tr("The land cover layer could not be saved."))
                self.iface.messageBar().pushWarning(
                    "AI Segmentation", message)
            except (RuntimeError, AttributeError):  # nosec B110
                pass
            return
        count = int(getattr(self, "_auto_export_feature_count", 0) or 0)
        recap_id = getattr(self, "_auto_export_layer_id", "")
        self._land_cover_teardown()
        self._reset_auto_for_new_run()
        try:
            if self.dock_widget:
                self.dock_widget.set_land_cover_export_success(
                    sum(1 for v in (self._lc_last_areas or {}).values() if v > 0),
                    count, layer_name, recap_id)
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _on_land_cover_rerun(self) -> None:

        self._land_cover_teardown()
        dock = self.dock_widget
        if dock is None:
            return
        try:
            dock.set_auto_review_active(False)
            self._set_zone_band_fill_visible(True)
            dock.set_auto_zone_state("zone_set")
            self._restore_tile_grid_after_run()
            self._refresh_exemplar_chips()
            dock.set_auto_status("idle")
        except (RuntimeError, AttributeError):
            pass

    def _on_land_cover_exit(self) -> None:

        from ..dialogs.confirm_dialog import DISCARD, PRIMARY, SECONDARY, ChoiceButton, ask_choice

        clicked = ask_choice(
            self.iface.mainWindow(), tr("Keep your land cover map?"),
            tr("Save the land cover map to a layer before leaving?"),
            [ChoiceButton("discard", tr("Discard && exit"), DISCARD),
             ChoiceButton("cancel", tr("Cancel"), SECONDARY),
             ChoiceButton("save", tr("Save && exit"), PRIMARY)],
            default="save", escape="cancel")
        if clicked == "save":
            if getattr(self, "_lc_result", None) is None:

                if self._land_cover_autosave_pending("exit_button") is False:
                    return
                self._reset_auto_for_new_run()
                return
            self._on_land_cover_export()
            return
        if clicked != "discard":
            return
        self._land_cover_teardown()
        self._reset_auto_for_new_run()

    def _land_cover_autosave_pending(self, exit_path: str = "other") -> bool:




        if (getattr(self, "_lc_result", None) is None
                and getattr(self, "_lc_store", None) is None):
            return True
        if getattr(self, "_lc_result", None) is None:
            if not self._land_cover_build_now():
                self._land_cover_save_refused()
                return False
        try:
            name = self._land_cover_write_layer(autosave=True)
            QgsMessageLog.logMessage(
                f"Land cover: result autosaved on leave ({exit_path})"
                if name else f"Land cover: autosave on leave failed ({exit_path})",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info if name else Qgis.MessageLevel.Warning)
        except Exception:  # noqa: BLE001
            name = None
        if not name:
            self._land_cover_save_refused()
            return False
        self._land_cover_teardown()
        return True

    def _land_cover_save_refused(self) -> None:

        self._headless_error = tr("The land cover layer could not be saved.")
        try:
            self.iface.messageBar().pushWarning("AI Segmentation", self._headless_error)
        except (RuntimeError, AttributeError):
            pass

    def _land_cover_build_now(self) -> bool:


        from ...core.land_cover_blocks import build_partition_blocked

        store = getattr(self, "_lc_store", None)
        grid = store.acquire() if store is not None else None
        if grid is None:
            return False
        try:
            plan = self._auto_land_cover or {}
            if getattr(self, "_lc_min_patch", None) is None:
                self._lc_min_patch = float(plan.get("min_patch_m2") or 0.0)
            self._lc_gen = getattr(self, "_lc_gen", 0) + 1
            self._lc_result = build_partition_blocked(
                grid, self._lc_geo, self._lc_zone_wkb, self._lc_min_patch,
                self._lc_area_scale)
            return bool(self._lc_result.get("patches"))
        except Exception as exc:  # noqa: BLE001
            QgsMessageLog.logMessage(
                f"Land cover: building the map failed ({type(exc).__name__})",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return False
        finally:
            store.release()

    def _land_cover_finish_headless(self, tiles_succeeded: int) -> None:


        self._lc_mosaic = None
        if not self._land_cover_build_now():
            self._land_cover_teardown()
            self._record_auto_zero_result(tiles_succeeded)
            return
        patches = len(self._lc_result.get("patches", []))
        if not patches:
            self._land_cover_teardown()
            self._record_auto_zero_result(tiles_succeeded)
            return
        layer_name = self._land_cover_write_layer()
        if not layer_name:

            self._last_auto_result = {
                "status": "error",
                "message": "The land cover layer could not be saved. It remains in the panel for export.",
                "instances": patches, "tiles_processed": tiles_succeeded,
                "layer_name": None}
            self._land_cover_fill_layer()
            self._land_cover_open_screen()
            return
        self._land_cover_teardown()
        result = {"status": "completed", "instances": patches,
                  "tiles_processed": tiles_succeeded, "layer_name": layer_name,
                  "layer_id": getattr(self, "_auto_export_layer_id", "") or None}
        prior = self._last_auto_result
        if isinstance(prior, dict) and prior.get("status") == "credits_exhausted":
            result["status"] = "credits_exhausted"
            result["credits_remaining"] = prior.get("credits_remaining", 0)
        self._last_auto_result = result
        QgsMessageLog.logMessage(
            f"Land cover: exported {patches} patch(es)", "AI Segmentation",
            level=Qgis.MessageLevel.Info)

    def _land_cover_teardown(self) -> None:

        self._lc_gen = getattr(self, "_lc_gen", 0) + 1
        self._lc_mosaic = None
        store = getattr(self, "_lc_store", None)
        self._lc_store = None
        task = getattr(self, "_lc_task", None)
        self._lc_task = None
        if task is not None:
            try:
                task.cancel()
            except RuntimeError:  # nosec B110
                pass
        if store is not None:
            try:

                store.close()
            except Exception:  # noqa: BLE001  # nosec B110
                pass
        self._lc_min_patch = None
        self._lc_zone_wkb = None
        self._lc_area_scale = 1.0
        self._lc_zone_m2 = 0.0
        self._lc_result = None
        self._lc_answer_legend = None
        self._lc_hidden = set()
        self._land_cover_drop_preview()
        layer = getattr(self, "_lc_layer", None)
        self._lc_layer = None
        if layer is not None:
            try:
                if QgsProject.instance().mapLayer(layer.id()) is not None:
                    QgsProject.instance().removeMapLayer(layer.id())
            except RuntimeError:  # nosec B110
                pass
        dock = self.dock_widget
        if dock is not None:
            try:
                dock.show_land_cover_result(False)


                if (getattr(self, "_auto_review", None) is None
                        and dock.__dict__.get("_auto_review_active")):
                    dock.set_auto_review_active(False)
            except (RuntimeError, AttributeError):  # nosec B110
                pass


def _land_cover_grid_path() -> str:

    import os
    import tempfile
    import uuid

    from ...core.land_cover import GRID_FILE_PREFIX, GRID_FILE_SUFFIX

    return os.path.join(tempfile.gettempdir(),
                        f"{GRID_FILE_PREFIX}{uuid.uuid4().hex[:12]}{GRID_FILE_SUFFIX}")
