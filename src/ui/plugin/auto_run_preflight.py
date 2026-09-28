







from __future__ import annotations

import math

from qgis.core import (
    Qgis,
    QgsMessageLog,
)

from ...core.i18n import tr


class AutoRunPreflightMixin:


    def _auto_raster_guard_message(self, layer) -> str | None:










        self._auto_raster_guard_reason = "raster_shape"




        try:
            crs_valid = layer.crs().isValid()
        except (RuntimeError, AttributeError):
            crs_valid = False
        if not crs_valid:
            self._auto_raster_guard_reason = "raster_no_crs"
            return tr(
                "This layer has no valid coordinate reference system. "
                "Set one in Layer Properties before detecting."
            )





        from ...core.layer_conventions import crs_disagrees_with_extent
        try:
            mismatch = crs_disagrees_with_extent(layer.crs(), layer.extent())
        except (RuntimeError, AttributeError):
            mismatch = ""
        if mismatch == "degrees_in_metric_crs":
            self._auto_raster_guard_reason = "crs_mismatch"
            return tr(
                "This layer is filed under {crs}, which counts in metres, but "
                "its coordinates are longitude and latitude. Detection would "
                "measure the whole image as under a millimetre of ground and "
                "return nothing. Set the layer's CRS to the one the pixels "
                "are really in (EPSG:4326 for plain longitude and latitude) "
                "in Layer Properties, or reproject it."
            ).format(crs=self._auto_layer_crs_label(layer))
        if mismatch == "metres_in_geographic_crs":
            self._auto_raster_guard_reason = "crs_mismatch"
            return tr(
                "This layer is filed under {crs}, which counts in degrees, "
                "but its coordinates are projected metres. Set the layer's "
                "CRS to the projected one the pixels are really in, in Layer "
                "Properties, before detecting."
            ).format(crs=self._auto_layer_crs_label(layer))
        try:
            online = self._needs_canvas_render(layer)
        except (RuntimeError, AttributeError):
            online = False
        if not online:
            try:
                georef = self._is_layer_georeferenced(layer)
            except (RuntimeError, AttributeError):
                georef = True
            if not georef:
                self._auto_raster_guard_reason = "raster_not_georeferenced"




                return tr(
                    "This image has no position on the map, so Automatic "
                    "cannot place what it finds. Give it one with the QGIS "
                    "Georeferencer, or use Semi-Auto mode on it as is."
                )
            if self._raster_is_rotated(layer):
                self._auto_raster_guard_reason = "raster_rotated"





                return tr(
                    "This raster is rotated. Run Warp (Reproject) on it to "
                    "straighten it first. Semi-Auto mode cannot read it "
                    "either."
                )
        return None

    @staticmethod
    def _auto_layer_crs_label(layer) -> str:






        try:
            crs = layer.crs()
            return str(crs.authid() or crs.description() or "").strip() or tr("its CRS")
        except (RuntimeError, AttributeError):
            return tr("its CRS")





    _DRAWN_MAP_BASEMAPS = ("OSM", "Carto")

    def _warn_drawn_map_basemap(self, layer) -> None:








        if getattr(self, "_auto_imagery_resume", None) is not None:
            return
        try:
            from ...core.basemap_label import detect_basemap_label
            label = detect_basemap_label(layer)
        except Exception:  # noqa: BLE001
            return
        if label not in self._DRAWN_MAP_BASEMAPS:
            return
        try:
            self.iface.messageBar().pushWarning(
                "AI Segmentation",
                tr("{basemap} is a drawn map, not aerial imagery, so detection "
                   "usually finds nothing on it and the tiles are still "
                   "charged. Switch the layer to a satellite basemap "
                   "(Google, Esri, Bing) or to your own raster first."
                   ).format(basemap=label))
        except (RuntimeError, AttributeError):
            pass

    def _auto_imagery_notice_passes(self, layer, grid, mupp_floor: float = 0.0) -> bool:












        if self._auto_headless_run or self.dock_widget is None:
            return True
        try:

            override_key = (layer.id(), self._imagery_layer_source(layer))
        except (RuntimeError, AttributeError):
            return True
        allowed = getattr(self, "_auto_imagery_notice_overrides", None)
        if allowed is None:
            allowed = set()
            self._auto_imagery_notice_overrides = allowed
        if override_key in allowed:
            return True
        copy = None
        notice_kind = ""
        gsd_m = 0.0
        try:
            kind = self._imagery_content_kind(layer, grid)
            if kind is not None:
                copy = self._imagery_content_copy(kind)
                notice_kind = kind
            else:
                coarse = self._coarse_imagery_copy(layer, grid, mupp_floor)
                if coarse is not None:
                    copy, gsd_m = coarse[:2], coarse[2]
                    notice_kind = "too_coarse"
        except Exception as exc:  # noqa: BLE001
            QgsMessageLog.logMessage(
                f"Auto detection: imagery check skipped ({type(exc).__name__})",
                "AI Segmentation", level=Qgis.MessageLevel.Info)
            copy = None
        if copy is None:
            return True
        title, body = copy
        from ..dialogs.confirm_dialog import (
            PRIMARY,
            SECONDARY,
            WARNING,
            ChoiceButton,
            ask_choice,
        )



        choice = ask_choice(
            self.iface.mainWindow(), title, body,
            [ChoiceButton("run", tr("Run anyway"), SECONDARY),
             ChoiceButton("cancel", tr("Cancel"), PRIMARY)],
            default="cancel", escape="cancel", tone=WARNING)
        self._track_imagery_notice(notice_kind, choice == "run", gsd_m)
        if choice != "run":
            QgsMessageLog.logMessage(
                "Auto detection: stopped at the imagery notice; nothing sent",
                "AI Segmentation", level=Qgis.MessageLevel.Info)
            return False
        allowed.add(override_key)
        return True

    def _track_imagery_notice(self, kind: str, run_anyway: bool, gsd_m: float) -> None:

        try:
            from ...core.telemetry_run_events import track_auto_imagery_notice

            track_auto_imagery_notice(
                kind, run_anyway, prompt=self._resolved_auto_object_class() or "",
                source_m_per_px=gsd_m)
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    @staticmethod
    def _imagery_layer_source(layer) -> str:

        try:
            return str(layer.dataProvider().dataSourceUri() or "")
        except (RuntimeError, AttributeError):
            return ""

    @staticmethod
    def _imagery_content_copy(kind: str) -> tuple[str, str]:
        from ...core.imagery_content import TERRAIN

        if kind == TERRAIN:
            return (tr("This looks like terrain shading, not a photo"),
                    tr("Detection works on aerial or satellite images."))
        return (tr("This looks like a drawn map, not a photo"),
                tr("Detection works on aerial or satellite images."))

    def _imagery_content_kind(self, layer, grid) -> str | None:








        from ...core.imagery_content import TERRAIN

        try:
            if layer.renderer().type() == "hillshade":
                return TERRAIN
        except (RuntimeError, AttributeError):
            pass
        kept = getattr(self, "_auto_imagery_content", None) or {}
        signature = self._imagery_probe_signature(layer, grid)
        if signature in kept:
            return kept[signature]
        if self._needs_canvas_render(layer):
            return None
        import time

        from ...core.cloud_detection import render_zone_to_image
        from ...core.server_dials import dial_in_range
        from ...core.shape_policy_dials import imagery_probe_px

        side = imagery_probe_px(256)
        render_crs = self._probe_render_crs(grid)



        grow = self._content_sample_growth(layer, grid)


        budget_ms = 1000.0 * dial_in_range(
            "tuning.preflight.content_check_budget_s", 1.5, 0.2, 10.0)
        started = time.monotonic()
        images = []
        for centre in self._probe_centres(layer, grid):
            extent = self._imagery_probe_extent(grid, centre)
            if extent is None:
                continue
            if grow > 1.0:
                extent.scale(grow)
            left_ms = int(budget_ms - 1000.0 * (time.monotonic() - started))
            if left_ms <= 0:
                return self._keep_imagery_content_verdict(signature, [])
            img, _actual = render_zone_to_image(
                layer, extent, side, side, timeout_ms=left_ms, render_crs=render_crs)
            if img is None and 1000.0 * (time.monotonic() - started) >= budget_ms:
                return self._keep_imagery_content_verdict(signature, [])
            if img is not None:
                images.append(img)
        return self._keep_imagery_content_verdict(signature, images)

    def _content_sample_growth(self, layer, grid) -> float:


        try:
            authid = (grid or {}).get("crs") or ""
            if authid and authid != layer.crs().authid():
                return 1.0
            run_mupp = self._grid_mupp(grid)
            native = max(float(layer.rasterUnitsPerPixelX()),
                         float(layer.rasterUnitsPerPixelY()))
            if run_mupp <= 0 or native <= run_mupp:
                return 1.0
            return native / run_mupp
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return 1.0

    def _keep_imagery_content_verdict(self, signature, images) -> str | None:






        verdict = None
        try:
            from ...core.imagery_content import (
                imagery_content_verdict,
                qimage_to_rgb_array,
            )

            verdicts = [imagery_content_verdict(qimage_to_rgb_array(img))
                        for img in images if img is not None]
            for kind in set(verdicts) - {None}:
                if 2 * verdicts.count(kind) > len(verdicts):
                    verdict = kind
        except Exception:  # noqa: BLE001
            verdict = None
        if signature is None:
            return verdict
        kept = getattr(self, "_auto_imagery_content", None)
        if kept is None:
            kept = {}
            self._auto_imagery_content = kept


        while len(kept) >= 8:
            kept.pop(next(iter(kept)))
        kept[signature] = verdict
        return verdict

    def _coarse_imagery_copy(self, layer, grid,
                             mupp_floor: float) -> tuple[str, str, float] | None:












        from qgis.core import QgsRectangle

        from ...core.detection_policy import seed_policy
        from ...core.prompt_taxonomy import normalize_prompt
        from ...core.server_dials import dial_in_range

        object_class = self._resolved_auto_object_class()
        if not object_class:
            return None
        tiers = seed_policy().get("object_tiers")
        if not isinstance(tiers, list):
            return None
        obj_m = self._largest_matching_object_m(normalize_prompt(object_class), tiers)
        max_obj_m = dial_in_range("tuning.preflight.coarse_object_max_m", 30.0, 1.0, 500.0)
        if obj_m <= 0 or obj_m > max_obj_m:
            return None
        if self._needs_canvas_render(layer):
            if mupp_floor <= 0:
                return None
            minx, miny, maxx, maxy = grid["bbox"]
            gsd = self._mupp_to_meters(layer, QgsRectangle(minx, miny, maxx, maxy), mupp_floor)
        else:
            gsd = self._native_ground_mupp(layer)
        if gsd <= 0:
            return None
        min_px = dial_in_range("tuning.preflight.coarse_min_object_px", 3.0, 0.5, 20.0)
        px = obj_m / gsd
        if px >= min_px:
            return None
        word = (self._current_auto_object_class() or object_class).strip()


        if px < 10:
            px_text = f"{max(0.1, math.floor(px * 10) / 10):.1f}"
        else:
            px_text = str(int(px))
        return (tr("At {gsd} m per pixel, one {object} is about {px} pixels wide").format(
                    gsd=f"{gsd:.1f}", object=word, px=px_text),
                tr("Detection needs sharper imagery to find it."), float(gsd))

    @staticmethod
    def _largest_matching_object_m(text: str, tiers: list) -> float:


        from ...core.prompt_taxonomy import iter_keywords, keyword_matches

        best = 0.0
        for entry in tiers:
            if not isinstance(entry, dict):
                continue
            if not any(keyword_matches(text, kw) for kw in iter_keywords(entry)):
                continue
            try:
                size_m = float(entry.get("size_m"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(size_m) and size_m > best:
                best = size_m
        return best

    @staticmethod
    def _native_ground_mupp(layer) -> float:


        try:
            from qgis.core import QgsDistanceArea, QgsPointXY, QgsProject

            from ...core.qt_compat import DistanceMeters

            upp_x = float(layer.rasterUnitsPerPixelX())
            upp_y = float(layer.rasterUnitsPerPixelY())
            if upp_x <= 0 or upp_y <= 0:
                return 0.0
            centre = layer.extent().center()
            da = QgsDistanceArea()
            da.setSourceCrs(layer.crs(), QgsProject.instance().transformContext())
            da.setEllipsoid("WGS84")
            cx, cy = centre.x(), centre.y()
            dist = max(da.measureLine(QgsPointXY(cx, cy), QgsPointXY(cx + upp_x, cy)),
                       da.measureLine(QgsPointXY(cx, cy), QgsPointXY(cx, cy + upp_y)))
            return float(da.convertLengthMeasurement(dist, DistanceMeters))
        except Exception:  # noqa: BLE001
            return 0.0

    def _warn_local_raster_quality(self, layer) -> None:














        try:
            if self._needs_canvas_render(layer):
                return
        except (RuntimeError, AttributeError):
            return
        try:
            if layer.crs().isValid() and layer.crs().isGeographic():
                zone = self._auto_zone
                if zone is None:






                    run_crs = layer.crs()
                else:
                    run_crs = self._run_crs_for_layer(
                        layer, self._zone_in_layer_crs(zone, layer))
                if run_crs is None or run_crs == layer.crs():
                    self.iface.messageBar().pushInfo(
                        "AI Segmentation",
                        tr("This raster uses a geographic CRS (degrees), which "
                           "distorts the imagery sent to the AI. For best "
                           "results, reproject it to a projected CRS (e.g. UTM)."))
                    return
        except (RuntimeError, AttributeError):
            pass


        try:
            source = layer.source() or ""
            low = source.lower()
            if low.startswith("/vsi") or "://" in low:
                return
            import os
            if not os.path.isfile(source):
                return
            from ...core.server_dials import dial_copy, dial_in_range
            tip_floor_bytes = dial_in_range(
                "tuning.processing.overview_tip_min_bytes", 512 * 1024 * 1024,
                1024 * 1024, 10 * 1024 * 1024 * 1024)
            if os.path.getsize(source) < tip_floor_bytes:
                return
            from ...core.raster_dataset_cache import acquire_gdal_dataset
            ds = acquire_gdal_dataset(source)
            if ds is None or ds.RasterCount < 1:
                return
            has_overviews = ds.GetRasterBand(1).GetOverviewCount() > 0
            if not has_overviews:
                self.iface.messageBar().pushInfo(
                    "AI Segmentation",
                    dial_copy("copy.auto.overview_tip", tr(
                        "Tip: this raster has no overviews (pyramids). "
                        "Build them (Raster menu, Miscellaneous, Build "
                        "Overviews) to make detection much faster.")))
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    @staticmethod
    def _raster_is_rotated(layer) -> bool:






        try:
            source = layer.source()
        except (RuntimeError, AttributeError):
            return False
        if not source:
            return False


        low = source.lower()
        if low.startswith("/vsi") or "://" in low:
            return False
        try:


            from ...core.raster_dataset_cache import acquire_gdal_dataset
            ds = acquire_gdal_dataset(source)
            if ds is None:
                return False
            gt = ds.GetGeoTransform()
            if gt is None:
                return False
            scale = max(abs(gt[1]), abs(gt[5]), 1e-9)
            return (abs(gt[2]) / scale > 1e-3) or (abs(gt[4]) / scale > 1e-3)
        except Exception:  # noqa: BLE001
            return False

    def _offer_automatic_setup(self, reason: str) -> None:







        from ..dialogs.confirm_dialog import (
            PRIMARY,
            SECONDARY,
            ChoiceButton,
            ask_choice,
        )

        if ask_choice(
            self.iface.mainWindow(), tr("One-time setup"), reason,
            [ChoiceButton("later", tr("Not now"), SECONDARY),
             ChoiceButton("setup", tr("Set up now"), PRIMARY)],
            default="setup", escape="later",
        ) != "setup":
            return



        self._on_install_requested(include_local_model=False)

    def _push_auto_warning(self, message: str) -> None:






        if self._auto_headless_run:
            return
        try:
            self.iface.messageBar().pushWarning("AI Segmentation", message)
        except (RuntimeError, AttributeError):
            pass

    def _probe_imagery_behind_banner(self, layer, grid) -> tuple[float, str | None] | None:














        resume = getattr(self, "_auto_imagery_resume", None)
        self._auto_imagery_resume = None
        if self._auto_headless_run or self.dock_widget is None:
            return self._online_imagery_verdict(layer, grid)
        signature = self._imagery_probe_signature(layer, grid)
        if resume is not None:
            if resume.get("signature") == signature:
                return resume["verdict"]
            QgsMessageLog.logMessage(
                "Auto detection: the zone changed during the imagery check; "
                "not starting", "AI Segmentation", level=Qgis.MessageLevel.Info)
            return None
        from ...core import run_timeline


        early = self._early_imagery_answer(signature)
        if early is not None:
            run_timeline.mark("imagery_probe_early_hit")
            return early
        probe = {"signature": signature, "banner": None}
        self._auto_imagery_probe = probe
        run_timeline.mark("imagery_probe_start")

        def _done(verdict) -> None:
            self._on_imagery_probe_done(probe, verdict)

        if self._adopt_early_imagery_probe(signature, probe):


            run_timeline.mark("imagery_probe_early_adopted")
        elif not self._start_online_imagery_verdict(layer, grid, _done):
            self._auto_imagery_probe = None
            return 0.0, None
        if self._auto_imagery_probe is not probe:


            return None
        try:
            banner = self.dock_widget.auto_status_banner
            banner.setText(tr("Preparing your zone..."))
            banner.setVisible(True)
            probe["banner"] = banner
        except (RuntimeError, AttributeError):
            probe["banner"] = None
        return None

    def _imagery_probe_signature(self, layer, grid) -> tuple:


        try:
            layer_id = layer.id()
        except (RuntimeError, AttributeError):
            layer_id = ""
        try:
            bbox = tuple(round(float(v), 6) for v in (grid or {}).get("bbox") or ())
        except (TypeError, ValueError):
            bbox = ()
        outline = b""
        try:
            polygon = getattr(self, "_auto_zone_polygon", None)
            if polygon is not None and not polygon.isEmpty():
                outline = bytes(polygon.asWkb())
        except (RuntimeError, AttributeError):
            outline = b""
        return (layer_id, self._imagery_layer_source(layer), bbox,
                (grid or {}).get("pixel_w"),
                (grid or {}).get("pixel_h"), (grid or {}).get("crs") or "",
                outline)

    def _on_imagery_probe_done(self, probe: dict, verdict) -> None:



        if getattr(self, "_auto_imagery_probe", None) is not probe:
            return
        self._auto_imagery_probe = None
        from ...core import run_timeline
        run_timeline.mark("imagery_probe_done")
        self._hide_imagery_probe_banner(probe)
        if verdict is None:
            return
        try:
            from ...core.qt_compat import safe_single_shot

            safe_single_shot(0, self.dock_widget,
                             lambda: self._resume_detect_after_imagery(probe, verdict))
        except (RuntimeError, AttributeError):
            pass

    def _resume_detect_after_imagery(self, probe: dict, verdict) -> None:


        if getattr(self, "_auto_imagery_probe", None) is not None:
            return
        dock = self.dock_widget
        if dock is None:
            return
        try:
            from ..dock.widgets import Mode
            if getattr(dock, "_mode", None) != Mode.AUTOMATIC:
                return
        except (ImportError, RuntimeError, AttributeError):
            return



        try:
            layer = self._get_active_raster_layer()
            grid = self._compute_auto_grid(layer) if layer is not None else None
        except (RuntimeError, AttributeError, TypeError, ValueError):
            grid = None
        if grid is None or self._imagery_probe_signature(layer, grid) != probe["signature"]:
            QgsMessageLog.logMessage(
                "Auto detection: the zone changed during the imagery check; "
                "not starting", "AI Segmentation", level=Qgis.MessageLevel.Info)
            return
        self._auto_imagery_resume = {
            "signature": probe["signature"], "verdict": verdict}
        try:
            self._start_auto_detection()
        finally:
            self._auto_imagery_resume = None

    def _abandon_imagery_probe(self) -> None:




        probe = getattr(self, "_auto_imagery_probe", None)
        self._auto_imagery_resume = None
        early_out = self._drop_early_imagery_probe()
        if probe is None and not early_out:
            return
        self._auto_imagery_probe = None
        if probe is not None:
            self._hide_imagery_probe_banner(probe)


        self._cancel_active_tile_render()

    @staticmethod
    def _hide_imagery_probe_banner(probe: dict) -> None:
        banner = probe.get("banner")
        probe["banner"] = None
        if banner is None:
            return
        try:
            banner.setVisible(False)
        except RuntimeError:
            pass

    def _abort_zone_outside_layer(self) -> None:



        msg = tr(
            "The zone is outside the selected raster layer. "
            "Pick the right layer or redraw the zone."
        )
        self._headless_error = msg
        try:
            self.dock_widget.set_auto_status("error", msg)
        except (RuntimeError, AttributeError):
            pass
        self._push_auto_warning(msg)
        QgsMessageLog.logMessage(
            "Auto detection: zone covers no ground on this layer; aborting",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
