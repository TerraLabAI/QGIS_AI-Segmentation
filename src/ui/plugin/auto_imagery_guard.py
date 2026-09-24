






























from __future__ import annotations

import time

from ...core.i18n import tr






_PROBE_SIDE_PX: int = 256




_EARLY_PROBE_DELAY_MS: int = 400




_EARLY_VERDICT_TTL_S: float = 600.0


class AutoImageryGuardMixin:


    def _online_imagery_verdict(self, layer, grid) -> tuple[float, str | None]:



















        plan = self._imagery_probe_plan(layer, grid)
        if plan is None:
            return 0.0, None
        from ...core.cloud_detection import probe_depth_chain

        try:
            chains = []
            for extent in plan["extents"]:
                chains.append(probe_depth_chain(
                    layer, extent, render_crs=plan["crs"], count=plan["count"],
                    side_px=None, min_side_px=plan["min_side_px"]))


                if self._imagery_chain_verdict(chains[-1], plan)[1] is not None:
                    break
            return self._imagery_verdict_from_chains(chains, plan)
        except Exception:  # noqa: BLE001
            return 0.0, None

    def _start_online_imagery_verdict(self, layer, grid, on_done,
                                      online: bool | None = None) -> bool:
















        plan = self._imagery_probe_plan(layer, grid, online=online)
        if plan is None:
            return False
        from ...core.cloud_detection import start_probe_depth_chain

        extents = plan["extents"]
        state = {"chains": [None] * len(extents), "pending": len(extents),
                 "cancelled": False, "reported": False}

        def _finish() -> None:
            if state["reported"] or state["pending"] > 0:
                return
            state["reported"] = True
            if state["cancelled"]:
                verdict = None
            else:
                try:
                    verdict = self._imagery_verdict_from_chains(
                        [c or [] for c in state["chains"]], plan)
                except Exception:  # noqa: BLE001
                    verdict = (0.0, None)
            state["chains"] = []
            try:
                on_done(verdict)
            except Exception as exc:  # noqa: BLE001
                from qgis.core import Qgis

                from ...core.logging_utils import log
                log(f"Imagery check callback failed: {exc}", Qgis.MessageLevel.Warning)

        def _chain_done(index: int):
            def _done(images, cancelled) -> None:
                state["chains"][index] = images
                state["cancelled"] = state["cancelled"] or bool(cancelled)
                state["pending"] -= 1
                _finish()
            return _done

        started = 0
        for index, extent in enumerate(extents):
            try:
                ok = start_probe_depth_chain(
                    layer, extent, _chain_done(index), render_crs=plan["crs"],
                    count=plan["count"], side_px=None,
                    min_side_px=plan["min_side_px"])
            except Exception:  # noqa: BLE001
                ok = False
            if ok:
                started += 1
            else:


                state["chains"][index] = []
                state["pending"] -= 1
        if started == 0:
            return False
        _finish()
        return True

    def _imagery_probe_plan(self, layer, grid, online: bool | None = None) -> dict | None:


        if online is None:
            online = bool(getattr(self, "_auto_source_is_online", False))
        if not online:
            return None
        from ...core.server_dials import feature_enabled




        if not feature_enabled("imagery_probe"):
            return None
        try:
            mupp = self._grid_mupp(grid)
            centres = self._probe_centres(layer, grid)
            if not centres or mupp <= 0:
                return None
            extents = [e for e in (self._imagery_probe_extent(grid, c)
                                   for c in centres) if e is not None]
            if not extents:
                return None
            from ...core.render_depth import LEVEL_SAMPLE_PX, backoff_steps

            steps = backoff_steps()
            return {"mupp": mupp, "extents": extents, "steps": steps,
                    "count": steps + 2, "min_side_px": LEVEL_SAMPLE_PX,
                    "crs": self._probe_render_crs(grid)}
        except Exception:  # noqa: BLE001
            return None

    def _imagery_verdict_from_chains(self, chains: list, plan: dict) -> tuple[float, str | None]:


        floor = 0.0
        for images in chains:
            level, refusal = self._imagery_chain_verdict(images, plan)


            if refusal is not None:
                return 0.0, refusal
            floor = max(floor, level)
        return floor, None

    def _note_imagery_backoff(self) -> None:









        if getattr(self, "_auto_headless_run", False):
            return
        try:
            from ...core.server_dials import dial_copy

            self.iface.messageBar().pushInfo("AI Segmentation", dial_copy(
                "auto.imagery_backoff",
                tr("This map source has no sharper picture of this area. The "
                   "run uses the sharpest one it has.")))
        except Exception:  # noqa: BLE001  # nosec B110
            pass



    def _schedule_early_imagery_probe(self) -> None:



        if getattr(self, "_auto_headless_run", False) or self.dock_widget is None:
            return
        timer = getattr(self, "_auto_imagery_early_timer", None)
        try:
            if timer is None:
                from qgis.PyQt.QtCore import QTimer



                timer = QTimer(self.dock_widget)
                timer.setSingleShot(True)
                timer.timeout.connect(self._run_early_imagery_probe)
                self._auto_imagery_early_timer = timer
            try:
                from ...core.server_dials import dial_in_range
                delay_ms = dial_in_range(
                    "tuning.auto.imagery_early_probe_delay_ms", _EARLY_PROBE_DELAY_MS, 100, 2000)
            except Exception:  # noqa: BLE001
                delay_ms = _EARLY_PROBE_DELAY_MS
            timer.start(delay_ms)
        except (RuntimeError, AttributeError, TypeError):
            self._auto_imagery_early_timer = None

    def _run_early_imagery_probe(self) -> None:


        try:
            self._start_early_imagery_probe()
        except Exception:  # noqa: BLE001
            self._auto_imagery_early = None

    def _start_early_imagery_probe(self) -> None:
        if not self._early_imagery_probe_enabled():
            return


        if (self._auto_worker is not None or self._auto_review is not None
                or getattr(self, "_auto_imagery_probe", None) is not None
                or getattr(self, "_auto_headless_run", False)
                or self._auto_zone is None):
            return
        layer = self._get_active_raster_layer()
        if layer is None or not self._needs_canvas_render(layer):
            return
        grid = self._compute_auto_grid(layer)
        if grid is None:
            return
        signature = self._imagery_probe_signature(layer, grid)
        early = getattr(self, "_auto_imagery_early", None)
        if early is not None and early.get("signature") == signature and (
                not early.get("done") or self._early_verdict_fresh(early)):
            return
        if early is not None and not early.get("done"):


            early["superseded"] = True
        state = {"signature": signature, "done": False, "verdict": None,
                 "at": 0.0, "waiter": None}
        self._auto_imagery_early = state

        def _done(verdict) -> None:
            self._on_early_imagery_probe_done(state, verdict)

        if not self._start_online_imagery_verdict(layer, grid, _done, online=True):

            state.update(done=True, verdict=(0.0, None), at=time.monotonic())

    def _on_early_imagery_probe_done(self, state: dict, verdict) -> None:


        if getattr(self, "_auto_imagery_early", None) is not state:
            return
        waiter = state.get("waiter")
        state["waiter"] = None
        if verdict is None:

            self._auto_imagery_early = None
        else:
            state.update(done=True, verdict=verdict, at=time.monotonic())
        if waiter is not None:
            self._on_imagery_probe_done(waiter, verdict)

    @staticmethod
    def _early_verdict_fresh(state: dict) -> bool:
        try:
            from ...core.server_dials import dial_in_range
            ttl_s = dial_in_range(
                "tuning.auto.imagery_early_verdict_ttl_s", _EARLY_VERDICT_TTL_S, 60, 3600)
        except Exception:  # noqa: BLE001
            ttl_s = _EARLY_VERDICT_TTL_S
        return (bool(state.get("done"))
                and time.monotonic() - float(state.get("at") or 0.0) < ttl_s)

    def _early_imagery_answer(self, signature):


        early = getattr(self, "_auto_imagery_early", None)
        if (early is None or early.get("signature") != signature
                or not self._early_verdict_fresh(early)
                or not self._early_imagery_probe_enabled()):
            return None
        return early.get("verdict")

    @staticmethod
    def _early_imagery_probe_enabled() -> bool:


        from ...core.server_dials import feature_enabled

        return feature_enabled("imagery_probe_early")

    def _adopt_early_imagery_probe(self, signature, probe: dict) -> bool:



        early = getattr(self, "_auto_imagery_early", None)
        if (early is None or early.get("done") or early.get("superseded")
                or early.get("signature") != signature
                or early.get("waiter") is not None
                or not self._early_imagery_probe_enabled()):
            return False
        early["waiter"] = probe
        return True

    def _retire_early_imagery_probe(self) -> None:



        early = getattr(self, "_auto_imagery_early", None)
        if early is None or early.get("done"):
            return
        self._auto_imagery_early = None
        self._cancel_active_tile_render()

    def _drop_early_imagery_probe(self) -> bool:


        timer = getattr(self, "_auto_imagery_early_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except RuntimeError:
                self._auto_imagery_early_timer = None
        early = getattr(self, "_auto_imagery_early", None)
        self._auto_imagery_early = None
        return early is not None and not early.get("done")

    def _imagery_chain_verdict(self, images: list, plan: dict):







        from ...core.cloud_detection import render_verdict
        from ...core.render_depth import agreement_min, level_agreement

        if len(images) < 2:
            return 0.0, None
        steps = plan["steps"]
        mupp = plan["mupp"]
        floor_min = agreement_min()
        for step in range(min(steps + 1, len(images) - 1)):
            fine, coarse = images[step], images[step + 1]
            if render_verdict(fine) != "unavailable":
                return self._level_floor(mupp, step), None



            if level_agreement(fine, coarse) >= floor_min:
                return self._level_floor(mupp, step), None
        return 0.0, tr(
            "This layer has no image over your zone at this precision. "
            "The map source answered with an empty tile, so there is nothing "
            "to detect on. Lower Precision, zoom the layer out until the "
            "imagery shows, or pick a layer that covers this area."
        )

    @staticmethod
    def _level_floor(mupp: float, step: int) -> float:





        return 0.0 if step <= 0 else float(mupp) * (2 ** step)

    @staticmethod
    def _grid_mupp(grid) -> float:

        try:
            minx, _miny, maxx, _maxy = (grid or {})["bbox"]
            pixel_w = int((grid or {})["pixel_w"])
        except (KeyError, TypeError, ValueError):
            return 0.0
        if pixel_w <= 0:
            return 0.0
        width = float(maxx) - float(minx)
        return width / pixel_w if width > 0 else 0.0

    @staticmethod
    def _probe_render_crs(grid):


        from qgis.core import QgsCoordinateReferenceSystem

        authid = (grid or {}).get("crs") or ""
        if not authid:
            return None
        crs = QgsCoordinateReferenceSystem(authid)
        return crs if crs.isValid() else None

    def _probe_centres(self, layer, grid) -> list:










        from qgis.core import QgsGeometry, QgsRectangle

        try:
            minx, miny, maxx, maxy = (grid or {})["bbox"]
        except (KeyError, TypeError, ValueError):
            return []
        centre = ((float(minx) + float(maxx)) / 2.0,
                  (float(miny) + float(maxy)) / 2.0)
        try:
            poly = self._polygon_in_run_crs(layer)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            poly = None
        if poly is None or poly.isEmpty():
            return [centre]
        points: list = []
        bb = poly.boundingBox()
        step = bb.width() / 3.0
        for i in range(3):
            band = QgsRectangle(bb.xMinimum() + i * step, bb.yMinimum(),
                                bb.xMinimum() + (i + 1) * step, bb.yMaximum())
            try:
                part = poly.intersection(QgsGeometry.fromRect(band))
                if part is None or part.isEmpty():
                    continue
                on_surface = part.pointOnSurface()
                if on_surface is None or on_surface.isEmpty():
                    continue
                point = on_surface.asPoint()
                points.append((point.x(), point.y()))
            except (RuntimeError, ValueError, TypeError):
                continue
        return points or [centre]

    @staticmethod
    def _imagery_probe_extent(grid, centre=None):










        from qgis.core import QgsRectangle

        try:
            minx, miny, maxx, maxy = (grid or {})["bbox"]
            pixel_w = int((grid or {})["pixel_w"])
            pixel_h = int((grid or {})["pixel_h"])
        except (KeyError, TypeError, ValueError):
            return None
        width = float(maxx) - float(minx)
        height = float(maxy) - float(miny)
        if width <= 0 or height <= 0 or pixel_w <= 0 or pixel_h <= 0:
            return None





        from ...core.shape_policy_dials import imagery_probe_px
        side_px = imagery_probe_px(_PROBE_SIDE_PX)
        half_w = min(width, side_px * width / pixel_w) / 2.0
        half_h = min(height, side_px * height / pixel_h) / 2.0
        if centre is not None:
            cx, cy = float(centre[0]), float(centre[1])
        else:
            cx = (float(minx) + float(maxx)) / 2.0
            cy = (float(miny) + float(maxy)) / 2.0
        return QgsRectangle(cx - half_w, cy - half_h, cx + half_w, cy + half_h)
