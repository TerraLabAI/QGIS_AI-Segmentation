
























from __future__ import annotations




_REACH_WATCH_MS = 400
_REACH_WATCH_TICKS = 25

_REACH_HELD_ZONES = 64

_REACH_PENDING = "pending"


class AutoGridReachMixin:


    def _grid_reach_native(self, layer, zone_in_layer, native: float) -> float:



        try:
            from ...core.server_dials import dial_in_range, feature_switch

            if not feature_switch("features.auto_grid_reach", True):
                return native
            zoom = self._grid_reach_zoom(layer, zone_in_layer)
            from ...core.online_zoom_reach import reach_scaled_native


            max_extra = int(dial_in_range(
                "tuning.auto.grid_reach_max_levels", 3, 0, 4))
            return reach_scaled_native(layer, native, zoom, max_extra)
        except Exception:  # noqa: BLE001
            return native

    def _grid_reach_zoom(self, layer, zone_in_layer) -> int | None:


        key = self._grid_reach_key(layer, zone_in_layer)
        if key is None:
            return None
        held = getattr(self, "_auto_grid_reach", None)
        if held is None:
            held = self._auto_grid_reach = {}
        fresh = key not in held
        if fresh or (held[key] == _REACH_PENDING and self._grid_reach_may_move()):
            if fresh and len(held) >= _REACH_HELD_ZONES:
                held.clear()
            point = self._grid_reach_point(layer, zone_in_layer)
            if point is None:
                held[key] = None
                return None
            from ...core.online_zoom_reach import reach_answer_near

            settled, zoom = reach_answer_near(layer, point)
            held[key] = zoom if settled else _REACH_PENDING
            if fresh and not settled:
                self._watch_grid_reach(key, layer, point)
            elif not fresh and settled:

                self._max_detail_cache = None
                self._tile_plan_cache = None
        zoom = held.get(key)
        return None if zoom == _REACH_PENDING else zoom

    def _grid_reach_levels(self, layer, zone_in_layer) -> int | None:



        try:
            from ...core.server_dials import dial_in_range, feature_switch
            from ...core.xyz_tile_fetch import _parse_layer_source

            parsed = _parse_layer_source(layer)
            if parsed is None:
                return None
            if not feature_switch("features.auto_grid_reach", True):
                return 0
            held = getattr(self, "_auto_grid_reach", None) or {}
            zoom = held.get(self._grid_reach_key(layer, zone_in_layer))
            if not isinstance(zoom, int):
                return 0
            max_extra = int(dial_in_range(
                "tuning.auto.grid_reach_max_levels", 3, 0, 4))
            return max(0, min(zoom - int(parsed[2]), max_extra))
        except Exception:  # noqa: BLE001
            return None

    def _await_grid_reach(self, layer) -> None:



        try:
            zone = getattr(self, "_auto_zone", None)
            if layer is None or zone is None:
                return
            zone_in_layer = self._reproject_zone_to_run_crs(zone, layer)
            self._grid_reach_zoom(layer, zone_in_layer)
            key = self._grid_reach_key(layer, zone_in_layer)
            held = getattr(self, "_auto_grid_reach", None) or {}
            if key is None or held.get(key) != _REACH_PENDING:
                return
            point = self._grid_reach_point(layer, zone_in_layer)
            if point is None:
                return
            from qgis.PyQt.QtCore import QEventLoop, QTimer

            from ...core.online_zoom_reach import reach_known_near
            from ...core.server_dials import dial_in_range

            budget_s = float(dial_in_range(
                "network.xyz.reach_probe_budget_s", 6.0, 1, 30)) + 1.0
            source = layer.source()
            loop = QEventLoop()
            timer = QTimer()
            ticks = {"left": int(budget_s * 1000 / 100)}

            def _tick():
                ticks["left"] -= 1
                if reach_known_near(source, point)[0] or ticks["left"] <= 0:
                    loop.quit()

            timer.timeout.connect(_tick)
            timer.start(100)
            try:
                loop.exec()
            finally:
                timer.stop()
            self._grid_reach_zoom(layer, zone_in_layer)
        except Exception:  # noqa: BLE001
            return

    @staticmethod
    def _grid_reach_key(layer, zone_in_layer):
        try:
            return (layer.id(), layer.source(),
                    round(zone_in_layer.xMinimum(), 3),
                    round(zone_in_layer.yMinimum(), 3),
                    round(zone_in_layer.xMaximum(), 3),
                    round(zone_in_layer.yMaximum(), 3))
        except (RuntimeError, AttributeError):
            return None

    def _grid_reach_point(self, layer, zone_in_layer):

        from qgis.core import (
            QgsCoordinateReferenceSystem,
            QgsCoordinateTransform,
            QgsPointXY,
            QgsProject,
        )

        crs = self._run_crs_now(layer) or layer.crs()
        if crs is None or not crs.isValid():
            return None
        centre = QgsPointXY(zone_in_layer.center())
        mercator = QgsCoordinateReferenceSystem("EPSG:3857")
        if crs.authid() != "EPSG:3857":
            centre = QgsCoordinateTransform(
                crs, mercator, QgsProject.instance()).transform(centre)
        return (centre.x(), centre.y())

    def _watch_grid_reach(self, key, layer, point) -> None:


        if self.dock_widget is None:
            return
        from qgis.PyQt.QtCore import QTimer

        old = getattr(self, "_auto_grid_reach_timer", None)
        if old is not None:
            try:
                old.stop()
                old.deleteLater()
            except RuntimeError:
                pass
        try:
            source = layer.source()
        except (RuntimeError, AttributeError):
            return

        from ...core.server_dials import dial_in_range

        timer = QTimer(self.dock_widget)
        timer.setInterval(int(dial_in_range(
            "tuning.auto.grid_reach_watch_ms", _REACH_WATCH_MS, 100, 5000)))
        self._auto_grid_reach_timer = timer
        self._auto_grid_reach_watch = {
            "key": key, "source": source, "point": point, "ticks": 0}
        timer.timeout.connect(self._on_grid_reach_tick)
        timer.start()

    def _on_grid_reach_tick(self) -> None:

        timer = getattr(self, "_auto_grid_reach_timer", None)
        watch = getattr(self, "_auto_grid_reach_watch", None)
        try:
            if timer is None or watch is None:
                return
            watch["ticks"] += 1
            from ...core.online_zoom_reach import reach_known_near

            known, zoom = reach_known_near(watch["source"], watch["point"])
            from ...core.server_dials import dial_in_range

            last_tick = int(dial_in_range(
                "tuning.auto.grid_reach_watch_ticks", _REACH_WATCH_TICKS, 1, 200))
            if not known and watch["ticks"] < last_tick:
                return
            timer.stop()
            self._auto_grid_reach_timer = None
            self._auto_grid_reach_watch = None
            held = getattr(self, "_auto_grid_reach", None)
            if (not known or zoom is None or held is None
                    or watch["key"] not in held
                    or not self._grid_reach_may_move()
                    or getattr(self, "_auto_headless_run", False)
                    or getattr(self, "_auto_zone", None) is None):
                return
            if held[watch["key"]] == _REACH_PENDING:
                held[watch["key"]] = zoom
                self._max_detail_cache = None
                self._tile_plan_cache = None


            self._update_credit_estimate()
        except Exception:  # noqa: BLE001
            self._auto_grid_reach_timer = None
            self._auto_grid_reach_watch = None

    def _grid_reach_may_move(self) -> bool:


        return (getattr(self, "_auto_worker", None) is None
                and getattr(self, "_auto_review", None) is None
                and getattr(self, "_auto_imagery_probe", None) is None)
