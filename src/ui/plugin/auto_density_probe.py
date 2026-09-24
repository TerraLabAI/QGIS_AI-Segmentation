


















from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog

from ...core.tile_manager import TILE_SIZE


def _density_log(message: str) -> None:
    QgsMessageLog.logMessage(
        f"Auto detection: {message}", "AI Segmentation",
        level=Qgis.MessageLevel.Info)


class AutoDensityProbeMixin:


    def _density_forced_side_m(self) -> float:

        forced = getattr(self, "_auto_density_forced", None)
        if not isinstance(forced, dict):
            return 0.0
        decision = forced.get("decision")
        side = getattr(decision, "side_m", 0.0)
        return float(side) if isinstance(side, (int, float)) and side > 0 else 0.0

    def _density_clear_forced(self) -> None:

        self._auto_density_forced = None

    def _density_probe_plan(self, layer, zone_in_layer, prompt: str,
                            tiles: list, has_exemplars: bool) -> dict | None:



        self._auto_density_props = {}
        forced = getattr(self, "_auto_density_forced", None)
        if isinstance(forced, dict):
            decision = forced.get("decision")
            try:
                props = decision.props()
            except AttributeError:
                props = {}
            props["density_from_m"] = int(round(forced.get("from_m", 0.0) or 0.0))
            self._auto_density_props = props
            return None
        try:
            from ...core import density_probe as dp
            from ...core.detection_policy import density_probe_config
            config = density_probe_config()
        except Exception:  # noqa: BLE001
            return None
        if config is None:
            return None

        def skip(reason: str) -> None:
            self._auto_density_props = {
                "density_branch": "skip", "density_reason": reason}

        if getattr(self, "_auto_headless_run", False):
            return skip(dp.SKIP_HEADLESS)
        if has_exemplars:
            return skip(dp.SKIP_EXEMPLAR)
        if (not self._tile_plan_active()
                or self._get_auto_detail_level() != self._tile_plan_centre_step()):
            return skip(dp.SKIP_USER_STEP)
        family = dp.family_for(prompt, config)
        if family is None:
            try:
                family = dp.family_for(self._resolved_auto_object_class(), config)
            except (RuntimeError, AttributeError):
                family = None
        if family is None:
            return skip(dp.SKIP_FAMILY)
        n = len(tiles)
        if n < config.min_zone_tiles:
            return skip(dp.SKIP_ZONE_SMALL)
        if n > config.max_zone_tiles:
            return skip(dp.SKIP_ZONE_LARGE)
        side_m = TILE_SIZE * float(getattr(self, "_auto_gsd_m", 0.0) or 0.0)
        if side_m <= 0:
            return skip(dp.SKIP_OFF)
        family = self._density_family_for_run(
            layer, zone_in_layer, family, config, side_m)
        k = dp.probe_k(n, config)
        centres = [(x + w / 2.0, y + h / 2.0) for (x, y, w, h) in tiles]
        order = dp.probe_order(centres, k)
        if len(order) < config.probe_tiles_min:
            return skip(dp.SKIP_ZONE_SMALL)
        self._auto_density_props = {
            "density_branch": "stay", "density_reason": "undecided"}
        _density_log(
            f"density probe armed, {len(order)} of {n} tiles first, tile "
            f"{side_m:.0f} m, family {family.name}"
            + ("" if family.refine_ground_m > 0 else ", refine off"))
        return {"indices": order, "config": config, "family": family,
                "side_m": side_m}

    def _density_family_for_run(self, layer, zone_in_layer, family, config,
                                side_m: float):





        from dataclasses import replace

        if family.refine_ground_m <= 0 or config.refine_p85_min <= 0:
            return family
        try:
            resolved = self._tile_plan_now(
                layer, zone_in_layer, density_side_m=family.refine_ground_m)
        except (RuntimeError, AttributeError, TypeError, ValueError,
                ZeroDivisionError):
            resolved = None
        ok = False
        if resolved is not None:
            plan = resolved[0]
            limit = config.refine_max_tiles
            try:
                _credits, is_free = self._auto_credit_snapshot()
            except Exception:  # noqa: BLE001
                is_free = False
            if is_free and config.refine_max_tiles_free > 0:
                limit = config.refine_max_tiles_free
            finer = plan.tile_ground_m < side_m * (1.0 - config.min_change_ratio)
            fits = limit <= 0 or 0 <= plan.tiles <= limit
            ok = finer and fits
            if not ok:
                _density_log(
                    f"density probe refine off for this zone (refined tile "
                    f"{plan.tile_ground_m:.0f} m, {plan.tiles} tiles, bound {limit})")
        return family if ok else replace(family, refine_ground_m=0.0)

    def _on_auto_density_replan(self, decision, worker=None) -> None:


        if worker is not None and worker is not self._auto_worker:
            return
        worker = self._auto_worker
        if worker is None:
            return
        if getattr(worker, "_stop_reason", None) != "replan":


            self._on_auto_cancelled(worker=worker)
            return
        from .auto_run_progress import _WIND_DOWN_DETACH
        from .shared import park_orphaned_worker

        from_m = TILE_SIZE * float(getattr(self, "_auto_gsd_m", 0.0) or 0.0)
        self._stop_auto_stall_watchdog()
        self._pop_nothing_found_notice()
        self._cancel_active_tile_render()
        for sig_name, slot_name in _WIND_DOWN_DETACH:
            try:
                getattr(worker, sig_name).disconnect(getattr(self, slot_name))
            except (TypeError, RuntimeError, AttributeError):
                pass
        for sig_name, slot in (("cancelled", getattr(self, "_auto_cancelled_slot", None)),
                               ("density_replan", getattr(self, "_auto_replan_slot", None))):
            if slot is None:
                continue
            try:
                getattr(worker, sig_name).disconnect(slot)
            except (TypeError, RuntimeError, AttributeError):
                pass
        if worker.isRunning():
            park_orphaned_worker(worker)
        self._auto_worker = None
        self._auto_cancelled_slot = None
        self._auto_replan_slot = None
        self._drop_auto_tile_bridge()
        self._auto_merger = None
        self._reset_auto_live_pipeline()
        self._remove_auto_selection_layer()
        self._auto_density_forced = {
            "decision": decision,
            "run_id": self._auto_run_id,
            "from_m": from_m,
            "click": getattr(self, "_auto_run_started_mono", None),
        }
        _density_log(
            f"density probe {decision.branch}: restarting the run at "
            f"{decision.side_m:.0f} m (was {from_m:.0f} m), same run id")
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(0, self._density_restart_run)

    def _density_restart_run(self) -> None:


        forced = getattr(self, "_auto_density_forced", None)
        if not isinstance(forced, dict):
            return
        if self.dock_widget is None or self._auto_worker is not None:
            self._density_clear_forced()
            return

        self._auto_click_mono = forced.get("click")
        self._start_auto_detection()

    def _density_after_start(self) -> None:


        if getattr(self, "_auto_density_forced", None) is None:
            return
        if getattr(self, "_auto_imagery_probe", None) is not None:
            return
        self._density_clear_forced()
        if self._auto_worker is None:


            _density_log("density probe restart did not start; the first pass "
                         "is dropped")
            try:
                self.dock_widget.set_auto_run_active(False)
                self._set_zone_badge_enabled(True)
            except (RuntimeError, AttributeError):
                pass
