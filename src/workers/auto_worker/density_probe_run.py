
















from __future__ import annotations

from collections import deque

from ...core import density_probe as _dp



STOP_REASON_REPLAN = "replan"


class AutoDensityProbeMixin:


    def _density_setup(self, density_probe) -> None:

        self._density_plan = None
        self._density_probe_set: frozenset = frozenset()
        self._density_counts: dict[int, int] = {}
        self._density_decision = None
        if not isinstance(density_probe, dict):
            return
        indices = density_probe.get("indices")
        config = density_probe.get("config")
        family = density_probe.get("family")
        side_m = density_probe.get("side_m")
        if (not isinstance(indices, (list, tuple)) or not indices
                or not isinstance(config, _dp.DensityProbeConfig)
                or not isinstance(family, _dp.DensityFamily)
                or not isinstance(side_m, (int, float)) or side_m <= 0):
            return
        n = len(self._tiles)
        kept = tuple(i for i in indices
                     if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < n)
        if len(kept) < config.probe_tiles_min:
            return
        self._density_plan = {
            "indices": kept, "config": config, "family": family,
            "side_m": float(side_m)}
        self._density_probe_set = frozenset(kept)

    def _density_pending_order(self) -> deque:


        if self._density_plan is None:
            return deque(enumerate(self._tiles))
        first = [(i, self._tiles[i]) for i in self._density_plan["indices"]]
        rest = [(i, t) for i, t in enumerate(self._tiles)
                if i not in self._density_probe_set]
        return deque(first + rest)

    def _density_note_count(self, tile_idx: int, count: int) -> None:


        if (self._density_plan is None or self._density_decision is not None
                or tile_idx not in self._density_probe_set):
            return
        if self._tile_depth.get(tile_idx, 0) != 0:
            return
        self._density_counts[tile_idx] = int(count)

    def _density_check(self, pending, resubmit, in_flight) -> None:



        if self._density_plan is None or self._density_decision is not None:
            return
        left = self._density_probe_set.difference(self._density_counts)
        if left:
            live = {entry[0] for entry in pending}
            live.update(entry[0] for entry in resubmit)
            live.update(entry[0] for entry in in_flight.values())
            live.update(entry[1] for entry in self._render_deferred)
            if left & live:
                return
        plan = self._density_plan
        counts = [self._density_counts[i] for i in plan["indices"]
                  if i in self._density_counts]
        decision = _dp.decide(counts, plan["side_m"], plan["family"], plan["config"])
        self._density_decision = decision
        try:
            from qgis.core import Qgis, QgsMessageLog

            QgsMessageLog.logMessage(
                f"Auto detection: density probe {decision.branch} "
                f"({decision.reason}), {decision.k} tiles, median "
                f"{decision.median:.0f}, p85 {decision.p85:.0f}, max "
                f"{decision.max:.0f}, tile {plan['side_m']:.0f} m -> "
                f"{decision.side_m:.0f} m",
                "AI Segmentation", level=Qgis.MessageLevel.Info)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        if decision.branch in (_dp.BRANCH_COARSEN, _dp.BRANCH_REFINE):
            if self._stop_reason is None and not self._stop_requested:
                self._stop_reason = STOP_REASON_REPLAN
                self._stop_requested = True

    def _density_replan_asked(self) -> bool:

        return (self._stop_reason == STOP_REASON_REPLAN
                and self._density_decision is not None)

    def _density_profile(self) -> dict:


        decision = self._density_decision
        if decision is None:
            if self._density_plan is not None:
                return {"density_branch": _dp.BRANCH_STAY,
                        "density_reason": "undecided"}
            return {}
        props = decision.props()
        props["density_from_m"] = int(round(self._density_plan["side_m"]))
        return props
