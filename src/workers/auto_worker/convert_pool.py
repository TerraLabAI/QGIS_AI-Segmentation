









from __future__ import annotations

import logging
import time

from qgis.core import Qgis

from ...core import transport_dials as _td
from ...core.server_dials import dial_in_range
from ..tile_convert_pool import PROCESS_MAX_WORKERS as _PROCESS_DEFAULT_MAX
from ..tile_convert_pool import (
    PROCESS_POOL_MIN_TILES,
    TileConvertPool,
    TileConvertProcessPool,
    process_workers,
    usable_cores,
)
from ..tile_convert_pool import SPARE_CORES as _CONVERT_SPARE_CORES

__all__ = [
    "AutoConvertPoolMixin",
    "_CONVERT_BACKLOG_PER_WORKER",
    "_CONVERT_DRAIN_BUDGET_S",
    "_CONVERT_RESCUE_PROBES",
    "_CONVERT_WORKERS",
    "_CONVERT_WORKERS_CEILING",
    "_GEOS_THREAD_LOCAL_MIN_VERSION",
    "_convert_failure_reason",
    "_resolve_convert_workers",
    "logger",
]

logger = logging.getLogger(__name__)




_CONVERT_WORKERS = 0





_CONVERT_WORKERS_CEILING = 8





_GEOS_THREAD_LOCAL_MIN_VERSION = 33800






_CONVERT_BACKLOG_PER_WORKER = 4





_CONVERT_DRAIN_BUDGET_S = 90.0






_CONVERT_RESCUE_PROBES = 8


def _convert_failure_reason(exc) -> str:








    try:
        name = type(exc).__name__
    except Exception:  # noqa: BLE001
        return "unknown"
    try:
        text = str(exc).strip().splitlines()[0][:160]
    except Exception:  # noqa: BLE001
        text = ""
    if not text:
        return name
    try:
        from ...core.run_log_capture import redact_line
        text = redact_line(text)
    except Exception:  # noqa: BLE001
        return name
    return f"{name}: {text}"


def _resolve_convert_workers(
    served: int, qgis_version_int: int, ceiling: int = _CONVERT_WORKERS_CEILING
) -> int:




    workers = min(max(1, ceiling), max(0, served))
    if qgis_version_int < _GEOS_THREAD_LOCAL_MIN_VERSION:
        return 1
    return workers


class AutoConvertPoolMixin:


    def _settle_converted_batch(self, items) -> None:





        for ok, job, payload in items:
            self._settle_converted(ok, job, payload)

    def _open_convert_pool(self, workers: int, while_booting=None):


















        children = process_workers(
            default_max=_td.convert_pool_default_max(_PROCESS_DEFAULT_MAX),
            spare_cores=_td.convert_pool_spare_cores(_CONVERT_SPARE_CORES))


        pool = self._take_prespawned_children(children)
        if len(self._tiles) < dial_in_range(
                "tuning.convert.process_pool_min_tiles", PROCESS_POOL_MIN_TILES, 1, 500):
            self._drop_early_children(pool)
            self._log_convert_pool("threads", "run too short for children")
            return TileConvertPool(self._convert_completed, workers=workers)
        snapshot = self._convert_snapshot()
        if snapshot is None:
            self._drop_early_children(pool)
            self._log_convert_pool("threads", "the run would not snapshot")
            return TileConvertPool(self._convert_completed, workers=workers)
        if children < 2:



            self._drop_early_children(pool)
            self._log_convert_pool("threads", f"{usable_cores()} usable core(s)")
            return TileConvertPool(self._convert_completed, workers=workers)
        if pool is not None:
            pool.set_snapshot(snapshot)
        else:
            pool = TileConvertProcessPool(
                self._convert_completed, snapshot, workers=children)
        reason = "the children would not come up"
        try:
            if pool.start(while_booting=while_booting):
                self._convert_pool_is_processes = True
                self._log_convert_pool("processes", "", pool.workers)
                return pool




            detail = str(getattr(pool, "last_failure", "") or "")
            if detail:
                reason = f"{reason}: {detail}"
        except Exception as exc:  # noqa: BLE001
            reason = f"spawn error {type(exc).__name__}"
            logger.info("AutoDetectionWorker: converter processes unavailable",
                        exc_info=True)
            try:
                pool.close(wait=False)
            except Exception:  # noqa: BLE001  # nosec B110
                pass
        self._log_convert_pool("threads", reason)
        return TileConvertPool(self._convert_completed, workers=workers)

    def _prespawn_convert_children(self) -> None:







        import sys

        if sys.platform != "win32" or self._convert_prespawned is not None:
            return
        if len(self._tiles) < dial_in_range(
                "tuning.convert.process_pool_min_tiles", PROCESS_POOL_MIN_TILES, 1, 500):
            return
        try:
            children = process_workers(
                default_max=_td.convert_pool_default_max(_PROCESS_DEFAULT_MAX),
                spare_cores=_td.convert_pool_spare_cores(_CONVERT_SPARE_CORES))
            if children < 2:
                return
            pool = TileConvertProcessPool(
                self._convert_completed, None, workers=children)
            if pool.spawn():
                self._convert_prespawned = pool
            else:
                pool.close(wait=False)
        except Exception:  # noqa: BLE001
            logger.info("AutoDetectionWorker: early converter launch failed",
                        exc_info=True)

    def _take_prespawned_children(self, children: int):


        pool, self._convert_prespawned = self._convert_prespawned, None
        if pool is not None and pool.workers != children:
            self._drop_early_children(pool)
            return None
        return pool

    def _drop_prespawned_children(self) -> None:

        pool, self._convert_prespawned = self._convert_prespawned, None
        self._drop_early_children(pool)

    @staticmethod
    def _drop_early_children(pool) -> None:
        if pool is None:
            return
        try:
            pool.close(wait=False)
        except Exception:  # noqa: BLE001
            pass  # nosec B110

    def _report_convert_failures(self) -> None:












        try:
            from ...core.telemetry_errors import track_plugin_error
            reason = self._convert_fail_reason or "no reason recorded"



            if self._convert_children_broken:
                track_plugin_error(
                    stage="segment",
                    error_code="convert_pool_children_dead",
                    message=(f"child converters answered nothing usable; "
                             f"{self._convert_rescued_tiles} tile(s) converted "
                             f"in QGIS instead: {reason}"),
                )
                return
            failed = int(self.tiles_convert_failed)
            answered = int(self.tiles_succeeded) + failed
            if failed <= 0 or answered <= 0 or failed * 2 < answered:
                return
            track_plugin_error(
                stage="segment",
                error_code="convert_failed_tiles",
                message=(f"{failed} of {answered} answered tile(s) failed to "
                         f"convert on {self._convert_pool_kind or 'unknown'}: "
                         f"{reason}"),
            )
        except Exception:  # noqa: BLE001
            pass  # nosec B110

    def _log_convert_pool(self, kind: str, reason: str, workers: int = 0) -> None:






        self._convert_pool_kind = kind
        self._convert_pool_workers = int(workers or 0)


        try:
            from ...core.run_log_capture import redact_line
            reason = redact_line(reason)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        self._convert_fallback_reason = reason if kind != "processes" else ""



        if kind != "processes" and reason.startswith(("the children", "spawn error")):
            try:
                from ...core.telemetry_errors import track_plugin_error
                track_plugin_error(stage="segment",
                                   error_code="convert_pool_fallback",
                                   message=reason)
            except Exception:  # noqa: BLE001  # nosec B110
                pass
        try:
            from qgis.core import QgsMessageLog

            tail = f" ({reason})" if reason else ""
            QgsMessageLog.logMessage(
                f"Auto detection: converting on {kind}"
                + (f", {workers} of them" if workers else "") + tail,
                "AI Segmentation", level=Qgis.MessageLevel.Info)
        except Exception:  # noqa: BLE001
            pass  # nosec B110

    def _convert_snapshot(self) -> dict | None:





















        import pickle  # nosec B403



        package = __name__.split(".")[0]
        unreadable = package.encode() if package != "src" else b""
        try:
            out = {}
            for name, value in list(self.__dict__.items()):
                try:
                    blob = pickle.dumps(value, protocol=4)
                except Exception:  # noqa: BLE001  # nosec B112
                    continue
                if unreadable and unreadable in blob:
                    continue
                out[name] = value


            if "_score_threshold" not in out or "_gsd" not in out:
                return None
            return out
        except Exception:  # noqa: BLE001
            logger.info("AutoDetectionWorker: could not snapshot the run",
                        exc_info=True)
            return None

    def _close_convert_pool(self, budget_s: float, emit: bool = True) -> None:




















        pool, self._convert_pool = self._convert_pool, None
        if pool is None:
            return
        started = time.monotonic()
        owed = int(pool.pending or 0)
        deadline = started + max(0.0, budget_s)
        stop_seen = self._stop_requested
        try:
            while pool.pending and time.monotonic() < deadline:
                items = pool.drain(timeout=0.25)
                if emit:
                    self._settle_converted_batch(items)
                if self._stop_requested and not stop_seen:
                    stop_seen = True
                    deadline = min(
                        deadline, time.monotonic() + self._stop_drain_budget_s)
        finally:




            leftover = pool.close(wait=False)
        if emit:
            self._settle_converted_batch(leftover)



        fold = getattr(pool, "fold_stats", None)
        if fold is not None:
            try:
                fold(self)
            except Exception:  # noqa: BLE001
                logger.warning("AutoDetectionWorker: converter counters were "
                               "not folded back", exc_info=True)





        spent = time.monotonic() - started
        if owed and spent >= 1.0:
            logger.info(
                "AutoDetectionWorker: converted the last %d tile(s) in %.1fs "
                "after the final reply", owed, spent,
            )

    def _retry_convert_in_process(self, job: dict):














        if self._convert_pool_kind != "processes":
            return None
        if not self._convert_children_broken:
            if self._convert_rescue_probes <= 0:
                return None
            self._convert_rescue_probes -= 1
        try:
            detections = self._convert_completed(job)
        except Exception:  # noqa: BLE001
            return None
        self._convert_rescued_tiles += 1
        if not self._convert_children_broken:
            self._convert_children_broken = True
            try:
                from qgis.core import QgsMessageLog
                QgsMessageLog.logMessage(
                    "Auto detection: the converter children answer nothing "
                    "usable; converting in QGIS instead for the rest of this "
                    f"run ({self._convert_fail_reason or 'no reason recorded'})",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
            except Exception:  # noqa: BLE001
                pass  # nosec B110
        return detections
