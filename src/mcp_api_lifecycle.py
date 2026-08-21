










from __future__ import annotations

import contextlib
import time



_LOAD_TIMEOUT_MAX_S = 600.0
_LOAD_TIMEOUT_DEFAULT_S = 180.0




_POLL_STEP_S = 0.05


def _caller_is_on_the_gui_thread() -> bool:




    try:
        from qgis.core import QgsApplication
        from qgis.PyQt.QtCore import QThread

        app = QgsApplication.instance()
        if app is None:
            return False
        return QThread.currentThread() is app.thread()
    except Exception:  # noqa: BLE001
        return False


def _pump_events_briefly() -> None:










    started = time.monotonic()

    with contextlib.suppress(Exception):
        from qgis.PyQt.QtCore import QCoreApplication, QEventLoop

        from .core.qt_compat import resolve_qt_enum

        QCoreApplication.processEvents(
            resolve_qt_enum(QEventLoop, "ProcessEventsFlag", "ExcludeUserInputEvents"),
            int(_POLL_STEP_S * 1000))
    left = _POLL_STEP_S - (time.monotonic() - started)
    if left > 0:
        time.sleep(left)


def _load_timeouts_in_force() -> tuple[float, float]:






    try:
        from .core.server_dials import dial_in_range

        max_s = float(dial_in_range("agent.load_timeout_max_s", _LOAD_TIMEOUT_MAX_S, 60.0, 7200.0))
        default_s = float(dial_in_range(
            "agent.load_timeout_default_s", _LOAD_TIMEOUT_DEFAULT_S, 1.0, 7200.0))
        return min(default_s, max_s), max_s
    except Exception:  # noqa: BLE001
        return _LOAD_TIMEOUT_DEFAULT_S, _LOAD_TIMEOUT_MAX_S




_READY_HINT = (
    "The model can answer now: call detect() for the object under one map "
    "point, or detect_points() to grow and cut an outline with several."
)


def _ready_hint() -> str:

    from .core.server_dials import dial_text

    return dial_text("tuning.agent.hints", "load_model_ready", 400) or _READY_HINT


class SegmentationLifecycleMixin:


    def install_status(self) -> dict:






















        plugin = self._plugin
        status: dict = {}

        try:
            from .core.checkpoint_manager import checkpoint_exists
            status["model_downloaded"] = bool(checkpoint_exists())
        except Exception:  # noqa: BLE001
            status["model_downloaded"] = False

        status["model_loaded"] = getattr(plugin, "predictor", None) is not None

        try:
            from .core.venv_manager import venv_exists
            status["environment_ready"] = bool(venv_exists())
        except Exception:  # noqa: BLE001
            status["environment_ready"] = False

        try:
            from .core.activation_manager import has_tos_accepted, is_plugin_activated
            status["account_active"] = bool(is_plugin_activated())
            status["terms_accepted"] = bool(has_tos_accepted())
        except Exception:  # noqa: BLE001
            status["account_active"] = False
            status["terms_accepted"] = False



        running = False
        try:
            probe = getattr(plugin, "_local_ai_install_running", None)
            running = bool(probe()) if callable(probe) else False
        except Exception:  # noqa: BLE001
            running = False
        status["install_running"] = running

        if not status["account_active"]:
            status["action_required"] = (
                "No account is signed in. Open the AI Segmentation panel and "
                "click Sign in. Only the person at this computer can do that."
            )
        elif running:
            status["action_required"] = (
                "The one-time setup is running. Nobody has to click anything "
                "while it does."
            )
            status["hint"] = (
                "Call install_status() again every 20 to 30 seconds until "
                "install_running is False. A first install takes minutes."
            )
        elif not status["environment_ready"] or not status["model_downloaded"]:
            status["action_required"] = (
                "The one-time setup has not run. Open the AI Segmentation "
                "panel and click Install. Only the person at this computer "
                "can do that."
            )
        elif not status["model_loaded"]:
            status["action_required"] = (
                "The model is installed but not loaded. Call load_model(), or "
                "click 'Start Semi-Auto AI Segmentation' in the panel."
            )
        return status

    def load_model(self, timeout_s: float | None = None) -> dict:
































        plugin = self._plugin

        if getattr(plugin, "predictor", None) is not None:
            return {"loaded": True, "already_loaded": True,
                    "waited_s": 0.0, "timeout_s": 0.0,
                    "hint": _ready_hint()}

        default_s, max_s = _load_timeouts_in_force()
        if timeout_s is None:
            timeout_s = default_s
        try:
            wait_s = float(timeout_s)
        except (TypeError, ValueError):
            return {"_error": f"timeout_s must be a number, got {timeout_s!r}."}
        if not 1.0 <= wait_s <= max_s:
            return {"_error": (
                f"timeout_s must be between 1 and {int(max_s)} "
                f"seconds, got {timeout_s!r}.")}


        with contextlib.suppress(Exception):
            from .core.checkpoint_manager import checkpoint_exists
            if not checkpoint_exists():
                return {
                    "loaded": False,
                    "already_loaded": False,
                    "state": "MODEL_NOT_DOWNLOADED",
                    "_error": (
                        "The model is not on this computer. Open the AI "
                        "Segmentation panel and click Install. This API does "
                        "not install anything."
                    ),
                }

        loader = getattr(plugin, "_load_predictor", None)
        if not callable(loader):
            return {"_error": "This build has no on-device model loader."}
        try:
            loader()
        except Exception as err:  # noqa: BLE001
            return {"_error": f"The model load did not start: {err}"}




        started = time.monotonic()
        self._wait_for_the_predictor(wait_s)

        loaded = getattr(plugin, "predictor", None) is not None
        out = {"loaded": loaded, "already_loaded": False,
               "waited_s": round(time.monotonic() - started, 2),
               "timeout_s": wait_s}
        if loaded:
            out["hint"] = _ready_hint()
        else:
            out["_error"] = (
                "The model did not finish loading in time. It may still be "
                "loading: call load_model() again, or read get_status()."
            )
        return out

    def _wait_for_the_predictor(self, wait_s: float) -> None:












        plugin = self._plugin
        deadline = time.monotonic() + max(0.0, float(wait_s))
        on_gui_thread = _caller_is_on_the_gui_thread()
        while time.monotonic() < deadline:
            if getattr(plugin, "predictor", None) is not None:
                return


            worker = getattr(plugin, "_predictor_worker", None)
            if worker is None:
                return
            running = getattr(worker, "isRunning", None)
            if callable(running):
                try:
                    if not running():
                        return
                except RuntimeError:
                    return
            if on_gui_thread:
                _pump_events_briefly()
            else:
                time.sleep(_POLL_STEP_S)
