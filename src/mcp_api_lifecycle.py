"""Model and account state for the public API. Read only, plus one loader.

Part of `SegmentationMCPAPI` (see `mcp_api.py`), split out so one concern sits
in one file.

Deliberately missing, and never to be added: anything that installs software,
downloads a model, signs the user in or writes an activation key. Those are the
user's decisions on the user's machine, and an outside agent must not make them
on their behalf. `get_status()` and `install_status()` say what is missing and
what the person has to click. That is the whole boundary.
"""
from __future__ import annotations

import contextlib
import time

# The longest a caller may ask load_model to wait, so a mistyped argument
# cannot park a QGIS session for an hour.
_LOAD_TIMEOUT_MAX_S = 600.0
_LOAD_TIMEOUT_DEFAULT_S = 180.0


# How long one poll step waits for the loader to answer. Short enough that a
# cancelled or finished load is noticed almost at once.
_POLL_STEP_S = 0.05


def _caller_is_on_the_gui_thread() -> bool:
    """True when this call runs on the thread QGIS builds its widgets on.

    Anything else is read as a worker thread, where pumping events is unsafe.
    """
    try:
        from qgis.core import QgsApplication
        from qgis.PyQt.QtCore import QThread

        app = QgsApplication.instance()
        if app is None:
            return False
        return QThread.currentThread() is app.thread()
    except Exception:  # noqa: BLE001 - an unanswered question is read as "not the GUI thread"
        return False


def _pump_events_briefly() -> None:
    """Let the GUI thread deliver the loader's result, for one poll step.

    User input stays excluded: a click on the panel while a blocking call is
    parked here re-enters the plugin from inside its own call.

    The step is spent whatever happens. Pumping returns at once when the queue
    is empty, so a caller looping on this alone would burn a core flat out for
    the whole of its deadline; the loader delivers on a queued signal, which
    the next pass picks up.
    """
    started = time.monotonic()
    # A build that cannot pump still waits out the step below.
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
    """The (default, ceiling) seconds load_model works with right now.

    Resolved at call time, never in a default argument: a default argument is
    evaluated once at import, which would freeze the value at plugin load.
    The default can never sit above the ceiling.
    """
    try:
        from .core.server_dials import dial_in_range

        max_s = float(dial_in_range("agent.load_timeout_max_s", _LOAD_TIMEOUT_MAX_S, 60.0, 7200.0))
        default_s = float(dial_in_range(
            "agent.load_timeout_default_s", _LOAD_TIMEOUT_DEFAULT_S, 1.0, 7200.0))
        return min(default_s, max_s), max_s
    except Exception:  # noqa: BLE001 - the shipped numbers always work
        return _LOAD_TIMEOUT_DEFAULT_S, _LOAD_TIMEOUT_MAX_S


# What a caller does the moment the model answers. Both success paths of
# load_model return it, so a caller that skipped the docs still knows.
_READY_HINT = (
    "The model can answer now: call detect() for the object under one map "
    "point, or detect_points() to grow and cut an outline with several."
)


class SegmentationLifecycleMixin:
    """Say what is installed, and start the on-device model when asked."""

    def install_status(self) -> dict:
        """Report what AI Segmentation has on this computer. Read only.

        Installs nothing, downloads nothing and writes nothing. When a piece is
        missing, ``action_required`` says in one sentence what the person has
        to do in the panel. An outside agent must relay that sentence rather
        than try to do it: this API has no install call and will not get one.

        Returns
        -------
        dict with keys:
            "model_downloaded"  -- bool, the model file is on disk.
            "model_loaded"      -- bool, it is loaded and ready to answer.
            "environment_ready" -- bool, the isolated Python environment exists.
            "account_active"    -- bool, an account is signed in.
            "terms_accepted"    -- bool, the terms have been accepted.
            "install_running"   -- bool, the one-time setup is running now.
            "action_required"   -- str, present only when something is missing.
            "hint"              -- str, present while the setup runs: how
                                   often to call this again.

        Costs nothing.
        """
        plugin = self._plugin
        status: dict = {}

        try:
            from .core.checkpoint_manager import checkpoint_exists
            status["model_downloaded"] = bool(checkpoint_exists())
        except Exception:  # noqa: BLE001 - a missing piece is a False, not a crash
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

        # An install already running is not a missing click. Telling a caller
        # to press Install while it installs gets it pressed twice.
        running = False
        try:
            probe = getattr(plugin, "_local_ai_install_running", None)
            running = bool(probe()) if callable(probe) else False
        except Exception:  # noqa: BLE001 - a probe that fails means "not running"
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
        """Load the on-device model, and return a status rather than hanging.

        The model loads on a background thread. This waits for it, but never
        past ``timeout_s``: it returns ``loaded: False`` with a message saying
        the load may still be running, so a caller gets an answer instead of a
        frozen session. Calling again is safe and free.

        It loads a model that is already on disk. It never downloads one and
        never installs anything: when the model is missing, the answer names
        what the person has to click.

        Parameters
        ----------
        timeout_s : float | None
            Seconds this call may wait, at least 1. None (the default) waits
            the product's current default. A value past the product's ceiling
            is refused, and the refusal names the ceiling. Loading is slow the
            first time in a session and quick afterwards.

        Returns
        -------
        dict with keys:
            "loaded"          -- bool, the model can answer now.
            "already_loaded"  -- bool, it was loaded before this call.
            "waited_s"        -- float, seconds this call actually waited.
            "timeout_s"       -- float, the cap this call was given.
            "state"           -- str, present when a precondition is missing.
            "hint"            -- str, present once loaded: the calls it unlocks.
            "_error"          -- str, present only on failure.

        Costs nothing.
        """
        plugin = self._plugin

        if getattr(plugin, "predictor", None) is not None:
            return {"loaded": True, "already_loaded": True,
                    "waited_s": 0.0, "timeout_s": 0.0,
                    "hint": _READY_HINT}

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

        # An unreadable disk is not a reason to refuse: the check is skipped.
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
        except Exception as err:  # noqa: BLE001 - the API never raises
            return {"_error": f"The model load did not start: {err}"}

        # The cap is not the wait. A load that answers in a third of a
        # second used to report the whole cap back, which reads as a slow
        # plugin to anyone timing it.
        started = time.monotonic()
        self._wait_for_the_predictor(wait_s)

        loaded = getattr(plugin, "predictor", None) is not None
        out = {"loaded": loaded, "already_loaded": False,
               "waited_s": round(time.monotonic() - started, 2),
               "timeout_s": wait_s}
        if loaded:
            out["hint"] = _READY_HINT
        else:
            out["_error"] = (
                "The model did not finish loading in time. It may still be "
                "loading: call load_model() again, or read get_status()."
            )
        return out

    def _wait_for_the_predictor(self, wait_s: float) -> None:
        """Hold until the model answers or the deadline passes. Never longer.

        A running loader is the only thing worth waiting for. The wait used to
        run an event loop that a finished worker never quits, so a second call
        made after the load ended sat there for the whole cap with nothing
        left to signal it.

        Events are pumped only on the GUI thread, since that is where the
        loader delivers its result and where pumping is safe. Called from a
        worker thread, this sleeps in short steps instead: the loader still
        finishes, and this call still notices.
        """
        plugin = self._plugin
        deadline = time.monotonic() + max(0.0, float(wait_s))
        on_gui_thread = _caller_is_on_the_gui_thread()
        while time.monotonic() < deadline:
            if getattr(plugin, "predictor", None) is not None:
                return
            # A loader that is gone or already finished leaves nothing to wait
            # for. The predictor check above is the answer either way.
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
