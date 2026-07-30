










from __future__ import annotations

import functools
import re
import threading

from . import telemetry_events as ev
from .telemetry import current_session_id, on_main_thread, scrub_payload_value, track







_PATH_PREFIX_RE = re.compile(
    r"(?:<USER>|~(?=[\\/])|(?<![\w.])[A-Za-z]:(?=[\\/])|\\\\|(?<![\w.:)\]])/)"
    r"[^\r\n'\"<>,;()=]*[\\/]"
)



_reported_lock = threading.Lock()
_reported_tracebacks: set[str] = set()
_reported = {"session": ""}


def keep_path_basenames(text: str) -> str:






    if not text:
        return text
    return _PATH_PREFIX_RE.sub("", text)


def _traceback_already_reported(traceback_hash: str) -> bool:




    if not traceback_hash:
        return False
    try:
        session = current_session_id()
    except Exception:  # noqa: BLE001
        session = ""
    with _reported_lock:
        if session != _reported["session"]:
            _reported["session"] = session
            _reported_tracebacks.clear()
        if traceback_hash in _reported_tracebacks:
            return True
        _reported_tracebacks.add(traceback_hash)
    return False


def track_plugin_error(
    stage: str,
    error_code: str,
    message: str,
    include_log_tail: bool = False,
    traceback_hash: str | None = None,
    module: str | None = None,
) -> None:
















    if traceback_hash and _traceback_already_reported(traceback_hash):
        return
    props = {
        "stage": stage,
        "error_code": error_code,
        "message": keep_path_basenames(scrub_payload_value(message or ""))[:500],
    }
    if traceback_hash:
        props["traceback_hash"] = traceback_hash
    if module:
        props["module"] = module
    if include_log_tail:
        try:
            from .log_scrub import get_recent_logs
            tail_lines = get_recent_logs().splitlines()[-20:]
            scrubbed = keep_path_basenames(scrub_payload_value("\n".join(tail_lines)))
            props["last_log_lines"] = scrubbed.encode("utf-8")[:4096].decode(
                "utf-8", errors="ignore"
            )
        except Exception:
            pass  # nosec B110
    track(ev.PLUGIN_ERROR, props)


    try:
        from .run_log_capture import send_run_log_on_error
        send_run_log_on_error(error_code)
    except Exception:  # noqa: BLE001
        pass  # nosec B110













def _short_traceback_hash(exc: BaseException) -> str:





    import hashlib
    import os as _os
    import traceback as _tb
    try:
        parts = [
            f"{_os.path.basename(fr.filename)}:{fr.lineno}:{fr.name}"
            for fr in _tb.extract_tb(exc.__traceback__)
        ]
        parts.append(exc.__class__.__name__)
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""


def report_exception(
    exc: BaseException,
    stage: str,
    module: str = "",
    user_message: str | None = None,
    parent=None,
) -> None:







    error_code = ""
    tb_hash = ""
    first_line = ""
    try:
        error_code = exc.__class__.__name__
        tb_hash = _short_traceback_hash(exc)
        text = str(exc)
        first_line = text.splitlines()[0] if text else ""
    except Exception:  # nosec B110
        pass
    try:
        track_plugin_error(
            stage=stage,
            error_code=error_code or "Exception",
            message=first_line,
            traceback_hash=tb_hash,
            module=module or None,
        )
    except Exception:  # nosec B110
        pass
    try:
        from qgis.core import Qgis, QgsMessageLog
        QgsMessageLog.logMessage(
            "Unhandled {code} in {mod} ({stage}) [{h}]".format(
                code=error_code or "Exception", mod=module or "?",
                stage=stage, h=tb_hash or "-"),
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
    except Exception:  # nosec B110
        pass
    if user_message and on_main_thread():
        try:
            from ..ui.error_report_dialog import ErrorReportDialog
            dialog = ErrorReportDialog(user_message, user_message, parent)
            dialog.exec()
        except Exception:  # nosec B110
            pass


def slot_guard(stage: str, user_message: str | None = None):




    def deco(fn):
        module = (fn.__module__ or "").rsplit(".", 1)[-1]






        try:
            import inspect
            params = list(inspect.signature(fn).parameters.values())[1:]
            _has_var = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params)
            _max_pos = None if _has_var else sum(
                1 for p in params
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD)
            )
        except Exception:  # noqa: BLE001
            _max_pos = None

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if _max_pos is not None and len(args) > _max_pos:
                args = args[:_max_pos]
            try:
                return fn(self, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                parent = None
                if user_message is not None:
                    try:
                        parent = self.iface.mainWindow()
                    except Exception:  # nosec B110
                        parent = None
                report_exception(
                    exc, stage=stage, module=module,
                    user_message=user_message, parent=parent,
                )
                return None
        return wrapper
    return deco






