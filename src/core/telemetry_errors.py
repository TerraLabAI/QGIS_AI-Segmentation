"""Telemetry for a failure, plus the two boundaries that catch one.

report_exception and slot_guard are the standard capture pattern: an
uncaught exception in a Qt slot or a worker body used to produce no
telemetry at all. There is no sys.excepthook on purpose, it conflicts
with QGIS and other plugins, so the entry points are wrapped instead.

Wrappers only: each one names its event's properties in one place so call sites
stay readable. The transport (consent, batching, scrubbing, the POST) is in
telemetry.py, and every function here ends in its track() call.
"""
from __future__ import annotations

import functools

from . import telemetry_events as ev
from .telemetry import on_main_thread, scrub_payload_value, track


def track_plugin_error(
    stage: str,
    error_code: str,
    message: str,
    include_log_tail: bool = False,
    traceback_hash: str | None = None,
    module: str | None = None,
) -> None:
    """Fire when an error is shown to the user or an exception is caught.

    stage: install | download | activate | segment | export | other
    error_code: short machine-friendly id (e.g. "PIP_TIMEOUT", "RUNTIME_ERROR")
    message: first line of the error, truncated to 500 chars, path + coord scrubbed
    include_log_tail: OFF by default. When True, the last 20 anonymized log lines
        are capped to ~4KB and coordinate-scrubbed before being sent.
    traceback_hash: optional short sha of the normalized traceback (groups
        recurrences of the same crash). Additive; omitted when unknown.
    module: optional source module the exception was caught in. Additive.
    """
    props = {
        "stage": stage,
        "error_code": error_code,
        "message": scrub_payload_value((message or "")[:500]),
    }
    if traceback_hash:
        props["traceback_hash"] = traceback_hash
    if module:
        props["module"] = module
    if include_log_tail:
        try:
            from .log_scrub import get_recent_logs
            tail_lines = get_recent_logs().splitlines()[-20:]
            scrubbed = scrub_payload_value("\n".join(tail_lines))
            props["last_log_lines"] = scrubbed.encode("utf-8")[:4096].decode(
                "utf-8", errors="ignore"
            )
        except Exception:
            pass  # nosec B110
    track(ev.PLUGIN_ERROR, props)

# --- Error capture (top-level slots + worker bodies) ----------------------
#
# The only error -> telemetry path used to be an explicit show_error_report()
# call, so an uncaught exception in a Qt slot or worker run() body produced NO
# telemetry at all. These helpers give a standard capture pattern: track a
# plugin_error with a stable English error_code, a traceback_hash (so the
# analytics backend groups recurrences of the same crash), and the source
# module; log a line; and
# show a dialog ONLY when an explicit user_message is passed. No sys.excepthook
# (that conflicts with QGIS and other plugins) - wrap the entry points instead.


def _short_traceback_hash(exc: BaseException) -> str:
    """A short, path-free fingerprint of an exception's traceback.

    Each frame contributes basename:lineno:function; the exception class name
    is appended. Filenames are reduced to their basename so the hash is stable
    across machines. Returns "" if anything goes wrong (never raises)."""
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
    """Capture an unhandled exception: track it, log it, optionally show it.

    Always tracks a plugin_error (error_code = exception class name, plus a
    traceback_hash and module) and writes one QgsMessageLog line. Shows the
    error-report dialog ONLY when user_message is given AND we are on the main
    thread. Never raises: this runs on failure paths where a second failure
    must stay invisible."""
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
    """Decorator for a top-level Qt slot: catch any unhandled exception, report
    it (telemetry + log, and a dialog only when user_message is given), and
    swallow it so a stray crash never leaves QGIS's console handler as the only
    trace. Do NOT stack on slots that already surface their own errors."""
    def deco(fn):
        module = (fn.__module__ or "").rsplit(".", 1)[-1]

        # Qt passes signal payloads (e.g. clicked's `checked` bool) into the
        # slot. Some PyQt builds introspect this wrapper's (*args) as "takes
        # anything" and forward them, so calling fn with the extra args raised
        # TypeError inside the guard and the slot silently did NOTHING (launch
        # bug: a dead Cancel button). Trim to fn's real positional arity.
        try:
            import inspect
            params = list(inspect.signature(fn).parameters.values())[1:]  # drop self
            _has_var = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params)
            _max_pos = None if _has_var else sum(
                1 for p in params
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD)
            )
        except Exception:  # noqa: BLE001 - introspection is best-effort
            _max_pos = None

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if _max_pos is not None and len(args) > _max_pos:
                args = args[:_max_pos]
            try:
                return fn(self, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - top-level slot boundary
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


# NOTE: worker run() bodies report inline (see AutoDetectionWorker.run): they
# pair the report with worker-specific cleanup (power inhibit, error signal),
# which a generic context manager cannot know about. slot_guard above is the
# one shared boundary; an unused error_boundary context manager was removed.
