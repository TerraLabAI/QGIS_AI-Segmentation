"""Prevent macOS App Nap during a long-running task.

macOS puts an app into "App Nap" when its windows are occluded (e.g. the user
opens another app on top of QGIS): timers are throttled, threads deprioritised,
and network-event delivery is choked. Our cloud detection worker drives blocking
network calls off a QThread whose run loop is then starved, so a run appears to
stall until QGIS is brought back to the foreground.

`NSProcessInfo -beginActivityWithOptions:reason:` declares a user-initiated
activity for which the system must NOT nap the process. We hold one token for
the duration of a run and end it afterwards.

Everything here is best-effort: a no-op on non-macOS, and any failure (missing
framework, ObjC runtime quirk) is swallowed so detection never breaks because of
power management. We talk to the Objective-C runtime directly via ctypes so there
is no pyobjc dependency (QGIS's bundled Python does not ship it).
"""
from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

# NSActivityOptions.NSActivityUserInitiated: "work the user asked for; do not
# nap me." Includes idle-system-sleep-disabled, which is acceptable for the few
# seconds-to-minutes a detection run lasts; the token is always ended in a
# finally, so the suppression can never outlive the run.
_NS_ACTIVITY_USER_INITIATED = 0x00FFFFFF

_IS_MACOS = sys.platform == "darwin"

# Cached ObjC primitives (resolved once, lazily). None until first use; the
# tuple holds (process_info_instance, sel_begin, sel_end, msgSend variants).
_objc_cache: dict | None = None


def _load_objc() -> dict | None:
    """Resolve the ObjC bits we need once. Returns None if unavailable."""
    global _objc_cache
    if _objc_cache is not None:
        return _objc_cache or None

    import ctypes
    import ctypes.util

    try:
        # Foundation must be loaded so NSProcessInfo/NSString are registered.
        ctypes.cdll.LoadLibrary(ctypes.util.find_library("Foundation"))
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))

        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]

        # objc_msgSend has no single signature; cast the one function pointer to
        # a distinct prototype per call shape (the standard ctypes ObjC pattern).
        def _msg(restype, argtypes):
            proto = ctypes.CFUNCTYPE(restype, *argtypes)
            return ctypes.cast(objc.objc_msgSend, proto)

        send_obj = _msg(ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p])
        send_str = _msg(
            ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
        )
        send_begin = _msg(
            ctypes.c_void_p,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p],
        )
        send_end = _msg(
            None, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        )

        process_info_cls = objc.objc_getClass(b"NSProcessInfo")
        nsstring_cls = objc.objc_getClass(b"NSString")
        sel_process_info = objc.sel_registerName(b"processInfo")
        sel_begin = objc.sel_registerName(b"beginActivityWithOptions:reason:")
        sel_end = objc.sel_registerName(b"endActivity:")
        sel_str = objc.sel_registerName(b"stringWithUTF8String:")

        process_info = send_obj(process_info_cls, sel_process_info)
        if not process_info:
            _objc_cache = {}
            return None

        _objc_cache = {
            "process_info": process_info,
            "nsstring_cls": nsstring_cls,
            "sel_begin": sel_begin,
            "sel_end": sel_end,
            "sel_str": sel_str,
            "send_str": send_str,
            "send_begin": send_begin,
            "send_end": send_end,
        }
        return _objc_cache
    except Exception as exc:  # any ObjC/ctypes failure -> silently no-op
        logger.debug("macos_activity: ObjC unavailable, App Nap not suppressed: %s", exc)
        _objc_cache = {}
        return None


def begin_activity(reason: str = "AI Segmentation task"):
    """Start a no-App-Nap activity. Returns an opaque token to pass to
    end_activity(), or None on non-macOS / any failure (callers must handle
    None as a no-op)."""
    if not _IS_MACOS:
        return None
    objc = _load_objc()
    if objc is None:
        return None
    try:
        reason_str = objc["send_str"](
            objc["nsstring_cls"], objc["sel_str"], reason.encode("utf-8")
        )
        token = objc["send_begin"](
            objc["process_info"],
            objc["sel_begin"],
            _NS_ACTIVITY_USER_INITIATED,
            reason_str,
        )
        return token or None
    except Exception as exc:
        logger.debug("macos_activity: begin_activity failed: %s", exc)
        return None


def end_activity(token) -> None:
    """End an activity started by begin_activity(). Safe to call with None."""
    if not _IS_MACOS or token is None:
        return
    objc = _load_objc()
    if objc is None:
        return
    try:
        objc["send_end"](objc["process_info"], objc["sel_end"], token)
    except Exception as exc:
        logger.debug("macos_activity: end_activity failed: %s", exc)
