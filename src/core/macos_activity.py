
















from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)





_NS_ACTIVITY_USER_INITIATED = 0x00FFFFFF

_IS_MACOS = sys.platform == "darwin"



_objc_memo: dict[str, dict] = {}


def _load_objc() -> dict | None:

    cached = _objc_memo.get("objc")
    if cached is not None:
        return cached or None

    import ctypes
    import ctypes.util

    try:

        ctypes.cdll.LoadLibrary(ctypes.util.find_library("Foundation"))
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))

        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]



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
            _objc_memo["objc"] = {}
            return None

        resolved = {
            "process_info": process_info,
            "nsstring_cls": nsstring_cls,
            "sel_begin": sel_begin,
            "sel_end": sel_end,
            "sel_str": sel_str,
            "send_str": send_str,
            "send_begin": send_begin,
            "send_end": send_end,
        }
        _objc_memo["objc"] = resolved
        return resolved
    except Exception as exc:
        logger.debug("macos_activity: ObjC unavailable, App Nap not suppressed: %s", exc)
        _objc_memo["objc"] = {}
        return None


def begin_app_nap_activity(reason: str = "AI Segmentation task"):



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
        logger.debug("macos_activity: begin_app_nap_activity failed: %s", exc)
        return None


def end_app_nap_activity(token) -> None:

    if not _IS_MACOS or token is None:
        return
    objc = _load_objc()
    if objc is None:
        return
    try:
        objc["send_end"](objc["process_info"], objc["sel_end"], token)
    except Exception as exc:
        logger.debug("macos_activity: end_app_nap_activity failed: %s", exc)
