






















from __future__ import annotations

import re
from itertools import islice

from .server_dials import ServerDialSet, read_value



TRANSIENT_CODES = ServerDialSet(
    "errors.transient_extra",
    {
        "NO_INTERNET", "TIMEOUT", "DNS_ERROR", "PROXY_ERROR",
        "SSL_ERROR", "SERVER_ERROR", "CONNECTION_REFUSED",
        "SERVICE_WARMING",
    },
    normalize=str.upper,
)








LINK_FAILURE_CODES = ServerDialSet(
    "errors.link_failure_extra",
    {"NO_INTERNET", "DNS_ERROR", "PROXY_ERROR", "SSL_ERROR",
     "CONNECTION_REFUSED"},
    normalize=str.upper,
)



EXHAUSTED_CODES = ServerDialSet(
    "errors.exhausted_extra",
    {"CREDITS_EXHAUSTED", "FREE_DETECTIONS_EXHAUSTED", "QUOTA_EXCEEDED",
     "TRIAL_EXHAUSTED"},
    normalize=str.upper,
)







BACKEND_UNAVAILABLE_CODES = ServerDialSet(
    "errors.backend_unavailable_extra",
    {
        "AUTH_BACKEND_UNAVAILABLE",


        "QUOTA_CHECK_FAILED",

        "JOB_INSERT_FAILED",
    },
    normalize=str.upper,
)




RUN_FATAL_CODES = ServerDialSet(
    "errors.fatal_extra",




    {"AUTH_ERROR", "INVALID_KEY", "SUBSCRIPTION_INACTIVE",
     "DEVICE_LIMIT_EXCEEDED", "WRONG_PRODUCT", "NO_ACCOUNT"},
    normalize=str.upper,
)










ACCOUNT_REFUSAL_REASONS = {
    "AUTH_ERROR": "SIGN_IN",
    "INVALID_KEY": "SIGN_IN",
    "SUBSCRIPTION_INACTIVE": "SUBSCRIPTION",


    "WRONG_PRODUCT": "SUBSCRIPTION",
    "NO_ACCOUNT": "NO_ACCOUNT",
}




OFFLINE_STOP_CODE = "NO_CONNECTION"




REPORTABLE_ERROR_CLASSES = ServerDialSet(
    "errors.report_link_extra",
    {"NETWORK", "SERVER", "TIMEOUT", "UNKNOWN"},
    normalize=str.upper,
)




RUN_ERROR_CLASSES = frozenset({
    "NETWORK", "AUTH", "CREDITS_EXHAUSTED", "SERVER", "CANCELLED", "TIMEOUT",
    "DEVICE_LIMIT", "UNKNOWN",
})




_MAX_RULES = 32
_MAX_MARKERS_PER_RULE = 16
_MAX_MARKER_CHARS = 64



_CODE_WORD_RE = re.compile(r"[a-z][a-z0-9_]{2,63}")
_MAX_CODE_WORDS = 32


def _codes_named_in(low: str) -> list[str]:







    return [word.upper() for word in islice(_CODE_WORD_RE.findall(low), _MAX_CODE_WORDS)]


def _served_classifier_rules() -> list[tuple[tuple[str, ...], str]]:





    rules: list[tuple[tuple[str, ...], str]] = []
    try:
        served = read_value("errors.classifier")
        if not isinstance(served, (list, tuple)):
            return rules
        for entry in islice(served, _MAX_RULES):
            if not isinstance(entry, dict):
                continue
            name = entry.get("class")
            if not isinstance(name, str) or name.strip().upper() not in RUN_ERROR_CLASSES:
                continue
            raw = entry.get("contains")
            if not isinstance(raw, (list, tuple)):
                continue
            markers = tuple(
                item.strip().lower()
                for item in islice(raw, _MAX_MARKERS_PER_RULE)
                if isinstance(item, str) and 0 < len(item) <= _MAX_MARKER_CHARS and item.strip()
            )
            if markers:
                rules.append((markers, name.strip().upper()))
    except Exception:  # noqa: BLE001  # nosec B110
        return rules
    return rules


def classify_run_error(message: str) -> str:










    low = (message or "").lower()
    if not low:
        return "UNKNOWN"
    for markers, name in _served_classifier_rules():
        if any(marker in low for marker in markers):
            return name
    return _shipped_error_class(low)


def account_refusal_reason(message: str) -> str:











    for code in _codes_named_in((message or "").lower()):
        reason = ACCOUNT_REFUSAL_REASONS.get(code)
        if reason:
            return reason
    return ""


def _shipped_error_class(low: str) -> str:











    named = _codes_named_in(low)
    for code in named:
        reason = ACCOUNT_REFUSAL_REASONS.get(code)
        if reason:
            return "AUTH"



    if any(code in BACKEND_UNAVAILABLE_CODES for code in named):
        return "SERVER"
    if (
        "backend_unavailable" in low
        or "quota_check_failed" in low
        or "service temporarily unavailable" in low
        or "503" in low
    ):
        return "SERVER"



    if "device_limit_exceeded" in low or "device limit" in low:
        return "DEVICE_LIMIT"




    if "internal error" in low:
        return "UNKNOWN"



    if any(code in EXHAUSTED_CODES for code in named):
        return "CREDITS_EXHAUSTED"
    if "credit" in low or "quota" in low or "402" in low:
        return "CREDITS_EXHAUSTED"



    if "warming" in low:
        return "SERVER"
    if "auth" in low or "401" in low or "403" in low or "sign in" in low:
        return "AUTH"
    if "timeout" in low or "timed out" in low:
        return "TIMEOUT"
    if "cancel" in low:
        return "CANCELLED"
    if any(t in low for t in ("network", "connection", "connect", "ssl", "dns",
                              "unreachable", "offline")):
        return "NETWORK"
    if any(t in low for t in ("server", "500", "502", "503", "504", "bad gateway")):
        return "SERVER"
    return "UNKNOWN"
