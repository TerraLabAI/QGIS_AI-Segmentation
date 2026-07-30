"""How a backend error code is handled, with the server able to retune it.

The plugin decides three things from a code the backend sent: retry the tile,
end the run, or end it as a clean out-of-credits. Those decisions used to be
frozen sets in the worker, so a code the backend started sending after a
release was handled by its worst-case branch until every user updated. On a
slow update curve that is weeks of runs failing on a code we already knew about
the day it shipped.

Each set here is a shipped base plus server extras, union-only, so a deploy can
teach the whole fleet about a new code and can never take a shipped one away
(see ``server_dials``: dropping a shipped safety code is the one thing a bad
configuration must not be able to do). The classifier is the same idea for the
message a failed run shows: served rules are consulted first, and the shipped
ladder still runs underneath when none of them match.

Codes are compared exactly as the backend spells them, in upper case. Served
extras are upper-cased on the way in so casing drift in the configuration
cannot defeat a match.

Pure Python, no Qt and no QGIS at import time, cache-only and never raising, so
the worker thread, the controller and the headless path all import it freely.
"""
from __future__ import annotations

from itertools import islice

from .server_dials import ServerDialSet, read_value

# Transient: the tile is worth sending again. Connectivity-side failures plus
# the service's own "still coming up" answer.
TRANSIENT_CODES = ServerDialSet(
    "errors.transient_extra",
    {
        "NO_INTERNET", "TIMEOUT", "DNS_ERROR", "PROXY_ERROR",
        "SSL_ERROR", "SERVER_ERROR", "CONNECTION_REFUSED",
        "SERVICE_WARMING",
    },
    normalize=str.upper,
)

# Out of credits: a clean end of run, never a tile failure and never a fault
# banner. One code per tier, and the server may name a new one.
EXHAUSTED_CODES = ServerDialSet(
    "errors.exhausted_extra",
    {"CREDITS_EXHAUSTED", "FREE_DETECTIONS_EXHAUSTED", "QUOTA_EXCEEDED"},
    normalize=str.upper,
)

# The service is up but a dependency it needs at submit time is not. Retryable
# on a short attempt count rather than the connectivity ladder, because the
# link is fine. Raised before any charge, so re-sending the tile cannot bill
# twice.
BACKEND_UNAVAILABLE_CODES = ServerDialSet(
    "errors.backend_unavailable_extra",
    {"AUTH_BACKEND_UNAVAILABLE"},
    normalize=str.upper,
)

# Run-level: every tile would fail the same way, so the run stops at once
# instead of burning the retry budget tile by tile. Anything not listed rejects
# only its own tile.
RUN_FATAL_CODES = ServerDialSet(
    "errors.fatal_extra",
    {"AUTH_ERROR", "INVALID_KEY", "SUBSCRIPTION_INACTIVE"},
    normalize=str.upper,
)

# Sentinel raised by the offline fast-fail so the terminal error carries a
# connectivity message instead of a raw code. Not a dial: it is produced by the
# client, never received from the backend.
OFFLINE_STOP_CODE = "NO_CONNECTION"

# Failure classes that may offer the bug-report link. A class the user caused
# (their credits, their sign-in, their cancel) is not a bug, so it is absent
# and stays absent unless the server names it.
REPORTABLE_ERROR_CLASSES = ServerDialSet(
    "errors.report_link_extra",
    {"NETWORK", "SERVER", "TIMEOUT", "UNKNOWN"},
    normalize=str.upper,
)

# Every class the failure banner knows how to render. A served rule naming
# anything else is dropped, so the configuration can re-route a failure but can
# never hand the UI a class it has no branch for.
RUN_ERROR_CLASSES = frozenset({
    "NETWORK", "AUTH", "CREDITS_EXHAUSTED", "SERVER", "CANCELLED", "TIMEOUT", "UNKNOWN",
})

# A served rule list is a handful of short markers. These caps are what stops a
# damaged configuration from making the classifier expensive, and they apply
# before anything is lower-cased or copied.
_MAX_RULES = 32
_MAX_MARKERS_PER_RULE = 16
_MAX_MARKER_CHARS = 64


def _served_classifier_rules() -> list[tuple[tuple[str, ...], str]]:
    """The served ``(markers, class)`` rules, in order, dropping bad entries.

    Validation is per rule, so one malformed entry never costs the others, and
    an entry naming an unknown class is dropped rather than rendered.
    """
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
    except Exception:  # noqa: BLE001 -- a bad rule is not an error  # nosec B110
        return rules
    return rules


def classify_run_error(message: str) -> str:
    """The failure class of an Automatic run, from the error message it ended on.

    One of ``RUN_ERROR_CLASSES``. It picks the banner the user sees, whether a
    bug-report link is offered, and the telemetry ``error_class``.

    Served rules are consulted first, in the order the configuration lists
    them, so a code we start sending tomorrow can be routed to the right banner
    the same day. The shipped ladder below runs when no served rule matches, so
    a configuration that says nothing leaves today's behaviour untouched.
    """
    low = (message or "").lower()
    if not low:
        return "UNKNOWN"
    for markers, name in _served_classifier_rules():
        if any(marker in low for marker in markers):
            return name
    return _shipped_error_class(low)


def _shipped_error_class(low: str) -> str:
    """The ladder that ships with the plugin, on an already lower-cased message.

    Order carries the meaning here and is not alphabetical: the outage markers
    are read before the keyword scan, because the service names two of its
    outage codes after the thing they could not reach. Read as plain keywords,
    an unreachable key check reads as "sign in again" and an unreachable quota
    store reads as "you are out of credits", when both are the same outage on
    our side.
    """
    if (
        "backend_unavailable" in low
        or "quota_check_failed" in low
        or "service temporarily unavailable" in low
        or "503" in low
    ):
        return "SERVER"
    # A fault on our own side, marked by the worker's last-resort net. Qt
    # exception text names the class it failed on, and a reply class name
    # carries the word "network", so the connectivity scan below would tell a
    # user with a good link to check their connection.
    if "internal error" in low:
        return "UNKNOWN"
    # "exhausted" is here because the free-tier code says nothing else: it
    # carries neither "credit" nor "quota".
    if "credit" in low or "quota" in low or "402" in low or "exhausted" in low:
        return "CREDITS_EXHAUSTED"
    # Transient service-side rejections carry AUTH in their code but are not
    # authentication failures: the session is valid and signing in again
    # changes nothing.
    if "backend_unavailable" in low or "warming" in low:
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
