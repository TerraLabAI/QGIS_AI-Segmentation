from __future__ import annotations

import json
import math
import time

from qgis.core import Qgis, QgsBlockingNetworkRequest, QgsMessageLog
from qgis.PyQt.QtCore import QByteArray, QUrl
from qgis.PyQt.QtNetwork import QNetworkReply, QNetworkRequest

from ..core.i18n import tr
from .request_compression import (
    answer_refused_the_body,
    note_gzip_request_refused,
    packed_request_body,
)

_TIMEOUT_API = 30_000
_TIMEOUT_INTERACTIVE = 10_000        # ms: lightweight config/usage/account GETs;
#                                      short so a bad connection surfaces fast.
_TIMEOUT_SUBMIT_DETECTION = 45_000   # ms: submit is small (base64 PNG inline)
# Direct mode: the submit BLOCKS for the whole inference (no async "pending"
# fallback like the backend route had), so a cold-start tile (idle backend
# model load, ~18-60s) must not trip the client timeout - that would retry and
# DOUBLE-CHARGE a tile the server already billed + computed. 110s waits a cold
# start out while staying under the detection endpoint's request timeout (120s).
# Server-overridable, so the two sides can move together: see _submit_timeout.
_TIMEOUT_SUBMIT_DETECTION_DIRECT = 110_000
_TIMEOUT_POLL_DETECTION = 15_000     # ms: poll is GET with tiny JSON response
_TIMEOUT_WARMUP = 5_000              # ms: warmup is a tiny best-effort ping
# ms: prompt translation holds a Detect back (it runs off the GUI thread, so it
# never freezes QGIS, but the user is waiting on it), so it gives up rather
# than hold the run. Long enough that a slow link still gets an answer: giving
# up costs the user the tuned run, which is worth more than the seconds saved.
_TIMEOUT_TRANSLATE = 12_000
# ms: final-output upload can carry a few MB of geometry; background task only.
_TIMEOUT_RUN_EXPORT = 60_000
# ms past a request's own deadline before the wall-clock backstop ends the wait.
# setTransferTimeout restarts on every byte, so a server that drips can hold a
# thread for good; this bounds the whole round trip instead. Sits past Qt's own
# deadline so it never fires first on a healthy slow answer.
_WALL_CLOCK_GUARD_MS = 5_000

# Qt6 (QGIS 4) uses scoped enums; Qt5 uses flat. Resolve at import time.
_NE = getattr(QNetworkReply, "NetworkError", QNetworkReply)
_HostNotFound = getattr(_NE, "HostNotFoundError", getattr(QNetworkReply, "HostNotFoundError", None))
_ConnRefused = getattr(_NE, "ConnectionRefusedError", getattr(QNetworkReply, "ConnectionRefusedError", None))
_Timeout = getattr(_NE, "TimeoutError", getattr(QNetworkReply, "TimeoutError", None))
# What Qt actually emits when a request's setTransferTimeout deadline expires:
# OperationCanceledError, NOT TimeoutError. A cold-starting / busy detection
# host that answers slower than the submit deadline lands here, so it must
# be read as "the service is warming up" (a transient, retry-worthy wait), not
# as the user's connection dying. reply.abort() on our own cancel path also
# raises this, but the cancel path never classifies+retries an aborted reply.
_OpCanceled = getattr(_NE, "OperationCanceledError", getattr(QNetworkReply, "OperationCanceledError", None))
_SslFailed = getattr(_NE, "SslHandshakeFailedError", getattr(QNetworkReply, "SslHandshakeFailedError", None))
_ContentDenied = getattr(_NE, "ContentAccessDenied", getattr(QNetworkReply, "ContentAccessDenied", None))
_AuthRequired = getattr(_NE, "AuthenticationRequiredError", getattr(QNetworkReply, "AuthenticationRequiredError", None))
_UnknownNetwork = getattr(_NE, "UnknownNetworkError", getattr(QNetworkReply, "UnknownNetworkError", None))
_NoError = getattr(_NE, "NoError", getattr(QNetworkReply, "NoError", 0))

# Genuine connect failures: the request never reached a server, so the link
# itself is down. Host-not-found and connection-refused have their own codes
# and are classified before this set is consulted; these are Qt's remaining
# "the network went away" cases. They are kept apart from the unnamed-failure
# fallthrough because ONLY a genuine connect failure may count toward the
# offline fast-fail (see workers.adaptive_concurrency.OfflineFastFail).
_CONNECT_FAILURE_ERRORS = set(filter(None, [
    getattr(_NE, "TemporaryNetworkFailureError",
            getattr(QNetworkReply, "TemporaryNetworkFailureError", None)),
    getattr(_NE, "NetworkSessionFailedError",
            getattr(QNetworkReply, "NetworkSessionFailedError", None)),
]))

_PROXY_ERRORS = set(filter(None, [
    getattr(_NE, "ProxyConnectionRefusedError", getattr(QNetworkReply, "ProxyConnectionRefusedError", None)),
    getattr(_NE, "ProxyConnectionClosedError", getattr(QNetworkReply, "ProxyConnectionClosedError", None)),
    getattr(_NE, "ProxyNotFoundError", getattr(QNetworkReply, "ProxyNotFoundError", None)),
    getattr(_NE, "ProxyTimeoutError", getattr(QNetworkReply, "ProxyTimeoutError", None)),
    getattr(_NE, "ProxyAuthenticationRequiredError", getattr(QNetworkReply, "ProxyAuthenticationRequiredError", None)),
    getattr(_NE, "UnknownProxyError", getattr(QNetworkReply, "UnknownProxyError", None)),
]))

# QNetworkRequest.Attribute.HttpStatusCodeAttribute (Qt6) vs QNetworkRequest.HttpStatusCodeAttribute (Qt5)
_Attr = getattr(QNetworkRequest, "Attribute", QNetworkRequest)
_HTTP_STATUS_ATTR = getattr(_Attr, "HttpStatusCodeAttribute", getattr(QNetworkRequest, "HttpStatusCodeAttribute", None))

# Follow same-or-safer redirects (e.g. signed-URL hops) without downgrading
# HTTPS. Resolved defensively for the Qt5 (flat) vs Qt6 (scoped) enum split.
_REDIRECT_ATTR = getattr(_Attr, "RedirectPolicyAttribute",
                         getattr(QNetworkRequest, "RedirectPolicyAttribute", None))
_RedirectPolicy = getattr(QNetworkRequest, "RedirectPolicy", QNetworkRequest)
_NO_LESS_SAFE_REDIRECT = getattr(_RedirectPolicy, "NoLessSafeRedirectPolicy",
                                 getattr(QNetworkRequest, "NoLessSafeRedirectPolicy", None))
_SAME_ORIGIN_REDIRECT = getattr(_RedirectPolicy, "SameOriginRedirectPolicy",
                                getattr(QNetworkRequest, "SameOriginRedirectPolicy", None))


def _log_warning(msg: str):
    QgsMessageLog.logMessage(msg, "AI Segmentation", level=Qgis.MessageLevel.Warning)


class _WallClockGuard:
    """End a blocking request that outlives its whole deadline.

    ``QgsBlockingNetworkRequest`` spins its own event loop, so the timer runs
    inside that loop, on the same thread, and the abort stays thread-affine.
    Built and stopped around the one call it guards; a timer with no event
    dispatcher behind it simply never fires, which is the same as not having
    one.
    """

    def __init__(self, blocker, timeout_ms: int) -> None:
        from qgis.PyQt.QtCore import QTimer

        self._blocker = blocker
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(max(1_000, int(timeout_ms)) + _WALL_CLOCK_GUARD_MS)
        self._timer.timeout.connect(self._end_it)
        try:
            self._timer.start()
        except (RuntimeError, TypeError):
            pass  # nosec B110 - no dispatcher on this thread, nothing to time with

    def _end_it(self) -> None:
        try:
            self._blocker.abort()
        except (AttributeError, RuntimeError) as err:
            _log_warning(f"Could not end a stalled request: {err}")

    def stop(self) -> None:
        try:
            self._timer.stop()
        except RuntimeError:
            pass  # nosec B110 - already gone with its thread


# How long one real answer keeps counting as proof the service is reachable.
# It has to outlast the longest single request deadline (the direct submit
# window), so a request that started while the link was good is still judged
# against that good link when its own deadline expires. Past it, the link has
# gone quiet for longer than any legitimate wait and a silent deadline is read
# as a dead link again.
_SERVER_CONTACT_TTL_S = 180.0
_last_server_contact_monotonic: float | None = None


def note_server_contact() -> None:
    """Record that a request came back with a real answer.

    Any HTTP status counts, including a failure status: the point is that the
    bytes made the round trip, not that the service liked them."""
    global _last_server_contact_monotonic
    _last_server_contact_monotonic = time.monotonic()


def server_reached_recently() -> bool:
    """Whether a real answer arrived inside _SERVER_CONTACT_TTL_S.

    This is what separates a service that is slow from a link that is gone. A
    firewall or captive portal that DROPs packets instead of refusing them ends
    a request exactly like a cold-starting service does, with a silent deadline
    and no HTTP status, so the deadline alone cannot tell them apart. A recent
    answer can. A clock jump backwards reads as expired, never as fresh."""
    stamp = _last_server_contact_monotonic
    if stamp is None:
        return False
    age = time.monotonic() - stamp
    return 0.0 <= age <= _SERVER_CONTACT_TTL_S


def _apply_redirect_policy(req: QNetworkRequest, has_auth: bool) -> None:
    """Pick the redirect policy for one request.

    Authenticated calls carry the activation key in a raw header and Qt replays
    raw headers on every hop it follows, so they are limited to same-origin
    redirects: a 3xx pointing at another host is refused instead of handing the
    key to that host. Unauthenticated calls keep the wider same-or-safer policy
    (signed-URL hops), which still never downgrades HTTPS to HTTP."""
    if _REDIRECT_ATTR is None:
        return
    policy = _SAME_ORIGIN_REDIRECT if has_auth else _NO_LESS_SAFE_REDIRECT
    if policy is not None:
        req.setAttribute(_REDIRECT_ATTR, policy)


def _parse_json_body(raw_body: str, allow_list: bool = False):
    """Parse a response body into a shape callers can actually read.

    Every caller does ``.get(...)`` on what the client returns, so a body that
    parses to ``null``, a bare string or a number must never be handed back:
    that turns a misbehaving gateway into an AttributeError deep in a run.
    Returns the parsed value when usable, else None so the caller falls back to
    its own error result. ``allow_list`` keeps the one route that legitimately
    answers a JSON array (the stored masks archive) working.

    Raises whatever json.loads raises; the callers already handle that."""
    parsed = json.loads(raw_body)
    if isinstance(parsed, dict):
        return parsed
    if allow_list and isinstance(parsed, list):
        return parsed
    return None


# Codes that mean the request never got an answer, so asking again is worth a
# round trip. Anything the server did answer, including a refusal, is its
# answer and stands.
_WORTH_ASKING_AGAIN_CODES = ("TIMEOUT", "NO_INTERNET")

# One retry, spread over this window so a fleet that lost the same link does
# not come back in step and finish the server off.
_RETRY_PAUSE_MIN_S = 0.3
_RETRY_PAUSE_MAX_S = 0.8


def _worth_asking_again(answer, http_status: int | None) -> bool:
    """Whether this answer is a failure that a second identical request may
    get past. Only ever consulted about a GET."""
    if http_status is not None and 500 <= http_status < 600:
        return True
    if http_status is not None:
        return False
    return (isinstance(answer, dict)
            and answer.get("code") in _WORTH_ASKING_AGAIN_CODES)


def _retry_pause_s() -> float:
    """How long to wait before the one retry."""
    import random  # noqa: PLC0415 -- only needed once a request has failed

    return random.uniform(  # nosec B311 -- spreading a retry, not security
        _RETRY_PAUSE_MIN_S, _RETRY_PAUSE_MAX_S)


def _note_skipped_tuning(what: str, answer) -> None:
    """Say in the log that a run went ahead without one of its tuned inputs.

    Both of these degrade in silence by design, so the run still happens; the
    line is the only way to tell a run tuned by the server from one that fell
    back. The name of the thing and the failure code, never a URL and never
    what the user typed.
    """
    if not isinstance(answer, dict) or "error" not in answer:
        return
    _log_warning(f"Ran without the {what} "
                 f"({answer.get('code') or 'no code'})")


def _unreadable_answer() -> dict:
    """The failure for a 2xx whose body is not ours to read.

    Its own code, apart from SERVER_ERROR, because the cause is between the two
    machines rather than on either of them: a wifi sign-in page answering every
    address, or a proxy that strips the body. The service never saw the request,
    so "try again in a few minutes" is the wrong advice and silence is worse:
    the account panel would just show stale numbers with no reason given."""
    return {
        "error": tr(
            "The reply did not come from the service. If this network shows a "
            "sign-in page, open it in your browser first, then try again."
        ),
        "code": "UNREADABLE_RESPONSE",
    }


# The private network managers, one per thread, shared by every client
# instance on that thread. Callers build a TerraLabClient per request (they are
# cheap and hold no state), so a cache living on the instance handed every one
# of them a manager of its own: a fresh name lookup, connection and TLS
# handshake for a call that repeats on a timer. Keyed here instead, the second
# call on a thread reuses the socket the first one left open.
#
# Kept in step with _THREAD_NAM_WATCHED, which remembers the threads whose end
# has already been armed to drop their entry (see _retain_thread_nam).
_THREAD_NAMS: dict = {}
_THREAD_NAM_WATCHED: set = set()


def _drop_thread_nam() -> None:
    """Destroy the calling thread's private manager as that thread ends.

    Armed by ``retain_thread_nam`` on the thread's own ``finished`` signal, so
    it runs in that thread's context and the C++ manager is destroyed where it
    was created. Never raises: it runs during a thread teardown.
    """
    try:
        from qgis.PyQt.QtCore import QThread

        thread = QThread.currentThread()
        _THREAD_NAMS.pop(thread, None)
        _THREAD_NAM_WATCHED.discard(thread)
    except Exception:  # noqa: BLE001 -- a teardown must never raise  # nosec B110
        pass


def _qobject_alive(obj) -> bool:
    """True while obj still has its C++ half.

    A destroyed QObject leaves a Python wrapper behind that raises RuntimeError
    ("wrapped C/C++ object ... has been deleted") on every attribute call, so
    the only probe is a call whose answer we throw away.
    """
    if obj is None:
        return False
    try:
        obj.thread()
    except RuntimeError:
        return False
    return True


def _http_status_of(reply) -> int | None:
    """HTTP status code of a reply, or None. Never raises (Qt can return an
    invalid/non-numeric attribute, which int() would choke on)."""
    if reply is None or _HTTP_STATUS_ATTR is None:
        return None
    attr = reply.attribute(_HTTP_STATUS_ATTR)
    if attr is None:
        return None
    try:
        return int(attr)
    except (TypeError, ValueError):
        return None


def _reply_was_packed(reply) -> bool:
    """Whether the request behind this reply went out gzipped.

    Read off the request itself rather than remembered beside it, because the
    replies on the concurrent paths are handed back to a caller that drives
    them and parses them wherever it likes.
    """
    try:
        return bytes(reply.request().rawHeader(b"Content-Encoding")) == b"gzip"
    except Exception:  # noqa: BLE001 -- an unreadable request was not packed
        return False


def _classify_network_error(blocker: QgsBlockingNetworkRequest) -> tuple[str, str]:
    reply = blocker.reply()
    qt_error = reply.error() if reply else _UnknownNetwork
    return _classify_qt_error(
        qt_error, blocker.errorMessage(), _http_status_of(reply),
        service_reachable=server_reached_recently(),
    )


# -- what the server may change about a network failure ---------------------
# Every sentence below is a served key under copy.network.<id>. Only the
# wording moves: the code returned beside it stays client-side, because
# consumers act on it (the offline fast-fail ends a run on a streak of
# hard-connectivity codes) and served text must never change what a run does.
#
# The wording is what points a user at the right fix, and it is release-blocked
# today. A sentence that sends people to the wrong place, or that reads badly
# in one language, has to be replaceable on the day it is reported.
#
# One key per distinct sentence, so two codes saying the same thing move
# together. An absent entry means the shipped sentence, which is what a user
# sees today.


def _classify_qt_error(
    qt_error,
    error_string: str,
    http_status: int | None,
    service_reachable: bool = False,
) -> tuple[str, str]:
    """Map a Qt NetworkError to a (code, user-message) pair. Shared by the
    blocking path (_classify_network_error) and the concurrent path
    (_parse_reply) so both classify identically.

    Contract: NO_INTERNET means the request never reached a server. Anything
    the server answered, and anything unnamed after a connection was
    established, is SERVER_ERROR. Consumers act on that split (the offline
    fast-fail ends a run on a streak of hard-connectivity codes), so an
    unnamed failure must never borrow the offline code.

    ``service_reachable`` says whether an answer came back recently (see
    server_reached_recently). It is the only evidence available on a silent
    deadline, so callers pass it; the default is the safe reading, no evidence
    of a reachable service."""
    # PyQt6's NetworkError is a real Python enum: int() raises TypeError on it
    # (Qt6 QGIS builds), which turned every network hiccup into a raw
    # "TypeError: int() argument..." instead of the friendly message + retry.
    qt_error_num = getattr(qt_error, "value", qt_error)
    # Qt's error strings for DNS, TLS and proxy failures quote the target host,
    # so the raw text would put the backend address in the QGIS message panel.
    # Scrub before logging (production-safe logging).
    try:
        from ..core.log_scrub import scrub_sensitive

        detail = scrub_sensitive(error_string or "")
    except Exception:  # noqa: BLE001 -- never let logging break a request
        detail = ""
    _log_warning(
        f"Network error: qt_error={qt_error_num}, http_status={http_status}, "
        f"detail={detail[:500]}"
    )
    # Served copy for the sentences below. Imported here like the scrubber
    # above, so plugin startup never pays for it. The read is cache-only and
    # fails open, and a module that cannot be imported at all leaves every
    # sentence exactly as it shipped.
    try:
        from ..core.server_dials import dial_copy
    except Exception:  # noqa: BLE001 -- served copy is best-effort  # nosec B110
        def dial_copy(_string_id: str, fallback: str) -> str:
            return fallback

    if qt_error == _HostNotFound:
        return "DNS_ERROR", dial_copy(
            "network.server_unreachable",
            tr("Cannot reach the server. Check your internet connection."),
        )
    if qt_error == _ConnRefused:
        return "CONNECTION_REFUSED", dial_copy(
            "network.connection_refused", tr("Server refused the connection."))
    if qt_error == _Timeout or (_OpCanceled is not None and qt_error == _OpCanceled):
        # No HTTP status means the socket never got an answer within the
        # deadline. Two very different things end that way: a cold-starting /
        # busy detection host, and a firewall or captive portal that DROPs
        # packets instead of refusing them. Only a recent real answer tells
        # them apart, so only then is this the service warming up. SERVICE_WARMING
        # holds the tile on the queue TIME budget and shows "your spot is held",
        # which a blocked link must never be allowed to claim: it would keep the
        # offline fast-fail permanently reset and leave the user watching a
        # waiting room that can never end.
        if http_status is None:
            if service_reachable:
                return "SERVICE_WARMING", dial_copy(
                    "network.service_waking_up",
                    tr("The AI service is waking up. Holding your spot…"),
                )
            # Nothing has answered for longer than any legitimate wait, and this
            # request got nothing either. That is the NO_INTERNET contract: the
            # request never reached a server. A hard-connectivity code, so a
            # streak of them ends a run instead of grinding every tile's retries.
            return "NO_INTERNET", dial_copy(
                "network.no_connection",
                tr("Network error. Check your internet connection."),
            )
        return "TIMEOUT", dial_copy(
            "network.request_timed_out",
            tr("Request timed out. Check your connection or try again."),
        )
    if qt_error == _SslFailed:
        return "SSL_ERROR", dial_copy(
            "network.secure_connection_blocked",
            tr("SSL certificate error. Your network may be blocking secure connections."),
        )
    if qt_error in _PROXY_ERRORS:
        return "PROXY_ERROR", dial_copy(
            "network.proxy_failed",
            tr(
                "Proxy connection failed. "
                "Check QGIS proxy settings (Settings > Options > Network)."
            ),
        )
    if qt_error in (_ContentDenied, _AuthRequired) and http_status == 401:
        # 401 only. Qt raises ContentAccessDenied for any 403, and on a blocked
        # network that 403 comes from a proxy or a portal, not from us. Telling
        # a user whose network is blocking the request to sign in again points
        # them at the wrong fix. A 401 or 403 that really is ours carries our
        # JSON error body, and both callers return that body before ever
        # reaching this classifier.
        return "AUTH_ERROR", dial_copy(
            "network.sign_in_again", tr("Authentication failed. Please sign in again."))
    # The server DID answer, with a failure status whose body was not
    # parseable JSON (typically an infrastructure incident page). The user's
    # connection worked, so "check your internet" points them at the wrong
    # side. SERVER_ERROR is already transient for every consumer: the tile
    # worker retries it and the revalidation path never signs out on it.
    if http_status is not None and http_status >= 500:
        return "SERVER_ERROR", dial_copy(
            "network.service_unavailable",
            tr(
                "The service is temporarily unavailable (server error). "
                "Your connection is fine - please try again in a few minutes."
            ),
        )
    # Any other HTTP status (400, 404, 409, 429...) whose body was not parseable
    # JSON: the request still completed a full round trip, so the link is fine
    # and the failure is on the service side. Same code as the 5xx case, which
    # every consumer already treats as transient and none counts as a
    # connectivity failure.
    if http_status is not None:
        return "SERVER_ERROR", dial_copy(
            "network.unexpected_response",
            tr("The server returned an unexpected response. Please try again."),
        )
    # No HTTP status: the link is the only thing that can be blamed, and only
    # for the errors that mean the request never reached a server. NO_INTERNET
    # is a hard-connectivity code (OfflineFastFail.HARD_CODES), so a run of
    # them ends a run early: it must never be returned for a condition the
    # server itself produced.
    if qt_error in _CONNECT_FAILURE_ERRORS:
        return "NO_INTERNET", dial_copy(
            "network.no_connection",
            tr("Network error. Check your internet connection."),
        )
    # Fallthrough: a peer accepted the connection and then broke the exchange
    # (remote host closed, malformed reply), or Qt could not name the failure.
    # Reported as a service-side failure so a live server is never announced as
    # "no internet" and a streak of them cannot end a paid run.
    return "SERVER_ERROR", dial_copy(
        "network.connection_interrupted",
        tr("The connection to the server was interrupted. Please try again."),
    )


class TerraLabClient:
    """HTTP client for the TerraLab backend - uses QgsBlockingNetworkRequest so
    requests pick up QGIS proxy / SSL / Network Logger settings."""

    def __init__(self, base_url: str | None = None):
        if base_url is None:
            base_url = self._read_base_url()
        self.base_url = base_url.rstrip("/")
        # Cloud detection (per-tile /predict) can talk DIRECTLY to the detection
        # endpoint instead of going through the main backend, removing a network
        # hop on the hot path. When a direct URL is configured the per-tile calls
        # use it (+ the direct route shapes); otherwise everything stays on
        # base_url unchanged. Resolved once at construction.
        direct = self._read_detection_base_url()
        if direct and not self._direct_detection_allowed():
            direct = ""
        self.detection_direct = bool(direct)
        self.detection_base_url = (direct or self.base_url).rstrip("/")

    @staticmethod
    def _direct_detection_allowed() -> bool:
        """Whether the one-hop detection route may be used.

        A server switch, so a region can be moved back onto the main route
        without a plugin release. Fails open: no configuration means the shipped
        behaviour is unchanged.
        """
        try:
            from ..core.server_dials import feature_enabled

            return feature_enabled("direct_inference")
        except Exception:  # noqa: BLE001 -- switch is best-effort  # nosec B110
            return True

    @staticmethod
    def _read_base_url() -> str:
        return TerraLabClient._read_env_value("TERRALAB_BASE_URL", "https://terra-lab.ai")

    @staticmethod
    def _read_detection_base_url() -> str:
        """Base URL for the per-tile cloud detection calls, when they should go
        DIRECT to the detection endpoint (one hop) instead of via the main backend.

        Resolution: an .env.local `TERRALAB_DETECTION_URL` (dev opt-in) wins, else the
        shipped default below. Empty string => not direct: detection keeps using
        the main base_url and the existing backend routes (zero behaviour change,
        the rollback: set the default back to "" to send everyone through
        the backend again).

        Either path is safe to switch between: both key the charge on the same
        (subscription, run, tile), so a tile that is sent twice on a flaky link
        is still paid for once."""
        # Shipped default: the neutral direct detection domain. Rollback stays
        # a one-line change: set back to "".
        _DEFAULT_DETECTION_DIRECT_URL = "https://inference.terra-lab.ai"
        return TerraLabClient._read_env_value("TERRALAB_DETECTION_URL", _DEFAULT_DETECTION_DIRECT_URL)

    @staticmethod
    def _read_env_value(name: str, default: str) -> str:
        import os
        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        env_path = os.path.join(plugin_dir, ".env.local")
        if os.path.isfile(env_path):
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"{name}="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        return default

    def _resolve_url(self, path_or_url: str) -> str:
        """A spec/path may be an absolute URL (the direct detection route) or a
        path relative to the main base_url (everything else). Absolute wins."""
        if path_or_url.startswith(("http://", "https://")):
            url = path_or_url
        else:
            url = f"{self.base_url}{path_or_url}"
        self._reject_cleartext_remote(url)
        return url

    @staticmethod
    def _reject_cleartext_remote(url: str) -> None:
        """Refuse to send requests (which carry the activation key) over plain
        HTTP to a non-local host. Only reachable via a hand-edited .env.local
        override; shipped defaults are HTTPS. Localhost stays allowed for dev.
        """
        if not url.startswith("http://"):
            return
        try:
            from urllib.parse import urlparse

            host = (urlparse(url).hostname or "").lower()
        except Exception:  # noqa: BLE001 - unparsable URL: treat as unsafe
            host = ""
        if host not in ("localhost", "127.0.0.1", "::1", ""):
            raise ValueError(
                "Refusing to send an authenticated request over plain HTTP to a "
                "remote host (the key would travel in cleartext). Use HTTPS."
            )

    def _detection_predict_url(self) -> str:
        """Absolute URL for the per-tile predict call. Direct mode hits the
        detection endpoint's /predict; otherwise the main backend route."""
        if self.detection_direct:
            return f"{self.detection_base_url}/predict"
        return f"{self.detection_base_url}/api/ai-segmentation/predict"

    def _detection_refine_url(self) -> str:
        """Absolute URL for one point-refine round trip. Same host split as the
        predict call, so the route follows whichever host the run itself used."""
        if self.detection_direct:
            return f"{self.detection_base_url}/refine"
        return f"{self.detection_base_url}/api/ai-segmentation/refine"

    def refine_endpoint_url(self) -> str:
        """The absolute address one point-refine round trip is sent to.

        For a caller that sends its own request and cannot use the waits here,
        so the host split stays decided in one place.
        """
        return self._resolve_url(self._detection_refine_url())

    def submit_refine(self, payload: dict, auth: dict, cancel_check=None) -> dict:
        """One point-refine call: a crop (or the token standing for it) plus the
        clicks, answered with masks, scores and the mask logits.

        Returns the parsed answer, or {"error", "code"} like every other route.
        The caller reads the code, because a crop the far side no longer holds
        is retried with the pixels rather than reported.

        ``cancel_check`` is an optional callable the drawing wait polls: as soon
        as it answers True the round trip ends and the answer is a refusal, so a
        session that ends mid-click is not left waiting on it.

        Shares the submit window with the per-tile call: both wait on the same
        inference, and it is already a server dial, so a slow day moves them
        together.
        """
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        # A click is answered on the thread that draws, so the ordinary wait
        # would stop the map for the length of the round trip. Try the wait
        # that keeps drawing first; it declines on any thread but that one, and
        # on anything it cannot do, and then this is the same call as ever.
        kept = self._refine_while_drawing(body, auth, cancel_check)
        if kept is not None:
            return kept
        return self._request(
            "POST",
            self._detection_refine_url(),
            auth=auth,
            body=body,
            timeout_ms=self._submit_timeout(),
        )

    def _refine_while_drawing(self, body: bytes, auth: dict,
                              cancel_check=None) -> dict | None:
        """One click round trip taken without freezing the map, or None when
        the caller should take its ordinary blocking path.

        Reads the answer exactly the way ``_request`` reads it, so a caller
        cannot tell which wait it got: the parsed body on success, and the
        parsed body on a refusal too.

        None is returned ONLY while nothing has been sent. Once the request is
        out, a wait that ends without an answer comes back as a refusal rather
        than as None, because the ordinary path would send the same click a
        second time and the far side may already be working on the first.

        A click that carries its own crop is large enough to travel gzipped
        when the server has asked for it. A body the server could not read is
        a body it never acted on, so that one is safe to send again plain, and
        it pins the session to plain bytes on the way through.
        """
        payload, packed = packed_request_body(body)
        answer, http_status, body_was_json = self._refine_once_while_drawing(
            payload, packed, auth, cancel_check)
        if answer is not None and packed and answer_refused_the_body(
                answer, http_status, body_was_json):
            _log_warning("A compressed request body was refused; sending "
                         "them plain for the rest of the session")
            note_gzip_request_refused()
            answer, _, _ = self._refine_once_while_drawing(
                body, False, auth, cancel_check)
        return answer

    def _refine_once_while_drawing(
        self, body: bytes, packed: bool, auth: dict, cancel_check=None,
    ) -> tuple[dict | None, int | None, bool]:
        """One send of an already-encoded click body, the HTTP status it came
        back with, and whether the answer carried a JSON body.

        ``(None, None, False)`` means nothing was ever sent. The last item is
        what tells a body the route never read from a route that read it and
        disagreed with it."""
        try:
            from .click_transport import ClickPostAbandoned
        except Exception:  # noqa: BLE001 -- the ordinary path still answers
            return None, None, False
        try:
            from qgis.core import QgsApplication
            from qgis.PyQt.QtCore import QThread

            app = QgsApplication.instance()
            if app is None or QThread.currentThread() is not app.thread():
                # Off the drawing thread there is nothing to keep alive, and
                # the crop hand-over already runs there.
                return None, None, False
            from .click_transport import post_and_keep_painting

            taken = post_and_keep_painting(
                self._resolve_url(self._detection_refine_url()), body, auth,
                self._submit_timeout(), _apply_redirect_policy,
                cancel_check=cancel_check, packed=packed)
        except ClickPostAbandoned as gone:
            return self._refine_abandoned(gone), None, False
        except Exception:  # noqa: BLE001 -- the ordinary path still answers
            return None, None, False
        if taken is None:
            return None, None, False
        raw, http_status, qt_error = taken
        raw_body = raw.decode("utf-8", "replace")
        if http_status is not None:
            note_server_contact()

        if qt_error != _NoError:
            if http_status is not None and http_status >= 400 and raw_body:
                try:
                    parsed = _parse_json_body(raw_body)
                    if parsed is not None:
                        return parsed, http_status, True
                except Exception:  # noqa: BLE001  # nosec B110
                    pass
            code, msg = _classify_qt_error(
                qt_error, "", http_status,
                service_reachable=server_reached_recently())
            return {"error": msg, "code": code}, http_status, False

        if http_status is not None and http_status >= 400:
            # Status only: error bodies can echo request URLs and the message
            # log must stay free of them.
            _log_warning(f"HTTP {http_status} error response")
            try:
                error_body = _parse_json_body(raw_body)
            except Exception:  # noqa: BLE001 - any unparsable body is a server error
                error_body = None
            if error_body is None:
                return ({"error": f"Server error (HTTP {http_status})",
                         "code": "SERVER_ERROR"}, http_status, False)
            if "error" in error_body:
                return error_body, http_status, True
            return ({"error": error_body.get("detail", raw_body[:200]),
                     "code": "SERVER_ERROR"}, http_status, True)
        if not raw_body:
            return {}, http_status, False
        try:
            parsed = _parse_json_body(raw_body)
        except json.JSONDecodeError:
            parsed = None
        if parsed is None:
            _log_warning(f"Invalid JSON response ({len(raw_body)} bytes)")
            return ({"error": "Invalid server response",
                     "code": "SERVER_ERROR"}, http_status, False)
        return parsed, http_status, True

    @staticmethod
    def _refine_abandoned(gone) -> dict:
        """The refusal for a click whose request went out and came back with
        nothing. A refusal and not a retry: the far side may already be working
        on it, and a second send would set two operations going on one crop."""
        if getattr(gone, "cancelled", False):
            return {"error": tr("The click was cancelled."), "code": "CANCELLED"}
        message = tr("Request timed out. Check your connection or try again.")
        try:
            from ..core.server_dials import dial_copy

            message = dial_copy("network.request_timed_out", message)
        except Exception:  # noqa: BLE001 -- served copy is best-effort  # nosec B110
            pass
        return {"error": message, "code": "TIMEOUT"}

    def submit_refine_register(self, payload: dict, auth: dict) -> dict:
        """Hand over a crop before any click, answered with the token for it.

        The click path calls this while the user is still choosing where to
        click, so the click itself sends points and not pixels. A failure here
        costs nothing: the caller falls back to sending the crop with the click.

        Nobody waits on this one, which is exactly why it needs the wall-clock
        backstop: a server that answers a byte at a time keeps Qt's own
        deadline alive for good, and the thread it holds is still running when
        unload comes to join it.
        """
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        return self._request(
            "POST",
            self._detection_refine_url() + "/register",
            auth=auth,
            body=body,
            timeout_ms=self._submit_timeout(),
            wall_clock=True,
        )

    def _detection_run_export_url(self) -> str:
        """Absolute URL for the end-of-run export summary. Same host split as
        the predict call."""
        if self.detection_direct:
            return f"{self.detection_base_url}/run-export"
        return f"{self.detection_base_url}/api/ai-segmentation/run-export"

    def post_run_export_body(self, body: bytes, auth: dict) -> dict:
        """Send the finished run's export summary (review settings + the final
        exported geometry when small enough) so the run record is complete.

        Takes the encoded body, not a dict: a dense run's geometry is
        serialized straight to text on the task thread, so re-encoding it here
        would double both the time and the peak memory.

        Additive and best-effort: callers fire-and-forget from a background
        task; a server without the route degrades to {"error", "code"} and the
        user's local export is never affected. Off-GUI-thread only.
        """
        return self._request(
            "POST", self._detection_run_export_url(), auth=auth, body=body,
            timeout_ms=_TIMEOUT_RUN_EXPORT,
        )

    def _submit_timeout(self) -> int:
        """Submit timeout. Direct mode blocks for the whole inference (no async
        fallback), so it needs a cold-start-proof window to avoid a retry that
        would double-charge; the backend route returns fast, so it keeps 45s.

        The direct window is a server dial (``network.submit_timeout_ms``),
        because it is pinned against a request timeout on the other side and
        that side must be able to move first. Read per request; the constant
        below is the fallback."""
        if not self.detection_direct:
            return _TIMEOUT_SUBMIT_DETECTION
        try:
            from ..core.detection_policy import submit_timeout_ms

            return submit_timeout_ms(_TIMEOUT_SUBMIT_DETECTION_DIRECT)
        except Exception:  # noqa: BLE001 -- a dial must never break a submit
            return _TIMEOUT_SUBMIT_DETECTION_DIRECT

    def charge_saved_object(self, auth: dict, session_id: str,
                            polygon_index: int,
                            area_m2: float | None = None,
                            polygon_wkt: str | None = None) -> dict:
        """Spend one credit for one object saved with its clicks answered here.

        Always the account host, never the direct inference one: the balance
        lives on the account side, and a session using the direct route still
        pays on this one.

        Keyed on (session, index) so the server dedupes: a Save re-sent after a
        dropped answer costs nothing more. The answer carries the account's own
        usage payload, so the caller sets the credit ring from the server's
        figure rather than a local subtraction. Off-GUI-thread only.

        ``area_m2`` and ``polygon_wkt`` are additive, optional and purely
        informational: an unencodable value falls back to the bare body, so
        the charge itself can never be lost to them.
        """
        fields: dict = {
            "session_id": str(session_id), "polygon_index": int(polygon_index)}
        if area_m2 is not None and math.isfinite(area_m2):
            fields["area_m2"] = float(area_m2)
        if polygon_wkt:
            fields["polygon_wkt"] = str(polygon_wkt)
        try:
            body = json.dumps(fields, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError):
            body = json.dumps(
                {"session_id": str(session_id),
                 "polygon_index": int(polygon_index)},
                allow_nan=False,
            ).encode("utf-8")
        return self._request(
            "POST", "/api/ai-segmentation/save-polygon", auth=auth, body=body,
            timeout_ms=_TIMEOUT_INTERACTIVE, require_body=True,
        )

    def get_usage(self, auth: dict) -> dict:
        return self._request(
            "GET", "/api/plugin/usage", auth=auth, timeout_ms=_TIMEOUT_INTERACTIVE,
            require_body=True,
        )

    def get_account(self, auth: dict) -> dict:
        return self._request(
            "GET", "/api/plugin/account", auth=auth, timeout_ms=_TIMEOUT_INTERACTIVE,
            require_body=True,
        )

    def get_account_and_usage(self, auth: dict) -> tuple[dict, dict]:
        """Fetch /account and /usage in ONE round-trip (concurrently) instead of
        two sequential blocking GETs. Halves the Account Settings load latency.

        Returns (account, usage) in the same dict shape each endpoint returns
        (an {"error", "code"} dict on failure). MUST be called off the GUI thread
        (request_many spins a nested event loop).
        """
        account, usage = self.request_many([
            {"method": "GET", "path": "/api/plugin/account",
             "auth": auth, "timeout_ms": _TIMEOUT_INTERACTIVE,
             "require_body": True},
            {"method": "GET", "path": "/api/plugin/usage",
             "auth": auth, "timeout_ms": _TIMEOUT_INTERACTIVE,
             "require_body": True},
        ])
        return account, usage

    def get_usage_with_account(self, auth: dict) -> dict:
        """/usage plus the /account envelope fields, as ONE task payload.

        The dock's balance refresh wants both: the usage payload keeps the
        wallet gauge alive on old servers, and the account rows carry the
        two-envelope fields (objects and km2) newer servers add. Answers
        {"usage": dict, "account": dict | None}. A usage error is returned
        AS the usage error dict, so GenericRequestTask routes it to failed
        with the same codes the plain get_usage path reports; an account
        error only drops the account half, never the refresh.
        MUST be called off the GUI thread (request_many nests an event loop).
        """
        try:
            account, usage = self.get_account_and_usage(auth=auth)
        finally:
            # The concurrent path lazily created a network manager with this
            # pooled thread's affinity. Keep it, so the next refresh on this
            # thread reuses the open connection instead of paying for a new one;
            # its own end drops it. Only a thread whose end cannot be watched
            # releases here, which is what every refresh used to do.
            try:
                if not self.retain_thread_nam():
                    self.release_thread_nam()
            except Exception:  # noqa: BLE001
                pass  # nosec B110
        if not isinstance(usage, dict) or "error" in usage:
            return usage if isinstance(usage, dict) else {
                "error": "Invalid server response", "code": "SERVER_ERROR"}
        account_ok: dict | None = (
            account if isinstance(account, dict) and "error" not in account
            else None)
        return {"usage": usage, "account": account_ok}

    def get_config(self, product: str) -> dict:
        """Fetch the product configuration.

        The request carries a little context (language, plugin version, QGIS
        version, platform) so the answer can be targeted. Every one of them is
        optional and omitted when it cannot be read, so the call still works
        with none of them.
        """
        from ..core.request_context import config_query

        return self._request(
            "GET", config_query(product), timeout_ms=_TIMEOUT_INTERACTIVE,
            require_body=True,
        )

    def translate_prompt(self, text: str, auth: dict | None = None) -> dict:
        """Resolve a short object prompt to its English equivalent (the server
        caches the answers). Additive endpoint: offline or not-yet-deployed
        degrades to {"error", "code"} and the caller keeps the typed text.

        Passing ``auth`` (the activation Bearer) puts the call on the per-key
        rate budget instead of the tight anonymous one; omit it and the server
        still serves the request anonymously."""
        body = json.dumps({"text": text}).encode("utf-8")
        answer = self._request(
            "POST", "/api/plugin/translate-prompt", auth=auth, body=body,
            timeout_ms=_TIMEOUT_TRANSLATE, require_body=True,
        )
        _note_skipped_tuning("prompt translation", answer)
        return answer

    def get_seg_run_plan(
        self,
        prompt: str,
        zone_area_m2: float | None,
        native_mupp: float | None,
        auth: dict | None = None,
    ) -> dict:
        """Server-computed run plan for one committed prompt: target resolution,
        recall floors, confidence and review shape defaults.

        Additive, fire-and-forget: offline or not-yet-deployed degrades to
        {"error", "code"} and the caller keeps the blob/generic path. Mirrors
        translate_prompt (same short interactive timeout, same auth pattern:
        pass the activation Bearer to land on the per-key rate budget)."""
        from ..core.request_context import plugin_version

        body = json.dumps({
            "prompt": prompt,
            "zone_area_m2": zone_area_m2,
            "native_mupp": native_mupp,
            "plugin_version": plugin_version() or "unknown",
        }).encode("utf-8")
        answer = self._request(
            "POST", "/api/plugin/seg-run-plan", auth=auth, body=body,
            timeout_ms=_TIMEOUT_TRANSLATE, require_body=True,
        )
        _note_skipped_tuning("run plan", answer)
        return answer

    # ---- Run history (Library 2.0) ----------------------------------------
    # Additive, off-GUI-thread only (the library dialog drives these from its
    # own QThread workers). Every method degrades gracefully when the endpoints
    # are not deployed yet: a 404 flows back as {"error", "code"} and the caller
    # shows the empty state instead of an error. Never logs URLs, keys, prompts.

    def get_seg_history(
        self,
        auth: dict,
        limit: int = 12,
        before: str | None = None,
        favorites_only: bool = False,
        deleted: bool = False,
    ) -> dict:
        """List the user's past cloud runs, newest first (run-grouped).

        Returns {"runs": [...], "has_more": bool} on success, or {"error",
        "code"} on failure (including a not-yet-deployed 404). ``before`` is an
        ISO cursor (the oldest run's started_at from the previous page)."""
        from urllib.parse import quote

        params = [f"limit={int(limit)}"]
        if before:
            params.append("before={}".format(quote(str(before), safe="")))
        if favorites_only:
            params.append("favorites_only=true")
        if deleted:
            params.append("deleted=true")
        path = "/api/ai-segmentation/history?" + "&".join(params)
        return self._request(
            "GET", path, auth=auth, timeout_ms=_TIMEOUT_API, require_body=True)

    def get_seg_run_detail(
        self,
        auth: dict,
        run_id: str | None = None,
        group_key: str | None = None,
    ) -> dict:
        """Per-tile detail (WITHOUT masks) for one run: the tile bboxes, CRS,
        pixel size, output dimensions and thresholds needed to rebuild it.

        Pass ``run_id`` for a real run, or ``group_key`` for a legacy day-bucket.
        Returns {"tiles": [...], ...} or {"error", "code"}."""
        from urllib.parse import quote

        if run_id:
            path = "/api/ai-segmentation/history/run?run_id={}".format(
                quote(str(run_id), safe=""))
        elif group_key:
            path = "/api/ai-segmentation/history/run?group_key={}".format(
                quote(str(group_key), safe=""))
        else:
            return {"error": "missing run identifier", "code": "CLIENT_ERROR"}
        return self._request(
            "GET", path, auth=auth, timeout_ms=_TIMEOUT_API, require_body=True)

    def set_seg_run_favorite(self, auth: dict, run_id: str, is_favorite: bool) -> dict:
        """Star / unstar a run (server-stored, cross-device)."""
        body = json.dumps({"run_id": run_id, "is_favorite": bool(is_favorite)}).encode("utf-8")
        return self._request(
            "POST", "/api/ai-segmentation/history/favorite",
            auth=auth, body=body, timeout_ms=_TIMEOUT_INTERACTIVE,
        )

    def delete_seg_run(self, auth: dict, run_id: str) -> dict:
        """Soft-delete a run (moves it to Recently deleted)."""
        body = json.dumps({"run_id": run_id}).encode("utf-8")
        return self._request(
            "POST", "/api/ai-segmentation/history/delete",
            auth=auth, body=body, timeout_ms=_TIMEOUT_INTERACTIVE,
        )

    def undelete_seg_run(self, auth: dict, run_id: str) -> dict:
        """Restore a soft-deleted run from Recently deleted."""
        body = json.dumps({"run_id": run_id}).encode("utf-8")
        return self._request(
            "POST", "/api/ai-segmentation/history/undelete",
            auth=auth, body=body, timeout_ms=_TIMEOUT_INTERACTIVE,
        )

    def fetch_run_masks(self, auth: dict, request_id: str) -> dict | list:
        """Fetch the full stored masks for one tile via the image route.

        Always uses ``stream=1`` so the server streams the bytes itself instead
        of issuing a signed redirect (a redirect would carry our Authorization
        header to a store that rejects it). Returns the parsed masks list on
        success, or {"error", "code"} on failure."""
        from urllib.parse import quote

        path = "/api/ai-segmentation/image/{}?type=masks&stream=1".format(
            quote(str(request_id), safe=""))
        return self._request(
            "GET", path, auth=auth, timeout_ms=_TIMEOUT_API, allow_list=True,
            require_body=True)

    def poll_pairing(self, code: str, timeout_ms: int = 10_000) -> dict:
        """Poll whether a pairing code has been bound to an activation key.

        Unauthenticated GET (the code itself is the bearer of trust). Returns
        {"status": "pending" | "ready" | "not_found", ...} or {"error", "code"}
        on a network/server failure (the caller retries those within a deadline).
        """
        from urllib.parse import quote
        return self._request(
            "GET",
            f"/api/plugin/pair/poll?code={quote(code, safe='')}",
            timeout_ms=timeout_ms,
            require_body=True,
        )

    def cancel_pairing(self, code: str) -> dict:
        """Retire an abandoned pairing code server-side, so a later Confirm in
        the browser shows expired instead of binding a key nobody polls for.

        Unauthenticated POST (the code itself is the bearer of trust).
        """
        body = json.dumps({"code": code, "product": "ai-segmentation"}).encode("utf-8")
        return self._request(
            "POST", "/api/plugin/pair/cancel", body=body, timeout_ms=5_000
        )

    def submit_detection(
        self,
        run_id: str,
        prompt: str,
        image_b64: str,
        tile_index: int,
        crs_authid: str,
        tile_bbox_wgs84: dict | None,
        tile_bbox_native: dict | None,
        pixel_size_m: float | None,
        max_masks: int | None,
        auth: dict,
        threshold: float | None = None,
        mask_threshold: float | None = None,
        exemplars: list[dict] | None = None,
    ) -> dict:
        """Submit one tile for cloud detection. Decrements 1 credit.

        Returns a dict with "request_id", "status", "poll_interval", "max_wait",
        "credits_remaining", "free_detections_remaining" on success.
        Returns {"error", "code"} on failure; "CREDITS_EXHAUSTED" and
        "FREE_DETECTIONS_EXHAUSTED" are sentinel codes the worker uses to
        stop submitting further tiles.

        Args:
            run_id:            UUID4 generated client-side, same for all tiles in
                               one run.
            prompt:            Text prompt describing the objects to detect.
            image_b64:         Raw base64 PNG string (no data-URI prefix).
            tile_index:        Zero-based tile index within the run.
            crs_authid:        CRS authority ID of the layer (e.g. "EPSG:32631").
            tile_bbox_wgs84:   Tile bbox in WGS84
                               {"xmin", "ymin", "xmax", "ymax"} or None.
            tile_bbox_native:  Tile bbox in the layer CRS
                               {"xmin", "ymin", "xmax", "ymax"} or None.
            pixel_size_m:      Ground sampling distance in metres or None.
            max_masks:         Maximum number of masks to return, or None for
                               the server default.
            auth:              Auth headers dict (Authorization, X-Product-ID, etc.).
            threshold:         detection-confidence cutoff (0..1), or None
                               for the server default. Lower = more detections.
            mask_threshold:    mask binarisation threshold (0..1), or None
                               for the server default. Wired end-to-end.
            exemplars:         Visual exemplars: a list of
                               {"box": [x0, y0, x1, y1], "label": 1|0} in the
                               image's pixel coords (xyxy), label 1=positive
                               (find similar) / 0=exclude. None for text-only.
        """
        body = self._build_predict_body(
            run_id, prompt, image_b64, tile_index, crs_authid,
            tile_bbox_wgs84, tile_bbox_native, pixel_size_m,
            max_masks, threshold, mask_threshold, exemplars,
        )
        return self._request(
            "POST",
            self._detection_predict_url(),
            auth=auth,
            body=body,
            timeout_ms=self._submit_timeout(),
        )

    # Additive, optional /predict request fields forwarded verbatim from a
    # worker submission dict when present. Old servers ignore unknown fields,
    # so forwarding is always backward-compatible:
    #   return_semantic  opt-in coverage-map read-out (zero-instance rescue)
    #   charge_tiles     decoupled scan-gate billing (a packed scan request
    #                    carries its whole tile group's charge; the kept
    #                    tiles' detect requests carry 0 = prepaid)
    #   mask_scale       per-run coarser mask grid (2) for classes that pass
    #                    the resolution floor; absent = the full grid
    #   plugin_version   which client build produced the run (per-run provenance)
    #   policy_rev       which policy revision produced the run (per-run
    #                    provenance)
    #   prompt_mode      "count" or "map", the client's read of the prompt kind
    #   zone_geojson     the drawn zone polygon (sent once, on the run's first
    #                    request, whatever its tile index)
    #   zone_wkt         the same zone as WGS84 WKT (sent once, same request)
    #   zone_km2         the zone's geodesic area in km2 (sent once, same
    #                    request)
    #   native_mupp      the run's ground meters per pixel (sent once, same
    #                    request)
    #   clean_image      the un-stamped tile image, present only when reference
    #                    stamps were composited into the sent tile
    _PREDICT_EXTRA_FIELDS = (
        "return_semantic", "charge_tiles", "mask_scale",
        "plugin_version", "policy_rev", "prompt_mode", "zone_geojson",
        "zone_wkt", "zone_km2", "native_mupp",
        "clean_image",
    )

    @classmethod
    def _predict_extras(cls, submission: dict) -> dict | None:
        """The submission's additive optional /predict fields, or None."""
        extra = {
            k: submission[k]
            for k in cls._PREDICT_EXTRA_FIELDS
            if submission.get(k) is not None
        }
        return extra or None

    @staticmethod
    def _build_predict_body(
        run_id, prompt, image_b64, tile_index, crs_authid,
        tile_bbox_wgs84, tile_bbox_native, pixel_size_m,
        max_masks, threshold, mask_threshold, exemplars,
        parent_tile_index=None, extra=None,
    ) -> bytes:
        """Serialize one /predict payload. Shared by submit_detection (one tile)
        and submit_detection_many (a concurrent batch). ``extra`` carries the
        additive optional fields (see _PREDICT_EXTRA_FIELDS), already filtered
        by the caller; None keeps the payload byte-identical to before."""
        payload: dict = {
            "image": image_b64,
            "run_id": run_id,
            "prompt": prompt,
            "tile_index": tile_index,
            "crs_authid": crs_authid,
        }
        if extra:
            payload.update(extra)
        if parent_tile_index is not None:
            # Re-split quadrant of an already-billed tile in the same run.
            payload["parent_tile_index"] = int(parent_tile_index)
        if tile_bbox_wgs84 is not None:
            payload["tile_bbox_wgs84"] = tile_bbox_wgs84
        if tile_bbox_native is not None:
            payload["tile_bbox_native"] = tile_bbox_native
        # Finite check, not just None: json.dumps emits a bare NaN/Infinity
        # token for non-finite floats, which is not valid JSON. A strict decoder
        # on the far side rejects the whole document, so one bad optional
        # analytics field silently costs the request its audit record. Dropping
        # the field keeps the tile itself intact.
        if pixel_size_m is not None and math.isfinite(pixel_size_m):
            payload["pixel_size_m"] = pixel_size_m
        if max_masks is not None:
            payload["max_masks"] = max_masks
        if threshold is not None:
            payload["threshold"] = threshold
        if mask_threshold is not None:
            payload["mask_threshold"] = mask_threshold
        if exemplars:
            payload["exemplars"] = exemplars
        # allow_nan=False turns a non-finite value into an error HERE instead
        # of a bare NaN/Infinity token in the body. Those tokens are not valid
        # JSON, and a strict decoder on the far side rejects the whole
        # document, which is how tiles were billed while their audit record was
        # silently dropped. Guarding the two known fields is not enough: this
        # catches every optional float, including ones added later.
        try:
            return json.dumps(payload, allow_nan=False).encode("utf-8")
        except ValueError:
            pass
        for key in ("pixel_size_m", "tile_bbox_native", "tile_bbox_wgs84"):
            payload.pop(key, None)
        _log_warning(
            "Dropped non-finite optional fields from a tile payload; the "
            "detection itself is unaffected."
        )
        try:
            return json.dumps(payload, allow_nan=False).encode("utf-8")
        except ValueError:
            # A REQUIRED field is non-finite. Send it as before rather than
            # losing the tile: this guard must never be worse than its absence.
            return json.dumps(payload).encode("utf-8")

    def submit_detection_many(
        self, submissions: list[dict], auth: dict, should_abort=None
    ) -> list[dict]:
        """Submit several tiles CONCURRENTLY; results in input order.

        Each item in `submissions` is the kwargs of submit_detection MINUS auth
        (run_id, prompt, image_b64, tile_index, crs_authid, tile_bbox_wgs84,
        tile_bbox_native, pixel_size_m, max_masks, threshold, mask_threshold,
        exemplars). Each result is the same dict shape submit_detection returns.
        This replaces N serial uploads (the dominant cost once polling went
        concurrent) with one batched round-trip. Off-GUI-thread only.
        """
        specs = []
        for s in submissions:
            body = self._build_predict_body(
                s["run_id"], s["prompt"], s["image_b64"], s["tile_index"],
                s["crs_authid"], s.get("tile_bbox_wgs84"), s.get("tile_bbox_native"),
                s.get("pixel_size_m"), s.get("max_masks"), s.get("threshold"),
                s.get("mask_threshold"), s.get("exemplars"),
                s.get("parent_tile_index"), self._predict_extras(s),
            )
            specs.append({
                "method": "POST",
                "path": self._detection_predict_url(),
                "auth": auth,
                "body": body,
                "timeout_ms": self._submit_timeout(),
            })
        return self.request_many(specs, should_abort=should_abort)

    def post_detection_async(self, nam, submission: dict, auth: dict):
        """Fire ONE /predict POST WITHOUT blocking; return the QNetworkReply.

        The caller drives completion itself (processEvents + reply.isFinished())
        and parses the finished reply with parse_reply(). This is what lets the
        auto worker run a CONTINUOUS sliding window on the synchronous direct
        endpoint: it keeps N posts in flight and, the instant any one returns,
        converts + emits that tile and fires the next - so the service never idles
        between barrier batches and tiles stream in one-by-one. `nam` is the
        caller-thread QgsNetworkAccessManager (must be created on the same thread
        that drives the event loop). Off-GUI-thread only.
        """
        from qgis.PyQt.QtCore import QByteArray

        body = self._build_predict_body(
            submission["run_id"], submission["prompt"], submission["image_b64"],
            submission["tile_index"], submission["crs_authid"],
            submission.get("tile_bbox_wgs84"), submission.get("tile_bbox_native"),
            submission.get("pixel_size_m"), submission.get("max_masks"),
            submission.get("threshold"), submission.get("mask_threshold"),
            submission.get("exemplars"), submission.get("parent_tile_index"),
            self._predict_extras(submission),
        )
        body, packed = packed_request_body(body)
        req = self._make_qnetwork_request(
            auth, self._submit_timeout(), self._detection_predict_url(), packed
        )
        # Route through our PRIVATE manager, not the caller's `nam`: a slow
        # /predict (cold start / busy service) must not trip QGIS's global network
        # timeout warning (see _predict_nam). The reply still drives on the
        # caller's event loop; only the manager differs. `nam` is kept in the
        # signature for backward compatibility with the worker call site.
        del nam
        return self._predict_nam().post(req, QByteArray(body))

    def parse_reply(self, reply) -> dict:
        """Public wrapper around the finished-reply parser, for callers that
        drive their own async replies (see post_detection_async)."""
        return self._parse_reply(reply)

    def warmup(self, auth: dict) -> bool:
        """Best-effort cold-start ping for the cloud detection backend.

        Fired when the user enters the Automatic flow so the idle
        backend instance is already spinning up by the time they hit Detect. The
        server pings the service and returns {"ok": true|false} (always 200 on
        a reachable server). Never raises: returns True only when the server
        confirms ok, False on any error (network, auth, timeout, server). Runs
        off the GUI thread via a hidden GenericRequestTask; never on the main thread.

        Args:
            auth: Auth headers dict (Authorization, X-Product-ID, etc.).
        """
        try:
            if self.detection_direct:
                # Direct mode: ping the detection endpoint's open /health probe
                # (GET, no body). It answers {"status": "ok"} once the model is
                # loaded, warming the idle instance.
                result = self._request(
                    "GET",
                    f"{self.detection_base_url}/health",
                    auth=auth,
                    timeout_ms=_TIMEOUT_WARMUP,
                )
                return result.get("status") == "ok"
            result = self._request(
                "POST",
                "/api/ai-segmentation/warmup",
                auth=auth,
                body=b"{}",
                timeout_ms=_TIMEOUT_WARMUP,
            )
            return result.get("ok") is True
        except Exception:
            return False

    def get_detection_status(self, request_id: str, auth: dict) -> dict:
        """Poll cloud detection status.

        Returns {"status": "pending", "retry_after": N} while waiting.
        Returns {"status": "completed", "masks": [...],
                 "credits_remaining": N, "width": W, "height": H} when done.
        Returns {"status": "failed", "error": "..."} on failure.
        Returns {"error", "code"} on network error.

        Note: width/height in the completed response may be null. The caller
        must fall back to its own tile dimensions for RLE decoding.

        Args:
            request_id: The request_id returned by submit_detection.
            auth:       Auth headers dict.
        """
        from urllib.parse import quote

        path = f"/api/ai-segmentation/predict/status?request_id={quote(request_id, safe='')}"
        return self._request(
            "GET", path, auth=auth, timeout_ms=_TIMEOUT_POLL_DETECTION
        )

    def get_detection_status_many(
        self, request_ids: list[str], auth: dict, should_abort=None
    ) -> list[dict]:
        """Poll several detection requests CONCURRENTLY; results in input order.

        Each result has the SAME shape as get_detection_status. Unlike calling
        get_detection_status N times (N serial blocking round-trips), this fires
        all polls at once, so a cycle costs ~1 RTT instead of N. The auto worker
        uses this to stop under-driving the cloud backend. Off-GUI-thread only.
        """
        from urllib.parse import quote

        specs = [
            {
                "method": "GET",
                "path": f"/api/ai-segmentation/predict/status?request_id={quote(rid, safe='')}",
                "auth": auth,
                "timeout_ms": _TIMEOUT_POLL_DETECTION,
            }
            for rid in request_ids
        ]
        return self.request_many(specs, should_abort=should_abort)

    def _make_qnetwork_request(self, auth: dict | None, timeout_ms: int, path: str,
                               packed: bool = False) -> QNetworkRequest:
        """Build a QNetworkRequest with the same headers/timeout/redirect policy
        _request applies (so the concurrent path behaves identically).

        ``packed`` says the body about to be posted is gzipped, and is the only
        thing that declares the encoding."""
        req = QNetworkRequest(QUrl(self._resolve_url(path)))
        req.setRawHeader(b"Content-Type", b"application/json")
        if packed:
            # Qt takes Content-Length from the byte array it is handed, so the
            # encoding is the only thing left to declare.
            req.setRawHeader(b"Content-Encoding", b"gzip")
        if hasattr(req, "setTransferTimeout"):
            req.setTransferTimeout(timeout_ms)
        _apply_redirect_policy(req, bool(auth))
        if auth:
            for key, value in auth.items():
                req.setRawHeader(key.encode("utf-8"), value.encode("utf-8"))
        return req

    def request_many(self, specs: list[dict], should_abort=None,
                     retry_reads: bool = True) -> list[dict]:
        """Execute several requests CONCURRENTLY, returning results in input order.

        Each spec is a dict: {"method": "GET"|"POST", "path": str, "auth": dict
        (optional), "body": bytes (optional), "timeout_ms": int (optional),
        "require_body": bool (optional, see _request)}.
        result[i] is the SAME dict shape _request returns for spec i.

        Where _request issues one blocking QgsBlockingNetworkRequest at a time,
        this fires every request on the thread-local QgsNetworkAccessManager and
        waits on ONE nested QEventLoop, so N requests cost ~1 round-trip instead
        of N. It still flows through QGIS's network stack (proxy / SSL / auth /
        Network Logger). MUST be called OFF the GUI thread: it spins a nested
        event loop, which would re-enter the UI on the main thread.

        should_abort: optional predicate polled ~4x/s on the SAME thread that
        drives the loop. When it returns True the loop quits at once and the
        pending replies are aborted, so a caller that sets a stop flag (the auto
        worker's request_stop) unblocks within ~0.25s instead of waiting out the
        full submit timeout (up to ~115s in direct mode). This is the shutdown
        guard: on unload the main thread joins the worker with wait(5000), and
        without this the worker is stuck mid-POST past that join, gets parked as a
        live QThread, and aborts QGIS at process teardown ("Destroyed while thread
        is still running"). The timer lives on this thread's spinning loop, so the
        whole abort stays thread-affine (no cross-thread reply.abort()).

        A GET that ends in a timeout, a dropped link or a server fault is sent
        once more, the same rule the one-at-a-time path follows: a batched read
        is idempotent, and one lost packet should not cost the user their
        balance. ``retry_reads`` is how that second batch says it is the last.
        A POST is never re-sent: the far side may already have acted on it.
        """
        from qgis.PyQt.QtCore import QEventLoop, QTimer

        if not specs:
            return []

        nam = self._predict_nam()
        loop = QEventLoop()
        replies = []
        max_timeout = _TIMEOUT_API
        for spec in specs:
            timeout_ms = spec.get("timeout_ms", _TIMEOUT_API)
            max_timeout = max(max_timeout, timeout_ms)
            body = spec.get("body") or b""
            packed = False
            if spec.get("method") == "POST" and body:
                body, packed = packed_request_body(body)
            req = self._make_qnetwork_request(
                spec.get("auth"), timeout_ms, spec["path"], packed)
            if spec.get("method") == "POST":
                reply = nam.post(req, QByteArray(body))
            else:
                reply = nam.get(req)
            replies.append(reply)

        remaining = [len(replies)]

        def _on_one_finished():
            remaining[0] -= 1
            if remaining[0] <= 0:
                loop.quit()

        for reply in replies:
            if reply.isFinished():
                # Extremely unlikely for a network reply, but if one resolved
                # before we reached the loop, account for it so we never hang.
                remaining[0] -= 1
            else:
                reply.finished.connect(_on_one_finished)

        if remaining[0] <= 0:
            results = [
                self._parse_reply(r, require_body=bool(s.get("require_body")))
                for s, r in zip(specs, replies)
            ]
            for r in replies:
                r.deleteLater()
            return results

        # Safety net: setTransferTimeout already bounds each reply, but a wedged
        # NAM could still stall the loop, so cap the whole batch defensively. A
        # stoppable timer (not singleShot) so it is cancelled right after exec()
        # instead of keeping the QEventLoop alive past this call and firing quit()
        # into a later batch's loop on this same thread.
        safety_timer = QTimer()
        safety_timer.setSingleShot(True)
        safety_timer.setInterval(max_timeout + 5_000)
        safety_timer.timeout.connect(loop.quit)
        safety_timer.start()
        # Cancellation net: poll the stop predicate on this loop's own thread so a
        # request_stop() during a long submit quits the loop within ~0.25s (the
        # shutdown-crash guard; see the docstring). Kept in a local so it stays
        # alive for the duration of exec() and is stopped right after.

        def _already_aborting():
            if should_abort is None:
                return False
            try:
                return bool(should_abort())
            except Exception:  # noqa: BLE001 - predicate must never crash the loop
                return False

        abort_timer = None
        if should_abort is not None:
            abort_timer = QTimer()
            abort_timer.setInterval(250)
            abort_timer.timeout.connect(
                lambda: loop.quit() if _already_aborting() else None)
            abort_timer.start()
        # Skip the loop entirely if the stop landed before we blocked (quit()
        # is a no-op before exec(), so guard here instead).
        if not _already_aborting():
            loop.exec()
        if abort_timer is not None:
            abort_timer.stop()
        safety_timer.stop()

        results = []
        statuses: list[int | None] = []
        for spec, reply in zip(specs, replies):
            # A reply whose C++ side is already gone raises RuntimeError on
            # every call below. It happens when the manager is torn down while
            # the loop above is spinning, on plugin unload or on a project
            # close, and one dead reply used to take the whole batch with it.
            # Its outcome is unknown, so report it as the transient the caller
            # already re-polls on and keep reading the others.
            try:
                results.append(self._parse_reply(
                    reply, require_body=bool(spec.get("require_body"))))
                statuses.append(_http_status_of(reply))
            except RuntimeError:
                results.append({
                    "error": tr("Request timed out. Check your connection or try again."),
                    "code": "TIMEOUT"})
                statuses.append(None)
                continue
            try:
                if not reply.isFinished():
                    reply.abort()  # release a still-pending socket on the timeout path
                reply.deleteLater()
            except RuntimeError:
                pass  # nosec B110 - already destroyed, nothing left to release
        if retry_reads and not _already_aborting():
            again = [i for i, spec in enumerate(specs)
                     if spec.get("method") != "POST"
                     and _worth_asking_again(results[i], statuses[i])]
            if again:
                time.sleep(_retry_pause_s())
                if not _already_aborting():
                    second = self.request_many(
                        [specs[i] for i in again], should_abort=should_abort,
                        retry_reads=False)
                    for slot, answer in zip(again, second):
                        results[slot] = answer
        return results

    def _predict_nam(self):
        """Plugin-owned plain QNetworkAccessManager for the concurrent /predict
        path. Using a PRIVATE manager (not QgsNetworkAccessManager.instance())
        stops QGIS's global network-timeout handler from popping a scary
        message-bar warning ("Network request ... timed out") on the long,
        already-retried /predict calls: cold start / a busy service is expected and
        the worker retries silently, so the user should not see a QGIS error.
        Our own setTransferTimeout + the worker retry still bound each request.
        Proxy mirrors QGIS's effective proxy so corporate-proxy users keep
        working (see _qgis_effective_proxy); auth is on the request headers
        (not QgsAuthManager). SSL trust parity is restored too (see
        _on_predict_ssl_errors): a private manager does not inherit the
        QGIS-stored per-host SSL exceptions, so without this a corporate-MITM /
        self-signed-CA user would pass the blocking GETs but TLS-fail every tile
        POST. Created lazily PER THREAD and reused there: QNetworkAccessManager
        has thread affinity, and one client instance can be driven from more
        than one thread (the account dialog's pooled QgsTasks land a Retry on
        whichever pool thread is free), so a single cached manager would be
        touched cross-thread and misbehave or crash.

        The cache is module-level, not per instance: callers build a client per
        request, and a per-instance cache made every one of them pay a fresh
        connection and TLS handshake."""
        from qgis.PyQt.QtCore import QThread

        nams = _THREAD_NAMS
        thread = QThread.currentThread()
        nam = nams.get(thread)
        if nam is not None and not _qobject_alive(nam):
            # The C++ manager went away while this cache still pointed at it
            # (see acquire_predict_nam). Posting on the dead wrapper raises
            # RuntimeError, and so does every reply already made by it, so
            # rebuild instead and let the caller retry its request.
            nam = None
        if nam is None:
            nam = self._new_private_nam()
            nams[thread] = nam
        return nam

    def acquire_predict_nam(self):
        """Return this thread's private manager for a caller that will keep a
        strong reference to it for as long as its replies must stay readable.

        Every reply is a CHILD of the manager, so destroying the manager
        destroys every reply still in flight at once, and the next read of one
        raises RuntimeError on a dead wrapper. The manager is Python-owned with
        no parent, so the per-thread cache above is otherwise its only owner: a
        pop, an overwrite, or a cyclic-GC pass on the cache would take a whole
        run's in-flight tiles down together. A caller holding the returned
        object for its run cannot be caught by that.
        """
        return self._predict_nam()

    def _new_private_nam(self):
        """Create a private QNetworkAccessManager configured like the QGIS one
        (proxy mirror + stored SSL exceptions honoured). The manager takes the
        affinity of the creating thread; destroy it on that same thread."""
        from qgis.PyQt.QtNetwork import QNetworkAccessManager

        nam = QNetworkAccessManager()
        try:
            nam.setProxy(self._qgis_effective_proxy())
        except (RuntimeError, AttributeError) as err:
            # Silence here reads as "no proxy needed". On a network that drops
            # direct egress it means every request fails with nothing in the
            # log pointing at the proxy that was never applied.
            _log_warning(f"Proxy mirror failed, continuing direct: {type(err).__name__}")
        # Honour QGIS-configured SSL exceptions on this private manager so it
        # trusts exactly what the blocking QGIS path trusts (corporate CA /
        # self-signed). Custom CAs added to QGIS land in the process-global
        # default QSslConfiguration and are already trusted here; the gap is
        # the per-host EXCEPTIONS, applied via the sslErrors handler below.
        try:
            nam.sslErrors.connect(self._on_predict_ssl_errors)
        except (RuntimeError, AttributeError):
            pass
        # A proxy that answers 407 asks the MANAGER for credentials. The QGIS
        # one answers from its own settings or prompts; a private manager is on
        # its own, so without this every request behind an authenticating
        # corporate proxy fails and the user is told to check a link that works.
        try:
            nam.proxyAuthenticationRequired.connect(self._on_proxy_auth_required)
        except (RuntimeError, AttributeError):
            pass
        # The two handlers above are bound methods of this client, and Qt does
        # not keep the client alive for them. The manager now outlives the
        # client that built it (the cache is per thread, not per instance), so
        # it pins its own maker: without this the SSL-exception and proxy-
        # credential handlers would fall away the moment that client is
        # collected, and a corporate-CA or authenticating-proxy user would
        # start failing on a manager that used to work.
        try:
            nam._terralab_owner = self
        except (AttributeError, RuntimeError):
            pass
        return nam

    def _on_proxy_auth_required(self, proxy, authenticator) -> None:
        """Answer a proxy's credential challenge from the QGIS proxy settings.

        Answered at most once per challenge: Qt leaves the rejected user on the
        authenticator when it re-asks, and replaying a refused pair makes it
        loop instead of failing. Leaving the authenticator empty ends the
        request with Qt's proxy error code, which the classifier already turns
        into "check your proxy settings" rather than "check your connection".
        Nothing here reaches a log: a proxy user name identifies the person.
        """
        try:
            if authenticator is None or authenticator.user():
                return
            from ..core.proxy_credentials import qgis_proxy_credentials

            user, password = qgis_proxy_credentials()
            if not user:
                # Once. A run works through hundreds of tiles and every one of
                # them would write this same line into the QGIS log.
                if not getattr(self, "_proxy_credentials_warned", False):
                    self._proxy_credentials_warned = True
                    _log_warning(
                        "The proxy asked for a user name and QGIS has none stored. "
                        "Set it in Settings > Options > Network."
                    )
                return
            authenticator.setUser(user)
            authenticator.setPassword(password)
        except Exception as err:  # noqa: BLE001 - a failed answer must not kill the reply
            _log_warning(f"Proxy credential lookup failed: {type(err).__name__}")

    def release_thread_nam(self) -> None:
        """Destroy the calling thread's cached private manager ON this thread.

        QNetworkAccessManager is a QObject with thread affinity: the detection
        worker creates one on its own thread, and if the last reference were
        only dropped later by main-thread GC of the worker, the C++ object
        would be destroyed from the wrong thread (Qt logs timer warnings and
        can crash during socket teardown, worst on Windows). The worker calls
        this at the end of run() while its thread is still alive; the account
        loader calls it at the end of its pooled-task body. Dropping the last
        Python reference here destroys the C++ object immediately, on the
        thread that owns it. Safe no-op when this thread has no manager."""
        if not _THREAD_NAMS:
            return
        from qgis.PyQt.QtCore import QThread

        thread = QThread.currentThread()
        _THREAD_NAMS.pop(thread, None)
        _THREAD_NAM_WATCHED.discard(thread)

    def retain_thread_nam(self) -> bool:
        """Keep this thread's private manager alive for the next request on it.

        A background call that repeats on a timer used to destroy its manager
        as it finished, which threw away the open connection: every refresh
        paid a name lookup, a connection and a TLS handshake again. Retaining
        it lets the next one start on a live socket.

        The manager still has to die on the thread that owns it, so the
        thread's own end is armed to drop it, once per thread. Returns False
        when that could not be armed; the caller then releases as before,
        rather than leave a manager nothing will destroy on the right thread.
        """
        from qgis.PyQt.QtCore import QThread

        thread = QThread.currentThread()
        if thread not in _THREAD_NAMS or thread in _THREAD_NAM_WATCHED:
            return True
        try:
            # Connected from this thread, so the slot runs here as the thread
            # winds down and the manager is destroyed where it belongs.
            thread.finished.connect(_drop_thread_nam)
        except (RuntimeError, AttributeError, TypeError):
            return False
        _THREAD_NAM_WATCHED.add(thread)
        return True

    def _qgis_effective_proxy(self):
        """Best mirror of QGIS's proxy for a private manager.

        A plain copy of QgsNetworkAccessManager.instance().proxy() misses the
        manual-proxy case: QGIS applies the proxy configured in its Options
        through a proxy FACTORY (with per-URL excludes), so .proxy() can stay
        DefaultProxy while the effective proxy lives only in the factory, and
        a private manager copying it would connect directly and fail behind a
        mandatory corporate proxy. Prefer the configured fallback proxy when
        one is set; otherwise keep .proxy() (DefaultProxy resolves through the
        app-level system-proxy configuration, which a private manager already
        inherits). Per-URL proxy excludes are deliberately not mirrored: they
        exist to BYPASS the proxy for specific hosts, and excluding this API
        host is not a supported setup."""
        from qgis.core import QgsNetworkAccessManager
        from qgis.PyQt.QtNetwork import QNetworkProxy

        # Never call setupDefaultProxyAndCache here. instance() already runs it
        # once per thread, and the body is not idempotent: it holds thirteen
        # connects with no uniqueness flag, so a second run doubles every
        # signal QGIS relays from a worker manager to the main one, each on a
        # blocking connection. On the main thread it would go further and
        # replace the handlers QGIS installs for the certificate and
        # credentials prompts, for the whole session.
        gnam = QgsNetworkAccessManager.instance()
        fallback = getattr(gnam, "fallbackProxy", None)
        if callable(fallback):
            try:
                proxy = fallback()
                if proxy is not None and proxy.type() not in (
                    QNetworkProxy.ProxyType.NoProxy,
                    QNetworkProxy.ProxyType.DefaultProxy,
                ):
                    return proxy
            except Exception as err:  # noqa: BLE001 - proxy mirror is best-effort
                _log_warning(
                    f"Reading the QGIS fallback proxy failed: {type(err).__name__}")
        return gnam.proxy()

    def _on_predict_ssl_errors(self, reply, errors) -> None:
        """Ignore ONLY the SSL errors the user chose to trust for this host in
        QGIS (Settings > Options > Network, or the "add exception" prompt).

        Mirrors what the blocking QgsBlockingNetworkRequest path already does:
        the private /predict manager does not inherit QGIS's stored SSL
        exceptions, so a self-signed-CA / corporate-MITM user would TLS-fail
        every tile while the lightweight GETs succeed. Consult QGIS's stored
        per-host exception and ignore only the exact errors it lists; anything
        else stands (secure default = current behaviour). Fully guarded: any
        missing/older API degrades to "do not ignore", i.e. today's behaviour,
        so it can never loosen trust it should not."""
        try:
            from qgis.core import QgsApplication

            url = reply.url()
            hostport = f"{url.host()}:{url.port(443)}"
            auth_mgr = QgsApplication.authManager()
            if auth_mgr is None:
                return
            config = auth_mgr.sslCertCustomConfigByHost(hostport)
            if config is None or config.isNull():
                return
            allowed = set(config.sslIgnoredErrorEnums())
            if not allowed:
                return
            # Only ignore when EVERY reported error is one the user pre-approved
            # for this host; a single unexpected error fails the handshake.
            if all(err.error() in allowed for err in errors):
                reply.ignoreSslErrors(errors)
        except Exception:  # noqa: BLE001 - parity is best-effort, never break the reply
            pass  # nosec B110

    def _parse_reply(self, reply, require_body: bool = False) -> dict:
        """Turn a finished QNetworkReply into the same dict _request returns.

        ``require_body`` has the same meaning as in _request: on a route whose
        whole point is the payload, a 2xx we cannot read is a failure.

        A reply to a body that went out gzipped and came back unread pins the
        session to plain bytes and is reported as the transient it is. Nobody
        waits on these replies, so the run loop above owns the retry, and by
        the time it re-sends the tile nothing is packed any more: that re-send
        IS the one uncompressed retry.
        """
        answer, body_was_json = self._parse_reply_once(reply, require_body)
        if _reply_was_packed(reply) and answer_refused_the_body(
                answer, _http_status_of(reply), body_was_json):
            _log_warning("A compressed request body was refused; sending "
                         "them plain for the rest of the session")
            note_gzip_request_refused()
            if isinstance(answer, dict):
                answer = dict(answer)
                answer["code"] = "SERVER_ERROR"
        return answer

    def _parse_reply_once(self, reply,
                          require_body: bool = False) -> tuple[dict, bool]:
        """The reply read as it stands, with no view on how the body travelled.

        Returns the answer and whether it came out of a JSON body, which is
        what tells a body the route never read from a route that read it and
        disagreed with it.
        """
        if not reply.isFinished():
            # Only reachable via the batch safety-net timeout (each reply also has
            # its own setTransferTimeout that fires first). Report it as a
            # transient TIMEOUT so the caller re-polls instead of dropping the tile.
            return ({"error": tr("Request timed out. Check your connection or try again."),
                     "code": "TIMEOUT"}, False)
        qt_error = reply.error()
        http_status = _http_status_of(reply)
        raw_body = bytes(reply.readAll()).decode("utf-8", "replace")
        if qt_error == _NoError or http_status is not None:
            note_server_contact()

        if qt_error != _NoError:
            # A 4xx/5xx still carries our JSON error body; prefer it over the
            # generic network classification so the worker sees the real code.
            if http_status is not None and http_status >= 400 and raw_body:
                try:
                    parsed = _parse_json_body(raw_body)
                    if parsed is not None:
                        return parsed, True
                except Exception:
                    pass  # nosec B110
            code, msg = _classify_qt_error(
                qt_error, reply.errorString(), http_status,
                service_reachable=server_reached_recently(),
            )
            return {"error": msg, "code": code}, False

        if http_status is not None and http_status >= 400:
            # Status only: error bodies can echo request URLs/details and the
            # message log must stay free of them (production-safe logging).
            _log_warning(f"HTTP {http_status} error response")
            try:
                error_body = _parse_json_body(raw_body)
            except Exception:  # noqa: BLE001 - any unparsable body is a server error
                error_body = None
            if error_body is None:
                return ({"error": f"Server error (HTTP {http_status})",
                         "code": "SERVER_ERROR"}, False)
            if "error" in error_body:
                return error_body, True
            return ({
                "error": error_body.get("detail", raw_body[:200]),
                "code": "SERVER_ERROR",
            }, True)

        if not raw_body:
            if require_body:
                _log_warning("Empty body on a route that must carry one")
                return _unreadable_answer(), False
            return {}, False
        try:
            parsed = _parse_json_body(raw_body)
        except json.JSONDecodeError:
            parsed = None
        if parsed is None:
            _log_warning(f"Invalid JSON response ({len(raw_body)} bytes)")
            if require_body:
                return _unreadable_answer(), False
            return ({"error": "Invalid server response",
                     "code": "SERVER_ERROR"}, False)
        return parsed, True

    def _request(
        self,
        method: str,
        path: str,
        auth: dict | None = None,
        body: bytes | None = None,
        timeout_ms: int = _TIMEOUT_API,
        allow_list: bool = False,
        require_body: bool = False,
        wall_clock: bool = False,
    ) -> dict | list:
        """One blocking round trip, returning the parsed body or {"error", "code"}.

        ``require_body`` marks a route whose whole point is the payload (usage,
        account, config, history, pairing). On those, a 2xx that carries nothing
        we can read is a failure, not an empty success: a transparent proxy that
        strips the body, or a captive portal answering every URL with its own
        sign-in page, would otherwise be handed to callers as a valid empty
        answer and rendered as a blank plan and no credits. Left off, an
        unreadable 2xx keeps the older behaviour, so nothing on the detection
        hot path changes shape.

        ``wall_clock`` bounds the WHOLE round trip, not each transfer. Qt's own
        deadline restarts on every byte received, so a server that answers one
        byte at a time holds the calling thread for as long as it likes, past
        the join unload gives it. Set it on any route a background thread takes
        and nobody waits on.

        A large POST body travels gzipped when the server has asked for it (see
        ``packed_request_body``). A server that cannot read that form refuses a
        request it would have answered plain, so the first refusal pins the
        session back to plain bytes and sends the same request again, once.
        With the dial off nothing is packed, no header is added, and this is
        the single round trip it has always been.

        A GET that ends in a timeout, a dropped link or a server fault is sent
        once more. On a link that drops one request in ten, one lost packet
        should not cost the user their credit balance or their configuration,
        and asking twice for the same thing changes nothing on the far side. A
        POST is never re-sent here: the far side may already have acted on the
        first one."""
        payload, packed = body, False
        # Only the inference service inflates a gzipped body, and it is the
        # only host reached through an absolute URL here. The website routes
        # take a relative path and read a packed body as an invalid one.
        if method == "POST" and body and path.startswith(("http://", "https://")):
            payload, packed = packed_request_body(body)
        answer, http_status, body_was_json = self._request_once(
            method, path, auth, payload, packed, timeout_ms, allow_list,
            require_body, wall_clock)
        if packed and answer_refused_the_body(answer, http_status,
                                              body_was_json):
            _log_warning("A compressed request body was refused; sending "
                         "them plain for the rest of the session")
            note_gzip_request_refused()
            answer, http_status, _ = self._request_once(
                method, path, auth, body, False, timeout_ms, allow_list,
                require_body, wall_clock)
        if method == "GET" and _worth_asking_again(answer, http_status):
            time.sleep(_retry_pause_s())
            answer, _, _ = self._request_once(
                method, path, auth, payload, packed, timeout_ms, allow_list,
                require_body, wall_clock)
        return answer

    def _request_once(
        self,
        method: str,
        path: str,
        auth: dict | None,
        body: bytes | None,
        packed: bool,
        timeout_ms: int,
        allow_list: bool,
        require_body: bool,
        wall_clock: bool,
    ) -> tuple[dict | list, int | None, bool]:
        """One send of an already-encoded body, the HTTP status it came back
        with, and whether the answer carried a JSON body.

        ``packed`` says the bytes are gzipped, and is the only thing that puts
        a ``Content-Encoding`` header on the request. The last item is what
        tells a body the route never read from a route that read it and
        disagreed with it."""
        url = self._resolve_url(path)
        req = QNetworkRequest(QUrl(url))
        req.setRawHeader(b"Content-Type", b"application/json")
        if packed:
            # Qt takes Content-Length from the byte array itself, so the
            # encoding is the only thing left to declare.
            req.setRawHeader(b"Content-Encoding", b"gzip")
        # Hard per-request deadline so a wedged connection can never hang the
        # task forever. setTransferTimeout arrived in Qt 5.15, and the oldest
        # QGIS this plugin runs on ships against older Qt on the long-term
        # Linux releases, so ask before calling it: the wall-clock guard and
        # the caller's own retry still bound the request without it.
        if hasattr(req, "setTransferTimeout"):
            req.setTransferTimeout(timeout_ms)
        _apply_redirect_policy(req, bool(auth))
        if auth:
            for key, value in auth.items():
                req.setRawHeader(key.encode("utf-8"), value.encode("utf-8"))

        blocker = QgsBlockingNetworkRequest()
        guard = _WallClockGuard(blocker, timeout_ms) if wall_clock else None
        try:
            if method == "GET":
                err = blocker.get(req, forceRefresh=True)
            elif method == "POST":
                payload = QByteArray(body) if body else QByteArray()
                err = blocker.post(req, payload)
            else:
                return ({"error": f"Unsupported method: {method}",
                         "code": "CLIENT_ERROR"}, None, False)
        finally:
            if guard is not None:
                guard.stop()

        if err != QgsBlockingNetworkRequest.ErrorCode.NoError:
            reply = blocker.reply()
            http_status = _http_status_of(reply)
            if http_status is not None:
                note_server_contact()
            if reply is not None and http_status is not None and http_status >= 400:
                # A misbehaving gateway/proxy can return a non-UTF-8 body; decode
                # leniently so a garbled error page never crashes the worker loop.
                raw = bytes(reply.content()).decode("utf-8", "replace")
                if raw:
                    try:
                        parsed = _parse_json_body(raw)
                        if parsed is not None:
                            return parsed, http_status, True
                    except Exception:
                        pass  # nosec B110
            code, msg = _classify_network_error(blocker)
            return {"error": msg, "code": code}, http_status, False

        reply = blocker.reply()
        http_status = _http_status_of(reply)
        raw_body = bytes(reply.content()).decode("utf-8", "replace")
        note_server_contact()

        if http_status is not None and http_status >= 400:
            # Status only: error bodies can echo request URLs/details and the
            # message log must stay free of them (production-safe logging).
            _log_warning(f"HTTP {http_status} error response")
            try:
                error_body = _parse_json_body(raw_body)
            except Exception:  # noqa: BLE001 - any unparsable body is a server error
                error_body = None
            if error_body is None:
                return ({"error": f"Server error (HTTP {http_status})",
                         "code": "SERVER_ERROR"}, http_status, False)
            if "error" in error_body:
                return error_body, http_status, True
            return ({
                "error": error_body.get("detail", raw_body[:200]),
                "code": "SERVER_ERROR",
            }, http_status, True)

        if not raw_body:
            if require_body:
                _log_warning("Empty body on a route that must carry one")
                return _unreadable_answer(), http_status, False
            return {}, http_status, False
        try:
            # allow_list defaults False so a stray array fails to parse here
            # rather than reach a caller typed -> dict; fetch_run_masks is
            # the one route that opts in (the stored-masks archive).
            parsed = _parse_json_body(raw_body, allow_list=allow_list)
        except json.JSONDecodeError:
            parsed = None
        if parsed is None:
            _log_warning(f"Invalid JSON response ({len(raw_body)} bytes)")
            if require_body:
                return _unreadable_answer(), http_status, False
            return ({"error": "Invalid server response",
                     "code": "SERVER_ERROR"}, http_status, False)
        return parsed, http_status, True
