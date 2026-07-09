from __future__ import annotations

import json

from qgis.core import Qgis, QgsBlockingNetworkRequest, QgsMessageLog
from qgis.PyQt.QtCore import QByteArray, QUrl
from qgis.PyQt.QtNetwork import QNetworkReply, QNetworkRequest

from ..core.i18n import tr

_TIMEOUT_API = 30_000
_TIMEOUT_INTERACTIVE = 10_000        # ms: lightweight config/usage/account GETs;
#                                      short so a bad connection surfaces fast.
_TIMEOUT_SUBMIT_DETECTION = 45_000   # ms: submit is small (base64 PNG inline)
# Direct mode: the submit BLOCKS for the whole inference (no async "pending"
# fallback like the backend route had), so a cold-start tile (idle backend
# model load, ~18-60s) must not trip the client timeout - that would retry and
# DOUBLE-CHARGE a tile the server already billed + computed. 110s waits a cold
# start out while staying under the inference service request timeout (120s).
_TIMEOUT_SUBMIT_DETECTION_DIRECT = 110_000
_TIMEOUT_POLL_DETECTION = 15_000     # ms: poll is GET with tiny JSON response
_TIMEOUT_WARMUP = 5_000              # ms: warmup is a tiny best-effort ping
_TIMEOUT_TRANSLATE = 6_000           # ms: prompt translation blocks a Detect
# ms: final-output upload can carry a few MB of geometry; background task only.
_TIMEOUT_RUN_EXPORT = 60_000
#                                      click, so fail fast and run as typed

# Qt6 (QGIS 4) uses scoped enums; Qt5 uses flat. Resolve at import time.
_NE = getattr(QNetworkReply, "NetworkError", QNetworkReply)
_HostNotFound = getattr(_NE, "HostNotFoundError", getattr(QNetworkReply, "HostNotFoundError", None))
_ConnRefused = getattr(_NE, "ConnectionRefusedError", getattr(QNetworkReply, "ConnectionRefusedError", None))
_Timeout = getattr(_NE, "TimeoutError", getattr(QNetworkReply, "TimeoutError", None))
# What Qt actually emits when a request's setTransferTimeout deadline expires:
# OperationCanceledError, NOT TimeoutError. A cold-starting / busy inference
# service that answers slower than the submit deadline lands here, so it must
# be read as "the service is warming up" (a transient, retry-worthy wait), not
# as the user's connection dying. reply.abort() on our own cancel path also
# raises this, but the cancel path never classifies+retries an aborted reply.
_OpCanceled = getattr(_NE, "OperationCanceledError", getattr(QNetworkReply, "OperationCanceledError", None))
_SslFailed = getattr(_NE, "SslHandshakeFailedError", getattr(QNetworkReply, "SslHandshakeFailedError", None))
_ContentDenied = getattr(_NE, "ContentAccessDenied", getattr(QNetworkReply, "ContentAccessDenied", None))
_AuthRequired = getattr(_NE, "AuthenticationRequiredError", getattr(QNetworkReply, "AuthenticationRequiredError", None))
_UnknownNetwork = getattr(_NE, "UnknownNetworkError", getattr(QNetworkReply, "UnknownNetworkError", None))
_NoError = getattr(_NE, "NoError", getattr(QNetworkReply, "NoError", 0))

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


def _log_warning(msg: str):
    QgsMessageLog.logMessage(msg, "AI Segmentation", level=Qgis.MessageLevel.Warning)


def _read_plugin_version() -> str:
    """Plugin version from metadata.txt, or "unknown". Never raises."""
    import os

    try:
        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(plugin_dir, "metadata.txt"), encoding="utf-8") as f:
            for line in f:
                if line.startswith("version="):
                    return line.strip().split("=", 1)[1]
    except Exception:  # noqa: BLE001
        pass  # nosec B110
    return "unknown"


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


def _classify_network_error(blocker: QgsBlockingNetworkRequest) -> tuple[str, str]:
    reply = blocker.reply()
    qt_error = reply.error() if reply else _UnknownNetwork
    return _classify_qt_error(qt_error, blocker.errorMessage(), _http_status_of(reply))


def _classify_qt_error(qt_error, error_string: str, http_status: int | None) -> tuple[str, str]:
    """Map a Qt NetworkError to a (code, user-message) pair. Shared by the
    blocking path (_classify_network_error) and the concurrent path
    (_parse_reply) so both classify identically."""
    _log_warning(
        f"Network error: qt_error={int(qt_error)}, http_status={http_status}, "
        f"detail={error_string[:500]}"
    )

    if qt_error == _HostNotFound:
        return "DNS_ERROR", tr("Cannot reach the server. Check your internet connection.")
    if qt_error == _ConnRefused:
        return "CONNECTION_REFUSED", tr("Server refused the connection.")
    if qt_error == _Timeout or (_OpCanceled is not None and qt_error == _OpCanceled):
        # No HTTP status means the socket never got an answer within the
        # deadline: on the detection hot path this is a cold-starting / busy
        # GPU, not a dead link. Code TIMEOUT keeps it a transient retry AND
        # (unlike NO_INTERNET) makes the worker surface the "your spot is held"
        # waiting state instead of blaming the user's connection.
        if http_status is None:
            return "SERVICE_WARMING", tr("The AI service is waking up. Holding your spot…")
        return "TIMEOUT", tr("Request timed out. Check your connection or try again.")
    if qt_error == _SslFailed:
        return "SSL_ERROR", tr("SSL certificate error. Your network may be blocking secure connections.")
    if qt_error in _PROXY_ERRORS:
        return "PROXY_ERROR", tr(
            "Proxy connection failed. "
            "Check QGIS proxy settings (Settings > Options > Network)."
        )
    if qt_error in (_ContentDenied, _AuthRequired):
        return "AUTH_ERROR", tr("Authentication failed. Please sign in again.")
    return "NO_INTERNET", tr("Network error. Check your internet connection.")


class TerraLabClient:
    """HTTP client for the TerraLab backend - uses QgsBlockingNetworkRequest so
    requests pick up QGIS proxy / SSL / Network Logger settings."""

    def __init__(self, base_url: str | None = None):
        if base_url is None:
            base_url = self._read_base_url()
        self.base_url = base_url.rstrip("/")
        # Cloud detection (per-tile /predict) can talk DIRECTLY to the inference
        # service instead of going through the main backend, removing a network
        # hop on the hot path. When a direct URL is configured the per-tile calls
        # use it (+ the direct route shapes); otherwise everything stays on
        # base_url unchanged. Resolved once at construction.
        direct = self._read_detection_base_url()
        self.detection_direct = bool(direct)
        self.detection_base_url = (direct or self.base_url).rstrip("/")

    @staticmethod
    def _read_base_url() -> str:
        return TerraLabClient._read_env_value("TERRALAB_BASE_URL", "https://terra-lab.ai")

    @staticmethod
    def _read_detection_base_url() -> str:
        """Base URL for the per-tile cloud detection calls, when they should go
        DIRECT to the inference service (one hop) instead of via the main backend.

        Resolution: an .env.local `TERRALAB_DETECTION_URL` (dev opt-in) wins, else the
        shipped default below. Empty string => not direct: detection keeps using
        the main base_url and the existing backend routes (zero behaviour change,
        the safe rollback: set the default back to "" to send everyone through
        the backend again)."""
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
        inference service's /predict; otherwise the main backend route."""
        if self.detection_direct:
            return f"{self.detection_base_url}/predict"
        return f"{self.detection_base_url}/api/ai-segmentation/predict"

    def _detection_run_export_url(self) -> str:
        """Absolute URL for the end-of-run export summary. Same host split as
        the predict call."""
        if self.detection_direct:
            return f"{self.detection_base_url}/run-export"
        return f"{self.detection_base_url}/api/ai-segmentation/run-export"

    def post_run_export(self, payload: dict, auth: dict) -> dict:
        """Send the finished run's export summary (review settings + the final
        exported geometry when small enough) so the run record is complete.

        Additive and best-effort: callers fire-and-forget from a background
        task; a server without the route degrades to {"error", "code"} and the
        user's local export is never affected. Off-GUI-thread only.
        """
        body = json.dumps(payload).encode("utf-8")
        return self._request(
            "POST", self._detection_run_export_url(), auth=auth, body=body,
            timeout_ms=_TIMEOUT_RUN_EXPORT,
        )

    def _submit_timeout(self) -> int:
        """Submit timeout. Direct mode blocks for the whole inference (no async
        fallback), so it needs a cold-start-proof window to avoid a retry that
        would double-charge; the backend route returns fast, so it keeps 45s."""
        return _TIMEOUT_SUBMIT_DETECTION_DIRECT if self.detection_direct else _TIMEOUT_SUBMIT_DETECTION

    def get_usage(self, auth: dict) -> dict:
        return self._request(
            "GET", "/api/plugin/usage", auth=auth, timeout_ms=_TIMEOUT_INTERACTIVE
        )

    def get_account(self, auth: dict) -> dict:
        return self._request(
            "GET", "/api/plugin/account", auth=auth, timeout_ms=_TIMEOUT_INTERACTIVE
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
             "auth": auth, "timeout_ms": _TIMEOUT_INTERACTIVE},
            {"method": "GET", "path": "/api/plugin/usage",
             "auth": auth, "timeout_ms": _TIMEOUT_INTERACTIVE},
        ])
        return account, usage

    def get_config(self, product: str) -> dict:
        return self._request(
            "GET", f"/api/plugin/config?product={product}", timeout_ms=_TIMEOUT_INTERACTIVE
        )

    def translate_prompt(self, text: str, auth: dict | None = None) -> dict:
        """Resolve a short object prompt to its English equivalent (the server
        caches the answers). Additive endpoint: offline or not-yet-deployed
        degrades to {"error", "code"} and the caller keeps the typed text.

        Passing ``auth`` (the activation Bearer) puts the call on the per-key
        rate budget instead of the tight anonymous one; omit it and the server
        still serves the request anonymously."""
        body = json.dumps({"text": text}).encode("utf-8")
        return self._request(
            "POST", "/api/plugin/translate-prompt", auth=auth, body=body,
            timeout_ms=_TIMEOUT_TRANSLATE,
        )

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
        body = json.dumps({
            "prompt": prompt,
            "zone_area_m2": zone_area_m2,
            "native_mupp": native_mupp,
            "plugin_version": _read_plugin_version(),
        }).encode("utf-8")
        return self._request(
            "POST", "/api/plugin/seg-run-plan", auth=auth, body=body,
            timeout_ms=_TIMEOUT_TRANSLATE,
        )

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

        params = ["limit={}".format(int(limit))]
        if before:
            params.append("before={}".format(quote(str(before), safe="")))
        if favorites_only:
            params.append("favorites_only=true")
        if deleted:
            params.append("deleted=true")
        path = "/api/ai-segmentation/history?" + "&".join(params)
        return self._request("GET", path, auth=auth, timeout_ms=_TIMEOUT_API)

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
        return self._request("GET", path, auth=auth, timeout_ms=_TIMEOUT_API)

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
        return self._request("GET", path, auth=auth, timeout_ms=_TIMEOUT_API)

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

    @staticmethod
    def _build_predict_body(
        run_id, prompt, image_b64, tile_index, crs_authid,
        tile_bbox_wgs84, tile_bbox_native, pixel_size_m,
        max_masks, threshold, mask_threshold, exemplars,
        parent_tile_index=None,
    ) -> bytes:
        """Serialize one /predict payload. Shared by submit_detection (one tile)
        and submit_detection_many (a concurrent batch)."""
        payload: dict = {
            "image": image_b64,
            "run_id": run_id,
            "prompt": prompt,
            "tile_index": tile_index,
            "crs_authid": crs_authid,
        }
        if parent_tile_index is not None:
            # Re-split quadrant of an already-billed tile in the same run.
            payload["parent_tile_index"] = int(parent_tile_index)
        if tile_bbox_wgs84 is not None:
            payload["tile_bbox_wgs84"] = tile_bbox_wgs84
        if tile_bbox_native is not None:
            payload["tile_bbox_native"] = tile_bbox_native
        if pixel_size_m is not None:
            payload["pixel_size_m"] = pixel_size_m
        if max_masks is not None:
            payload["max_masks"] = max_masks
        if threshold is not None:
            payload["threshold"] = threshold
        if mask_threshold is not None:
            payload["mask_threshold"] = mask_threshold
        if exemplars:
            payload["exemplars"] = exemplars
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
                s.get("parent_tile_index"),
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
        )
        req = self._make_qnetwork_request(
            auth, self._submit_timeout(), self._detection_predict_url()
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
                # Direct mode: ping the inference service's open /health probe
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

    def _make_qnetwork_request(self, auth: dict | None, timeout_ms: int, path: str) -> QNetworkRequest:
        """Build a QNetworkRequest with the same headers/timeout/redirect policy
        _request applies (so the concurrent path behaves identically)."""
        req = QNetworkRequest(QUrl(self._resolve_url(path)))
        req.setRawHeader(b"Content-Type", b"application/json")
        req.setTransferTimeout(timeout_ms)
        if _REDIRECT_ATTR is not None and _NO_LESS_SAFE_REDIRECT is not None:
            req.setAttribute(_REDIRECT_ATTR, _NO_LESS_SAFE_REDIRECT)
        if auth:
            for key, value in auth.items():
                req.setRawHeader(key.encode("utf-8"), value.encode("utf-8"))
        return req

    def request_many(self, specs: list[dict], should_abort=None) -> list[dict]:
        """Execute several requests CONCURRENTLY, returning results in input order.

        Each spec is a dict: {"method": "GET"|"POST", "path": str, "auth": dict
        (optional), "body": bytes (optional), "timeout_ms": int (optional)}.
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
            req = self._make_qnetwork_request(spec.get("auth"), timeout_ms, spec["path"])
            if spec.get("method") == "POST":
                reply = nam.post(req, QByteArray(spec.get("body") or b""))
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
            results = [self._parse_reply(r) for r in replies]
            for r in replies:
                r.deleteLater()
            return results

        # Safety net: setTransferTimeout already bounds each reply, but a wedged
        # NAM could still stall the loop, so cap the whole batch defensively.
        QTimer.singleShot(max_timeout + 5_000, loop.quit)
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

        results = []
        for reply in replies:
            results.append(self._parse_reply(reply))
            if not reply.isFinished():
                reply.abort()  # release a still-pending socket on the timeout path
            reply.deleteLater()
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
        touched cross-thread and misbehave or crash."""
        from qgis.PyQt.QtCore import QThread

        nams = getattr(self, "_own_nams", None)
        if nams is None:
            nams = {}
            self._own_nams = nams
        thread = QThread.currentThread()
        nam = nams.get(thread)
        if nam is None:
            nam = self._new_private_nam()
            nams[thread] = nam
        return nam

    def _new_private_nam(self):
        """Create a private QNetworkAccessManager configured like the QGIS one
        (proxy mirror + stored SSL exceptions honoured). The manager takes the
        affinity of the creating thread; destroy it on that same thread."""
        from qgis.PyQt.QtNetwork import QNetworkAccessManager

        nam = QNetworkAccessManager()
        try:
            nam.setProxy(self._qgis_effective_proxy())
        except (RuntimeError, AttributeError):
            pass
        # Honour QGIS-configured SSL exceptions on this private manager so it
        # trusts exactly what the blocking QGIS path trusts (corporate CA /
        # self-signed). Custom CAs added to QGIS land in the process-global
        # default QSslConfiguration and are already trusted here; the gap is
        # the per-host EXCEPTIONS, applied via the sslErrors handler below.
        try:
            nam.sslErrors.connect(self._on_predict_ssl_errors)
        except (RuntimeError, AttributeError):
            pass
        return nam

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
        nams = getattr(self, "_own_nams", None)
        if not nams:
            return
        from qgis.PyQt.QtCore import QThread

        nams.pop(QThread.currentThread(), None)

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
            except Exception:  # noqa: BLE001 - proxy mirror is best-effort
                pass  # nosec B110
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
            hostport = "{host}:{port}".format(host=url.host(), port=url.port(443))
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

    def _parse_reply(self, reply) -> dict:
        """Turn a finished QNetworkReply into the same dict _request returns."""
        if not reply.isFinished():
            # Only reachable via the batch safety-net timeout (each reply also has
            # its own setTransferTimeout that fires first). Report it as a
            # transient TIMEOUT so the caller re-polls instead of dropping the tile.
            return {"error": tr("Request timed out. Check your connection or try again."),
                    "code": "TIMEOUT"}
        qt_error = reply.error()
        http_status = _http_status_of(reply)
        raw_body = bytes(reply.readAll()).decode("utf-8", "replace")

        if qt_error != _NoError:
            # A 4xx/5xx still carries our JSON error body; prefer it over the
            # generic network classification so the worker sees the real code.
            if http_status is not None and http_status >= 400 and raw_body:
                try:
                    return json.loads(raw_body)
                except Exception:
                    pass  # nosec B110
            code, msg = _classify_qt_error(qt_error, reply.errorString(), http_status)
            return {"error": msg, "code": code}

        if http_status is not None and http_status >= 400:
            _log_warning(f"HTTP {http_status}: {raw_body[:500]}")
            try:
                error_body = json.loads(raw_body)
                if "error" in error_body:
                    return error_body
                return {
                    "error": error_body.get("detail", raw_body[:200]),
                    "code": "SERVER_ERROR",
                }
            except Exception:
                return {"error": f"Server error (HTTP {http_status})", "code": "SERVER_ERROR"}

        if not raw_body:
            return {}
        try:
            return json.loads(raw_body)
        except json.JSONDecodeError:
            _log_warning(f"Invalid JSON response: {raw_body[:500]}")
            return {"error": "Invalid server response", "code": "SERVER_ERROR"}

    def _request(
        self,
        method: str,
        path: str,
        auth: dict | None = None,
        body: bytes | None = None,
        timeout_ms: int = _TIMEOUT_API,
    ) -> dict:
        url = self._resolve_url(path)
        req = QNetworkRequest(QUrl(url))
        req.setRawHeader(b"Content-Type", b"application/json")
        # Hard per-request deadline so a wedged connection can never hang the
        # task forever. setTransferTimeout exists on Qt 5.15+ (QGIS >= 3.22),
        # which is our floor, so this is unconditional.
        req.setTransferTimeout(timeout_ms)
        if _REDIRECT_ATTR is not None and _NO_LESS_SAFE_REDIRECT is not None:
            req.setAttribute(_REDIRECT_ATTR, _NO_LESS_SAFE_REDIRECT)
        if auth:
            for key, value in auth.items():
                req.setRawHeader(key.encode("utf-8"), value.encode("utf-8"))

        blocker = QgsBlockingNetworkRequest()
        if method == "GET":
            err = blocker.get(req, forceRefresh=True)
        elif method == "POST":
            payload = QByteArray(body) if body else QByteArray()
            err = blocker.post(req, payload)
        else:
            return {"error": f"Unsupported method: {method}", "code": "CLIENT_ERROR"}

        if err != QgsBlockingNetworkRequest.ErrorCode.NoError:
            reply = blocker.reply()
            http_status = _http_status_of(reply)
            if reply is not None and http_status is not None and http_status >= 400:
                # A misbehaving gateway/proxy can return a non-UTF-8 body; decode
                # leniently so a garbled error page never crashes the worker loop.
                raw = bytes(reply.content()).decode("utf-8", "replace")
                if raw:
                    try:
                        return json.loads(raw)
                    except Exception:
                        pass  # nosec B110
            code, msg = _classify_network_error(blocker)
            return {"error": msg, "code": code}

        reply = blocker.reply()
        http_status = _http_status_of(reply)
        raw_body = bytes(reply.content()).decode("utf-8", "replace")

        if http_status is not None and http_status >= 400:
            _log_warning(f"HTTP {http_status}: {raw_body[:500]}")
            try:
                error_body = json.loads(raw_body)
                if "error" in error_body:
                    return error_body
                return {
                    "error": error_body.get("detail", raw_body[:200]),
                    "code": "SERVER_ERROR",
                }
            except Exception:
                return {"error": f"Server error (HTTP {http_status})", "code": "SERVER_ERROR"}

        if not raw_body:
            return {}
        try:
            return json.loads(raw_body)
        except json.JSONDecodeError:
            _log_warning(f"Invalid JSON response: {raw_body[:500]}")
            return {"error": "Invalid server response", "code": "SERVER_ERROR"}
