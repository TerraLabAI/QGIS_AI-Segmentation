













from __future__ import annotations

import hashlib
import re
import secrets
import string
import time
from urllib.parse import urlsplit

from qgis.core import QgsTask
from qgis.PyQt.QtCore import pyqtSignal

from ..core.activation_manager import ACTIVATION_KEY_RE
from ..core.i18n import tr
from ..core.logging_utils import log
from ..core.server_dials import dial_in_range

PAIRING_PRODUCT = "ai-segmentation"
_CLEARED = ""


_USER_CODE_ALPHABET = "".join(
    c for c in string.digits + string.ascii_uppercase if c not in "01IO")
_USER_CODE_LENGTH = 6



_SERVER_CODE_RE = re.compile(r"^[A-Za-z0-9_-]{32,128}$")



START_FAILURES_BEFORE_LEGACY = 3


def make_secret() -> str:

    return secrets.token_urlsafe(32)


def secret_hash(secret: str) -> str:


    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def normalize_user_code(raw: str) -> str:


    cleaned = "".join(ch for ch in str(raw or "") if ch not in " -\t").upper()
    if len(cleaned) != _USER_CODE_LENGTH:
        return ""
    if any(ch not in _USER_CODE_ALPHABET for ch in cleaned):
        return ""
    return cleaned


def pairing_no_plan_message() -> str:
    return tr("This account has no active AI Segmentation plan. "
              "Reactivate it on terra-lab.ai, then click Sign in again.")


def pairing_cancelled_message() -> str:
    return tr("Sign-in was cancelled in the browser. Click Sign in to try again.")


def _answer_http_status(answer) -> int | None:
    if isinstance(answer, dict):
        status = answer.get("http_status")
        if isinstance(status, int):
            return status
    return None


def _answer_retry_after_s(answer, default: float, ceiling: float) -> float:
    raw = answer.get("retry_after") if isinstance(answer, dict) else None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if not 0 < value < float("inf"):
        return default
    return min(value, ceiling)


def _is_site_connect_url(url: str, base_url: str) -> bool:




    try:
        page, base = urlsplit(url), urlsplit(base_url)
    except ValueError:
        return False

    def bare(host: str | None) -> str:
        host = (host or "").lower()
        return host[4:] if host.startswith("www.") else host

    return (bool(base.hostname) and page.scheme == base.scheme
            and bare(page.hostname) == bare(base.hostname)
            and page.port == base.port and page.path == "/connect"
            and not page.username and not page.password)


class _PairingStepTask(QgsTask):
    def _sleep_cancellable(self, seconds: float) -> None:
        end = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < end:
            if self.isCanceled():
                return
            time.sleep(0.1)

    def is_active(self) -> bool:
        try:
            return self.status() in (
                QgsTask.TaskStatus.Running,
                QgsTask.TaskStatus.Queued,
                QgsTask.TaskStatus.OnHold,
            )
        except Exception:
            return False


class PairingStartTask(_PairingStepTask):








    started = pyqtSignal(str, str, int)
    use_legacy = pyqtSignal(str)

    def __init__(self, client, secret_digest: str, loopback_port: int | None,
                 ttl_s: int):
        super().__init__(tr("Connecting AI Segmentation"), QgsTask.Flag.CanCancel)
        self._client = client
        self._digest = secret_digest
        self._port = loopback_port
        self._ttl_s = int(ttl_s)
        self._answer: tuple[str, str, int] | None = None
        self._legacy_reason = ""

    def _read_started(self, answer) -> tuple[str, str, int] | None:
        if not isinstance(answer, dict) or "error" in answer:
            return None
        code = answer.get("code")
        connect_url = answer.get("connect_url")
        if not isinstance(code, str) or not _SERVER_CODE_RE.match(code):
            return None
        if not isinstance(connect_url, str) or not _is_site_connect_url(
                connect_url, str(getattr(self._client, "base_url", ""))):
            return None
        try:
            expires_in = int(answer.get("expires_in") or self._ttl_s)
        except (TypeError, ValueError):
            expires_in = self._ttl_s
        return code, connect_url, max(60, min(expires_in, self._ttl_s))

    def run(self) -> bool:
        failures = 0
        waited_out_rate_limit = False
        pause_s = dial_in_range("tuning.pairing.start_retry_pause_s", 1.0, 0.2, 10.0)
        while not self.isCanceled():
            try:
                answer = self._client.start_pairing(self._digest, self._port, self._ttl_s)
            except Exception:  # noqa: BLE001
                answer = {"error": "start failed", "code": "NO_INTERNET"}
            if self.isCanceled():
                return False
            started = self._read_started(answer)
            if started is not None:
                self._answer = started
                return True
            status = _answer_http_status(answer)
            if status == 400:
                log("Pairing: the sign-in service refused the request (HTTP 400); "
                    "using the previous sign-in")
                self._legacy_reason = "start_invalid_request"
                return False
            if status == 429 and not waited_out_rate_limit:
                waited_out_rate_limit = True
                self._sleep_cancellable(_answer_retry_after_s(answer, 5.0, 60.0))
                continue
            failures += 1


            if status is not None:
                detail = status
            elif isinstance(answer, dict) and "error" in answer:
                detail = str(answer.get("code") or "error")
            else:
                detail = "unusable answer"
            log(f"Pairing: sign-in service did not start ({detail}), "
                f"attempt {failures} of {START_FAILURES_BEFORE_LEGACY}")
            if failures >= START_FAILURES_BEFORE_LEGACY:
                self._legacy_reason = "start_unavailable"
                return False
            self._sleep_cancellable(pause_s)
        return False

    def finished(self, result: bool) -> None:
        if self.isCanceled():
            return
        if result and self._answer is not None:
            self.started.emit(*self._answer)
        elif self._legacy_reason:
            self.use_legacy.emit(self._legacy_reason)


class PairingClaimTask(_PairingStepTask):








    answered = pyqtSignal(dict)

    def __init__(self, client, code: str, secret: str, grant: str = "",
                 user_code: str = ""):
        super().__init__(tr("Connecting AI Segmentation"), QgsTask.Flag.CanCancel)
        self._client = client
        self._code = code
        self._secret = secret
        self._grant = grant
        self._user_code = user_code
        self._result: dict = {"status": "network"}

    @staticmethod
    def _read(answer) -> dict:
        if not isinstance(answer, dict):
            return {"status": "network"}
        status = answer.get("status")
        if status == "ready":
            raw_key = answer.get("activation_key")
            key = raw_key.strip() if isinstance(raw_key, str) else ""
            if ACTIVATION_KEY_RE.match(key):
                return {"status": "ready", "key": key}
            return {"status": "bad_key"}
        if status in ("pending", "no_plan", "cancelled", "invalid", "locked",
                      "not_found", "invalid_request", "rate_limited", "error"):
            out = {"status": status}
            if status == "rate_limited":
                out["retry_after"] = answer.get("retry_after")
            return out
        if _answer_http_status(answer) == 503:
            return {"status": "error"}
        return {"status": "network"}

    def run(self) -> bool:
        retried_unavailable = False
        retried_rate_limit = False
        try:
            while not self.isCanceled():
                try:
                    answer = self._client.claim_pairing(
                        self._code, self._secret, grant=self._grant,
                        user_code=self._user_code)
                except Exception:  # noqa: BLE001
                    answer = None
                result = self._read(answer)
                status = result["status"]
                if status == "error" and not retried_unavailable:
                    retried_unavailable = True
                    self._sleep_cancellable(1.0)
                    continue
                if status == "rate_limited" and not retried_rate_limit:
                    retried_rate_limit = True
                    self._sleep_cancellable(_answer_retry_after_s(result, 2.0, 8.0))
                    continue
                result.pop("retry_after", None)
                self._result = result
                return status == "ready"
            return False
        finally:
            self._secret = _CLEARED
            self._grant = _CLEARED
            self._user_code = ""

    def finished(self, _result: bool) -> None:
        self.answered.emit(dict(self._result))
        self._result = {"status": "network"}
