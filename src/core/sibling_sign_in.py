# SPDX-FileCopyrightText: 2026 TerraLab <yvann.barbot@terra-lab.ai>
# SPDX-License-Identifier: GPL-2.0-or-later





















from __future__ import annotations

import base64
import json
import re
import sys

from qgis.core import Qgis, QgsApplication, QgsMessageLog, QgsSettings, QgsTask

KEY_RE = re.compile(r"^tl_[0-9a-f]{32}$")
TIMEOUT_MS = 15000


PRODUCTS = {
    "ai-agent": {
        "label": "AI Agent",
        "authcfg": "TerraLab/AIAgent/authcfg_id",
        "plain": "TerraLab/AIAgent/activation_key",
        "sealed": "TerraLab/AIAgent/activation_key_dpapi",
    },
    "ai-segmentation": {
        "label": "AI Segmentation",
        "authcfg": "AISegmentation/authcfg_id",
        "plain": "AISegmentation/activation_key",
        "sealed": "",
    },
    "ai-edit": {
        "label": "AI Edit",
        "authcfg": "AIEdit/authcfg_id",
        "plain": "AIEdit/activation_key",
        "sealed": "",
    },
}
_HELD_KEY = "TerraLab/sibling_sign_in/{}"
_DPAPI_PREFIX = "dpapi1:"
_DPAPI_ENTROPY = b"TerraLab AI Agent activation key"
_TAG = "TerraLab sign-in"

_running = {}


def _log(message: str, warning: bool = False) -> None:
    level = Qgis.MessageLevel.Warning if warning else Qgis.MessageLevel.Info
    QgsMessageLog.logMessage(message, _TAG, level=level)


def _text(settings: QgsSettings, name: str) -> str:
    if not name:
        return ""
    value = settings.value(name, "")
    return value.strip() if isinstance(value, str) else ""


def _unseal(value: str) -> str:

    if sys.platform != "win32" or not value.startswith(_DPAPI_PREFIX):
        return ""
    try:
        import ctypes
        from ctypes import wintypes

        class _Blob(ctypes.Structure):
            _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

        def _blob(raw: bytes):
            buffer = ctypes.create_string_buffer(raw, len(raw))
            return _Blob(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))), buffer

        crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        crypt32.CryptUnprotectData.argtypes = [ctypes.POINTER(_Blob), ctypes.c_void_p, ctypes.POINTER(_Blob),
                                               ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD,
                                               ctypes.POINTER(_Blob)]
        crypt32.CryptUnprotectData.restype = wintypes.BOOL
        kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        kernel32.LocalFree.restype = ctypes.c_void_p
        source, _keep = _blob(base64.b64decode(value[len(_DPAPI_PREFIX):]))
        entropy, _keep_entropy = _blob(_DPAPI_ENTROPY)
        out = _Blob()

        if not crypt32.CryptUnprotectData(ctypes.byref(source), None, ctypes.byref(entropy), None, None, 0x1,
                                          ctypes.byref(out)):
            return ""
        try:
            return ctypes.string_at(out.pbData, out.cbData).decode("utf-8")
        finally:
            kernel32.LocalFree(ctypes.cast(out.pbData, ctypes.c_void_p))
    except Exception:  # noqa: BLE001
        return ""


def _from_auth_database(authcfg_id: str) -> str:

    if not authcfg_id:
        return ""
    try:
        from qgis.core import QgsAuthMethodConfig

        manager = QgsApplication.authManager()
        if manager is None or not manager.masterPasswordIsSet():
            return ""
        config = QgsAuthMethodConfig()
        loaded = manager.loadAuthenticationConfig(authcfg_id, config, True)
        if isinstance(loaded, tuple):
            loaded, config = (loaded + (config,))[:2]
        return (config.config("password", "") or "").strip() if loaded and config is not None else ""
    except Exception:  # noqa: BLE001
        return ""


def stored_key(product: str, settings: QgsSettings | None = None) -> str:

    names = PRODUCTS[product]
    settings = settings or QgsSettings()
    key = _from_auth_database(_text(settings, names["authcfg"]))
    if not key:
        key = _unseal(_text(settings, names["sealed"]))
    if not key:
        key = _text(settings, names["plain"])
    return key if KEY_RE.match(key) else ""


def holds_a_key(product: str, settings: QgsSettings | None = None) -> bool:

    names = PRODUCTS[product]
    settings = settings or QgsSettings()
    return any(_text(settings, names[kind]) for kind in ("authcfg", "sealed", "plain"))


def start(own: str, base_url: str, device_hash: str, on_done) -> bool:







    if own not in PRODUCTS or own in _running:
        return False
    settings = QgsSettings()
    held = _HELD_KEY.format(own)
    once_held = settings.value(held, False, type=bool)
    if holds_a_key(own, settings):


        if not once_held:
            settings.setValue(held, True)
        return False
    if once_held:
        return False
    siblings = [(product, key) for product, key in ((p, stored_key(p, settings)) for p in PRODUCTS if p != own)
                if key]
    if not siblings:
        return False
    task = _SiblingKeyTask(own, str(base_url or "").rstrip("/"), str(device_hash or ""), siblings, on_done)
    _running[own] = task
    QgsApplication.taskManager().addTask(task)
    return True


def cancel(own: str) -> None:

    task = _running.pop(own, None)
    if task is not None:
        task.on_done = None
        try:
            task.cancel()
        except RuntimeError:
            pass  # nosec B110


def _enum(owner, group: str, name: str):
    return getattr(getattr(owner, group, owner), name)


def _call(method: str, url: str, product: str, key: str, device_hash: str, body: dict | None = None):

    from qgis.core import QgsBlockingNetworkRequest
    from qgis.PyQt.QtCore import QByteArray, QUrl
    from qgis.PyQt.QtNetwork import QNetworkRequest

    request = QNetworkRequest(QUrl(url))
    request.setRawHeader(b"Authorization", f"Bearer {key}".encode("ascii"))
    request.setRawHeader(b"X-Product-ID", product.encode("ascii"))
    if device_hash:
        request.setRawHeader(b"X-Device-Hash", device_hash.encode("ascii", "ignore"))
    request.setRawHeader(b"Content-Type", b"application/json")
    try:
        request.setTransferTimeout(TIMEOUT_MS)
    except AttributeError:
        pass  # nosec B110
    from .gil_safe_qobject import prime



    blocker = prime(QgsBlockingNetworkRequest())
    if method == "POST":
        blocker.post(request, QByteArray(json.dumps(body or {}).encode("utf-8")))
    else:
        blocker.get(request, True)
    reply = blocker.reply()
    status = None
    payload = {}
    try:
        status = reply.attribute(_enum(QNetworkRequest, "Attribute", "HttpStatusCodeAttribute"))
        status = int(status) if status is not None else None
        raw = bytes(reply.content())
        parsed = json.loads(raw.decode("utf-8")) if raw else {}
        payload = parsed if isinstance(parsed, dict) else {}
    except Exception:  # noqa: BLE001
        payload = {}
    return status, payload


class _SiblingKeyTask(QgsTask):
    def __init__(self, own: str, base_url: str, device_hash: str, siblings: list, on_done):
        flags = QgsTask.Flag.CanCancel
        for name in ("Hidden", "Silent"):
            extra = getattr(QgsTask.Flag, name, None)
            if extra is not None:
                flags = flags | extra
        super().__init__("TerraLab sign-in", flags)
        self.own = own
        self.base_url = base_url
        self.device_hash = device_hash
        self.siblings = siblings
        self.on_done = on_done
        self.result = {"ok": False, "reason": "not_tried"}

    def run(self) -> bool:
        try:
            self.result = self._ask()
        except Exception as exc:  # noqa: BLE001
            self.result = {"ok": False, "reason": f"error:{type(exc).__name__}"}
        return bool(self.result.get("ok"))

    def _ask(self) -> dict:
        emails = {}
        if len(self.siblings) > 1:
            for product, key in self.siblings:
                if self.isCanceled():
                    return {"ok": False, "reason": "cancelled"}
                status, account = _call("GET", f"{self.base_url}/api/plugin/account", product, key,
                                        self.device_hash)
                if status == 200 and isinstance(account.get("email"), str):
                    emails[product] = account["email"].strip().lower()
                elif status not in (401, 403):
                    return {"ok": False, "reason": f"unreachable:{status}"}
            if len(set(emails.values())) > 1:
                return {"ok": False, "reason": "different_accounts"}
            self.siblings = [(product, key) for product, key in self.siblings if product in emails]
        sibling_keys = {key for _product, key in self.siblings}
        reason = "refused"
        for product, key in self.siblings:
            if self.isCanceled():
                return {"ok": False, "reason": "cancelled"}
            status, answer = _call("POST", f"{self.base_url}/api/plugin/sibling-key", product, key,
                                   self.device_hash, {"product": self.own, "device_id": self.device_hash})
            if status in (401, 403):
                reason = "sibling_refused"
                continue
            if status != 200:
                return {"ok": False, "reason": f"{answer.get('reason') or 'unreachable'}:{status}"}
            own_key = str(answer.get("activation_key") or "")
            if not KEY_RE.match(own_key) or own_key in sibling_keys:
                return {"ok": False, "reason": str(answer.get("reason") or "no_key")}
            email = str(answer.get("email") or emails.get(product) or "")
            return {"ok": True, "key": own_key, "email": email, "sibling": product,
                    "label": PRODUCTS[product]["label"]}
        return {"ok": False, "reason": reason}

    def finished(self, result: bool) -> None:
        if _running.get(self.own) is self:
            _running.pop(self.own, None)
        callback, self.on_done = self.on_done, None
        outcome, self.result = self.result, {}
        self.siblings = []
        if outcome.get("ok"):

            _log(f"{PRODUCTS[self.own]['label']}: signed in with the account of "
                 f"{outcome['label']}")
        else:
            _log(f"{PRODUCTS[self.own]['label']}: no sign-in from a sibling plugin "
                 f"({outcome.get('reason')})")
        if callback is not None and not self.isCanceled():
            try:
                callback(outcome)
            except Exception as exc:  # noqa: BLE001
                _log(f"Sign-in from a sibling plugin not applied: {type(exc).__name__}", warning=True)
