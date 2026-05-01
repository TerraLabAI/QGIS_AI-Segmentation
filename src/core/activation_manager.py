"""Activation manager for the AI Segmentation plugin."""
from __future__ import annotations

import re
import uuid

from qgis.core import QgsSettings

PRODUCT_ID = "ai-segmentation"

_KEY_RE = re.compile(r"^tl_[0-9a-f]{32}$")

SETTINGS_PREFIX = "AISegmentation/"
TERRALAB_PREFIX = "TerraLab/"
DEVICE_ID_KEY = f"{TERRALAB_PREFIX}device_id"

TUTORIAL_URL_FALLBACK = "https://youtu.be/lbADk75l-mk?si=q6WnwyV2NcmQYuhI"
_SIGN_IN_BASE = (
    "https://terra-lab.ai/login"
    "?utm_source=qgis&utm_medium=plugin&utm_campaign=ai-segmentation"
    "&utm_content=sign_in&product=ai-segmentation"
)
_SIGN_UP_BASE = (
    "https://terra-lab.ai/register"
    "?utm_source=qgis&utm_medium=plugin&utm_campaign=ai-segmentation"
    "&utm_content=sign_up&product=ai-segmentation"
)

_cached_config: dict | None = None


def _client():
    from ..api.terralab_client import TerraLabClient
    return TerraLabClient()


# -- session state ---------------------------------------------------------


def get_auth_token(settings=None) -> str:
    s = settings or QgsSettings()
    return s.value(f"{SETTINGS_PREFIX}activation_key", "") or ""


def save_auth_token(token: str, settings=None):
    s = settings or QgsSettings()
    s.setValue(f"{SETTINGS_PREFIX}activation_key", token.strip())
    s.setValue(f"{SETTINGS_PREFIX}activated", True)


def clear_auth(settings=None):
    s = settings or QgsSettings()
    s.setValue(f"{SETTINGS_PREFIX}activation_key", "")
    s.setValue(f"{SETTINGS_PREFIX}activated", False)


def is_plugin_activated(settings=None) -> bool:
    return bool(get_auth_token(settings))


# -- terms of service consent (required to run a segmentation) -------------


def has_tos_accepted(settings=None) -> bool:
    """True only after the user has explicitly accepted Terms + Privacy.

    Stored separately from telemetry consent: ToS is mandatory to use the
    service, telemetry is optional opt-in handled elsewhere.
    """
    s = settings or QgsSettings()
    return bool(s.value(f"{SETTINGS_PREFIX}tos_accepted", False, type=bool))


def set_tos_accepted(granted: bool, settings=None):
    """Persist the user's Terms + Privacy acceptance decision."""
    s = settings or QgsSettings()
    s.setValue(f"{SETTINGS_PREFIX}tos_accepted", bool(granted))


def has_tos_locked(settings=None) -> bool:
    """True once the user has run at least one segmentation with consent.

    After the first successful Start click the Terms + Privacy gate is sealed
    shut: we stop showing the checkbox and treat consent as permanently given,
    even across plugin updates or fresh sessions. The reasoning is that by
    running the service the user has already accepted the ToS in practice, so
    re-prompting is pure friction.
    """
    s = settings or QgsSettings()
    return bool(s.value(f"{SETTINGS_PREFIX}tos_locked", False, type=bool))


def lock_tos():
    """Seal the Terms + Privacy acceptance. Irreversible by design."""
    s = QgsSettings()
    s.setValue(f"{SETTINGS_PREFIX}tos_locked", True)
    s.setValue(f"{SETTINGS_PREFIX}tos_accepted", True)


def get_auth_header(settings=None) -> dict:
    token = get_auth_token(settings)
    if not token:
        return {}
    return {
        "Authorization": f"Bearer {token}",
        "X-Product-ID": PRODUCT_ID,
    }


# -- device id (stable per machine, shared across TerraLab plugins) --------


def get_device_id(settings=None) -> str:
    s = settings or QgsSettings()
    device_id = s.value(DEVICE_ID_KEY, "")
    if not device_id:
        device_id = str(uuid.uuid4())
        s.setValue(DEVICE_ID_KEY, device_id)
    return device_id


# -- server config ---------------------------------------------------------


def get_server_config() -> dict:
    global _cached_config
    if _cached_config is not None:
        return _cached_config
    try:
        result = _client().get_config(PRODUCT_ID)
        if "error" not in result:
            _cached_config = result
            return result
    except Exception:
        pass  # nosec B110
    return {}


def clear_config_cache():
    global _cached_config
    _cached_config = None


def get_tutorial_url() -> str:
    config = get_server_config()
    if config and "tutorial_url" in config:
        return config["tutorial_url"]
    return TUTORIAL_URL_FALLBACK


def get_sign_in_url() -> str:
    return f"{_SIGN_IN_BASE}&device_id={get_device_id()}"


def get_sign_up_url() -> str:
    return f"{_SIGN_UP_BASE}&device_id={get_device_id()}"


# -- activation key validation ---------------------------------------------


def validate_key_with_server(key: str) -> tuple[bool, str]:
    key = (key or "").strip()
    if not key:
        return False, "Please enter your activation key."
    if not _KEY_RE.match(key):
        return False, "Invalid key format. Keys look like tl_ followed by 32 characters."

    auth = {"Authorization": f"Bearer {key}", "X-Product-ID": PRODUCT_ID}
    try:
        result = _client().get_usage(auth=auth)
    except Exception:
        return False, "Cannot reach server. Check your internet connection."

    if "error" in result:
        code = (result.get("code") or "").strip().upper()
        if code == "INVALID_KEY":
            return False, "Invalid activation key."
        if code == "SUBSCRIPTION_INACTIVE":
            return False, "Subscription inactive. Check your account."
        return False, result.get("error", "Validation failed.")

    server_product = result.get("product_id", "")
    if server_product and server_product != PRODUCT_ID:
        return False, "This key belongs to a different product. Use your AI Segmentation key."

    save_auth_token(key)
    return True, "Activation key verified!"


def revalidate_stored_key() -> bool:
    """Re-check the stored activation key against the server.

    Returns True if the key is still valid (or if no key is stored).
    Returns False and clears auth if the server rejects the key.
    Network errors are silently ignored (benefit of the doubt).
    """
    key = get_auth_token()
    if not key:
        return False

    auth = {"Authorization": f"Bearer {key}", "X-Product-ID": PRODUCT_ID}
    try:
        result = _client().get_usage(auth=auth)
    except Exception:
        return True

    if "error" in result:
        code = (result.get("code") or "").strip().upper()
        if code in ("INVALID_KEY", "SUBSCRIPTION_INACTIVE"):
            clear_auth()
            return False

    return True
