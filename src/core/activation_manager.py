"""Activation manager for the AI Segmentation plugin."""
from __future__ import annotations

import uuid

from qgis.core import QgsSettings

PRODUCT_ID = "ai-segmentation"

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
    if not key.startswith("tl_"):
        return False, "Invalid key format. Keys start with tl_"

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
