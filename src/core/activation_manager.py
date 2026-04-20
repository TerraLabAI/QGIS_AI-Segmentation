"""TerraLab account state for the AI Segmentation plugin (v1.0.0).

Uses the shared QSettings('TerraLab/*') namespace so a user signed in through
AI Edit is recognised here without re-authentication, and vice versa.

Backwards-compatible stubs kept:
- is_plugin_activated() → true when a TerraLab token is stored.
- get_tutorial_url() → fetched from /api/plugin/config, with fallback.
- get_shared_email() / save_shared_email() → shared TerraLab email helpers.
"""
from __future__ import annotations

import uuid

from qgis.core import QgsSettings

PRODUCT_ID = "ai-segmentation"

TERRALAB_PREFIX = "TerraLab"
AUTH_TOKEN_KEY = f"{TERRALAB_PREFIX}/auth/token"
AUTH_EMAIL_KEY = f"{TERRALAB_PREFIX}/auth/email"
AUTH_PRODUCT_KEY = f"{TERRALAB_PREFIX}/auth/product"
SHARED_EMAIL_KEY = f"{TERRALAB_PREFIX}/user_email"
DEVICE_ID_KEY = f"{TERRALAB_PREFIX}/device_id"

_LEGACY_ACTIVATION_KEY = "AISegmentation/activated"

TUTORIAL_URL_FALLBACK = "https://youtu.be/lbADk75l-mk?si=q6WnwyV2NcmQYuhI"
_SIGN_IN_BASE = (
    "https://terra-lab.ai/en/login"
    "?utm_source=qgis&utm_medium=plugin&utm_campaign=ai-segmentation"
    "&utm_content=sign_in&product=ai-segmentation"
)

_cached_config: dict | None = None


def _client():
    from ..api.terralab_client import TerraLabClient
    return TerraLabClient()


# -- session state (shared with AI Edit via QSettings prefix) --------------


def get_auth_token(settings=None) -> str:
    s = settings or QgsSettings()
    return s.value(AUTH_TOKEN_KEY, "") or ""


def save_auth_token(token: str, email: str, product: str = PRODUCT_ID, settings=None):
    s = settings or QgsSettings()
    s.setValue(AUTH_TOKEN_KEY, token.strip())
    s.setValue(AUTH_EMAIL_KEY, email.strip())
    s.setValue(AUTH_PRODUCT_KEY, product)
    if email:
        s.setValue(SHARED_EMAIL_KEY, email.strip())


def clear_auth(settings=None):
    s = settings or QgsSettings()
    s.setValue(AUTH_TOKEN_KEY, "")
    s.setValue(AUTH_EMAIL_KEY, "")
    s.setValue(AUTH_PRODUCT_KEY, "")
    # Legacy key kept false so old builds don't re-trigger activation UI.
    s.setValue(_LEGACY_ACTIVATION_KEY, False)


def is_plugin_activated(settings=None) -> bool:
    """Activated means: the user has a TerraLab auth token stored."""
    return bool(get_auth_token(settings))


def get_auth_header(settings=None) -> dict:
    token = get_auth_token(settings)
    if not token:
        return {}
    return {
        "Authorization": f"Bearer {token}",
        "X-Product-ID": PRODUCT_ID,
    }


# -- shared TerraLab email (legacy key name kept) --------------------------


def get_shared_email(settings=None) -> str:
    s = settings or QgsSettings()
    return s.value(SHARED_EMAIL_KEY, "") or ""


def save_shared_email(email: str, settings=None):
    s = settings or QgsSettings()
    s.setValue(SHARED_EMAIL_KEY, email.strip())


# -- device id (stable per machine) ----------------------------------------


def get_device_id(settings=None) -> str:
    s = settings or QgsSettings()
    device_id = s.value(DEVICE_ID_KEY, "")
    if not device_id:
        device_id = str(uuid.uuid4())
        s.setValue(DEVICE_ID_KEY, device_id)
    return device_id


# -- server config (tutorial URL, etc.) ------------------------------------


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
        pass
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


# -- activation key validation -----------------------------------------------


def validate_key_with_server(key: str) -> tuple[bool, str]:
    """Validate an activation key against the server.

    Returns (success, english_message) — caller wraps message with tr().
    """
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

    save_auth_token(key, result.get("email", ""), PRODUCT_ID)
    return True, "Activation key verified!"
