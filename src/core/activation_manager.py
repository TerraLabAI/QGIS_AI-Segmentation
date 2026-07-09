"""Activation manager for the AI Segmentation plugin."""
from __future__ import annotations

import re

from qgis.core import QgsSettings

from .auth_helper import (
    clear_activation as _auth_clear_activation,
)
from .auth_helper import (
    get_activation_key as _auth_get_activation_key,
)
from .auth_helper import (
    migrate_legacy_key as _auth_migrate_legacy_key,
)
from .auth_helper import (
    save_activation as _auth_save_activation,
)

PRODUCT_ID = "ai-segmentation"

_KEY_RE = re.compile(r"^tl_[0-9a-f]{32}$")

SETTINGS_PREFIX = "AISegmentation/"
TERRALAB_PREFIX = "TerraLab/"

TUTORIAL_URL_FALLBACK = "https://youtu.be/lbADk75l-mk?si=q6WnwyV2NcmQYuhI"
TERMS_URL = (
    "https://terra-lab.ai/terms-of-sale"
    "?utm_source=qgis&utm_medium=plugin&utm_campaign=ai-segmentation"
    "&utm_content=settings_terms"
)
PRIVACY_URL = (
    "https://terra-lab.ai/privacy-policy"
    "?utm_source=qgis&utm_medium=plugin&utm_campaign=ai-segmentation"
    "&utm_content=settings_privacy"
)
DASHBOARD_URL = (
    "https://terra-lab.ai/dashboard/ai-segmentation"
    "?utm_source=qgis&utm_medium=plugin&utm_campaign=ai-segmentation"
    "&utm_content=dashboard"
)

_cached_config: dict | None = None


def _client():
    from ..api.terralab_client import TerraLabClient
    return TerraLabClient()


# -- session state ---------------------------------------------------------


def get_auth_token(settings=None) -> str:
    return _auth_get_activation_key(settings)


def save_auth_token(token: str, settings=None):
    _auth_save_activation(token, settings)
    s = settings or QgsSettings()
    s.setValue(f"{SETTINGS_PREFIX}activated", bool((token or "").strip()))


def clear_auth(settings=None):
    _auth_clear_activation(settings)
    s = settings or QgsSettings()
    s.setValue(f"{SETTINGS_PREFIX}activated", False)


def migrate_legacy_key(settings=None) -> bool:
    """Move any QSettings-only key into QgsAuthManager. Idempotent."""
    return _auth_migrate_legacy_key(settings)


def is_plugin_activated(settings=None) -> bool:
    return bool(get_auth_token(settings))


# -- terms of service consent (required to run a segmentation) -------------


def has_tos_accepted(settings=None) -> bool:
    """Whether the Terms + Privacy box is currently ticked.

    Defaults to True (checked) to remove first-run friction: the box shows
    pre-ticked and the user still performs an affirmative act (clicking
    Detect / Start with the agreement text right there) before anything runs.
    Explicitly unticking persists False, which re-adds the gate. Stored
    separately from telemetry consent (optional opt-in, handled elsewhere).
    """
    s = settings or QgsSettings()
    return bool(s.value(f"{SETTINGS_PREFIX}tos_accepted", True, type=bool))


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
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Product-ID": PRODUCT_ID,
    }
    # Anonymous per-machine hash so the server can apply the device limit.
    # Best-effort: a hash failure must never strip auth.
    try:
        from .device_id import get_device_hash, get_device_platform

        headers["X-Device-Hash"] = get_device_hash()
        platform = get_device_platform()
        if platform:
            headers["X-Device-Platform"] = platform
    except Exception:  # nosec B110
        pass
    return headers


# -- server config ---------------------------------------------------------


def get_server_config() -> dict:
    """Return the cached server config, or {} if not yet fetched.

    Cache-only by design: this is called on the GUI thread (kill-switch and
    tutorial-url lookups), so it must NEVER do network here. The fetch happens
    once off-thread via the plugin's hidden config-prefetch task, which calls
    set_cached_config(). Callers all fail open on an empty dict.
    """
    return _cached_config or {}


def set_cached_config(config: dict) -> None:
    """Populate the config cache from the off-thread prefetch result."""
    global _cached_config
    if isinstance(config, dict) and config:
        _cached_config = config


def is_automatic_mode_enabled() -> bool:
    """Server-side kill switch for Automatic mode.

    Fails open: if the config is unreachable or the field is absent
    (older server), Automatic mode stays available.
    """
    config = get_server_config()
    if not config:
        return True
    return bool(config.get("automatic_mode_enabled", True))


def get_tutorial_url() -> str:
    config = get_server_config()
    if config and "tutorial_url" in config:
        return config["tutorial_url"]
    return TUTORIAL_URL_FALLBACK


def get_terms_url() -> str:
    return TERMS_URL


def get_privacy_url() -> str:
    return PRIVACY_URL


def get_dashboard_url() -> str:
    return DASHBOARD_URL


def get_upgrade_url() -> str:
    """URL for the Pro upgrade checkout/dashboard page with UTM attribution."""
    base = get_dashboard_url().split("?")[0]
    return (
        f"{base}?utm_source=qgis&utm_medium=plugin"
        "&utm_campaign=ai-segmentation-pro&utm_content=upgrade_cta"
    )


# -- activation key validation ---------------------------------------------
#
# The manual key-entry validation helpers that used to live here are gone:
# sign-in is the browser pairing flow (env_setup + pairing_poll_task), and
# the stored key is re-validated by env_setup's async usage fetch. Only the
# rejection-code test survives, shared by both.


def is_rejection_code(code: str) -> bool:
    """True when a /usage error code means the stored key is no longer usable."""
    return (code or "").strip().upper() in ("INVALID_KEY", "SUBSCRIPTION_INACTIVE")
