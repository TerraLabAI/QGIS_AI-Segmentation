"""Activation manager for the AI Segmentation plugin."""
from __future__ import annotations

import re

from qgis.core import QgsSettings

from .auth_helper import SETTINGS_PREFIX
from .auth_helper import (
    clear_activation as _auth_clear_activation,
)
from .auth_helper import (
    get_activation_key as _auth_get_activation_key,
)
from .auth_helper import (
    migrate_legacy_activation_key as _auth_migrate_legacy_key,
)
from .auth_helper import (
    save_activation as _auth_save_activation,
)

PRODUCT_ID = "ai-segmentation"

# Shape of an activation key. Public: the pairing worker checks a key the
# server just handed back before anything persists it.
ACTIVATION_KEY_RE = re.compile(r"^tl_[0-9a-f]{32}$")
_KEY_RE = ACTIVATION_KEY_RE  # old private name, kept for existing callers

TERRALAB_PREFIX = "TerraLab/"

TUTORIAL_URL_FALLBACK = "https://youtu.be/lbADk75l-mk?si=q6WnwyV2NcmQYuhI"
CONTACT_CALL_URL_FALLBACK = "https://calendly.com/barbot-yvann/30min"
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
#
# The store itself lives in config_cache (memory, the disk copy left by an
# earlier session, and the local override merged on top). It is pure Python, so
# the dial readers reach it without going through this QGIS-bound module. What
# stays here is the API the rest of the plugin has always called.


def get_server_config() -> dict:
    """Return the cached server config, or {} if none is available.

    Cache-only by design: this is called on the GUI thread (kill-switch and
    tutorial-url lookups), so it must NEVER do network here. The fetch happens
    once off-thread via the plugin's hidden config-prefetch task, which calls
    set_cached_config(). Callers all fail open on an empty dict.
    """
    from .config_cache import get_config

    return get_config()


def get_server_config_age_s() -> float | None:
    """Seconds since the configuration in use was fetched, or None."""
    from .config_cache import config_age_s

    return config_age_s()


def set_cached_config(config: dict) -> None:
    """Populate the config cache from the off-thread prefetch result.

    Also mirrors it to disk so the next cold start is not empty. The write is
    best-effort and holds only values the server serves to any caller, so a
    failure costs nothing and nothing secret lands on disk.

    The disk copy carries no kill switch on purpose, so what the server turned
    OFF is remembered separately (kill_switch_memory) and consulted only until
    the next fetch lands. Without it a feature withdrawn because it is broken
    came back on at every restart.
    """
    from .config_cache import set_config
    from .kill_switch_memory import remember_from_live_config

    set_config(config)
    remember_from_live_config(config)


def is_feature_enabled(name: str) -> bool:
    """Server kill switch for one named feature, under the `features` key.

    Fails open: an absent map, an absent key or garbage all mean enabled. Only
    an explicit false disables.
    """
    from .server_dials import feature_enabled

    return feature_enabled(name)


def is_automatic_mode_enabled() -> bool:
    """Server-side kill switch for Automatic mode.

    Fails open: if the config is unreachable or both fields are absent
    (older server), Automatic mode stays available. Honours the original
    top-level flag and the generic `features.automatic_mode` switch.
    """
    from .server_dials import automatic_mode_enabled

    return automatic_mode_enabled()


def is_update_recommended(installed_version: str) -> bool:
    """Whether the server asks this build to update (min_recommended_version).

    Fails closed on garbage: no version, an unparsable one on either side, or
    no server value all mean no.
    """
    from .server_dials import is_served_update_recommended

    return is_served_update_recommended(installed_version)


def get_tutorial_url() -> str:
    """The tutorial address: the served one only when it is a usable https URL.

    Guarded rather than trusted. The caller opens this in the browser and pastes
    it into the href of a rich-text label, both on the GUI thread, so a served
    list, number or odd scheme has to be turned away here and not discovered by
    the widget. Anything that is not a plain https web address with a host
    yields the shipped constant.
    """
    from .server_dials import dial_url

    return dial_url("tutorial_url", TUTORIAL_URL_FALLBACK)


def get_contact_call_url() -> str:
    """The book-a-call address behind Contact us, served like the tutorial one.

    Same guard: the caller hands it to the desktop URL handler, so anything
    that is not a plain https web address with a host yields the shipped
    constant."""
    from .server_dials import dial_url

    return dial_url("contact_call_url", CONTACT_CALL_URL_FALLBACK)


def get_terms_url() -> str:
    return TERMS_URL


def get_privacy_url() -> str:
    return PRIVACY_URL


def get_dashboard_url() -> str:
    return DASHBOARD_URL


# Every paid CTA lands on the dashboard checkout for AI Segmentation Pro.
# cta_source names the plugin surface the click came from, so the website can
# tell the km2 wall from the objects wall without a plugin release.
PRO_CHECKOUT_URL_BASE = (
    "https://terra-lab.ai/dashboard"
    "?action=checkout&product=ai-segmentation-pro"
)


def get_pro_checkout_url(cta_source: str) -> str:
    """The Pro checkout URL, tagged with the surface that sent the click."""
    source = "".join(
        ch for ch in str(cta_source or "") if ch.isalnum() or ch == "_"
    ) or "plugin"
    return (
        f"{PRO_CHECKOUT_URL_BASE}&cta_source={source}"
        "&utm_source=qgis&utm_medium=plugin&utm_campaign=ai-segmentation-pro"
    )


def get_upgrade_url() -> str:
    """The account dialog's Upgrade CTA destination."""
    return get_pro_checkout_url("plugin_account_dialog")


# -- activation key validation ---------------------------------------------
#
# The manual key-entry validation helpers that used to live here are gone:
# sign-in is the browser pairing flow (env_setup + pairing_poll_task), and
# the stored key is re-validated by env_setup's async usage fetch. Only the
# rejection-code test survives, shared by both.


def is_rejection_code(code: str) -> bool:
    """True when a /usage error code means the stored key is no longer usable."""
    return (code or "").strip().upper() in ("INVALID_KEY", "SUBSCRIPTION_INACTIVE")
