
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



ACTIVATION_KEY_RE = re.compile(r"^tl_[0-9a-f]{32}$")

TERRALAB_PREFIX = "TerraLab/"

TUTORIAL_URL_FALLBACK = "https://youtu.be/lbADk75l-mk?si=q6WnwyV2NcmQYuhI"
CONTACT_CALL_URL_FALLBACK = "https://calendly.com/barbot-yvann/30min"


_LEGAL_UTM_STEM = "utm_source=qgis&utm_medium=plugin&utm_campaign=ai-segmentation"

TERMS_URL_FALLBACK = "https://terra-lab.ai/terms-of-sale"
CONSENT_TERMS_URL_FALLBACK = "https://terra-lab.ai/terms-of-use"
PRIVACY_URL_FALLBACK = "https://terra-lab.ai/privacy-policy"
TERMS_URL = f"{TERMS_URL_FALLBACK}?{_LEGAL_UTM_STEM}&utm_content=settings_terms"
PRIVACY_URL = f"{PRIVACY_URL_FALLBACK}?{_LEGAL_UTM_STEM}&utm_content=settings_privacy"

SUPPORT_EMAIL_FALLBACK = "yvann.barbot@terra-lab.ai"
DASHBOARD_URL_FALLBACK = "https://terra-lab.ai/dashboard/ai-segmentation"
_DASHBOARD_UTM = (
    "utm_source=qgis&utm_medium=plugin&utm_campaign=ai-segmentation"
    "&utm_content=dashboard"
)
DASHBOARD_URL = f"{DASHBOARD_URL_FALLBACK}?{_DASHBOARD_UTM}"


def _client():
    from ..api.terralab_client import TerraLabClient
    return TerraLabClient()




_auth_revision = 0


def auth_revision() -> int:

    return _auth_revision


def get_auth_token(settings=None) -> str:
    return _auth_get_activation_key(settings)


def _forget_account_fingerprint() -> None:








    try:
        from .presets.run_history_cache import reset_account_fingerprint_cache

        reset_account_fingerprint_cache()
    except Exception:  # noqa: BLE001
        pass  # nosec B110


def save_auth_token(token: str, settings=None):
    global _auth_revision
    _auth_revision += 1
    _auth_save_activation(token, settings)
    _forget_account_fingerprint()
    s = settings or QgsSettings()
    s.setValue(f"{SETTINGS_PREFIX}activated", bool((token or "").strip()))


def clear_auth(settings=None):
    global _auth_revision
    _auth_revision += 1
    _auth_clear_activation(settings)
    _forget_account_fingerprint()
    s = settings or QgsSettings()
    s.setValue(f"{SETTINGS_PREFIX}activated", False)


def migrate_legacy_key(settings=None) -> bool:

    return _auth_migrate_legacy_key(settings)


def is_plugin_activated(settings=None) -> bool:
    return bool(get_auth_token(settings))





def has_tos_accepted(settings=None) -> bool:











    s = settings or QgsSettings()
    return bool(s.value(f"{SETTINGS_PREFIX}tos_accepted", False, type=bool))


def has_tos_locked(settings=None) -> bool:






    s = settings or QgsSettings()
    return bool(s.value(f"{SETTINGS_PREFIX}tos_locked", False, type=bool))


def lock_tos():

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


    try:
        from .device_id import get_device_hash, get_device_platform

        headers["X-Device-Hash"] = get_device_hash()
        platform = get_device_platform()
        if platform:
            headers["X-Device-Platform"] = platform
    except Exception:  # nosec B110
        pass
    return headers










def get_server_config() -> dict:







    from .config_cache import get_config

    return get_config()


def set_cached_config(config: dict) -> None:











    from .config_cache import set_config
    from .kill_switch_memory import remember_from_live_config

    set_config(config)
    remember_from_live_config(config)


def is_automatic_mode_enabled() -> bool:






    from .server_dials import automatic_mode_enabled

    return automatic_mode_enabled()


def is_update_recommended(installed_version: str) -> bool:





    from .server_dials import is_served_update_recommended

    return is_served_update_recommended(installed_version)


def is_update_available(installed_version: str) -> bool:




    from .server_dials import is_served_update_available

    return is_served_update_available(installed_version)


def get_latest_version() -> str | None:

    from .server_dials import served_latest_version

    return served_latest_version()


def get_release_notes_line() -> str | None:

    from .server_dials import served_release_notes_line

    return served_release_notes_line()


def get_marketplace_url() -> str:




    from .server_dials import served_marketplace_url

    return served_marketplace_url()


def get_tutorial_url() -> str:








    from .server_dials import dial_url

    return dial_url("tutorial_url", TUTORIAL_URL_FALLBACK)


def get_contact_call_url() -> str:





    from .server_dials import dial_url

    return dial_url("contact_call_url", CONTACT_CALL_URL_FALLBACK)


def with_legal_utm(url: str, content: str) -> str:

    joiner = "&" if "?" in url else "?"
    return f"{url}{joiner}{_LEGAL_UTM_STEM}&utm_content={content}"


def get_terms_url(content: str = "settings_terms") -> str:

    from .server_dials import dial_url

    return with_legal_utm(dial_url("terms_url", TERMS_URL_FALLBACK), content)


def get_consent_terms_url(content: str = "consent_terms") -> str:

    from .server_dials import dial_url

    return with_legal_utm(dial_url("consent_terms_url", CONSENT_TERMS_URL_FALLBACK), content)


def get_privacy_url(content: str = "settings_privacy") -> str:

    from .server_dials import dial_url

    return with_legal_utm(dial_url("privacy_url", PRIVACY_URL_FALLBACK), content)


_SUPPORT_EMAIL_MAX_CHARS = 120


def get_support_email(fallback: str = SUPPORT_EMAIL_FALLBACK) -> str:


    from .server_dials import dial_str

    served = dial_str("support_email", fallback)
    if (
        served.count("@") == 1
        and len(served) <= _SUPPORT_EMAIL_MAX_CHARS
        and served.isprintable()
        and not any(c.isspace() for c in served)
        and not served.startswith("@")
        and not served.endswith("@")
    ):
        return served
    return fallback


def get_dashboard_url() -> str:


    from .server_dials import dial_url

    url = dial_url("tuning.links.dashboard", DASHBOARD_URL_FALLBACK)
    joiner = "&" if "?" in url else "?"
    return f"{url}{joiner}{_DASHBOARD_UTM}"





PRO_CHECKOUT_URL_BASE = (
    "https://terra-lab.ai/dashboard"
    "?action=checkout&product=ai-segmentation-pro"
)


def get_pro_checkout_url(cta_source: str) -> str:

    from .server_dials import dial_url

    source = "".join(
        ch for ch in str(cta_source or "") if ch.isalnum() or ch == "_"
    ) or "plugin"


    base = dial_url("tuning.links.pro_checkout", PRO_CHECKOUT_URL_BASE)
    if "?" not in base:
        base = PRO_CHECKOUT_URL_BASE
    return (
        f"{base}&cta_source={source}"
        "&utm_source=qgis&utm_medium=plugin&utm_campaign=ai-segmentation-pro"
    )


def get_upgrade_url() -> str:

    return get_pro_checkout_url("plugin_account_dialog")





PLANS_URL_FALLBACK = "https://terra-lab.ai/pricing"


def book_a_call_url() -> str:






    from .server_dials import dial_url

    return dial_url("contact.book_call_url", "")


def get_plans_page_url(cta_source: str = "plugin") -> str:





    from .server_dials import dial_url

    source = "".join(
        ch for ch in str(cta_source or "") if ch.isalnum() or ch == "_"
    ) or "plugin"
    url = dial_url("plans_url", PLANS_URL_FALLBACK)
    joiner = "&" if "?" in url else "?"
    return (f"{url}{joiner}utm_source=qgis&utm_medium=plugin"
            f"&utm_campaign=ai-segmentation-pro&utm_content={source}")








