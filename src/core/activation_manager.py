"""
Activation manager for the AI Segmentation plugin.
Handles plugin activation state using QSettings.
"""

import json
from typing import Tuple, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

from qgis.core import QgsSettings

# Hardcoded fallback codes (used when server is unreachable)
_FALLBACK_CODES = ["fromage", "baguette"]

# QSettings keys
SETTINGS_PREFIX = "AISegmentation"
ACTIVATION_KEY = f"{SETTINGS_PREFIX}/activated"
TERRALAB_PREFIX = "TerraLab"

# TerraLab URLs
TERRALAB_WEBSITE = "https://terra-lab.ai"
TERRALAB_NEWSLETTER = "https://terra-lab.ai/mail-verification"
_CONFIG_URL = f"{TERRALAB_WEBSITE}/api/plugin/config?product=ai-segmentation"

# Server config cache (in-memory, one fetch per session)
_cached_config: Optional[dict] = None


def _fetch_server_config() -> Optional[dict]:
    """Fetch plugin config from server. Returns None on failure."""
    global _cached_config
    if _cached_config is not None:
        return _cached_config
    try:
        req = Request(_CONFIG_URL, headers={"Accept": "application/json"})
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            _cached_config = data
            return data
    except (URLError, json.JSONDecodeError, OSError):
        return None


def _get_unlock_codes() -> list:
    """Get unlock codes from server config, falling back to hardcoded list."""
    config = _fetch_server_config()
    if config and "verification_codes" in config:
        return [c.lower() for c in config["verification_codes"]]
    return _FALLBACK_CODES


def is_plugin_activated() -> bool:
    """Check if the plugin has been activated."""
    settings = QgsSettings()
    return settings.value(ACTIVATION_KEY, False, type=bool)


def activate_plugin(code: str) -> Tuple[bool, str]:
    """
    Attempt to activate the plugin with the given code.

    Returns:
        (success, message) tuple
    """
    codes = _get_unlock_codes()
    if code.strip().lower() in codes:
        settings = QgsSettings()
        settings.setValue(ACTIVATION_KEY, True)
        return True, "Plugin activated successfully!"
    else:
        return False, "Invalid code. Please check and try again."


def deactivate_plugin():
    """Deactivate the plugin (for testing purposes)."""
    settings = QgsSettings()
    settings.setValue(ACTIVATION_KEY, False)


def get_newsletter_url() -> str:
    """Get the URL for the newsletter signup page."""
    config = _fetch_server_config()
    if config and "newsletter_url" in config:
        return config["newsletter_url"]
    return TERRALAB_NEWSLETTER


def get_tutorial_url() -> str:
    """Get tutorial URL from server config, falling back to hardcoded default."""
    config = _fetch_server_config()
    if config and "tutorial_url" in config:
        return config["tutorial_url"]
    return "https://youtu.be/lbADk75l-mk?si=q6WnwyV2NcmQYuhI"


def get_website_url() -> str:
    """Get the main TerraLab website URL."""
    return TERRALAB_WEBSITE


def get_shared_email() -> str:
    """Get email from shared TerraLab namespace (set by any plugin)."""
    settings = QgsSettings()
    return settings.value(f"{TERRALAB_PREFIX}/user_email", "")


def save_shared_email(email: str):
    """Save email to shared TerraLab namespace for cross-plugin use."""
    settings = QgsSettings()
    settings.setValue(f"{TERRALAB_PREFIX}/user_email", email.strip())


def get_newsletter_url_with_email(email: str) -> str:
    """Build newsletter URL with pre-filled email."""
    from urllib.parse import urlencode
    base = get_newsletter_url()
    if email:
        return f"{base}?{urlencode({'email': email, 'plugin': 'ai-segmentation'})}"
    return base
