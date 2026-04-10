"""
Activation manager for the AI Segmentation plugin.
Handles plugin activation state using QSettings.
"""
from __future__ import annotations

import json

from qgis.core import QgsBlockingNetworkRequest, QgsSettings
from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest

# Hardcoded fallback codes (used when server is unreachable)
_FALLBACK_CODES = ["fromage", "baguette"]

# QSettings keys
SETTINGS_PREFIX = "AISegmentation"
ACTIVATION_KEY = f"{SETTINGS_PREFIX}/activated"
TERRALAB_PREFIX = "TerraLab"

# TerraLab URLs
TERRALAB_WEBSITE = "https://terra-lab.ai?utm_source=qgis&utm_medium=plugin&utm_campaign=ai-segmentation&utm_content=website"
TERRALAB_NEWSLETTER = "https://terra-lab.ai/mail-verification?utm_source=qgis&utm_medium=plugin&utm_campaign=ai-segmentation&utm_content=get_code"
_CONFIG_URL = f"{TERRALAB_WEBSITE}/api/plugin/config?product=ai-segmentation"

# Server config cache (in-memory, one fetch per session)
_cached_config: dict | None = None


def _fetch_server_config() -> dict | None:
    """Fetch plugin config from server. Returns None on failure."""
    global _cached_config
    if _cached_config is not None:
        return _cached_config
    try:
        if not _CONFIG_URL.startswith("https://"):
            return None
        req = QNetworkRequest(QUrl(_CONFIG_URL))
        req.setRawHeader(b"Accept", b"application/json")
        blocker = QgsBlockingNetworkRequest()
        err = blocker.get(req)
        if err != QgsBlockingNetworkRequest.NoError:
            return None
        reply = blocker.reply()
        data = json.loads(bytes(reply.content()).decode("utf-8"))
        _cached_config = data
        return data
    except (json.JSONDecodeError, Exception):
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


def activate_plugin(code: str) -> tuple[bool, str]:
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
    return False, "Invalid code. Please check and try again."


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
        sep = "&" if "?" in base else "?"
        return f"{base}{sep}{urlencode({'email': email, 'plugin': 'ai-segmentation'})}"
    return base
