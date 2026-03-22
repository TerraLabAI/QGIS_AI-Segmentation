"""
Activation manager for the AI Segmentation plugin.
Handles plugin activation state using QSettings.
"""

from typing import Tuple
from qgis.core import QgsSettings

# The unlock codes
UNLOCK_CODES = ["fromage", "baguette"]

# QSettings keys
SETTINGS_PREFIX = "AISegmentation"
ACTIVATION_KEY = f"{SETTINGS_PREFIX}/activated"
PRO_API_KEY_SETTING = f"{SETTINGS_PREFIX}/pro_api_key"

# TerraLab URLs
TERRALAB_WEBSITE = "https://terra-lab.ai"
TERRALAB_NEWSLETTER = "https://terra-lab.ai/mail-verification"


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
    if code.strip().lower() in UNLOCK_CODES:
        settings = QgsSettings()
        settings.setValue(ACTIVATION_KEY, True)
        return True, "Plugin activated successfully!"
    else:
        return False, "Invalid code. Please check and try again."


def deactivate_plugin():
    """Deactivate the plugin (for testing purposes)."""
    settings = QgsSettings()
    settings.setValue(ACTIVATION_KEY, False)


def get_pro_api_key() -> str:
    """Return the stored PRO API key, or empty string if not set."""
    settings = QgsSettings()
    return settings.value(PRO_API_KEY_SETTING, "", type=str)


def set_pro_api_key(key: str) -> None:
    """Store the PRO API key in QSettings."""
    settings = QgsSettings()
    settings.setValue(PRO_API_KEY_SETTING, key.strip())


def get_newsletter_url() -> str:
    """Get the URL for the newsletter signup page."""
    return TERRALAB_NEWSLETTER


def get_website_url() -> str:
    """Get the main TerraLab website URL."""
    return TERRALAB_WEBSITE
