"""
Activation manager for the AI Segmentation plugin.
Handles plugin activation state using QSettings.
"""

from typing import Tuple
from qgis.core import QgsSettings

# The unlock code
UNLOCK_CODE = "baguette"

# QSettings keys
SETTINGS_PREFIX = "AISegmentation"
ACTIVATION_KEY = f"{SETTINGS_PREFIX}/activated"

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
    if code.strip().lower() == UNLOCK_CODE.lower():
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
    return TERRALAB_NEWSLETTER


def get_website_url() -> str:
    """Get the main TerraLab website URL."""
    return TERRALAB_WEBSITE
