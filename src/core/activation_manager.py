"""Activation manager for the AI Segmentation plugin."""

from typing import Tuple

from qgis.core import QgsSettings

UNLOCK_CODES = ["fromage", "baguette"]
SETTINGS_PREFIX = "AISegmentation"
ACTIVATION_KEY = f"{SETTINGS_PREFIX}/activated"
PRO_API_KEY_SETTING = f"{SETTINGS_PREFIX}/pro_api_key"
TERRALAB_WEBSITE = "https://terra-lab.ai"
TERRALAB_NEWSLETTER = "https://terra-lab.ai/mail-verification"


def is_plugin_activated() -> bool:
    settings = QgsSettings()
    return settings.value(ACTIVATION_KEY, False, type=bool)


def activate_plugin(code: str) -> Tuple[bool, str]:
    if code.strip().lower() in UNLOCK_CODES:
        settings = QgsSettings()
        settings.setValue(ACTIVATION_KEY, True)
        return True, "Plugin activated successfully!"
    return False, "Invalid code. Please check and try again."


def deactivate_plugin():
    settings = QgsSettings()
    settings.setValue(ACTIVATION_KEY, False)


def get_pro_api_key() -> str:
    settings = QgsSettings()
    return settings.value(PRO_API_KEY_SETTING, "", type=str)


def set_pro_api_key(key: str) -> None:
    settings = QgsSettings()
    settings.setValue(PRO_API_KEY_SETTING, key.strip())


def get_newsletter_url() -> str:
    return TERRALAB_NEWSLETTER


def get_website_url() -> str:
    return TERRALAB_WEBSITE
