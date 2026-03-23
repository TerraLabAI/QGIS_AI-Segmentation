# SHARED MODULE v1.0 — keep in sync between AI Canvas and AI Segmentation
"""Shared activation logic for TerraLab QGIS plugins."""

from typing import Optional, Tuple

from .constants import NEWSLETTER_URL, PRODUCTS

UNLOCK_CODES = ["fromage", "baguette"]


def _get_settings(settings=None):
    """Get or create QgsSettings."""
    if settings is not None:
        return settings
    from qgis.core import QgsSettings

    return QgsSettings()


def _prefix(product_id: str) -> str:
    return PRODUCTS[product_id]["qsettings_prefix"]


def is_activated(product_id: str, settings=None) -> bool:
    s = _get_settings(settings)
    return s.value(f"{_prefix(product_id)}activated", False, type=bool)


def activate_plugin(product_id: str, code: str, settings=None) -> Tuple[bool, str]:
    if code.strip().lower() in UNLOCK_CODES:
        s = _get_settings(settings)
        s.setValue(f"{_prefix(product_id)}activated", True)
        return True, "Plugin activated successfully!"
    return False, "Invalid code. Please check and try again."


def get_user_email(product_id: str, settings=None) -> str:
    s = _get_settings(settings)
    email = s.value(f"{_prefix(product_id)}user_email", "")
    if email:
        return email
    # Check sibling plugins for email
    for pid in PRODUCTS:
        if pid != product_id:
            sibling_email = s.value(f"{_prefix(pid)}user_email", "")
            if sibling_email:
                s.setValue(f"{_prefix(product_id)}user_email", sibling_email)
                return sibling_email
    return ""


def set_user_email(product_id: str, email: str, settings=None):
    s = _get_settings(settings)
    s.setValue(f"{_prefix(product_id)}user_email", email)


def detect_sibling_activation(product_id: str, settings=None) -> Optional[str]:
    """Check if any sibling plugin is activated. Returns sibling product_id or None."""
    s = _get_settings(settings)
    for pid in PRODUCTS:
        if pid != product_id and is_activated(pid, s):
            return pid
    return None


def auto_activate_from_sibling(product_id: str, settings=None) -> bool:
    """Auto-activate if a sibling is already activated. Returns True if auto-activated."""
    s = _get_settings(settings)
    sibling = detect_sibling_activation(product_id, s)
    if sibling:
        s.setValue(f"{_prefix(product_id)}activated", True)
        return True
    return False


def get_newsletter_url(product_id: str) -> str:
    param = PRODUCTS[product_id]["newsletter_param"]
    return f"{NEWSLETTER_URL}?{param}"
