"""Activation key in QgsAuthManager with legacy QSettings fallback.

Mirrors AI Edit's storage: the key lives encrypted in QgsAuthManager when a
master password is available, with the historical plain-QSettings key kept as
a fallback (and migrated on load). Existing installs that stored the key at
AISegmentation/activation_key keep working unchanged.
"""
from __future__ import annotations

from qgis.core import Qgis, QgsApplication, QgsAuthMethodConfig, QgsSettings

from .logging_utils import log

SETTINGS_PREFIX = "AISegmentation/"
_AUTHCFG_KEY = f"{SETTINGS_PREFIX}authcfg_id"
_LEGACY_KEY = f"{SETTINGS_PREFIX}activation_key"
_MIGRATION_PENDING_KEY = f"{SETTINGS_PREFIX}auth_migration_pending"


def _get_auth_manager():
    try:
        return QgsApplication.authManager()
    except Exception:
        return None


def _can_use_auth_manager() -> bool:
    # Never prompt for the master password - would block QGIS startup.
    am = _get_auth_manager()
    if am is None:
        return False
    try:
        return bool(am.masterPasswordIsSet())
    except Exception:
        return False


def _read_from_auth_manager(authcfg_id: str) -> str:
    am = _get_auth_manager()
    if am is None or not authcfg_id:
        return ""
    try:
        cfg = QgsAuthMethodConfig()
        ok = am.loadAuthenticationConfig(authcfg_id, cfg, True)
        if not ok:
            return ""
        return cfg.config("password", "") or ""
    except Exception:
        return ""


def _store_to_auth_manager(key: str) -> str:
    am = _get_auth_manager()
    if am is None:
        return ""
    try:
        cfg = QgsAuthMethodConfig()
        cfg.setName("AI Segmentation activation key")
        cfg.setMethod("Basic")
        cfg.setConfig("password", key)
        ok = am.storeAuthenticationConfig(cfg)
        if not ok:
            return ""
        return cfg.id() or ""
    except Exception:
        return ""


def get_activation_key(settings=None) -> str:
    s = settings or QgsSettings()
    authcfg_id = s.value(_AUTHCFG_KEY, "", type=str)
    if authcfg_id and _can_use_auth_manager():
        key = _read_from_auth_manager(authcfg_id)
        if key:
            return key
    return s.value(_LEGACY_KEY, "", type=str)


def save_activation(key: str, settings=None) -> None:
    key = (key or "").strip()
    s = settings or QgsSettings()
    if not key:
        clear_activation(s)
        return

    if _can_use_auth_manager():
        authcfg_id = _store_to_auth_manager(key)
        if authcfg_id:
            s.setValue(_AUTHCFG_KEY, authcfg_id)
            s.setValue(_LEGACY_KEY, "")
            s.setValue(_MIGRATION_PENDING_KEY, False)
            return

    s.setValue(_LEGACY_KEY, key)
    s.setValue(_AUTHCFG_KEY, "")
    s.setValue(_MIGRATION_PENDING_KEY, True)


def clear_activation(settings=None) -> None:
    s = settings or QgsSettings()
    authcfg_id = s.value(_AUTHCFG_KEY, "", type=str)
    if authcfg_id:
        am = _get_auth_manager()
        if am is not None:
            try:
                am.removeAuthenticationConfig(authcfg_id)
            except Exception:  # nosec B110
                pass
    s.setValue(_AUTHCFG_KEY, "")
    s.setValue(_LEGACY_KEY, "")
    s.setValue(_MIGRATION_PENDING_KEY, False)
    try:
        s.sync()
    except Exception:  # nosec B110
        pass


def migrate_legacy_key(settings=None) -> bool:
    """Idempotent. False = retry on next plugin load (legacy key still in QSettings)."""
    s = settings or QgsSettings()
    legacy = s.value(_LEGACY_KEY, "", type=str)
    authcfg = s.value(_AUTHCFG_KEY, "", type=str)

    if not legacy:
        return True
    if authcfg:
        s.setValue(_LEGACY_KEY, "")
        s.setValue(_MIGRATION_PENDING_KEY, False)
        try:
            s.sync()
        except Exception:  # nosec B110
            pass
        return True

    if not _can_use_auth_manager():
        # No master password = QgsAuthManager unavailable. QSettings fallback
        # is the normal path for most users; retry silently next load.
        s.setValue(_MIGRATION_PENDING_KEY, True)
        return False

    new_id = _store_to_auth_manager(legacy)
    if not new_id:
        s.setValue(_MIGRATION_PENDING_KEY, True)
        log("Auth migration failed: storeAuthenticationConfig returned empty id",
            Qgis.MessageLevel.Warning)
        return False

    s.setValue(_AUTHCFG_KEY, new_id)
    s.setValue(_LEGACY_KEY, "")
    s.setValue(_MIGRATION_PENDING_KEY, False)
    try:
        s.sync()
    except Exception:  # nosec B110
        pass
    return True
