






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
        loaded = am.loadAuthenticationConfig(authcfg_id, cfg, True)
        if isinstance(loaded, tuple):
            ok, cfg = (loaded + (cfg,))[:2]
        else:
            ok = loaded
        if not ok or cfg is None:
            return ""
        return cfg.config("password", "") or ""
    except Exception:
        return ""


def _store_to_auth_manager(key: str, authcfg_id: str = "") -> str:
    am = _get_auth_manager()
    if am is None:
        return ""
    try:
        cfg = QgsAuthMethodConfig()
        cfg.setName("AI Segmentation activation key")
        cfg.setMethod("Basic")
        cfg.setConfig("password", key)
        if authcfg_id:
            cfg.setId(authcfg_id)
            if am.updateAuthenticationConfig(cfg):
                return authcfg_id
            cfg.setId("")
        stored = am.storeAuthenticationConfig(cfg)
        if isinstance(stored, tuple):
            ok, cfg = (stored + (cfg,))[:2]
        else:
            ok = stored
        if not ok or cfg is None:
            return ""
        return cfg.id() or ""
    except Exception:
        return ""


def get_activation_key(settings=None) -> str:







    s = settings or QgsSettings()
    legacy = s.value(_LEGACY_KEY, "", type=str)
    if legacy:
        return legacy
    authcfg_id = s.value(_AUTHCFG_KEY, "", type=str)
    if authcfg_id and _can_use_auth_manager():
        key = _read_from_auth_manager(authcfg_id)
        if key:
            return key
    return ""


def activation_key_is_locked(settings=None) -> bool:







    s = settings or QgsSettings()
    if not s.value(_AUTHCFG_KEY, "", type=str):
        return False
    if s.value(_LEGACY_KEY, "", type=str):
        return False
    return not _can_use_auth_manager()


def save_activation(key: str, settings=None) -> None:
    key = (key or "").strip()
    s = settings or QgsSettings()
    if not key:
        clear_activation(s)
        return

    if _can_use_auth_manager():
        authcfg_id = _store_to_auth_manager(key, s.value(_AUTHCFG_KEY, "", type=str))
        if authcfg_id:
            s.setValue(_AUTHCFG_KEY, authcfg_id)
            s.setValue(_LEGACY_KEY, "")
            s.setValue(_MIGRATION_PENDING_KEY, False)
            return

    s.setValue(_LEGACY_KEY, key)
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


def migrate_legacy_activation_key(settings=None) -> bool:

    s = settings or QgsSettings()
    legacy = s.value(_LEGACY_KEY, "", type=str)
    authcfg = s.value(_AUTHCFG_KEY, "", type=str)

    if not legacy:
        return True
    if authcfg and _can_use_auth_manager() and _read_from_auth_manager(authcfg) == legacy:
        s.setValue(_LEGACY_KEY, "")
        s.setValue(_MIGRATION_PENDING_KEY, False)
        try:
            s.sync()
        except Exception:  # nosec B110
            pass
        return True

    if not _can_use_auth_manager():


        s.setValue(_MIGRATION_PENDING_KEY, True)
        return False

    new_id = _store_to_auth_manager(legacy, authcfg)
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
