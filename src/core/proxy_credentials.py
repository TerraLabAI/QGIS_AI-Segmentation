














from __future__ import annotations

from qgis.core import Qgis

from .logging_utils import log



_PROXY_KEY_RENAMES = {
    f"proxy/{old}": f"proxy/{new}"
    for old, new in (
        ("proxyEnabled", "proxy-enabled"),
        ("proxyType", "proxy-type"),
        ("proxyHost", "proxy-host"),
        ("proxyPort", "proxy-port"),
        ("proxyUser", "proxy-user"),
        ("proxyPassword", "proxy-password"),
        ("noProxyUrls", "no-proxy-urls"),
        ("authcfg", "auth-cfg"),
    )
}


def qgis_proxy_setting(settings, key: str, default, value_type=None):


    renamed = _PROXY_KEY_RENAMES.get(key)
    if renamed and settings.contains(renamed):
        key = renamed
    if value_type is None:
        return settings.value(key, default)
    return settings.value(key, default, type=value_type)


def qgis_proxy_credentials() -> tuple[str, str]:















    try:
        from qgis.core import QgsSettings

        settings = QgsSettings()
        if not qgis_proxy_setting(settings, "proxy/proxyEnabled", False, bool):
            return "", ""
        authcfg = qgis_proxy_setting(settings, "proxy/authcfg", "", str) or ""
        if authcfg:
            user, password = _credentials_from_auth_config(authcfg)
            if user:
                return user, password
        user = qgis_proxy_setting(settings, "proxy/proxyUser", "", str) or ""
        password = qgis_proxy_setting(settings, "proxy/proxyPassword", "", str) or ""
        return user, password
    except Exception as err:  # noqa: BLE001
        log(f"Reading the QGIS proxy credentials failed: {type(err).__name__}",
            Qgis.MessageLevel.Warning)
        return "", ""


def _credentials_from_auth_config(authcfg: str) -> tuple[str, str]:





    from qgis.core import QgsApplication, QgsAuthMethodConfig

    auth_mgr = QgsApplication.authManager()
    if auth_mgr is None or not auth_mgr.masterPasswordIsSet():
        return "", ""
    config = QgsAuthMethodConfig()



    loaded = auth_mgr.loadAuthenticationConfig(authcfg, config, True)
    if isinstance(loaded, tuple):
        ok, config = (loaded + (config,))[:2]
    else:
        ok = loaded
    if not ok or config is None:
        return "", ""
    return config.config("username", "") or "", config.config("password", "") or ""
