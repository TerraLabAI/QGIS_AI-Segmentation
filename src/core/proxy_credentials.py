














from __future__ import annotations

from qgis.core import Qgis

from .logging_utils import log


def qgis_proxy_credentials() -> tuple[str, str]:















    try:
        from qgis.core import QgsSettings

        settings = QgsSettings()
        if not settings.value("proxy/proxyEnabled", False, type=bool):
            return "", ""
        authcfg = settings.value("proxy/authcfg", "", type=str) or ""
        if authcfg:
            user, password = _credentials_from_auth_config(authcfg)
            if user:
                return user, password
        user = settings.value("proxy/proxyUser", "", type=str) or ""
        password = settings.value("proxy/proxyPassword", "", type=str) or ""
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
