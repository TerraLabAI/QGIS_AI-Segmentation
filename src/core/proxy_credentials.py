"""The proxy user name and password a user configured in QGIS.

One reader for the two places that need them: the requests the plugin makes
while it works, and the package installs it runs as separate programs. A
company proxy that asks for a password blocks both, and reading the setting in
only one of them leaves the other failing with a message about the internet
connection.

QGIS keeps these in two places. The plain settings pair is what the Network
page writes. The authentication configuration is what a user picks instead
when they do not want the password stored in the clear, and it is read only
once the master password is already unlocked: asking for it here would stop a
background request behind a prompt the user cannot connect to anything they
did.
"""
from __future__ import annotations

from qgis.core import Qgis

from .logging_utils import log


def qgis_proxy_credentials() -> tuple[str, str]:
    """Return (user, password), each empty when QGIS has none to give.

    Two empty strings whenever the user has not ticked "use a proxy". QGIS
    leaves the old user name and password in place when the box is unticked,
    and sending a pair the user has retired is not merely useless: a company
    domain that locks an account after three bad attempts would lock it on the
    first install, since every package download retries.

    The authentication configuration is read before the plain pair, which is
    the order QGIS itself resolves them in: it builds the proxy from the plain
    pair and then lets the configuration overwrite it. Reading them the other
    way round would sign the plugin in as somebody else than the rest of QGIS.

    Never logs either value: a proxy user name identifies the person.
    """
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
    except Exception as err:  # noqa: BLE001 - absent credentials, not a crash
        log(f"Reading the QGIS proxy credentials failed: {type(err).__name__}",
            Qgis.MessageLevel.Warning)
        return "", ""


def _credentials_from_auth_config(authcfg: str) -> tuple[str, str]:
    """Read the pair out of a QGIS authentication configuration.

    Two empty strings whenever the master password is still locked, so this
    never raises the prompt itself.
    """
    from qgis.core import QgsApplication, QgsAuthMethodConfig

    auth_mgr = QgsApplication.authManager()
    if auth_mgr is None or not auth_mgr.masterPasswordIsSet():
        return "", ""
    config = QgsAuthMethodConfig()
    # The binding hands the config back as the second half of a tuple rather
    # than filling the one passed in, so read the answer, never the argument:
    # testing the call's truth alone tests a tuple, which is always true.
    loaded = auth_mgr.loadAuthenticationConfig(authcfg, config, True)
    if isinstance(loaded, tuple):
        ok, config = (loaded + (config,))[:2]
    else:
        ok = loaded
    if not ok or config is None:
        return "", ""
    return config.config("username", "") or "", config.config("password", "") or ""
