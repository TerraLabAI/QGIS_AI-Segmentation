









from __future__ import annotations

import os
import sys

from qgis.core import Qgis

from .gui_thread import on_gui_thread
from .logging_utils import log as _log


def _insecure_install_opt_in() -> bool:











    if os.environ.get("QGIS_AI_ALLOW_INSECURE_INSTALL", "").strip().lower() in ("1", "true", "yes"):
        return True
    try:
        from qgis.PyQt.QtCore import QSettings

        val = QSettings().value("TerraLab/allow_insecure_install", False, type=bool)
        return bool(val)
    except Exception:  # noqa: BLE001
        return False





CA_BUNDLE_ENV_VARS = ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
                      "CURL_CA_BUNDLE")


def env_without_ca_bundle_overrides(env: dict) -> tuple[dict, list[str]]:







    out = dict(env)
    dropped = [name for name in CA_BUNDLE_ENV_VARS if out.pop(name, None)]
    return out, dropped


def windows_trust_store_bundle(cache_dir: str) -> tuple[str | None, int]:

















    if not sys.platform.startswith("win"):
        return None, 0
    try:
        import ssl

        seen: set[bytes] = set()
        pems: list[str] = []
        try:
            import certifi

            with open(certifi.where(), encoding="utf-8") as fh:
                pems.append(fh.read())
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        for store in ("ROOT", "CA"):
            try:
                entries = ssl.enum_certificates(store)
            except Exception:  # noqa: BLE001  # nosec B112
                continue
            for der, _encoding, trust in entries:


                if trust is not True and (
                        not trust or "1.3.6.1.5.5.7.3.1" not in trust):
                    continue
                if der in seen:
                    continue
                seen.add(der)
                pems.append(ssl.DER_cert_to_PEM_cert(der))
        if not seen:
            return None, 0
        path = os.path.join(cache_dir, "os_trust_store.pem")
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("\n".join(pems))
        os.replace(tmp, path)
        return path, len(seen)
    except Exception as e:  # noqa: BLE001
        _log(f"Could not export the OS certificate store: {e}",
             Qgis.MessageLevel.Warning)
        return None, 0



_auto_config_proxy: dict[str, tuple[str | None]] = {}


def _get_auto_config_proxy_settings() -> str | None:


























    cached = _auto_config_proxy.get("answer")
    if cached is not None:
        return cached[0]


    if on_gui_thread():
        return None
    try:
        from urllib.parse import quote as url_quote

        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtNetwork import (
            QNetworkProxy,
            QNetworkProxyFactory,
            QNetworkProxyQuery,
        )

        from .qt_compat import resolve_qt_enum

        http_type = resolve_qt_enum(QNetworkProxy, "ProxyType", "HttpProxy")
        caching_type = resolve_qt_enum(QNetworkProxy, "ProxyType", "HttpCachingProxy")
        query = QNetworkProxyQuery(QUrl("https://pypi.org/simple"))
        for proxy in QNetworkProxyFactory.systemProxyForQuery(query) or []:
            if proxy.type() not in (http_type, caching_type):
                continue
            host = proxy.hostName()
            if not host:
                continue
            from .proxy_credentials import qgis_proxy_credentials

            user, password = qgis_proxy_credentials()
            prefix = ""
            if user:
                prefix = url_quote(user, safe="")
                if password:
                    prefix += ":" + url_quote(password, safe="")
                prefix += "@"
            port = proxy.port()
            resolved = (f"http://{prefix}{host}:{port}" if port
                        else f"http://{prefix}{host}")
            _auto_config_proxy["answer"] = (resolved,)
            _log("Using the proxy the machine's automatic configuration names",
                 Qgis.MessageLevel.Info)
            return resolved
    except Exception as e:  # noqa: BLE001
        _log(f"Could not resolve an automatic proxy configuration: {e}",
             Qgis.MessageLevel.Warning)
    _auto_config_proxy["answer"] = (None,)
    return None


def _get_system_proxy_settings() -> str | None:







    try:
        import urllib.request

        proxies = urllib.request.getproxies()
        proxy_url = proxies.get("https") or proxies.get("http")
        plain = proxies.get("http") or ""
        if (sys.platform == "win32" and proxy_url and plain
                and proxy_url.lower().startswith("https://")
                and plain.lower() == "http://" + proxy_url[len("https://"):].lower()):




            proxy_url = plain
        if proxy_url and proxy_url.lower().startswith(("http://", "https://")):
            return _with_qgis_proxy_credentials(proxy_url)
    except Exception as e:
        _log(f"Could not read system proxy settings: {e}", Qgis.MessageLevel.Warning)
    return None


def _with_qgis_proxy_credentials(proxy_url: str) -> str:







    from urllib.parse import quote as url_quote
    from urllib.parse import urlsplit

    parts = urlsplit(proxy_url)
    if "@" in parts.netloc:
        return proxy_url
    from .proxy_credentials import qgis_proxy_credentials

    user, password = qgis_proxy_credentials()
    if not user:
        return proxy_url
    prefix = url_quote(user, safe="")
    if password:
        prefix += ":" + url_quote(password, safe="")
    return f"{parts.scheme}://{prefix}@{parts.netloc}{parts.path}"


def _get_effective_proxy_url() -> str | None:











    from .qgis_proxy_reader import qgis_proxy_url

    return qgis_proxy_url() or _get_system_proxy_settings() or _get_auto_config_proxy_settings()


def _get_pip_proxy_args() -> list[str]:








    proxy_url = _get_effective_proxy_url()
    if not proxy_url:
        return []
    if "@" in proxy_url:
        _log("Using proxy for pip: {}".format(proxy_url.split("@")[-1]),
             Qgis.MessageLevel.Info)
        return []
    _log(f"Using proxy for pip: {proxy_url}", Qgis.MessageLevel.Info)
    return ["--proxy", proxy_url]
