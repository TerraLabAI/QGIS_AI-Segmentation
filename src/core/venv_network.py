









from __future__ import annotations

import os
import sys

from qgis.core import Qgis

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


def _host_without_scheme(host: str) -> str:

    host = (host or "").strip()
    if "://" in host:
        host = host.split("://", 1)[1]
    return host.strip("/").split("/", 1)[0]


def _bracketed_host(host: str) -> str:





    if host.startswith("[") or host.count(":") < 2:
        return host
    return f"[{host}]"


def _host_carries_port(host: str) -> bool:




    tail = host.rsplit("]", 1)[-1] if host.startswith("[") else host
    return ":" in tail


def _get_qgis_proxy_settings() -> str | None:





    try:
        from urllib.parse import quote as url_quote

        from qgis.core import QgsSettings

        settings = QgsSettings()
        enabled = settings.value("proxy/proxyEnabled", False, type=bool)
        if not enabled:
            return None



        proxy_type = settings.value("proxy/proxyType", "", type=str)
        if proxy_type == "Socks5Proxy":
            _log(
                "QGIS is configured with a SOCKS5 proxy, which is not "
                "supported for dependency installs. Trying a direct "
                "connection instead.",
                Qgis.MessageLevel.Warning
            )
            return None

        host = settings.value("proxy/proxyHost", "", type=str)
        if not host:
            return None

        port = settings.value("proxy/proxyPort", "", type=str)




        from .proxy_credentials import qgis_proxy_credentials

        user, password = qgis_proxy_credentials()





        host = _bracketed_host(_host_without_scheme(host))
        if not host:
            return None

        proxy_url = "http://"
        if user:
            proxy_url += url_quote(user, safe="")
            if password:
                proxy_url += ":" + url_quote(password, safe="")
            proxy_url += "@"
        proxy_url += host


        if port and not _host_carries_port(host):
            proxy_url += f":{port}"

        return proxy_url
    except Exception as e:
        _log(f"Could not read QGIS proxy settings: {e}", Qgis.MessageLevel.Warning)
        return None



_auto_config_proxy: dict[str, tuple[str | None]] = {}


def _get_auto_config_proxy_settings() -> str | None:


























    cached = _auto_config_proxy.get("answer")
    if cached is not None:
        return cached[0]
    if _on_gui_thread():
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


def _on_gui_thread() -> bool:






    try:
        from qgis.PyQt.QtCore import QCoreApplication, QThread

        app = QCoreApplication.instance()
        if app is None:
            return False
        return QThread.currentThread() is app.thread()
    except Exception:  # noqa: BLE001
        return False


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
            return proxy_url
    except Exception as e:
        _log(f"Could not read system proxy settings: {e}", Qgis.MessageLevel.Warning)
    return None


def _get_qgis_no_proxy_hosts() -> str:





    try:
        from urllib.parse import urlparse

        from qgis.core import QgsSettings

        settings = QgsSettings()
        raw = settings.value("proxy/noProxyUrls", [])
        if isinstance(raw, str):
            raw = [raw]
        hosts = []
        for entry in raw or []:
            text = str(entry).strip()
            if not text:
                continue
            host = urlparse(text).hostname if "://" in text else text.split("/")[0]
            host = (host or "").strip()
            if host and host not in hosts:
                hosts.append(host)
        return ",".join(hosts)
    except Exception as e:  # noqa: BLE001
        _log(f"Could not read the QGIS proxy exclusions: {e}", Qgis.MessageLevel.Warning)
        return ""


def _get_effective_proxy_url() -> str | None:










    return _get_qgis_proxy_settings() or _get_system_proxy_settings() or _get_auto_config_proxy_settings()


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
