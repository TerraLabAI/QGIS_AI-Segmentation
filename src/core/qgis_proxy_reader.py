














from __future__ import annotations

from urllib.parse import quote, urlparse








_HTTP_PROXY_KINDS = ("", "HttpProxy", "HttpCachingProxy")


def _read_proxy_setup() -> tuple[bool, str, str, str]:


    from qgis.core import QgsSettings

    from .proxy_credentials import qgis_proxy_setting

    settings = QgsSettings()
    return (
        bool(qgis_proxy_setting(settings, "proxy/proxyEnabled", False, bool)),
        qgis_proxy_setting(settings, "proxy/proxyType", "", str) or "",
        qgis_proxy_setting(settings, "proxy/proxyHost", "", str),
        qgis_proxy_setting(settings, "proxy/proxyPort", "", str),
    )


def _read_excluded_hosts() -> tuple[str, ...]:



    from qgis.core import QgsSettings

    from .proxy_credentials import qgis_proxy_setting

    raw = qgis_proxy_setting(QgsSettings(), "proxy/noProxyUrls", [])
    if isinstance(raw, str):
        raw = [raw]
    hosts: list[str] = []
    for entry in raw or []:
        text = str(entry).strip()
        if not text:
            continue
        host = urlparse(text).hostname if "://" in text else text.split("/")[0]
        host = (host or "").strip()
        if host and host not in hosts:
            hosts.append(host)
    return tuple(hosts)


def _credentials_prefix(*, blank_password: bool) -> str:






    from .proxy_credentials import qgis_proxy_credentials

    user, password = qgis_proxy_credentials()
    if not user:
        return ""
    if blank_password:
        return f"{quote(user, safe='')}:{quote(password, safe='')}@"
    prefix = quote(user, safe="")
    if password:
        prefix += ":" + quote(password, safe="")
    return prefix + "@"


def _log_proxy_warning(message: str) -> None:
    from qgis.core import Qgis

    from .logging_utils import log

    log(message, Qgis.MessageLevel.Warning)


def qgis_proxy_url() -> str | None:








    try:
        enabled, kind, host, port = _read_proxy_setup()
        if not enabled:
            return None
        if kind == "Socks5Proxy":
            _log_proxy_warning(
                "QGIS is configured with a SOCKS5 proxy, which is not "
                "supported for dependency installs. Trying a direct "
                "connection instead.")
            return None
        if kind not in _HTTP_PROXY_KINDS or not host:
            return None




        host = _bracketed_host(_host_without_scheme(host))
        if not host:
            return None
        url = "http://" + _credentials_prefix(blank_password=False) + host


        if port and not _host_carries_port(host):
            url += f":{port}"
        return url
    except Exception as err:  # noqa: BLE001
        _log_proxy_warning(f"Could not read the QGIS proxy settings: {type(err).__name__}")
        return None


def qgis_proxy_bypass() -> str:





    try:
        return ",".join(_read_excluded_hosts())
    except Exception as err:  # noqa: BLE001
        _log_proxy_warning(f"Could not read the QGIS proxy exclusions: {type(err).__name__}")
        return ""


def qgis_urllib_proxies() -> dict[str, str]:








    try:
        enabled, kind, host, port = _read_proxy_setup()
        if not enabled:
            return {}
        proxies: dict[str, str] = {}
        try:
            excluded = _read_excluded_hosts()
        except Exception:  # noqa: BLE001
            excluded = ()
        if excluded:
            proxies["no"] = ",".join(excluded)
        if kind not in _HTTP_PROXY_KINDS:
            return proxies
        host = _bracketed_host(_host_without_scheme(host))
        carries_port = _host_carries_port(host)
        if not host or not (port or carries_port):
            return proxies
        target = f"http://{_credentials_prefix(blank_password=True)}{host}"
        if not carries_port:
            target += f":{port}"
        proxies["http"] = target
        proxies["https"] = target
        return proxies
    except Exception:  # noqa: BLE001
        return {}


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
