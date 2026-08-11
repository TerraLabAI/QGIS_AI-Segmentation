"""Network and TLS policy for dependency installs.

Owns the proxy discovery chain used by the pip/uv install subprocesses
(QGIS-configured proxy first, then the OS-level system proxy) and the
explicit opt-in that governs whether TLS verification may be disabled
during a corporate-CA install. These helpers hold no venv state: they
read QGIS/OS settings and translate them into pip/uv arguments, so they
live apart from the venv build logic in venv_manager.
"""

from __future__ import annotations

import os
import sys

from qgis.core import Qgis

from .logging_utils import log as _log


def _insecure_install_opt_in() -> bool:
    """True only if the user has explicitly allowed disabling TLS verification
    during dependency install.

    The safe corporate-CA path (--native-tls, OS trust store) is always
    tried first. Fully disabling TLS verification is an integrity risk (an
    active MITM could force the initial verified attempt to fail and then
    serve a poisoned wheel, which runs code on install), so it never happens
    silently: it requires an explicit opt-in via the QSettings key
    ``TerraLab/allow_insecure_install`` or the ``QGIS_AI_ALLOW_INSECURE_INSTALL``
    environment variable. Default is off.
    """
    if os.environ.get("QGIS_AI_ALLOW_INSECURE_INSTALL", "").strip().lower() in ("1", "true", "yes"):
        return True
    try:
        from qgis.PyQt.QtCore import QSettings

        val = QSettings().value("TerraLab/allow_insecure_install", False, type=bool)
        return bool(val)
    except Exception:  # noqa: BLE001 - settings unavailable: fail closed (secure default)
        return False


# Variables that REPLACE the trust store instead of adding to it: while one of
# them is set, that file is the only thing uv trusts and every other source,
# including the OS store, is ignored.
CA_BUNDLE_ENV_VARS = ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
                      "CURL_CA_BUNDLE")


def env_without_ca_bundle_overrides(env: dict) -> tuple[dict, list[str]]:
    """A copy of ``env`` with the inherited CA-bundle variables removed, plus
    the names that were dropped (for the log line).

    Whatever bundle the machine handed us was already used by the attempt that
    just failed, and it hides every other trust source. The retry starts from a
    clean slate so the certificates it does install are the ones that count.
    """
    out = dict(env)
    dropped = [name for name in CA_BUNDLE_ENV_VARS if out.pop(name, None)]
    return out, dropped


def windows_trust_store_bundle(cache_dir: str) -> tuple[str | None, int]:
    """Write the machine's own certificate authorities to a PEM file and return
    (path, certificate count). ``(None, 0)`` off Windows, or if nothing could
    be read.

    A company proxy opens every TLS connection and re-signs it with a company
    CA, which the package index's public roots do not vouch for: the install
    fails as an unknown issuer. Asking uv for the platform store is the
    documented answer and it is what the first attempt already does, so when
    that attempt fails the CA is somewhere the platform verifier does not look:
    a per-user store, an intermediate store, a hand-imported certificate. This
    reads all of them (the Python runtime exposes the same stores the rest of
    Windows uses) and hands uv the file, which outranks every other source.

    Public roots come first, from certifi when it is importable, so a bundle
    built on a machine whose root store is still half-empty (Windows fills it
    on demand) can never trust LESS than the attempt before it.
    """
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
        except Exception:  # noqa: BLE001 - no certifi: the OS store alone  # nosec B110
            pass
        for store in ("ROOT", "CA"):
            try:
                entries = ssl.enum_certificates(store)
            except Exception:  # noqa: BLE001 - store unreadable: try the next  # nosec B112
                continue
            for der, _encoding, trust in entries:
                # trust is True (every purpose) or the set of purpose OIDs;
                # 1.3.6.1.5.5.7.3.1 is server authentication.
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
    except Exception as e:  # noqa: BLE001 - the retry falls back to the flag
        _log(f"Could not export the OS certificate store: {e}",
             Qgis.MessageLevel.Warning)
        return None, 0


def _host_without_scheme(host: str) -> str:
    """The bare host part of whatever sits in the QGIS proxy host setting."""
    host = (host or "").strip()
    if "://" in host:
        host = host.split("://", 1)[1]
    return host.strip("/").split("/", 1)[0]


def _bracketed_host(host: str) -> str:
    """The host as a URL may carry it: an IPv6 address gains its brackets.

    The Network page asks for an address, so a v6 one is typed bare. Without
    the brackets the colons of the address read as the port separator and the
    URL is rejected."""
    if host.startswith("[") or host.count(":") < 2:
        return host
    return f"[{host}]"


def _host_carries_port(host: str) -> bool:
    """Does this host string already end in ":port"?

    Only the colon after the closing bracket can be the port on an IPv6
    address; the ones inside it belong to the address."""
    tail = host.rsplit("]", 1)[-1] if host.startswith("[") else host
    return ":" in tail


def _get_qgis_proxy_settings() -> str | None:
    """Read proxy configuration from QGIS settings.

    Returns a proxy URL string (with optional authentication)
    or None if proxy is not configured or disabled.
    """
    try:
        from urllib.parse import quote as url_quote

        from qgis.core import QgsSettings

        settings = QgsSettings()
        enabled = settings.value("proxy/proxyEnabled", False, type=bool)
        if not enabled:
            return None

        # pip/uv only speak HTTP proxies; SOCKS needs extra packages we
        # don't ship, so fall back to a direct/environment connection.
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
        # Also covers the user who stored the pair in an authentication
        # configuration rather than in the clear, which the Network page
        # offers and which used to leave the install with no credentials at
        # all behind a proxy that demands them.
        from .proxy_credentials import qgis_proxy_credentials

        user, password = qgis_proxy_credentials()

        # The Network page asks for a host, but a user who typed a full URL
        # there stored the scheme with it. Prefixing that again gives
        # "http://http://proxy", which pip and uv reject with a message
        # naming neither the setting nor the doubled scheme.
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
        # A host typed as a full URL usually carries its port too, and
        # appending the port setting on top of it gives "proxy:3128:3128".
        if port and not _host_carries_port(host):
            proxy_url += f":{port}"

        return proxy_url
    except Exception as e:
        _log(f"Could not read QGIS proxy settings: {e}", Qgis.MessageLevel.Warning)
        return None


_auto_config_proxy_cached: tuple[str | None] | None = None


def _get_auto_config_proxy_settings() -> str | None:
    """Ask Qt which proxy the machine would use for the package index.

    A company that publishes its proxy through an auto-configuration script
    (the usual corporate setup, PAC or WPAD) writes nothing in the plain
    system setting, so ``urllib`` finds nothing and the install goes out
    direct on a network with no direct way out. Qt runs the script and
    answers with a real host and port. Credentials come from QGIS, since a
    resolved script never carries any.

    Never asked from the thread that draws QGIS. Running the script means a
    name lookup and a download, which on a machine set to discover its script
    automatically can sit there for seconds, and the whole window would be
    frozen for all of it. Every caller that needs this is an install running
    on its own thread; a call from the drawing thread simply reads as no
    script, which is what happened before this existed. Asked once per
    session either way: it is the same answer all afternoon.

    Nothing to expect on macOS: Qt cannot run the script there, and says so,
    so a Mac on a script-only network is no worse and no better than before.
    On Linux Qt only reads it when the build has the helper library, and
    otherwise falls back to the same environment variable the caller has
    already tried.

    None when there is no script, when it names no proxy for this address, or
    when it names a kind pip cannot speak.
    """
    global _auto_config_proxy_cached
    if _auto_config_proxy_cached is not None:
        return _auto_config_proxy_cached[0]
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
            _auto_config_proxy_cached = (resolved,)
            _log("Using the proxy the machine's automatic configuration names",
                 Qgis.MessageLevel.Info)
            return resolved
    except Exception as e:  # noqa: BLE001 - no proxy found is a normal answer
        _log(f"Could not resolve an automatic proxy configuration: {e}",
             Qgis.MessageLevel.Warning)
    _auto_config_proxy_cached = (None,)
    return None


def _on_gui_thread() -> bool:
    """Whether this call is on the thread that draws QGIS.

    False where there is no application at all, which is the tests and any
    headless run: there is no window to freeze, so the work the caller guards
    with this is safe to do.
    """
    try:
        from qgis.PyQt.QtCore import QCoreApplication, QThread

        app = QCoreApplication.instance()
        if app is None:
            return False
        return QThread.currentThread() is app.thread()
    except Exception:  # noqa: BLE001 - no Qt, no window to freeze
        return False


def _get_system_proxy_settings() -> str | None:
    """Fall back to the OS-level proxy (Windows registry, macOS network
    settings, HTTP(S)_PROXY env vars) when no proxy is set in QGIS.

    Corporate machines often have a system proxy that browsers and QGIS
    use implicitly, while our pip/uv subprocesses get a clean environment
    and fail with a network error unless we pass it explicitly.
    """
    try:
        import urllib.request

        proxies = urllib.request.getproxies()
        proxy_url = proxies.get("https") or proxies.get("http")
        if proxy_url and proxy_url.lower().startswith(("http://", "https://")):
            return proxy_url
    except Exception as e:
        _log(f"Could not read system proxy settings: {e}", Qgis.MessageLevel.Warning)
    return None


def _get_qgis_no_proxy_hosts() -> str:
    """The addresses QGIS is told to reach without the proxy, comma separated.

    Empty when the user listed none. Only the host part of each entry is kept,
    which is the form pip and uv read; QGIS stores whole URLs.
    """
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
    except Exception as e:  # noqa: BLE001 - an unreadable list is an empty one
        _log(f"Could not read the QGIS proxy exclusions: {e}", Qgis.MessageLevel.Warning)
        return ""


def _get_effective_proxy_url() -> str | None:
    """QGIS-configured proxy first, then the plain OS setting, then an
    automatic configuration script.

    The script comes LAST, not before the plain setting. On Linux the two read
    the same environment variables, and going through Qt there would throw away
    the credentials the variable already carries and put QGIS's in their place,
    signing the install in as somebody else. The script is worth asking only
    where the first two found nothing, which is the Windows machine that
    publishes its proxy that way and writes it nowhere else.
    """
    return _get_qgis_proxy_settings() or _get_system_proxy_settings() or _get_auto_config_proxy_settings()


def _get_pip_proxy_args() -> list[str]:
    """Get pip --proxy argument if a QGIS or system proxy is configured.

    A proxy URL carrying a username and password NEVER goes on the command
    line: argv is readable by any local user (``/proc/<pid>/cmdline``, ``ps``)
    for the whole multi-minute install, and pip echoes it back in proxy error
    text. Those credentials travel only through the HTTP_PROXY/HTTPS_PROXY
    variables of the install environment, which pip and uv both honour.
    """
    proxy_url = _get_effective_proxy_url()
    if not proxy_url:
        return []
    if "@" in proxy_url:
        _log("Using proxy for pip: {}".format(proxy_url.split("@")[-1]),
             Qgis.MessageLevel.Info)
        return []
    _log(f"Using proxy for pip: {proxy_url}", Qgis.MessageLevel.Info)
    return ["--proxy", proxy_url]
