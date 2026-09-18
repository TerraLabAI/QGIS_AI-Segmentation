







from __future__ import annotations

from ..core import transport_dials as _td
from .terralab_client_primitives import (
    _log_warning,
)
from .terralab_client_retry import (
    _WINDOW_HINT_MAX,
)









_CONNECTIONS_PER_MANAGER = 6















_THREAD_NAMS: dict = {}
_THREAD_NAM_CURSOR: dict = {}
_THREAD_NAM_WATCHED: set = set()


def _predict_manager_count() -> int:






    try:
        ceiling = _td.window_hint_ceiling(_WINDOW_HINT_MAX)
    except Exception:  # noqa: BLE001
        ceiling = _WINDOW_HINT_MAX
    return max(1, -(-int(ceiling) // _CONNECTIONS_PER_MANAGER))


def _drop_thread_nam() -> None:






    try:
        from qgis.PyQt.QtCore import QThread

        thread = QThread.currentThread()
        _THREAD_NAMS.pop(thread, None)
        _THREAD_NAM_CURSOR.pop(thread, None)
        _THREAD_NAM_WATCHED.discard(thread)
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def _qobject_alive(obj) -> bool:






    if obj is None:
        return False
    try:
        obj.thread()
    except RuntimeError:
        return False
    return True


class TerraLabManagerPoolMixin:







    def _predict_nam(self):




























        return self._predict_nam_at(0)

    def _predict_nam_at(self, slot: int):






        pool = self._predict_nam_pool()
        slot %= len(pool)
        nam = pool[slot]
        if nam is not None and not _qobject_alive(nam):




            nam = None
        if nam is None:
            nam = self._new_private_nam()
            pool[slot] = nam
        return nam

    def _predict_nam_pool(self) -> list:






        from qgis.PyQt.QtCore import QThread

        thread = QThread.currentThread()
        pool = _THREAD_NAMS.get(thread)
        if not pool:
            pool = [None] * _predict_manager_count()
            _THREAD_NAMS[thread] = pool
            _THREAD_NAM_CURSOR[thread] = 0
        return pool

    def _next_predict_nam(self):





        from qgis.PyQt.QtCore import QThread

        thread = QThread.currentThread()
        slot = _THREAD_NAM_CURSOR.get(thread, 0)
        _THREAD_NAM_CURSOR[thread] = slot + 1
        return self._predict_nam_at(slot)

    def acquire_predict_nam(self):












        return self._predict_nam_pool()

    def _new_private_nam(self):



        from qgis.PyQt.QtNetwork import QNetworkAccessManager

        nam = QNetworkAccessManager()
        try:
            nam.setProxy(self._qgis_effective_proxy())
        except (RuntimeError, AttributeError) as err:



            _log_warning(f"Proxy mirror failed, continuing direct: {type(err).__name__}")





        try:
            nam.sslErrors.connect(self._on_predict_ssl_errors)
        except (RuntimeError, AttributeError):
            pass




        try:
            nam.proxyAuthenticationRequired.connect(self._on_proxy_auth_required)
        except (RuntimeError, AttributeError):
            pass







        try:
            nam._terralab_owner = self
        except (AttributeError, RuntimeError):
            pass
        return nam

    def _on_proxy_auth_required(self, proxy, authenticator) -> None:









        try:
            if authenticator is None or authenticator.user():
                return
            from ..core.proxy_credentials import qgis_proxy_credentials

            user, password = qgis_proxy_credentials()
            if not user:


                if not getattr(self, "_proxy_credentials_warned", False):
                    self._proxy_credentials_warned = True
                    _log_warning(
                        "The proxy asked for a user name and QGIS has none stored. "
                        "Set it in Settings > Options > Network."
                    )
                return
            authenticator.setUser(user)
            authenticator.setPassword(password)
        except Exception as err:  # noqa: BLE001
            _log_warning(f"Proxy credential lookup failed: {type(err).__name__}")

    def release_thread_nam(self) -> None:











        if not _THREAD_NAMS:
            return
        from qgis.PyQt.QtCore import QThread

        thread = QThread.currentThread()
        _THREAD_NAMS.pop(thread, None)
        _THREAD_NAM_CURSOR.pop(thread, None)
        _THREAD_NAM_WATCHED.discard(thread)

    def retain_thread_nam(self) -> bool:












        from qgis.PyQt.QtCore import QThread

        thread = QThread.currentThread()
        if thread not in _THREAD_NAMS or thread in _THREAD_NAM_WATCHED:
            return True
        try:


            thread.finished.connect(_drop_thread_nam)
        except (RuntimeError, AttributeError, TypeError):
            return False
        _THREAD_NAM_WATCHED.add(thread)
        return True

    def _qgis_effective_proxy(self):













        from qgis.core import QgsNetworkAccessManager
        from qgis.PyQt.QtNetwork import QNetworkProxy








        gnam = QgsNetworkAccessManager.instance()
        fallback = getattr(gnam, "fallbackProxy", None)
        if callable(fallback):
            try:
                proxy = fallback()
                if proxy is not None and proxy.type() not in (
                    QNetworkProxy.ProxyType.NoProxy,
                    QNetworkProxy.ProxyType.DefaultProxy,
                ):
                    return proxy
            except Exception as err:  # noqa: BLE001
                _log_warning(
                    f"Reading the QGIS fallback proxy failed: {type(err).__name__}")
        return gnam.proxy()

    def _on_predict_ssl_errors(self, reply, errors) -> None:











        try:
            from qgis.core import QgsApplication

            url = reply.url()
            hostport = f"{url.host()}:{url.port(443)}"
            auth_mgr = QgsApplication.authManager()
            if auth_mgr is None:
                return
            config = auth_mgr.sslCertCustomConfigByHost(hostport)
            if config is None or config.isNull():
                return
            allowed = set(config.sslIgnoredErrorEnums())
            if not allowed:
                return


            if all(err.error() in allowed for err in errors):
                reply.ignoreSslErrors(errors)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
