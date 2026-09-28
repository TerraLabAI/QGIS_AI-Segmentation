














from __future__ import annotations

from qgis.PyQt.QtCore import QObject, QTimer, pyqtSignal

from ..core.i18n import tr
from ..core.server_dials import dial_in_range

UPDATE_STATE_FETCHING = "fetching"
UPDATE_STATE_INSTALLING = "installing"
UPDATE_STATE_FALLBACK = "fallback"

UPDATE_RESULT_INSTALLED = "installed"
UPDATE_RESULT_RESTART = "restart_needed"
UPDATE_RESULT_FELL_BACK = "fell_back"
UPDATE_RESULT_FAILED = "failed"

_FETCH_TIMEOUT_S = 30

_RESULT_NOTE_SECONDS = 10


_BUSY_RECHECK_MS = 1000


class PluginSelfUpdate(QObject):


    state_changed = pyqtSignal(str)

    def __init__(self, installer_key: str, target_version: str, fallback, parent=None,
                 on_result=None, is_busy=None):
        super().__init__(parent)
        self._key = installer_key
        self._target = target_version

        self._fallback = fallback


        self._on_result = on_result

        self._is_busy = is_busy
        self._done = False
        self._timer: QTimer | None = None
        self._repositories = None


        self._installed_note = tr("Version {version} is installed.")
        self._restart_note = tr("Version {version} is installed. Restart QGIS to use it.")

    def start(self) -> None:

        try:
            import pyplugin_installer  # noqa: F401
            from pyplugin_installer.installer_data import plugins, repositories
        except Exception:  # noqa: BLE001
            self._give_up()
            return
        self._repositories = repositories
        try:
            if self._listed_upgradeable(plugins):
                self._schedule_install()
                return
            enabled = list(repositories.allEnabled())
            if not enabled:
                self._give_up()
                return
            self.state_changed.emit(UPDATE_STATE_FETCHING)
            repositories.checkingDone.connect(self._on_fetched)
            for key in enabled:
                request_repository_fetch(repositories, key)
        except Exception:  # noqa: BLE001
            self._give_up()
            return
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._give_up)
        self._timer.start(int(1000 * dial_in_range(
            "tuning.update.fetch_timeout_s", _FETCH_TIMEOUT_S, 5, 180)))

    def _listed_upgradeable(self, plugins) -> bool:
        data = plugins.all().get(self._key)
        return bool(data) and data.get("status") == "upgradeable"

    def _disconnect_fetch(self) -> None:
        if self._timer is not None:
            self._timer.stop()
        try:
            self._repositories.checkingDone.disconnect(self._on_fetched)
        except (TypeError, RuntimeError, AttributeError):
            pass  # nosec B110

    def _on_fetched(self) -> None:
        if self._done:
            return
        self._disconnect_fetch()
        try:
            from pyplugin_installer.installer_data import plugins

            plugins.rebuild()
            listed = self._listed_upgradeable(plugins)
        except Exception:  # noqa: BLE001
            listed = False
        if listed:
            self._schedule_install()
        else:
            self._give_up()

    def _schedule_install(self) -> None:

        self.state_changed.emit(UPDATE_STATE_INSTALLING)
        QTimer.singleShot(0, self._install)

    def _work_in_progress(self) -> bool:
        if self._is_busy is None:
            return False
        try:
            return bool(self._is_busy())
        except Exception:  # noqa: BLE001
            return False

    def _install(self) -> None:
        if self._done:
            return
        if self._work_in_progress():


            QTimer.singleShot(_BUSY_RECHECK_MS, self._install)
            return
        self._done = True
        version = self._target
        try:
            import pyplugin_installer
            from pyplugin_installer.installer_data import plugins

            listed = plugins.all().get(self._key) or {}
            version = str(listed.get("version_available") or "") or version

            _flush_telemetry()
            installer = pyplugin_installer.instance()
            try:
                installer.installPlugin(self._key, quiet=True)
            except TypeError:
                installer.installPlugin(self._key)
            data = plugins.all().get(self._key) or {}
            installed = data.get("status") == "installed" and not data.get("error")
            version = str(data.get("version_installed") or "") or version
        except Exception:  # noqa: BLE001
            installed = False
        if not installed:
            self._run_fallback(UPDATE_RESULT_FAILED)
            return
        self._report(UPDATE_RESULT_INSTALLED if self._push_result_note(version)
                     else UPDATE_RESULT_RESTART, version)
        self.deleteLater()

    def _report(self, result: str, version: str = "") -> None:
        if self._on_result is None:
            return
        try:
            self._on_result(result, version)
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _push_result_note(self, version: str) -> bool:




        loaded = False
        try:
            from qgis.core import Qgis
            from qgis.utils import iface, isPluginLoaded

            loaded = isPluginLoaded(self._key)
            iface.messageBar().pushMessage(
                "AI Segmentation",
                (self._installed_note if loaded else self._restart_note).format(
                    version=version),
                level=Qgis.MessageLevel.Success if loaded else Qgis.MessageLevel.Warning,
                duration=_RESULT_NOTE_SECONDS)
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return bool(loaded)

    def _give_up(self) -> None:
        if self._done:
            return
        self._done = True
        self._disconnect_fetch()
        self._run_fallback()

    def _run_fallback(self, result: str = UPDATE_RESULT_FELL_BACK) -> None:
        self._report(result)
        self.state_changed.emit(UPDATE_STATE_FALLBACK)
        try:
            self._fallback()
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        self.deleteLater()


def _flush_telemetry() -> None:

    try:
        from ..core.telemetry import flush

        flush()
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def request_repository_fetch(repositories, key: str) -> None:




    try:
        repositories.requestFetching(key, force_reload=True)
    except TypeError:
        repositories.requestFetching(key)



_ACTIVE_UPDATE: list = []


def start_plugin_self_update(installer_key: str, target_version: str,
                             fallback, on_state=None, on_result=None,
                             is_busy=None) -> PluginSelfUpdate:

    parent = None
    try:
        from qgis.utils import iface

        parent = iface.mainWindow()
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    runner = PluginSelfUpdate(installer_key, target_version, fallback, parent,
                              on_result=on_result, is_busy=is_busy)
    _ACTIVE_UPDATE[:] = [runner]
    if on_state is not None:
        runner.state_changed.connect(on_state)
    runner.start()
    return runner
