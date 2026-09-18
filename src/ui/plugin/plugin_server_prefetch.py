







from __future__ import annotations

import time

from ...core.i18n import tr


class ServerPrefetchMixin:





    def _prefetch_server_config(self) -> None:











        if self._config_prefetch_task is not None and self._config_prefetch_task.is_active():
            return
        from qgis.core import QgsApplication

        from ...api.terralab_client import TerraLabClient
        from ...core.activation_manager import PRODUCT_ID
        from ...workers.generic_request_task import GenericRequestTask
        client = TerraLabClient()
        self._config_prefetch_task = GenericRequestTask(
            tr("Loading AI Segmentation settings"),
            lambda: self._refresh_server_config(client, PRODUCT_ID),
            hidden=True,
        )
        self._config_last_fetch_unix = time.time()
        self._config_prefetch_task.succeeded.connect(self._on_config_prefetched)
        self._config_prefetch_task.failed.connect(self._on_config_prefetch_failed)
        QgsApplication.taskManager().addTask(self._config_prefetch_task)

    @staticmethod
    def _refresh_server_config(client, product_id: str) -> dict:








        from ...core.config_cache import prime_from_disk
        try:
            prime_from_disk()
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        config = client.get_config(product_id)
        if isinstance(config, dict) and "error" not in config:
            from ...core.activation_manager import set_cached_config
            set_cached_config(config)
        return config

    def _on_config_prefetched(self, _config: object) -> None:

        self._config_prefetch_task = None
        self._reapply_server_switches()
        self._arm_config_refresh()

    def _on_config_prefetch_failed(self, message: str, code: str) -> None:
        self._config_prefetch_task = None


        self._reapply_server_switches()
        self._notify_connection_issue(code, message)



        self._arm_config_refresh()

    def _arm_config_refresh(self) -> None:

















        if self._config_refresh_timer is not None or self.dock_widget is None:
            return


        try:
            if not self.dock_widget.isVisible():
                return
        except (RuntimeError, AttributeError):
            return
        from qgis.PyQt.QtCore import QTimer



        timer = QTimer(self.dock_widget)
        timer.setInterval(self._config_refresh_interval_ms())
        timer.timeout.connect(self._refresh_config_if_idle)
        timer.start()
        self._config_refresh_timer = timer

    def _set_config_refresh_paused(self, paused: bool) -> None:





        timer = self._config_refresh_timer
        if timer is None:
            return
        try:
            if paused:
                timer.stop()
            elif not timer.isActive():
                timer.start()
        except RuntimeError:
            self._config_refresh_timer = None

    def _resume_config_refresh(self) -> None:









        self._arm_config_refresh()
        self._set_config_refresh_paused(False)
        try:
            interval_s = self._config_refresh_interval_ms() / 1000.0
        except Exception:  # noqa: BLE001
            return
        if (time.time() - self._config_last_fetch_unix) >= interval_s:
            self._prefetch_server_config()

    @staticmethod
    def _config_refresh_interval_ms() -> int:






        from ...core.server_dials import dial_in_range

        minutes = dial_in_range("config_refresh_minutes", 30, 5, 720)
        return int(minutes) * 60 * 1000

    def _refresh_config_if_idle(self) -> None:







        if getattr(self, "_auto_worker", None) is not None:
            return
        if getattr(self, "_auto_review", None):
            return
        try:
            if self.dock_widget is None or not self.dock_widget.isVisible():
                return
        except RuntimeError:
            return
        self._prefetch_server_config()


        timer = self._config_refresh_timer
        if timer is not None:
            try:
                wanted = self._config_refresh_interval_ms()
                if timer.interval() != wanted:
                    timer.setInterval(wanted)
            except RuntimeError:
                self._config_refresh_timer = None

    def _reapply_server_switches(self) -> None:

        if self.dock_widget is None:
            return
        try:
            self.dock_widget.apply_server_feature_switches()

            self.dock_widget.check_for_updates()
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _prefetch_segment_catalog(self) -> None:





        if self._catalog_prefetch_task is not None and self._catalog_prefetch_task.is_active():
            return
        from qgis.core import QgsApplication

        from ...core.presets.segmentation_presets_client import fetch_catalog
        from ...workers.generic_request_task import GenericRequestTask
        self._catalog_prefetch_task = GenericRequestTask(
            tr("Loading segment library"),
            lambda: fetch_catalog(force=False),
            hidden=True,
        )
        self._catalog_prefetch_task.succeeded.connect(self._on_catalog_prefetched)
        self._catalog_prefetch_task.failed.connect(
            lambda *_a: setattr(self, "_catalog_prefetch_task", None))
        QgsApplication.taskManager().addTask(self._catalog_prefetch_task)

    def _on_catalog_prefetched(self, _result: object) -> None:


        self._catalog_prefetch_task = None
