







from __future__ import annotations

from qgis.core import QgsProject

from ...core.i18n import tr


class DockLifecycleMixin:





    def closeEvent(self, event):




        if self._auto_run_active:
            try:
                from qgis.utils import iface as _iface
                _iface.messageBar().pushInfo(
                    "AI Segmentation",
                    tr("Detection continues in the background. "
                       "Reopen AI Segmentation to follow it."))
            except Exception:
                pass  # nosec B110
        elif bool(getattr(self, "_qgis_bridge_active_ui", False)):



            try:
                self.auto_qgis_bridge_done_requested.emit()
            except (TypeError, RuntimeError):
                pass
        elif self._refine_handoff:




            try:
                self.auto_reshape_done_requested.emit()
            except (TypeError, RuntimeError):
                pass



        if self._segmentation_active:
            self.stop_segmentation_requested.emit()
            if self._segmentation_active:
                event.ignore()
                return

        try:
            from ...core.telemetry import flush as _telemetry_flush
            _telemetry_flush()
        except Exception:
            pass  # nosec B110
        super().closeEvent(event)

    def cleanup_signals(self):




        try:
            self._close_manual_install_window()
        except (TypeError, RuntimeError, AttributeError):

            pass
        try:
            self.layer_combo.cleanup()
        except (TypeError, RuntimeError, AttributeError):

            pass

        try:
            self._abandon_prompt_lookup()
        except (TypeError, RuntimeError, AttributeError):
            pass


        try:
            self.auto_layer_combo.cleanup()
        except (TypeError, RuntimeError, AttributeError):

            pass
        try:
            QgsProject.instance().layersAdded.disconnect(self._on_layers_added)
        except (TypeError, RuntimeError):
            pass
        try:
            QgsProject.instance().layersRemoved.disconnect(self._on_layers_removed)
        except (TypeError, RuntimeError):
            pass
        try:
            QgsProject.instance().layerTreeRoot().visibilityChanged.disconnect(
                self._on_layer_visibility_changed)
        except (TypeError, RuntimeError, AttributeError):

            pass



        window = getattr(self, "_shortcut_arming_window", None)
        filt = getattr(self, "_shortcut_arming_filter", None)
        if window is not None and filt is not None:
            try:
                window.removeEventFilter(filt)
            except (RuntimeError, AttributeError):
                pass
        self._shortcut_arming_window = None


        self._shortcut_arming_filter = None




        for shortcut in getattr(self, "_dock_shortcuts", ()):
            if shortcut is None:
                continue
            try:
                shortcut.activated.disconnect()
                shortcut.deleteLater()
            except (TypeError, RuntimeError, AttributeError):

                pass
        self._dock_shortcuts = []


        for name in (
                "_progress_timer",
                "_refine_debounce_timer",
                "_auto_review_debounce_timer",
                "_auto_conf_debounce_timer",
                "_auto_conf_preview_timer",
                "_auto_progress_ease_timer",
                "_auto_prompt_focus_timer",
                "_auto_detail_emit_timer",
                "_visibility_debounce_timer",
                "_auto_warmup_timer",
                "_pairing_anim_timer"):
            timer = getattr(self, name, None)
            if timer is None:
                continue
            try:
                timer.blockSignals(True)
                timer.stop()
                timer.timeout.disconnect()
            except (TypeError, RuntimeError, AttributeError):

                pass

        if getattr(self, "_checkbox_icon_dir", None):
            import shutil
            shutil.rmtree(self._checkbox_icon_dir, ignore_errors=True)
            self._checkbox_icon_dir = None

    def is_activated(self) -> bool:

        return self._plugin_activated

    def changeEvent(self, event):  # noqa: N802




        super().changeEvent(event)
        try:
            from qgis.PyQt.QtCore import QEvent

            kinds = tuple(
                getattr(QEvent.Type, name) for name in
                ("PaletteChange", "ApplicationPaletteChange", "ThemeChange")
                if hasattr(QEvent.Type, name))
            if event.type() not in kinds or getattr(self, "_theme_follow_queued", False):
                return
            from ...core.qt_compat import safe_single_shot

            self._theme_follow_queued = True
            safe_single_shot(0, self, self._follow_qgis_theme_now)
        except Exception:  # noqa: BLE001
            self._theme_follow_queued = False

    def _follow_qgis_theme_now(self) -> None:
        self._theme_follow_queued = False
        from .styles import follow_qgis_theme

        follow_qgis_theme(self)

    def showEvent(self, event):







        super().showEvent(event)





        for combo_name in ("layer_combo", "auto_layer_combo"):
            try:
                getattr(self, combo_name)._schedule_refresh()
            except (RuntimeError, AttributeError):

                pass
        if getattr(self, "_plugin_opened_emitted", False):
            return
        try:
            from ...core.telemetry import is_telemetry_enabled
            from ...core.telemetry_session_events import track_plugin_first_open, track_plugin_opened



            track_plugin_first_open()
            track_plugin_opened()



            self._plugin_opened_emitted = bool(is_telemetry_enabled())
        except Exception:
            pass  # nosec B110
