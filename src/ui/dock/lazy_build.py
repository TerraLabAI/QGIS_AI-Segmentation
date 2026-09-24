




















from __future__ import annotations

import sys
import time
import traceback

from qgis.core import Qgis, QgsMessageLog, QgsProject
from qgis.PyQt.QtCore import QCoreApplication, QEvent, QMetaObject, QObject, Qt, QThread, pyqtSlot
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core.qt_compat import safe_single_shot
from .styles import (
    apply_input_theme_to_tree,
    apply_keyboard_focus_policy,
    apply_quiet_scrollbar,
)
from .widgets import _WheelGuard



_WHEEL_GUARDED = (QComboBox, QSpinBox, QDoubleSpinBox, QSlider)


def _on_gui_thread() -> bool:
    app = QCoreApplication.instance()
    return app is not None and QThread.currentThread() is app.thread()


class _DockQueuedCall(QObject):





    def __init__(self, parent: QObject, callback) -> None:
        super().__init__(parent)
        self._callback = callback
        QMetaObject.invokeMethod(self, "run_queued_call", Qt.ConnectionType.QueuedConnection)

    @pyqtSlot()
    def run_queued_call(self) -> None:
        callback, self._callback = self._callback, None
        self.deleteLater()
        if callback is not None:
            callback()


class _DockFirstPaintHook(QObject):


    def __init__(self, watched: QWidget, callback) -> None:
        super().__init__(watched)
        self._callback = callback
        watched.installEventFilter(self)

    def eventFilter(self, watched, event):  # noqa: N802

        try:
            if event.type() == QEvent.Type.Paint and self._callback is not None:
                callback, self._callback = self._callback, None
                watched.removeEventFilter(self)


                safe_single_shot(0, watched, callback)
                self.deleteLater()
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return False


class DockLazyBuildMixin:




    def __getattr__(self, name):










        if not name.startswith("__") and _on_gui_thread():
            state = self.__dict__.get("_dock_content_state")
            if state == "pending":
                self.ensure_dock_content(trigger=name)
                return getattr(self, name)
            if (state == "built" and self.__dict__.get("_dock_pending_parts")
                    and not self.__dict__.get("_dock_parts_building")):
                self.ensure_dock_parts(trigger=name)
                return getattr(self, name)
        return super().__getattr__(name)

    def setVisible(self, visible: bool) -> None:  # noqa: N802

        if visible and self.__dict__.get("_dock_content_state") == "pending":
            try:
                self._queue_dock_content_check()
            except Exception:  # noqa: BLE001
                self._log_dock_content_failure()
        super().setVisible(visible)

    def _queue_dock_content_check(self) -> None:











        if self.__dict__.get("_dock_content_check_queued"):
            return
        self._dock_content_check_queued = True
        _DockQueuedCall(self, self._build_dock_content_if_shown)

    def _build_dock_content_if_shown(self) -> None:
        self._dock_content_check_queued = False
        if self.__dict__.get("_dock_content_state") != "pending" or not self.isVisible():
            return
        try:
            self.ensure_dock_content(trigger="show")
        except Exception:  # noqa: BLE001
            self._log_dock_content_failure()

    @property
    def dock_content_built(self) -> bool:

        return self.__dict__.get("_dock_content_state") == "built"

    def when_dock_content_built(self, callback) -> None:

        if self.dock_content_built:
            callback()
        else:
            self._dock_content_callbacks.append(callback)

    def stop_dock_content_build(self) -> None:



        if self.__dict__.get("_dock_content_state") == "pending":
            self._dock_content_state = "closed"
        self._dock_content_callbacks = []
        self._dock_pending_parts = []

    def _log_dock_content_failure(self) -> None:
        QgsMessageLog.logMessage(
            "AI Segmentation panel could not be built:\n" + traceback.format_exc(),
            "AI Segmentation", level=Qgis.MessageLevel.Critical)



    def ensure_dock_content(self, trigger: str = "") -> None:

        if self.__dict__.get("_dock_content_state") != "pending":
            return
        self._dock_content_state = "building"
        started = time.perf_counter()




        was_blocked = self.blockSignals(True)
        try:
            self._build_dock_content()
        except Exception:
            self._dock_content_state = "failed"
            raise
        finally:
            self.blockSignals(was_blocked)
        self._dock_content_state = "built"
        callbacks, self._dock_content_callbacks = self._dock_content_callbacks, []
        for callback in callbacks:
            callback()
        if self._dock_pending_parts:
            _DockFirstPaintHook(self.main_widget, self._build_parts_after_paint)
        self._dock_content_trigger = trigger or "direct"
        QgsMessageLog.logMessage(
            "Panel built on first use ({}, {:.0f} ms)".format(
                trigger or "direct", (time.perf_counter() - started) * 1000),
            "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _build_dock_content(self) -> None:


        self._setup_title_bar()

        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setSpacing(8)
        self.main_layout.setContentsMargins(8, 8, 8, 8)

        self._setup_ui()



        from .font_scale import apply_font_scale_to_tree

        apply_font_scale_to_tree(self.main_widget)




        apply_input_theme_to_tree(self.main_widget)


        apply_keyboard_focus_policy(self.main_widget)
        apply_keyboard_focus_policy(self._custom_title_bar)

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.main_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)



        apply_quiet_scrollbar(scroll_area)


        body_holder = QWidget()
        body_col = QVBoxLayout(body_holder)
        body_col.setContentsMargins(0, 0, 0, 0)
        body_col.setSpacing(0)
        body_col.addWidget(scroll_area, 1)
        body_col.addWidget(self.update_gate_page, 1)
        apply_keyboard_focus_policy(self.update_gate_page)
        self.setWidget(body_holder)


        self._dock_scroll_area = scroll_area




        self._wheel_guard = _WheelGuard(scroll_area.viewport(), self)



        for _w in self.main_widget.findChildren(_WHEEL_GUARDED):
            _w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            _w.installEventFilter(self._wheel_guard)




        QgsProject.instance().layersAdded.connect(self._on_layers_added)
        QgsProject.instance().layersRemoved.connect(self._on_layers_removed)

        QgsProject.instance().layerTreeRoot().visibilityChanged.connect(
            self._on_layer_visibility_changed)



        self.visibilityChanged.connect(self._on_dock_hidden_reset_engine)


        if sys.platform == "win32" and self.isFloating():
            self._swap_title_bar_for_floating(True)


        self._update_full_ui()


        self._sync_pro_pill()



    def _defer_dock_part(self, builder) -> None:







        self._dock_pending_parts.append(builder)

    def _build_parts_after_paint(self) -> None:


        try:
            self.ensure_dock_parts(trigger="after first paint", limit=1)
        except Exception:  # noqa: BLE001
            self._log_dock_content_failure()
            return
        if self._dock_pending_parts and self.__dict__.get("main_widget") is not None:
            safe_single_shot(0, self.main_widget, self._build_parts_after_paint)

    def ensure_dock_parts(self, trigger: str = "", limit: int = 0) -> None:


        if (self.__dict__.get("_dock_content_state") != "built"
                or self._dock_parts_building or not self._dock_pending_parts):
            return
        self._dock_parts_building = True
        started = time.perf_counter()
        built = 0


        was_blocked = self.blockSignals(True)
        try:
            while self._dock_pending_parts and not (limit and built >= limit):
                self._dock_pending_parts.pop(0)()
                built += 1
        finally:
            self.blockSignals(was_blocked)
            self._dock_parts_building = False
        QgsMessageLog.logMessage(
            "Panel: {} deferred part(s) built ({}, {:.0f} ms)".format(
                built, trigger or "direct", (time.perf_counter() - started) * 1000),
            "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _finish_dock_part(self, root: QWidget) -> None:



        from .font_scale import apply_font_scale_to_tree

        apply_font_scale_to_tree(root)
        apply_input_theme_to_tree(root)
        apply_keyboard_focus_policy(root)
        guard = self.__dict__.get("_wheel_guard")
        if guard is None:
            return
        for _w in root.findChildren(_WHEEL_GUARDED):
            _w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            _w.installEventFilter(guard)
