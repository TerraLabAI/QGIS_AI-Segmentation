









from __future__ import annotations

try:
    from qgis.PyQt.QtCore import QCoreApplication, QThread
except ImportError:
    QCoreApplication = QThread = None


def on_gui_thread(unknown: bool = False) -> bool:








    try:
        app = QCoreApplication.instance()
        if app is None:
            return unknown
        return QThread.currentThread() is app.thread()
    except Exception:  # noqa: BLE001
        return unknown
