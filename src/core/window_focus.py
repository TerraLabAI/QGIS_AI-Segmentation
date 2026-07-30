









from __future__ import annotations

import sys


_SW_RESTORE = 9


def bring_qgis_window_to_front(main_window, dock_widget=None) -> bool:





    if main_window is None:
        return False
    raised = False
    try:
        if main_window.isMinimized():
            main_window.showNormal()
        main_window.raise_()
        main_window.activateWindow()
        raised = True
    except Exception:  # nosec B110
        pass
    if sys.platform.startswith("win"):



        raised = _force_foreground_on_windows(main_window)
    if dock_widget is not None:
        try:
            dock_widget.raise_()
        except Exception:  # nosec B110
            pass
    if not raised:
        _flash_taskbar_entry(main_window)
    return raised


def _force_foreground_on_windows(main_window) -> bool:

    try:
        import ctypes




        user32 = ctypes.WinDLL("user32")


        user32.GetForegroundWindow.restype = ctypes.c_void_p
        user32.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        user32.GetWindowThreadProcessId.restype = ctypes.c_ulong
        user32.IsIconic.argtypes = [ctypes.c_void_p]
        user32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
        user32.BringWindowToTop.argtypes = [ctypes.c_void_p]
        user32.SetForegroundWindow.argtypes = [ctypes.c_void_p]
        user32.SetForegroundWindow.restype = ctypes.c_bool
        user32.AttachThreadInput.argtypes = [ctypes.c_ulong, ctypes.c_ulong, ctypes.c_bool]
        user32.AttachThreadInput.restype = ctypes.c_bool

        hwnd = ctypes.c_void_p(int(main_window.winId()))
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, _SW_RESTORE)
        foreground = ctypes.c_void_p(user32.GetForegroundWindow())
        target_thread = user32.GetWindowThreadProcessId(hwnd, None)
        front_thread = user32.GetWindowThreadProcessId(foreground, None)
        attached = False
        if front_thread and target_thread and front_thread != target_thread:
            attached = bool(user32.AttachThreadInput(front_thread, target_thread, True))
        try:
            user32.BringWindowToTop(hwnd)
            return bool(user32.SetForegroundWindow(hwnd))
        finally:
            if attached:
                user32.AttachThreadInput(front_thread, target_thread, False)
    except Exception:  # nosec B110
        return False


def _flash_taskbar_entry(main_window) -> None:

    try:
        from qgis.PyQt.QtWidgets import QApplication

        QApplication.alert(main_window, 3000)
    except Exception:  # nosec B110
        pass
