"""Bring the QGIS main window back to the front after a browser round trip.

``activateWindow()`` alone is enough on macOS and most Linux desktops. On
Windows it is not: SetForegroundWindow refuses a process that does not own the
foreground window, so a plugin calling it while the browser is in front only
gets a flashing taskbar button. The Windows branch below attaches to the
foreground thread's input queue first, which is what the OS accepts.

Kept in sync with the same file in the AI Edit plugin.
"""
from __future__ import annotations

import sys

# Win32 constant, ShowWindow(SW_RESTORE): un-minimize without changing size.
_SW_RESTORE = 9


def bring_qgis_window_to_front(main_window, dock_widget=None) -> bool:
    """Raise and focus the QGIS window; return False if the OS refused.

    Never raises: a failed raise leaves the user on the browser tab, which is
    the behaviour we already had, so it must not break the caller's flow.
    """
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
        raised = _force_foreground_on_windows(main_window) or raised
    if dock_widget is not None:
        try:
            dock_widget.raise_()
        except Exception:  # nosec B110
            pass
    if not raised:
        _flash_taskbar_entry(main_window)
    return raised


def _force_foreground_on_windows(main_window) -> bool:
    """SetForegroundWindow with the input-queue attach Windows requires."""
    try:
        import ctypes

        user32 = ctypes.windll.user32
        # Handles are pointer sized: without these the default c_int return
        # truncates every HWND on 64-bit Windows.
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
    """Last resort when the OS keeps the browser in front: flash the button."""
    try:
        from qgis.PyQt.QtWidgets import QApplication

        QApplication.alert(main_window, 3000)
    except Exception:  # nosec B110
        pass
