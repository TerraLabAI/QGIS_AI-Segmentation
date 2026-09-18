





from __future__ import annotations

import os

from ..core.i18n import tr
from ..workers.generic_request_task import GenericRequestTask
from .external_links import open_local_path




_DIR_SIZE_CACHE: dict[str, str] = {}







_REMOVAL_WATCHDOG_MS = 300_000


class AccountRemovalMixin:





    def _on_remove_ai_data_clicked(self):




        from .dialogs.confirm_dialog import (
            DANGER,
            SECONDARY,
            WARNING,
            ChoiceButton,
            ask_choice,
            info_box,
            warning_box,
        )

        if self._removal_running:
            return



        if self._is_busy_check is not None:
            try:
                if self._is_busy_check():
                    info_box(
                        self,
                        tr("An install or detection is still running. Wait for "
                           "it to finish, then try again."))
                    return
            except Exception:  # nosec B110
                pass


        choice = ask_choice(
            self, tr("Remove the downloaded AI data from this computer?"),
            tr("Deletes the offline AI files and signs you out. Your account stays."),
            [ChoiceButton("cancel", tr("Cancel"), SECONDARY),
             ChoiceButton("remove", tr("Remove"), DANGER)],
            default=None, escape="cancel", tone=WARNING)
        if choice != "remove":
            return

        self._removal_running = True
        self._removal_generation += 1
        generation = self._removal_generation
        if self._remove_btn is not None:
            self._remove_btn.setEnabled(False)
            self._remove_btn.setText(tr("Removing..."))
        self._set_remove_status(tr("Removing the downloaded AI data..."))
        try:
            started, message = self._on_remove_ai_data(
                self._on_remove_ai_data_progress,
                self._on_remove_ai_data_finished)
        except Exception:  # nosec B110
            started, message = False, tr("Could not remove the AI data. Try again.")

        if not started:


            self._removal_running = False
            self._set_remove_status("")
            self._reset_remove_button()
            warning_box(self, message)
            return



        try:



            from ..core.qt_compat import safe_single_shot
            from ..core.surface_dials import removal_watchdog_ms
            safe_single_shot(
                removal_watchdog_ms(_REMOVAL_WATCHDOG_MS), self,
                lambda g=generation: self._on_removal_watchdog(g))
        except (RuntimeError, AttributeError):
            pass

    def _on_removal_watchdog(self, generation: int) -> None:




        if generation != getattr(self, "_removal_generation", 0):
            return
        if not self._removal_running:
            return
        try:
            self._on_remove_ai_data_finished(False, tr(
                "The removal did not finish. Close this window, then check "
                "the AI data folder before trying again."))
        except RuntimeError:

            self._removal_running = False

    def _on_remove_ai_data_progress(self, text: str):

        self._set_remove_status(text)

    def _on_remove_ai_data_finished(self, ok: bool, message: str):




        from .dialogs.confirm_dialog import success_box, warning_box

        if not self._removal_running:
            return
        self._removal_running = False
        self._set_remove_status("")
        self._reset_remove_button()
        if ok:

            _DIR_SIZE_CACHE.clear()
            success_box(self, message)

            self.accept()
        else:
            warning_box(self, message)

    def _set_remove_status(self, text: str):

        label = getattr(self, "_remove_status", None)
        if label is None:
            return
        try:
            label.setText(text)
            label.setVisible(bool(text))
        except RuntimeError:
            self._remove_status = None

    def _reset_remove_button(self):



        btn = getattr(self, "_remove_btn", None)
        if btn is None:
            return
        busy = False
        check = getattr(self, "_is_busy_check", None)
        if check is not None:
            try:
                busy = bool(check())
            except Exception:  # noqa: BLE001  # nosec B110
                busy = False
        try:
            btn.setEnabled(not busy)
            btn.setText(tr("Remove"))
            btn.setToolTip(
                tr("Available once the current install or detection finishes.")
                if busy else "")
        except RuntimeError:
            self._remove_btn = None

    @staticmethod
    def _open_install_folder(path: str):


        target = path
        while target and not os.path.isdir(target):
            parent = os.path.dirname(target)
            if parent == target:
                break
            target = parent
        open_local_path(target)

    def _start_dir_size_task(self, path: str):






        from qgis.core import QgsApplication

        from .plugin.shared import dir_size_label



        task = GenericRequestTask(
            tr("Measuring AI data size..."),
            lambda: {"path": path, "size": dir_size_label(path)},
            hidden=True,
        )
        task.succeeded.connect(self._on_dir_size_ready)
        self._size_task = task
        QgsApplication.taskManager().addTask(task)

    def _on_dir_size_ready(self, data: dict):
        self._size_task = None
        size = str(data.get("size") or "-")
        _DIR_SIZE_CACHE[str(data.get("path") or "")] = size
        label = self._size_label
        if label is None:
            return
        try:
            label.setText(f"{tr('On disk')}: {size}")
        except RuntimeError:

            self._size_label = None

    def _cancel_size_task(self):

        if self._size_task is None:
            return
        try:
            self._size_task.succeeded.disconnect()
        except (RuntimeError, TypeError):  # nosec B110
            pass
        try:
            self._size_task.cancel()
        except Exception:  # nosec B110
            pass
        self._size_task = None
