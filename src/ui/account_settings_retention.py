





from __future__ import annotations

from qgis.PyQt.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

from ..core.i18n import tr
from ..core.qt_compat import safe_disconnect
from ..workers.generic_request_task import GenericRequestTask

_CHOICES = (0, 30, 90, 180, 365, None)


def _days_label(days) -> str:
    if days in (1, 7):
        return tr("1 day") if days == 1 else tr("7 days")
    return {
        0: tr("No copies (0 days)"),
        30: tr("30 days"),
        90: tr("90 days"),
        180: tr("6 months"),
        365: tr("1 year"),
    }.get(days, tr("Until I delete it"))


def _rank(days) -> float:


    return float("inf") if days is None or days == 0 else float(days)


class AccountRetentionMixin:


    def _add_retention_row(self, group) -> None:
        from .settings.account_page import _row_tile
        from .settings.settings_widgets import SettingRow

        self._retention_tasks: list = []
        self._retention_state: dict | None = None
        self._retention_busy = False
        self._retention_control = QWidget()
        box = QHBoxLayout(self._retention_control)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(8)
        self._retention_row = SettingRow(
            tr("History retention"), "", self._retention_control,
            lead=_row_tile("clock", "green"))
        self._retention_row.setVisible(False)
        group.add_row(self._retention_row)
        if not self._auth or self._client is None:
            return
        client, auth = self._client, self._auth
        self._run_retention_task(lambda: client.get_data_retention(auth),
                                 self._on_retention_loaded, self._on_retention_load_failed)

    def _run_retention_task(self, fn, ok, ko) -> None:
        from qgis.core import QgsApplication

        task = GenericRequestTask(tr("History retention..."), fn, hidden=True)
        task.succeeded.connect(ok)
        task.failed.connect(ko)
        self._retention_tasks.append(task)
        QgsApplication.taskManager().addTask(task)

    def _cancel_retention_tasks(self) -> None:
        for task in getattr(self, "_retention_tasks", []):
            safe_disconnect(task, "succeeded")
            safe_disconnect(task, "failed")
            try:
                task.cancel()
            except Exception:  # nosec B110
                pass
        self._retention_tasks = []



    def _on_retention_loaded(self, data) -> None:
        if not isinstance(data, dict) or data.get("tier") not in ("free", "pro", "zero"):
            self._on_retention_load_failed("", "")
            return
        self._retention_state = dict(data)
        try:
            self._paint_retention()
        except RuntimeError:
            pass  # nosec B110

    def _on_retention_load_failed(self, _message: str = "", _code: str = "") -> None:
        try:
            self._retention_row.setVisible(False)
        except (RuntimeError, AttributeError):
            pass  # nosec B110



    def _paint_retention(self) -> None:
        from .dock.styles import _BTN_SETTINGS_GHOST
        from .settings.settings_widgets import ROW_NOTE_QSS, clear_layout, settings_button

        state = self._retention_state or {}
        tier = state.get("tier")
        days = state.get("history_retention_days")
        box = self._retention_control.layout()
        clear_layout(box)
        self._retention_combo = None
        row = self._retention_row

        if tier == "pro":
            combo = QComboBox(self._retention_control)
            for value in _CHOICES:
                combo.addItem(_days_label(value), value)
            if days in (1, 7):

                combo.addItem(_days_label(days), days)
                item = combo.model().item(combo.count() - 1)
                if item is not None:
                    item.setEnabled(False)
            combo.setCurrentIndex(self._retention_index(combo, days))
            combo.setEnabled(bool(state.get("can_change", True)))
            combo.currentIndexChanged.connect(self._on_retention_combo_changed)
            combo.setAccessibleName(tr("History retention"))
            box.addWidget(combo)
            self._retention_combo = combo
            row.set_note(self._retention_pro_note(days))
        elif tier == "zero":
            box.addWidget(QLabel(tr("Set by your contract"), self._retention_control))
            row.set_note(tr("Zero data retention: no copy of your work is kept beyond "
                            "what your contract allows."))
        elif days is None:
            box.addWidget(QLabel(tr("Until you delete it"), self._retention_control))
            row.set_note(tr("Free plan. Pro lets you choose 30 days to 1 year."))
        else:
            box.addWidget(QLabel(_days_label(days), self._retention_control))
            if state.get("can_change"):
                keep = settings_button(tr("Keep until I delete it"), _BTN_SETTINGS_GHOST)
                keep.clicked.connect(lambda: self._save_retention(None))
                box.addWidget(keep)
            row.set_note(tr("Free plan. Pro lets you choose 30 days to 1 year."))
        row.note_label.setStyleSheet(ROW_NOTE_QSS)
        row.note_label.setVisible(True)
        row.setVisible(True)



    def _on_retention_combo_changed(self, index: int) -> None:
        combo = self._retention_combo
        if combo is None or self._retention_busy:
            return
        new = combo.itemData(index)
        current = (self._retention_state or {}).get("history_retention_days")
        if new == current:
            return
        if _rank(new) < _rank(current):
            from .dialogs.confirm_dialog import question

            if not question(self, tr("History retention"),
                            tr("Older history will be deleted for good, files "
                               "included, within two days. Continue?"),
                            default_yes=False, destructive=True):
                self._revert_retention_combo()
                return
        self._save_retention(new)

    def _save_retention(self, days) -> None:
        if self._retention_busy:
            return
        self._retention_busy = True
        self._retention_control.setEnabled(False)
        client, auth = self._client, self._auth
        self._run_retention_task(lambda: client.set_data_retention(auth, days),
                                 self._on_retention_saved, self._on_retention_save_failed)

    def _on_retention_saved(self, data) -> None:
        self._retention_busy = False
        try:
            self._retention_control.setEnabled(True)
            if isinstance(data, dict) and data.get("tier") in ("free", "pro", "zero"):
                self._retention_state = dict(data)
            self._paint_retention()
            self._show_saved()
        except RuntimeError:
            pass  # nosec B110

    def _on_retention_save_failed(self, message: str = "", _code: str = "") -> None:
        from .settings.settings_widgets import ROW_ERROR_QSS

        self._retention_busy = False
        try:
            self._retention_control.setEnabled(True)
            self._revert_retention_combo()
            text = (message or "").strip() or tr("Could not save. Try again.")
            self._retention_row.set_note(text)
            self._retention_row.note_label.setStyleSheet(ROW_ERROR_QSS)
            self._retention_row.note_label.setVisible(True)
        except RuntimeError:
            pass  # nosec B110

    def _revert_retention_combo(self) -> None:
        combo = self._retention_combo
        if combo is None:
            return
        current = (self._retention_state or {}).get("history_retention_days")
        combo.blockSignals(True)
        combo.setCurrentIndex(self._retention_index(combo, current))
        combo.blockSignals(False)
        try:
            self._retention_row.set_note(self._retention_pro_note(current))
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    @staticmethod
    def _retention_index(combo, days) -> int:
        values = [combo.itemData(i) for i in range(combo.count())]
        for i, value in enumerate(values):
            if value is days or (days is not None and value == days):
                return i
        return values.index(None) if None in values else 0

    @staticmethod
    def _retention_pro_note(days) -> str:
        if days == 0:
            return tr("No copy of your work is written. Your AI Edit generations and "
                      "saved outlines stay until you delete them.")
        return tr("How long your history stays on our servers, for all TerraLab "
                  "plugins. Seen by you only, never used to improve anything.")
