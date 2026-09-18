







from __future__ import annotations

import os

from qgis.PyQt.QtWidgets import QLabel

from ...core.i18n import tr
from ..dock.styles import _BTN_SETTINGS_DANGER, _BTN_SETTINGS_GHOST
from .category_tile import TILE_SMALL_GLYPH_PX, TILE_SMALL_PX, category_icon_tile
from .settings_widgets import (
    ROW_NOTE_QSS,
    ElidedLabel,
    SettingGroup,
    SettingRow,
    SettingsPage,
    settings_button,
)


class LocalModelPageMixin:


    def _build_local_model_page(self) -> SettingsPage:
        from ...core.cache_paths import PLUGIN_CACHE_DIR
        from ...core.venv_manager import VENV_DIR
        from ..account_settings_removal import _DIR_SIZE_CACHE

        page = SettingsPage(tr("Local model"), tr("Semi-Auto's offline AI files."), self,
                            glyph="package", category="teal")
        group = SettingGroup()
        installed = os.path.isdir(VENV_DIR)
        cached = _DIR_SIZE_CACHE.get(PLUGIN_CACHE_DIR)
        size = (cached or tr("Calculating...")) if installed else tr("Not installed")
        open_btn = settings_button(tr("Open folder"), _BTN_SETTINGS_GHOST)
        open_btn.clicked.connect(lambda: self._open_install_folder(PLUGIN_CACHE_DIR))
        files_row = SettingRow(f"{tr('On disk')}: {size}", "", open_btn,
                               lead=category_icon_tile("laptop", "teal", None,
                                                       TILE_SMALL_PX, TILE_SMALL_GLYPH_PX))
        path_label = ElidedLabel(PLUGIN_CACHE_DIR, files_row)
        path_label.setStyleSheet(ROW_NOTE_QSS)
        files_row.words.addWidget(path_label)
        self._size_label = files_row.title_label
        group.add_row(files_row)
        if installed and cached is None:
            self._start_dir_size_task(PLUGIN_CACHE_DIR)



        if self._on_remove_ai_data is not None:
            busy = False
            if self._is_busy_check is not None:
                try:
                    busy = bool(self._is_busy_check())
                except Exception:  # noqa: BLE001
                    busy = False  # nosec B110
            self._remove_btn = settings_button(tr("Remove"), _BTN_SETTINGS_DANGER)
            self._remove_btn.setEnabled(not busy)
            self._remove_btn.setAccessibleName(tr("Remove downloaded AI data"))
            if busy:
                self._remove_btn.setToolTip(
                    tr("Available once the current install or detection finishes."))
            self._remove_btn.clicked.connect(self._on_remove_ai_data_clicked)
            remove_row = SettingRow(tr("Remove AI files"), tr("Also signs you out"),
                                    self._remove_btn,
                                    lead=category_icon_tile("trash", "coral", None,
                                                            TILE_SMALL_PX, TILE_SMALL_GLYPH_PX))
            remove_row.setToolTip(tr(
                "Deletes the local model files, signs you out and resets the plugin. "
                "Your account and your cloud detections are not affected."))
            self._remove_status = QLabel("")
            self._remove_status.setWordWrap(True)
            self._remove_status.setStyleSheet(ROW_NOTE_QSS)
            self._remove_status.setVisible(False)
            remove_row.words.addWidget(self._remove_status)
            group.add_row(remove_row)
        page.add(group)
        return page
