
from qgis.PyQt.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QProgressBar,
    QFrame,
    QMessageBox,
    QComboBox,
    QToolButton,
    QSizePolicy,
)
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QStandardItem, QStandardItemModel, QFont

from qgis.core import QgsMapLayerProxyModel
from qgis.gui import QgsMapLayerComboBox

from typing import List


class AISegmentationDockWidget(QDockWidget):

    install_dependencies_requested = pyqtSignal()
    download_models_requested = pyqtSignal()
    install_model_requested = pyqtSignal(str)  
    cancel_download_requested = pyqtSignal()
    cancel_preparation_requested = pyqtSignal()
    start_segmentation_requested = pyqtSignal(object)  # QgsRasterLayer
    clear_points_requested = pyqtSignal()
    undo_requested = pyqtSignal()
    finish_segmentation_requested = pyqtSignal()  
    model_changed = pyqtSignal(str)  # model_id

    def __init__(self, parent=None):
        super().__init__("AI Segmentation by Terralab", parent)

        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setSpacing(8)
        self.main_layout.setContentsMargins(8, 8, 8, 8)

        self._setup_ui()

        self.setWidget(self.main_widget)

        self._dependencies_ok = False
        self._models_ok = False
        self._segmentation_active = False
        self._has_mask = False  # Track if there's a current mask (points placed)
        self._installed_models: List[str] = []
        self._current_model_id: str = None
        self._downloading_model_id: str = None  # Track which model is being downloaded

    def _setup_ui(self):
        self._setup_dependencies_section()

        self._setup_model_install_section()

        self._setup_segmentation_section()

        self.main_layout.addStretch()

        self._setup_status_bar()

    def _setup_dependencies_section(self):
        self.deps_group = QGroupBox("Dependencies")
        layout = QVBoxLayout(self.deps_group)

        self.deps_status_label = QLabel("Checking dependencies...")
        layout.addWidget(self.deps_status_label)

        self.deps_progress = QProgressBar()
        self.deps_progress.setRange(0, 100)
        self.deps_progress.setVisible(False)
        layout.addWidget(self.deps_progress)

        self.deps_progress_label = QLabel("")
        self.deps_progress_label.setStyleSheet("color: #666; font-size: 10px;")
        self.deps_progress_label.setVisible(False)
        self.deps_progress_label.setWordWrap(True)
        layout.addWidget(self.deps_progress_label)

        self.install_button = QPushButton("Install Dependencies")
        self.install_button.clicked.connect(self._on_install_clicked)
        self.install_button.setVisible(False)
        self.install_button.setToolTip("Install required Python packages (onnxruntime, numpy)")
        layout.addWidget(self.install_button)

        self.main_layout.addWidget(self.deps_group)

    def _setup_model_install_section(self):
        self.model_install_container = QWidget()
        container_layout = QVBoxLayout(self.model_install_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(4)

        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)

        self.model_install_toggle = QToolButton()
        self.model_install_toggle.setArrowType(Qt.RightArrow)
        self.model_install_toggle.setCheckable(True)
        self.model_install_toggle.setChecked(False)
        self.model_install_toggle.clicked.connect(self._on_model_install_toggle)
        self.model_install_toggle.setStyleSheet("QToolButton { border: none; }")
        header_layout.addWidget(self.model_install_toggle)

        header_label = QLabel("Install AI Models")
        header_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(header_label)
        header_layout.addStretch()

        container_layout.addWidget(header_widget)

        self.model_install_content = QWidget()
        self.model_install_content.setVisible(False)
        content_layout = QVBoxLayout(self.model_install_content)
        content_layout.setContentsMargins(20, 4, 0, 4)
        content_layout.setSpacing(6)

        self.model_rows = {}
        self._create_model_rows(content_layout)

        container_layout.addWidget(self.model_install_content)

        self.models_progress = QProgressBar()
        self.models_progress.setRange(0, 100)
        self.models_progress.setVisible(False)
        container_layout.addWidget(self.models_progress)

        self.models_progress_label = QLabel("")
        self.models_progress_label.setStyleSheet("color: #666; font-size: 10px;")
        self.models_progress_label.setVisible(False)
        container_layout.addWidget(self.models_progress_label)

        self.cancel_download_button = QPushButton("Cancel")
        self.cancel_download_button.clicked.connect(self._on_cancel_download_clicked)
        self.cancel_download_button.setVisible(False)
        self.cancel_download_button.setStyleSheet("background-color: #d32f2f; color: white;")
        container_layout.addWidget(self.cancel_download_button)

        self.main_layout.addWidget(self.model_install_container)

    def _create_model_rows(self, layout):
        from .core.model_registry import get_all_models

        for config in get_all_models():
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 2, 0, 2)

            info_label = QLabel(f"{config.display_name} - {config.total_size_mb}MB")
            info_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            row_layout.addWidget(info_label)

            status_label = QLabel("")
            status_label.setVisible(False)
            row_layout.addWidget(status_label)

            install_btn = QPushButton("Install")
            install_btn.setFixedWidth(70)
            install_btn.clicked.connect(lambda checked, mid=config.model_id: self._on_model_install_clicked(mid))
            row_layout.addWidget(install_btn)

            self.model_rows[config.model_id] = {
                "row": row,
                "info_label": info_label,
                "status_label": status_label,
                "install_btn": install_btn,
            }

            layout.addWidget(row)

    def _setup_segmentation_section(self):
        
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        self.main_layout.addWidget(sep)

        seg_widget = QWidget()
        layout = QVBoxLayout(seg_widget)
        layout.setContentsMargins(0, 8, 0, 0)

        model_label = QLabel("AI Model:")
        layout.addWidget(model_label)

        self.model_combo = QComboBox()
        self.model_combo.setToolTip(
            "Select the AI model for segmentation.\n"
            "Grayed out models are not installed yet."
        )
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        layout.addWidget(self.model_combo)

        self._populate_model_combo()

        layout.addSpacing(8)

        layer_label = QLabel("Raster layer to segment:")
        layout.addWidget(layer_label)

        self.layer_combo = QgsMapLayerComboBox()
        self.layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.layer_combo.setAllowEmptyLayer(True)
        self.layer_combo.setShowCrs(True)
        self.layer_combo.layerChanged.connect(self._on_layer_changed)
        self.layer_combo.setToolTip("Select a raster layer (GeoTIFF, XYZ tiles, WMS, etc.)")
        layout.addWidget(self.layer_combo)

        self.active_instructions_label = QLabel(
            "CLICK ON THE MAP\n\n"
            "Left-click = INCLUDE\n"
            "Right-click = EXCLUDE"
        )
        self.active_instructions_label.setStyleSheet(
            "color: #1976d2; font-size: 11px; font-weight: bold; "
            "background-color: #e3f2fd; padding: 10px; border-radius: 4px;"
        )
        self.active_instructions_label.setWordWrap(True)
        self.active_instructions_label.setVisible(False)
        layout.addWidget(self.active_instructions_label)

        self.prep_progress = QProgressBar()
        self.prep_progress.setRange(0, 100)
        self.prep_progress.setVisible(False)
        layout.addWidget(self.prep_progress)

        self.prep_status_label = QLabel("")
        self.prep_status_label.setStyleSheet("color: #666; font-size: 10px;")
        self.prep_status_label.setVisible(False)
        layout.addWidget(self.prep_status_label)

        self.cancel_prep_button = QPushButton("Cancel")
        self.cancel_prep_button.clicked.connect(self._on_cancel_prep_clicked)
        self.cancel_prep_button.setVisible(False)
        self.cancel_prep_button.setStyleSheet("background-color: #d32f2f; color: white;")
        layout.addWidget(self.cancel_prep_button)

        self.start_button = QPushButton("Start AI Segmentation")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._on_start_clicked)
        self.start_button.setToolTip(
            "Activate point mode to click on the map.\n"
            "Left-click to include, right-click to exclude."
        )
        layout.addWidget(self.start_button)

        self.undo_button = QPushButton("Undo Last Point")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self._on_undo_clicked)
        self.undo_button.setVisible(False)
        self.undo_button.setToolTip("Remove the last point (Ctrl+Z)")
        layout.addWidget(self.undo_button)

        self.finish_button = QPushButton("Finish Segmentation")
        self.finish_button.clicked.connect(self._on_finish_clicked)
        self.finish_button.setVisible(False)
        self.finish_button.setStyleSheet("background-color: #388e3c; color: white;")
        self.finish_button.setToolTip("Save the segmentation as a new layer and start fresh")
        layout.addWidget(self.finish_button)

        self.main_layout.addWidget(seg_widget)

    def _populate_model_combo(self):
        from .core.model_registry import get_all_models

        self.model_combo.blockSignals(True)
        self.model_combo.clear()

        model = QStandardItemModel()
        for config in get_all_models():
            item = QStandardItem(config.display_name)
            item.setData(config.model_id, Qt.UserRole)
            item.setEnabled(False)
            model.appendRow(item)

        self.model_combo.setModel(model)
        self.model_combo.blockSignals(False)

    def _setup_status_bar(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        self.main_layout.addWidget(sep)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #666; font-size: 10px;")
        self.status_label.setWordWrap(True)
        self.main_layout.addWidget(self.status_label)

    def _on_install_clicked(self):
        self.install_button.setEnabled(False)
        self.install_dependencies_requested.emit()

    def _on_model_install_toggle(self, checked):
        self.model_install_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.model_install_content.setVisible(checked)

    def _on_model_install_clicked(self, model_id: str):
        self._downloading_model_id = model_id
        for mid, widgets in self.model_rows.items():
            widgets["install_btn"].setEnabled(False)
        self.install_model_requested.emit(model_id)

    def _on_cancel_download_clicked(self):
        self.cancel_download_requested.emit()

    def _on_cancel_prep_clicked(self):
        self.cancel_preparation_requested.emit()

    def _on_layer_changed(self, layer):
        self._update_ui_state()

        if self._segmentation_active:
            self._segmentation_active = False
            self._update_button_visibility()
            QMessageBox.warning(
                self,
                "Layer Changed",
                "Segmentation cancelled because the layer was changed."
            )

    def _on_model_combo_changed(self, index):
        if index < 0:
            return

        model = self.model_combo.model()
        item = model.item(index)
        model_id = item.data(Qt.UserRole)

        if not item.isEnabled():
            self._select_model_in_combo(self._current_model_id)
            QMessageBox.information(
                self,
                "Model Not Installed",
                f"Please install this model first using the 'Install AI Models' section above."
            )
            return

        if self._segmentation_active:
            self._select_model_in_combo(self._current_model_id)
            QMessageBox.warning(
                self,
                "Segmentation Active",
                "Please finish or cancel the current segmentation before switching models."
            )
            return

        if model_id != self._current_model_id:
            self._current_model_id = model_id
            self.model_changed.emit(model_id)

    def _on_start_clicked(self):
        layer = self.layer_combo.currentLayer()
        if layer:
            self.start_segmentation_requested.emit(layer)

    def _on_undo_clicked(self):
        
        self.undo_requested.emit()

    def _on_finish_clicked(self):
        self.finish_segmentation_requested.emit()

    def set_dependency_status(self, ok: bool, message: str):
        self._dependencies_ok = ok
        self.deps_status_label.setText(message)

        if ok:
            self.deps_status_label.setStyleSheet("color: #388e3c; font-weight: bold;")
            self.install_button.setVisible(False)
            self.deps_progress.setVisible(False)
            self.deps_progress_label.setVisible(False)
            self.deps_group.setVisible(False)
        else:
            self.deps_status_label.setStyleSheet("color: #f57c00;")
            self.install_button.setVisible(True)
            self.install_button.setEnabled(True)
            self.deps_group.setVisible(True)

        self._update_ui_state()

    def set_install_progress(self, percent: int, message: str):
        self.deps_progress.setValue(percent)
        self.deps_progress_label.setText(message)

        if percent == 0:
            self.deps_progress.setVisible(True)
            self.deps_progress_label.setVisible(True)
            self.install_button.setEnabled(False)
            self.install_button.setText("Installing...")
        elif percent >= 100:
            self.deps_progress.setVisible(False)
            if "failed" in message.lower():
                self.install_button.setEnabled(True)
                self.install_button.setText("Retry")

    def set_models_status(self, installed_models: List[str], current_model_id: str = None):
        self._installed_models = installed_models
        self._models_ok = len(installed_models) > 0

        if current_model_id:
            self._current_model_id = current_model_id

        for model_id, widgets in self.model_rows.items():
            is_installed = model_id in installed_models
            if is_installed:
                widgets["status_label"].setText("Installed")
                widgets["status_label"].setStyleSheet("color: #388e3c; font-weight: bold;")
                widgets["status_label"].setVisible(True)
                widgets["install_btn"].setVisible(False)
            else:
                widgets["status_label"].setVisible(False)
                widgets["install_btn"].setVisible(True)
                widgets["install_btn"].setEnabled(True)

        self._update_model_combo_state()

        if current_model_id:
            self._select_model_in_combo(current_model_id)

        if self._models_ok:
            self.model_install_toggle.setChecked(False)
            self.model_install_content.setVisible(False)
            self.model_install_toggle.setArrowType(Qt.RightArrow)
        else:
            self.model_install_toggle.setChecked(True)
            self.model_install_content.setVisible(True)
            self.model_install_toggle.setArrowType(Qt.DownArrow)

        self._update_ui_state()

    def _update_model_combo_state(self):
        
        model = self.model_combo.model()
        for i in range(model.rowCount()):
            item = model.item(i)
            model_id = item.data(Qt.UserRole)
            is_installed = model_id in self._installed_models
            item.setEnabled(is_installed)

            from .core.model_registry import get_model_config
            config = get_model_config(model_id)
            if is_installed:
                item.setText(config.display_name)
            else:
                item.setText(f"{config.display_name} (not installed)")

    def _select_model_in_combo(self, model_id: str):
        if not model_id:
            return

        model = self.model_combo.model()
        self.model_combo.blockSignals(True)
        for i in range(model.rowCount()):
            item = model.item(i)
            if item.data(Qt.UserRole) == model_id:
                self.model_combo.setCurrentIndex(i)
                break
        self.model_combo.blockSignals(False)

    def set_download_progress(self, percent: int, message: str):
        self.models_progress.setValue(percent)
        self.models_progress_label.setText(message)

        if percent == 0:
            self.models_progress.setVisible(True)
            self.models_progress_label.setVisible(True)
            self.cancel_download_button.setVisible(True)
            self.model_install_toggle.setChecked(True)
            self.model_install_content.setVisible(True)
            self.model_install_toggle.setArrowType(Qt.DownArrow)
        elif percent >= 100 or "cancel" in message.lower():
            self.models_progress.setVisible(False)
            self.models_progress_label.setVisible(False)
            self.cancel_download_button.setVisible(False)
            self._downloading_model_id = None
            for mid, widgets in self.model_rows.items():
                if mid not in self._installed_models:
                    widgets["install_btn"].setEnabled(True)

    def set_preparation_progress(self, percent: int, message: str):
        self.prep_progress.setValue(percent)
        self.prep_status_label.setText(message)

        if percent == 0:
            self.prep_progress.setVisible(True)
            self.prep_status_label.setVisible(True)
            self.start_button.setEnabled(False)
            self.cancel_prep_button.setVisible(True)
        elif percent >= 100 or "cancel" in message.lower():
            self.prep_progress.setVisible(False)
            self.prep_status_label.setVisible(False)
            self.cancel_prep_button.setVisible(False)
            self._update_ui_state()

    def set_segmentation_active(self, active: bool):
        self._segmentation_active = active
        self._update_button_visibility()
        self._update_ui_state()

    def _update_button_visibility(self):
        if self._segmentation_active:
            self.start_button.setVisible(False)
            self.active_instructions_label.setVisible(True)
            self.undo_button.setVisible(True)
            self.finish_button.setVisible(True)
            self.finish_button.setEnabled(self._has_mask)
            self.model_combo.setEnabled(False)
        else:
            self.start_button.setVisible(True)
            self.active_instructions_label.setVisible(False)
            self.undo_button.setVisible(False)
            self.finish_button.setVisible(False)
            self.model_combo.setEnabled(True)

    def set_point_count(self, positive: int, negative: int):
        total = positive + negative
        has_points = total > 0
        self._has_mask = has_points

        self.undo_button.setEnabled(has_points and self._segmentation_active)
        self.finish_button.setEnabled(has_points)

        if has_points:
            self.status_label.setText(f"Points: {positive} + / {negative} -")

    def set_status(self, message: str):
        self.status_label.setText(message)

    def reset_session(self):
        self._has_mask = False
        self._segmentation_active = False
        self._update_button_visibility()
        self._update_ui_state()

    def get_selected_model_id(self) -> str:
        return self._current_model_id

    def _update_ui_state(self):
        has_layer = self.layer_combo.currentLayer() is not None

        can_start = (
            self._dependencies_ok and
            self._models_ok and
            has_layer and
            self._current_model_id is not None
        )
        self.start_button.setEnabled(can_start or self._segmentation_active)
