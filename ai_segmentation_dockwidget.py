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
    QSizePolicy,
)
from qgis.PyQt.QtCore import Qt, pyqtSignal

from qgis.core import QgsMapLayerProxyModel
from qgis.gui import QgsMapLayerComboBox


class AISegmentationDockWidget(QDockWidget):

    install_dependencies_requested = pyqtSignal()
    cancel_deps_install_requested = pyqtSignal()
    download_checkpoint_requested = pyqtSignal()
    cancel_download_requested = pyqtSignal()
    cancel_preparation_requested = pyqtSignal()
    start_segmentation_requested = pyqtSignal(object)
    clear_points_requested = pyqtSignal()
    undo_requested = pyqtSignal()
    save_polygon_requested = pyqtSignal()
    export_layer_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("AI Segmentation by TerraLab", parent)

        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setSpacing(8)
        self.main_layout.setContentsMargins(8, 8, 8, 8)

        self._setup_ui()

        self.setWidget(self.main_widget)

        self._dependencies_ok = False
        self._checkpoint_ok = False
        self._segmentation_active = False
        self._has_mask = False
        self._saved_polygon_count = 0

    def _setup_ui(self):
        self._setup_dependencies_section()
        self._setup_checkpoint_section()
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
        layout.addWidget(self.deps_progress_label)

        self.install_button = QPushButton("Install Dependencies")
        self.install_button.clicked.connect(self._on_install_clicked)
        self.install_button.setVisible(False)
        self.install_button.setToolTip(
            "Create isolated virtual environment and install required packages\n"
            "(PyTorch, Segment Anything, pandas, rasterio)\n"
            "Download size: ~2.5GB"
        )
        layout.addWidget(self.install_button)

        self.cancel_deps_button = QPushButton("Cancel")
        self.cancel_deps_button.clicked.connect(self._on_cancel_deps_clicked)
        self.cancel_deps_button.setVisible(False)
        self.cancel_deps_button.setStyleSheet("background-color: #d32f2f; color: white;")
        layout.addWidget(self.cancel_deps_button)

        self.main_layout.addWidget(self.deps_group)

    def _setup_checkpoint_section(self):
        self.checkpoint_group = QGroupBox("AI Model")
        layout = QVBoxLayout(self.checkpoint_group)

        self.checkpoint_status_label = QLabel("Checking model...")
        layout.addWidget(self.checkpoint_status_label)

        self.checkpoint_progress = QProgressBar()
        self.checkpoint_progress.setRange(0, 100)
        self.checkpoint_progress.setVisible(False)
        layout.addWidget(self.checkpoint_progress)

        self.checkpoint_progress_label = QLabel("")
        self.checkpoint_progress_label.setStyleSheet("color: #666; font-size: 10px;")
        self.checkpoint_progress_label.setVisible(False)
        layout.addWidget(self.checkpoint_progress_label)

        self.download_button = QPushButton("Download SAM Model (~375MB)")
        self.download_button.clicked.connect(self._on_download_clicked)
        self.download_button.setVisible(False)
        self.download_button.setToolTip("Download the SAM vit_b checkpoint for segmentation")
        layout.addWidget(self.download_button)

        self.cancel_download_button = QPushButton("Cancel")
        self.cancel_download_button.clicked.connect(self._on_cancel_download_clicked)
        self.cancel_download_button.setVisible(False)
        self.cancel_download_button.setStyleSheet("background-color: #d32f2f; color: white;")
        layout.addWidget(self.cancel_download_button)

        self.main_layout.addWidget(self.checkpoint_group)

    def _setup_segmentation_section(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        self.main_layout.addWidget(sep)

        seg_widget = QWidget()
        layout = QVBoxLayout(seg_widget)
        layout.setContentsMargins(0, 8, 0, 0)

        layer_label = QLabel("Raster layer to segment:")
        layout.addWidget(layer_label)

        self.layer_combo = QgsMapLayerComboBox()
        self.layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.layer_combo.setAllowEmptyLayer(True)
        self.layer_combo.setShowCrs(True)
        self.layer_combo.layerChanged.connect(self._on_layer_changed)
        self.layer_combo.setToolTip("Select a file-based raster layer (GeoTIFF, etc.)")
        layout.addWidget(self.layer_combo)

        note_label = QLabel("Note: Only file-based rasters supported (no web layers)")
        note_label.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        note_label.setWordWrap(True)
        layout.addWidget(note_label)

        self.active_instructions_label = QLabel(
            "CLICK ON THE MAP\n\n"
            "Left-click = INCLUDE\n"
            "Right-click = EXCLUDE\n"
            "S = Save polygon"
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

        self.save_polygon_button = QPushButton("Save Polygon (S)")
        self.save_polygon_button.clicked.connect(self._on_save_polygon_clicked)
        self.save_polygon_button.setVisible(False)
        self.save_polygon_button.setEnabled(False)
        self.save_polygon_button.setStyleSheet("background-color: #1976d2; color: white;")
        self.save_polygon_button.setToolTip("Save current polygon and start a new one (S)")
        layout.addWidget(self.save_polygon_button)

        self.export_button = QPushButton("Export to Layer")
        self.export_button.clicked.connect(self._on_export_clicked)
        self.export_button.setVisible(False)
        self.export_button.setEnabled(False)
        self.export_button.setStyleSheet("background-color: #388e3c; color: white;")
        self.export_button.setToolTip("Export all saved polygons as a new layer")
        layout.addWidget(self.export_button)

        self.main_layout.addWidget(seg_widget)

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

    def _on_cancel_deps_clicked(self):
        self.cancel_deps_install_requested.emit()

    def _on_download_clicked(self):
        self.download_button.setEnabled(False)
        self.download_checkpoint_requested.emit()

    def _on_cancel_download_clicked(self):
        self.cancel_download_requested.emit()

    def _on_cancel_prep_clicked(self):
        self.cancel_preparation_requested.emit()

    def _on_layer_changed(self, _layer):
        self._update_ui_state()

        if self._segmentation_active:
            self._segmentation_active = False
            self._update_button_visibility()
            QMessageBox.warning(
                self,
                "Layer Changed",
                "Segmentation cancelled because the layer was changed."
            )

    def _on_start_clicked(self):
        layer = self.layer_combo.currentLayer()
        if layer:
            self.start_segmentation_requested.emit(layer)

    def _on_undo_clicked(self):
        self.undo_requested.emit()

    def _on_save_polygon_clicked(self):
        self.save_polygon_requested.emit()

    def _on_export_clicked(self):
        self.export_layer_requested.emit()

    def set_dependency_status(self, ok: bool, message: str):
        self._dependencies_ok = ok
        self.deps_status_label.setText(message)

        if ok:
            self.deps_status_label.setStyleSheet("color: #388e3c; font-weight: bold;")
            self.install_button.setVisible(False)
            self.cancel_deps_button.setVisible(False)
            self.deps_progress.setVisible(False)
            self.deps_progress_label.setVisible(False)
            self.deps_group.setVisible(False)
        else:
            self.deps_status_label.setStyleSheet("color: #f57c00;")
            self.install_button.setVisible(True)
            self.install_button.setEnabled(True)
            self.deps_group.setVisible(True)

        self._update_ui_state()

    def set_deps_install_progress(self, percent: int, message: str):
        self.deps_progress.setValue(percent)
        self.deps_progress_label.setText(message)

        if percent == 0:
            self.deps_progress.setVisible(True)
            self.deps_progress_label.setVisible(True)
            self.cancel_deps_button.setVisible(True)
            self.install_button.setEnabled(False)
            self.install_button.setText("Installing...")
        elif percent >= 100 or "cancel" in message.lower() or "failed" in message.lower():
            self.deps_progress.setVisible(False)
            self.deps_progress_label.setVisible(False)
            self.cancel_deps_button.setVisible(False)
            self.install_button.setEnabled(True)
            self.install_button.setText("Install Dependencies")


    def set_checkpoint_status(self, ok: bool, message: str):
        self._checkpoint_ok = ok
        self.checkpoint_status_label.setText(message)

        if ok:
            self.checkpoint_status_label.setStyleSheet("color: #388e3c; font-weight: bold;")
            self.download_button.setVisible(False)
            self.checkpoint_progress.setVisible(False)
            self.checkpoint_progress_label.setVisible(False)
            self.checkpoint_group.setVisible(False)
        else:
            self.checkpoint_status_label.setStyleSheet("color: #f57c00;")
            self.download_button.setVisible(True)
            self.download_button.setEnabled(True)
            self.checkpoint_group.setVisible(True)

        self._update_ui_state()

    def set_download_progress(self, percent: int, message: str):
        self.checkpoint_progress.setValue(percent)
        self.checkpoint_progress_label.setText(message)

        if percent == 0:
            self.checkpoint_progress.setVisible(True)
            self.checkpoint_progress_label.setVisible(True)
            self.cancel_download_button.setVisible(True)
            self.download_button.setEnabled(False)
            self.download_button.setText("Downloading...")
        elif percent >= 100 or "cancel" in message.lower():
            self.checkpoint_progress.setVisible(False)
            self.checkpoint_progress_label.setVisible(False)
            self.cancel_download_button.setVisible(False)
            self.download_button.setEnabled(True)
            self.download_button.setText("Download SAM Model (~375MB)")

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
            self.save_polygon_button.setVisible(True)
            self.save_polygon_button.setEnabled(self._has_mask)
            self.export_button.setVisible(True)
            self.export_button.setEnabled(self._saved_polygon_count > 0 or self._has_mask)
        else:
            self.start_button.setVisible(True)
            self.active_instructions_label.setVisible(False)
            self.undo_button.setVisible(False)
            self.save_polygon_button.setVisible(False)
            self.export_button.setVisible(False)

    def set_point_count(self, positive: int, negative: int):
        total = positive + negative
        has_points = total > 0
        self._has_mask = has_points

        self.undo_button.setEnabled(has_points and self._segmentation_active)
        self.save_polygon_button.setEnabled(has_points)

        if has_points:
            self.status_label.setText(f"Points: {positive} + / {negative} -")

    def set_status(self, message: str):
        self.status_label.setText(message)

    def reset_session(self):
        self._has_mask = False
        self._segmentation_active = False
        self._saved_polygon_count = 0
        self._update_button_visibility()
        self._update_ui_state()

    def set_saved_polygon_count(self, count: int):
        self._saved_polygon_count = count
        self._update_button_visibility()
        if count > 0:
            self.export_button.setText(f"Export to Layer ({count})")
        else:
            self.export_button.setText("Export to Layer")

    def _update_ui_state(self):
        layer = self.layer_combo.currentLayer()
        has_layer = layer is not None

        is_file_based = False
        if has_layer:
            source = layer.source()
            is_file_based = not any(x in source.lower() for x in ['http://', 'https://', 'wms', 'xyz', 'url='])

        can_start = (
            self._dependencies_ok and
            self._checkpoint_ok and
            has_layer and
            is_file_based
        )
        self.start_button.setEnabled(can_start or self._segmentation_active)

        if has_layer and not is_file_based and not self._segmentation_active:
            self.set_status("Web layers not supported - select a file-based raster")
