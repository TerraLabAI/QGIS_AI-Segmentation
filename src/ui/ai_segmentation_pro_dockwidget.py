import pathlib

from qgis.PyQt.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QFrame,
    QSpinBox,
    QToolButton,
    QStyle,
    QScrollArea,
    QSizePolicy,
)
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.core import QgsMapLayerProxyModel, QgsProject

from qgis.gui import QgsMapLayerComboBox

from ..core.activation_manager import is_plugin_activated  # noqa: E402
from ..core.i18n import tr  # noqa: E402


class AISegmentationProDockWidget(QDockWidget):

    start_pro_segmentation_requested = pyqtSignal(object)  # layer
    save_polygon_requested = pyqtSignal()
    export_layer_requested = pyqtSignal()
    undo_requested = pyqtSignal()
    stop_segmentation_requested = pyqtSignal()
    pro_detect_requested = pyqtSignal()
    clear_points_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(tr("AI Segmentation PRO by TerraLab"), parent)

        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        self._setup_title_bar()

        self._segmentation_active = False
        self._has_mask = False
        self._saved_polygon_count = 0
        self._positive_count = 0  # batch detection count in PRO mode

        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setSpacing(8)
        self.main_layout.setContentsMargins(8, 8, 8, 8)

        self._setup_ui()

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.main_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidget(scroll_area)

        QgsProject.instance().layersAdded.connect(self._on_layers_changed)
        QgsProject.instance().layersRemoved.connect(self._on_layers_changed)

        self._update_ui_state()

    def _setup_title_bar(self):
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(4, 0, 0, 0)
        title_layout.setSpacing(0)

        title_label = QLabel(
            'AI Segmentation PRO by '
            '<a href="https://terra-lab.ai" style="color: #1976d2; text-decoration: none;">TerraLab</a>'
        )
        title_label.setOpenExternalLinks(True)
        title_layout.addWidget(title_label)
        title_layout.addStretch()

        icon_size = self.style().pixelMetric(QStyle.PixelMetric.PM_SmallIconSize)

        float_btn = QToolButton()
        float_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarNormalButton))
        float_btn.setFixedSize(icon_size + 4, icon_size + 4)
        float_btn.setAutoRaise(True)
        float_btn.clicked.connect(lambda: self.setFloating(not self.isFloating()))
        title_layout.addWidget(float_btn)

        close_btn = QToolButton()
        close_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton))
        close_btn.setFixedSize(icon_size + 4, icon_size + 4)
        close_btn.setAutoRaise(True)
        close_btn.clicked.connect(self.close)
        title_layout.addWidget(close_btn)

        self.setTitleBarWidget(title_widget)

    def _setup_ui(self):
        # Layer combo
        self.layer_combo = QgsMapLayerComboBox()
        self.layer_combo.setFilters(QgsMapLayerProxyModel.Filter.RasterLayer)
        self.layer_combo.setAllowEmptyLayer(False)
        self.layer_combo.setToolTip(tr("Select a raster layer (GeoTIFF, WMS, XYZ tiles, etc.)"))
        self.main_layout.addWidget(self.layer_combo)

        # No rasters warning
        self.no_rasters_widget = QWidget()
        self.no_rasters_widget.setStyleSheet(
            "QWidget { background-color: rgb(255, 230, 150); "
            "border: 1px solid rgba(255, 152, 0, 0.6); border-radius: 4px; }"
            "QLabel { background: transparent; border: none; color: #333333; }"
        )
        no_rasters_layout = QHBoxLayout(self.no_rasters_widget)
        no_rasters_layout.setContentsMargins(8, 8, 8, 8)
        no_rasters_layout.setSpacing(8)

        warning_icon_label = QLabel()
        style = self.no_rasters_widget.style()
        warning_icon = style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
        warning_icon_label.setPixmap(warning_icon.pixmap(16, 16))
        warning_icon_label.setFixedSize(16, 16)
        no_rasters_layout.addWidget(warning_icon_label, 0, Qt.AlignmentFlag.AlignTop)

        self.no_rasters_label = QLabel(
            tr("No raster layer found. Add a GeoTIFF, image file, "
               "or online layer (WMS, XYZ) to your project."))
        self.no_rasters_label.setWordWrap(True)
        no_rasters_layout.addWidget(self.no_rasters_label, 1)

        self.no_rasters_widget.setVisible(False)
        self.main_layout.addWidget(self.no_rasters_widget)

        # Instructions label
        self.instructions_label = QLabel("")
        self.instructions_label.setWordWrap(True)
        self.instructions_label.setStyleSheet(
            "background-color: rgba(128, 128, 128, 0.08); "
            "border: 1px solid rgba(128, 128, 128, 0.2); "
            "border-radius: 4px; padding: 8px; color: palette(text); font-size: 12px;"
        )
        self.instructions_label.setVisible(False)
        self.main_layout.addWidget(self.instructions_label)

        # Start PRO button
        self.start_pro_button = QPushButton(tr("Start AI Segmentation PRO"))
        self.start_pro_button.setEnabled(False)
        self.start_pro_button.clicked.connect(self._on_start_pro_clicked)
        self.start_pro_button.setStyleSheet(
            "QPushButton { background-color: #2e7d32; padding: 8px 16px; }"
            "QPushButton:disabled { background-color: #c8e6c9; }"
        )
        self.main_layout.addWidget(self.start_pro_button)

        # PRO controls container (visible during segmentation)
        self.pro_controls_container = QWidget()
        pro_ctrl_layout = QVBoxLayout(self.pro_controls_container)
        pro_ctrl_layout.setContentsMargins(0, 4, 0, 4)
        pro_ctrl_layout.setSpacing(6)

        # Tag grid
        pro_tags_widget = QWidget()
        tags_vbox = QVBoxLayout(pro_tags_widget)
        tags_vbox.setContentsMargins(0, 0, 0, 0)
        tags_vbox.setSpacing(4)

        tags_label = QLabel(tr("Select object type:"))
        tags_label.setStyleSheet("font-size: 11px; color: palette(text);")
        tags_vbox.addWidget(tags_label)

        tags_grid_widget = QWidget()
        tags_grid = QGridLayout(tags_grid_widget)
        tags_grid.setContentsMargins(0, 0, 0, 0)
        tags_grid.setSpacing(3)

        _tags = [
            "Roof", "Building", "Warehouse",
            "Solar panel", "Tree", "Bush",
            "Grass", "Forest", "Car",
            "Truck", "Road", "Parking",
            "Railway", "Pool", "River",
            "Field", "Crop", "Shadow",
            "Greenhouse",
        ]
        self._tag_buttons = []
        for i, tag in enumerate(_tags):
            btn = QPushButton(tag)
            btn.setCheckable(True)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.setStyleSheet(
                "QPushButton {"
                "  padding: 3px 8px;"
                "  border-radius: 10px;"
                "  font-size: 11px;"
                "  border: 1px solid rgba(128,128,128,0.35);"
                "  background-color: transparent;"
                "}"
                "QPushButton:checked {"
                "  background-color: #2e7d32;"
                "  color: white;"
                "  border: 1px solid #2e7d32;"
                "}"
                "QPushButton:hover:!checked {"
                "  background-color: rgba(128,128,128,0.12);"
                "}"
            )
            btn.clicked.connect(lambda checked, b=btn: self._on_tag_clicked(b, checked))
            tags_grid.addWidget(btn, i // 4, i % 4)
            self._tag_buttons.append(btn)

        tags_vbox.addWidget(tags_grid_widget)
        pro_ctrl_layout.addWidget(pro_tags_widget)

        self.pro_detect_button = QPushButton(tr("Detect objects"))
        self.pro_detect_button.setVisible(False)
        self.pro_detect_button.clicked.connect(self.pro_detect_requested)
        pro_ctrl_layout.addWidget(self.pro_detect_button)

        # Min confidence (score threshold)
        score_layout = QHBoxLayout()
        score_label = QLabel(tr("Min. confidence"))
        score_label.setStyleSheet("font-size: 12px; color: palette(text);")
        self.score_threshold_spinbox = QSpinBox()
        self.score_threshold_spinbox.setRange(0, 100)
        self.score_threshold_spinbox.setValue(30)
        self.score_threshold_spinbox.setSuffix(" %")
        self.score_threshold_spinbox.setMinimumWidth(80)
        score_layout.addWidget(score_label)
        score_layout.addStretch()
        score_layout.addWidget(self.score_threshold_spinbox)
        pro_ctrl_layout.addLayout(score_layout)

        self.pro_controls_container.setVisible(False)
        self.main_layout.addWidget(self.pro_controls_container)

        # Save polygon button
        self.save_mask_button = QPushButton(tr("Save polygon") + "  (Shortcut: S)")
        self.save_mask_button.clicked.connect(self.save_polygon_requested)
        self.save_mask_button.setVisible(False)
        self.save_mask_button.setEnabled(False)
        self.save_mask_button.setStyleSheet(
            "QPushButton { background-color: #1976d2; padding: 6px 12px; }"
            "QPushButton:disabled { background-color: #b0bec5; }"
        )
        self.save_mask_button.setToolTip(tr("Save current polygon to your session"))
        self.main_layout.addWidget(self.save_mask_button)

        # Export button
        self.export_button = QPushButton(tr("Export polygon to a layer"))
        self.export_button.clicked.connect(self.export_layer_requested)
        self.export_button.setVisible(False)
        self.export_button.setEnabled(False)
        self.export_button.setStyleSheet(
            "QPushButton { background-color: #b0bec5; padding: 6px 12px; }"
        )
        self.main_layout.addWidget(self.export_button)

        # Secondary buttons (undo + stop)
        self.secondary_buttons_widget = QWidget()
        secondary_layout = QHBoxLayout(self.secondary_buttons_widget)
        secondary_layout.setContentsMargins(0, 0, 0, 0)
        secondary_layout.setSpacing(4)

        self.undo_button = QPushButton(tr("Undo last point"))
        self.undo_button.clicked.connect(self.undo_requested)
        self.undo_button.setEnabled(False)
        secondary_layout.addWidget(self.undo_button)

        self.stop_button = QPushButton(tr("Stop segmentation"))
        self.stop_button.clicked.connect(self.stop_segmentation_requested)
        secondary_layout.addWidget(self.stop_button)

        self.secondary_buttons_widget.setVisible(False)
        self.main_layout.addWidget(self.secondary_buttons_widget)

        # Disjoint warning
        self.disjoint_warning_widget = QWidget()
        self.disjoint_warning_widget.setStyleSheet(
            "QWidget { background-color: rgba(255, 180, 50, 0.12); "
            "border: 1px solid rgba(255, 180, 50, 0.4); border-radius: 4px; }"
            "QLabel { background: transparent; border: none; }"
        )
        disjoint_layout = QHBoxLayout(self.disjoint_warning_widget)
        disjoint_layout.setContentsMargins(8, 6, 8, 6)
        disjoint_layout.setSpacing(8)

        disjoint_icon = QLabel()
        warn_icon = style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
        disjoint_icon.setPixmap(warn_icon.pixmap(14, 14))
        disjoint_icon.setFixedSize(14, 14)
        disjoint_layout.addWidget(disjoint_icon, 0, Qt.AlignmentFlag.AlignTop)

        disjoint_msg = "{}\n{}".format(
            tr("Disconnected parts detected in your polygon."),
            tr("For best accuracy, segment one element at a time."))
        disjoint_text = QLabel(disjoint_msg)
        disjoint_text.setWordWrap(True)
        disjoint_text.setStyleSheet("font-size: 11px; color: palette(text);")
        disjoint_layout.addWidget(disjoint_text, 1)

        self.disjoint_warning_widget.setVisible(False)
        self.main_layout.addWidget(self.disjoint_warning_widget)

        self.main_layout.addStretch()

    # ── Public interface (mirrors AISegmentationDockWidget for shared callsites) ──

    def set_segmentation_active(self, active: bool):
        self._segmentation_active = active
        self._update_button_visibility()
        self._update_ui_state()
        if active:
            self._update_instructions()

    def set_point_count(self, positive: int, negative: int):
        self._positive_count = positive
        has_detections = positive > 0
        old_has_mask = self._has_mask
        self._has_mask = has_detections

        self.undo_button.setEnabled(
            (has_detections or self._saved_polygon_count > 0) and self._segmentation_active
        )
        self.save_mask_button.setEnabled(has_detections)

        if self._segmentation_active:
            self._update_instructions()
            self._update_export_button_style()

    def set_mask_available(self, available: bool):
        self._has_mask = available
        self._update_button_visibility()

    def set_saved_polygon_count(self, count: int):
        self._saved_polygon_count = count
        self._update_export_button_style()
        if self._segmentation_active:
            can_undo = self._positive_count > 0 or count > 0
            self.undo_button.setEnabled(can_undo)

    def set_disjoint_warning(self, visible: bool):
        self.disjoint_warning_widget.setVisible(visible)

    def reset_session(self):
        self._has_mask = False
        self._segmentation_active = False
        self._saved_polygon_count = 0
        self._positive_count = 0
        self.disjoint_warning_widget.setVisible(False)
        for btn in self._tag_buttons:
            btn.setChecked(False)
        self.pro_detect_button.setVisible(False)
        self.score_threshold_spinbox.blockSignals(True)
        self.score_threshold_spinbox.setValue(30)
        self.score_threshold_spinbox.blockSignals(False)
        self._update_button_visibility()
        self._update_ui_state()

    def get_score_threshold(self) -> float:
        return self.score_threshold_spinbox.value() / 100.0

    def get_pro_text_prompt(self) -> str:
        for btn in self._tag_buttons:
            if btn.isChecked():
                return btn.text().lower()
        return ""

    def is_activated(self) -> bool:
        return is_plugin_activated()

    def cleanup_signals(self):
        try:
            QgsProject.instance().layersAdded.disconnect(self._on_layers_changed)
        except (TypeError, RuntimeError):
            pass
        try:
            QgsProject.instance().layersRemoved.disconnect(self._on_layers_changed)
        except (TypeError, RuntimeError):
            pass

    # ── Private ──────────────────────────────────────────────────────────────

    def _update_button_visibility(self):
        if self._segmentation_active:
            self.start_pro_button.hide()
            self.instructions_label.show()
            self.pro_controls_container.show()
            self.save_mask_button.show()
            self.save_mask_button.setEnabled(self._has_mask)
            self.export_button.show()
            self.secondary_buttons_widget.show()
        else:
            self.start_pro_button.show()
            self.instructions_label.hide()
            self.pro_controls_container.hide()
            self.save_mask_button.hide()
            self.export_button.hide()
            self.secondary_buttons_widget.hide()

    def _update_export_button_style(self):
        count = self._saved_polygon_count
        if count > 1:
            self.export_button.setText(
                tr("Export {count} polygons to a layer").format(count=count)
            )
        else:
            self.export_button.setText(tr("Export polygon to a layer"))

        if count > 0:
            self.export_button.setEnabled(True)
            self.export_button.setStyleSheet(
                "QPushButton { background-color: #4CAF50; padding: 6px 12px; }"
            )
        else:
            self.export_button.setEnabled(False)
            self.export_button.setStyleSheet(
                "QPushButton { background-color: #b0bec5; padding: 6px 12px; }"
            )

    def _update_instructions(self):
        if self._positive_count == 0:
            text = tr("Click on the area to detect and segment objects")
        else:
            text = tr("Click another area to detect more objects")
        self.instructions_label.setText(text)

    def _update_ui_state(self):
        has_rasters = self.layer_combo.count() > 0
        self.no_rasters_widget.setVisible(not has_rasters and not self._segmentation_active)
        self.layer_combo.setVisible(has_rasters)

        layer = self.layer_combo.currentLayer()
        has_layer = layer is not None
        activated = is_plugin_activated()
        self.start_pro_button.setEnabled(
            has_layer and activated and not self._segmentation_active
        )

    def _on_tag_clicked(self, clicked_btn: QPushButton, checked: bool):
        for btn in self._tag_buttons:
            if btn is not clicked_btn:
                btn.setChecked(False)
        self.pro_detect_button.setVisible(checked)

    def _on_layers_changed(self, *args):
        self._update_ui_state()

    def _on_start_pro_clicked(self):
        from qgis.PyQt.QtWidgets import QMessageBox

        env_path = pathlib.Path(__file__).parent.parent.parent / ".env"
        api_key = ""
        if env_path.exists():
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("PRO_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

        if not api_key:
            QMessageBox.warning(
                self,
                tr("PRO API Key Missing"),
                tr(
                    "PRO API key is not configured.\n\n"
                    "Create the file .env at the root of the plugin directory\n"
                    "with the content:\n"
                    "PRO_API_KEY=your_key_here"
                ),
            )
            return

        layer = self.layer_combo.currentLayer()
        if layer:
            self.start_pro_segmentation_requested.emit(layer)
