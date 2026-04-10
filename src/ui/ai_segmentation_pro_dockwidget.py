from qgis.core import QgsMapLayerProxyModel
from qgis.gui import QgsMapLayerComboBox
from qgis.PyQt.QtCore import QStringListModel, Qt, QTimer, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QCompleter,
    QDockWidget,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.activation_manager import is_plugin_activated  # noqa: E402
from ..core.i18n import tr  # noqa: E402

_REFINE_COLLAPSED_HEIGHT = 25

_KNOWN_OBJECTS = [
    "Roof",
    "Building",
    "Warehouse",
    "Solar panel",
    "Car",
    "Truck",
    "Road",
    "Parking",
    "Railway",
    "Greenhouse",
    "Tree",
    "Bush",
    "Grass",
    "Forest",
    "Crop",
    "Field",
    "River",
    "Pool",
    "Shadow",
]


class AISegmentationProDockWidget(QDockWidget):
    start_pro_segmentation_requested = pyqtSignal(object)  # layer
    export_layer_requested = pyqtSignal()
    undo_requested = pyqtSignal()
    stop_segmentation_requested = pyqtSignal()
    pro_detect_requested = pyqtSignal()
    clear_points_requested = pyqtSignal()
    refine_settings_changed = pyqtSignal(
        int, int, bool, int
    )  # expand, simplify, fill_holes, min_area
    zone_select_requested = pyqtSignal()  # User wants to draw a zone
    zone_clear_requested = pyqtSignal()  # User wants to reset to full image

    def __init__(self, parent=None):
        super().__init__(tr("AI Segmentation PRO by TerraLab"), parent)

        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )

        self._setup_title_bar()

        self._segmentation_active = False
        self._has_mask = False
        self._saved_polygon_count = 0
        self._positive_count = 0  # batch detection count in PRO mode
        self._refine_expanded = False

        # Debounce timer for refinement sliders
        self._refine_debounce_timer = QTimer(self)
        self._refine_debounce_timer.setSingleShot(True)
        self._refine_debounce_timer.timeout.connect(self._emit_refine_changed)

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

        self._update_ui_state()

    def _setup_title_bar(self):
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(4, 0, 0, 0)
        title_layout.setSpacing(0)

        title_label = QLabel(
            "AI Segmentation PRO by "
            '<a href="https://terra-lab.ai" style="color: #1976d2; text-decoration: none;">TerraLab</a>'
        )
        title_label.setOpenExternalLinks(True)
        title_layout.addWidget(title_label)
        title_layout.addStretch()

        icon_size = self.style().pixelMetric(QStyle.PixelMetric.PM_SmallIconSize)

        float_btn = QToolButton()
        float_btn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarNormalButton)
        )
        float_btn.setFixedSize(icon_size + 4, icon_size + 4)
        float_btn.setAutoRaise(True)
        float_btn.clicked.connect(lambda: self.setFloating(not self.isFloating()))
        title_layout.addWidget(float_btn)

        close_btn = QToolButton()
        close_btn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton)
        )
        close_btn.setFixedSize(icon_size + 4, icon_size + 4)
        close_btn.setAutoRaise(True)
        close_btn.clicked.connect(self.close)
        title_layout.addWidget(close_btn)

        self.setTitleBarWidget(title_widget)

    def _setup_ui(self):
        _neutral_btn = (
            "QPushButton { background-color: rgba(128, 128, 128, 0.12);"
            " padding: 6px 12px; border: 1px solid rgba(128, 128, 128, 0.2);"
            " border-radius: 4px; }"
        )
        self._zone_selected_btn = (
            "QPushButton { background-color: rgba(66, 133, 244, 0.25);"
            " padding: 6px 12px; border: 1px solid rgba(66, 133, 244, 0.6);"
            " border-radius: 4px; color: palette(text); }"
        )
        self._zone_neutral_btn = _neutral_btn

        # Layer combo
        self.layer_combo = QgsMapLayerComboBox()
        self.layer_combo.setFilters(QgsMapLayerProxyModel.Filter.RasterLayer)
        self.layer_combo.setAllowEmptyLayer(False)
        self.layer_combo.setToolTip(
            tr("Select a raster layer (GeoTIFF, WMS, XYZ tiles, etc.)")
        )
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
            tr(
                "No raster layer found. Add a GeoTIFF, image file, "
                "or online layer (WMS, XYZ) to your project."
            )
        )
        self.no_rasters_label.setWordWrap(True)
        no_rasters_layout.addWidget(self.no_rasters_label, 1)

        self.no_rasters_widget.setVisible(False)
        self.main_layout.addWidget(self.no_rasters_widget)

        # Zone selection row — both always visible, "Full image" selected by default
        zone_row = QHBoxLayout()
        self.zone_select_btn = QPushButton(tr("Select zone"))
        self.zone_select_btn.setToolTip(
            tr("Draw a rectangle to limit the segmentation area")
        )
        self.zone_select_btn.setStyleSheet(_neutral_btn)
        self.zone_select_btn.clicked.connect(self.zone_select_requested.emit)

        self.zone_clear_btn = QPushButton(tr("Full image"))
        self.zone_clear_btn.setToolTip(tr("Use the entire image"))
        self.zone_clear_btn.setStyleSheet(self._zone_selected_btn)
        self.zone_clear_btn.clicked.connect(self.zone_clear_requested.emit)

        zone_row.addWidget(self.zone_select_btn)
        zone_row.addWidget(self.zone_clear_btn)
        self.main_layout.addLayout(zone_row)

        # Autocomplete search field (replaces category + tag combos)
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText(tr("Search object to detect..."))
        self._search_edit.setClearButtonEnabled(True)

        completer = QCompleter()
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        model = QStringListModel(_KNOWN_OBJECTS)
        completer.setModel(model)
        self._search_edit.setCompleter(completer)
        self._search_edit.textChanged.connect(self._on_search_text_changed)
        self.main_layout.addWidget(self._search_edit)

        # Detect button (only green element — includes credit count)
        self._credit_count = 0
        self.pro_detect_button = QPushButton(tr("Detect objects"))
        self.pro_detect_button.setEnabled(False)
        self.pro_detect_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; padding: 8px 16px;"
            " border-radius: 4px; }"
            "QPushButton:disabled { background-color: rgba(128, 128, 128, 0.12); }"
        )
        self.pro_detect_button.clicked.connect(self.pro_detect_requested)
        self.main_layout.addWidget(self.pro_detect_button)

        # Min confidence (score threshold) — default 25%
        score_layout = QHBoxLayout()
        score_label = QLabel(tr("Min. confidence"))
        score_label.setStyleSheet("font-size: 12px; color: palette(text);")
        self.score_threshold_spinbox = QSpinBox()
        self.score_threshold_spinbox.setRange(0, 100)
        self.score_threshold_spinbox.setValue(25)
        self.score_threshold_spinbox.setSuffix(" %")
        self.score_threshold_spinbox.setMinimumWidth(80)
        score_layout.addWidget(score_label)
        score_layout.addStretch()
        score_layout.addWidget(self.score_threshold_spinbox)
        self.main_layout.addLayout(score_layout)

        # Credit info hint
        self.credit_info_label = QLabel(
            tr("The larger the zone, the more credits are used.")
        )
        self.credit_info_label.setWordWrap(True)
        self.credit_info_label.setStyleSheet("color: palette(text); font-size: 11px;")
        self.main_layout.addWidget(self.credit_info_label)

        # Post-detection status banner
        self.status_banner = QLabel("")
        self.status_banner.setWordWrap(True)
        self.status_banner.setStyleSheet(
            "background-color: rgba(128, 128, 128, 0.08); "
            "border: 1px solid rgba(128, 128, 128, 0.2); "
            "border-radius: 4px; padding: 8px; color: palette(text); font-size: 12px;"
        )
        self.status_banner.setVisible(False)
        self.main_layout.addWidget(self.status_banner)

        # Refine polygons panel (collapsed, post-detection only)
        self._setup_refine_panel(self.main_layout)

        # Export button (visible when polygons exist)
        self.export_button = QPushButton(tr("Export polygon to a layer"))
        self.export_button.clicked.connect(self.export_layer_requested)
        self.export_button.setVisible(False)
        self.export_button.setEnabled(False)
        self.export_button.setStyleSheet(
            "QPushButton { background-color: rgba(128, 128, 128, 0.12);"
            " padding: 6px 12px; border-radius: 4px; }"
        )
        self.main_layout.addWidget(self.export_button)

        # Secondary buttons (undo + stop) — neutral style
        self.secondary_buttons_widget = QWidget()
        secondary_layout = QHBoxLayout(self.secondary_buttons_widget)
        secondary_layout.setContentsMargins(0, 0, 0, 0)
        secondary_layout.setSpacing(4)

        self.undo_button = QPushButton(tr("Undo"))
        self.undo_button.setStyleSheet(_neutral_btn)
        self.undo_button.clicked.connect(self.undo_requested)
        self.undo_button.setEnabled(False)
        secondary_layout.addWidget(self.undo_button)

        self.stop_button = QPushButton(tr("Stop"))
        self.stop_button.setStyleSheet(_neutral_btn)
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
            tr("For best accuracy, segment one element at a time."),
        )
        disjoint_text = QLabel(disjoint_msg)
        disjoint_text.setWordWrap(True)
        disjoint_text.setStyleSheet("font-size: 11px; color: palette(text);")
        disjoint_layout.addWidget(disjoint_text, 1)

        self.disjoint_warning_widget.setVisible(False)
        self.main_layout.addWidget(self.disjoint_warning_widget)

        # Tile progress bar (hidden by default)
        self.tile_progress = QProgressBar()
        self.tile_progress.setVisible(False)
        self.tile_progress.setTextVisible(True)
        self.tile_progress.setFormat("Tile %v/%m")
        self.main_layout.addWidget(self.tile_progress)

        # Debug: tile grid overlay toggle
        self.tile_grid_button = QPushButton(tr("Show tile grid"))
        self.tile_grid_button.setStyleSheet(_neutral_btn)
        self.main_layout.addWidget(self.tile_grid_button)

        self.main_layout.addStretch()

    # ── Public interface (mirrors AISegmentationDockWidget for shared callsites) ──

    def set_segmentation_active(self, active: bool):
        self._segmentation_active = active
        self._update_button_visibility()
        self._update_ui_state()

    def set_point_count(self, positive: int, negative: int):
        self._positive_count = positive
        has_detections = positive > 0
        self._has_mask = has_detections

        self.undo_button.setEnabled(
            (has_detections or self._saved_polygon_count > 0)
            and self._segmentation_active
        )

        if self._segmentation_active:
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
        self._credit_count = 0
        self.disjoint_warning_widget.setVisible(False)
        self._search_edit.clear()
        self.score_threshold_spinbox.blockSignals(True)
        self.score_threshold_spinbox.setValue(25)
        self.score_threshold_spinbox.blockSignals(False)
        self.reset_refine_sliders()
        self._update_detect_button_label()
        self._update_button_visibility()
        self._update_ui_state()

    def get_score_threshold(self) -> float:
        return self.score_threshold_spinbox.value() / 100.0

    def get_max_objects(self) -> int:
        """Max objects is no longer exposed in the UI — always return 32."""
        return 32

    def get_pro_text_prompt(self) -> str:
        return self._search_edit.text().strip().lower()

    def is_activated(self) -> bool:
        return is_plugin_activated()

    # ── Private ──────────────────────────────────────────────────────────────

    def _on_search_text_changed(self, text: str):
        has_text = bool(text.strip())
        activated = is_plugin_activated()
        has_layer = self.layer_combo.currentLayer() is not None
        self.pro_detect_button.setEnabled(has_text and activated and has_layer)

    def _update_detect_button_label(self):
        if self._credit_count > 0:
            self.pro_detect_button.setText(
                tr("Detect objects ({credits} credits)").format(
                    credits=self._credit_count
                )
            )
        else:
            self.pro_detect_button.setText(tr("Detect objects"))

    def _update_button_visibility(self):
        if self._segmentation_active:
            self.status_banner.setVisible(self.status_banner.text() != "")
            has_results = self._saved_polygon_count > 0 or self._positive_count > 0
            self.export_button.setVisible(has_results)
            self.secondary_buttons_widget.show()
            self.refine_group.setVisible(has_results)
        else:
            self.status_banner.hide()
            self.export_button.hide()
            self.secondary_buttons_widget.hide()
            self.refine_group.hide()

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
            self.export_button.setVisible(True)
            self.export_button.setStyleSheet(
                "QPushButton { background-color: #66BB6A; padding: 6px 12px;"
                " border-radius: 4px; }"
            )
        else:
            self.export_button.setEnabled(False)
            self.export_button.setVisible(False)

    def _update_ui_state(self):
        has_rasters = self.layer_combo.count() > 0
        self.no_rasters_widget.setVisible(
            not has_rasters and not self._segmentation_active
        )
        self.layer_combo.setVisible(has_rasters)

        # Update detect button enabled state
        has_text = bool(self._search_edit.text().strip())
        activated = is_plugin_activated()
        has_layer = self.layer_combo.currentLayer() is not None
        self.pro_detect_button.setEnabled(has_text and activated and has_layer)

    def set_reference_pending(self, prompt: str):
        self.status_banner.setText(
            tr("Click on one {prompt} as reference, then click Detect").format(
                prompt=prompt
            )
        )
        self.status_banner.setVisible(True)

    def set_reference_set(self, prompt: str):
        self.status_banner.setText(
            tr("Reference set. Click Detect to find all {prompt}.").format(
                prompt=prompt
            )
        )
        self.status_banner.setVisible(True)

    def set_batch_done(self, count: int):
        self.status_banner.setText(
            "{}\n{}".format(
                tr("{count} objects detected").format(count=count),
                tr("Right-click a polygon to remove it"),
            )
        )
        self.status_banner.setVisible(True)
        # Show refine and export after first detection
        self.refine_group.setVisible(True)
        self._update_export_button_style()

    def _on_layers_changed(self, *args):
        self._update_ui_state()

    def _setup_refine_panel(self, parent_layout):
        """Setup the collapsible Refine mask panel."""
        self.refine_group = QGroupBox("▶ " + tr("Refine polygons"))
        self.refine_group.setCheckable(False)
        self.refine_group.setVisible(False)
        self.refine_group.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refine_group.mousePressEvent = self._on_refine_group_clicked
        self.refine_group.setStyleSheet("""
            QGroupBox {
                background-color: transparent;
                border: none;
                border-radius: 0px;
                margin: 0px;
                padding: 0px;
                padding-top: 20px;
            }
            QGroupBox::title {
                subcontrol-origin: padding;
                subcontrol-position: top left;
                padding: 2px 4px;
                background-color: transparent;
                border: none;
            }
        """)
        refine_layout = QVBoxLayout(self.refine_group)
        refine_layout.setSpacing(0)
        refine_layout.setContentsMargins(0, 0, 0, 0)

        self.refine_content_widget = QWidget()
        self.refine_content_widget.setObjectName("refineContentWidget")
        self.refine_content_widget.setStyleSheet("""
            QWidget#refineContentWidget {
                background-color: rgba(128, 128, 128, 0.08);
                border: 1px solid rgba(128, 128, 128, 0.2);
                border-radius: 4px;
            }
            QLabel {
                background: transparent;
                border: none;
            }
        """)
        refine_content_layout = QVBoxLayout(self.refine_content_widget)
        refine_content_layout.setContentsMargins(10, 10, 10, 10)
        refine_content_layout.setSpacing(8)

        # Expand/Contract
        expand_layout = QHBoxLayout()
        expand_label = QLabel(tr("Expand/Contract:"))
        expand_label.setToolTip(
            tr("Positive = expand outward, Negative = shrink inward")
        )
        self.expand_spinbox = QSpinBox()
        self.expand_spinbox.setRange(-1000, 1000)
        self.expand_spinbox.setValue(0)
        self.expand_spinbox.setSuffix(" px")
        self.expand_spinbox.setMinimumWidth(80)
        expand_layout.addWidget(expand_label)
        expand_layout.addStretch()
        expand_layout.addWidget(self.expand_spinbox)
        refine_content_layout.addLayout(expand_layout)

        # Simplify outline
        simplify_layout = QHBoxLayout()
        simplify_label = QLabel(tr("Simplify outline:"))
        simplify_label.setToolTip(
            tr("Reduce small variations in the outline (0 = no change)")
        )
        self.simplify_spinbox = QSpinBox()
        self.simplify_spinbox.setRange(0, 1000)
        self.simplify_spinbox.setValue(3)
        self.simplify_spinbox.setMinimumWidth(80)
        simplify_layout.addWidget(simplify_label)
        simplify_layout.addStretch()
        simplify_layout.addWidget(self.simplify_spinbox)
        refine_content_layout.addLayout(simplify_layout)

        # Fill holes
        fill_holes_layout = QHBoxLayout()
        self.fill_holes_checkbox = QCheckBox(tr("Fill holes"))
        self.fill_holes_checkbox.setChecked(False)
        self.fill_holes_checkbox.setToolTip(tr("Fill interior holes in the selection"))
        fill_holes_layout.addWidget(self.fill_holes_checkbox)
        fill_holes_layout.addStretch()
        refine_content_layout.addLayout(fill_holes_layout)

        # Min. region size
        min_area_layout = QHBoxLayout()
        min_area_label = QLabel(tr("Min. region size:"))
        min_area_label.setToolTip(
            "{}\n{}\n{}".format(
                tr("Remove disconnected regions smaller than this area (in pixels²)."),
                tr("Example: 100 = ~10x10 pixel regions, 900 = ~30x30."),
                tr("0 = keep all."),
            )
        )
        self.min_area_spinbox = QSpinBox()
        self.min_area_spinbox.setRange(0, 100000)
        self.min_area_spinbox.setValue(100)
        self.min_area_spinbox.setSuffix(" px²")
        self.min_area_spinbox.setSingleStep(50)
        self.min_area_spinbox.setMinimumWidth(80)
        min_area_layout.addWidget(min_area_label)
        min_area_layout.addStretch()
        min_area_layout.addWidget(self.min_area_spinbox)
        refine_content_layout.addLayout(min_area_layout)

        refine_layout.addWidget(self.refine_content_widget)
        self.refine_content_widget.setVisible(self._refine_expanded)
        if not self._refine_expanded:
            self.refine_group.setMaximumHeight(_REFINE_COLLAPSED_HEIGHT)

        self.expand_spinbox.valueChanged.connect(self._on_refine_changed)
        self.simplify_spinbox.valueChanged.connect(self._on_refine_changed)
        self.fill_holes_checkbox.stateChanged.connect(self._on_refine_changed)
        self.min_area_spinbox.valueChanged.connect(self._on_refine_changed)

        parent_layout.addWidget(self.refine_group)

    def _on_refine_group_clicked(self, event):
        """Toggle the refine panel expanded/collapsed state."""
        if event.pos().y() > _REFINE_COLLAPSED_HEIGHT:
            return
        self._refine_expanded = not self._refine_expanded
        self.refine_content_widget.setVisible(self._refine_expanded)
        arrow = "▼" if self._refine_expanded else "▶"
        self.refine_group.setTitle(f"{arrow} " + tr("Refine polygons"))
        if self._refine_expanded:
            self.refine_group.setMaximumHeight(16777215)
        else:
            self.refine_group.setMaximumHeight(_REFINE_COLLAPSED_HEIGHT)

    def _on_refine_changed(self, value=None):
        """Handle refine control changes with debounce."""
        self._refine_debounce_timer.start(150)

    def _emit_refine_changed(self):
        """Emit the refine settings changed signal after debounce."""
        self.refine_settings_changed.emit(
            self.expand_spinbox.value(),
            self.simplify_spinbox.value(),
            self.fill_holes_checkbox.isChecked(),
            self.min_area_spinbox.value(),
        )

    def reset_refine_sliders(self):
        """Reset refinement controls to default values without emitting signals."""
        self.expand_spinbox.blockSignals(True)
        self.simplify_spinbox.blockSignals(True)
        self.fill_holes_checkbox.blockSignals(True)
        self.min_area_spinbox.blockSignals(True)

        self.expand_spinbox.setValue(0)
        self.simplify_spinbox.setValue(3)
        self.fill_holes_checkbox.setChecked(False)
        self.min_area_spinbox.setValue(100)

        self.expand_spinbox.blockSignals(False)
        self.simplify_spinbox.blockSignals(False)
        self.fill_holes_checkbox.blockSignals(False)
        self.min_area_spinbox.blockSignals(False)

    def set_credit_estimate(self, credits: int):
        """Update the credit count shown in the Detect button."""
        if credits < 0:
            self._credit_count = 0
            self.credit_info_label.setText(
                tr("Zone too large. Please reduce the selection.")
            )
            self.credit_info_label.setStyleSheet("color: #f44336; font-size: 11px;")
        else:
            self._credit_count = credits
            self.credit_info_label.setText(
                tr("The larger the zone, the more credits are used.")
            )
            self.credit_info_label.setStyleSheet(
                "color: palette(text); font-size: 11px;"
            )
        self._update_detect_button_label()

    def set_zone_active(self, active: bool):
        """Toggle zone selection UI state and highlight the active choice."""
        if active:
            self.zone_select_btn.setText(tr("Redraw zone"))
            self.zone_select_btn.setStyleSheet(self._zone_selected_btn)
            self.zone_clear_btn.setStyleSheet(self._zone_neutral_btn)
        else:
            self.zone_select_btn.setText(tr("Select zone"))
            self.zone_select_btn.setStyleSheet(self._zone_neutral_btn)
            self.zone_clear_btn.setStyleSheet(self._zone_selected_btn)

    def set_tile_progress(self, current: int, total: int):
        """Update tile progress bar."""
        self.tile_progress.setMaximum(total)
        self.tile_progress.setValue(current)
        self.tile_progress.setVisible(total > 1)

    def hide_tile_progress(self):
        """Hide the tile progress bar."""
        self.tile_progress.setVisible(False)
