import os
from pathlib import Path
from typing import Optional, List, Tuple
import sys

from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal

from qgis.core import (
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsMessageLog,
    Qgis,
    QgsWkbTypes,
    QgsGeometry,
    QgsFeature,
    QgsFillSymbol,
    QgsSingleSymbolRenderer,
    QgsRectangle,
    QgsCoordinateReferenceSystem,
)
from qgis.gui import QgisInterface, QgsRubberBand
from qgis.PyQt.QtGui import QColor

from .ai_segmentation_dockwidget import AISegmentationDockWidget
from .ai_segmentation_maptool import AISegmentationMapTool




class DownloadWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        try:
            from .core.checkpoint_manager import download_checkpoint
            success, message = download_checkpoint(
                progress_callback=lambda p, m: self.progress.emit(p, m)
            )
            self.finished.emit(success, message)
        except Exception as e:
            self.finished.emit(False, str(e))


class EncodingWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, raster_path: str, output_dir: str, checkpoint_path: str, layer_crs_wkt: str = None, parent=None):
        super().__init__(parent)
        self.raster_path = raster_path
        self.output_dir = output_dir
        self.checkpoint_path = checkpoint_path
        self.layer_crs_wkt = layer_crs_wkt
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from .core.feature_encoder import encode_raster_to_features
            success, message = encode_raster_to_features(
                self.raster_path,
                self.output_dir,
                self.checkpoint_path,
                layer_crs_wkt=self.layer_crs_wkt,
                progress_callback=lambda p, m: self.progress.emit(p, m),
                cancel_check=lambda: self._cancelled,
            )
            self.finished.emit(success, message)
        except Exception as e:
            import traceback
            QgsMessageLog.logMessage(
                f"Encoding error: {str(e)}\n{traceback.format_exc()}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            self.finished.emit(False, str(e))


class PromptManager:
    def __init__(self):
        self.positive_points: List[Tuple[float, float]] = []
        self.negative_points: List[Tuple[float, float]] = []
        self._history: List[Tuple[str, Tuple[float, float]]] = []

    def add_positive_point(self, x: float, y: float):
        self.positive_points.append((x, y))
        self._history.append(("positive", (x, y)))

    def add_negative_point(self, x: float, y: float):
        self.negative_points.append((x, y))
        self._history.append(("negative", (x, y)))

    def undo(self) -> Optional[Tuple[str, Tuple[float, float]]]:
        if not self._history:
            return None

        label, point = self._history.pop()
        if label == "positive" and point in self.positive_points:
            self.positive_points.remove(point)
        elif label == "negative" and point in self.negative_points:
            self.negative_points.remove(point)

        return label, point

    def clear(self):
        self.positive_points = []
        self.negative_points = []
        self._history = []

    @property
    def point_count(self) -> Tuple[int, int]:
        return len(self.positive_points), len(self.negative_points)

    def get_points_for_predictor(self, transform) -> Tuple[Optional['np.ndarray'], Optional['np.ndarray']]:
        import numpy as np
        from rasterio import transform as rio_transform

        all_points = self.positive_points + self.negative_points
        if not all_points:
            return None, None

        point_coords = []
        point_labels = []

        for x, y in self.positive_points:
            col, row = rio_transform.rowcol(transform, x, y)
            point_coords.append([row, col])
            point_labels.append(1)

        for x, y in self.negative_points:
            col, row = rio_transform.rowcol(transform, x, y)
            point_coords.append([row, col])
            point_labels.append(0)

        return np.array(point_coords), np.array(point_labels)


class AISegmentationPlugin:

    def __init__(self, iface: QgisInterface):
        self.iface = iface
        self.plugin_dir = Path(__file__).parent

        self.dock_widget: Optional[AISegmentationDockWidget] = None
        self.map_tool: Optional[AISegmentationMapTool] = None
        self.action: Optional[QAction] = None

        self.predictor = None
        self.feature_dataset = None
        self.prompts = PromptManager()

        self.current_mask = None
        self.current_score = 0.0
        self.current_transform_info = None
        self.saved_polygons = []

        self._initialized = False
        self._current_layer = None
        self._current_layer_name = ""
        self._segmentation_counter = 0

        self.download_worker = None
        self.encoding_worker = None

        self.mask_rubber_band: Optional[QgsRubberBand] = None
        self.saved_rubber_bands: List[QgsRubberBand] = []

    def initGui(self):
        icon_path = str(self.plugin_dir / "resources" / "icons" / "ai_segmentation_icon.svg")
        if not os.path.exists(icon_path):
            icon = QIcon()
        else:
            icon = QIcon(icon_path)

        self.action = QAction(
            icon,
            "AI Segmentation",
            self.iface.mainWindow()
        )
        self.action.setCheckable(True)
        self.action.triggered.connect(self.toggle_dock_widget)

        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&AI Segmentation", self.action)

        self.dock_widget = AISegmentationDockWidget(self.iface.mainWindow())
        self.dock_widget.setVisible(False)
        self.dock_widget.visibilityChanged.connect(self._on_dock_visibility_changed)

        self.dock_widget.install_dependencies_requested.connect(self._on_install_requested)
        self.dock_widget.download_checkpoint_requested.connect(self._on_download_checkpoint_requested)
        self.dock_widget.cancel_download_requested.connect(self._on_cancel_download)
        self.dock_widget.cancel_preparation_requested.connect(self._on_cancel_preparation)
        self.dock_widget.start_segmentation_requested.connect(self._on_start_segmentation)
        self.dock_widget.save_polygon_requested.connect(self._on_save_polygon)
        self.dock_widget.export_layer_requested.connect(self._on_export_layer)
        self.dock_widget.clear_points_requested.connect(self._on_clear_points)
        self.dock_widget.undo_requested.connect(self._on_undo)

        self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock_widget)

        self.map_tool = AISegmentationMapTool(self.iface.mapCanvas())
        self.map_tool.positive_click.connect(self._on_positive_click)
        self.map_tool.negative_click.connect(self._on_negative_click)
        self.map_tool.tool_deactivated.connect(self._on_tool_deactivated)
        self.map_tool.undo_requested.connect(self._on_undo)
        self.map_tool.save_polygon_requested.connect(self._on_save_polygon)

        self.mask_rubber_band = QgsRubberBand(
            self.iface.mapCanvas(),
            QgsWkbTypes.PolygonGeometry
        )
        self.mask_rubber_band.setColor(QColor(0, 120, 255, 100))
        self.mask_rubber_band.setStrokeColor(QColor(0, 80, 200))
        self.mask_rubber_band.setWidth(2)

        QgsMessageLog.logMessage(
            "AI Segmentation plugin loaded (checks deferred until panel opens)",
            "AI Segmentation",
            level=Qgis.Info
        )

    def unload(self):
        self.iface.removePluginMenu("&AI Segmentation", self.action)
        self.iface.removeToolBarIcon(self.action)

        if self.dock_widget:
            self.iface.removeDockWidget(self.dock_widget)
            self.dock_widget.deleteLater()
            self.dock_widget = None

        if self.map_tool:
            if self.iface.mapCanvas().mapTool() == self.map_tool:
                self.iface.mapCanvas().unsetMapTool(self.map_tool)
            self.map_tool = None

        if self.mask_rubber_band:
            self.iface.mapCanvas().scene().removeItem(self.mask_rubber_band)
            self.mask_rubber_band = None

        for rb in self.saved_rubber_bands:
            self.iface.mapCanvas().scene().removeItem(rb)
        self.saved_rubber_bands = []

        for worker in [self.download_worker, self.encoding_worker]:
            if worker and worker.isRunning():
                worker.terminate()
                worker.wait()

    def toggle_dock_widget(self, checked: bool):
        if self.dock_widget:
            self.dock_widget.setVisible(checked)

    def _on_dock_visibility_changed(self, visible: bool):
        if self.action:
            self.action.setChecked(visible)

        if visible and not self._initialized:
            self._initialized = True
            self._do_first_time_setup()

    def _do_first_time_setup(self):
        QgsMessageLog.logMessage(
            "Panel opened - checking dependencies...",
            "AI Segmentation",
            level=Qgis.Info
        )

        try:
            from .core.dependency_manager import all_dependencies_installed, get_missing_dependencies

            if all_dependencies_installed():
                self.dock_widget.set_dependency_status(True, "Dependencies OK")
                self._check_checkpoint()
            else:
                missing = get_missing_dependencies()
                missing_str = ", ".join([f"{name}" for name, _ in missing])
                self.dock_widget.set_dependency_status(
                    False,
                    f"Missing: {missing_str}"
                )
                self.dock_widget.set_status("Click 'Install Dependencies' to continue")

        except Exception as e:
            QgsMessageLog.logMessage(
                f"Dependency check error: {str(e)}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            self.dock_widget.set_dependency_status(False, f"Error: {str(e)[:50]}")

    def _check_checkpoint(self):
        try:
            from .core.checkpoint_manager import checkpoint_exists

            if checkpoint_exists():
                self.dock_widget.set_checkpoint_status(True, "SAM model ready")
                self._load_predictor()
                self.dock_widget.set_status("Ready - select a raster layer and start!")
            else:
                self.dock_widget.set_checkpoint_status(False, "Model not downloaded")
                self.dock_widget.set_status("Download the AI model to get started")

        except Exception as e:
            QgsMessageLog.logMessage(
                f"Checkpoint check error: {str(e)}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            self.dock_widget.set_checkpoint_status(False, f"Error: {str(e)[:50]}")

    def _load_predictor(self):
        try:
            from .core.checkpoint_manager import get_checkpoint_path
            from .core.sam_predictor import build_sam_vit_b_no_encoder, SamPredictorNoImgEncoder

            checkpoint_path = get_checkpoint_path()
            sam = build_sam_vit_b_no_encoder(checkpoint=checkpoint_path)
            self.predictor = SamPredictorNoImgEncoder(sam)

            QgsMessageLog.logMessage(
                "SAM predictor loaded successfully",
                "AI Segmentation",
                level=Qgis.Info
            )

        except Exception as e:
            import traceback
            QgsMessageLog.logMessage(
                f"Failed to load predictor: {str(e)}\n{traceback.format_exc()}",
                "AI Segmentation",
                level=Qgis.Warning
            )

    def _on_install_requested(self):
        from .core.dependency_manager import get_manual_install_instructions, get_missing_dependencies

        missing = get_missing_dependencies()
        missing_str = ", ".join([pip_name for _, pip_name, _ in missing])

        instructions = get_manual_install_instructions()

        QMessageBox.information(
            self.iface.mainWindow(),
            "Install Dependencies",
            f"Missing packages: {missing_str}\n\n"
            f"Please install manually:\n\n{instructions}\n\n"
            "After installation, restart QGIS."
        )

        self.dock_widget.set_status("Install dependencies manually, then restart QGIS")

    def _on_download_checkpoint_requested(self):
        self.dock_widget.set_download_progress(0, "Downloading SAM checkpoint...")

        self.download_worker = DownloadWorker()
        self.download_worker.progress.connect(self._on_download_progress)
        self.download_worker.finished.connect(self._on_download_finished)
        self.download_worker.start()

    def _on_download_progress(self, percent: int, message: str):
        self.dock_widget.set_download_progress(percent, message)

    def _on_download_finished(self, success: bool, message: str):
        if success:
            self.dock_widget.set_download_progress(100, "Download complete!")
            self.dock_widget.set_checkpoint_status(True, "SAM model ready")
            self._load_predictor()
            self.dock_widget.set_status("Ready - select a raster layer and start!")
        else:
            self.dock_widget.set_download_progress(0, "")
            self.dock_widget.set_status(f"Error: {message[:50]}")

            QMessageBox.warning(
                self.iface.mainWindow(),
                "Download Failed",
                f"Failed to download model:\n{message}\n\n"
                "Please check your internet connection and try again."
            )

    def _on_cancel_download(self):
        if self.download_worker and self.download_worker.isRunning():
            self.download_worker.terminate()
            self.download_worker.wait()
            self.dock_widget.set_download_progress(0, "Cancelled")
            self.dock_widget.set_status("Download cancelled by user")

    def _on_cancel_preparation(self):
        if self.encoding_worker and self.encoding_worker.isRunning():
            self.encoding_worker.cancel()
            self.dock_widget.set_preparation_progress(0, "Cancelled")
            self.dock_widget.set_status("Encoding cancelled by user")

    def _on_start_segmentation(self, layer: QgsRasterLayer):
        if self.predictor is None:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Not Ready",
                "Please wait for the SAM model to load."
            )
            return

        self._reset_session()

        self._current_layer = layer
        self._current_layer_name = layer.name().replace(" ", "_")
        raster_path = layer.source()

        from .core.checkpoint_manager import has_features_for_raster, get_raster_features_dir, get_checkpoint_path

        if has_features_for_raster(raster_path):
            self._load_features_and_activate(raster_path)
        else:
            output_dir = get_raster_features_dir(raster_path)
            checkpoint_path = get_checkpoint_path()

            layer_crs_wkt = None
            if layer.crs().isValid():
                layer_crs_wkt = layer.crs().toWkt()

            self.dock_widget.set_preparation_progress(0, "Encoding raster (first time)...")
            self.dock_widget.set_status("This may take a few minutes...")

            self.encoding_worker = EncodingWorker(raster_path, output_dir, checkpoint_path, layer_crs_wkt)
            self.encoding_worker.progress.connect(self._on_encoding_progress)
            self.encoding_worker.finished.connect(
                lambda s, m: self._on_encoding_finished(s, m, raster_path)
            )
            self.encoding_worker.start()

    def _on_encoding_progress(self, percent: int, message: str):
        self.dock_widget.set_preparation_progress(percent, message)

    def _on_encoding_finished(self, success: bool, message: str, raster_path: str):
        if success:
            self.dock_widget.set_preparation_progress(100, "Ready!")
            self._load_features_and_activate(raster_path)
        else:
            self.dock_widget.set_preparation_progress(0, "")
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Encoding Failed",
                f"Failed to encode raster:\n{message}"
            )

    def _load_features_and_activate(self, raster_path: str):
        try:
            from .core.checkpoint_manager import get_raster_features_dir
            from .core.feature_dataset import FeatureDataset

            features_dir = get_raster_features_dir(raster_path)
            self.feature_dataset = FeatureDataset(features_dir, cache=True)

            QgsMessageLog.logMessage(
                f"Loaded {len(self.feature_dataset)} feature tiles",
                "AI Segmentation",
                level=Qgis.Info
            )

            self._activate_segmentation_tool()

        except Exception as e:
            import traceback
            QgsMessageLog.logMessage(
                f"Failed to load features: {str(e)}\n{traceback.format_exc()}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Load Failed",
                f"Failed to load feature data:\n{str(e)}"
            )

    def _activate_segmentation_tool(self):
        self.iface.mapCanvas().setMapTool(self.map_tool)
        self.dock_widget.set_segmentation_active(True)
        self.dock_widget.set_status("Click on the map to segment")

    def _on_save_polygon(self):
        if self.current_mask is None:
            self.dock_widget.set_status("No polygon to save")
            return

        from .core.polygon_exporter import mask_to_polygons

        geometries = mask_to_polygons(self.current_mask, self.current_transform_info)

        if not geometries:
            self.dock_widget.set_status("Could not generate polygon")
            return

        combined = QgsGeometry.unaryUnion(geometries)
        if combined and not combined.isEmpty():
            self.saved_polygons.append({
                'geometry': combined,
                'score': self.current_score,
                'transform_info': self.current_transform_info.copy() if self.current_transform_info else None,
            })

            saved_rb = QgsRubberBand(self.iface.mapCanvas(), QgsWkbTypes.PolygonGeometry)
            saved_rb.setColor(QColor(0, 200, 100, 120))
            saved_rb.setFillColor(QColor(0, 200, 100, 80))
            saved_rb.setWidth(2)
            saved_rb.setToGeometry(combined, None)
            self.saved_rubber_bands.append(saved_rb)

            QgsMessageLog.logMessage(
                f"Saved polygon #{len(self.saved_polygons)}",
                "AI Segmentation",
                level=Qgis.Info
            )

            self.dock_widget.set_saved_polygon_count(len(self.saved_polygons))
            self.dock_widget.set_status(f"Polygon saved ({len(self.saved_polygons)} total) - click for next object")

        self.prompts.clear()
        if self.map_tool:
            self.map_tool.clear_markers()
        self._clear_mask_visualization()
        self.current_mask = None
        self.current_score = 0.0
        self.dock_widget.set_point_count(0, 0)

    def _on_export_layer(self):
        from .core.polygon_exporter import mask_to_polygons

        polygons_to_export = list(self.saved_polygons)

        if self.current_mask is not None:
            geometries = mask_to_polygons(self.current_mask, self.current_transform_info)
            if geometries:
                combined = QgsGeometry.unaryUnion(geometries)
                if combined and not combined.isEmpty():
                    polygons_to_export.append({
                        'geometry': combined,
                        'score': self.current_score,
                        'transform_info': self.current_transform_info.copy() if self.current_transform_info else None,
                    })

        if not polygons_to_export:
            self.dock_widget.set_status("No polygons to export")
            return

        self.iface.mapCanvas().unsetMapTool(self.map_tool)

        self._segmentation_counter += 1
        layer_name = f"{self._current_layer_name}_segmentation_{self._segmentation_counter}"

        crs_str = None
        for pg in polygons_to_export:
            ti = pg.get('transform_info')
            if ti:
                crs_str = ti.get('crs', None)
                if crs_str and not (isinstance(crs_str, float) and str(crs_str) == 'nan'):
                    break
        if crs_str is None or (isinstance(crs_str, float) and str(crs_str) == 'nan'):
            if self.current_transform_info:
                crs_str = self.current_transform_info.get('crs', None)
        if crs_str is None or (isinstance(crs_str, float) and str(crs_str) == 'nan'):
            crs_str = self._current_layer.crs().authid() if self._current_layer and self._current_layer.crs().isValid() else 'EPSG:4326'
        if isinstance(crs_str, str) and crs_str.strip():
            crs = QgsCoordinateReferenceSystem(crs_str)
        else:
            crs = self._current_layer.crs() if self._current_layer else QgsCoordinateReferenceSystem('EPSG:4326')

        crs_authid = crs.authid() if crs.isValid() else 'EPSG:4326'
        if not crs_authid:
            crs_authid = 'EPSG:4326'

        result_layer = QgsVectorLayer(
            f"Polygon?crs={crs_authid}&field=id:integer&field=score:double&field=area:double",
            layer_name,
            "memory"
        )

        if not result_layer.isValid():
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Layer Creation Failed",
                "Could not create the output layer."
            )
            return

        result_layer.startEditing()
        for i, polygon_data in enumerate(polygons_to_export):
            feature = QgsFeature(result_layer.fields())
            geom = polygon_data['geometry']
            feature.setGeometry(geom)
            area = geom.area() if geom and not geom.isEmpty() else 0.0
            feature.setAttributes([i + 1, polygon_data['score'], area])
            result_layer.addFeature(feature)
            QgsMessageLog.logMessage(
                f"Added polygon {i+1}: area={area}, valid={geom.isGeosValid() if geom else False}",
                "AI Segmentation",
                level=Qgis.Info
            )
        result_layer.commitChanges()
        result_layer.updateExtents()

        symbol = QgsFillSymbol.createSimple({
            'color': '50,150,255,100',
            'outline_color': '0,100,200,255',
            'outline_width': '1.5'
        })
        renderer = QgsSingleSymbolRenderer(symbol)
        result_layer.setRenderer(renderer)

        QgsProject.instance().addMapLayer(result_layer)
        result_layer.triggerRepaint()
        self.iface.mapCanvas().refresh()

        QgsMessageLog.logMessage(
            f"Layer extent: {result_layer.extent().toString()}",
            "AI Segmentation",
            level=Qgis.Info
        )

        QgsMessageLog.logMessage(
            f"Created segmentation layer: {layer_name} with {len(polygons_to_export)} polygons (featureCount={result_layer.featureCount()})",
            "AI Segmentation",
            level=Qgis.Info
        )

        self._reset_session()
        self.dock_widget.reset_session()
        self.dock_widget.set_status(f"Exported: {layer_name}")

    def _on_tool_deactivated(self):
        self.dock_widget.set_segmentation_active(False)

    def _on_positive_click(self, point):
        if self.predictor is None or self.feature_dataset is None:
            self.dock_widget.set_status("Model not ready - please wait")
            return

        QgsMessageLog.logMessage(
            f"LEFT-CLICK (POSITIVE) at ({point.x():.2f}, {point.y():.2f})",
            "AI Segmentation",
            level=Qgis.Info
        )

        self.prompts.add_positive_point(point.x(), point.y())
        self._run_prediction()

    def _on_negative_click(self, point):
        if self.predictor is None or self.feature_dataset is None:
            self.dock_widget.set_status("Model not ready - please wait")
            return

        QgsMessageLog.logMessage(
            f"RIGHT-CLICK (NEGATIVE) at ({point.x():.2f}, {point.y():.2f})",
            "AI Segmentation",
            level=Qgis.Info
        )

        self.prompts.add_negative_point(point.x(), point.y())
        self._run_prediction()

    def _run_prediction(self):
        import numpy as np
        from rasterio.transform import from_bounds as transform_from_bounds
        from .core.feature_dataset import FeatureSampler

        all_points = self.prompts.positive_points + self.prompts.negative_points
        if not all_points:
            return

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]

        bounds = self.feature_dataset.bounds
        roi = (
            min(xs), max(xs),
            min(ys), max(ys),
            bounds[4], bounds[5]
        )

        sampler = FeatureSampler(self.feature_dataset, roi)
        if len(sampler) == 0:
            QgsMessageLog.logMessage(
                "No feature found for click location",
                "AI Segmentation",
                level=Qgis.Warning
            )
            self.dock_widget.set_status("Click outside feature area - try elsewhere")
            return

        for query in sampler:
            sample = self.feature_dataset[query]
            break

        bbox = sample["bbox"]
        features = sample["image"]

        img_size = self.predictor.model.image_encoder.img_size
        img_height = img_width = img_size
        input_height = input_width = img_size

        if "img_shape" in sample:
            img_height = sample["img_shape"][0]
            img_width = sample["img_shape"][1]
            input_height = sample["input_shape"][0]
            input_width = sample["input_shape"][1]

        self.predictor.set_image_feature(
            img_features=features.numpy(),
            img_size=(img_height, img_width),
            input_size=(input_height, input_width),
        )

        minx, maxx, miny, maxy = bbox[0], bbox[1], bbox[2], bbox[3]
        img_clip_transform = transform_from_bounds(minx, miny, maxx, maxy, img_width, img_height)

        point_coords, point_labels = self.prompts.get_points_for_predictor(img_clip_transform)

        if point_coords is None:
            return

        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )

        self.current_mask = masks[0]
        self.current_score = float(scores[0])
        self.current_transform_info = {
            "bbox": bbox,
            "img_shape": (img_height, img_width),
            "crs": self.feature_dataset.crs,
        }

        self._update_ui_after_prediction()

    def _update_ui_after_prediction(self):
        pos_count, neg_count = self.prompts.point_count
        self.dock_widget.set_point_count(pos_count, neg_count)

        if self.current_mask is not None:
            mask_pixels = int(self.current_mask.sum())
            QgsMessageLog.logMessage(
                f"Segmentation result: score={self.current_score:.3f}, mask_pixels={mask_pixels}",
                "AI Segmentation",
                level=Qgis.Info
            )
            self.dock_widget.set_status(f"Segmented - Score: {self.current_score:.2f}")
            self._update_mask_visualization()
        else:
            self.dock_widget.set_status("No mask generated - try clicking elsewhere")
            self._clear_mask_visualization()

    def _update_mask_visualization(self):
        if self.mask_rubber_band is None:
            return

        if self.current_mask is None:
            self._clear_mask_visualization()
            return

        try:
            from .core.polygon_exporter import mask_to_polygons

            geometries = mask_to_polygons(self.current_mask, self.current_transform_info)

            if geometries:
                combined = QgsGeometry.unaryUnion(geometries)
                if combined and not combined.isEmpty():
                    self.mask_rubber_band.setToGeometry(combined, None)
                else:
                    self._clear_mask_visualization()
            else:
                self._clear_mask_visualization()

        except Exception as e:
            import traceback
            QgsMessageLog.logMessage(
                f"Failed to visualize mask: {str(e)}\n{traceback.format_exc()}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            self._clear_mask_visualization()

    def _clear_mask_visualization(self):
        if self.mask_rubber_band:
            self.mask_rubber_band.reset(QgsWkbTypes.PolygonGeometry)

    def _on_clear_points(self):
        self.prompts.clear()

        if self.map_tool:
            self.map_tool.clear_markers()

        self._clear_mask_visualization()

        self.current_mask = None
        self.current_score = 0.0
        self.dock_widget.set_point_count(0, 0)
        self.dock_widget.set_status("Points cleared")

    def _on_undo(self):
        result = self.prompts.undo()
        if result is None:
            return

        if self.map_tool:
            self.map_tool.remove_last_marker()

        if self.prompts.point_count[0] + self.prompts.point_count[1] > 0:
            self._run_prediction()
        else:
            self.current_mask = None
            self.current_score = 0.0
            self._clear_mask_visualization()
            self.dock_widget.set_point_count(0, 0)
            self.dock_widget.set_status("All points removed")

    def _reset_session(self):
        self.prompts.clear()
        self.saved_polygons = []

        for rb in self.saved_rubber_bands:
            self.iface.mapCanvas().scene().removeItem(rb)
        self.saved_rubber_bands = []

        if self.map_tool:
            self.map_tool.clear_markers()

        self._clear_mask_visualization()

        self.current_mask = None
        self.current_score = 0.0
        self.current_transform_info = None
        self.dock_widget.set_point_count(0, 0)
        self.dock_widget.set_saved_polygon_count(0)
