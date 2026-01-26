import os
import glob
import sys
from typing import Optional, Dict, Any, Iterator, Tuple, List

from qgis.core import QgsMessageLog, Qgis

from .venv_manager import ensure_venv_packages_available
ensure_venv_packages_available()

import numpy as np
import pandas as pd

from .device_manager import get_optimal_device


class FeatureDataset:
    def __init__(self, root: str, cache: bool = True):
        self.root = root
        self.cache = cache
        self._cache = {}

        self.index = SpatialIndex()
        self.crs = None
        self.res = None
        self.model_type = None

        self._load_index()

    def _load_index(self):
        dir_name = os.path.basename(self.root)
        csv_path = os.path.join(self.root, dir_name + ".csv")

        tif_files = glob.glob(os.path.join(self.root, "*.tif"))
        if not tif_files:
            raise FileNotFoundError(f"No feature files found in {self.root}")

        for name in tif_files:
            basename = os.path.basename(name)
            if "vit_b" in basename:
                self.model_type = "vit_b"
                break
            elif "vit_l" in basename:
                self.model_type = "vit_l"
                break
            elif "vit_h" in basename:
                self.model_type = "vit_h"
                break

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            if len(df) == len(tif_files):
                for _, row in df.iterrows():
                    if self.crs is None:
                        crs_val = row["crs"]
                        if pd.notna(crs_val) and str(crs_val).strip():
                            self.crs = str(crs_val)
                    if self.res is None:
                        self.res = row["res"]

                    bounds = (
                        row["minx"], row["maxx"],
                        row["miny"], row["maxy"],
                        row["mint"], row["maxt"]
                    )
                    filepath = os.path.join(self.root, os.path.basename(row["filepath"]))
                    self.index.insert(int(row["id"]), bounds, filepath)

                QgsMessageLog.logMessage(
                    f"Loaded index from: {csv_path} ({len(df)} features)",
                    "AI Segmentation",
                    level=Qgis.Info
                )
                return

        self._build_index_from_files(tif_files)

    def _build_index_from_files(self, tif_files: List[str]):
        import rasterio

        QgsMessageLog.logMessage(
            f"Building index from {len(tif_files)} files...",
            "AI Segmentation",
            level=Qgis.Info
        )

        for i, filepath in enumerate(tif_files):
            try:
                with rasterio.open(filepath) as src:
                    if self.crs is None:
                        self.crs = str(src.crs)
                    if self.res is None:
                        self.res = src.res[0]

                    minx, miny, maxx, maxy = src.bounds
                    bounds = (minx, maxx, miny, maxy, 0, sys.maxsize)
                    self.index.insert(i, bounds, filepath)

            except Exception as e:
                QgsMessageLog.logMessage(
                    f"Failed to index {filepath}: {e}",
                    "AI Segmentation",
                    level=Qgis.Warning
                )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, query: Dict[str, Any]) -> Dict[str, Any]:
        import rasterio
        import torch

        bbox = query["bbox"]
        filepath = query["path"]

        if self.cache and filepath in self._cache:
            return self._cache[filepath]

        with rasterio.open(filepath) as src:
            data = src.read()
            tags = src.tags()

            img_shape = None
            input_shape = None
            if "img_shape" in tags:
                img_shape = eval(tags["img_shape"])
                input_shape = eval(tags["input_shape"])

            if data.dtype == np.uint16:
                data = data.astype(np.int32)
            elif data.dtype == np.uint32:
                data = data.astype(np.int64)

            device = get_optimal_device()
            tensor = torch.tensor(data, dtype=torch.float32, device=device)

            sample = {
                "crs": self.crs,
                "bbox": bbox,
                "path": filepath,
                "image": tensor,
            }

            if img_shape:
                sample["img_shape"] = img_shape
                sample["input_shape"] = input_shape

            if self.cache:
                self._cache[filepath] = sample

            return sample

    @property
    def bounds(self):
        return self.index.bounds


class SpatialIndex:
    def __init__(self):
        self._items = []
        self._bounds = None

    def insert(self, id: int, bounds: Tuple, obj: Any):
        self._items.append({
            "id": id,
            "bounds": bounds,
            "object": obj
        })
        self._update_bounds(bounds)

    def _update_bounds(self, bounds):
        minx, maxx, miny, maxy, mint, maxt = bounds
        if self._bounds is None:
            self._bounds = [minx, maxx, miny, maxy, mint, maxt]
        else:
            self._bounds[0] = min(self._bounds[0], minx)
            self._bounds[1] = max(self._bounds[1], maxx)
            self._bounds[2] = min(self._bounds[2], miny)
            self._bounds[3] = max(self._bounds[3], maxy)
            self._bounds[4] = min(self._bounds[4], mint)
            self._bounds[5] = max(self._bounds[5], maxt)

    @property
    def bounds(self):
        if self._bounds is None:
            return (0, 0, 0, 0, 0, 0)
        return tuple(self._bounds)

    def __len__(self) -> int:
        return len(self._items)

    def intersection(self, query_bounds: Tuple, objects: bool = False):
        qminx, qmaxx, qminy, qmaxy = query_bounds[0], query_bounds[1], query_bounds[2], query_bounds[3]

        results = []
        for item in self._items:
            minx, maxx, miny, maxy, _, _ = item["bounds"]

            if not (qmaxx < minx or qminx > maxx or qmaxy < miny or qminy > maxy):
                if objects:
                    results.append(type('Hit', (), {
                        'id': item['id'],
                        'bounds': item['bounds'],
                        'object': item['object']
                    })())
                else:
                    results.append(item['id'])

        return results


class FeatureSampler:
    def __init__(self, dataset: FeatureDataset, roi: Tuple[float, float, float, float, float, float]):
        self.dataset = dataset
        self.roi = roi

        self.q_bbox = None
        self.q_path = None
        self.dist_roi = None

        center_x_roi = (roi[0] + roi[1]) / 2
        center_y_roi = (roi[2] + roi[3]) / 2

        hits = list(dataset.index.intersection(roi, objects=True))
        
        QgsMessageLog.logMessage(
            f"Sampler query: ROI=({roi[0]:.6f}, {roi[1]:.6f}, {roi[2]:.6f}, {roi[3]:.6f}), "
            f"center=({center_x_roi:.6f}, {center_y_roi:.6f}), "
            f"found {len(hits)} intersecting features",
            "AI Segmentation",
            level=Qgis.Info
        )

        if len(hits) == 0:
            # If no hits, try to find nearest feature by expanding search
            # This handles edge cases where ROI is just outside feature bounds
            dataset_bounds = dataset.bounds
            QgsMessageLog.logMessage(
                f"Dataset bounds: ({dataset_bounds[0]:.6f}, {dataset_bounds[1]:.6f}, "
                f"{dataset_bounds[2]:.6f}, {dataset_bounds[3]:.6f})",
                "AI Segmentation",
                level=Qgis.Info
            )
            
            # Check if ROI center is within dataset bounds
            if (dataset_bounds[0] <= center_x_roi <= dataset_bounds[1] and
                dataset_bounds[2] <= center_y_roi <= dataset_bounds[3]):
                # Center is within bounds but no intersection - might be a precision issue
                # Try a slightly expanded ROI
                expansion_factor = 1.1
                expanded_roi = (
                    center_x_roi - (roi[1] - roi[0]) * expansion_factor / 2,
                    center_x_roi + (roi[1] - roi[0]) * expansion_factor / 2,
                    center_y_roi - (roi[3] - roi[2]) * expansion_factor / 2,
                    center_y_roi + (roi[3] - roi[2]) * expansion_factor / 2,
                    roi[4], roi[5]
                )
                hits = list(dataset.index.intersection(expanded_roi, objects=True))
                QgsMessageLog.logMessage(
                    f"Expanded ROI search found {len(hits)} features",
                    "AI Segmentation",
                    level=Qgis.Info
                )

        for hit in hits:
            bbox = hit.bounds
            center_x = (bbox[0] + bbox[1]) / 2
            center_y = (bbox[2] + bbox[3]) / 2

            dist = (center_x - center_x_roi) ** 2 + (center_y - center_y_roi) ** 2

            if self.dist_roi is None or dist < self.dist_roi:
                self.dist_roi = dist
                self.q_bbox = bbox
                self.q_path = hit.object

        self.length = 1 if self.q_bbox is not None else 0

        if self.length > 0:
            QgsMessageLog.logMessage(
                f"Selected nearest feature: bbox=({self.q_bbox[0]:.6f}, {self.q_bbox[1]:.6f}, "
                f"{self.q_bbox[2]:.6f}, {self.q_bbox[3]:.6f}), path={os.path.basename(self.q_path)}",
                "AI Segmentation",
                level=Qgis.Info
            )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self.q_bbox is not None:
            yield {"bbox": self.q_bbox, "path": self.q_path}

    def __len__(self) -> int:
        return self.length
