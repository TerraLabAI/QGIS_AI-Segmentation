
import os
from typing import Tuple, Optional
import numpy as np

from qgis.core import (
    QgsRasterLayer,
    QgsRectangle,
    QgsMessageLog,
    Qgis,
)


def _is_web_layer(layer: QgsRasterLayer) -> bool:
    source = layer.source()
    web_indicators = ['http://', 'https://', 'type=xyz', 'type=wms', 'url=']
    for indicator in web_indicators:
        if indicator in source.lower():
            return True
    return not os.path.isfile(source)


def _get_numpy_dtype(qgis_dtype):
    dtype_map = {
        Qgis.DataType.Byte: np.uint8,
        Qgis.DataType.Int8: np.int8,
        Qgis.DataType.UInt16: np.uint16,
        Qgis.DataType.Int16: np.int16,
        Qgis.DataType.UInt32: np.uint32,
        Qgis.DataType.Int32: np.int32,
        Qgis.DataType.Float32: np.float32,
        Qgis.DataType.Float64: np.float64,
        Qgis.DataType.ARGB32: np.uint8,  
        Qgis.DataType.ARGB32_Premultiplied: np.uint8,
    }
    return dtype_map.get(qgis_dtype, np.float64)


def raster_to_numpy(
    layer: QgsRasterLayer,
    extent: QgsRectangle = None,
    max_size: int = None
) -> Tuple[Optional[np.ndarray], dict]:
    try:
        from .debug_settings import get_settings
        settings = get_settings()

        if max_size is None:
            max_size = settings.max_image_size

        provider = layer.dataProvider()

        if extent is None:
            extent = layer.extent()

        width = provider.xSize()
        height = provider.ySize()

        is_web = _is_web_layer(layer)

        if width == 0 or height == 0 or is_web:
            extent_width = extent.width()
            extent_height = extent.height()

            if extent_width == 0 or extent_height == 0:
                QgsMessageLog.logMessage(
                    "Layer has zero extent",
                    "AI Segmentation",
                    level=Qgis.Warning
                )
                return None, {}

            aspect_ratio = extent_width / extent_height
            if aspect_ratio > 1:
                width = max_size
                height = int(max_size / aspect_ratio)
            else:
                height = max_size
                width = int(max_size * aspect_ratio)

            scale = 1.0
            QgsMessageLog.logMessage(
                f"Web layer detected, using resolution: {width}x{height}",
                "AI Segmentation",
                level=Qgis.Info
            )
        else:
            scale = 1.0
            if max(width, height) > max_size:
                scale = max_size / max(width, height)
                width = int(width * scale)
                height = int(height * scale)

        band_count = provider.bandCount()
        data_type = provider.dataType(1)
        numpy_dtype = _get_numpy_dtype(data_type)

        QgsMessageLog.logMessage(
            f"Reading layer: bands={band_count}, dtype={data_type}, size={width}x{height}",
            "AI Segmentation",
            level=Qgis.Info
        )

        if data_type in (Qgis.DataType.ARGB32, Qgis.DataType.ARGB32_Premultiplied):
            block = provider.block(1, extent, width, height)
            if block is None or not block.isValid():
                QgsMessageLog.logMessage(
                    "Failed to read raster block",
                    "AI Segmentation",
                    level=Qgis.Warning
                )
                return None, {}

            raw_data = block.data()
            if len(raw_data) == 0:
                QgsMessageLog.logMessage(
                    "Empty raster data received",
                    "AI Segmentation",
                    level=Qgis.Warning
                )
                return None, {}

            pixels = np.frombuffer(raw_data, dtype=np.uint8)
            expected_size = width * height * 4
            if len(pixels) != expected_size:
                actual_pixels = len(pixels) // 4
                QgsMessageLog.logMessage(
                    f"Adjusting dimensions: expected {width*height}, got {actual_pixels}",
                    "AI Segmentation",
                    level=Qgis.Info
                )
                height = int(np.sqrt(actual_pixels * height / width))
                width = actual_pixels // height
                if width * height * 4 > len(pixels):
                    return None, {}

            pixels = pixels.reshape(height, width, 4)
            image = pixels[:, :, [2, 1, 0]].astype(np.float32)

        elif band_count >= 3:
            block_r = provider.block(1, extent, width, height)
            block_g = provider.block(2, extent, width, height)
            block_b = provider.block(3, extent, width, height)

            if not all(b and b.isValid() for b in [block_r, block_g, block_b]):
                QgsMessageLog.logMessage(
                    "Failed to read RGB bands",
                    "AI Segmentation",
                    level=Qgis.Warning
                )
                return None, {}

            r = np.frombuffer(block_r.data(), dtype=numpy_dtype).reshape(height, width)
            g = np.frombuffer(block_g.data(), dtype=numpy_dtype).reshape(height, width)
            b = np.frombuffer(block_b.data(), dtype=numpy_dtype).reshape(height, width)

            image = np.stack([r, g, b], axis=-1).astype(np.float32)

        elif band_count == 1:
            block = provider.block(1, extent, width, height)
            if not block or not block.isValid():
                QgsMessageLog.logMessage(
                    "Failed to read raster block",
                    "AI Segmentation",
                    level=Qgis.Warning
                )
                return None, {}

            gray = np.frombuffer(block.data(), dtype=numpy_dtype).reshape(height, width)
            image = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

        else:
            QgsMessageLog.logMessage(
                f"Unsupported band count: {band_count}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            return None, {}

        image = normalize_image(image)

        geo_transform = {
            "x_min": extent.xMinimum(),
            "y_max": extent.yMaximum(),
            "pixel_width": extent.width() / width,
            "pixel_height": extent.height() / height,
            "width": width,
            "height": height,
            "scale": scale,
        }

        QgsMessageLog.logMessage(
            f"Successfully read raster: shape={image.shape}",
            "AI Segmentation",
            level=Qgis.Info
        )

        return image.astype(np.float32), geo_transform

    except Exception as e:
        QgsMessageLog.logMessage(
            f"Failed to convert raster to numpy: {str(e)}",
            "AI Segmentation",
            level=Qgis.Critical
        )
        import traceback
        QgsMessageLog.logMessage(
            traceback.format_exc(),
            "AI Segmentation",
            level=Qgis.Critical
        )
        return None, {}


def normalize_image(image: np.ndarray) -> np.ndarray:
    
    img_min = np.nanmin(image)
    img_max = np.nanmax(image)

    if img_max - img_min < 1e-8:
        return np.full_like(image, 127.5)

    if img_min >= 0 and img_max <= 255:
        return image

    normalized = (image - img_min) / (img_max - img_min) * 255.0
    return np.clip(normalized, 0, 255)


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int]
) -> np.ndarray:
    
    src_h, src_w = image.shape[:2]
    dst_h, dst_w = target_size

    if src_h == dst_h and src_w == dst_w:
        return image.copy()

    x_ratio = src_w / dst_w
    y_ratio = src_h / dst_h

    x_coords = (np.arange(dst_w) * x_ratio).astype(np.int32)
    y_coords = (np.arange(dst_h) * y_ratio).astype(np.int32)

    x_coords = np.clip(x_coords, 0, src_w - 1)
    y_coords = np.clip(y_coords, 0, src_h - 1)

    if image.ndim == 3:
        resized = image[y_coords[:, np.newaxis], x_coords[np.newaxis, :], :]
    else:
        resized = image[y_coords[:, np.newaxis], x_coords[np.newaxis, :]]

    return resized


def pixel_to_coord(
    pixel_x: int,
    pixel_y: int,
    geo_transform: dict
) -> Tuple[float, float]:
    
    x = geo_transform["x_min"] + pixel_x * geo_transform["pixel_width"]
    y = geo_transform["y_max"] - pixel_y * geo_transform["pixel_height"]
    return x, y


def coord_to_pixel(
    x: float,
    y: float,
    geo_transform: dict
) -> Tuple[int, int]:
    
    pixel_x = int((x - geo_transform["x_min"]) / geo_transform["pixel_width"])
    pixel_y = int((geo_transform["y_max"] - y) / geo_transform["pixel_height"])
    return pixel_x, pixel_y


def map_point_to_sam_coords(
    map_x: float,
    map_y: float,
    transform_info: dict,
    scale_to_sam_space: bool = True
) -> Tuple[float, float]:
    extent = transform_info["extent"]
    original_size = transform_info["original_size"]
    scale = transform_info.get("scale", 1.0)

    x_min, y_min, x_max, y_max = extent
    geo_width = x_max - x_min
    geo_height = y_max - y_min

    if geo_width <= 0 or geo_height <= 0:
        QgsMessageLog.logMessage(
            f"Invalid extent: width={geo_width}, height={geo_height}",
            "AI Segmentation",
            level=Qgis.Warning
        )
        return 0.0, 0.0

    pixel_x = (map_x - x_min) / geo_width * original_size[1]
    pixel_y = (y_max - map_y) / geo_height * original_size[0]

    pixel_x = max(0, min(pixel_x, original_size[1] - 1))
    pixel_y = max(0, min(pixel_y, original_size[0] - 1))

    if scale_to_sam_space:
        out_x = pixel_x * scale
        out_y = pixel_y * scale
        input_size = transform_info.get("input_size", 1024)
        out_x = max(0, min(out_x, input_size - 1))
        out_y = max(0, min(out_y, input_size - 1))
    else:
        out_x = pixel_x
        out_y = pixel_y

    QgsMessageLog.logMessage(
        f"[COORD] map({map_x:.2f}, {map_y:.2f}) -> pixel({pixel_x:.2f}, {pixel_y:.2f}) -> sam({out_x:.2f}, {out_y:.2f})",
        "AI Segmentation",
        level=Qgis.Info
    )

    return out_x, out_y
