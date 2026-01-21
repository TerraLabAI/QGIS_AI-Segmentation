"""
Image Utilities for AI Segmentation

Handles conversion between QGIS raster layers and numpy arrays,
as well as image preprocessing operations.
"""

from typing import Tuple, Optional
import numpy as np

from qgis.core import (
    QgsRasterLayer,
    QgsRectangle,
    QgsMessageLog,
    Qgis,
)


def raster_to_numpy(
    layer: QgsRasterLayer,
    extent: QgsRectangle = None,
    max_size: int = 2048
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Convert a QGIS raster layer to a numpy array.

    Args:
        layer: QgsRasterLayer to convert
        extent: Optional extent to read (defaults to full layer)
        max_size: Maximum dimension for the output (for memory management)

    Returns:
        Tuple of (image_array, geo_transform) or (None, {}) on error
        image_array is in RGB format (H, W, 3)
        geo_transform contains pixel-to-coordinate mapping info
    """
    try:
        provider = layer.dataProvider()

        if extent is None:
            extent = layer.extent()

        # Calculate output size respecting max_size
        width = provider.xSize()
        height = provider.ySize()

        # Scale down if needed
        scale = 1.0
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

        # Read bands
        band_count = provider.bandCount()

        if band_count >= 3:
            # Read RGB bands
            block_r = provider.block(1, extent, width, height)
            block_g = provider.block(2, extent, width, height)
            block_b = provider.block(3, extent, width, height)

            r = np.frombuffer(block_r.data(), dtype=np.float64).reshape(height, width)
            g = np.frombuffer(block_g.data(), dtype=np.float64).reshape(height, width)
            b = np.frombuffer(block_b.data(), dtype=np.float64).reshape(height, width)

            # Stack to RGB
            image = np.stack([r, g, b], axis=-1)

        elif band_count == 1:
            # Single band - convert to grayscale RGB
            block = provider.block(1, extent, width, height)
            gray = np.frombuffer(block.data(), dtype=np.float64).reshape(height, width)
            image = np.stack([gray, gray, gray], axis=-1)

        else:
            QgsMessageLog.logMessage(
                f"Unsupported band count: {band_count}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            return None, {}

        # Normalize to 0-255 range
        image = normalize_image(image)

        # Geo transform info for coordinate mapping
        geo_transform = {
            "x_min": extent.xMinimum(),
            "y_max": extent.yMaximum(),
            "pixel_width": extent.width() / width,
            "pixel_height": extent.height() / height,
            "width": width,
            "height": height,
            "scale": scale,
        }

        return image.astype(np.float32), geo_transform

    except Exception as e:
        QgsMessageLog.logMessage(
            f"Failed to convert raster to numpy: {str(e)}",
            "AI Segmentation",
            level=Qgis.Critical
        )
        return None, {}


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image values to 0-255 range.

    Handles various input ranges (0-1, 0-255, 0-65535, etc.)

    Args:
        image: Input image array

    Returns:
        Normalized image in 0-255 range
    """
    # Get min/max
    img_min = np.nanmin(image)
    img_max = np.nanmax(image)

    # Handle constant images
    if img_max - img_min < 1e-8:
        return np.full_like(image, 127.5)

    # Check if already in 0-255 range
    if img_min >= 0 and img_max <= 255:
        return image

    # Normalize to 0-255
    normalized = (image - img_min) / (img_max - img_min) * 255.0
    return np.clip(normalized, 0, 255)


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int]
) -> np.ndarray:
    """
    Resize image using simple interpolation.

    Uses a basic bilinear-like approach without external dependencies.
    For production use, consider PIL or skimage for better quality.

    Args:
        image: Input image (H, W, C) or (H, W)
        target_size: Target size as (height, width)

    Returns:
        Resized image
    """
    src_h, src_w = image.shape[:2]
    dst_h, dst_w = target_size

    # Handle case where no resize needed
    if src_h == dst_h and src_w == dst_w:
        return image.copy()

    # Create coordinate mappings
    x_ratio = src_w / dst_w
    y_ratio = src_h / dst_h

    # Generate destination coordinates
    x_coords = (np.arange(dst_w) * x_ratio).astype(np.int32)
    y_coords = (np.arange(dst_h) * y_ratio).astype(np.int32)

    # Clip to valid range
    x_coords = np.clip(x_coords, 0, src_w - 1)
    y_coords = np.clip(y_coords, 0, src_h - 1)

    # Simple nearest neighbor resize
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
    """
    Convert pixel coordinates to geographic coordinates.

    Args:
        pixel_x: X pixel coordinate
        pixel_y: Y pixel coordinate
        geo_transform: Geo transform dictionary from raster_to_numpy

    Returns:
        Tuple of (x_coord, y_coord) in the layer's CRS
    """
    x = geo_transform["x_min"] + pixel_x * geo_transform["pixel_width"]
    y = geo_transform["y_max"] - pixel_y * geo_transform["pixel_height"]
    return x, y


def coord_to_pixel(
    x: float,
    y: float,
    geo_transform: dict
) -> Tuple[int, int]:
    """
    Convert geographic coordinates to pixel coordinates.

    Args:
        x: X coordinate in layer's CRS
        y: Y coordinate in layer's CRS
        geo_transform: Geo transform dictionary from raster_to_numpy

    Returns:
        Tuple of (pixel_x, pixel_y)
    """
    pixel_x = int((x - geo_transform["x_min"]) / geo_transform["pixel_width"])
    pixel_y = int((geo_transform["y_max"] - y) / geo_transform["pixel_height"])
    return pixel_x, pixel_y


def map_point_to_sam_coords(
    map_x: float,
    map_y: float,
    transform_info: dict
) -> Tuple[float, float]:
    """
    Convert map coordinates to SAM input coordinates (0-1024 range).

    Args:
        map_x: X coordinate in map CRS
        map_y: Y coordinate in map CRS
        transform_info: Transform info from encoding

    Returns:
        Tuple of (sam_x, sam_y) in SAM coordinate space (0-1024)
    """
    extent = transform_info["extent"]
    original_size = transform_info["original_size"]
    scale = transform_info["scale"]

    # Map coordinates to original image pixel coordinates
    x_min, y_min, x_max, y_max = extent
    width = x_max - x_min
    height = y_max - y_min

    # Pixel in original image
    pixel_x = (map_x - x_min) / width * original_size[1]
    pixel_y = (y_max - map_y) / height * original_size[0]

    # Scale to SAM input size
    sam_x = pixel_x * scale
    sam_y = pixel_y * scale

    return sam_x, sam_y
