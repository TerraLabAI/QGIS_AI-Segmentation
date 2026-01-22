import os
from typing import Tuple, Optional, Callable
import numpy as np

from qgis.core import QgsMessageLog, Qgis, QgsRasterLayer


TILE_SIZE = 1024
STRIDE = 512


def get_preprocess_shape(old_h: int, old_w: int, long_side_length: int) -> Tuple[int, int]:
    scale = long_side_length * 1.0 / max(old_h, old_w)
    new_h, new_w = old_h * scale, old_w * scale
    new_w = int(new_w + 0.5)
    new_h = int(new_h + 0.5)
    return new_h, new_w


def pad_to_size(arr: np.ndarray, target_size: int) -> np.ndarray:
    h, w = arr.shape[-2:]
    pad_h = target_size - h
    pad_w = target_size - w
    if pad_h <= 0 and pad_w <= 0:
        return arr
    pad_h = max(0, pad_h)
    pad_w = max(0, pad_w)
    if arr.ndim == 3:
        return np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    else:
        return np.pad(arr, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)


def encode_raster_to_features(
    raster_path: str,
    output_dir: str,
    checkpoint_path: str,
    layer_crs_wkt: Optional[str] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[bool, str]:
    try:
        import torch
        import rasterio
        import pandas as pd
        from rasterio.windows import Window
        from segment_anything import sam_model_registry

        if progress_callback:
            progress_callback(0, "Loading SAM encoder...")

        QgsMessageLog.logMessage(
            f"Loading SAM model from: {checkpoint_path}",
            "AI Segmentation",
            level=Qgis.Info
        )

        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        sam.eval()

        if progress_callback:
            progress_callback(5, "Opening raster...")

        with rasterio.open(raster_path) as src:
            raster_width = src.width
            raster_height = src.height
            raster_transform = src.transform
            pixel_size_x = abs(raster_transform.a)
            pixel_size_y = abs(raster_transform.e)

            raster_crs = None
            if src.crs is not None:
                raster_crs = src.crs
            elif layer_crs_wkt and layer_crs_wkt.strip():
                try:
                    raster_crs = rasterio.crs.CRS.from_wkt(layer_crs_wkt)
                except Exception:
                    raster_crs = None

            crs_str = str(raster_crs) if raster_crs else "None (will save without CRS)"
            QgsMessageLog.logMessage(
                f"Raster: {raster_width}x{raster_height}, CRS: {crs_str}",
                "AI Segmentation",
                level=Qgis.Info
            )

            num_tiles_x = max(1, (raster_width + STRIDE - 1) // STRIDE)
            num_tiles_y = max(1, (raster_height + STRIDE - 1) // STRIDE)
            total_tiles = num_tiles_x * num_tiles_y

            QgsMessageLog.logMessage(
                f"Processing {total_tiles} tiles ({num_tiles_x}x{num_tiles_y})",
                "AI Segmentation",
                level=Qgis.Info
            )

            index_data = []
            processed = 0

            for ty in range(num_tiles_y):
                for tx in range(num_tiles_x):
                    if cancel_check and cancel_check():
                        return False, "Encoding cancelled by user"

                    row_off = ty * STRIDE
                    col_off = tx * STRIDE

                    actual_height = min(TILE_SIZE, raster_height - row_off)
                    actual_width = min(TILE_SIZE, raster_width - col_off)

                    if actual_height <= 0 or actual_width <= 0:
                        continue

                    window = Window(col_off, row_off, actual_width, actual_height)

                    tile_data = src.read(window=window)

                    if tile_data.shape[0] == 1:
                        tile_data = np.repeat(tile_data, 3, axis=0)
                    elif tile_data.shape[0] == 4:
                        tile_data = tile_data[:3, :, :]
                    elif tile_data.shape[0] > 3:
                        tile_data = tile_data[:3, :, :]

                    if tile_data.dtype != np.uint8:
                        tile_min = tile_data.min()
                        tile_max = tile_data.max()
                        if tile_max > tile_min:
                            tile_data = ((tile_data - tile_min) / (tile_max - tile_min) * 255).astype(np.uint8)
                        else:
                            tile_data = np.zeros_like(tile_data, dtype=np.uint8)

                    input_h, input_w = get_preprocess_shape(actual_height, actual_width, TILE_SIZE)

                    tile_hwc = np.transpose(tile_data, (1, 2, 0))

                    from segment_anything.utils.transforms import ResizeLongestSide
                    transform = ResizeLongestSide(TILE_SIZE)
                    tile_resized = transform.apply_image(tile_hwc)

                    tile_padded = pad_to_size(
                        np.transpose(tile_resized, (2, 0, 1)),
                        TILE_SIZE
                    )

                    tile_tensor = torch.as_tensor(tile_padded, dtype=torch.float32)
                    tile_tensor = tile_tensor.unsqueeze(0)

                    tile_tensor = sam.preprocess(tile_tensor)

                    with torch.no_grad():
                        features = sam.image_encoder(tile_tensor)

                    features_np = features.squeeze(0).cpu().numpy()

                    tile_minx = src.bounds.left + col_off * pixel_size_x
                    tile_maxx = tile_minx + actual_width * pixel_size_x
                    tile_maxy = src.bounds.top - row_off * pixel_size_y
                    tile_miny = tile_maxy - actual_height * pixel_size_y

                    feature_filename = f"tile_{tx}_{ty}_vit_b.tif"
                    feature_path = os.path.join(output_dir, feature_filename)

                    feature_transform = rasterio.transform.from_bounds(
                        tile_minx, tile_miny, tile_maxx, tile_maxy,
                        features_np.shape[2], features_np.shape[1]
                    )

                    with rasterio.open(
                        feature_path,
                        'w',
                        driver='GTiff',
                        height=features_np.shape[1],
                        width=features_np.shape[2],
                        count=features_np.shape[0],
                        dtype=features_np.dtype,
                        crs=raster_crs,
                        transform=feature_transform,
                    ) as dst:
                        dst.write(features_np)
                        dst.update_tags(
                            img_shape=str((actual_height, actual_width)),
                            input_shape=str((input_h, input_w))
                        )

                    index_data.append({
                        'id': processed,
                        'minx': tile_minx,
                        'maxx': tile_maxx,
                        'miny': tile_miny,
                        'maxy': tile_maxy,
                        'mint': 0,
                        'maxt': float('inf'),
                        'filepath': feature_filename,
                        'crs': str(raster_crs) if raster_crs else "",
                        'res': pixel_size_x,
                    })

                    processed += 1
                    percent = int(5 + (processed / total_tiles) * 90)
                    if progress_callback:
                        progress_callback(percent, f"Encoding tile {processed}/{total_tiles}...")

            dir_name = os.path.basename(output_dir)
            csv_path = os.path.join(output_dir, dir_name + ".csv")
            df = pd.DataFrame(index_data)
            df.to_csv(csv_path, index=False)

            QgsMessageLog.logMessage(
                f"Saved {processed} feature tiles and index to: {output_dir}",
                "AI Segmentation",
                level=Qgis.Success
            )

            if progress_callback:
                progress_callback(100, "Encoding complete!")

            return True, f"Encoded {processed} tiles"

    except Exception as e:
        import traceback
        error_msg = f"Encoding failed: {str(e)}\n{traceback.format_exc()}"
        QgsMessageLog.logMessage(
            error_msg,
            "AI Segmentation",
            level=Qgis.Warning
        )
        return False, str(e)
