#!/usr/bin/env python3
import sys
import json
import os
import numpy as np


TILE_SIZE = 1024
STRIDE = 512


def send_progress(percent: int, message: str):
    progress = {"type": "progress", "percent": percent, "message": message}
    print(json.dumps(progress), flush=True)


def send_error(error_message: str):
    error = {"type": "error", "message": error_message}
    print(json.dumps(error), flush=True)


def send_success(tiles_processed: int):
    result = {"type": "success", "tiles_processed": tiles_processed}
    print(json.dumps(result), flush=True)


def get_optimal_device():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Prevent MPS OOM by disabling memory pool upper limit
            os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
            return torch.device("mps")
        else:
            return torch.device("cpu")
    except Exception:
        import torch
        return torch.device("cpu")


def synchronize_device(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def get_preprocess_shape(old_h: int, old_w: int, long_side_length: int):
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


def encode_raster(config):
    try:
        import torch
        import rasterio
        import pandas as pd
        from rasterio.windows import Window
        from segment_anything import sam_model_registry
        from segment_anything.utils.transforms import ResizeLongestSide

        raster_path = config['raster_path']
        output_dir = config['output_dir']
        checkpoint_path = config['checkpoint_path']
        layer_crs_wkt = config.get('layer_crs_wkt')
        layer_extent = config.get('layer_extent')

        send_progress(0, "Preparing AI model...")

        device = get_optimal_device()
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        sam.to(device)
        sam.eval()

        send_progress(5, "Reading image...")

        with rasterio.open(raster_path) as src:
            raster_width = src.width
            raster_height = src.height
            raster_transform = src.transform

            use_layer_extent = False

            if layer_extent:
                xmin, ymin, xmax, ymax = layer_extent

                if src.crs is None:
                    use_layer_extent = True
                else:
                    raster_bounds = src.bounds
                    left_near_zero = abs(raster_bounds.left) < 10
                    bottom_near_zero = abs(raster_bounds.bottom) < 10
                    right_near_width = abs(raster_bounds.right - raster_width) < 10
                    top_near_height = abs(raster_bounds.top - raster_height) < 10
                    all_near = left_near_zero and bottom_near_zero
                    bounds_look_like_pixels = all_near and right_near_width and top_near_height

                    if bounds_look_like_pixels:
                        use_layer_extent = True

            if use_layer_extent and layer_extent:
                xmin, ymin, xmax, ymax = layer_extent
                pixel_size_x = (xmax - xmin) / raster_width
                pixel_size_y = (ymax - ymin) / raster_height
                raster_bounds_left = xmin
                raster_bounds_top = ymax
            else:
                pixel_size_x = abs(raster_transform.a)
                pixel_size_y = abs(raster_transform.e)
                raster_bounds_left = src.bounds.left
                raster_bounds_top = src.bounds.top

            raster_crs = None
            if src.crs is not None:
                raster_crs = src.crs
            elif layer_crs_wkt and layer_crs_wkt.strip():
                try:
                    raster_crs = rasterio.crs.CRS.from_wkt(layer_crs_wkt)
                except Exception:
                    raster_crs = None

            num_tiles_x = max(1, (raster_width + STRIDE - 1) // STRIDE)
            num_tiles_y = max(1, (raster_height + STRIDE - 1) // STRIDE)
            total_tiles = num_tiles_x * num_tiles_y

            index_data = []
            processed = 0

            transform_obj = ResizeLongestSide(TILE_SIZE)

            for ty in range(num_tiles_y):
                for tx in range(num_tiles_x):
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
                    tile_resized = transform_obj.apply_image(tile_hwc)

                    tile_padded = pad_to_size(
                        np.transpose(tile_resized, (2, 0, 1)),
                        TILE_SIZE
                    )

                    tile_tensor = torch.as_tensor(tile_padded, dtype=torch.float32, device=device)
                    tile_tensor = tile_tensor.unsqueeze(0)
                    tile_tensor = sam.preprocess(tile_tensor)

                    with torch.no_grad():
                        features = sam.image_encoder(tile_tensor)

                    synchronize_device(device)
                    features_np = features.squeeze(0).cpu().numpy()

                    tile_minx = raster_bounds_left + col_off * pixel_size_x
                    tile_maxx = tile_minx + actual_width * pixel_size_x
                    tile_maxy = raster_bounds_top - row_off * pixel_size_y
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
                    send_progress(percent, f"Encoding tile {processed}/{total_tiles}...")

            dir_name = os.path.basename(output_dir)
            csv_path = os.path.join(output_dir, dir_name + ".csv")
            df = pd.DataFrame(index_data)
            df.to_csv(csv_path, index=False)

            send_progress(100, "Done! Cached for instant access")
            send_success(processed)

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        send_error(error_msg)
        sys.exit(1)


if __name__ == "__main__":
    try:
        config_json = sys.stdin.read()
        config = json.loads(config_json)
        encode_raster(config)
    except Exception as e:
        import traceback
        send_error(f"Failed to parse config: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)
