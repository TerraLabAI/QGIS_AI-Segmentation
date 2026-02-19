#!/usr/bin/env python3
import sys
import json
import os
import gc

# Ensure consistent GPU ordering on multi-GPU systems
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import numpy as np  # noqa: E402


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
            best_idx = -1
            best_mem = 0
            count = torch.cuda.device_count()
            for i in range(count):
                try:
                    mem = torch.cuda.get_device_properties(i).total_memory
                    if mem >= 2 * 1024 ** 3 and mem > best_mem:
                        best_mem = mem
                        best_idx = i
                except Exception:
                    continue
            if best_idx < 0:
                return torch.device("cpu")
            # Verify CUDA kernels actually work
            cuda_dev = "cuda:{}".format(best_idx)
            t = torch.zeros(1, device=cuda_dev)
            _ = t + 1
            torch.cuda.synchronize(best_idx)
            del t
            torch.cuda.empty_cache()
            return torch.device(cuda_dev)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Prevent MPS OOM by disabling memory pool upper limit
            os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
            # Verify MPS actually works
            try:
                t = torch.zeros(1, device="mps")
                _ = t + 1
                torch.mps.synchronize()
                del t
            except Exception:
                return torch.device("cpu")
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
        visible_extent = config.get('visible_extent')  # (xmin, ymin, xmax, ymax) in raster CRS

        send_progress(0, "Preparing AI model...")

        device = get_optimal_device()

        # Log environment info for diagnostics
        device_name = str(device)
        if device.type == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(device)
                gpu_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                device_name = "cuda ({}, {:.1f}GB)".format(gpu_name, gpu_mem)
            except Exception:
                pass
        sys.stderr.write("[encoding_worker] PyTorch={}, device={}, Python={}.{}.{}\n".format(
            torch.__version__, device_name,
            sys.version_info.major, sys.version_info.minor, sys.version_info.micro
        ))
        sys.stderr.flush()
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        # Free memory before loading model onto GPU
        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        elif device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
            gc.collect()
            torch.mps.empty_cache()
        sam.to(device)
        sam.eval()

        # Verify CUDA kernels are available for this GPU
        if device.type == "cuda":
            try:
                test = torch.zeros(1, device=device)
                _ = test + 1  # Force a kernel execution
                torch.cuda.synchronize()
                del test
            except RuntimeError:
                send_progress(0, "GPU not compatible with installed CUDA version, using CPU...")
                device = torch.device("cpu")
                sam.to(device)

        send_progress(5, "Reading image...")

        try:
            src_dataset = rasterio.open(raster_path)
        except rasterio.errors.RasterioIOError:
            ext = os.path.splitext(raster_path)[1].upper()
            send_error(
                "{ext} format is not supported by the encoding engine.\n"
                "Please convert your raster to GeoTIFF (.tif) using:\n"
                "  QGIS: Raster > Conversion > Translate\n"
                "  Or: gdal_translate input{ext} output.tif".format(ext=ext)
            )
            sys.exit(1)

        with src_dataset as src:
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

            # Apply visible extent crop if specified
            crop_col_off = 0
            crop_row_off = 0
            crop_width = raster_width
            crop_height = raster_height

            if visible_extent:
                vxmin, vymin, vxmax, vymax = visible_extent
                # Convert geo coords to pixel coords
                try:
                    if use_layer_extent and layer_extent:
                        # Non-georeferenced: compute pixel coords from layer extent
                        le_xmin, le_ymin, le_xmax, le_ymax = layer_extent
                        c_left = (vxmin - le_xmin) / pixel_size_x
                        c_right = (vxmax - le_xmin) / pixel_size_x
                        r_top = (le_ymax - vymax) / pixel_size_y
                        r_bottom = (le_ymax - vymin) / pixel_size_y
                    else:
                        # Georeferenced: use rasterio transform
                        from rasterio.transform import rowcol
                        r_top, c_left = rowcol(raster_transform, vxmin, vymax)
                        r_bottom, c_right = rowcol(raster_transform, vxmax, vymin)

                    # Clamp to raster bounds
                    c_left = max(0, min(c_left, raster_width))
                    c_right = max(0, min(c_right, raster_width))
                    r_top = max(0, min(r_top, raster_height))
                    r_bottom = max(0, min(r_bottom, raster_height))

                    # Ensure valid range
                    if c_left > c_right:
                        c_left, c_right = c_right, c_left
                    if r_top > r_bottom:
                        r_top, r_bottom = r_bottom, r_top

                    crop_col_off = int(c_left)
                    crop_row_off = int(r_top)
                    crop_width = int(c_right - c_left)
                    crop_height = int(r_bottom - r_top)

                    if crop_width <= 0 or crop_height <= 0:
                        send_error("Visible area does not overlap with the raster.")
                        sys.exit(1)

                    # Update bounds for the cropped region
                    raster_bounds_left = raster_bounds_left + crop_col_off * pixel_size_x
                    raster_bounds_top = raster_bounds_top - crop_row_off * pixel_size_y

                    sys.stderr.write(
                        "[encoding_worker] Visible extent crop: "
                        "col_off={}, row_off={}, width={}, height={}\n".format(
                            crop_col_off, crop_row_off, crop_width, crop_height)
                    )
                    sys.stderr.flush()
                except Exception as crop_err:
                    sys.stderr.write(
                        "[encoding_worker] Failed to apply visible extent crop: {}\n".format(
                            crop_err)
                    )
                    sys.stderr.flush()
                    # Fall back to full raster
                    crop_col_off = 0
                    crop_row_off = 0
                    crop_width = raster_width
                    crop_height = raster_height

            num_tiles_x = max(1, (crop_width + STRIDE - 1) // STRIDE)
            num_tiles_y = max(1, (crop_height + STRIDE - 1) // STRIDE)
            total_tiles = num_tiles_x * num_tiles_y

            # Log raster info for diagnostics (no paths - privacy)
            sys.stderr.write(
                "[encoding_worker] Raster: {}x{}px, bands={}, dtype={}, "
                "CRS={}, tiles={}x{}={}{}\n".format(
                    raster_width, raster_height, src.count, src.dtypes[0],
                    raster_crs if raster_crs else "none",
                    num_tiles_x, num_tiles_y, total_tiles,
                    " (visible crop: {}x{})".format(crop_width, crop_height) if visible_extent else ""
                )
            )
            sys.stderr.flush()

            index_data = []
            processed = 0

            transform_obj = ResizeLongestSide(TILE_SIZE)

            for ty in range(num_tiles_y):
                for tx in range(num_tiles_x):
                    row_off = crop_row_off + ty * STRIDE
                    col_off = crop_col_off + tx * STRIDE

                    actual_height = min(TILE_SIZE, crop_row_off + crop_height - row_off)
                    actual_width = min(TILE_SIZE, crop_col_off + crop_width - col_off)

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
                        try:
                            features = sam.image_encoder(tile_tensor)
                        except RuntimeError:
                            if device.type != "cpu":
                                # GPU error (OOM, illegal access, no kernel image, etc.)
                                pct = int(5 + (processed / total_tiles) * 90)
                                send_progress(
                                    pct,
                                    "GPU error, falling back to CPU..."
                                )
                                try:
                                    del tile_tensor
                                    if device.type == "cuda":
                                        torch.cuda.empty_cache()
                                    elif device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
                                        torch.mps.empty_cache()
                                    gc.collect()
                                except Exception:
                                    pass
                                device = torch.device("cpu")
                                sam.to(device)
                                # Re-create tile tensor on CPU
                                tile_tensor = torch.as_tensor(
                                    tile_padded, dtype=torch.float32, device=device
                                ).unsqueeze(0)
                                tile_tensor = sam.preprocess(tile_tensor)
                                try:
                                    features = sam.image_encoder(tile_tensor)
                                except Exception as cpu_err:
                                    send_error("CPU retry also failed: {}".format(cpu_err))
                                    sys.exit(1)
                            else:
                                raise

                    synchronize_device(device)
                    features_np = features.squeeze(0).cpu().numpy()

                    # Free GPU memory after each tile to prevent OOM accumulation
                    del tile_tensor, features
                    if processed % 10 == 0:
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        elif device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                        gc.collect()

                    # Use relative offset within crop region for geo bounds
                    rel_col = col_off - crop_col_off
                    rel_row = row_off - crop_row_off
                    tile_minx = raster_bounds_left + rel_col * pixel_size_x
                    tile_maxx = tile_minx + actual_width * pixel_size_x
                    tile_maxy = raster_bounds_top - rel_row * pixel_size_y
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
