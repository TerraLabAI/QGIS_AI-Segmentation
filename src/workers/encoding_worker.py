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
            return torch.device("mps")
        else:
            return torch.device("cpu")
    except Exception:
        import torch
        return torch.device("cpu")


def _save_sam2_high_res_feats(feats, output_dir, tx, ty, feature_suffix):
    """Save SAM2 high_res_feats as float16 .npy files alongside the tile .tif."""
    high_res_feats = feats.get("high_res_feats", [])
    for idx, hr_feat in enumerate(high_res_feats):
        hr_np = hr_feat.squeeze(0).cpu().numpy().astype(np.float16)
        hr_filename = "tile_{}_{}_{}_hr{}.npy".format(
            tx, ty, feature_suffix, idx)
        hr_path = os.path.join(output_dir, hr_filename)
        np.save(hr_path, hr_np)


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


def _load_sam1_model(config, device):
    """Load a SAM1 model (ViT-B or ViT-L) and return (model, None)."""
    import torch
    from segment_anything import sam_model_registry

    checkpoint_path = config['checkpoint_path']
    registry_key = config.get('registry_key', 'vit_b')

    sam = sam_model_registry[registry_key](checkpoint=checkpoint_path)
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    sam.to(device)
    sam.eval()

    # Verify CUDA kernels are available for this GPU
    if device.type == "cuda":
        try:
            test = torch.zeros(1, device=device)
            _ = test + 1
            torch.cuda.synchronize()
            del test
        except RuntimeError:
            send_progress(0, "GPU not compatible with installed CUDA version, using CPU...")
            device = torch.device("cpu")
            sam.to(device)

    return sam, device


def _load_sam2_model(config, device):
    """Load a SAM2 model and return (predictor, device)."""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint_path = config['checkpoint_path']
    sam2_model_cfg = config.get('sam2_model_cfg', 'configs/sam2.1/sam2.1_hiera_l.yaml')

    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    model = build_sam2(sam2_model_cfg, checkpoint_path, device=str(device))
    predictor = SAM2ImagePredictor(model)

    return predictor, device


def encode_raster(config):
    try:
        import torch
        import rasterio
        import pandas as pd
        from rasterio.windows import Window

        raster_path = config['raster_path']
        output_dir = config['output_dir']
        layer_crs_wkt = config.get('layer_crs_wkt')
        layer_extent = config.get('layer_extent')
        model_family = config.get('model_family', 'sam1')
        feature_suffix = config.get('feature_suffix', 'vit_b')

        send_progress(0, "Preparing AI model...")

        device = get_optimal_device()

        # Load model based on family
        if model_family == "sam2":
            sam2_predictor, device = _load_sam2_model(config, device)
            sam = None
        else:
            sam, device = _load_sam1_model(config, device)
            sam2_predictor = None
            from segment_anything.utils.transforms import ResizeLongestSide

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

            if model_family != "sam2":
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
                    elif tile_data.shape[0] == 2:
                        # 2-band (e.g. SAR, spectral index): duplicate first band
                        tile_data = np.concatenate(
                            [tile_data, tile_data[:1, :, :]], axis=0)
                    elif tile_data.shape[0] > 3:
                        tile_data = tile_data[:3, :, :]

                    if tile_data.dtype != np.uint8:
                        tile_min = tile_data.min()
                        tile_max = tile_data.max()
                        if tile_max > tile_min:
                            tile_data = ((tile_data - tile_min) / (tile_max - tile_min) * 255).astype(np.uint8)
                        else:
                            tile_data = np.zeros_like(tile_data, dtype=np.uint8)

                    tile_hwc = np.transpose(tile_data, (1, 2, 0))

                    if model_family == "sam2":
                        # SAM2: use SAM2ImagePredictor to encode
                        # set_image handles preprocessing internally
                        with torch.no_grad():
                            try:
                                sam2_predictor.set_image(tile_hwc)
                                # Extract internal features after set_image
                                feats = sam2_predictor._features
                                # feats["image_embed"] shape: (1, C, H, W)
                                features_np = feats["image_embed"].squeeze(0).cpu().numpy()
                                _save_sam2_high_res_feats(
                                    feats, output_dir, tx, ty,
                                    feature_suffix)
                                input_h, input_w = actual_height, actual_width
                            except RuntimeError:
                                if device.type != "cpu":
                                    pct = int(5 + (processed / total_tiles) * 90)
                                    send_progress(pct, "GPU error, falling back to CPU...")
                                    try:
                                        if device.type == "cuda":
                                            torch.cuda.empty_cache()
                                    except Exception:
                                        pass
                                    device = torch.device("cpu")
                                    sam2_predictor, device = _load_sam2_model(config, device)
                                    sam2_predictor.set_image(tile_hwc)
                                    feats = sam2_predictor._features
                                    features_np = feats["image_embed"].squeeze(0).cpu().numpy()
                                    _save_sam2_high_res_feats(
                                        feats, output_dir, tx, ty,
                                        feature_suffix)
                                    input_h, input_w = actual_height, actual_width
                                else:
                                    raise
                    else:
                        # SAM1: existing encoding path
                        input_h, input_w = get_preprocess_shape(actual_height, actual_width, TILE_SIZE)
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
                                    pct = int(5 + (processed / total_tiles) * 90)
                                    send_progress(
                                        pct,
                                        "GPU error, falling back to CPU..."
                                    )
                                    try:
                                        if device.type == "cuda":
                                            torch.cuda.empty_cache()
                                    except Exception:
                                        pass
                                    device = torch.device("cpu")
                                    sam.to(device)
                                    tile_tensor = tile_tensor.to(device)
                                    try:
                                        features = sam.image_encoder(tile_tensor)
                                    except Exception as cpu_err:
                                        send_error("CPU retry also failed: {}".format(cpu_err))
                                        sys.exit(1)
                                else:
                                    raise

                        synchronize_device(device)
                        features_np = features.squeeze(0).cpu().numpy()

                    tile_minx = raster_bounds_left + col_off * pixel_size_x
                    tile_maxx = tile_minx + actual_width * pixel_size_x
                    tile_maxy = raster_bounds_top - row_off * pixel_size_y
                    tile_miny = tile_maxy - actual_height * pixel_size_y

                    feature_filename = "tile_{}_{}_{}.tif".format(tx, ty, feature_suffix)
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
                    send_progress(percent, "Encoding tile {}/{}...".format(processed, total_tiles))

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
