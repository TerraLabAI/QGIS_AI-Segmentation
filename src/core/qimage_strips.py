









from __future__ import annotations

import numpy as np


_STRIP_BYTES = 32768


def qimage_array_in_strips(image, image_format, channels: int) -> np.ndarray | None:



    width, height = image.width(), image.height()
    if width <= 0 or height <= 0:
        return None
    row_bytes = max(image.bytesPerLine(), width * max(channels, 4))
    rows = max(1, _STRIP_BYTES // row_bytes)
    parts = []
    for top in range(0, height, rows):
        strip = image.copy(0, top, width, min(rows, height - top))
        if strip.format() != image_format:
            strip = strip.convertToFormat(image_format)
        buffer = strip.constBits()
        buffer.setsize(strip.sizeInBytes())
        flat = np.frombuffer(bytes(buffer), dtype=np.uint8)
        parts.append(flat.reshape(strip.height(), strip.bytesPerLine())
                     [:, :width * channels].reshape(strip.height(), width, channels))
    return np.concatenate(parts, axis=0)


def _resample_matrix(src: int, dst: int) -> np.ndarray:


    if dst < src:
        scale = src / dst
        starts = np.arange(dst) * scale
        ends = starts + scale
        edges = np.arange(src + 1, dtype=np.float64)
        lo = np.maximum(starts[:, None], edges[None, :-1])
        hi = np.minimum(ends[:, None], edges[None, 1:])
        weights = np.clip(hi - lo, 0.0, None)
    else:
        centres = (np.arange(dst) + 0.5) * src / dst - 0.5
        centres = np.clip(centres, 0.0, src - 1)
        left = np.floor(centres).astype(int)
        right = np.minimum(left + 1, src - 1)
        frac = centres - left
        weights = np.zeros((dst, src), dtype=np.float64)
        rows = np.arange(dst)
        np.add.at(weights, (rows, left), 1.0 - frac)
        np.add.at(weights, (rows, right), frac)
    return weights / weights.sum(axis=1, keepdims=True)


def smooth_scaled_in_python(image, width: int, height: int, keep_aspect: bool = False):







    from qgis.PyQt.QtGui import QImage

    src_w, src_h = image.width(), image.height()
    if src_w <= 0 or src_h <= 0:
        return image
    if keep_aspect:

        fit_w = height * src_w // src_h
        if fit_w <= width:
            width = fit_w
        else:
            height = width * src_h // src_w
    width, height = max(1, int(width)), max(1, int(height))
    fmt = QImage.Format.Format_ARGB32_Premultiplied
    arr = qimage_array_in_strips(image, fmt, 4)
    if arr is None:
        return image
    if src_w == 2 * width and src_h == 2 * height:




        blocks = arr.astype(np.float32).reshape(height, 2, src_w, 4)
        half = blocks[:, 0] * np.float32(0.5) + blocks[:, 1] * np.float32(0.5)
        half = half.reshape(height, width, 2, 4)
        out = half[:, :, 0] * np.float32(0.5) + half[:, :, 1] * np.float32(0.5)
        out = np.ascontiguousarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))
        return QImage(out.tobytes(), width, height, width * 4, fmt).copy()
    rows = _resample_matrix(src_h, height).astype(np.float32)
    cols = _resample_matrix(src_w, width).astype(np.float32)

    out = rows @ arr.reshape(src_h, src_w * 4).astype(np.float32)
    out = out.reshape(height, src_w, 4).transpose(0, 2, 1) @ cols.T
    out = out.transpose(0, 2, 1)
    out = np.ascontiguousarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))
    data = out.tobytes()

    return QImage(data, width, height, width * 4, fmt).copy()


def nearest_rgb_sample_in_strips(image, width: int, height: int) -> np.ndarray | None:




    from qgis.PyQt.QtGui import QImage

    arr = qimage_array_in_strips(image, QImage.Format.Format_RGB888, 3)
    if arr is None:
        return None
    src_h, src_w = arr.shape[0], arr.shape[1]
    width, height = max(1, int(width)), max(1, int(height))
    rows = ((np.arange(height) + 0.5) * src_h / height).astype(int).clip(0, src_h - 1)
    cols = ((np.arange(width) + 0.5) * src_w / width).astype(int).clip(0, src_w - 1)
    return np.ascontiguousarray(arr[rows][:, cols])
