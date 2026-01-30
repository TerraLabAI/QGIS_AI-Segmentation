









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



    out = _resample_axis(arr, height, 0)
    out = _resample_axis(out, width, 1)
    out = np.ascontiguousarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))
    data = out.tobytes()

    return QImage(data, width, height, width * 4, fmt).copy()


def pixmap_from_image(image):













    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtGui import QImage, QPixmap

    if image is None or image.isNull():
        return QPixmap()
    fmt = (QImage.Format.Format_ARGB32_Premultiplied if image.hasAlphaChannel()
           else QImage.Format.Format_RGB32)
    if image.format() != fmt:
        arr = qimage_array_in_strips(image, fmt, 4)
        if arr is None:
            return QPixmap()
        height, width = arr.shape[0], arr.shape[1]
        data = np.ascontiguousarray(arr).tobytes()
        ratio = image.devicePixelRatio()

        image = QImage(data, width, height, width * 4, fmt).copy()
        image.setDevicePixelRatio(ratio)
    return QPixmap.fromImage(image, Qt.ImageConversionFlag.NoFormatConversion)


def pixmap_from_file(path: str):

    from qgis.PyQt.QtGui import QImage

    return pixmap_from_image(QImage(str(path)))


def pixmap_from_bytes(data: bytes):


    from qgis.PyQt.QtGui import QImage

    image = QImage()
    image.loadFromData(data)
    return pixmap_from_image(image)


def pixmap_from_bounded_bytes(data: bytes, max_pixels: int):

    from qgis.PyQt.QtCore import QBuffer, QByteArray, QIODevice
    from qgis.PyQt.QtGui import QImageReader, QPixmap

    buffer = QBuffer()
    buffer.setData(QByteArray(data))
    buffer.open(QIODevice.OpenModeFlag.ReadOnly)
    reader = QImageReader(buffer)
    size = reader.size()
    if not size.isValid() or size.width() * size.height() > max_pixels:
        return QPixmap()
    return pixmap_from_image(reader.read())


def icon_from_file(path: str):

    from qgis.PyQt.QtGui import QIcon

    pixmap = pixmap_from_file(path)
    return QIcon(pixmap) if not pixmap.isNull() else QIcon()


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


def _resample_axis(arr: np.ndarray, dst: int, axis: int) -> np.ndarray:






    src = arr.shape[axis]
    if dst == src:
        return arr
    shape = [1] * arr.ndim
    shape[axis] = -1
    if dst < src:
        scale = src / dst
        starts = np.arange(dst, dtype=np.float64) * scale
        ends = np.minimum(starts + scale, src)
        first = np.floor(starts).astype(np.intp)
        taps = int(np.ceil(scale)) + 1
        out = None
        for k in range(taps):
            idx = first + k

            weight = np.clip(np.minimum(ends, idx + 1) - np.maximum(starts, idx), 0.0, None) / scale
            part = np.take(arr, np.minimum(idx, src - 1), axis=axis) * weight.astype(np.float32).reshape(shape)
            out = part if out is None else out + part
        return out
    centres = np.clip((np.arange(dst) + 0.5) * src / dst - 0.5, 0.0, src - 1)
    left = np.floor(centres).astype(np.intp)
    right = np.minimum(left + 1, src - 1)
    frac = (centres - left).astype(np.float32).reshape(shape)
    return (np.take(arr, left, axis=axis) * (1.0 - frac)
            + np.take(arr, right, axis=axis) * frac)


def smooth_pixmap_for_paint(image, width: int, height: int):









    from qgis.PyQt.QtGui import QImage, QPixmap

    if image is None or image.isNull():
        return QPixmap()
    width, height = max(1, int(width)), max(1, int(height))
    fmt = QImage.Format.Format_ARGB32_Premultiplied
    arr = qimage_array_in_strips(image, fmt, 4)
    if arr is None:
        return QPixmap()
    out = _resample_axis(arr.astype(np.float32), height, 0)
    out = _resample_axis(out, width, 1)
    out = np.ascontiguousarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))
    if not image.hasAlphaChannel():

        fmt = QImage.Format.Format_RGB32

    return pixmap_from_image(QImage(out.tobytes(), width, height, width * 4, fmt).copy())
