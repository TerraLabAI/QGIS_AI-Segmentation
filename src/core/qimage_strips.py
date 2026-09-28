









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
