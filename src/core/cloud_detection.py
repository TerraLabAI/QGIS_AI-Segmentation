"""RLE decoding and tile image encoding for Pro automatic detection mode."""
from __future__ import annotations

import base64
import logging
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qgis.core import QgsRasterLayer, QgsRectangle

# RLE format used by the detection service.
# Each mask RLE is a space-separated string of "offset count" pairs.
# Offsets are ONE-BASED, ROW-MAJOR:
#   idx = offset - 1
#   y   = idx // width
#   x   = idx % width
# Then `count` consecutive foreground pixels starting at idx.
# This is NOT COCO column-major {counts, size}.
_RLE_FORMAT: str = "offset_count_row_major_one_based"

# Tiles are uploaded as JPEG, not PNG. The detection backend decodes with a
# format-agnostic PIL Image.open(), so the wire format is purely a client-side
# size choice, and JPEG is ~4-8x smaller than lossless PNG for aerial imagery.
# Upload bytes dominate the per-tile time on slow connections (and the submits
# are serialized on the worker thread), so this is the single biggest network
# win for users on poor links. Quality 90 keeps libjpeg at 4:4:4 chroma
# (no colour subsampling), so detection quality is effectively unchanged while
# the payload stays a fraction of the PNG size. Lower it toward 80-85 to trade
# a little fidelity for even smaller uploads.
_TILE_IMAGE_FORMAT: str = "JPEG"
_TILE_JPEG_QUALITY: int = 90

# Blank-tile skip: a tile that renders as essentially one colour (nodata fill,
# a mosaic gap, an out-of-footprint rectangle corner, a rendered ocean pad)
# returns no detections but still costs one credit. Detecting it cheaply BEFORE
# submit and dropping it uncharged directly protects the free trial. This is a
# DOMINANT-VALUE test, deliberately NOT a low-variance / low-texture test: only
# a tile whose single quantized-colour bucket covers _BLANK_TILE_DOMINANT_FRAC
# is skipped, so genuinely uniform-but-real terrain (a solid field, calm water,
# snow) keeps its natural render micro-variation and is never over-culled.
_BLANK_TILE_SAMPLE_PX: int = 32
_BLANK_TILE_DOMINANT_FRAC: float = 0.995
# Colour quantization step for the dominant-bucket test: near-identical values
# (render antialiasing, a faint gradient) collapse into one bucket while a real
# scene still spreads across several. 16 keeps the top 4 bits per channel.
_BLANK_TILE_QUANT: int = 16

logger = logging.getLogger(__name__)


def decode_rle_to_mask(rle: str | dict, height: int, width: int) -> np.ndarray:
    """Decode one mask RLE from the backend response to a boolean numpy array.

    RLE format: space-separated "offset count" pairs, offsets 1-based, row-major.
    See the _RLE_FORMAT constant.

    Args:
        rle:    RLE string from the backend ("offset count offset count ...").
                A dict argument is not supported by this implementation; if a dict
                is received the function logs a warning and returns an empty mask.
        height: Tile height in pixels (the actual tile, not padded).
        width:  Tile width in pixels.

    Returns:
        Boolean numpy array of shape (height, width). True = foreground.
    """
    flat = np.zeros(height * width, dtype=np.uint8)

    if isinstance(rle, dict):
        logger.warning(
            "decode_rle_to_mask: received dict RLE (unsupported format); "
            "returning empty mask for tile %dx%d",
            width,
            height,
        )
        return flat.reshape((height, width)).astype(bool)

    if not isinstance(rle, str) or not rle.strip():
        return flat.reshape((height, width)).astype(bool)

    tokens = rle.split()
    total = height * width

    for i in range(0, len(tokens) - 1, 2):
        try:
            offset = int(tokens[i])
            count = int(tokens[i + 1])
        except ValueError:
            logger.warning(
                "decode_rle_to_mask: invalid token pair at index %d; skipping", i
            )
            continue

        # Convert 1-based offset to 0-based flat index.
        idx = offset - 1
        if idx < 0:
            logger.warning(
                "decode_rle_to_mask: offset %d is below 1; skipping pair", offset
            )
            continue

        if count <= 0:
            # A zero/negative run length is malformed; skip rather than silently
            # producing an empty or reversed slice.
            logger.warning(
                "decode_rle_to_mask: non-positive count %d at offset %d; skipping",
                count, offset,
            )
            continue

        end = idx + count
        if end > total:
            logger.warning(
                "decode_rle_to_mask: run [%d, %d) exceeds mask size %d; clipping",
                idx,
                end,
                total,
            )
            end = total

        flat[idx:end] = 1

    return flat.reshape((height, width)).astype(bool)


def _set_quality_render_flags(settings) -> None:
    """Turn on antialiasing + high-quality image transforms for a detection
    render. Without these the raster is resampled nearest-neighbour whenever
    the render resolution differs from the source/tile-pyramid native grid,
    which feeds blocky staircase pixels to the model and degrades mask edges.
    Best-effort per flag: enum names moved across QGIS versions (legacy
    QgsMapSettings attributes vs Qgis.MapSettingsFlag), and a missing flag on
    an old QGIS must never break the render itself."""
    from qgis.core import Qgis, QgsMapSettings

    for name in ("Antialiasing", "HighQualityImageTransforms"):
        flag = getattr(QgsMapSettings, name, None) or getattr(
            getattr(Qgis, "MapSettingsFlag", None), name, None)
        if flag is None:
            continue
        try:
            settings.setFlag(flag, True)
        except (TypeError, AttributeError):  # enum type mismatch on this QGIS
            pass


def render_zone_to_image(
    layer: "QgsRasterLayer",
    extent: "QgsRectangle",
    width: int,
    height: int,
    timeout_ms: int = 120000,
):
    """Render a whole zone to one QImage off the main thread, without freezing.

    Uses QgsMapRendererParallelJob, which renders on QGIS's own background
    thread pool and emits finished() on the main thread. We wait on a local
    QEventLoop so the UI keeps repainting and stays responsive (the
    QgsMapRendererCustomPainterJob path it replaces blocked the UI, and doing it
    once per tile froze QGIS on large online zones).

    Critically for georeferencing: QgsMapSettings expands the requested extent to
    match the output aspect ratio. We return settings.visibleExtent() (the extent
    actually rendered) so the caller can map mask pixels back to geography with
    zero offset. Render once, slice into tiles: every tile then shares this exact
    pixel-to-map mapping, so detections land precisely on the imagery.

    Args:
        layer:   Raster layer to render (local or online: XYZ/WMS/WMTS).
        extent:  Requested zone extent in the layer's CRS.
        width:   Output image width in pixels.
        height:  Output image height in pixels.
        timeout_ms: Hard cap so a stalled network render cannot hang forever.

    Returns:
        (QImage, QgsRectangle actual_extent) on success, or (None, None).
    """
    from qgis.core import QgsMapRendererParallelJob, QgsMapSettings
    from qgis.PyQt.QtCore import QEventLoop, QSize, QTimer
    from qgis.PyQt.QtGui import QColor

    if width <= 0 or height <= 0:
        logger.warning("render_zone_to_image: invalid dimensions %dx%d", width, height)
        return None, None

    t0 = time.monotonic()
    try:
        settings = QgsMapSettings()
        settings.setOutputSize(QSize(width, height))
        settings.setExtent(extent)
        settings.setLayers([layer])
        settings.setDestinationCrs(layer.crs())
        settings.setBackgroundColor(QColor(0, 0, 0))
        _set_quality_render_flags(settings)
        # The extent QGIS will actually render (aspect-corrected). Using this for
        # the geo-transform is what keeps detections aligned with the imagery.
        actual_extent = settings.visibleExtent()

        job = QgsMapRendererParallelJob(settings)
        loop = QEventLoop()
        job.finished.connect(loop.quit)
        # Safety net: never block the UI loop indefinitely on a stalled render.
        QTimer.singleShot(timeout_ms, loop.quit)
        job.start()
        loop.exec()

        if not job.isActive():
            img = job.renderedImage()
        else:
            # Timed out: stop the job and bail.
            job.cancelWithoutBlocking()
            logger.warning("render_zone_to_image: render timed out after %d ms", timeout_ms)
            return None, None
    except Exception as exc:
        logger.warning("render_zone_to_image: failed for %dx%d: %s", width, height, exc)
        return None, None

    if img is None or img.isNull():
        return None, None

    logger.debug(
        "render_zone_to_image: rendered %dx%d in %d ms",
        width, height, int((time.monotonic() - t0) * 1000),
    )
    return img, actual_extent


def visible_extent_for(extent: "QgsRectangle", width: int, height: int):
    """Return the extent QGIS would actually render for (extent, width, height).

    QgsMapSettings expands the requested extent to match the output aspect ratio
    (visibleExtent). render_zone_to_image returns that expanded extent so masks
    map back to ground with zero offset. This computes the SAME value WITHOUT
    starting a render job (no basemap fetch), so the per-tile JIT path can build
    the global geo_transform up front, identically to the old full-zone render,
    before any tile is rendered.

    MAIN THREAD ONLY (QgsMapSettings construction). Returns the input extent
    unchanged on any failure, which is exact when the requested aspect already
    matches width:height (the automatic grid guarantees square pixels, so it
    does).
    """
    from qgis.core import QgsMapSettings
    from qgis.PyQt.QtCore import QSize

    try:
        settings = QgsMapSettings()
        settings.setOutputSize(QSize(int(width), int(height)))
        settings.setExtent(extent)
        return settings.visibleExtent()
    except Exception as exc:  # noqa: BLE001 - fall back to the requested extent
        logger.warning("visible_extent_for: failed (%dx%d): %s", width, height, exc)
        return extent


# The one in-flight per-tile render job (MAIN THREAD only, no locking needed).
# The JIT tile render blocks in a nested event loop, so teardown code can run
# re-entrantly inside it; tracking the live job lets that teardown cancel the
# render synchronously BEFORE the raster layer it is reading can be deleted.
_active_render_job = None


def cancel_active_tile_render() -> None:
    """Synchronously cancel the in-flight per-tile render, if any. MAIN THREAD.

    QgsMapRendererParallelJob.cancel() blocks until the render threads have
    stopped touching the layer, which closes the use-after-free window where a
    re-entrant layer removal (layersWillBeRemoved, project clear, unload)
    would free the QgsRasterLayer under an active render job. As a bonus it
    makes Cancel feel instant instead of waiting out a slow basemap render.
    Safe no-op when nothing is rendering."""
    global _active_render_job
    job = _active_render_job
    _active_render_job = None
    if job is None:
        return
    try:
        job.cancel()
    except (RuntimeError, AttributeError):
        pass


def _configure_downsample_resampling(layer, qgis_module) -> bool:
    """Configure an averaged downsample (+ smooth upsample) on a THROWAWAY raster
    layer clone. Returns True when at least one resampling path was applied.

    Prefer provider-stage resampling (the QGIS 3.16+ enum method API): GDAL then
    applies a true AVERAGE box filter on downsample, which is the correct
    antialiasing filter for a large decimation factor (a fine ortho shrunk to the
    coarse run resolution). Fall back to the raster pipe resample filter
    (bilinear out / cubic in) on bindings where the provider-stage API is
    missing. Everything is getattr-guarded so an older QGIS just returns False
    and the caller keeps the original layer.
    """
    try:
        provider = layer.dataProvider()
    except (AttributeError, RuntimeError):
        provider = None

    # Preferred: provider-stage resampling via the enum method API (3.16+).
    try:
        from qgis.core import QgsRasterDataProvider

        rm = getattr(QgsRasterDataProvider, "ResamplingMethod", None)
        stage = getattr(
            getattr(qgis_module, "RasterResamplingStage", None), "Provider", None
        )
        if (
            provider is not None
            and rm is not None
            and stage is not None
            and hasattr(provider, "enableProviderResampling")
            and hasattr(provider, "setZoomedOutResamplingMethod")
            and hasattr(provider, "setZoomedInResamplingMethod")
            and hasattr(layer, "setResamplingStage")
        ):
            # Averaged box filter for downsampling (fine -> coarse), the true
            # antialiasing choice; cubic keeps any zoomed-in read smooth. Fall
            # back to bilinear only if a build lacks these enum members.
            out_method = getattr(rm, "Average", None)
            if out_method is None:
                out_method = getattr(rm, "Bilinear", None)
            in_method = getattr(rm, "Cubic", None)
            if in_method is None:
                in_method = getattr(rm, "Bilinear", None)
            provider.enableProviderResampling(True)
            if out_method is not None:
                provider.setZoomedOutResamplingMethod(out_method)
            if in_method is not None:
                provider.setZoomedInResamplingMethod(in_method)
            if hasattr(provider, "setMaxOversampling"):
                provider.setMaxOversampling(2.0)
            layer.setResamplingStage(stage)
            return True
    except (AttributeError, RuntimeError, TypeError):
        pass

    # Fallback: the raster pipe resample filter with resampler objects, present
    # on very old QGIS where the provider-stage API does not exist.
    try:
        from qgis.core import QgsBilinearRasterResampler, QgsCubicRasterResampler

        resample_filter = layer.resampleFilter() if hasattr(layer, "resampleFilter") else None
        if resample_filter is not None:
            resample_filter.setZoomedOutResampler(QgsBilinearRasterResampler())
            resample_filter.setZoomedInResampler(QgsCubicRasterResampler())
            return True
    except (AttributeError, RuntimeError, TypeError):
        pass

    return False


def _local_raster_render_clone(layer: "QgsRasterLayer"):
    """Return a private, resampling-configured CLONE of a local file-based GDAL
    raster, or None when the layer is not an eligible local raster (or the clone
    fails). The caller renders detection tiles through this clone so the user's
    own on-screen layer, style and resampling are never touched.

    Why: a fine local ortho (say 5-8 cm/px) rendered at the run's coarser ground
    resolution (~0.30-0.45 m/px) is decimated NEAREST by the GDAL provider by
    default, which aliases object edges and starves the model of the cleanly
    averaged pixels a resampled XYZ/WMS basemap already delivers. Configuring an
    averaged downsample on a throwaway clone recovers that edge detail with no
    visible change for the user. Online providers (wms/xyz/wcs/arcgis*) already
    serve pyramid-resampled tiles, so they are deliberately left untouched.

    Best-effort: any failure returns None and the caller falls back to the
    original layer, so a missing API on an older QGIS never breaks the render.
    The clone is never added to the project; the caller keeps it referenced only
    for the duration of the render job and lets it be freed afterwards.
    """
    from qgis.core import QgsRasterLayer

    # Only local, file-based GDAL rasters benefit; every online provider already
    # delivers resampled tiles, so that path must stay byte-for-byte unchanged.
    try:
        if not isinstance(layer, QgsRasterLayer) or layer.providerType() != "gdal":
            return None
    except (AttributeError, RuntimeError):
        return None

    clone = None
    try:
        clone = layer.clone()
    except (AttributeError, RuntimeError):
        clone = None
    if clone is None:
        # Older bindings without clone(): rebuild from the source and copy the
        # renderer so band selection / contrast (WYSIWYG) is preserved.
        try:
            rebuilt = QgsRasterLayer(layer.source(), layer.name(), "gdal")
            if not rebuilt.isValid():
                return None
            renderer = layer.renderer()
            if renderer is not None:
                rebuilt.setRenderer(renderer.clone())
            clone = rebuilt
        except (AttributeError, RuntimeError, TypeError):
            return None

    try:
        if clone is None or not clone.isValid():
            return None
    except (AttributeError, RuntimeError):
        return None

    from qgis.core import Qgis

    if not _configure_downsample_resampling(clone, Qgis):
        return None
    return clone


def render_tile_qimage(
    layer: "QgsRasterLayer",
    tile_extent: "QgsRectangle",
    width: int,
    height: int,
    timeout_ms: int = 60000,
):
    """Render ONE tile's ground sub-extent to a width x height QImage.

    Same QgsMapSettings recipe as render_zone_to_image (layer, destinationCrs,
    output size, extent, parallel job, wait on a local QEventLoop) but for a
    single tile rectangle instead of the whole zone. Because the per-tile extent
    is the tile's bbox_native (derived from the global geo_transform) and the
    output size is the tile's (tw, th) at the SAME ground-per-pixel as the zone,
    the rendered pixels are identical to slicing the same tile out of one big
    zone render: same destination CRS, same map units per pixel, same origin.

    MAIN THREAD ONLY (QgsMapRendererParallelJob requires the GUI thread). The
    AutoDetectionWorker calls this on the main thread via a bridge handshake, so
    the heavy basemap fetch happens just-in-time per tile, overlapped with the
    in-flight detections, instead of blocking on the whole zone up front.

    Args:
        layer:      Raster layer to render (local or online: XYZ/WMS/WMTS).
        tile_extent: The tile's bbox_native as a QgsRectangle, in the layer CRS.
        width:      Tile width in pixels (tw).
        height:     Tile height in pixels (th).
        timeout_ms: Hard cap so a stalled network render cannot hang forever.

    Returns:
        A QImage on success, or None on timeout / failure.
    """
    from qgis.core import QgsMapRendererParallelJob, QgsMapSettings
    from qgis.PyQt.QtCore import QEventLoop, QSize, QTimer
    from qgis.PyQt.QtGui import QColor

    global _active_render_job

    if width <= 0 or height <= 0:
        logger.warning("render_tile_qimage: invalid dimensions %dx%d", width, height)
        return None

    try:
        settings = QgsMapSettings()
        settings.setOutputSize(QSize(int(width), int(height)))
        settings.setExtent(tile_extent)
        # Render a fine LOCAL raster through a resampling-configured clone so the
        # coarse detection-resolution tile is cleanly averaged, not decimated
        # nearest (which aliases object edges). Best-effort: None -> original
        # layer, and online providers are never cloned. The clone MUST outlive
        # the render job, so render_clone stays referenced here (never added to
        # the project) until the QImage is extracted below, then is freed.
        render_clone = _local_raster_render_clone(layer)
        render_layer = render_clone if render_clone is not None else layer
        settings.setLayers([render_layer])
        settings.setDestinationCrs(layer.crs())
        settings.setBackgroundColor(QColor(0, 0, 0))
        _set_quality_render_flags(settings)

        job = QgsMapRendererParallelJob(settings)
        loop = QEventLoop()
        job.finished.connect(loop.quit)
        QTimer.singleShot(timeout_ms, loop.quit)
        # Publish the job so a teardown running re-entrantly INSIDE loop.exec()
        # (layer removal, project clear, unload, Cancel) can cancel it
        # synchronously before the rendered layer can be freed underneath it.
        _active_render_job = job
        try:
            job.start()
            loop.exec()
        finally:
            if _active_render_job is job:
                _active_render_job = None

        if not job.isActive():
            img = job.renderedImage()
        else:
            job.cancelWithoutBlocking()
            logger.warning("render_tile_qimage: render timed out after %d ms", timeout_ms)
            return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("render_tile_qimage: failed for %dx%d: %s", width, height, exc)
        return None

    if img is None or img.isNull():
        return None
    return img


def tile_is_blank_array(
    arr: np.ndarray, dominant_frac: float = _BLANK_TILE_DOMINANT_FRAC
) -> bool:
    """True when a small RGB sample is essentially a single colour (nodata /
    uniform fill), using a coarse dominant-bucket test.

    NOT a low-variance / low-texture test: quantizing to coarse buckets and
    requiring one bucket to cover ``dominant_frac`` of the pixels keeps a real,
    textured scene (even a plain field or calm water) from being culled, while
    a perfectly flat render (nodata, mosaic gap, ocean pad) is caught.

    Args:
        arr: (H, W, 3+) uint8-ish array. Extra channels beyond RGB are ignored.
        dominant_frac: fraction the top bucket must reach to call it blank.

    Returns:
        True only when the sample is confidently uniform; False for anything it
        cannot classify (so it never over-skips).
    """
    if arr is None or arr.ndim != 3 or arr.shape[2] < 3 or arr.size == 0:
        return False
    rgb = arr[:, :, :3].astype(np.int64)
    q = rgb // _BLANK_TILE_QUANT
    # Pack the three quantized channels into one integer per pixel.
    packed = (q[:, :, 0] << 16) | (q[:, :, 1] << 8) | q[:, :, 2]
    flat = packed.reshape(-1)
    if flat.size == 0:
        return False
    counts = np.unique(flat, return_counts=True)[1]
    dominant = int(counts.max())
    return (dominant / float(flat.size)) >= dominant_frac


def tile_is_blank(img) -> bool:
    """True when a rendered tile QImage is essentially a single colour, so it
    can be skipped before submit and never billed. Cheap: downsamples to a
    small sample first. Reentrant (QImage), so safe on the worker thread.
    Returns False on any conversion failure (never over-skips)."""
    try:
        from qgis.PyQt.QtCore import QSize, Qt
        from qgis.PyQt.QtGui import QImage

        if img is None or img.isNull():
            return False
        small = img.scaled(
            QSize(_BLANK_TILE_SAMPLE_PX, _BLANK_TILE_SAMPLE_PX),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        ).convertToFormat(QImage.Format.Format_RGB32)
        w, h = small.width(), small.height()
        if w <= 0 or h <= 0:
            return False
        ptr = small.bits()
        ptr.setsize(h * w * 4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)
        # Qt BGRA byte order -> RGB.
        rgb = arr[:, :, [2, 1, 0]]
        return tile_is_blank_array(rgb)
    except Exception as exc:  # noqa: BLE001 - a failed check must never skip a tile
        logger.debug("tile_is_blank: check failed: %s", exc)
        return False


def encode_tile_png(
    img,
    tx: int,
    ty: int,
    tw: int,
    th: int,
) -> tuple[tuple[int, int, int, int], bytes] | None:
    """Encode one tile sub-rectangle of a rendered zone QImage to JPEG bytes.

    Pure image work (QImage.copy + JPEG encode), no QGIS layer/provider access.
    QImage and QBuffer are reentrant, so this is safe to call from a worker
    thread (the AutoDetectionWorker does exactly that): the encode is kept off
    the GUI thread and interleaved with the per-tile network waits, instead of
    blocking the UI up front during "Preparing tiles".

    Tiles ship as JPEG (see _TILE_IMAGE_FORMAT / _TILE_JPEG_QUALITY): ~4-8x
    smaller than PNG over the wire, which is the dominant cost per tile on slow
    connections since the worker uploads tiles serially.

    Args:
        img:            The QImage returned by render_zone_to_image.
        tx, ty, tw, th: Tile pixel rectangle from the tile grid.

    Returns:
        ((tx, ty, cw, ch), image_bytes) with the rectangle clamped to the image
        bounds, or None if the rectangle is empty / encoding produced no bytes.
    """
    from qgis.PyQt.QtCore import QBuffer, QRect

    from .qt_compat import WriteOnly

    # Clamp to the image bounds (the last tile may run a hair past due to
    # snapping); QImage.copy pads with transparent pixels otherwise.
    cw = min(tw, img.width() - tx)
    ch = min(th, img.height() - ty)
    if cw <= 0 or ch <= 0:
        return None
    sub = img.copy(QRect(tx, ty, cw, ch))
    buf = QBuffer()
    buf.open(WriteOnly)
    sub.save(buf, _TILE_IMAGE_FORMAT, _TILE_JPEG_QUALITY)
    data = bytes(buf.data())
    buf.close()
    if not data:
        return None
    return (tx, ty, cw, ch), data


def composite_tile_with_stamps(img, tx, ty, tw, th, stamps):
    """Encode one tile, with reference-example crops STAMPED in a row along its
    top edge (composite-per-tile). The cloud model's exemplars are same-image only, so to make
    one drawn example work on every tile we paste its natural-context crop into
    each tile and point the example box at the OBJECT inside the pasted pixels.

    The pasted crop carries a little surrounding context on purpose (the
    model's attention and pooling both use the region around the
    box as signal for boundaries and scale), but the example box we SEND is tight
    to the drawn object so the object dominates the pooled feature, not its
    surroundings.

    Args:
        img:    The whole-zone QImage (full detail resolution).
        tx,ty,tw,th: Tile pixel rectangle.
        stamps: list of ``(crop QImage, label int, obj_box)`` already sized for
                stamping, where ``obj_box`` is ``[x0,y0,x1,y1]`` in crop-pixel
                coords framing the drawn object (or None to use the whole crop).
                A plain ``(crop, label)`` pair is also accepted (whole-crop box).

    Returns ``((tx, ty, cw, ch), jpeg_bytes, exemplar_boxes, stamp_norm)`` where
    ``exemplar_boxes`` are ``[{"box":[x0,y0,x1,y1], "label":1|0}]`` in TILE pixel
    coords and ``stamp_norm`` is ``[0,0,nx,ny]`` (normalized) covering the stamp
    region so detections that land on it (the example itself) can be dropped.
    Returns None on an empty/failed tile.
    """
    from qgis.PyQt.QtCore import QBuffer, QRect

    from .qt_compat import WriteOnly
    from qgis.PyQt.QtGui import QPainter

    cw = min(tw, img.width() - tx)
    ch = min(th, img.height() - ty)
    if cw <= 0 or ch <= 0:
        return None
    sub = img.copy(QRect(tx, ty, cw, ch))

    boxes: list[dict] = []
    max_x = 0
    max_y = 0
    pad = 3
    if stamps:
        painter = QPainter(sub)
        # Stamps go in a HORIZONTAL row along the TOP edge (wrapping to a second
        # row only if it overflows the width). Combined with the paste-size cap
        # (< tile overlap, see _prepare_stamps) the whole stamp band stays inside
        # the vertical overlap: the ground it hides is always seen clean by the
        # tile above, so dropping stamp-region detections never leaves holes.
        # The old vertical stack grew past the overlap from the second stamp on.
        x = pad
        y = pad
        row_h = 0
        for stamp in stamps:
            if len(stamp) == 3:
                crop, label, obj_box = stamp
            else:
                crop, label = stamp
                obj_box = None
            sw = crop.width()
            sh = crop.height()
            # Wrap to a new row when the row would overflow the tile width.
            if x + sw + pad > cw and x > pad:
                y = y + row_h + pad
                x = pad
                row_h = 0
            if y + sh > ch:  # no room: skip this stamp on this tile
                continue
            painter.drawImage(QRect(x, y, sw, sh), crop)
            # Send a box tight to the object (obj_box) offset by the paste
            # position, clamped to the pasted crop. The whole pasted crop still
            # provides the surrounding context; only the box is tightened.
            if obj_box is not None and len(obj_box) == 4:
                bx0 = x + max(0.0, min(float(obj_box[0]), sw))
                by0 = y + max(0.0, min(float(obj_box[1]), sh))
                bx1 = x + max(0.0, min(float(obj_box[2]), sw))
                by1 = y + max(0.0, min(float(obj_box[3]), sh))
                if bx1 - bx0 < 1 or by1 - by0 < 1:  # degenerate: fall back
                    bx0, by0, bx1, by1 = x, y, x + sw, y + sh
            else:
                bx0, by0, bx1, by1 = x, y, x + sw, y + sh
            boxes.append({
                "box": [float(bx0), float(by0), float(bx1), float(by1)],
                "label": int(label),
            })
            row_h = max(row_h, sh)
            max_x = max(max_x, x + sw)
            max_y = max(max_y, y + sh)
            x += sw + pad
        painter.end()

    stamp_norm = None
    if boxes:
        stamp_norm = [
            0.0, 0.0,
            min(1.0, (max_x + pad) / cw),
            min(1.0, (max_y + pad) / ch),
        ]

    buf = QBuffer()
    buf.open(WriteOnly)
    sub.save(buf, _TILE_IMAGE_FORMAT, _TILE_JPEG_QUALITY)
    data = bytes(buf.data())
    buf.close()
    if not data:
        return None
    return (tx, ty, cw, ch), data, boxes, stamp_norm


def tile_png_to_base64(image_bytes: bytes) -> str:
    """Encode encoded-image bytes (JPEG today) to base64 (no data-URI prefix).

    Format-agnostic: the backend decodes whatever PIL recognizes, so this just
    base64s the bytes encode_tile_png produced regardless of container format.
    """
    return base64.b64encode(image_bytes).decode("ascii")


def decode_detection_response(
    response: dict,
    tile_w: int,
    tile_h: int,
    score_threshold: float = 0.0,
) -> list[tuple[np.ndarray, float, list[float]]]:
    """Decode the completed status response into (mask, score, box) tuples.

    The server response contains a "masks" list where each entry has:
      - "rle":   space-separated "offset count" string (1-based, row-major)
      - "score": float confidence
      - "box":   [cx, cy, w, h] NORMALIZED (YOLO style) -- not pixel coords

    When response["width"] or ["height"] is None (the server may omit them),
    the caller-supplied tile_w / tile_h are used for RLE decoding.

    Args:
        response:        Completed status dict with a "masks" list.
        tile_w:          Actual tile pixel width (used when server omits dimensions).
        tile_h:          Actual tile pixel height.
        score_threshold: Masks with score < threshold are discarded.

    Returns:
        List of (mask, score, box) where mask is bool (H, W), score is float,
        box is [cx, cy, w, h] normalized (may be [0,0,0,0] if server omits it).
    """
    raw_masks = response.get("masks") or []
    if not isinstance(raw_masks, list):
        logger.warning("decode_detection_response: 'masks' is not a list; returning []")
        return []

    # Prefer server-reported dimensions; fall back to caller-supplied tile dims.
    srv_w = response.get("width")
    srv_h = response.get("height")
    decode_w = int(srv_w) if srv_w is not None else tile_w
    decode_h = int(srv_h) if srv_h is not None else tile_h

    results: list[tuple[np.ndarray, float, list[float]]] = []
    for entry in raw_masks:
        if not isinstance(entry, dict):
            continue
        score = float(entry.get("score", 0.0))
        if score < score_threshold:
            continue
        rle = entry.get("rle", "")
        mask = decode_rle_to_mask(rle, decode_h, decode_w)
        raw_box = entry.get("box")
        if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
            box = [float(v) for v in raw_box]
        else:
            box = [0.0, 0.0, 0.0, 0.0]
        results.append((mask, score, box))

    return results
