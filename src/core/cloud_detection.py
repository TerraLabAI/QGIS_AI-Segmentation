"""RLE decoding and tile image encoding for Pro automatic detection mode."""
from __future__ import annotations

import base64
import logging
import time
from typing import TYPE_CHECKING

# Automatic is a cloud mode, but it still decodes masks and encodes tiles in
# process, so it needs the venv's numpy. polygon_exporter is the only other
# module that sets the venv path up before importing numpy, and nothing on the
# Automatic path imports it, so a session that only ever used Automatic reached
# the import below with the venv still off sys.path.
try:
    from .venv_manager import ensure_venv_packages_available
except ImportError:
    # venv_manager needs qgis. Outside QGIS (tests, tooling) there is no venv
    # to add and numpy is already importable, so there is nothing to set up.
    pass
else:
    ensure_venv_packages_available()

import numpy as np  # noqa: E402

from .review_defaults import HOLE_NOISE_CEILING_M  # noqa: E402

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
# size choice, and JPEG stays a fraction of lossless PNG for aerial imagery.
# Upload bytes dominate the per-tile time on slow connections, so the encode
# quality is a fidelity/bandwidth trade-off. Detection scores sitting near the
# review confidence cutoff are sensitive to compression noise, so the default
# stays near-lossless (Qt's JPEG writer keeps 4:4:4 chroma from quality 90 up);
# the server policy can lower it fleet-wide for bandwidth without a release
# (seed.tile_jpeg_quality, read at encode time by _tile_jpeg_quality).
_TILE_IMAGE_FORMAT: str = "JPEG"
_TILE_JPEG_QUALITY: int = 98


def _tile_jpeg_quality() -> int:
    """Encode quality resolved from the cached server policy when present, else
    the built-in default. Cache-only (no network), safe on any thread; fails to
    the constant so offline and older servers are unaffected."""
    try:
        from .detection_policy import tile_jpeg_quality  # noqa: PLC0415
        return tile_jpeg_quality(_TILE_JPEG_QUALITY)
    except Exception:  # noqa: BLE001 - policy is optional, never break encode
        return _TILE_JPEG_QUALITY


# Reference-example ("exemplar") crops are pasted in ONE horizontal row along one
# edge of a tile (composite_tile_with_stamps). The whole band MUST stay inside the
# vertical tile overlap, so the ground it hides is always re-seen clean by the
# neighbouring row (stamp-region detections are dropped). A normal tile bands its
# TOP edge, whose ground is re-seen by the tile above. The FIRST grid row has no
# tile above, so when a row below overlaps enough the top row bands its BOTTOM
# edge instead (the mirror case, gated by top_row_bottom_stamp_ok): that ground
# then falls in the overlap and is re-seen clean by the row below. stamp_size_cap
# keeps the band inside the overlap for any exemplar count; both the worker
# (pre-sizing) and the compositor (layout) read it, so the two never drift.
_STAMP_PAD: int = 3            # gap around each pasted crop
# A pasted crop's longest side must fit inside the inter-row overlap band with its
# padding, so the ground it hides is re-seen clean by the neighbouring row. That
# is the binding constraint (it equals the band budget below); stamp_size_cap also
# caps by each crop's share of the tile width, whichever is smaller.
_STAMP_MAX: int = 195


def should_paste_stamp(
    run_scale_side: float, cap: int, has_prompt: bool,
    min_scale: float = 0.85,
) -> bool:
    """Decide whether an example crop is worth PASTING into tiles.

    ``run_scale_side`` is the crop's long side (px) when rendered at the run's
    tile resolution (true apparent scale); ``cap`` is the band budget
    (stamp_size_cap). A paste that would shrink the crop below ``min_scale``
    of true scale shows the object smaller than the tile's real objects, and
    the model matches by apparent scale, so with a prompt present the paste
    is skipped (the exemplar still counts in-situ on tiles that fully contain
    it). Without a prompt the paste is the run's only query, so it always
    goes through (a shrunk reference beats an empty query). Excludes follow
    the same rule: a wrong-scale negative misleads the same matching.
    """
    if not has_prompt:
        return True
    if run_scale_side <= 0 or cap <= 0:
        return True
    return cap >= run_scale_side * min_scale


def _stamp_pad() -> int:
    """Crop padding resolved from the cached server policy when present, else
    the built-in default. Cache-only (no network), safe on any thread; fails to
    the constant so offline and older servers are unaffected."""
    try:
        from .detection_policy import exemplar_stamp_pad_px  # noqa: PLC0415
        return exemplar_stamp_pad_px(_STAMP_PAD)
    except Exception:  # noqa: BLE001 - policy is optional, never break layout
        return _STAMP_PAD


def _stamp_max() -> int:
    """Crop size cap resolved the same way as _stamp_pad."""
    try:
        from .detection_policy import exemplar_stamp_max_px  # noqa: PLC0415
        return exemplar_stamp_max_px(_STAMP_MAX)
    except Exception:  # noqa: BLE001 - policy is optional, never break layout
        return _STAMP_MAX


def stamp_size_cap(n: int) -> int:
    """Longest-side pixel cap for ``n`` exemplar crops sharing one band row.

    Two ceilings keep the band inside the overlap for any ``n``: its height must
    fit the vertical overlap band (minus padding), and ``n`` crops must fit the
    tile width. The binding (smallest) ceiling wins; the band budget binds for
    every supported count and the width share only bites at larger ``n``.

    The size cap and the padding are server dials (``exemplar.stamp_max_px``,
    ``exemplar.stamp_pad_px``) resolved HERE rather than by the callers, so the
    worker's pre-sizing and the compositor's layout can never read two
    different values.
    """
    from .tile_manager import OVERLAP_FRACTION, TILE_SIZE

    n = max(1, int(n))
    pad = _stamp_pad()
    overlap_px = int(TILE_SIZE * OVERLAP_FRACTION)
    band_budget = overlap_px - 2 * pad
    width_budget = (TILE_SIZE - pad) // n - pad
    return max(1, min(_stamp_max(), band_budget, width_budget))


def in_situ_exemplar_box(
    full_box, tx: int, ty: int, tw: int, th: int,
) -> list[float] | None:
    """Tile-local pixel box for an exemplar this tile FULLY contains, else None.

    ``full_box`` is the exemplar's drawn box in FULL-image pixel coords (xyxy,
    UNCLAMPED). On the one tile whose grid rect contains the whole box, the
    worker sends the example box pointing at the REAL in-situ object instead of
    at a pasted crop (and skips that crop's paste there), so the model sees the
    object in its true context and fewer real pixels are occluded. A box that
    straddles a tile edge, or overflows the zone image, must NOT be sent
    in-situ (a partial view is not a faithful reference): this returns None and
    the caller keeps the pasted-stamp path for that exemplar on that tile. The
    mapped box is clamped to the tile and rejected when degenerate (< 1 px on a
    side).
    """
    if not full_box or len(full_box) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in full_box)
    except (TypeError, ValueError):
        return None
    # Full containment in the tile's grid rect; an unclamped box overflowing
    # the zone image fails this for every tile, by construction.
    if not (x0 >= tx and y0 >= ty and x1 <= tx + tw and y1 <= ty + th):
        return None
    bx0 = min(max(x0 - tx, 0.0), float(tw))
    by0 = min(max(y0 - ty, 0.0), float(th))
    bx1 = min(max(x1 - tx, 0.0), float(tw))
    by1 = min(max(y1 - ty, 0.0), float(th))
    if bx1 - bx0 < 1.0 or by1 - by0 < 1.0:
        return None
    return [bx0, by0, bx1, by1]


def region_exemplar_box(
    full_box, tx: int, ty: int, tw: int, th: int, min_side: float = 8.0,
) -> list[float] | None:
    """Tile-local pixel box for a REGION marker, clipped to this tile.

    A region exemplar (a review correction box) flags an AREA to look at
    again, not the outline of one object, so a partial view keeps its meaning:
    unlike ``in_situ_exemplar_box`` it does not require full containment. The
    intersection with the tile's grid rect is returned whenever both clipped
    sides reach ``min_side`` px (below that the sliver carries no usable
    signal); None otherwise.
    """
    if not full_box or len(full_box) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in full_box)
    except (TypeError, ValueError):
        return None
    bx0 = min(max(x0 - tx, 0.0), float(tw))
    by0 = min(max(y0 - ty, 0.0), float(th))
    bx1 = min(max(x1 - tx, 0.0), float(tw))
    by1 = min(max(y1 - ty, 0.0), float(th))
    if bx1 - bx0 < min_side or by1 - by0 < min_side:
        return None
    return [bx0, by0, bx1, by1]


def top_row_bottom_stamp_ok(
    top_ty: int, top_th: int, next_ty: int, band_content_h: int,
    pad: int = 0,
) -> bool:
    """Whether the TOP grid row can stamp its band on the BOTTOM edge safely.

    The top row has no row above to re-see its top-band ground clean, so it may
    band its bottom edge instead and let the row BELOW re-see that ground. That
    only works when the row below does not itself band the same strip: the top
    row's bottom band and the next row's top band must not cover shared ground.
    Bottom band ground is ``[top_ty + top_th - pad - band_content_h, ...)``; the
    next row's top band ground is ``[next_ty, next_ty + pad + band_content_h)``.
    They are clear of each other exactly when the inequality below holds; when the
    overlap is too small they would hide the same strip, so the top row keeps its
    top band (the top-edge residual is unchanged, never traded for an interior
    hole). ``band_content_h`` is the tallest stamp, an upper bound on the band.
    ``pad`` <= 0 resolves the same padding stamp_size_cap uses, so this guard
    and the layout can never disagree.
    """
    if pad <= 0:
        pad = _stamp_pad()
    return next_ty + pad + band_content_h <= top_ty + top_th - pad - band_content_h


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

# Degenerate-tile prefilter, no-data by COLOUR. _tile_render_settings paints an
# OPAQUE black background (setBackgroundColor(QColor(0, 0, 0))), so ground the
# layer does not cover comes back as RGB (0, 0, 0) at alpha 255, never as a
# transparent pixel. An alpha-only no-data rule therefore reads every tile as
# fully valid and can never fire.
#
# EXACT black is what separates no-data from dark ground, which is why the
# tolerance is 0 and not a margin. A background fill is flat black, so widening
# the tolerance adds nothing on a no-data region and starts matching deep
# water, night scenes and shadow, which carry sensor noise a fill does not.
_PREFILTER_NODATA_RGB_EPS: int = 0
# Valid (non-no-data) pixel count under which a tile provably cannot produce a
# shipped object, so it settles as empty with no request. The pipeline's own
# noise floor is MIN_SIZE_NOISE_MASK_PX (3) mask pixels across, and a detection
# smaller than that is dropped as noise at ANY review setting, so a valid region
# under 3 x 3 pixels cannot survive whatever the user does with the Min size
# spinbox. Deliberately the provable bound and nothing wider: a sliver of real
# ground along a footprint edge still returns real objects, so nothing supports
# a cut above the floor. The server can raise it once evidence exists.
_PREFILTER_MIN_VALID_PX: float = 9.0

# Unavailable-imagery skip. An online tile source that holds no image for a
# tile does not fail the request: it answers with a placeholder card, a flat
# neutral-grey tile carrying a line of grey text. Those pixels reach the render
# like any others, so nothing downstream can tell them from ground, and the
# blank test above misses them because the text breaks the single-bucket rule.
# The signature is the combination the blank test cannot use on its own: every
# pixel neutral (R, G and B within a few levels of each other) AND one coarse
# bucket still covering most of the tile. Real imagery fails one or the other,
# because an aerial scene is neither perfectly neutral nor that flat. Applied
# to ONLINE sources only (the caller decides), so a grey local ortho is never
# read as a load failure.
_UNAVAILABLE_NEUTRAL_EPS: int = 6
_UNAVAILABLE_NEUTRAL_FRAC: float = 0.98
_UNAVAILABLE_DOMINANT_FRAC: float = 0.80

logger = logging.getLogger(__name__)


def decode_rle_to_mask(rle: str | dict, height: int, width: int,
                       strict: bool = False) -> np.ndarray:
    """Decode one mask RLE from the backend response to a boolean numpy array.

    RLE format: space-separated "offset count" pairs, offsets 1-based, row-major.
    See the _RLE_FORMAT constant.

    Args:
        rle:    RLE string from the backend ("offset count offset count ...").
                An empty or whitespace-only string is the encoder's legal form
                for an all-background mask and decodes to the all-zero grid in
                both modes. A dict argument is not supported by this
                implementation; if a dict is received the function logs a
                warning and returns an empty mask.
        height: Tile height in pixels (the actual tile, not padded).
        width:  Tile width in pixels.
        strict: Raise ValueError instead of answering with a mask the encoding
                did not describe. That covers an unusable argument, a token
                that is not a number, a dangling run token, an offset below 1,
                a non-positive count and a run reaching past the end of the
                mask. A caller that shows the answer to a user must set this:
                an empty or half-set mask reads on screen as a modelling
                result, so a changed encoding would never be reported. Off by
                default, because a tile that legitimately carries nothing is
                normal and must stay cheap.

    Returns:
        Boolean numpy array of shape (height, width). True = foreground.
    """
    # Allocate the boolean grid directly: a uint8 buffer plus astype(bool) is two
    # full HxW allocations per mask, and a dense tile decodes over a hundred masks.
    flat = np.zeros(height * width, dtype=bool)

    if isinstance(rle, dict):
        if strict:
            raise ValueError("mask encoding is not readable")
        logger.warning(
            "decode_rle_to_mask: received dict RLE (unsupported format); "
            "returning empty mask for tile %dx%d",
            width,
            height,
        )
        return flat.reshape((height, width))

    if not isinstance(rle, str):
        if strict:
            raise ValueError("mask encoding is not readable")
        return flat.reshape((height, width))

    if not rle.strip():
        # The server's own encoder emits an empty string for an all-background
        # mask, so this is a described all-zero grid, valid in both modes.
        return flat.reshape((height, width))

    tokens = rle.split()
    total = height * width

    if strict and len(tokens) % 2:
        raise ValueError("mask encoding ends on a run with no count")

    pairs = _rle_pairs(tokens)
    if pairs is None:
        if strict:
            raise ValueError("mask encoding carries a token that is not a number")
        # A token is not an integer: walk the pairs one by one so the log names
        # the bad pair instead of just counting it.
        _apply_rle_pairs_slow(flat, tokens, total)
    else:
        offsets, counts = pairs
        idx = offsets - 1
        bad = (idx < 0) | (counts <= 0)
        if bad.any():
            if strict:
                raise ValueError("mask encoding carries a malformed run")
            logger.warning(
                "decode_rle_to_mask: %d malformed run pair(s) skipped "
                "(offset below 1 or non-positive count)",
                int(bad.sum()),
            )
        ends = idx + counts
        over = ends > total
        if over.any():
            if strict:
                raise ValueError("mask encoding carries a run past the end of the mask")
            logger.warning(
                "decode_rle_to_mask: %d run(s) exceed mask size %d; clipping",
                int(over.sum()), total,
            )
            ends = np.minimum(ends, total)
        keep = ~bad
        for start, end in zip(idx[keep].tolist(), ends[keep].tolist()):
            flat[start:end] = True

    return flat.reshape((height, width))


def _rle_pairs(tokens: list[str]):
    """Parse the RLE tokens into (offsets, counts) int64 arrays in one numpy
    pass. Returns None when a token is not an integer, so the caller can fall
    back to the per-pair loop that reports which pair is malformed."""
    n = (len(tokens) // 2) * 2
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    try:
        nums = np.asarray(tokens[:n], dtype=np.int64)
    except (ValueError, TypeError, OverflowError):
        return None
    return nums[0::2], nums[1::2]


def _apply_rle_pairs_slow(flat: np.ndarray, tokens: list[str], total: int) -> None:
    """Set the runs of a token list that failed the vectorized parse, reporting
    each malformed pair by index."""
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

        flat[idx:end] = True


# ---------------------------------------------------------------------------
# Vectorization scale: parameters that live on the returned mask's pixel grid
# ---------------------------------------------------------------------------
# The service reports each mask's own width/height and may return masks on a
# COARSER grid than the uploaded tile. Any per-pixel vectorization parameter
# (the staircase simplify tolerance, the pinhole-fill ceiling) must then be
# keyed to the mask's own grid cell, not the run's native pixel: a tolerance
# below the actual staircase step cannot remove the stairs. CONSTRAINT: when
# the mask grid matches the native pixel these derivations must reduce
# BIT-EXACTLY to the historical native-gsd formulas, so already-shipped runs
# never change geometry. The slack factor keeps float noise in the
# ground-size / pixel-count divisions from ever flipping that equivalence;
# only a genuinely coarser grid takes over.
_MASK_CELL_SLACK: float = 1.000001


def mask_cell_size(
    ground_w: float, ground_h: float, mask_w: int, mask_h: int
) -> float:
    """Ground size of ONE cell of a returned mask's own pixel grid.

    This is the staircase step of any polygon vectorized from that mask. The
    max of the two axes is used so an anisotropic ratio never understates the
    step. Returns 0.0 for degenerate inputs (no mask pixels / no ground size),
    which every consumer treats as "grid unknown, use the native pixel".
    """
    if mask_w <= 0 or mask_h <= 0 or ground_w <= 0:
        return 0.0
    return max(ground_w / float(mask_w), ground_h / float(mask_h))


def _vector_step(native_gsd: float, mask_cell: float) -> float:
    """The grid step vectorization parameters scale with: the mask's own cell
    when it is genuinely coarser than the native pixel, else the native pixel
    exactly (see _MASK_CELL_SLACK for why not a plain max)."""
    if mask_cell > native_gsd * _MASK_CELL_SLACK:
        return mask_cell
    return native_gsd


# Multiple of the grid step used as the staircase simplify tolerance: enough to
# strip the pixel staircase without moving a true corner by more than a cell.
# Client fallback for the server policy's `review.tile_simplify_mult`.
_TILE_SIMPLIFY_MULT: float = 0.75
# Ground size (metres across) of the largest interior hole treated as a fillable
# pinhole. Client fallback for `review.pinhole_m`. Imported rather than repeated:
# Manual's own Fill-holes ceiling starts at the same judgement, and two copies of
# "this hole is noise" that can drift is exactly how a courtyard gets deleted on
# one path and kept on the other.
_PINHOLE_GROUND_M: float = HOLE_NOISE_CEILING_M


def tile_simplify_tolerance(
    native_gsd: float, mask_cell: float = 0.0, mult: float = 0.0
) -> float:
    """Simplify tolerance (ground units) for polygonizing one returned mask.

    ``mult`` times the vector step. Keyed to the MASK grid so a coarser
    returned grid is de-staircased at its true step; identical to the
    historical multiple of the native gsd whenever the mask grid matches the
    tile. 0.0 (no simplify) when the run has no metric scale.

    ``mult`` <= 0 uses the client constant. A live run passes the value it
    resolved from the server policy ONCE at its start (see
    ``detection_policy.tile_simplify_mult``); a replay of an archived run must
    pass that run's own value, never a fresh policy read, or its geometry would
    change under it.
    """
    if native_gsd <= 0:
        return 0.0
    if mult <= 0:
        mult = _TILE_SIMPLIFY_MULT
    return mult * _vector_step(native_gsd, mask_cell)


def pinhole_fill_limit_px(
    native_gsd: float, mask_cell: float = 0.0, ground_m: float = 0.0
) -> int:
    """Max hole area (in MASK-grid pixels) that counts as a fillable pinhole.

    The pinhole ceiling is fixed in GROUND terms (texture pits fill, real
    courtyards stay, see the worker's fill site); converting it to pixels of
    the mask's own grid keeps that ground meaning when the returned grid is
    coarser than the native pixel. Identical to the historical native-pixel
    conversion when the grids match; 36 px when the run has no metric scale;
    never below 9 px (a 3x3 speck is always noise, so that floor is mechanism
    and stays client-side).

    ``ground_m`` <= 0 uses the client constant; a run passes the value it
    resolved from ``detection_policy.pinhole_fill_m`` at its start.
    """
    if native_gsd <= 0:
        return 36
    if ground_m <= 0:
        ground_m = _PINHOLE_GROUND_M
    step = _vector_step(native_gsd, mask_cell)
    return max(9, int((ground_m / step) ** 2))


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
        # Tested with `is None`, never chained with `or`: a scoped enum member
        # whose value is 0 is falsy, and an `or` chain would drop it and reach
        # for the other spelling that does not exist on that QGIS.
        flag = getattr(QgsMapSettings, name, None)
        if flag is None:
            scoped = getattr(Qgis, "MapSettingsFlag", None)
            flag = None if scoped is None else getattr(scoped, name, None)
        if flag is None:
            continue
        try:
            settings.setFlag(flag, True)
        except (TypeError, AttributeError):  # enum type mismatch on this QGIS
            pass


def _set_blocking_remote_fetch(settings) -> None:
    """Make the render job fetch remote layer data (XYZ/WMS/WMTS tiles)
    synchronously before finishing, the way QGIS exports do. Without it an
    offscreen render of an online basemap can return with EMPTY cells for
    every imagery tile the provider had not cached yet (it schedules an async
    download and a canvas refresh that never applies to a one-shot job), which
    reads as a blank/partial tile downstream. Best-effort: enum location moved
    across QGIS versions, and a missing flag must never break the render."""
    from qgis.core import Qgis, QgsMapSettings

    # `is None` rather than an `or` chain, for the reason given in
    # _set_quality_render_flags: a 0-valued enum member is falsy.
    flag = getattr(QgsMapSettings, "RenderBlocking", None)
    if flag is None:
        scoped = getattr(Qgis, "MapSettingsFlag", None)
        flag = None if scoped is None else getattr(scoped, "RenderBlocking", None)
    if flag is None:
        return
    try:
        settings.setFlag(flag, True)
    except (TypeError, AttributeError):  # enum type mismatch on this QGIS
        pass


# Hard cap on one zone render. The wait runs on the calling thread, which is
# the GUI thread, so it is sized for a slow basemap to answer and not for a
# dead one to be waited out.
_RENDER_ZONE_TIMEOUT_MS: int = 25000

# How many render waits are on this thread's stack. A caller that can be
# reached from a queued signal reads render_loop_is_active() to know it is
# running INSIDE one, where starting another render, or freeing the layer the
# running job reads, is not safe.
_render_loop_depth: int = 0


def render_loop_is_active() -> bool:
    """True while a render wait (nested event loop) is on the stack."""
    return _render_loop_depth > 0


def _exclude_user_input_flag():
    """QEventLoop's "hold back user input" flag, for Qt5 and Qt6."""
    from qgis.PyQt.QtCore import QEventLoop

    from .qt_compat import resolve_qt_enum

    return resolve_qt_enum(
        QEventLoop, "ProcessEventsFlag", "ExcludeUserInputEvents")


def render_zone_to_image(
    layer: QgsRasterLayer,
    extent: QgsRectangle,
    width: int,
    height: int,
    timeout_ms: int = _RENDER_ZONE_TIMEOUT_MS,
    resample_local: bool = False,
    render_crs=None,
):
    """Render a whole zone to one QImage, waiting on a nested event loop.

    Uses QgsMapRendererParallelJob, which renders on QGIS's own background
    thread pool and emits finished() on the main thread. The caller waits on a
    local QEventLoop, so this call still BLOCKS the caller (the whole run sits
    here) and the flag RenderBlocking makes every uncached basemap tile
    download before the loop quits. What the loop buys is repainting and timers,
    not a free thread. It holds back user input for the whole wait: a click or a
    window close delivered inside this frame would reenter the caller, and
    quitting the application from there unwinds every loop on the stack and
    returns into code whose objects are being destroyed. Callers reachable from
    a queued signal can read render_loop_is_active() to tell they are already
    inside such a wait.

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
        resample_local: Render a fine LOCAL raster through the same averaged
                     downsample clone the tile path uses, so a stamp shrunk to
                     the run resolution is smoothed, not decimated nearest, and
                     carries the SAME texture statistics as the tiles it must
                     match. No-op for online providers (already resampled).
        render_crs:  CRS to render into, which ``extent`` must be expressed in.
                     None renders in the layer's own CRS, as it always did.

    Returns:
        (QImage, QgsRectangle actual_extent) on success, or (None, None).
    """
    from qgis.core import QgsMapRendererParallelJob, QgsMapSettings, QgsProject
    from qgis.PyQt.QtCore import QEventLoop, QSize, QTimer
    from qgis.PyQt.QtGui import QColor

    if width <= 0 or height <= 0:
        logger.warning("render_zone_to_image: invalid dimensions %dx%d", width, height)
        return None, None

    t0 = time.monotonic()
    render_clone = None
    try:
        settings = QgsMapSettings()
        settings.setOutputSize(QSize(width, height))
        settings.setExtent(extent)
        # Same reason as the tile render: the destination CRS can differ from
        # the layer's, so the project's datum transforms have to travel with it.
        settings.setTransformContext(QgsProject.instance().transformContext())
        # A local raster is decimated nearest when shrunk to the run resolution
        # by default; render through the resampling clone so the stamp matches
        # the tiles' averaged texture. None (and unchanged behaviour) for online
        # providers or when the caller does not opt in.
        if resample_local:
            render_clone = _local_raster_render_clone(layer)
        render_layer = render_clone if render_clone is not None else layer
        settings.setLayers([render_layer])
        settings.setDestinationCrs(_render_crs_or_layer(layer, render_crs))
        settings.setBackgroundColor(QColor(0, 0, 0))
        _set_quality_render_flags(settings)
        # Fetch remote imagery synchronously so uncached areas render with
        # real pixels, not blank cells (see _set_blocking_remote_fetch).
        _set_blocking_remote_fetch(settings)
        # The extent QGIS will actually render (aspect-corrected). Using this for
        # the geo-transform is what keeps detections aligned with the imagery.
        actual_extent = settings.visibleExtent()

        job = QgsMapRendererParallelJob(settings)
        loop = QEventLoop()
        job.finished.connect(loop.quit)
        # Safety net: never block the UI loop indefinitely on a stalled render.
        QTimer.singleShot(timeout_ms, loop.quit)
        # Publish the job so a teardown running re-entrantly INSIDE loop.exec()
        # (layer removal, project clear, unload, Cancel) can cancel it
        # synchronously before the rendered layer can be freed underneath it,
        # exactly like the per-tile render path. Queued teardown still arrives
        # here: only user input is held back.
        _active_render_jobs.append(job)
        global _render_loop_depth
        _render_loop_depth += 1
        try:
            job.start()
            loop.exec(_exclude_user_input_flag())
        finally:
            _render_loop_depth -= 1
            if job in _active_render_jobs:
                _active_render_jobs.remove(job)

        if not job.isActive():
            img = job.renderedImage()
        else:
            # Timed out: stop the job and bail.
            job.cancelWithoutBlocking()
            logger.warning("render_zone_to_image: render timed out after %d ms", timeout_ms)
            return None, None
        # The clone (when any) backed the render layer; release it now the job
        # has finished, mirroring the tile render path's cleanup.
        del render_clone
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


def visible_extent_for(extent: QgsRectangle, width: int, height: int):
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


# The in-flight per-tile render jobs (MAIN THREAD only, no locking needed).
# The synchronous path blocks in a nested event loop, and the async prefetch
# path can hold several jobs at once; tracking every live job lets teardown
# cancel them synchronously BEFORE the raster layer they read can be deleted.
_active_render_jobs: list = []


def cancel_active_tile_render() -> None:
    """Synchronously cancel every in-flight per-tile render. MAIN THREAD.

    QgsMapRendererParallelJob.cancel() blocks until the render threads have
    stopped touching the layer, which closes the use-after-free window where a
    re-entrant layer removal (layersWillBeRemoved, project clear, unload)
    would free the QgsRasterLayer under an active render job. As a bonus it
    makes Cancel feel instant instead of waiting out a slow basemap render.
    Safe no-op when nothing is rendering."""
    jobs = list(_active_render_jobs)
    _active_render_jobs.clear()
    for job in jobs:
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
        can_resample = provider is not None
        can_resample = can_resample and rm is not None
        can_resample = can_resample and stage is not None
        can_resample = can_resample and hasattr(provider, "enableProviderResampling")
        can_resample = can_resample and hasattr(provider, "setZoomedOutResamplingMethod")
        can_resample = can_resample and hasattr(provider, "setZoomedInResamplingMethod")
        can_resample = can_resample and hasattr(layer, "setResamplingStage")
        if can_resample:
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


def _local_raster_render_clone(layer: QgsRasterLayer):
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


def _render_crs_or_layer(layer, render_crs):
    """The CRS a render targets: the run's when it is usable, else the layer's.

    The layer CRS is the answer for every run whose axes already agree on the
    ground, and it is the answer whenever the run CRS is missing or unusable,
    so a caller that passes nothing renders exactly as it always did.
    """
    try:
        if render_crs is not None and render_crs.isValid():
            return render_crs
    except (RuntimeError, AttributeError):
        pass
    return layer.crs()


def _tile_render_settings(layer, tile_extent, width: int, height: int,
                          render_clone=None, render_crs=None):
    """The QgsMapSettings recipe for one tile render.

    Returns (settings, render_clone). The clone (None for online providers)
    MUST be kept referenced until the job has finished: it backs the layer the
    job reads. Render a fine LOCAL raster through a resampling-configured
    clone so the coarse detection-resolution tile is cleanly averaged, not
    decimated nearest (which aliases object edges).

    Pass ``render_clone`` to reuse one the caller already built and owns.
    Building one reopens the GDAL dataset on the MAIN thread, which is also
    the thread that paints, so it belongs once per run and not once per tile.
    Sharing one layer object across the overlapping tile jobs is what the
    online path below already does: it takes the ``else layer`` branch and
    hands every concurrent job the user's single layer."""
    from qgis.core import QgsMapSettings, QgsProject
    from qgis.PyQt.QtCore import QSize
    from qgis.PyQt.QtGui import QColor

    settings = QgsMapSettings()
    settings.setOutputSize(QSize(int(width), int(height)))
    settings.setExtent(tile_extent)
    # The destination CRS is the run CRS, which is no longer always the layer's
    # own, so this render can now reproject for real. Without the project's
    # context that happens outside the datum transforms the user configured,
    # and the imagery is sampled from ground shifted against the coordinates
    # the detections are stamped with.
    settings.setTransformContext(QgsProject.instance().transformContext())
    if render_clone is None:
        render_clone = _local_raster_render_clone(layer)
    render_layer = render_clone if render_clone is not None else layer
    settings.setLayers([render_layer])
    settings.setDestinationCrs(_render_crs_or_layer(layer, render_crs))
    settings.setBackgroundColor(QColor(0, 0, 0))
    _set_quality_render_flags(settings)
    # Online basemaps MUST download their imagery tiles before the job
    # finishes; otherwise an uncached area (never viewed at detection
    # zoom) can render blank and the detection tile is silently skipped
    # as "empty".
    _set_blocking_remote_fetch(settings)
    return settings, render_clone


def start_tile_render_job(
    layer,
    tile_extent,
    width: int,
    height: int,
    on_done,
    timeout_ms: int = 60000,
    render_clone=None,
    render_crs=None,
) -> bool:
    """Start ONE tile render as an ASYNC job and return immediately. MAIN THREAD.

    ``on_done(img_or_None)`` is called exactly once on the main thread when the
    job finishes, times out, or is torn down by cancel_active_tile_render()
    (a cancelled job reports None). This never blocks the caller and never
    opens a nested event loop, so several tile renders overlap: each parallel
    job renders in its own threads, and the blocking remote fetches of N tiles
    run side by side instead of one after the other. Returns False when the job
    could not be started (on_done is then NOT called).
    """
    from qgis.core import QgsMapRendererParallelJob
    from qgis.PyQt.QtCore import QTimer

    if width <= 0 or height <= 0:
        logger.warning("start_tile_render_job: invalid dimensions %dx%d", width, height)
        return False
    try:
        settings, render_clone = _tile_render_settings(
            layer, tile_extent, width, height, render_clone, render_crs)
        job = QgsMapRendererParallelJob(settings)
    except Exception as exc:  # noqa: BLE001
        logger.warning("start_tile_render_job: failed for %dx%d: %s", width, height, exc)
        return False

    state = {"done": False, "clone": render_clone}

    def _finish(timed_out: bool = False) -> None:
        if state["done"]:
            return
        state["done"] = True
        if job in _active_render_jobs:
            _active_render_jobs.remove(job)
        img = None
        try:
            if not job.isActive():
                img = job.renderedImage()
            else:
                # Blocking cancel: the render threads must stop touching the
                # layer/clone before we release our reference to it.
                job.cancel()
                if timed_out:
                    logger.warning(
                        "start_tile_render_job: render timed out after %d ms", timeout_ms)
        except (RuntimeError, AttributeError):
            img = None
        state["clone"] = None
        if img is not None and img.isNull():
            img = None
        try:
            on_done(img)
        except Exception:  # noqa: BLE001 - a broken callback must not leak jobs
            logger.warning("start_tile_render_job: on_done callback failed")

    job.finished.connect(_finish)
    QTimer.singleShot(timeout_ms, lambda: _finish(timed_out=True))
    _active_render_jobs.append(job)
    try:
        job.start()
    except Exception as exc:  # noqa: BLE001
        logger.warning("start_tile_render_job: start failed: %s", exc)
        state["done"] = True
        if job in _active_render_jobs:
            _active_render_jobs.remove(job)
        return False
    return True


def tile_is_blank_array(
    arr: np.ndarray, dominant_frac: float = _BLANK_TILE_DOMINANT_FRAC,
    quant: int = _BLANK_TILE_QUANT,
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
        quant: colour quantization step (bucket width per channel).

    Returns:
        True only when the sample is confidently uniform; False for anything it
        cannot classify (so it never over-skips).
    """
    if arr is None or arr.ndim != 3 or arr.shape[2] < 3 or arr.size == 0:
        return False
    rgb = arr[:, :, :3].astype(np.int64)
    q = rgb // max(1, int(quant))
    # Pack the three quantized channels into one integer per pixel.
    packed = (q[:, :, 0] << 16) | (q[:, :, 1] << 8) | q[:, :, 2]
    flat = packed.reshape(-1)
    if flat.size == 0:
        return False
    counts = np.unique(flat, return_counts=True)[1]
    dominant = int(counts.max())
    return (dominant / float(flat.size)) >= dominant_frac


def tile_is_degenerate_array(
    arr: np.ndarray, nodata_frac: float = 1.0, band_eps: float = 2.0,
    nodata_rgb_eps: int = _PREFILTER_NODATA_RGB_EPS,
    min_valid_px: float = _PREFILTER_MIN_VALID_PX,
) -> bool:
    """True when a FULL-RESOLUTION tile array provably cannot contain an
    object, so it can settle as an empty result with no model pass at all.

    A pixel is NO-DATA when it is fully transparent (alpha channel 0) or when
    all three of its channels sit at or below ``nodata_rgb_eps``. The colour
    half is what makes the rule work at all on the tile path: the render paints
    an opaque black background, so ground the layer does not cover arrives as
    black at alpha 255 and an alpha-only test sees a fully valid tile (see
    _PREFILTER_NODATA_RGB_EPS for the measurement). Every other pixel is VALID.

    Three rules, all zero-loss by construction:

    - No-data: the no-data fraction reaches ``nodata_frac``. There is no
      imagery there to detect on.
    - Too small: fewer than ``min_valid_px`` valid pixels. Anything the model
      could return from this tile is confined to that region, and a detection
      under the pipeline's own noise floor is dropped at any review setting.
    - Uniform: over the VALID pixels, every band's full spread (max - min)
      stays within ``band_eps``. A truly uniform region has no edges, and an
      object would create at least one.

    This is deliberately NOT a low-variance / low-texture test: a calm lake
    with one boat or a plain roof carrying panels has low variance but real
    content, and must never be settled here (the broader dominant-bucket
    ``tile_is_blank`` covers the retry-then-drop path for near-uniform
    renders). Nor is it a mostly-black test: a tile can be almost all no-data
    and still hold the objects the run is looking for along the footprint edge,
    so no no-data fraction separates a barren tile from a productive one and
    none is used.

    The zero-loss claim also requires FULL-RES input: a downsampled sample can
    lose a small object entirely, so it can never prove absence.

    Args:
        arr: (H, W, 3) RGB or (H, W, 4) RGBA uint8-ish array. With 4 channels
             the last one is read as alpha; with 3 opacity is assumed.
        nodata_frac: no-data fraction at/above which the tile settles. Only
             fractions in (0, 1] are honoured (anything else disables the
             no-data rule, never loosens it).
        nodata_rgb_eps: max channel value for a pixel to count as no-data by
             colour. Negative disables the colour half (alpha only).
        min_valid_px: valid-pixel count under which the tile settles. Negative
             values are clamped to 0, which disables the rule.
        band_eps: max per-band spread (negative values are clamped to 0).

    Returns:
        True only when the tile is provably empty; False for anything the
        check cannot classify (never over-skips).
    """
    if arr is None or arr.ndim != 3 or arr.shape[2] < 3 or arr.size == 0:
        return False
    eps = max(0.0, float(band_eps))
    floor_px = max(0.0, float(min_valid_px))
    # Keep the tile's own dtype: widening a full-resolution tile to int64 costs
    # 8x its size for a max-min that only ever needs the per-band extremes. The
    # subtraction below is the one place that must not wrap, and it runs on 3
    # values, so it takes the wide type there instead.
    rgb = arr[:, :, :3].reshape(-1, 3)
    n_px = int(arr.shape[0] * arr.shape[1])
    valid = None
    if arr.shape[2] >= 4:
        valid = arr[:, :, 3].reshape(-1) != 0
    if int(nodata_rgb_eps) >= 0:
        lit = rgb.max(axis=1) > int(nodata_rgb_eps)
        valid = lit if valid is None else (valid & lit)
    if valid is not None:
        n_valid = int(valid.sum())
        frac = (n_px - n_valid) / float(n_px)
        if 0.0 < float(nodata_frac) <= 1.0 and frac >= float(nodata_frac):
            return True
        if n_valid == 0:
            return True  # nothing but no-data, whatever the threshold
        if n_valid < floor_px:
            return True  # too little ground to hold anything shippable
        rgb = rgb[valid]
    spread = np.subtract(rgb.max(axis=0), rgb.min(axis=0), dtype=np.float64)
    return bool((spread <= eps).all())


def tile_is_degenerate(
    img, nodata_frac: float = 1.0, band_eps: float = 2.0,
    nodata_rgb_eps: int = _PREFILTER_NODATA_RGB_EPS,
    min_valid_px: float = _PREFILTER_MIN_VALID_PX,
) -> bool:
    """True when a rendered tile QImage is provably empty (all no-data, too
    little ground to hold anything, or uniform in every band), per
    tile_is_degenerate_array. Unlike tile_is_blank this reads the
    FULL-resolution pixels (required for the zero-loss guarantee) including the
    alpha channel. Reentrant (QImage), safe on the worker thread; returns False
    on any conversion failure."""
    try:
        from qgis.PyQt.QtGui import QImage

        if img is None or img.isNull():
            return False
        # Unpremultiplied ARGB so alpha reads directly and RGB is unscaled.
        full = img.convertToFormat(QImage.Format.Format_ARGB32)
        w, h = full.width(), full.height()
        if w <= 0 or h <= 0:
            return False
        ptr = full.bits()
        ptr.setsize(h * full.bytesPerLine())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, full.bytesPerLine() // 4, 4)
        arr = arr[:, :w, :]
        # Qt BGRA byte order -> RGBA.
        rgba = arr[:, :, [2, 1, 0, 3]]
        return tile_is_degenerate_array(
            rgba, nodata_frac, band_eps, nodata_rgb_eps, min_valid_px)
    except Exception as exc:  # noqa: BLE001 - a failed check must never skip a tile
        logger.debug("tile_is_degenerate: check failed: %s", exc)
        return False


def _blank_tile_dials() -> tuple[int, float, int]:
    """(sample side, dominant fraction, quantization step) resolved from the
    cached server policy when present, else the built-in defaults. Cache-only
    (no network), safe on any thread; fails to the constants so offline and
    older servers skip exactly the tiles they skip today."""
    try:
        from .detection_policy import (  # noqa: PLC0415
            blank_dominant_frac,
            blank_quant,
            blank_sample_px,
        )
        return (
            blank_sample_px(_BLANK_TILE_SAMPLE_PX),
            blank_dominant_frac(_BLANK_TILE_DOMINANT_FRAC),
            blank_quant(_BLANK_TILE_QUANT),
        )
    except Exception:  # noqa: BLE001 - policy is optional, never break the check
        return (_BLANK_TILE_SAMPLE_PX, _BLANK_TILE_DOMINANT_FRAC, _BLANK_TILE_QUANT)


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
        sample_px, dominant_frac, quant = _blank_tile_dials()
        small = img.scaled(
            QSize(sample_px, sample_px),
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
        return tile_is_blank_array(rgb, dominant_frac, quant)
    except Exception as exc:  # noqa: BLE001 - a failed check must never skip a tile
        logger.debug("tile_is_blank: check failed: %s", exc)
        return False


def tile_is_unavailable_array(
    arr: np.ndarray,
    neutral_eps: int = _UNAVAILABLE_NEUTRAL_EPS,
    neutral_frac: float = _UNAVAILABLE_NEUTRAL_FRAC,
    dominant_frac: float = _UNAVAILABLE_DOMINANT_FRAC,
    quant: int = _BLANK_TILE_QUANT,
) -> bool:
    """True when a small RGB sample looks like an online source's "no image
    here" placeholder card rather than ground.

    Two conditions, both required (see the _UNAVAILABLE_* constants):

    - Neutral: at least ``neutral_frac`` of the pixels have their three
      channels within ``neutral_eps`` levels of each other. A placeholder card
      is drawn in greys; an aerial scene carries colour almost everywhere.
    - Flat: one coarse colour bucket still covers ``dominant_frac`` of the
      sample. The card's text is the only thing breaking its single colour.

    ONLINE SOURCES ONLY. A panchromatic local raster is neutral by nature, and
    the flatness rule alone is not enough to tell one from a placeholder.

    Args:
        arr: (H, W, 3+) uint8-ish array. Extra channels beyond RGB are ignored.
        neutral_eps: max spread between a pixel's three channels.
        neutral_frac: share of pixels that must be neutral.
        dominant_frac: share the top quantized bucket must reach.
        quant: colour quantization step (bucket width per channel).

    Returns:
        True only when both conditions hold; False for anything it cannot
        classify, so it never over-skips.
    """
    if arr is None or arr.ndim != 3 or arr.shape[2] < 3 or arr.size == 0:
        return False
    rgb = arr[:, :, :3].astype(np.int16)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    neutral = float((spread <= max(0, int(neutral_eps))).mean())
    if neutral < float(neutral_frac):
        return False
    return tile_is_blank_array(arr, float(dominant_frac), quant)


def _unavailable_tile_dials() -> tuple[int, int, float, float]:
    """(sample side, neutral eps, neutral fraction, dominant fraction) resolved
    from the cached server policy when present, else the built-in defaults.
    Cache-only (no network), safe on any thread; fails to the constants so an
    offline plugin and an older server skip exactly what they skip today."""
    try:
        from .detection_policy import (  # noqa: PLC0415
            blank_sample_px,
            unavailable_dominant_frac,
            unavailable_neutral_eps,
            unavailable_neutral_frac,
        )
        return (
            blank_sample_px(_BLANK_TILE_SAMPLE_PX),
            unavailable_neutral_eps(_UNAVAILABLE_NEUTRAL_EPS),
            unavailable_neutral_frac(_UNAVAILABLE_NEUTRAL_FRAC),
            unavailable_dominant_frac(_UNAVAILABLE_DOMINANT_FRAC),
        )
    except Exception:  # noqa: BLE001 - policy is optional, never break the check
        return (
            _BLANK_TILE_SAMPLE_PX,
            _UNAVAILABLE_NEUTRAL_EPS,
            _UNAVAILABLE_NEUTRAL_FRAC,
            _UNAVAILABLE_DOMINANT_FRAC,
        )


def tile_is_unavailable(img) -> bool:
    """True when a rendered tile QImage is an online source's "no image here"
    placeholder, per tile_is_unavailable_array, so the tile can be re-rendered
    and then dropped before submit instead of billed for a picture of nothing.
    Cheap: downsamples to a small sample first. Reentrant (QImage), so safe on
    the worker thread. Returns False on any conversion failure."""
    try:
        from qgis.PyQt.QtCore import QSize, Qt
        from qgis.PyQt.QtGui import QImage

        if img is None or img.isNull():
            return False
        sample_px, neutral_eps, neutral_frac, dominant_frac = _unavailable_tile_dials()
        small = img.scaled(
            QSize(sample_px, sample_px),
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
        return tile_is_unavailable_array(
            rgb, neutral_eps, neutral_frac, dominant_frac, _BLANK_TILE_QUANT)
    except Exception as exc:  # noqa: BLE001 - a failed check must never skip a tile
        logger.debug("tile_is_unavailable: check failed: %s", exc)
        return False


# Side (px) and time budget of the pre-run imagery probe. Small on purpose: it
# has to render at the RUN's ground resolution to ask the right question, but
# it only needs enough pixels to recognise a placeholder card, and the user is
# waiting on it before the run starts.
_IMAGERY_PROBE_PX: int = 256
_IMAGERY_PROBE_TIMEOUT_MS: int = 12000


def probe_imagery_availability(
    layer,
    extent,
    render_crs=None,
    width: int = _IMAGERY_PROBE_PX,
    height: int = _IMAGERY_PROBE_PX,
    timeout_ms: int = _IMAGERY_PROBE_TIMEOUT_MS,
):
    """Render one small window at the run's resolution and say what came back.

    Answers the question a run cannot ask afterwards: does this source actually
    hold a picture of this ground at this level of detail? Asked BEFORE any
    billable work, and the pixels it fetches warm the provider cache for the
    first real tile, so the cost is close to free.

    ``extent`` must already be in ``render_crs`` (the run CRS), and must span
    the ground of one probe window at the run's resolution, not the whole zone:
    a whole-zone probe renders at a coarser zoom, where a source usually does
    have imagery even when it has none at the run's.

    Returns one of:
        "ok"           - real imagery came back
        "unavailable"  - the source answered with a "no image here" card
        "blank"        - a single flat colour (nodata, out of footprint)
        "failed"       - nothing rendered (timeout, provider error)

    MAIN THREAD ONLY (it waits on a nested event loop, see render_zone_to_image).
    """
    try:
        img, _actual = render_zone_to_image(
            layer, extent, int(width), int(height),
            timeout_ms=int(timeout_ms), render_crs=render_crs)
    except Exception as exc:  # noqa: BLE001 - a probe must never block a run
        logger.debug("probe_imagery_availability: render failed: %s", exc)
        return "failed"
    if img is None:
        return "failed"
    if tile_is_unavailable(img):
        return "unavailable"
    if tile_is_blank(img):
        return "blank"
    return "ok"


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
    sub.save(buf, _TILE_IMAGE_FORMAT, _tile_jpeg_quality())
    data = bytes(buf.data())
    buf.close()
    if not data:
        return None
    return (tx, ty, cw, ch), data


def composite_tile_with_stamps(img, tx, ty, tw, th, stamps, bottom=False):
    """Encode one tile, with reference-example crops STAMPED in a single row along
    one edge (composite-per-tile). The cloud model's exemplars are same-image only, so to make
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
        bottom: paste the band along the BOTTOM edge instead of the top (the
                first-row mirror case; see top_row_bottom_stamp_ok). Either way
                the band stays inside the inter-row overlap.

    Returns ``((tx, ty, cw, ch), jpeg_bytes, exemplar_boxes, stamp_norm)`` where
    ``exemplar_boxes`` are ``[{"box":[x0,y0,x1,y1], "label":1|0}]`` in TILE pixel
    coords and ``stamp_norm`` is ``[nx0,ny0,nx1,ny1]`` (normalized): the REAL
    pasted rectangle wherever the band landed, so detections that fall on it (the
    example itself) can be dropped. Returns None on an empty/failed tile.
    """
    from qgis.PyQt.QtCore import QBuffer, QRect
    from qgis.PyQt.QtGui import QPainter

    from .qt_compat import WriteOnly

    cw = min(tw, img.width() - tx)
    ch = min(th, img.height() - ty)
    if cw <= 0 or ch <= 0:
        return None
    sub = img.copy(QRect(tx, ty, cw, ch))

    boxes: list[dict] = []
    min_x = min_y = None
    max_x = 0
    max_y = 0
    pad = _stamp_pad()
    if stamps:
        from .tile_manager import OVERLAP_FRACTION, TILE_SIZE

        # Stamps go in a SINGLE HORIZONTAL row pinned to the top edge, or the
        # bottom edge when `bottom` is set (the first grid row's mirror case).
        # The band must never reach past the vertical tile overlap, or ground
        # hidden under it is not guaranteed to be re-seen clean by the
        # neighbouring row (stamp-region detections are dropped). Stamps are
        # pre-sized (stamp_size_cap) so all of them fit one row inside this
        # budget; anything that still would not fit (only possible on a
        # degenerate narrow edge tile) is skipped here, never wrapped onto a
        # second row that would poke past the overlap.
        band_h = min(ch, int(TILE_SIZE * OVERLAP_FRACTION))
        painter = QPainter(sub)
        x = pad
        for stamp in stamps:
            if len(stamp) == 3:
                crop, label, obj_box = stamp
            else:
                crop, label = stamp
                obj_box = None
            if crop is None:  # in-situ-only exemplar: nothing to paste
                continue
            sw = crop.width()
            sh = crop.height()
            # Row full: stop (never wrap down past the overlap band).
            if x + sw + pad > cw and x > pad:
                break
            if sh + pad > band_h:  # taller than the overlap band: skip
                continue
            # Pin to the chosen edge: `pad` from the top, or `pad` from the
            # bottom so the band still lies inside the bottom overlap strip.
            y = (ch - pad - sh) if bottom else pad
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
            min_x = x if min_x is None else min(min_x, x)
            min_y = y if min_y is None else min(min_y, y)
            max_x = max(max_x, x + sw)
            max_y = max(max_y, y + sh)
            x += sw + pad
        painter.end()

    stamp_norm = None
    if boxes:
        # The REAL pasted rectangle (normalized), padded by one gap on each side
        # and clamped to the tile. For top placement min_y ~ pad so ny0 ~ 0
        # (the historical [0,0,nx,ny]); for bottom placement it hugs the bottom.
        stamp_norm = [
            max(0.0, (min_x - pad) / cw),
            max(0.0, (min_y - pad) / ch),
            min(1.0, (max_x + pad) / cw),
            min(1.0, (max_y + pad) / ch),
        ]

    buf = QBuffer()
    buf.open(WriteOnly)
    sub.save(buf, _TILE_IMAGE_FORMAT, _tile_jpeg_quality())
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


def encoded_image_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) of an encoded JPEG or PNG from its header bytes only.

    The submission path needs the SENT pixel size, not the source-rect size:
    a re-split quadrant ships the same ground extent at a finer pixel grid, so
    reading the actual encoded dimensions is the one measure that is right at
    every re-split depth. No decoder, no image library: PNG stores the size in
    the fixed IHDR position, JPEG in the first SOF segment. Returns None for
    anything unrecognized or truncated; callers treat that as unknown.
    """
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return (
            int.from_bytes(data[16:20], "big"),
            int.from_bytes(data[20:24], "big"),
        )
    if len(data) >= 4 and data[:2] == b"\xff\xd8":
        i = 2
        n = len(data)
        while i + 9 < n:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            # Standalone markers carry no length segment.
            if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h = int.from_bytes(data[i + 5:i + 7], "big")
                w = int.from_bytes(data[i + 7:i + 9], "big")
                return (w, h) if w > 0 and h > 0 else None
            seg_len = int.from_bytes(data[i + 2:i + 4], "big")
            i += 2 + max(seg_len, 2)
    return None


def should_request_semantic(
    enabled: bool, has_prompt: bool, merge_separate: bool
) -> bool:
    """Whether to ask the service for a semantic coverage map on this run.

    Only a map-like TEXT prompt qualifies: the coverage map is a fallback for
    continuous features the per-instance pass can miss whole, and it needs a
    text query to answer, so it is requested only when the server dial is on, a
    text prompt is present, and that prompt's merge policy is map-like (not the
    counting / SEPARATE policy). Off by default (the dial is fail-closed)."""
    return bool(enabled) and bool(has_prompt) and not bool(merge_separate)


def mask_scale_field(scale: int | None) -> int | None:
    """The ``mask_scale`` value to attach to a /predict request, or None to omit.

    The service accepts a per-request mask grid keyed to 1 or 2 (absent = the
    full grid). Only the coarser grid is worth sending, and only scale 2 is
    validated, so the client clamps the run's resolved decision to {1, 2} and
    attaches the field ONLY for 2; scale 1 (the default full grid) and every
    other value omit it, keeping the request byte-identical to a client that
    never learned the field. Additive and optional: an older service ignores an
    unknown field, so sending it is always backward-compatible."""
    return 2 if scale == 2 else None


def _opt_float(val: object) -> float | None:
    """A plain float from an optional numeric field, else None. A JSON bool is
    rejected (never read as 0.0/1.0)."""
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return None


def parse_semantic_fields(
    response: dict,
) -> tuple[str | None, float | None, float | None]:
    """(semantic_rle, semantic_coverage, presence) from a response, each None
    when the field is absent or malformed.

    All three are additive and optional: an older service, or a tile with no
    coverage output, omits them, and the caller treats absence as "no rescue
    possible", never an error. The RLE is a non-empty string; coverage and
    presence are plain floats."""
    rle = response.get("semantic_rle")
    if not isinstance(rle, str) or not rle.strip():
        rle = None
    return (
        rle,
        _opt_float(response.get("semantic_coverage")),
        _opt_float(response.get("presence")),
    )


def should_rescue_with_semantic(
    instance_count: int,
    coverage: float | None,
    has_rle: bool,
    enabled: bool,
    coverage_floor: float,
) -> bool:
    """Whether a tile's semantic coverage mask should stand in for a missing
    per-instance result.

    Fires ONLY when the rescue is enabled, the per-instance pass returned
    NOTHING for this tile (``instance_count == 0``), a coverage mask is present,
    and its coverage reaches the floor. A tile with any instance keeps today's
    result untouched; a missing mask or a coverage below the floor leaves the
    tile empty."""
    if not enabled or instance_count > 0 or not has_rle:
        return False
    if not isinstance(coverage, (int, float)) or isinstance(coverage, bool):
        return False
    return float(coverage) >= float(coverage_floor)


def _response_mask_list(response: dict) -> list:
    """The response's "masks" list, or [] when absent or of the wrong type."""
    raw_masks = response.get("masks") or []
    return raw_masks if isinstance(raw_masks, list) else []


def _iter_mask_entries(raw_masks: list, score_threshold: float):
    """Yield the (entry, score) pairs of a masks list that pass the score gate,
    in server order.

    Pure dict work: the RLE stays a string, so a caller that only needs the
    count never allocates a pixel grid.
    """
    for entry in raw_masks:
        if not isinstance(entry, dict):
            continue
        score = float(entry.get("score", 0.0))
        if score < score_threshold:
            continue
        yield entry, score


def detection_mask_count(response: dict, score_threshold: float = 0.0) -> int:
    """How many masks a response decodes to, WITHOUT decoding any of them.

    Lets a caller take a decision that only needs the count (the saturation
    cap) while still consuming the masks one at a time.
    """
    return sum(1 for _ in _iter_mask_entries(
        _response_mask_list(response), score_threshold))


def _iter_masks(raw_masks: list, decode_h: int, decode_w: int, score_threshold: float):
    """The lazy half of :func:`iter_detection_masks`: one decoded mask per step."""
    for entry, score in _iter_mask_entries(raw_masks, score_threshold):
        mask = decode_rle_to_mask(entry.get("rle", ""), decode_h, decode_w)
        raw_box = entry.get("box")
        if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
            box = [float(v) for v in raw_box]
        else:
            box = [0.0, 0.0, 0.0, 0.0]
        yield mask, score, box


def iter_detection_masks(
    response: dict,
    tile_w: int,
    tile_h: int,
    score_threshold: float = 0.0,
):
    """Decode a completed status response into (mask, score, box), ONE AT A TIME.

    Same values, same order as :func:`decode_detection_response`, but a mask is
    built only when the consumer asks for it. A saturated tile can carry over a
    hundred full-tile boolean grids, and holding them together is a large
    simultaneous allocation on the machines least able to afford it; a consumer
    that turns each mask into geometry and drops it keeps only one alive.

    The masks-list check and the decode dimensions are resolved EAGERLY (this is
    a plain function returning an iterator, not a generator), so a malformed
    response fails at the call, exactly where it did before.

    The server response contains a "masks" list where each entry has:
      - "rle":   space-separated "offset count" string (1-based, row-major)
      - "score": float confidence
      - "box":   [cx, cy, w, h] NORMALIZED (YOLO style) -- not pixel coords

    When response["width"] or ["height"] is None (the server may omit them),
    the caller-supplied tile_w / tile_h are used for RLE decoding.
    """
    raw_masks = response.get("masks") or []
    if not isinstance(raw_masks, list):
        logger.warning("decode_detection_response: 'masks' is not a list; returning []")
        return iter(())

    # Prefer server-reported dimensions; fall back to caller-supplied tile dims.
    srv_w = response.get("width")
    srv_h = response.get("height")
    decode_w = int(srv_w) if srv_w is not None else tile_w
    decode_h = int(srv_h) if srv_h is not None else tile_h

    return _iter_masks(raw_masks, decode_h, decode_w, score_threshold)


def decode_detection_response(
    response: dict,
    tile_w: int,
    tile_h: int,
    score_threshold: float = 0.0,
) -> list[tuple[np.ndarray, float, list[float]]]:
    """Every mask of one tile, materialized together.

    Prefer :func:`iter_detection_masks` on any path that processes a whole run:
    this holds every mask of the tile at once. Kept for callers that genuinely
    need the list (and for the characterization tests).

    Args:
        response:        Completed status dict with a "masks" list.
        tile_w:          Actual tile pixel width (used when server omits dimensions).
        tile_h:          Actual tile pixel height.
        score_threshold: Masks with score < threshold are discarded.

    Returns:
        List of (mask, score, box) where mask is bool (H, W), score is float,
        box is [cx, cy, w, h] normalized (may be [0,0,0,0] if server omits it).
    """
    return list(iter_detection_masks(response, tile_w, tile_h, score_threshold))
