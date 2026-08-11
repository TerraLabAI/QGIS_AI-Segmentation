"""Where to cut the square crop that one object gets refined in.

The local model encodes a whole crop before it can answer a single click, and
that encode is the entire cost of opening an object. Two things follow, and both
are this module's job:

- Neighbours should share a crop. A crop chosen from an object's own bounds is
  unique to that object, so walking a street of houses pays one encode per
  house. Chosen on a grid instead, the houses in one cell land on the SAME crop,
  and every one after the first is already encoded.
- The same object must always give the same crop. The window is picked twice for
  one gesture (once when the crop is warmed ahead of the click, once when the
  click opens the object), and the two must agree to the byte or the warm-up is
  wasted.

Pure geometry, no Qt and no QGIS: the caller supplies the object bounds and the
raster's native pixel size, and gets back a window it can hand to the crop
reader. Both callers must go through here; computing it twice by hand is how the
two stop agreeing.
"""
from __future__ import annotations

import math

# Ground size of one grid cell, as a fraction of the crop's own ground size. A
# quarter keeps the shared crop close enough around each of its objects that
# none of them sits near an edge.
GRID_CELL_FRACTION = 0.25

# The window is only allowed to grow in these steps, so two objects that need
# slightly different resolutions still round to one shared crop. Growing is
# always safe (a bigger window still contains the object); shrinking is not.
SCALE_STEP = 1.15

# An object must clear the crop edge by this fraction of the crop's ground size,
# or it does not belong in that shared crop and gets its own centred one.
EDGE_CLEARANCE = 0.05

# Native resolution: one source pixel per crop pixel. The floor for every window
# the two helpers below pick. See their docstrings for why.
NATIVE_SCALE = 1.0


def scale_at_least_native(exact_scale: float) -> float:
    """A file raster's zoom-out factor, never below native resolution.

    A crop is answered at a fixed square size whatever it holds, so a window
    cut from fewer source pixels than that square has to be blown up to fill
    it. Blowing a crop up invents no detail: it spends the whole answer on
    pixels that were interpolated, and the answer comes back worse than the
    same click answered from a larger window read at native resolution. So a
    canvas zoomed in past the raster's own pixels takes more ground rather than
    fewer pixels.

    Nothing here can ask for pixels the raster does not have. The reader clamps
    the window to the raster, so a source coarser than the square simply sends
    what it holds, which is the one case where there is nothing else to send.
    """
    try:
        value = float(exact_scale)
    except (TypeError, ValueError):
        return NATIVE_SCALE
    if not value > 0:
        return NATIVE_SCALE
    return max(NATIVE_SCALE, value)


def ground_per_pixel_at_least_native(requested: float, native: float) -> float:
    """The same floor for a tiled source, whose windows are sized in ground units.

    ``native`` is the finest ground size per pixel the source actually serves,
    or 0 when that is unknown, in which case the request is left alone. Asking
    a tiled source for a finer step than it serves buys interpolated pixels at
    the price of ground, exactly what ``scale_at_least_native`` refuses.
    """
    try:
        want = float(requested)
        floor = float(native)
    except (TypeError, ValueError):
        return requested
    if not want > 0 or not floor > 0:
        return requested
    return max(want, floor)


def scale_floor_is_usable(min_scale) -> bool:
    """Can this floor be the base of the scale ladder below?

    The tiled path measures its floor off the canvas, so a window with no size
    yet reports zero and a transform that failed reports a number that is not
    one. Both reach `log()`, where zero divides and the rest raises.
    """
    try:
        value = float(min_scale)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0


def crop_scale_for_bounds(
    width: float,
    height: float,
    native_pixel_size: float,
    crop_size: int = 1024,
    margin: float = 1.4,
    min_scale: float = 1.0,
    max_scale: float = 8.0,
) -> float:
    """Smallest allowed scale whose crop still holds a width x height object.

    ``scale`` is the crop reader's zoom-out factor: it reads
    ``crop_size * scale`` native pixels and resamples them down to
    ``crop_size``, so 1.0 is native resolution and 8.0 is the reader's ceiling.

    ``min_scale`` comes from the canvas on the tiled path, where a window not
    yet laid out reports zero and a broken transform reports a number that is
    not one. Both reach the logarithm below, so the floor is checked here
    rather than at every caller.

    Bounds that cover no ground, or none that can be read, answer with the
    caller's own floor, and 0.0 when the caller gave no usable one. 0.0 means
    "no size to give", which every reader of this number already treats as
    "measure it yourself". The alternative was ``NATIVE_SCALE``, and that is a
    zoom factor on source pixels: handed to a tiled caller, whose windows are
    sized in ground units, it asks for one ground unit per pixel and reads a
    crop a thousand units wide.
    """
    floored = scale_floor_is_usable(min_scale)
    unsized = float(min_scale) if floored else 0.0
    if native_pixel_size <= 0:
        return unsized
    needed_ground = max(width, height) * margin
    exact = needed_ground / (crop_size * native_pixel_size)
    if not math.isfinite(exact) or not exact > 0:
        # A coordinate that is not a number reaches the logarithm below and
        # raises there, on the click path. Refused here, like the floor.
        return unsized
    if not floored:
        # No floor to ladder from. The window that just holds the object is
        # the honest answer, and raising it is all the floor was ever for.
        return min(max_scale, exact)
    if exact <= min_scale:
        return min_scale
    if exact >= max_scale:
        return max_scale
    steps = math.ceil(math.log(exact / min_scale) / math.log(SCALE_STEP))
    return min(max_scale, min_scale * SCALE_STEP ** steps)


def snap_center_to_grid(
    center_x: float,
    center_y: float,
    scale: float,
    native_pixel_size: float,
    crop_size: int = 1024,
) -> tuple[float, float]:
    """Move a crop centre onto the shared grid, keeping its scale.

    For a crop chosen from a bare point rather than an object's bounds, which is
    what a Manual click gives. One cell is a quarter of the crop, so the point
    stays well inside the snapped window.
    """
    if native_pixel_size <= 0 or scale <= 0:
        return center_x, center_y
    step = crop_size * scale * native_pixel_size * GRID_CELL_FRACTION
    if step <= 0:
        return center_x, center_y
    return round(center_x / step) * step, round(center_y / step) * step


def neighborhood_crop_window(
    bounds: tuple[float, float, float, float],
    native_pixel_size: float,
    crop_size: int = 1024,
    margin: float = 1.4,
    min_scale: float = 1.0,
    max_scale: float = 8.0,
) -> tuple[float, float, float]:
    """The crop an object should be refined in: ``(center_x, center_y, scale)``.

    ``bounds`` is the object's ``(minx, miny, maxx, maxy)`` in the raster CRS.
    The centre is snapped onto a grid so neighbouring objects share one crop,
    unless snapping would push this object too close to the crop edge, in which
    case it gets its own centred crop. Deterministic: same object, same raster,
    same window.
    """
    minx, miny, maxx, maxy = bounds
    exact_cx = (minx + maxx) / 2.0
    exact_cy = (miny + maxy) / 2.0
    scale = crop_scale_for_bounds(
        maxx - minx, maxy - miny, native_pixel_size,
        crop_size=crop_size, margin=margin,
        min_scale=min_scale, max_scale=max_scale)
    if native_pixel_size <= 0:
        return exact_cx, exact_cy, scale

    ground = crop_size * scale * native_pixel_size
    step = ground * GRID_CELL_FRACTION
    if step <= 0:
        return exact_cx, exact_cy, scale
    cx = round(exact_cx / step) * step
    cy = round(exact_cy / step) * step

    # Does the object still sit comfortably inside the shared crop? A large one
    # does not, and takes its own centred window rather than being clipped.
    half = ground / 2.0 - ground * EDGE_CLEARANCE
    if (minx < cx - half or maxx > cx + half or miny < cy - half or maxy > cy + half):
        return exact_cx, exact_cy, scale
    return cx, cy, scale


def crop_window_key(
    center_x: float, center_y: float, scale: float
) -> tuple[float, float, float]:
    """Comparable identity for one crop window, safe against float drift."""
    return (round(float(center_x), 6), round(float(center_y), 6),
            round(float(scale), 6))


def crop_pixel_of_point(
    bounds: tuple[float, float, float, float],
    img_shape: tuple[int, int],
    x: float,
    y: float,
) -> tuple[int, int] | None:
    """(row, col) of a ground point inside a crop, floored like a raster read.

    The same answer a raster library's inverse transform gives, written out so
    the click path does not need one. Rows run top down, which is why the
    northing is subtracted rather than added.

    None when the window has no size. (0, 0) would be a valid pixel address,
    and the caller cannot tell it from an answer: it prompts the model at the
    crop's top-left corner and the user gets a mask nowhere near their click.

    Clamped to the grid, because a point on the right or bottom bound divides
    out to exactly the pixel count and would index one past the last column.
    """
    minx, miny, maxx, maxy = bounds
    height, width = int(img_shape[0]), int(img_shape[1])
    span_x = maxx - minx
    span_y = maxy - miny
    if span_x <= 0 or span_y <= 0 or width <= 0 or height <= 0:
        return None
    col = math.floor((x - minx) * width / span_x)
    row = math.floor((maxy - y) * height / span_y)
    return (min(max(int(row), 0), height - 1),
            min(max(int(col), 0), width - 1))


def window_frames_bounds(
    window: tuple[float, float, float],
    bounds: tuple[float, float, float, float],
    native_pixel_size: float,
    crop_size: int = 1024,
) -> bool:
    """Does an existing crop window hold this object, clear of its edges?

    The same clearance the grid guarantees, so a window accepted here is as good
    a frame as one the grid would have picked."""
    if native_pixel_size <= 0 or window is None:
        return False
    cx, cy, scale = window
    if scale <= 0:
        return False
    ground = crop_size * scale * native_pixel_size
    half = ground / 2.0 - ground * EDGE_CLEARANCE
    minx, miny, maxx, maxy = bounds
    return (minx >= cx - half and maxx <= cx + half and miny >= cy - half and maxy <= cy + half)


def crop_window_for_object(
    bounds: tuple[float, float, float, float],
    native_pixel_size: float,
    held_window: tuple[float, float, float] | None = None,
    crop_size: int = 1024,
    margin: float = 1.4,
    min_scale: float = 1.0,
    max_scale: float = 8.0,
) -> tuple[float, float, float]:
    """The window to refine one object in, preferring the crop already in hand.

    ``held_window`` is the window the model currently holds encoded. Reusing it
    when it frames the object properly is what makes a run of neighbours cost
    one encode: on the grid alone, two objects a few metres apart still land in
    different cells whenever a cell boundary happens to fall between them, and
    the second one pays in full for imagery already in memory.

    Reuse needs the SAME scale, not merely a window large enough: scale is the
    resolution the object is refined at, and quietly refining one object coarser
    than it asked for is a change to the result, not to the timing.
    """
    needed_scale = crop_scale_for_bounds(
        bounds[2] - bounds[0], bounds[3] - bounds[1], native_pixel_size,
        crop_size=crop_size, margin=margin,
        min_scale=min_scale, max_scale=max_scale)
    held_fits = held_window is not None and round(float(held_window[2]), 6) == round(needed_scale, 6)
    held_fits = held_fits and window_frames_bounds(held_window, bounds, native_pixel_size, crop_size=crop_size)
    if held_fits:
        return (float(held_window[0]), float(held_window[1]), needed_scale)
    return neighborhood_crop_window(
        bounds, native_pixel_size, crop_size=crop_size, margin=margin,
        min_scale=min_scale, max_scale=max_scale)
