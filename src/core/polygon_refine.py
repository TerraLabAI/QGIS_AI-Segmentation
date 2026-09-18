







# ruff: noqa: E402


from __future__ import annotations

from .venv_manager import ensure_venv_packages_available

ensure_venv_packages_available()

import numpy as np

from .shape_policy_dials import refine_margin_px


def apply_mask_refinement(
    mask: np.ndarray,
    expand_value: int = 0,
    fill_holes: bool = False,
    min_area: int = 0,
    max_hole_px: int | None = None,
) -> np.ndarray:


















    result = mask.astype(np.uint8, order="C")
    if expand_value == 0 and not fill_holes and min_area <= 0:
        return result




    window = _mask_bounding_window(
        result, abs(int(expand_value)) + refine_margin_px(_REFINE_MARGIN_PX))
    if window is None:
        return result
    row0, row1, col0, col1 = window
    region = _apply_refinement_steps(
        np.ascontiguousarray(result[row0:row1, col0:col1]),
        expand_value, fill_holes, min_area, max_hole_px)
    if region.shape == result.shape:
        return region
    result[row0:row1, col0:col1] = region
    return result





_REFINE_MARGIN_PX = 2


def _mask_bounding_window(mask: np.ndarray, margin: int = 0) -> tuple | None:
















    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    height, width = mask.shape
    return (max(0, int(rows[0]) - margin),
            min(height, int(rows[-1]) + 1 + margin),
            max(0, int(cols[0]) - margin),
            min(width, int(cols[-1]) + 1 + margin))


def _apply_refinement_steps(
    mask: np.ndarray,
    expand_value: int,
    fill_holes: bool,
    min_area: int,
    max_hole_px: int | None,
) -> np.ndarray:


    result = mask


    if expand_value != 0:
        iterations = abs(expand_value)
        if expand_value > 0:
            result = _numpy_dilate(result, iterations)
        else:
            result = _numpy_erode(result, iterations)


    if fill_holes:
        if max_hole_px is None:
            result = _fill_holes(result)
        else:
            result = fill_small_holes(result, max(0, int(max_hole_px)))


    if min_area > 0:
        result = _remove_small_regions(result, min_area)

    return result


def fill_small_holes(mask: np.ndarray, max_hole_px: int) -> np.ndarray:










    try:
        from scipy import ndimage
    except ImportError:
        ndimage = None
    try:
        solid = mask.astype(bool, copy=False)
        if solid.size == 0:
            return solid.astype(np.uint8)







        labels, n = (_label_components(~solid) if ndimage is None
                     else ndimage.label(~solid))
        if n == 0:
            return solid.astype(np.uint8)
        border = np.concatenate((labels[0], labels[-1],
                                 labels[:, 0], labels[:, -1]))
        outside = np.zeros(n + 1, dtype=bool)
        outside[border] = True
        outside[0] = True
        counts = np.bincount(labels.ravel(), minlength=n + 1)
        fillable = ~outside & (counts <= max_hole_px)
        if not fillable.any():
            return solid.astype(np.uint8)
        return (solid | fillable[labels]).astype(np.uint8)
    except Exception:



        return mask


def _label_components(mask_bool: np.ndarray) -> tuple[np.ndarray, int]:












    h, w = mask_bool.shape
    labels = np.zeros((h, w), dtype=np.int32)
    if h == 0 or w == 0 or not mask_bool.any():
        return labels, 0


    padded = np.zeros((h, w + 2), dtype=bool)
    padded[:, 1:-1] = mask_bool
    flat = padded.ravel()
    edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = edges[0::2]
    stops = edges[1::2]
    rows = starts // (w + 2)
    col_start = starts - rows * (w + 2) - 1
    col_stop = stops - rows * (w + 2) - 1

    n_runs = int(starts.size)
    parent = list(range(n_runs))

    def _find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root


    row_lo = np.searchsorted(rows, np.arange(h), side="left").tolist()
    row_hi = np.searchsorted(rows, np.arange(h), side="right").tolist()
    cs = col_start.tolist()
    ce = col_stop.tolist()
    for r in range(h - 1):
        i, i_end = row_lo[r], row_hi[r]
        j, j_end = row_lo[r + 1], row_hi[r + 1]
        while i < i_end and j < j_end:
            if cs[i] < ce[j] and cs[j] < ce[i]:
                ri, rj = _find(i), _find(j)
                if ri != rj:
                    if ri < rj:
                        parent[rj] = ri
                    else:
                        parent[ri] = rj
            if ce[i] < ce[j]:
                i += 1
            else:
                j += 1


    numbering: dict[int, int] = {}
    run_label = [0] * n_runs
    for i in range(n_runs):
        root = _find(i)
        lab = numbering.get(root)
        if lab is None:
            lab = len(numbering) + 1
            numbering[root] = lab
        run_label[i] = lab

    row_list = rows.tolist()
    for i in range(n_runs):
        labels[row_list[i], cs[i]:ce[i]] = run_label[i]
    return labels, len(numbering)


def _fill_holes(mask: np.ndarray) -> np.ndarray:






    try:
        from scipy import ndimage
        return ndimage.binary_fill_holes(mask).astype(np.uint8)
    except ImportError:
        pass
    except Exception:  # noqa: BLE001  # nosec B110




        pass





    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = mask

    background = (padded == 0)
    labels, _count = _label_components(background)


    exterior = labels == labels[0, 0]

    result = padded.copy()
    result[background & ~exterior] = 1

    return result[1:-1, 1:-1]


def _remove_small_regions(mask: np.ndarray, min_area: int) -> np.ndarray:




    if min_area <= 1:
        return mask.copy()


    try:
        from scipy import ndimage
        labeled, num_features = ndimage.label(mask)
        if num_features == 0:
            return mask.copy()


        component_sizes = np.bincount(labeled.ravel())


        keep_mask = component_sizes >= min_area
        keep_mask[0] = False


        return keep_mask[labeled].astype(np.uint8)

    except ImportError:
        pass



    labels, count = _label_components(mask.astype(bool))
    if count == 0:
        return mask.copy()

    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    small = np.flatnonzero(sizes < min_area)
    small = small[small != 0]

    if small.size:
        result = mask.copy()
        result[np.isin(labels, small)] = 0
        return result

    return mask.copy()


def count_significant_regions(mask: np.ndarray, min_ratio: float = 0.01) -> int:





    if mask is None or mask.sum() == 0:
        return 0


    bridged = _numpy_dilate(mask.astype(np.uint8), 1)

    sizes = _label_region_sizes(bridged)
    if len(sizes) == 0:
        return 0

    largest = max(sizes)
    threshold = largest * min_ratio
    return sum(1 for s in sizes if s >= threshold)


def _label_region_sizes(mask: np.ndarray) -> list:

    try:
        from scipy import ndimage
        labeled, num_features = ndimage.label(mask)
        if num_features == 0:
            return []
        return list(np.bincount(labeled.ravel())[1:])
    except ImportError:
        pass



    labels, count = _label_components(mask.astype(bool))
    if count == 0:
        return []
    return list(np.bincount(labels.ravel(), minlength=count + 1)[1:])


def _numpy_dilate(mask: np.ndarray, iterations: int) -> np.ndarray:





    try:
        from scipy.ndimage import binary_dilation
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        return binary_dilation(
            mask, structure=struct, iterations=iterations
        ).astype(np.uint8)
    except ImportError:
        pass

    result = mask.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=0)
        center = padded[1:-1, 1:-1]
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        dilated = center | up | down | left | right
        result = dilated.astype(np.uint8)
    return result


def _numpy_erode(mask: np.ndarray, iterations: int) -> np.ndarray:





    try:
        from scipy.ndimage import binary_erosion
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        return binary_erosion(
            mask, structure=struct, iterations=iterations
        ).astype(np.uint8)
    except ImportError:
        pass

    result = mask.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=0)
        center = padded[1:-1, 1:-1]
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        eroded = center & up & down & left & right
        result = eroded.astype(np.uint8)
    return result
