








from __future__ import annotations


def exemplar_min_side_px(rect_w: float, rect_h: float, run_mupp: float) -> float:




    if run_mupp <= 0:
        return -1.0
    return min(rect_w, rect_h) / run_mupp


def exemplar_too_small(rect_w: float, rect_h: float, run_mupp: float, floor: float) -> bool:





    side = exemplar_min_side_px(rect_w, rect_h, run_mupp)
    if side < 0:
        return False
    return side < floor


def exemplar_at_max_detail(current_detail: int, max_detail: int) -> bool:




    return current_detail >= max_detail
