








from __future__ import annotations

import math



SELF_EXEMPLAR_INDEX_OFFSET = 500000

SELF_EXEMPLAR_TOP_K = 3
SELF_EXEMPLAR_MIN_SCORE = 0.5


def self_exemplar_wire_index(base_tile_index: int) -> int:
    return SELF_EXEMPLAR_INDEX_OFFSET + int(base_tile_index)


def parse_self_exemplar_decision(value: object) -> dict | None:






    if value is True:
        return {"top_k": SELF_EXEMPLAR_TOP_K, "min_score": SELF_EXEMPLAR_MIN_SCORE}
    if not isinstance(value, dict):
        return None
    top_k = value.get("top_k", SELF_EXEMPLAR_TOP_K)
    min_score = value.get("min_score", SELF_EXEMPLAR_MIN_SCORE)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 3:
        return None
    if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
        return None
    min_score = float(min_score)
    if not math.isfinite(min_score) or not 0.0 <= min_score < 1.0:
        return None
    return {"top_k": top_k, "min_score": min_score}


def resolve_self_exemplar_settings(flag_on: bool, decisions: object) -> dict | None:


    if flag_on is not True or not isinstance(decisions, dict):
        return None
    return parse_self_exemplar_decision(decisions.get("self_exemplar"))


def _score_of(entry: dict) -> float:
    score = entry.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return 0.0
    score = float(score)
    return score if math.isfinite(score) else 0.0


def select_self_exemplar_boxes(
    answer: object, image_w: int, image_h: int,
    top_k: int = SELF_EXEMPLAR_TOP_K, min_score: float = SELF_EXEMPLAR_MIN_SCORE,
) -> list[dict]:








    if not isinstance(answer, dict) or image_w <= 0 or image_h <= 0:
        return []
    masks = answer.get("masks")
    if not isinstance(masks, list):
        return []
    scored = [m for m in masks if isinstance(m, dict)]
    scored.sort(key=_score_of, reverse=True)
    picked = [m for m in scored if _score_of(m) > min_score][:max(0, int(top_k))]
    out = []
    for entry in picked:
        raw = entry.get("box")
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            continue
        try:
            cx, cy, w, h = (float(v) for v in raw)
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (cx, cy, w, h)):
            continue
        x0, x1 = (cx - w / 2) * image_w, (cx + w / 2) * image_w
        y0, y1 = (cy - h / 2) * image_h, (cy + h / 2) * image_h
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue
        out.append({"box": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                    "label": 1})
    return out


def self_exemplar_boxes_for_image(
    answer: object, image_bytes: bytes,
    top_k: int = SELF_EXEMPLAR_TOP_K, min_score: float = SELF_EXEMPLAR_MIN_SCORE,
) -> list[dict]:



    from .tile_images import encoded_image_size

    size = encoded_image_size(image_bytes or b"")
    if size is None:
        return []
    return select_self_exemplar_boxes(answer, size[0], size[1], top_k, min_score)
