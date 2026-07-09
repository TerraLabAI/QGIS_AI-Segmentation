






from __future__ import annotations

from .manual_predict_clicks import (
    MASK_UNDO_DEPTH,
    QUIET_CLICK_REFUSED,
    QUIET_CLICK_REREAD,
    QUIET_CLICK_SUPERSEDED,
    ManualClickMixin,
    _click_refusal_answer,
    _click_was_superseded,
)
from .manual_predict_masks import (
    ManualMaskMixin,
)
from .manual_predict_shapes import (
    FILL_HOLES_CAP_UNKNOWN,
    ManualShapeMixin,
)


class ManualPredictMixin(
    ManualClickMixin,
    ManualShapeMixin,
    ManualMaskMixin,
):
    pass


__all__ = [
    "ManualPredictMixin",
    "FILL_HOLES_CAP_UNKNOWN",
    "MASK_UNDO_DEPTH",
    "QUIET_CLICK_REFUSED",
    "QUIET_CLICK_REREAD",
    "QUIET_CLICK_SUPERSEDED",
    "_click_refusal_answer",
    "_click_was_superseded",
]
