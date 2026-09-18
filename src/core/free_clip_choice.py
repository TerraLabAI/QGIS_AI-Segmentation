




from __future__ import annotations

from typing import NamedTuple

DETECT_CLIPPED = "detect_clipped"
DRAW_SMALLER = "draw_smaller"
UPGRADE = "upgrade"
DISMISSED = "dismissed"


class ChoicePlan(NamedTuple):
    run: bool
    redraw: bool
    upgrade: bool


def plan_for_choice(choice: str) -> ChoicePlan:


    if choice == DETECT_CLIPPED:
        return ChoicePlan(run=True, redraw=False, upgrade=False)
    if choice == DRAW_SMALLER:
        return ChoicePlan(run=False, redraw=True, upgrade=False)
    if choice == UPGRADE:
        return ChoicePlan(run=False, redraw=False, upgrade=True)
    return ChoicePlan(run=False, redraw=False, upgrade=False)
