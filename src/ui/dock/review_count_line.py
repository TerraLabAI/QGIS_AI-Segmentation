"""The review card's one-line count readout.

A pure function, not a mixin method: it reads no widget, and the review panel
mixin was over its size band. The panel keeps its method name and delegates.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QLocale

from ...core.i18n import tr
from .styles import BRAND_GREEN


def format_review_count_line(
    visible: int, total: int, pct: int, bound: str = "confidence",
) -> str:
    """ONE compact review readout line, always honest: green check + bold
    shown-count, then a muted tail counting what the filters hide.
    Sits at the top of the review card (it is the live readout of the
    filters below it). A run that found something NEVER reads as '0
    detected'. ``bound`` (only read when visible == 0) names the filter that
    hides everything, ``"confidence"``, ``"min"`` or ``"max"``, so the reveal
    hint points at the dial that will actually work.
    The check is the lime success accent (the CTA green never announces
    success).

    The tail never names Confidence while objects are still shown. Every
    run already arrives with a recall floor applied, so at the lowest
    cutoff the slider hides NOTHING and the hidden cohort is entirely Min
    size and hand deletions: a 700-tile run read "53 389 below 10%" with
    Confidence sitting at its own floor, which sent the user to the one
    slider that could not move."""
    check = f'<span style="color:{BRAND_GREEN};">&#10003;</span> '
    muted = 'style="color: rgba(128,128,128,0.95);"'
    if total <= 0:
        # Empty runs use the guidance box instead of this label; safe fallback.
        return "<b>{title}</b>".format(title=tr("No objects found"))
    loc = QLocale()
    if visible >= total:
        bold = (tr("1 object found") if total == 1
                else tr("{n} objects found").format(n=loc.toString(total)))
        tail = tr("all shown")
    elif visible > 0:
        bold = tr("{visible} of {n} shown").format(
            visible=loc.toString(visible), n=loc.toString(total))
        tail = tr("{hidden} hidden by the filters").format(
            hidden=loc.toString(total - visible))
    else:
        # No green check at 0 visible: nothing is shown, but the count is
        # honest and the tail tells the user how to reveal them - naming the
        # binding filter (Min size vs Confidence) so they pull the right one.
        bold = (tr("1 object found") if total == 1
                else tr("{n} objects found").format(n=loc.toString(total)))
        if bound == "min":
            tail = tr("0 shown - lower the Min size filter to reveal them")
        elif bound == "max":
            tail = tr("0 shown - raise the Max size filter to reveal them")
        else:
            tail = tr(
                "0 shown at {pct}% - lower Confidence to reveal them").format(pct=pct)
        return f"<b>{bold}</b> <span {muted}>· {tail}</span>"
    return f"{check}<b>{bold}</b> <span {muted}>· {tail}</span>"
