"""Whether a render asked a map source for more detail than it holds.

A tile pyramid declares zoom levels. It does not promise a picture at those
levels. Past its real depth an online source answers with a placeholder card,
which renders as real pixels, so a run reads a page of cards as ground and the
user pays for it.

The single-picture test for that card is in ``cloud_detection``
(``tile_is_unavailable``): flat, and drawn in greys. It is cheap and it works,
but a flat grey card and flat grey GROUND are the same picture. Salt flats,
ice, a bare desert: every one of them is a run refused for imagery that is
really there.

This module adds the piece the single picture cannot carry: a SECOND render of
the same ground one level coarser, and the question of whether the two show the
same place. Real ground looks like itself at both levels, however flat it is.
A card does not, because one level down the source has a picture. The test
knows nothing about any provider, any artwork or any byte count, so a source
whose card is blue and says something else in another language fails it just
the same.

Both values are server dials with shipped fallbacks, and every entry point
fails OPEN: a render that did not come back, an image the check cannot read, or
an ambiguous answer all report agreement, which leaves the run exactly as it
was.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# How much of the fine render's picture the coarser one must also show before
# the two are read as the same ground. Below this the finer level is showing
# something the coarser one does not have, which is what a placeholder is.
#
# The number is a correlation, so it lives in [-1, 1] and carries no unit. It
# is only ever consulted for a render the flat-and-grey test has already
# accused, so it is a confirmation and never an accusation of its own.
LEVEL_AGREEMENT_MIN: float = 0.27

# Side of the square both renders are reduced to before they are compared.
# Small on purpose: the question is whether the same PLACE is in both pictures,
# which survives heavy reduction, and the check runs while the user waits.
LEVEL_SAMPLE_PX: int = 32

# How many times a run may step back to a coarser level before it gives up.
# Each step doubles the ground a tile covers, so two steps is a factor of four
# on the resolution and already a different run from the one asked for.
LEVEL_BACKOFF_STEPS: int = 2

# Correlation is meaningless when one of the two pictures carries no variation
# at all, and a divide by its own zero would report perfect disagreement on an
# image that simply has nothing in it. Anything at or below this counts as no
# variation, and the pair reports agreement.
_FLAT_IMAGE_EPS: float = 1e-6


def level_agreement(fine_img, coarse_img, sample_px: int = 0) -> float:
    """How much of one render's picture the other also shows, in [-1, 1].

    ``fine_img`` and ``coarse_img`` are QImages of the SAME ground, the second
    rendered into fewer pixels so the source is asked for one level less. Both
    are reduced to one small square and their greys correlated, so the answer
    is about content and not about sharpness: a coarser picture of the same
    place still scores near 1.

    Returns 1.0 (perfect agreement, so no caller acts) for anything it cannot
    measure. Reentrant: QImage only, safe off the GUI thread.
    """
    try:
        import numpy as np  # noqa: PLC0415 - keep plugin load light

        side = int(sample_px) if sample_px else _sample_px()
        fine = _grey_sample(fine_img, side, np)
        coarse = _grey_sample(coarse_img, side, np)
        if fine is None or coarse is None:
            return 1.0
        fine = fine - fine.mean()
        coarse = coarse - coarse.mean()
        norm = float(np.linalg.norm(fine)) * float(np.linalg.norm(coarse))
        if norm <= _FLAT_IMAGE_EPS:
            return 1.0
        return float(fine.ravel() @ coarse.ravel() / norm)
    except Exception as exc:  # noqa: BLE001 - a failed check never stops a run
        logger.debug("level_agreement: comparison failed: %s", exc)
        return 1.0


def levels_disagree(fine_img, coarse_img) -> bool:
    """True when the two renders are not showing the same ground.

    The confirming half of the placeholder verdict: read it only for a render
    the flat-and-grey test has already accused. False whenever the comparison
    cannot be made, so an unreadable pair leaves the run alone.
    """
    return level_agreement(fine_img, coarse_img) < agreement_min()


def agreement_min(policy: dict | None = None) -> float:
    """The agreement below which two renders are read as different ground.

    Served, then held inside [-1, 1] because a correlation cannot leave that
    band and a value outside it would either accuse everything or nothing.
    """
    from .detection_policy import unavailable_agreement_min

    try:
        value = float(unavailable_agreement_min(LEVEL_AGREEMENT_MIN, policy))
    except (TypeError, ValueError):
        return LEVEL_AGREEMENT_MIN
    if value != value or not -1.0 <= value <= 1.0:
        return LEVEL_AGREEMENT_MIN
    return value


def backoff_steps(policy: dict | None = None) -> int:
    """How many coarser levels a run may fall back to before it is refused.

    0 turns the fall-back off and restores the plain refusal, which is what
    shipped before this existed, so the switch is safe to pull mid-incident.
    """
    from .detection_policy import unavailable_backoff_steps

    try:
        value = int(unavailable_backoff_steps(LEVEL_BACKOFF_STEPS, policy))
    except (TypeError, ValueError):
        return LEVEL_BACKOFF_STEPS
    return max(0, min(value, 8))


def _sample_px(policy: dict | None = None) -> int:
    """Side of the comparison square, held in a band a correlation can use.

    Under a handful of pixels the two samples correlate on almost nothing; far
    above it the check pays for detail the question does not need.
    """
    from .detection_policy import unavailable_agreement_sample_px

    try:
        value = int(unavailable_agreement_sample_px(LEVEL_SAMPLE_PX, policy))
    except (TypeError, ValueError):
        return LEVEL_SAMPLE_PX
    if not 8 <= value <= 256:
        return LEVEL_SAMPLE_PX
    return value


def _grey_sample(img, side: int, np):
    """``img`` reduced to a ``side`` square of greys, or None.

    Smooth reduction on purpose: the coarser render has fewer pixels to start
    with, and a nearest-neighbour reduction would compare two different
    samplings of the same ground rather than the ground.
    """
    from qgis.PyQt.QtCore import QSize, Qt
    from qgis.PyQt.QtGui import QImage

    if img is None or img.isNull():
        return None
    small = img.scaled(
        QSize(side, side),
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    ).convertToFormat(QImage.Format.Format_RGB32)
    if small.width() != side or small.height() != side:
        return None
    ptr = small.bits()
    ptr.setsize(side * side * 4)
    arr = np.frombuffer(bytes(ptr), dtype=np.uint8).reshape(side, side, 4)
    # Qt keeps RGB32 in BGRA byte order.
    return (arr[:, :, 2] * 0.299 + arr[:, :, 1] * 0.587
            + arr[:, :, 0] * 0.114).astype(np.float32)
