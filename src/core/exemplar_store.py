"""In-memory store of visual exemplars for the current Automatic run.

A visual exemplar is a single example box the user drew on the canvas to tell
the cloud model "find more things like this" (positive) or "ignore things like
this" (exclude). The store keeps the boxes (in the map canvas CRS) plus their
labels for the lifetime of one Automatic run; it is cleared when the zone is
redrawn or the run resets. Main thread only, no Qt widgets, no persistence.

Visual prompting plateaus the gain after a few exemplars, so the store caps
insertions PER LABEL: at most EXEMPLAR_MAX_POSITIVE positive examples and at
most EXEMPLAR_MAX_EXCLUDE exclude examples (the two are counted independently).
"""
from __future__ import annotations

from dataclasses import dataclass

from qgis.core import QgsGeometry, QgsRectangle
from qgis.PyQt.QtGui import QImage

# Hard per-label ceilings: at most 3 positive and at most 2 exclude examples.
# Client fallbacks for the server policy's `exemplar.max_positive` /
# `exemplar.max_exclude`, read through the resolvers below.
EXEMPLAR_MAX_POSITIVE = 3
EXEMPLAR_MAX_EXCLUDE = 2
# REGION markers (review correction boxes) have their own ceiling, separate
# from the object-exemplar caps above: they are never pasted (no band space)
# and a request sends at most 8 exemplars anyway, so correction gestures must
# not eat the user's chip slots, and chips must not silently evict recorded
# corrections.
EXEMPLAR_MAX_REGION = 8
# Total ceiling (both labels at their cap). Kept for callers that only need the
# combined count; the real gate is the per-label cap above.
EXEMPLAR_MAX = EXEMPLAR_MAX_POSITIVE + EXEMPLAR_MAX_EXCLUDE
LABEL_POSITIVE = 1
LABEL_EXCLUDE = 0


def _policy_cap(getter_name: str, fallback: int) -> int:
    """One exemplar ceiling resolved from the cached server policy, else the
    client constant. Cache-only (no network), safe on any thread; any failure
    keeps the shipped ceiling, so an offline run behaves exactly as before."""
    try:
        from . import detection_policy
        return int(getattr(detection_policy, getter_name)(fallback))
    except Exception:  # noqa: BLE001 - policy is optional, never break the store
        return fallback


def max_positive() -> int:
    """How many positive examples this run may hold."""
    return _policy_cap("exemplar_max_positive", EXEMPLAR_MAX_POSITIVE)


def max_exclude() -> int:
    """How many exclude examples this run may hold."""
    return _policy_cap("exemplar_max_exclude", EXEMPLAR_MAX_EXCLUDE)


def max_region() -> int:
    """How many region markers this run may hold."""
    return _policy_cap("exemplar_max_region", EXEMPLAR_MAX_REGION)


@dataclass
class Exemplar:
    id: str
    map_rect: QgsRectangle   # in the MAP CANVAS CRS (project CRS)
    label: int               # 1 = positive (find similar), 0 = exclude
    thumbnail: QImage | None = None
    # Optional precise outline (canvas CRS) the user drew. map_rect is its
    # bounding box. When present the worker masks the stamped crop to this
    # polygon so the cloud model's box exemplar isn't polluted by neighbours.
    polygon: QgsGeometry | None = None
    # Cached model-ready stamp, rendered WHEN THE EXEMPLAR IS DRAWN (the canvas
    # already shows those pixels) so Detect never blocks on a per-exemplar
    # basemap render. stamp_layer_id keys the cache to the source
    # raster; a mismatch at Detect forces a rebuild. Cleared implicitly when the
    # exemplar is removed (the whole entry is dropped).
    stamp_img: QImage | None = None
    stamp_obj_box: list | None = None
    stamp_layer_id: str | None = None
    # Layer-CRS ground units per pixel the stamp was rendered at. The stamp
    # must match the RUN's tile resolution (the model matches by apparent
    # scale: a reference rendered 2x finer than the tiles pushes it toward
    # bigger, coarser structures). 0.0 = legacy source-finest render; a
    # mismatch with the run's mupp at Detect forces a rebuild.
    stamp_gsd: float = 0.0
    # Longest side (px) the cached stamp was actually rendered at. Crops share
    # one paste row whose per-crop size cap can shrink as more examples are
    # added, so a stamp cached wider than the current cap is rebuilt at Detect.
    stamp_side: int = 0
    # True for a REGION marker (a review correction box): the box flags an area
    # to look at again, not the outline of one object. Region boxes are sent
    # clipped to each tile they overlap instead of requiring full containment,
    # and are never pasted as crops.
    region: bool = False


class ExemplarStore:
    """Session-only ordered store of visual exemplars (boxes + labels).

    Insertion order is preserved (Python 3.7+ dict). Ids are short, unique and
    deterministic within a store instance (an incrementing counter), so no
    randomness or wall-clock is involved.
    """

    def __init__(self) -> None:
        self._exemplars: dict[str, Exemplar] = {}
        self._seq = 0

    def add(
        self,
        map_rect: QgsRectangle,
        label: int,
        thumbnail: QImage | None = None,
        polygon: QgsGeometry | None = None,
        region: bool = False,
    ) -> str | None:
        """Store a new exemplar. Returns its id, or None if that label is full.

        Args:
            map_rect: Box in the map canvas CRS (project CRS).
            label: LABEL_POSITIVE (1) or LABEL_EXCLUDE (0).
            thumbnail: Optional preview image (not required).
            polygon: Optional precise outline (canvas CRS); map_rect is its bbox.
            region: Area marker (correction box) rather than an object outline.
        """
        if region:
            if self.regions() >= max_region():
                return None
        elif self.is_full_for(label):
            return None
        self._seq += 1
        exemplar_id = f"ex{self._seq}"
        self._exemplars[exemplar_id] = Exemplar(
            id=exemplar_id,
            map_rect=QgsRectangle(map_rect),
            label=label,
            thumbnail=thumbnail,
            polygon=QgsGeometry(polygon) if polygon is not None else None,
            region=bool(region),
        )
        return exemplar_id

    def remove(self, exemplar_id: str) -> None:
        """Remove an exemplar by id. No-op if missing."""
        self._exemplars.pop(exemplar_id, None)

    def clear(self) -> None:
        """Remove all exemplars (does not reset the id counter)."""
        self._exemplars.clear()

    def count(self) -> int:
        return len(self._exemplars)

    def positives(self) -> int:
        return sum(1 for e in self._exemplars.values()
                   if e.label == LABEL_POSITIVE and not e.region)

    def excludes(self) -> int:
        return sum(1 for e in self._exemplars.values()
                   if e.label == LABEL_EXCLUDE and not e.region)

    def regions(self) -> int:
        return sum(1 for e in self._exemplars.values() if e.region)

    def is_full_for(self, label: int) -> bool:
        """True when the OBJECT-exemplar cap for THIS label is reached (no
        more chips of it can be added). Positive and exclude examples are
        capped independently; REGION markers have their own separate ceiling
        (EXEMPLAR_MAX_REGION, checked in add) so correction gestures and chips
        never compete for slots."""
        if label == LABEL_POSITIVE:
            return self.positives() >= max_positive()
        return self.excludes() >= max_exclude()

    def is_full(self) -> bool:
        """True only when BOTH labels are at their cap (nothing more can be
        added at all). Per-label gating uses is_full_for."""
        return self.is_full_for(LABEL_POSITIVE) and self.is_full_for(LABEL_EXCLUDE)

    def list(self) -> list[Exemplar]:
        """Return exemplars in insertion order."""
        return list(self._exemplars.values())

    def get(self, exemplar_id: str) -> Exemplar | None:
        return self._exemplars.get(exemplar_id)
