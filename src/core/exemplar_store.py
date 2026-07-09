"""In-memory store of visual exemplars for the current Automatic run.

A visual exemplar is a single example box the user drew on the canvas to tell
the cloud model "find more things like this" (positive) or "ignore things like
this" (exclude). The store keeps the boxes (in the map canvas CRS) plus their
labels for the lifetime of one Automatic run; it is cleared when the zone is
redrawn or the run resets. Main thread only, no Qt widgets, no persistence.

Documentation on visual prompting plateaus the gain at 3-4 exemplars, so the
store caps insertions at EXEMPLAR_MAX.
"""
from __future__ import annotations

from dataclasses import dataclass

from qgis.core import QgsGeometry, QgsRectangle
from qgis.PyQt.QtGui import QImage

EXEMPLAR_MAX = 4          # doc plateau is 3-4 exemplars
LABEL_POSITIVE = 1
LABEL_EXCLUDE = 0


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
    ) -> str | None:
        """Store a new exemplar. Returns its id, or None if already full.

        Args:
            map_rect: Box in the map canvas CRS (project CRS).
            label: LABEL_POSITIVE (1) or LABEL_EXCLUDE (0).
            thumbnail: Optional preview image (not required).
            polygon: Optional precise outline (canvas CRS); map_rect is its bbox.
        """
        if self.is_full():
            return None
        self._seq += 1
        exemplar_id = f"ex{self._seq}"
        self._exemplars[exemplar_id] = Exemplar(
            id=exemplar_id,
            map_rect=QgsRectangle(map_rect),
            label=label,
            thumbnail=thumbnail,
            polygon=QgsGeometry(polygon) if polygon is not None else None,
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
        return sum(1 for e in self._exemplars.values() if e.label == LABEL_POSITIVE)

    def excludes(self) -> int:
        return sum(1 for e in self._exemplars.values() if e.label == LABEL_EXCLUDE)

    def is_full(self) -> bool:
        return len(self._exemplars) >= EXEMPLAR_MAX

    def list(self) -> list[Exemplar]:
        """Return exemplars in insertion order."""
        return list(self._exemplars.values())

    def get(self, exemplar_id: str) -> Exemplar | None:
        return self._exemplars.get(exemplar_id)
