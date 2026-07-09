














from __future__ import annotations

import math
from dataclasses import dataclass

from qgis.core import QgsGeometry, QgsRectangle
from qgis.PyQt.QtGui import QImage







EXEMPLAR_MAX_POSITIVE = 4
EXEMPLAR_MAX_EXCLUDE = 2
EXEMPLAR_MAX_TOTAL = 5


EXEMPLAR_MAX_POSITIVE_FREE = 1
EXEMPLAR_MAX_EXCLUDE_FREE = 0





EXEMPLAR_MAX_REGION = 8

EXEMPLAR_MAX = EXEMPLAR_MAX_TOTAL
LABEL_POSITIVE = 1
LABEL_EXCLUDE = 0


def _policy_cap(getter_name: str, fallback: int) -> int:



    try:
        from . import detection_policy
        return int(getattr(detection_policy, getter_name)(fallback))
    except Exception:  # noqa: BLE001
        return fallback


def max_positive(free_tier: bool = False) -> int:

    if free_tier:
        return _policy_cap("exemplar_max_positive_free", EXEMPLAR_MAX_POSITIVE_FREE)
    return _policy_cap("exemplar_max_positive", EXEMPLAR_MAX_POSITIVE)


def max_exclude(free_tier: bool = False) -> int:

    if free_tier:
        return _policy_cap("exemplar_max_exclude_free", EXEMPLAR_MAX_EXCLUDE_FREE)
    return _policy_cap("exemplar_max_exclude", EXEMPLAR_MAX_EXCLUDE)


def max_total() -> int:


    return _policy_cap("exemplar_max_total", EXEMPLAR_MAX_TOTAL)


def max_region() -> int:

    return _policy_cap("exemplar_max_region", EXEMPLAR_MAX_REGION)


@dataclass
class Exemplar:
    id: str
    map_rect: QgsRectangle
    label: int
    thumbnail: QImage | None = None



    polygon: QgsGeometry | None = None





    stamp_img: QImage | None = None
    stamp_obj_box: list | None = None
    stamp_layer_id: str | None = None





    stamp_gsd: float = 0.0



    stamp_side: int = 0




    region: bool = False


class ExemplarStore:







    def __init__(self) -> None:
        self._exemplars: dict[str, Exemplar] = {}
        self._seq = 0



        self.free_tier: bool = False

    def add(
        self,
        map_rect: QgsRectangle,
        label: int,
        thumbnail: QImage | None = None,
        polygon: QgsGeometry | None = None,
        region: bool = False,
    ) -> str | None:









        if label not in (LABEL_POSITIVE, LABEL_EXCLUDE):
            return None
        try:
            if (map_rect.isEmpty()
                    or not all(math.isfinite(v) for v in (
                        map_rect.xMinimum(), map_rect.yMinimum(),
                        map_rect.xMaximum(), map_rect.yMaximum()))):
                return None
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return None
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

        self._exemplars.pop(exemplar_id, None)

    def clear(self) -> None:

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





        if label not in (LABEL_POSITIVE, LABEL_EXCLUDE):
            return True
        free = self.free_tier
        if not free and self.positives() + self.excludes() >= max_total():
            return True
        if label == LABEL_POSITIVE:
            return self.positives() >= max_positive(free)
        return self.excludes() >= max_exclude(free)

    def is_full(self) -> bool:


        return self.is_full_for(LABEL_POSITIVE) and self.is_full_for(LABEL_EXCLUDE)

    def is_full_on_free_only(self, label: int) -> bool:


        if not self.free_tier:
            return False
        if not self.is_full_for(label):
            return False
        if self.positives() + self.excludes() >= max_total():
            return False
        if label == LABEL_POSITIVE:
            return self.positives() < max_positive(False)
        return self.excludes() < max_exclude(False)

    def list(self) -> list[Exemplar]:

        return list(self._exemplars.values())

    def get(self, exemplar_id: str) -> Exemplar | None:
        return self._exemplars.get(exemplar_id)
