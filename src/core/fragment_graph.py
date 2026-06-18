








from __future__ import annotations

from qgis.core import QgsFeature, QgsGeometry, QgsSpatialIndex

MERGE_ALGORITHM = "fragment_graph_v1"


class FragmentReference:


    __slots__ = ("wkb", "score", "area", "bbox", "centroid")

    def __init__(self, geom: QgsGeometry, score: float) -> None:
        self.wkb = bytes(geom.asWkb())
        self.score = float(score)
        self.area = geom.area()
        self.bbox = geom.boundingBox()
        self.centroid = None

    def geometry(self) -> QgsGeometry:
        geom = QgsGeometry()
        geom.fromWkb(self.wkb)
        return geom

    def centre(self, geom: QgsGeometry | None = None):
        if self.centroid is None:
            self.centroid = (geom if geom is not None else self.geometry()).centroid().asPoint()
        return self.centroid


class FragmentGraph:


    def __init__(self) -> None:
        self.references: list[FragmentReference] = []
        self._by_wkb: dict[bytes, int] = {}
        self._parent: list[int] = []
        self._members: dict[int, list[int]] = {}
        self._keepers: dict[int, set[int]] = {}
        self._keeper_root: dict[int, int] = {}
        self._index = QgsSpatialIndex()
        self.wkb_bytes = 0
        self.candidate_hits = 0

    def root(self, index: int) -> int:

        while self._parent[index] != index:
            self._parent[index] = self._parent[self._parent[index]]
            index = self._parent[index]
        return index

    def add(self, geom: QgsGeometry, score: float, matches) -> tuple[int, bool]:





        candidate = FragmentReference(geom, score)
        existing = self._by_wkb.get(candidate.wkb)
        if existing is not None:
            reference = self.references[existing]
            changed = candidate.score > reference.score
            if changed:
                reference.score = candidate.score
            return self.root(existing), changed

        index = len(self.references)
        self.references.append(candidate)
        self._by_wkb[candidate.wkb] = index
        self.wkb_bytes += len(candidate.wkb)
        self._parent.append(index)
        self._members[index] = [index]
        self._keepers[index] = set()
        roots = set()
        neighbours = self._index.intersects(candidate.bbox)
        self.candidate_hits += len(neighbours)
        for other_index in neighbours:
            root = self.root(other_index)
            if root not in roots and matches(candidate, self.references[other_index], geom):
                roots.add(root)

        root = index
        for other in roots:
            root = self._join(root, other)
        feature = QgsFeature(index)
        feature.setGeometry(geom)
        self._index.addFeature(feature)
        return root, True

    def _join(self, first: int, second: int) -> int:
        first, second = self.root(first), self.root(second)
        if first == second:
            return first
        if len(self._members[first]) < len(self._members[second]):
            first, second = second, first
        self._parent[second] = first
        self._members[first].extend(self._members.pop(second))
        keepers = self._keepers.pop(second)
        self._keepers[first].update(keepers)
        for fid in keepers:
            self._keeper_root[fid] = first
        return first

    def members(self, root: int) -> list[FragmentReference]:

        return sorted((self.references[i] for i in self._members[self.root(root)]),
                      key=lambda reference: (-reference.area, reference.wkb))

    def keepers(self, root: int) -> set[int]:
        return set(self._keepers[self.root(root)])

    def bind(self, root: int, fids) -> None:
        root = self.root(root)
        for fid in self._keepers[root]:
            self._keeper_root.pop(fid, None)
        self._keepers[root] = set(fids)
        for fid in self._keepers[root]:
            self._keeper_root[fid] = root

    def replace_keeper(self, fid: int, replacements) -> None:

        root = self._keeper_root.pop(fid, None)
        if root is None:
            return
        root = self.root(root)
        self._keepers[root].discard(fid)
        for replacement in replacements:
            self._keepers[root].add(replacement)
            self._keeper_root[replacement] = root
