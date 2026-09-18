







from __future__ import annotations

from typing import Any

from qgis.core import Qgis, QgsFeature, QgsGeometry, QgsSpatialIndex

from .polygon_masks import (
    polygonal_part_of,
)
from .qt_compat import PolygonGeometry
from .shape_policy_dials import close_max_area_growth, smooth_area_keep, smooth_diet_fraction


def _overlap_metrics(g1: QgsGeometry, g2: QgsGeometry) -> tuple[float, float]:








    if g1 is None or g2 is None:
        return 0.0, 0.0
    if g1.isEmpty() or g2.isEmpty():
        return 0.0, 0.0
    try:
        intersection = g1.intersection(g2)
        if intersection is None or intersection.isEmpty():
            return 0.0, 0.0
        inter_area = intersection.area()
        if inter_area <= 0.0:
            return 0.0, 0.0
        a1 = g1.area()
        a2 = g2.area()
        union_area = a1 + a2 - inter_area
        min_area = min(a1, a2)
        iou = inter_area / union_area if union_area > 0.0 else 0.0
        containment = inter_area / min_area if min_area > 0.0 else 0.0
        return iou, containment
    except Exception:
        return 0.0, 0.0


def _buffer_square_corners(g: QgsGeometry, dist: float) -> QgsGeometry | None:






    cap = getattr(getattr(Qgis, "EndCapStyle", None), "Round", None)
    join = getattr(getattr(Qgis, "JoinStyle", None), "Miter", None)
    if cap is None or join is None:
        cap = getattr(QgsGeometry, "CapRound", None)
        join = getattr(QgsGeometry, "JoinStyleMiter", None)
    if cap is not None and join is not None:
        try:
            return g.buffer(dist, 8, cap, join, 2.0)
        except (TypeError, AttributeError):
            pass
    return g.buffer(dist, 8)


def _keep_largest_part(g: QgsGeometry) -> QgsGeometry | None:






    try:
        if g is None or g.isEmpty() or not g.isMultipart():
            return g
        parts = g.asGeometryCollection()
        if not parts:
            return g
        best = max(parts, key=lambda p: p.area())
        return QgsGeometry(best)
    except Exception:  # noqa: BLE001  # nosec B110
        return None





_CLOSE_MAX_AREA_GROWTH = 1.25


def _geometry_part_count(geom: QgsGeometry) -> int:


    try:
        if not geom.isMultipart():
            return 1
        return int(geom.constGet().numGeometries())
    except Exception:  # noqa: BLE001
        return 1


def despike_thin_necks(
    g: QgsGeometry,
    despike_m: float,
    *,
    preserve_parts: bool = False,
) -> QgsGeometry:


















    if g is None or g.isEmpty() or not despike_m or despike_m <= 0.0:
        return g
    try:
        shrunk = _buffer_square_corners(g, -despike_m)
        if shrunk is None or shrunk.isEmpty():
            return g
        grown = _buffer_square_corners(shrunk, despike_m)
        if grown is None or grown.isEmpty():
            return g
        return grown if preserve_parts else (_keep_largest_part(grown) or grown)
    except Exception:  # noqa: BLE001  # nosec B110
        return g





_SMOOTH_AREA_KEEP = 0.5



_SMOOTH_DIET_FRACTION = 0.5


def _mean_segment_length(g: QgsGeometry) -> float:





    try:
        length = float(g.length())
        count = int(g.constGet().nCoordinates()) if g.constGet() else 0
    except (AttributeError, TypeError, ValueError):
        return 0.0
    if length <= 0.0 or count < 2:
        return 0.0
    return length / float(count)


def rounded_corner_outline(
    g: QgsGeometry, simplify_tol: float = 0.0,
    settings: dict | None = None,
) -> QgsGeometry:




























    if g is None or g.isEmpty():
        return g
    passes, offset, max_angle = 1, 0.25, 120.0
    area_keep = smooth_area_keep(_SMOOTH_AREA_KEEP)
    diet = smooth_diet_fraction(_SMOOTH_DIET_FRACTION)
    try:
        served = settings
        if served is None:
            from .detection_policy import smooth_pass_settings

            served = smooth_pass_settings()
        passes = int(served["iterations"])
        offset = float(served["offset"])
        max_angle = float(served["max_angle_deg"])
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    try:
        src = g
        tol = float(simplify_tol) if simplify_tol and simplify_tol > 0.0 else 0.0





        if tol > 0.0 and _mean_segment_length(src) < tol:
            thinned = src.simplify(tol)
            if (thinned is not None and not thinned.isEmpty()
                    and thinned.area() >= area_keep * src.area()):
                src = thinned




        r = src.smooth(passes, offset, -1.0, max_angle)


        if r is None or r.isEmpty() or r.area() < area_keep * src.area():
            return src
        g = r
        if tol > 0.0:



            r2 = g.simplify(tol * diet)
            if (r2 is not None and not r2.isEmpty()
                    and r2.area() >= area_keep * g.area()):
                g = r2
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return g


def _regularize_shape_kwargs(
    diagonal_reduction: float | None,
    circle_threshold: float | None,
    multi_direction: bool = False,
    multi_max_groups: int | None = None,
    multi_min_separation_deg: float | None = None,
) -> dict:










    kwargs: dict = {}
    if diagonal_reduction is not None:
        kwargs["diagonal_reduction"] = float(diagonal_reduction)
    if circle_threshold is not None and circle_threshold > 0:
        kwargs["circle_threshold"] = float(circle_threshold)
    if multi_direction:
        kwargs["multi_direction"] = True
        if multi_max_groups is not None:
            kwargs["multi_max_groups"] = int(multi_max_groups)
        if multi_min_separation_deg is not None:
            kwargs["multi_min_separation_deg"] = float(multi_min_separation_deg)
    return kwargs


def apply_geometry_refinement(
    geom: QgsGeometry,
    *,
    smooth_settings: dict | None = None,
    simplify_tol: float = 0.0,
    smooth: bool = False,
    expand_dist: float = 0.0,
    fill_holes: bool = False,
    fill_holes_max_area: float = 0.0,
    open_dist: float = 0.0,
    close_dist: float = 0.0,
    despike_m: float = 0.0,
    vertex_spacing: float = 0.0,
    vertex_min: int = 8,
    vertex_max_deviation: float = 0.0,
    vertex_max_deviation_fraction: float = 0.0,
    vertex_keep_fraction: float = 0.0,
    vertex_dial_max_cap_fraction: float | None = None,
    ortho: bool = False,
    ortho_tol: float = 0.0,
    regularize: bool = False,
    regularize_tol: float = 0.0,
    allow_diagonal: bool = True,
    allow_circles: bool = False,
    regularize_min_iou: float = 0.0,
    diagonal_reduction: float | None = None,
    circle_threshold: float | None = None,
    multi_direction: bool = False,
    multi_max_groups: int | None = None,
    multi_min_separation_deg: float | None = None,
    envelope: Any = None,
    unit_aspect: float = 1.0,
) -> QgsGeometry:











































































    if geom is None or geom.isEmpty():
        return geom
    g = geom





    if g.type() != PolygonGeometry:
        areal = polygonal_part_of(g)
        if areal is None or areal.isEmpty():
            return geom
        g = areal





    preserve_input_parts = bool(g.isMultipart())
    budget_on = bool((vertex_spacing and vertex_spacing > 0.0) or (vertex_keep_fraction and vertex_keep_fraction > 0.0))
    if ortho and ortho_tol > 0.0:








        simplify_tol = max(simplify_tol or 0.0, ortho_tol)
    if fill_holes:
        from .hole_size import REMOVE_ALL_RINGS

        cutoff = (float(fill_holes_max_area) if fill_holes_max_area and fill_holes_max_area > 0.0 else REMOVE_ALL_RINGS)
        try:
            r = g.removeInteriorRings(cutoff)
            if isinstance(r, QgsGeometry) and not r.isEmpty():
                g = r
        except (AttributeError, TypeError, ValueError):
            pass







    from .ground_frame import stretch_y, unstretch_y, usable_aspect

    aspect = usable_aspect(unit_aspect)
    unstretched = g
    stretched = stretch_y(g, aspect)
    if stretched is not None:
        g = stretched


        step_aspect = 1.0
    else:
        unstretched = None
        step_aspect = aspect


    g = despike_thin_necks(g, despike_m, preserve_parts=preserve_input_parts)
    if close_dist and close_dist > 0.0:











        try:
            before_parts = _geometry_part_count(g)
            before_area = g.area()



            grown = _buffer_square_corners(g, close_dist)
            r = (None if grown is None or grown.isEmpty()
                 else _buffer_square_corners(grown, -close_dist))
            if (r is not None and not r.isEmpty()
                    and _geometry_part_count(r) >= before_parts
                    and (before_area <= 0.0
                         or r.area() <= before_area * close_max_area_growth(_CLOSE_MAX_AREA_GROWTH))):
                g = r
        except Exception:  # noqa: BLE001  # nosec B110
            pass
    if open_dist and open_dist > 0.0:









        try:
            before_parts = _geometry_part_count(g)
            r = g.buffer(-open_dist, 8).buffer(open_dist, 8)
            if (r is not None and not r.isEmpty()
                    and _geometry_part_count(r) >= before_parts):
                g = r
        except Exception:  # noqa: BLE001  # nosec B110
            pass
    if simplify_tol and simplify_tol > 0.0:
        try:
            r = g.simplify(simplify_tol)
            if r is not None and not r.isEmpty():
                g = r
        except Exception:  # noqa: BLE001  # nosec B110
            pass
    if budget_on:



        try:
            from .vertex_budget import (
                _DIAL_MAX_CAP_NARROW_FRACTION,
                simplify_to_budget,
            )




            dial_cap = (_DIAL_MAX_CAP_NARROW_FRACTION
                        if vertex_dial_max_cap_fraction is None
                        else float(vertex_dial_max_cap_fraction))
            r = simplify_to_budget(
                g, spacing=vertex_spacing, min_vertices=vertex_min,
                max_deviation=vertex_max_deviation,
                max_deviation_fraction=vertex_max_deviation_fraction,
                keep_fraction=vertex_keep_fraction,
                dial_max_cap_fraction=dial_cap,





                unit_aspect=step_aspect)
            if r is not None and not r.isEmpty():
                g = r
        except Exception:  # noqa: BLE001  # nosec B110
            pass
    regularized_ok = False
    if regularize:





        tol = regularize_tol if regularize_tol > 0.0 else simplify_tol
        if tol and tol > 0.0:
            try:
                from .building_regularizer import regularize_qgs_geometry_ex
                res = regularize_qgs_geometry_ex(
                    g,
                    tolerance_m=tol,
                    allow_diagonal=allow_diagonal,
                    allow_circles=allow_circles,
                    min_keep_iou=regularize_min_iou,
                    policy=envelope,


                    unit_aspect=step_aspect,
                    **_regularize_shape_kwargs(
                        diagonal_reduction, circle_threshold,
                        multi_direction, multi_max_groups,
                        multi_min_separation_deg),
                )
                if res.regularized and not res.geometry.isEmpty():
                    g = res.geometry
                    regularized_ok = True
            except Exception:  # noqa: BLE001  # nosec B110
                pass







    if ortho and not regularized_ok and not regularize:



        try:
            r = g.orthogonalize(1.0e-8, 1000, 15.0)
            if r is not None and not r.isEmpty():
                g = r
        except Exception:  # noqa: BLE001  # nosec B110
            pass
    if expand_dist and expand_dist != 0.0:
        try:



            shrinking = expand_dist < 0.0
            before_parts = _geometry_part_count(g) if shrinking else 0


            r = (
                _buffer_square_corners(g, expand_dist)
                if (ortho or regularize)
                else g.buffer(expand_dist, 8)
            )
            if (r is not None and not r.isEmpty()
                    and (not shrinking
                         or _geometry_part_count(r) >= before_parts)):
                g = r
        except Exception:  # noqa: BLE001  # nosec B110
            pass
    if smooth and not (ortho or regularize):











        g = rounded_corner_outline(g, simplify_tol, smooth_settings)
    if unstretched is not None:



        back = unstretch_y(g, aspect)
        g = back if back is not None else unstretched
    return g


def apply_right_angles(
    geom: QgsGeometry,
    destair_tol: float = 0.0,
    *,
    tolerance_m: float = 0.0,
    allow_diagonal: bool = True,
    allow_circles: bool = False,
    min_keep_iou: float = 0.7,
    diagonal_reduction: float | None = None,
    circle_threshold: float | None = None,
    multi_direction: bool = False,
    multi_max_groups: int | None = None,
    multi_min_separation_deg: float | None = None,
    envelope: Any = None,
    unit_aspect: float = 1.0,
) -> QgsGeometry:



































    if geom is None or geom.isEmpty():
        return geom
    g = geom
    destaired = False
    if destair_tol and destair_tol > 0.0:
        try:
            r = g.simplify(destair_tol)
            if r is not None and not r.isEmpty():
                g = r
                destaired = True
        except Exception:  # noqa: BLE001  # nosec B110
            pass
    tol = tolerance_m if tolerance_m and tolerance_m > 0.0 else destair_tol
    if tol and tol > 0.0:
        try:
            from .building_regularizer import regularize_qgs_geometry_ex
            res = regularize_qgs_geometry_ex(
                g,
                tolerance_m=tol,
                allow_diagonal=allow_diagonal,
                allow_circles=allow_circles,
                min_keep_iou=min_keep_iou,
                policy=envelope,
                unit_aspect=unit_aspect,
                **_regularize_shape_kwargs(
                    diagonal_reduction, circle_threshold,
                    multi_direction, multi_max_groups,
                    multi_min_separation_deg),
            )



            if res.regularized and not res.geometry.isEmpty():
                return res.geometry
        except Exception:  # noqa: BLE001  # nosec B110
            pass



    if not destaired and tol and tol > 0.0:
        try:
            r = geom.simplify(tol)
            if r is not None and not r.isEmpty():
                return r
        except Exception:  # noqa: BLE001  # nosec B110
            pass
    return g


def shape_polygon_geometry(
    geom: QgsGeometry,
    mupp: float,
    simplify_px: float = 0.0,
    smooth: bool = False,
    expand_px: int = 0,
    fill_holes: bool = False,
    ortho: bool = False,
    fill_holes_max_area: float = 0.0,
    *,
    open_dist: float = 0.0,
    vertex_spacing: float = 0.0,
    vertex_min: int = 8,
    vertex_max_deviation: float = 0.0,
    vertex_max_deviation_fraction: float = 0.0,
    vertex_keep_fraction: float = 0.0,
    vertex_dial_max_cap_fraction: float | None = None,
    regularize_tol: float = 0.0,
    destair_tol: float = 0.0,
    allow_diagonal: bool = True,
    allow_circles: bool = False,
    min_keep_iou: float = 0.7,
    diagonal_reduction: float | None = None,
    circle_threshold: float | None = None,
    multi_direction: bool = False,
    multi_max_groups: int | None = None,
    multi_min_separation_deg: float | None = None,
    envelope: Any = None,
    unit_aspect: float = 1.0,
) -> QgsGeometry:



















































    if geom is None or geom.isEmpty() or not mupp or mupp <= 0:
        return geom
    g = QgsGeometry(geom)
    try:
        if fill_holes:
            if fill_holes_max_area and fill_holes_max_area > 0.0:
                r = g.removeInteriorRings(float(fill_holes_max_area))
            else:
                parts = g.asMultiPolygon() if g.isMultipart() else [g.asPolygon()]
                shells = [[rings[0]] for rings in parts if rings]
                r = QgsGeometry.fromMultiPolygonXY(shells) if shells else None
            if r is not None and not r.isEmpty():
                g = r






        if open_dist and open_dist > 0.0:
            before_parts = _geometry_part_count(g)
            r = g.buffer(-open_dist, 8).buffer(open_dist, 8)
            if (r is not None and not r.isEmpty()
                    and _geometry_part_count(r) >= before_parts):
                g = r



        tolerance = simplify_px * mupp if simplify_px > 0 else 0.0
        if tolerance > 0:
            r = g.simplify(tolerance)
            if r is not None and not r.isEmpty():
                g = r
        if ((vertex_spacing and vertex_spacing > 0.0) or (vertex_keep_fraction and vertex_keep_fraction > 0.0)):


            from .vertex_budget import (
                _DIAL_MAX_CAP_NARROW_FRACTION,
                simplify_to_budget,
            )



            dial_cap = (_DIAL_MAX_CAP_NARROW_FRACTION
                        if vertex_dial_max_cap_fraction is None
                        else float(vertex_dial_max_cap_fraction))
            r = simplify_to_budget(
                g, spacing=vertex_spacing, min_vertices=vertex_min,
                max_deviation=vertex_max_deviation,
                max_deviation_fraction=vertex_max_deviation_fraction,
                keep_fraction=vertex_keep_fraction,
                dial_max_cap_fraction=dial_cap,




                unit_aspect=unit_aspect)
            if r is not None and not r.isEmpty():
                g = r
        if ortho:



            reg_tol = regularize_tol if regularize_tol and regularize_tol > 0.0 else 1.5 * mupp
            destair = destair_tol if destair_tol and destair_tol > 0.0 else reg_tol
            g = apply_right_angles(
                g, destair_tol=max(0.0, destair - tolerance),
                tolerance_m=reg_tol,
                allow_diagonal=allow_diagonal,
                allow_circles=allow_circles,
                min_keep_iou=min_keep_iou,
                diagonal_reduction=diagonal_reduction,
                circle_threshold=circle_threshold,
                multi_direction=multi_direction,
                multi_max_groups=multi_max_groups,
                multi_min_separation_deg=multi_min_separation_deg,
                unit_aspect=unit_aspect,
                envelope=envelope)
        if expand_px:




            dist = expand_px * mupp
            shrinking = dist < 0.0
            before_parts = _geometry_part_count(g) if shrinking else 0
            r = _buffer_square_corners(g, dist) if ortho else g.buffer(dist, 8)
            if (r is not None and not r.isEmpty()
                    and (not shrinking
                         or _geometry_part_count(r) >= before_parts)):
                g = r
        if smooth and not ortho:









            g = rounded_corner_outline(g, tolerance)
        if g.isGeosValid() is False:
            r = g.makeValid()
            if r is not None and not r.isEmpty():
                g = r
    except Exception:  # noqa: BLE001
        return geom
    return g if g is not None and not g.isEmpty() else geom


def suppress_redundant_hypotheses(
    items: list[tuple[QgsGeometry, float]],
    ios_threshold: float = 0.5,
    dup_ios_floor: float = 0.3,
    dup_centroid_frac: float = 0.35,
) -> list[tuple[QgsGeometry, float]]:























    if len(items) < 2:
        return list(items)
    order = sorted(items, key=lambda t: -t[1])
    kept: list[tuple[QgsGeometry, float]] = []
    kept_meta: list[tuple] = []
    overlap_floor = min(ios_threshold, dup_ios_floor)

    def _meta(geom):
        bb = geom.boundingBox()
        c = geom.centroid().asPoint()
        return bb, geom.area(), c, max(bb.width(), bb.height())

    for geom, score in order:
        bb, area, cen, dim = _meta(geom)
        conflict = False
        for (kbb, karea, kcen, kdim), kept_pair in zip(kept_meta, kept):



            iw = min(bb.xMaximum(), kbb.xMaximum()) - max(bb.xMinimum(), kbb.xMinimum())
            ih = min(bb.yMaximum(), kbb.yMaximum()) - max(bb.yMinimum(), kbb.yMinimum())
            if iw <= 0.0 or ih <= 0.0:
                continue
            small = min(karea, area)
            if small <= 0.0 or (iw * ih) / small < overlap_floor:
                continue
            inter = geom.intersection(kept_pair[0])
            ia = inter.area() if inter is not None and not inter.isEmpty() else 0.0
            ios = ia / small
            if ios >= ios_threshold:
                conflict = True
                break
            if ios >= dup_ios_floor:
                smax = dim if area <= karea else kdim
                if smax > 0.0:
                    dist = ((cen.x() - kcen.x()) ** 2 + (cen.y() - kcen.y()) ** 2) ** 0.5
                    if dist < dup_centroid_frac * smax:
                        conflict = True
                        break
        if not conflict:
            kept.append((geom, score))
            kept_meta.append((bb, area, cen, dim))
    return kept






COVER_THRESHOLD_DEFAULT = 0.40


def drop_covered_objects(
    items: list[tuple[int, QgsGeometry, float]],
    cover_threshold: float | None = None,
) -> list[tuple[int, QgsGeometry, float]]:































    sweep = CoverSweep(items, cover_threshold=cover_threshold)
    sweep.step(len(items))
    return sweep.result()


class CoverSweep:






    def __init__(
        self,
        items: list[tuple[int, QgsGeometry, float]],
        cover_threshold: float | None = None,
    ) -> None:
        self._items = items


        self._threshold = (
            COVER_THRESHOLD_DEFAULT if cover_threshold is None
            else float(cover_threshold)
        )
        if cover_threshold is None:
            try:
                from .detection_policy import merge_scalar

                self._threshold = merge_scalar("cover_threshold", COVER_THRESHOLD_DEFAULT)
            except Exception:  # noqa: BLE001  # nosec B110
                pass
        self._n = len(items)
        self._i = 0
        self._keep = [True] * self._n
        self._done = self._n < 2
        self._metas: list = []
        self._index = QgsSpatialIndex()
        if self._done:
            self._order: list[int] = []
            return
        for pos, (_sid, geom, _score) in enumerate(items):
            try:
                bb, area = geom.boundingBox(), geom.area()
            except Exception:
                self._metas.append((None, 0.0))
                continue
            self._metas.append((bb, area))
            feat = QgsFeature(pos)
            feat.setGeometry(QgsGeometry.fromRect(bb))



            self._index.addFeature(feat)



        self._order = sorted(
            range(self._n), key=lambda pos: -self._metas[pos][1])

    def step(self, max_items: int) -> bool:

        if self._done:
            return True
        items, metas, index, keep = self._items, self._metas, self._index, self._keep
        threshold = self._threshold
        processed = 0
        while self._i < self._n and processed < max_items:
            i = self._order[self._i]
            self._i += 1
            processed += 1
            bb, area = metas[i]
            if bb is None or area <= 0.0:
                continue
            covers = []
            best_cover_score = None









            engine = None
            for j in index.intersects(bb):
                if j == i:
                    continue
                jbb, jarea = metas[j]
                if jbb is None or jarea <= area or not keep[j]:
                    continue
                iw = min(bb.xMaximum(), jbb.xMaximum()) - max(bb.xMinimum(), jbb.xMinimum())
                ih = min(bb.yMaximum(), jbb.yMaximum()) - max(bb.yMinimum(), jbb.yMinimum())
                if iw <= 0.0 or ih <= 0.0 or (iw * ih) / area < threshold * 0.5:
                    continue
                try:
                    if engine is None:
                        engine = QgsGeometry.createGeometryEngine(items[i][1].constGet())
                        engine.prepareGeometry()
                    jg = items[j][1].constGet()
                    if not engine.intersects(jg):
                        continue
                    raw = engine.intersection(jg)
                    inter = None if raw is None else QgsGeometry(raw)
                except Exception:  # nosec B112
                    continue
                if inter is None or inter.isEmpty():
                    continue
                covers.append(inter)
                s = items[j][2]
                if best_cover_score is None or s > best_cover_score:
                    best_cover_score = s
            if not covers:
                continue




            if len(covers) == 1:
                merged = covers[0]
            else:
                merged = QgsGeometry.unaryUnion(covers)
                if merged is None or merged.isEmpty():
                    merged = covers[0]
                    for g in covers[1:]:
                        u = merged.combine(g)
                        if u is not None and not u.isEmpty():
                            merged = u
            if merged.area() / area >= threshold:
                if best_cover_score is not None and items[i][2] > best_cover_score:
                    continue
                keep[i] = False
        if self._i >= self._n:
            self._done = True
        return self._done

    def result(self) -> list[tuple[int, QgsGeometry, float]]:
        return [it for it, k in zip(self._items, self._keep) if k]





OVERLAP_COUNT_MAX_OBJECTS = 5000

OVERLAP_COUNT_MAX_PAIRS = 20000


def count_overlapping_pairs(geoms: list) -> int | None:











    from .server_dials import dial_in_range
    max_objects = dial_in_range(
        "tuning.export.overlap_count_max_objects", OVERLAP_COUNT_MAX_OBJECTS, 100, 50_000)
    max_pairs = dial_in_range(
        "tuning.export.overlap_count_max_pairs", OVERLAP_COUNT_MAX_PAIRS, 500, 200_000)
    usable = [g for g in geoms if g is not None and not g.isEmpty()]
    if len(usable) < 2 or len(usable) > max_objects:
        return None
    try:
        index = QgsSpatialIndex()
        boxes = []
        for i, geom in enumerate(usable):
            box = geom.boundingBox()
            boxes.append(box)
            index.addFeature(i, box)
        pairs = 0
        examined = 0
        for i, geom in enumerate(usable):
            for j in index.intersects(boxes[i]):
                if j <= i:
                    continue
                examined += 1
                if examined > max_pairs:
                    return None
                other = usable[j]
                inter = geom.intersection(other)
                if (inter is not None and not inter.isEmpty()
                        and inter.area() > 0.0):
                    pairs += 1
        return pairs
    except Exception:  # noqa: BLE001
        return None
