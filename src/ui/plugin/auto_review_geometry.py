






from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsMessageLog,
    QgsPointXY,
)

from ...core.interaction_dials import (
    ground_scale_band_deg,
    ground_scale_band_m,
    live_refiner_memo_max,
    rescue_refine_budget_s,
    review_reslice_parked_geoms_max,
    review_reslice_parked_keys_max,
)
from ...core.live_refine import (
    LiveRefiner,
    plain_outline_geom,
    points_dial_fraction,
    refine_review_geom,
)







_LIVE_REFINER_MEMO_MAX = 8








_GROUND_SCALE_BAND_DEG = 0.5
_GROUND_SCALE_BAND_M = 50000.0






_RESLICE_PARKED_KEYS_MAX = 2
_RESLICE_PARKED_GEOMS_MAX = 40000









_RESCUE_REFINE_BUDGET_S = 3.0


def rescue_refine_budget() -> float:


    return rescue_refine_budget_s(_RESCUE_REFINE_BUDGET_S)


def _geom_centre_xy(geom) -> tuple[float, float] | None:








    try:
        if geom is None or geom.isEmpty():
            return None
        bbox = geom.boundingBox()
        if bbox.isNull():
            return None
        centre = bbox.center()
        return float(centre.x()), float(centre.y())
    except (AttributeError, RuntimeError, TypeError):
        return None


class AutoReviewGeometryMixin:


    def _auto_crs_metres_per_unit(self, ref_x: float, ref_y: float) -> float:













        authid = self._auto_crs_authid or "EPSG:4326"
        try:
            crs = QgsCoordinateReferenceSystem(authid)
            if not crs.isValid():
                return 1.0
            geographic = bool(crs.isGeographic())
        except Exception:  # noqa: BLE001
            return 1.0


        bucket = round(ref_y, 2) if geographic else round(ref_y / 10000.0)
        key = (authid, bucket)
        cached = getattr(self, "_auto_crs_mpu", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        factor = 1.0
        try:
            from ...core.layer_conventions import make_area_measurer
            step = 0.001 if geographic else 1.0
            metres = float(make_area_measurer(crs).measureLine(
                QgsPointXY(ref_x, ref_y), QgsPointXY(ref_x + step, ref_y)))
            if metres > 0:
                factor = metres / step
        except Exception:  # noqa: BLE001
            factor = 1.0
        self._auto_crs_mpu = (key, factor)
        return factor

    def _auto_crs_unit_aspect(self, ref_x: float, ref_y: float) -> float:





        authid = self._auto_crs_authid or "EPSG:4326"
        try:
            crs = QgsCoordinateReferenceSystem(authid)
            if not crs.isValid():
                return 1.0
            geographic = bool(crs.isGeographic())
        except Exception:  # noqa: BLE001
            return 1.0
        bucket = round(ref_y, 2) if geographic else round(ref_y / 10000.0)
        key = (authid, bucket)
        cached = getattr(self, "_auto_crs_aspect", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            from ...core.layer_conventions import ground_unit_aspect
            aspect = ground_unit_aspect(crs, ref_x, ref_y)
        except Exception:  # noqa: BLE001
            aspect = 1.0
        self._auto_crs_aspect = (key, aspect)
        return aspect

    def _auto_ground_scale_band(self) -> float:








        authid = getattr(self, "_auto_crs_authid", None) or "EPSG:4326"
        cached = getattr(self, "_auto_crs_scale_band", None)
        if cached is not None and cached[0] == authid:
            return cached[1]
        band = ground_scale_band_m(_GROUND_SCALE_BAND_M)
        try:
            from ...core.layer_conventions import crs_measures_in_ground_metres
            crs = QgsCoordinateReferenceSystem(authid)
            if crs_measures_in_ground_metres(crs):
                band = 0.0
            elif crs.isGeographic():
                band = ground_scale_band_deg(_GROUND_SCALE_BAND_DEG)
        except Exception:  # noqa: BLE001
            band = ground_scale_band_m(_GROUND_SCALE_BAND_M)
        self._auto_crs_scale_band = (authid, band)
        return band

    def _auto_ground_scale_bucket(self, ref_y: float) -> int:


        band = self._auto_ground_scale_band()
        if band <= 0.0:
            return 0
        try:
            return int(float(ref_y) // band)
        except (TypeError, ValueError, OverflowError):
            return 0

    def _refine_geom_for_review(self, base, params: dict, pixel_size: float):









        refiner = self._review_refiner_for(base, params, pixel_size)
        if refiner is None:
            return None
        return refiner.refine(base)

    def _review_refiner_for(self, base, params: dict, pixel_size: float,
                            shape_key: tuple | None = None):


















        ref = _geom_centre_xy(base)
        if ref is None:



            ref = getattr(self, "_review_ground_ref_xy", None)
            if ref is None:
                return None
        else:
            self._review_ground_ref_xy = ref





        key = (getattr(self, "_auto_crs_authid", None),
               self._auto_ground_scale_bucket(ref[1]),
               shape_key if shape_key is not None
               else AutoReviewGeometryMixin._review_shape_key(params, pixel_size))
        refiners = getattr(self, "_review_live_refiners", None)
        if not isinstance(refiners, dict):
            refiners = {}
            self._review_live_refiners = refiners
        refiner = refiners.get(key)
        if refiner is None:

            while len(refiners) >= live_refiner_memo_max(_LIVE_REFINER_MEMO_MAX):
                refiners.pop(next(iter(refiners)))
            refiner = LiveRefiner(
                params, pixel_size,
                self._auto_crs_metres_per_unit(ref[0], ref[1]),
                self._auto_crs_unit_aspect(ref[0], ref[1]))
            refiners[key] = refiner
        return refiner

    @staticmethod
    def _review_shape_key(params: dict, pixel_size: float) -> tuple:




        fill_holes = bool(params.get("fill_holes", False))
        return (
            round(float(params.get("simplify_px", 0) or 0.0), 4),
            bool(params.get("smooth", False)),
            round(float(params.get("expand_px", 0) or 0.0), 4),
            fill_holes,





            round(float(params.get("fill_max_m2", 0) or 0.0), 4) if fill_holes
            else 0.0,
            round(float(params.get("open_px", 0) or 0.0), 4),




            round(float(params.get("close_notches_m", 0) or 0.0), 4),
            bool(params.get("ortho", False)),
            round(float(pixel_size or 0.0), 6),



            round(float(params.get("vertex_spacing_m", 0) or 0.0), 4),
            round(points_dial_fraction(params), 4),





            bool(params.get("snap_boundaries", False)),
        )

    def _boundary_snap_offered(self) -> bool:


        from .review_boundary_snap import boundary_snap_is_offered

        return boundary_snap_is_offered(self)

    def _apply_boundary_snap(self, geoms: list, params: dict) -> list:








        from .review_boundary_snap import apply_boundary_snap_to_set

        return apply_boundary_snap_to_set(self, geoms, params)

    def _begin_boundary_snap(self, geoms: list, params: dict) -> tuple:


        from .review_boundary_snap import begin_boundary_snap_pass

        return begin_boundary_snap_pass(self, geoms, params)

    def _finish_boundary_snap(self, geoms: list, started: tuple) -> list:


        from .review_boundary_snap import finish_boundary_snap_pass

        return finish_boundary_snap_pass(self, geoms, started)

    def _reset_review_refine_cache(self) -> None:















        stop_refine_thread = getattr(self, "_stop_review_refine_thread", None)
        if stop_refine_thread is not None:
            stop_refine_thread()
        self._auto_reslice_cache = {
            "key": None, "geoms": {}, "parked": {}, "areas": {}}

        self._review_refine_seq = {}
        self._review_fid_map = {}
        self._review_live_refiners = {}


        self._boundary_snap_memo = None
        self._gap_fill_memo = None


        self._review_ground_ref_xy = None

    def _invalidate_review_refine(self, indices) -> None:






















        cache = getattr(self, "_auto_reslice_cache", None)
        touched = [int(idx) for idx in indices]



        bump = getattr(self, "_bump_review_refine_seq", None)
        if bump is not None:
            bump(touched)


        self._boundary_snap_memo = None
        self._gap_fill_memo = None
        if isinstance(cache, dict):
            areas = cache.get("areas")
            if isinstance(areas, dict):
                for idx in touched:
                    areas.pop(idx, None)




            entries = [cache.get("geoms")]
            entries.extend((cache.get("parked") or {}).values())
            for geoms in entries:
                if isinstance(geoms, dict):
                    for idx in touched:
                        geoms.pop(idx, None)
        fid_map = getattr(self, "_review_fid_map", None)
        if not touched or not isinstance(fid_map, dict) or not fid_map:
            return
        for idx in touched:
            det_id = self._object_fid_for(idx)
            rec = fid_map.get(det_id)
            if rec is not None:


                fid_map[det_id] = (rec[0], None, rec[2], rec[3])

    def _adopt_reslice_shape_key(self, cache: dict, key: tuple) -> None:













        parked = cache.get("parked")
        if not isinstance(parked, dict):
            parked = {}
            cache["parked"] = parked
        old_key, old_geoms = cache.get("key"), cache.get("geoms")
        if old_key is not None and old_geoms:


            parked.pop(old_key, None)
            parked[old_key] = old_geoms
        revived = parked.pop(key, None)
        held = sum(len(g) for g in parked.values())
        keys_max = review_reslice_parked_keys_max(_RESLICE_PARKED_KEYS_MAX)
        geoms_max = review_reslice_parked_geoms_max(_RESLICE_PARKED_GEOMS_MAX)
        while parked and (len(parked) > keys_max or held > geoms_max):
            held -= len(parked.pop(next(iter(parked))))
        cache["key"] = key
        cache["geoms"] = revived if revived is not None else {}


        cache["areas"] = {}

    @staticmethod
    def _plain_outline_geom(base):












        return plain_outline_geom(base)

    def _review_refined_geom(self, det_idx: int, base, params: dict,
                             pixel_size: float):














        cache = self._auto_reslice_cache
        key = self._review_shape_key(params, pixel_size)
        if cache.get("key") != key:
            self._adopt_reslice_shape_key(cache, key)
        geoms = cache["geoms"]
        if det_idx in geoms:
            return geoms[det_idx]










        _per_shape = getattr(self, "_shape_params_for_object", None)
        _shared_params = params
        if _per_shape is not None:
            params = _per_shape(det_idx, params)









        refiner = self._review_refiner_for(
            base, params, pixel_size,
            shape_key=key if params is _shared_params else None)
        if refiner is None:
            result = self._plain_outline_geom(base)
            geoms[det_idx] = result
            return result
        result, err = refine_review_geom(refiner, base)
        if err is not None:
            self._log_review_refine_failure(det_idx, err)
        geoms[det_idx] = result
        return result

    def _log_review_refine_failure(self, det_idx: int, exc: Exception) -> None:





        try:
            QgsMessageLog.logMessage(
                f"Auto review: refine failed on object {det_idx}, "
                f"kept as traced ({exc})",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        except Exception:  # nosec B110
            pass

    def _note_stitch_shapes_dirty(self, fids) -> None:








        if not fids:
            return
        dirty = getattr(self, "_auto_stitch_dirty_fids", None)
        if dirty is None:
            dirty = set()
            self._auto_stitch_dirty_fids = dirty
        dirty.update(fids)

    def _stitch_shapes_are_reusable(self, pixel_size: float,
                                    objects: list) -> bool:











        if getattr(self, "_auto_stitch_shapes_stale", True):
            return False
        if not objects:


            return False
        stitch_px = float(getattr(self, "_auto_stitch_shape_px", 0.0) or 0.0)
        if abs(stitch_px - float(pixel_size or 0.0)) > 1e-9:
            return False
        factor = float(getattr(self, "_auto_stitch_shape_mpu", 0.0) or 0.0)
        if factor <= 0.0:
            return False









        band = None
        try:




            if self._auto_ground_scale_band() > 0.0:
                for row in objects:
                    ref = _geom_centre_xy(row[0])
                    if ref is None:
                        return False
                    here = self._auto_ground_scale_bucket(ref[1])
                    if band is None:
                        band = here
                    elif here != band:
                        return False
            for row in (objects[0], objects[-1]):
                ref = _geom_centre_xy(row[0])
                if ref is None:
                    return False
                here = self._auto_crs_metres_per_unit(ref[0], ref[1])
                if abs(float(here) - factor) > factor * 1e-6:
                    return False
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return False
        return True

    def _seed_review_refine_cache(self, params: dict, pixel_size: float,
                                  objects: list, object_fids: list) -> int:













        shapes = getattr(self, "_auto_stitch_shapes", None)
        if not shapes or not objects or len(object_fids) != len(objects):
            return 0
        if not self._stitch_shapes_are_reusable(pixel_size, objects):
            return 0
        from ...core.layer_conventions import repair_polygon, to_multipolygon
        cache = self._auto_reslice_cache
        key = self._review_shape_key(params, pixel_size)
        if cache.get("key") != key:
            self._adopt_reslice_shape_key(cache, key)
        geoms = cache["geoms"]
        per_shape = getattr(self, "_shape_params_for_object", None)



        dirty = getattr(self, "_auto_stitch_dirty_fids", None) or ()
        seeded = 0
        for det_idx, fid in enumerate(object_fids):
            if det_idx in geoms or fid in dirty:
                continue
            shape = shapes.get(fid)
            if shape is None:
                continue



            if per_shape is not None and per_shape(det_idx, params) is not params:
                continue
            try:




                g = to_multipolygon(repair_polygon(shape) or shape)
            except Exception:  # noqa: BLE001  # nosec B112
                continue
            if g is None or g.isEmpty():
                continue
            geoms[det_idx] = g
            seeded += 1
        return seeded

    def _compute_visible_objects(
        self, params: dict, pixel_size: float, with_scores: bool = False,
        refine_budget_s: float = 0.0, with_ids: bool = False,
    ) -> list | tuple[list, list] | tuple[list, list, list]:
















        import time as _t

        removed = self._review_removed_fids()



        cache = self._auto_reslice_cache
        shape_key = self._review_shape_key(params, pixel_size)
        if cache.get("key") != shape_key:
            self._adopt_reslice_shape_key(cache, shape_key)
        cached_geoms = cache["geoms"]
        deadline = (_t.monotonic() + refine_budget_s) if refine_budget_s > 0 else None
        unshaped = 0
        unrepairable = 0
        out = []
        out_scores = []
        out_ids = []
        for det_idx, (base, score, area) in enumerate(self._auto_objects):
            if det_idx in removed or base is None or base.isEmpty():
                continue
            if not (self._object_is_manual(det_idx) or self._passes_review_filters(score, area, params)):
                continue
            if deadline is not None and det_idx not in cached_geoms and _t.monotonic() >= deadline:
                g = self._plain_outline_geom(base)
                unshaped += 1
            else:
                g = self._review_refined_geom(det_idx, base, params, pixel_size)
            if g is not None:
                out.append(g)
                out_scores.append(float(score))
                if with_ids:
                    out_ids.append(self._object_fid_for(det_idx))
            else:
                unrepairable += 1
        if unshaped:
            QgsMessageLog.logMessage(
                f"Auto review: rescue export ran out of its {refine_budget_s:.0f}s "
                f"shaping budget; {unshaped} hidden object(s) saved with their "
                f"traced outline", "AI Segmentation", level=Qgis.MessageLevel.Info)
        if unrepairable:


            QgsMessageLog.logMessage(
                f"Auto review: {unrepairable} object(s) left out; their geometry "
                f"could not be repaired", "AI Segmentation",
                level=Qgis.MessageLevel.Warning)


        out = self._apply_boundary_snap(out, params)
        if with_ids:
            return out, out_scores, out_ids
        if with_scores:
            return out, out_scores
        return out
