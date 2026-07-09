


from __future__ import annotations

import os
import sys
import time
import weakref

from qgis.core import QgsFeatureSink

from ...core.i18n import tr
from ...core.interaction_dials import (
    auto_pump_budget_s,
    live_frame_cost_ratio,
    live_repaint_max_ms,
    live_repaint_ms,
)
from ...core.qt_compat import field_type_double, field_type_int, field_type_string



__all__ = [
    "AutoRerunSignature",
    "SETTINGS_KEY_LAST_MANUAL_SESSION_TS",
    "SETTINGS_KEY_TUTORIAL_SHOWN",
    "_FIELD_TYPE_DOUBLE",
    "_FIELD_TYPE_INT",
    "_FIELD_TYPE_STRING",
    "_RECALL_FLOOR",
    "_RECALL_FLOOR_EXEMPLAR_ONLY",
    "_WEBMERC_MUPP_Z0",
    "_add_features_fast",
    "_add_features_with_ids",
    "_apply_fast_render",
    "_clear_all_features",
    "_debounce_timer",
    "_get_change_path_instructions",
    "_notify_provider_write",
    "_provider_name_for_log",
    "auto_live_repaint_settings",
    "auto_pump_budget",
    "auto_rerun_scope",
    "backend_stalled_flag",
    "clip_served_hint",
    "detach_widget_from_main_window",
    "dir_size_label",
    "free_zone_cap_km2",
    "is_layer_georeferenced",
    "join_orphaned_workers",
    "looks_like_pixel_image",
    "max_tiles_per_run_cap",
    "park_orphaned_worker",
    "pixel_grid_crs",
    "zone_over_free_cap_message",
    "zone_too_large_message",
]




_FIELD_TYPE_STRING = field_type_string()
_FIELD_TYPE_DOUBLE = field_type_double()
_FIELD_TYPE_INT = field_type_int()


SETTINGS_KEY_TUTORIAL_SHOWN = "AISegmentation/tutorial_simple_shown"



SETTINGS_KEY_LAST_MANUAL_SESSION_TS = "AISegmentation/last_manual_session_ts"








FREE_TRIAL_MAX_ZONE_KM2 = 3.0


def free_zone_cap_km2() -> float:






    try:
        from ...core.detection_policy import free_zone_max_km2
        return free_zone_max_km2(FREE_TRIAL_MAX_ZONE_KM2)
    except Exception:  # noqa: BLE001
        return FREE_TRIAL_MAX_ZONE_KM2






SERVED_HINT_MAX_CHARS = 240


def clip_served_hint(text: str) -> str:






    text = (text or "").strip()
    if len(text) <= SERVED_HINT_MAX_CHARS:
        return text
    head = text[:SERVED_HINT_MAX_CHARS]
    cut = head.rfind(" ")
    return (head[:cut] if cut > 0 else head).rstrip(" ,;:.") + "..."


def max_tiles_per_run_cap(zone_km2: float | None = None) -> int:



















    from ...core.tile_manager import (
        MAX_TILES,
        MAX_TILES_FLOOR,
        MAX_TILES_PER_KM2,
    )
    try:
        from ...core.detection_policy import (
            max_tiles_floor,
            max_tiles_per_km2,
            max_tiles_per_run,
        )
        ceiling = max_tiles_per_run(MAX_TILES)
        if zone_km2 is None or zone_km2 <= 0:
            return ceiling
        per_km2 = max_tiles_per_km2(MAX_TILES_PER_KM2)
        floor = max_tiles_floor(MAX_TILES_FLOOR)
    except Exception:  # noqa: BLE001
        ceiling = MAX_TILES
        if zone_km2 is None or zone_km2 <= 0:
            return ceiling
        per_km2 = MAX_TILES_PER_KM2
        floor = MAX_TILES_FLOOR
    try:
        allowed = int(float(zone_km2) * float(per_km2))
    except (TypeError, ValueError):
        return ceiling
    return max(int(floor), min(int(ceiling), allowed))












def zone_too_large_message(max_tiles: int, fallback: str | None = None) -> str:







    from ...core.server_dials import dial_copy

    if fallback is None:
        fallback = tr("Zone too large. Draw a zone of {max} tiles or fewer.")
    text = dial_copy("zone.too_large", fallback)
    return text.replace("{max}", str(int(max_tiles)))


def zone_over_free_cap_message(area_km2: float) -> str:








    from ...core.server_dials import dial_copy

    text = dial_copy(
        "zone.free_cap",
        "Zone is {area} km2; free trial zones go up to {max} km2. "
        "Use a smaller zone, or subscribe to segment areas of any size.")
    return _fill_free_cap_text(text.replace("\n", " "), area_km2, "")


def zone_over_free_cap_lines(area_km2: float, upgrade_url: str) -> tuple[str, str]:











    from ...core.server_dials import dial_copy








    shipped_head = tr(
        "This zone is {area} km². Free zones stop at {max} km².")
    shipped_tail = tr(
        '<a href="{url}">Upgrade to Pro</a> to run this zone as drawn, or make '
        'it smaller.')



    text = dial_copy("zone.free_cap", shipped_head + "\n" + shipped_tail,
                     escape=True)
    head, _, _served_tail = text.partition("\n")
    return (_fill_free_cap_text(head, area_km2, upgrade_url),
            _fill_free_cap_text(shipped_tail, area_km2, upgrade_url))


def _fill_free_cap_text(text: str, area_km2: float, upgrade_url: str) -> str:


    return (text
            .replace("{area}", f"{area_km2:.1f}")
            .replace("{max}", f"{free_zone_cap_km2():g}")
            .replace("{url}", upgrade_url)
            .strip())


def _zone_shape_key(plugin) -> str | None:










    import hashlib

    crs_id = ""
    try:
        crs_id = plugin.iface.mapCanvas().mapSettings().destinationCrs().authid()
    except (RuntimeError, AttributeError):
        crs_id = ""
    outline = ""
    polygon = getattr(plugin, "_auto_zone_polygon", None)
    if polygon is not None:
        try:
            outline = polygon.asWkt(6) or ""
        except (RuntimeError, AttributeError, TypeError):
            outline = ""
    if not outline:

        zone = getattr(plugin, "_auto_zone", None)
        if zone is None:
            return None
        try:
            outline = (f"{zone.xMinimum():.6f} {zone.yMinimum():.6f} "
                       f"{zone.xMaximum():.6f} {zone.yMaximum():.6f}")
        except (AttributeError, TypeError, ValueError):
            return None
    digest = hashlib.sha256(outline.encode("utf-8")).hexdigest()[:16]
    return f"{crs_id}:{digest}"


def auto_rerun_scope(plugin) -> tuple:

    zone_key = _zone_shape_key(plugin)
    layer_id = ""
    try:
        layer = plugin._get_active_raster_layer()
        if layer is not None:
            layer_id = layer.id()
    except (RuntimeError, AttributeError):
        layer_id = ""
    return (zone_key, layer_id)


class AutoRerunSignature(tuple):










    def __new__(cls, plugin, fields, scope):
        obj = super().__new__(cls, fields)
        obj._plugin = weakref.ref(plugin)
        obj._scope = scope
        return obj

    def __eq__(self, other):
        if not isinstance(other, tuple):
            return NotImplemented
        if tuple(self) != tuple(other):
            return False
        plugin = self._plugin()
        return plugin is not None and self._scope == auto_rerun_scope(plugin)

    def __ne__(self, other):
        equal = self.__eq__(other)
        return equal if equal is NotImplemented else not equal

    __hash__ = tuple.__hash__


def backend_stalled_flag(tiles_done: int, warming_ms: int,
                         submit_retries: int, tiles_skipped_network: int,
                         tiles_timed_out: int = 0) -> bool:























    if tiles_done > 0:
        return tiles_timed_out >= tiles_done
    return bool(warming_ms > 0 or submit_retries > 0
                or tiles_skipped_network > 0 or tiles_timed_out > 0)






_RECALL_FLOOR = 0.10





_RECALL_FLOOR_EXEMPLAR_ONLY = 0.20









_WEBMERC_MUPP_Z0 = 156543.033928










_AUTO_PUMP_BUDGET_S = 0.02




_AUTO_LIVE_REPAINT_MS = 300







_AUTO_LIVE_FRAME_COST_RATIO = 3.0

_AUTO_LIVE_REPAINT_MAX_MS = 6000


def auto_pump_budget() -> float:


    return auto_pump_budget_s(_AUTO_PUMP_BUDGET_S)


def auto_live_repaint_settings() -> tuple[int, float, int]:


    return (
        live_repaint_ms(_AUTO_LIVE_REPAINT_MS),
        live_frame_cost_ratio(_AUTO_LIVE_FRAME_COST_RATIO),
        live_repaint_max_ms(_AUTO_LIVE_REPAINT_MAX_MS),
    )





_FAST_INSERT = getattr(
    getattr(QgsFeatureSink, "Flag", QgsFeatureSink), "FastInsert", None
)


def _get_change_path_instructions():

    if sys.platform == "win32":
        steps = tr(
            "1. Open Windows Settings > System > Advanced system settings\n"
            "2. Click 'Environment Variables'\n"
            "3. Under 'User variables', click 'New'\n"
            "4. Variable name: AI_SEGMENTATION_CACHE_DIR\n"
            "5. Variable value: the folder path you want to use\n"
            "6. Click OK and restart QGIS"
        )
    elif sys.platform == "darwin":
        steps = tr(
            "Run this command in Terminal, then restart QGIS:\n\n"
            "launchctl setenv AI_SEGMENTATION_CACHE_DIR /your/path"
        )
    else:
        steps = tr(
            "Add this line to your ~/.bashrc or ~/.profile, "
            "then restart QGIS:\n\n"
            "export AI_SEGMENTATION_CACHE_DIR=/your/path"
        )
    return "{}\n\n{}".format(
        tr("To install in a different folder, set the environment "
           "variable AI_SEGMENTATION_CACHE_DIR:"),
        steps)


def _apply_fast_render(layer) -> None:








    from ...core.output_store import apply_fast_canvas_render

    apply_fast_canvas_render(layer)


def _notify_provider_write(layer) -> None:





















    try:
        layer.dataChanged.emit()
    except (RuntimeError, AttributeError):
        pass


def _clear_all_features(provider) -> None:















    try:
        ids = list(provider.allFeatureIds())
    except (AttributeError, RuntimeError):
        ids = []
    if ids:
        provider.deleteFeatures(ids)
    else:
        provider.truncate()


def _add_features_fast(provider, features) -> bool:









    if _FAST_INSERT is not None:
        result = provider.addFeatures(features, _FAST_INSERT)
    else:
        result = provider.addFeatures(features)

    if isinstance(result, tuple):
        return bool(result[0]) if result else False
    return bool(result)


def _add_features_with_ids(provider, features):





    res = provider.addFeatures(features)
    if isinstance(res, tuple):
        ok, added = res
        return bool(ok), list(added or [])
    return bool(res), []


def _debounce_timer(owner, attr_name: str, parent, interval_ms: int, slot) -> None:







    from qgis.PyQt.QtCore import QTimer

    timer = getattr(owner, attr_name, None)
    if timer is None:
        timer = QTimer(parent)
        timer.setSingleShot(True)
        timer.timeout.connect(slot)
        setattr(owner, attr_name, timer)
    timer.start(interval_ms)


def _provider_name_for_log(layer) -> str:





    try:
        prov = layer.dataProvider()
        return prov.name() if prov is not None else "unknown"
    except (RuntimeError, AttributeError):
        return "unknown"








_ORPHANED_WORKERS: list = []


def park_orphaned_worker(worker) -> None:









    bucket = _ORPHANED_WORKERS
    bucket.append(worker)
    released = {"done": False}

    def _release() -> None:
        if released["done"]:
            return
        released["done"] = True
        try:
            bucket.remove(worker)
        except ValueError:
            pass




        try:
            worker.wait(10000)
        except (RuntimeError, AttributeError):
            pass
        try:
            worker.deleteLater()
        except RuntimeError:
            pass

    worker.finished.connect(_release)






    try:
        finished_in_gap = worker.isFinished() and not worker.isRunning()
    except (RuntimeError, AttributeError):
        finished_in_gap = True
    if finished_in_gap:
        _release()


def join_orphaned_workers(budget_seconds: float) -> int:
























    workers = list(_ORPHANED_WORKERS)
    if not workers:
        return 0
    for worker in workers:
        try:
            worker.requestInterruption()
        except (RuntimeError, AttributeError):
            pass
        try:
            cancel = getattr(worker, "cancel", None)
            if callable(cancel):
                cancel()
        except (RuntimeError, AttributeError, TypeError):
            pass
    deadline = time.monotonic() + budget_seconds
    joined = 0
    for worker in workers:
        left_ms = int(max(0.0, deadline - time.monotonic()) * 1000)
        if left_ms <= 0:
            break
        try:
            if worker.wait(left_ms):
                joined += 1
        except (RuntimeError, AttributeError):
            pass
    return joined


def detach_widget_from_main_window(widget) -> bool:















    if widget is None:
        return False
    from qgis.PyQt import sip
    try:
        if sip.isdeleted(widget):
            return False
    except (RuntimeError, TypeError):
        return False
    try:
        widget.hide()
    except (RuntimeError, AttributeError):
        pass
    try:
        widget.setParent(None)
    except (RuntimeError, AttributeError):
        pass
    try:
        sip.transferto(widget, None)
    except (RuntimeError, TypeError, ValueError):
        pass
    try:
        widget.deleteLater()
    except (RuntimeError, AttributeError):
        return False
    return True


def dir_size_label(path: str) -> str:


    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:  # nosec B112
                    continue
    except OSError:
        return "-"
    mb = total / (1024 * 1024)
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"





_PIXEL_GRID_WKT = (
    'ENGCRS["Pixel grid",EDATUM["Image origin"],CS[Cartesian,2],'
    'AXIS["x",east,ORDER[1]],AXIS["y",north,ORDER[2]],'
    'LENGTHUNIT["metre",1]]',
    'LOCAL_CS["Pixel grid",UNIT["metre",1],AXIS["X",EAST],AXIS["Y",NORTH]]',
)


def pixel_grid_crs():












    from qgis.core import QgsCoordinateReferenceSystem

    for wkt in _PIXEL_GRID_WKT:
        crs = QgsCoordinateReferenceSystem(wkt)
        if crs.isValid():
            return crs
    return QgsCoordinateReferenceSystem("EPSG:3857")




NON_GEOREF_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")


def looks_like_pixel_image(layer) -> bool:



    try:
        source = (layer.source() or "").lower()
    except (RuntimeError, AttributeError):
        return False
    return any(source.endswith(ext) for ext in NON_GEOREF_IMAGE_EXTENSIONS)


def is_layer_georeferenced(layer) -> bool:














    from qgis.core import QgsRasterLayer
    if layer is None or not isinstance(layer, QgsRasterLayer):
        return False
    try:
        if not layer.crs().isValid():
            return False
    except RuntimeError:
        return False
    try:
        source = layer.source().lower()
    except RuntimeError:
        return False
    if any(source.endswith(ext) for ext in NON_GEOREF_IMAGE_EXTENSIONS):
        try:
            extent = layer.extent()
            if extent.xMinimum() == 0 and extent.yMinimum() == 0:

                return False
        except RuntimeError:
            return False
    return True
