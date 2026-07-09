













from __future__ import annotations

from qgis.core import Qgis

from ....core import detection_history
from ....core.i18n import tr
from ....core.logging_utils import log
from ....core.presets.segmentation_presets import pick_label
from .common import _fmt_count, _relative_when


def merge_local_recents(history: list[dict], legacy: list[dict]) -> list[dict]:



    merged = list(history)
    seen = {(e.get("prompt") or "").strip() for e in merged}
    merged.extend(
        e for e in legacy if (e.get("prompt") or "").strip() not in seen)
    merged.sort(key=lambda e: str(e.get("ts") or ""), reverse=True)
    return merged


def recent_view(entry: dict, by_token: dict) -> dict:




    token = entry.get("prompt", "")
    known = by_token.get(token)
    label = pick_label(known.get("label"), token) if known else token
    bits: list[str] = []
    objects = entry.get("objects")
    det = entry.get("detections")
    if isinstance(objects, int):
        bits.append(
            tr("1 object") if objects == 1
            else tr("{n} objects").format(n=_fmt_count(objects)))
    elif isinstance(det, int):
        bits.append(
            tr("1 detection") if det == 1
            else tr("{n} detections").format(n=_fmt_count(det)))
    when = _relative_when(entry.get("ts", ""))
    if when:
        bits.append(when)
    view = {"prompt": token, "label": label, "_meta": "  ·  ".join(bits)}
    thumb = detection_history.thumb_abspath(entry)
    if thumb:
        view["_thumb"] = thumb



    for key in ("extent", "zone_wkt", "crs", "layer_name"):
        if entry.get(key):
            view[key] = entry[key]
    return view


def restore_recent_on_map(plugin, entry: dict) -> None:



    try:
        canvas = _map_canvas(plugin)
        if canvas is not None:
            _zoom_to_recent_extent(canvas, entry)
        _activate_recent_layer(plugin, entry)
    except Exception as err:  # noqa: BLE001
        log(f"Recent restore skipped: {err}", Qgis.MessageLevel.Info)


def _iface(plugin):


    iface_obj = getattr(plugin, "iface", None)
    if iface_obj is None:
        try:
            from qgis.utils import iface as qgis_iface
            iface_obj = qgis_iface
        except ImportError:
            return None
    return iface_obj


def _map_canvas(plugin):
    iface_obj = _iface(plugin)
    try:
        return iface_obj.mapCanvas() if iface_obj is not None else None
    except (RuntimeError, AttributeError):
        return None


def _zoom_to_recent_extent(canvas, entry: dict) -> None:



    from qgis.core import (
        QgsCoordinateReferenceSystem,
        QgsCoordinateTransform,
        QgsProject,
        QgsRectangle,
    )
    ext = entry.get("extent")
    authid = str(entry.get("crs") or "")
    if not ext or len(ext) != 4 or not authid:
        return
    rect = QgsRectangle(
        float(ext[0]), float(ext[1]), float(ext[2]), float(ext[3]))
    if rect.isEmpty():
        return
    src = QgsCoordinateReferenceSystem(authid)
    if not src.isValid():
        return
    dest = canvas.mapSettings().destinationCrs()
    if dest.isValid() and src != dest:
        try:
            xform = QgsCoordinateTransform(src, dest, QgsProject.instance())
            rect = xform.transformBoundingBox(rect)
        except Exception:  # noqa: BLE001
            return
        if rect.isEmpty():
            return
    rect.scale(1.1)
    canvas.setExtent(rect)
    canvas.refresh()


def _activate_recent_layer(plugin, entry: dict) -> None:


    name = str(entry.get("layer_name") or "")
    if not name:
        return
    from qgis.core import QgsProject
    layers = QgsProject.instance().mapLayersByName(name)
    if not layers:
        return
    iface_obj = _iface(plugin)
    if iface_obj is not None:
        iface_obj.setActiveLayer(layers[0])
