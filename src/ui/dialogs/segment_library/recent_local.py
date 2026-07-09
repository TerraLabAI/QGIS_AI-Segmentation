"""Local "Recently detected" support for the Segment library's Recent tab.

The tab's signed-out / endpoint-less face renders from two local stores:
``core/detection_history.py`` (rich runs recorded at Finish: prompt, zone
extent + CRS authid, exported layer name, object count, thumbnail) and the
legacy prompt-only recents (``core/presets/segment_history``, QSettings).

This module merges the two feeds newest-first, decorates entries for the
card widget (label + "N objects · when" caption + thumbnail path + restore
payload), and implements the one-click restore: zoom the canvas back to the
stored zone (defensive CRS transform, skip on failure) and re-activate the
exported layer when it is still in the project. Everything is best-effort;
the caller's prompt reuse never depends on a restore step succeeding.
"""
from __future__ import annotations

from qgis.core import Qgis

from ....core import detection_history
from ....core.i18n import tr
from ....core.logging_utils import log
from ....core.presets.segmentation_presets import pick_label
from .common import _relative_when


def merge_local_recents(history: list[dict], legacy: list[dict]) -> list[dict]:
    """Rich detection-history runs plus legacy prompt-only recents (kept for
    prompts not covered by any stored run), sorted newest first (both stores
    share the '%Y-%m-%dT%H:%M:%SZ' timestamp shape, so ISO strings sort)."""
    merged = list(history)
    seen = {(e.get("prompt") or "").strip() for e in merged}
    merged.extend(
        e for e in legacy if (e.get("prompt") or "").strip() not in seen)
    merged.sort(key=lambda e: str(e.get("ts") or ""), reverse=True)
    return merged


def recent_view(entry: dict, by_token: dict) -> dict:
    """Decorate a stored recent entry with a display label, a stats line,
    the thumbnail path and the restore payload (zone extent + CRS + exported
    layer name) when the entry carries them. ``by_token`` maps catalogue
    prompt tokens to presets so a known object borrows its localized label."""
    token = entry.get("prompt", "")
    known = by_token.get(token)
    label = pick_label(known.get("label"), token) if known else token
    bits: list[str] = []
    objects = entry.get("objects")
    det = entry.get("detections")
    if isinstance(objects, int):
        bits.append(tr("{n} object(s)").format(n=objects))
    elif isinstance(det, int):
        bits.append(tr("{n} detection(s)").format(n=det))
    when = _relative_when(entry.get("ts", ""))
    if when:
        bits.append(when)
    view = {"prompt": token, "label": label, "_meta": "  ·  ".join(bits)}
    thumb = detection_history.thumb_abspath(entry)
    if thumb:
        view["_thumb"] = thumb
    for key in ("extent", "crs", "layer_name"):
        if entry.get(key):
            view[key] = entry[key]
    return view


def restore_recent_on_map(plugin, entry: dict) -> None:
    """Bring the user back to the run: zoom to the stored zone extent and
    re-activate the exported layer so export/inspect is immediate. Every
    step is best-effort (logged quietly, never raises)."""
    try:
        canvas = _map_canvas(plugin)
        if canvas is not None:
            _zoom_to_recent_extent(canvas, entry)
        _activate_recent_layer(plugin, entry)
    except Exception as err:  # noqa: BLE001 -- restore is best-effort
        log(f"Recent restore skipped: {err}", Qgis.MessageLevel.Info)


def _iface(plugin):
    """The QGIS interface, via the plugin when available (the dialog
    tolerates plugin=None) or the global iface fallback. May be None."""
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
    """Zoom the canvas back to the run's stored zone extent, transformed
    defensively to the current canvas CRS (a failed transform skips the zoom
    rather than sending the user to a wrong place)."""
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
        except Exception:  # noqa: BLE001 -- transform failure: skip the zoom
            return
        if rect.isEmpty():
            return
    rect.scale(1.1)  # a little air around the zone
    canvas.setExtent(rect)
    canvas.refresh()


def _activate_recent_layer(plugin, entry: dict) -> None:
    """If the run's exported layer (by stored name) is still in the project,
    make it the active layer."""
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
