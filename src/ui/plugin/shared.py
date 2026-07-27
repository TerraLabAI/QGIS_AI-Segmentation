"""Module-level constants and free helper functions shared by the
AISegmentationPlugin mixins (extracted from ai_segmentation_plugin.py so
mixin modules can import them without a circular import)."""
from __future__ import annotations

import os
import sys

from qgis.core import QgsFeatureSink

from ...core.i18n import tr
from ...core.qt_compat import field_type_double, field_type_int, field_type_string

# QgsField type args (QGIS 4 rejects raw int, #25/#36): resolved once in
# qt_compat (QVariant on QGIS 3, QMetaType on QGIS 4). Kept as module constants
# because several mixins import them by name.
_FIELD_TYPE_STRING = field_type_string()
_FIELD_TYPE_DOUBLE = field_type_double()
_FIELD_TYPE_INT = field_type_int()

# QSettings keys for tutorial flags
SETTINGS_KEY_TUTORIAL_SHOWN = "AISegmentation/tutorial_simple_shown"

# QSettings key for the last Manual-session timestamp (drives the predictive
# model warm-up on a returning Manual user, see env_setup._manual_used_recently).
SETTINGS_KEY_LAST_MANUAL_SESSION_TS = "AISegmentation/last_manual_session_ts"

# Free-trial zone cap: a free-tier user's Automatic zone may not exceed this
# geodesic area (km2, ~5 x 5 km). All features and detail levels stay
# available under the cap; subscribers are never capped. Enforced at zone
# commit time (interactive draw AND the MCP/headless paths). What a run at
# that size costs is a separate gate: the credit estimate and the per-run cap
# still apply, so a wide zone buys a coarser grid rather than a free ride.
FREE_TRIAL_MAX_ZONE_KM2 = 25.0


def free_zone_cap_km2() -> float:
    """The free-trial zone cap (km2), resolved from the server detection policy
    when present, else the built-in FREE_TRIAL_MAX_ZONE_KM2. Lets the cap be
    tuned server-side (a growth dial) without a plugin release. Reads the cached
    config only (no network), so it is safe on the GUI thread at zone-commit
    time; fails to the constant so Manual/offline and older servers are
    unaffected."""
    try:
        from ...core.detection_policy import free_zone_max_km2
        return free_zone_max_km2(FREE_TRIAL_MAX_ZONE_KM2)
    except Exception:  # noqa: BLE001 -- policy is best-effort; fall to the constant
        return FREE_TRIAL_MAX_ZONE_KM2


def max_tiles_per_run_cap() -> int:
    """Hard per-run tile (credit) ceiling, resolved from the server detection
    policy when present, else the built-in MAX_TILES. Lets the cost ceiling be
    tuned server-side without a plugin release. Cached config only (no network),
    so it is safe on the GUI thread and in the live credit estimate; fails to
    the constant offline and on older servers."""
    from ...core.tile_manager import MAX_TILES
    try:
        from ...core.detection_policy import max_tiles_per_run
        return max_tiles_per_run(MAX_TILES)
    except Exception:  # noqa: BLE001 -- policy is best-effort; fall to the constant
        return MAX_TILES


# -- the two zone refusals -------------------------------------------------
# Both sentences quote a number that is already a server dial, so they are
# built from the LIVE value here rather than from a copy of it. A refusal that
# names a different limit from the one that refused is worse than no number.
# The wording itself is served copy (ids ``zone.too_large`` and
# ``zone.free_cap``), so a confusing refusal is one deploy away from fixed.
# Placeholders are filled with str.replace, never format(), so a stray brace in
# a served sentence cannot raise on the draw path.


def zone_too_large_message(max_tiles: int) -> str:
    """Why a zone was refused for exceeding the per-run tile ceiling."""
    from ...core.server_dials import dial_copy

    text = dial_copy(
        "zone.too_large",
        tr("Zone too large. Reduce the area to {max} tiles or fewer."))
    return text.replace("{max}", str(int(max_tiles)))


def zone_over_free_cap_message(area_km2: float) -> str:
    """Why a free-tier zone was refused for exceeding the free-trial cap.

    The one-sentence form, for the paths that answer with a plain string (the
    MCP API and the headless run). Those answers go to a program, not to a
    dock label, so the shipped wording is untranslated exactly as it was. The
    dock renders the same refusal as two rich-text lines in
    ``auto_state.set_auto_zone_rejected``; that call site should read this dial
    too rather than keep its own wording.
    """
    from ...core.server_dials import dial_copy

    text = dial_copy(
        "zone.free_cap",
        "Zone is {area} km2; free trial zones go up to {max} km2. "
        "Use a smaller zone, or subscribe to segment areas of any size.")
    return (text
            .replace("{area}", f"{area_km2:.1f}")
            .replace("{max}", f"{free_zone_cap_km2():g}"))


def backend_stalled_flag(tiles_done: int, warming_ms: int,
                         submit_retries: int, tiles_skipped_network: int) -> bool:
    """Whether an ending run was a STALLED BACKEND rather than a healthy run the
    user simply cancelled.

    True only when the run billed ZERO tiles AND the service showed distress:
    time spent in the waiting room (warming_ms), transient submit retries, or
    tiles dropped after their submit-retry budget was exhausted. A user who
    cancels a run that already produced tiles, or a run with no distress signal,
    stays False. Carried on auto_detect_cancelled so a service outage the user
    gives up on does not read as a user-initiated cancel in analytics."""
    if tiles_done > 0:
        return False
    return bool(warming_ms > 0 or submit_retries > 0 or tiles_skipped_network > 0)


# The low recall floor sent to the server so every plausible mask comes back and
# the review confidence slider can re-filter client-side with no re-detection
# (free, instant). The UI default cutoff (_AUTO_DEFAULT_CONFIDENCE) is imported
# from core/review_defaults.py, the single source shared with the dock.
_RECALL_FLOOR = 0.10
# Exemplar-ONLY runs (a drawn example, no text prompt) get a higher floor:
# without the open-vocab text prior the model floods low-confidence visual
# matches (big context blobs), which drowns the merger and the review in
# junk. 0.20 keeps the review slider meaningful while cutting the noise tier;
# a text prompt restores the full 0.10 recall floor.
_RECALL_FLOOR_EXEMPLAR_ONLY = 0.20

# The object currently OPEN for editing (refine/handoff) reuses pending-blue
# (PENDING_STROKE / PENDING_FILL in ui/canvas_palette.py) with a bolder outline,
# so the canvas keeps a two-color language: blue = editable, green = validated.

# EPSG:3857 ground resolution (map units/px) at XYZ zoom 0 with 256 px tiles:
# 2*pi*6378137 / 256. Divided by 2**zmax it gives an online basemap's finest
# native resolution, used to clamp the render so an online source is never
# upsampled past the detail it actually has (mirrors the local native clamp).
_WEBMERC_MUPP_Z0 = 156543.033928

# Automatic-review shape-refine defaults (Simplify / Clean / Round corners /
# Fill holes / Expand) now live in core/review_defaults.py, imported at the top
# as the private _AUTO_REVIEW_* aliases so both this file and the dock share one
# source. No local copies to keep in sync.

# The finalize and reslice chains are cooperatively time-sliced on the GUI
# thread so a big run never freezes QGIS. Each turn works for at most this many
# seconds, then yields to the event loop (cursor, repaints, window switches stay
# live) and reschedules itself while work remains.
_AUTO_PUMP_BUDGET_S = 0.02
# The live preview writes arriving objects to the provider on a coalesced tick
# rather than once per completed tile. The tick applies a DELTA, so a burst of
# tiles costs one write of what changed; this is how often the canvas is asked
# to redraw, not how much work the GUI does per object.
_AUTO_LIVE_REPAINT_MS = 300
# What the live preview may cost the run. A frame's draw time grows with the
# objects already found, and the canvas draws on the same machine as the tiles
# are folded on, so a preview that repaints as fast as the map can draw ends up
# owning the machine for the whole run. After each frame the preview sits out
# this many times its own measured cost before asking for the next one: at 40 ms
# a frame it is imperceptible, at 1.5 s it settles to a repaint every few
# seconds and gives the run its cores back.
_AUTO_LIVE_FRAME_COST_RATIO = 3.0
# However slow one frame gets, the preview still shows something this often.
_AUTO_LIVE_REPAINT_MAX_MS = 6000

# Bulk-insert flag: tells the provider not to populate feature IDs back onto the
# input features, which we never read. A recognised QGIS speed-up for bulk
# addFeatures (live rebuilds + export). Version-defensive (Qt5 flat / Qt6 scoped).
_FAST_INSERT = getattr(
    getattr(QgsFeatureSink, "Flag", QgsFeatureSink), "FastInsert", None
)


def _get_change_path_instructions():
    """Return platform-specific instructions for changing the install path."""
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
    """Make a detection result layer cheap to pan/zoom when it holds thousands of
    polygons. This is the fix for "QGIS lags after a big run": SAM footprints
    carry far more vertices than matter at map scale, and redrawing them all on
    every canvas refresh is what stutters. Two display-only levers (the stored
    geometry is never altered):

    - On-the-fly geometry simplification (QgsVectorSimplifyMethod): QGIS drops
      sub-pixel vertices at draw time, so a zoomed-out canvas redraws a fraction
      of the points. Full detail returns as you zoom in. This is the dominant win.
    - A spatial index on the provider so a pan/zoom fetches only the features in
      view instead of scanning every one.

    Best-effort and version-defensive (Qt5 flat vs Qt6 scoped enums); never
    raises into the caller.
    """
    try:
        from qgis.core import QgsVectorSimplifyMethod

        method = QgsVectorSimplifyMethod()
        hint_scope = getattr(QgsVectorSimplifyMethod, "SimplifyHint", QgsVectorSimplifyMethod)
        algo_scope = getattr(QgsVectorSimplifyMethod, "SimplifyAlgorithm", QgsVectorSimplifyMethod)
        # FullSimplification (not just GeometrySimplification) unlocks QGIS's
        # antialiasing-disabling draw path on TOP of vertex dropping. QGIS only
        # honours it when the threshold is > 1.0, which ours is. Fall back to
        # GeometrySimplification on older builds that lack the flag.
        full_hint = getattr(hint_scope, "FullSimplification",
                            getattr(QgsVectorSimplifyMethod, "FullSimplification", None))
        geom_hint = getattr(hint_scope, "GeometrySimplification",
                            getattr(QgsVectorSimplifyMethod, "GeometrySimplification", None))
        distance_algo = getattr(algo_scope, "Distance",
                                getattr(QgsVectorSimplifyMethod, "Distance", None))
        hint = full_hint if full_hint is not None else geom_hint
        if hint is not None:
            method.setSimplifyHints(hint)
        if distance_algo is not None:
            method.setSimplifyAlgorithm(distance_algo)
        # Threshold in pixels: >1 both simplifies a touch more aggressively AND is
        # the condition QGIS requires to apply FullSimplification's AA path.
        method.setThreshold(1.5)
        method.setForceLocalOptimization(True)
        layer.setSimplifyMethod(method)
    except Exception:  # noqa: BLE001 - display nicety, never break a run on it  # nosec B110
        pass
    try:
        layer.dataProvider().createSpatialIndex()
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def _notify_provider_write(layer) -> None:
    """Tell QGIS that features changed under it after a PROVIDER-level write.

    We rewrite the review layer through the data provider (deleteFeatures /
    addFeatures / truncate / changeGeometryValues) because it is far cheaper than
    an edit session. The catch: a provider write emits ONLY ``repaintRequested``.
    None of the five signals ``QgsPointLocator`` invalidates on (featureAdded,
    featureDeleted, geometryChanged, attributeValueChanged, dataChanged) are
    emitted, because those come from the edit buffer. So QGIS's snapping index
    and the vertex tool's geometry cache keep describing the PREVIOUS features.

    Measured on QGIS 3.44: after a delete + re-add, ``snapToMap`` still returned
    a confident match on a feature id that no longer existed, so clicking a
    vertex did nothing. After ``changeGeometryValues`` it returned a LIVE id
    carrying the OLD shape, so an edit landed on the wrong vertex.
    ``startEditing()`` does not rebuild the index, which is why simply
    re-entering the hand-edit bridge never cleared it.

    ``dataChanged`` is the one call that clears both (the locator destroys its
    index, the vertex tool clears its geometry cache). It is layer-scoped and
    lazy (the index rebuilds on the next snap), unlike ``clearAllLocators()``,
    which would also throw away every other layer's index. Never raises."""
    try:
        layer.dataChanged.emit()
    except (RuntimeError, AttributeError):
        pass


def _clear_all_features(provider) -> None:
    """Empty a provider we rewrite in full, WITHOUT using ``truncate()``.

    ``QgsMemoryProvider::truncate()`` drops the feature map but leaves every
    entry in the spatial index it built. The provider then hands out the same
    small feature ids again, so after K truncate + re-add cycles a bbox request
    returns each feature K times and the renderer paints each polygon K times.
    ``featureCount()`` stays correct throughout, which is why this reads as
    "QGIS got slow" rather than as a bug, and why the stacked translucent fills
    start looking opaque. Calling ``createSpatialIndex()`` again does not repair
    it: the provider returns early when an index already exists.

    Deleting by id keeps the index consistent. It costs more than a truncate on
    a large set, but it runs on the full-rebuild path only, which already
    follows a whole reslice, and it buys back the per-frame multiplication.
    """
    try:
        ids = list(provider.allFeatureIds())
    except (AttributeError, RuntimeError):
        ids = []
    if ids:
        provider.deleteFeatures(ids)
    else:
        provider.truncate()


def _add_features_fast(provider, features) -> None:
    """Bulk-add features with the FastInsert flag when the running QGIS exposes
    it. FastInsert skips writing provider feature IDs back onto the input
    features (which we never read), the standard speed-up for bulk loads."""
    if _FAST_INSERT is not None:
        provider.addFeatures(features, _FAST_INSERT)
    else:
        provider.addFeatures(features)


def _add_features_with_ids(provider, features):
    """Bulk-add features and return (ok, added) where ``added`` holds the
    provider's ASSIGNED feature copies. PyQGIS never writes the new ids back
    onto the features you pass in (the C++ in-out list arrives as a sip copy,
    so reading .id() on an input after the call yields the invalid sentinel):
    any caller that needs the provider fids must read them off ``added``."""
    res = provider.addFeatures(features)
    if isinstance(res, tuple):
        ok, added = res
        return bool(ok), list(added or [])
    return bool(res), []


def _debounce_timer(owner, attr_name: str, parent, interval_ms: int, slot) -> None:
    """Lazily create (once) and (re)start a single-shot debounce timer stored
    as ``owner.<attr_name>``. The plugin controller is a plain class, not a
    QObject, so ``parent`` must be a QObject the caller already relies on to
    own the timer's lifetime (usually ``self.dock_widget``). The slot is
    connected only at creation, never re-connected on later calls; every call
    restarts the countdown (trailing-edge debounce), so the slot fires once
    ``interval_ms`` has passed with no further call."""
    from qgis.PyQt.QtCore import QTimer

    timer = getattr(owner, attr_name, None)
    if timer is None:
        timer = QTimer(parent)
        timer.setSingleShot(True)
        timer.timeout.connect(slot)
        setattr(owner, attr_name, timer)
    timer.start(interval_ms)


def _provider_name_for_log(layer) -> str:
    """Best-effort data-provider name of a raster layer for diagnostic logs
    (e.g. 'wms' for XYZ/WMS basemaps, 'gdal' for local files). Online providers
    make the upfront zone render slow (basemap tile fetches), so this tag is the
    first clue when 'generating tiles' is slow. Never raises; production-safe
    (no URLs/keys, just the provider type)."""
    try:
        prov = layer.dataProvider()
        return prov.name() if prov is not None else "unknown"
    except (RuntimeError, AttributeError):
        return "unknown"


# Workers whose OS thread outlived plugin unload (blocked in a long network
# call). Holding the last Python reference here prevents the C++ QThread
# from being garbage-collected while running, which would hard-abort QGIS
# ("QThread: Destroyed while thread is still running"). Entries remove
# themselves when the thread finishes; on interpreter exit the daemon-like
# leak is bounded by the client's longest timeout (110 s).
_ORPHANED_WORKERS: list = []


def park_orphaned_worker(worker) -> None:
    """Keep `worker` alive until its finished signal fires, then release it.

    Two callers rely on this: a still-running orphan at teardown, and a
    NOT-YET-STARTED worker parked as a lifetime anchor right before its
    start() call (e.g. the manual encode worker). Both must be safe here.
    """
    # Capture the list object itself (not the module-global name) so a
    # reload of this module can never rebind `_ORPHANED_WORKERS` out from
    # under an already-parked worker and drop its last strong reference.
    bucket = _ORPHANED_WORKERS
    bucket.append(worker)
    released = {"done": False}

    def _release() -> None:
        if released["done"]:
            return  # idempotent: the re-check below and the signal can both fire
        released["done"] = True
        try:
            bucket.remove(worker)
        except ValueError:
            pass
        # Join the native thread before destruction: finished() is emitted a
        # hair before the thread has fully stopped, and destroying a QThread
        # inside that window hard-aborts all of QGIS. wait() returns
        # immediately for a thread that never started or already stopped.
        try:
            worker.wait(10000)
        except (RuntimeError, AttributeError):
            pass
        try:
            worker.deleteLater()
        except RuntimeError:
            pass  # C++ object already gone

    worker.finished.connect(_release)
    # A thread can finish between the caller's isRunning() check and the
    # connect above: its finished signal never reaches the slot and the
    # worker would leak in the bucket forever. Release now ONLY when the
    # thread actually ran to completion: a parked-before-start worker
    # (isRunning False, isFinished False) must stay anchored, or the pending
    # delete would destroy the thread the moment start() runs and abort QGIS.
    try:
        finished_in_gap = worker.isFinished() and not worker.isRunning()
    except (RuntimeError, AttributeError):
        finished_in_gap = True  # dead C++ object: just drop the anchor
    if finished_in_gap:
        _release()


def dir_size_label(path: str) -> str:
    """Human-readable total size of a directory tree (cache-size labels,
    removal log lines). Best-effort, never raises."""
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


# Plain image formats that carry no georeferencing of their own (a world file
# or an explicit CRS is what makes one usable on a map).
NON_GEOREF_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")


def looks_like_pixel_image(layer) -> bool:
    """True when the layer's source is a plain image format. Used to tell an
    expected pixel-mode run (a PNG screenshot) apart from a surprising one (a
    GeoTIFF whose CRS is missing), so only the latter warrants a heads-up."""
    try:
        source = (layer.source() or "").lower()
    except (RuntimeError, AttributeError):
        return False
    return any(source.endswith(ext) for ext in NON_GEOREF_IMAGE_EXTENSIONS)


def is_layer_georeferenced(layer) -> bool:
    """True when a raster layer is properly georeferenced.

    A missing CRS is decisive on its OWN, whatever the file extension: without
    one there is no pixel-to-ground mapping, so the layer belongs in Manual's
    pixel-coordinate mode. Deciding this by extension alone (the previous rule)
    classified a CRS-less GeoTIFF/IMG/JP2 as georeferenced, which then failed
    the caller's CRS guard and left Manual with no way to run at all.

    The extension list stays as a SECOND, narrower test: an image format can
    carry a valid CRS and still be laid out at bare pixel coordinates (extent
    anchored at the origin), which the CRS check alone cannot catch.

    Guarded against a layer deleted mid-check (RuntimeError from the sip
    wrapper)."""
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
                # Likely bare pixel dimensions, not ground coordinates.
                return False
        except RuntimeError:
            return False
    return True
