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
# geodesic area (km2, ~2.2 x 2.2 km). All features and detail levels stay
# available under the cap; subscribers are never capped. Enforced at zone
# commit time (interactive draw AND the MCP/headless paths).
FREE_TRIAL_MAX_ZONE_KM2 = 5.0

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

# Live tile processing is cooperatively time-sliced on the GUI thread so a big
# run never freezes QGIS. Each pump turn converts queued masks to geometry for
# at most this many seconds, then yields to the event loop (cursor, repaints,
# window switches stay live) and reschedules itself while work remains.
_AUTO_PUMP_BUDGET_S = 0.02
# The live preview rebuilds the whole selection layer, so it is coalesced to at
# most one repaint per this many ms instead of one per completed tile (a tile
# burst used to trigger a full-layer rebuild storm on the GUI thread).
_AUTO_LIVE_REPAINT_MS = 600
# Per-repaint budget for applying the run's smart preset (fill holes, right
# angles, min size...) to NEWLY arrived objects in the live preview. Refined
# results are cached per merger keeper (keepers are immutable, merges insert a
# new fid), so this only bounds the burst after a dense tile: objects past the
# budget show the cheap simplify fallback for one repaint cycle and pick up
# their preset on the next.
_AUTO_LIVE_REFINE_BUDGET_S = 0.08

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


def _probe_writable(directory: str) -> bool:
    probe = os.path.join(directory, f".ai_seg_write_probe_{os.getpid()}")
    try:
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return True
    except OSError:
        return False


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
    """Keep `worker` alive until its finished signal fires, then release it."""
    # Capture the list object itself (not the module-global name) so a
    # reload of this module can never rebind `_ORPHANED_WORKERS` out from
    # under an already-parked worker and drop its last strong reference.
    bucket = _ORPHANED_WORKERS
    bucket.append(worker)

    def _release() -> None:
        try:
            bucket.remove(worker)
        except ValueError:
            pass
        worker.deleteLater()

    worker.finished.connect(_release)
