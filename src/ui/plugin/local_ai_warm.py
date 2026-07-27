"""Get the local model and the imagery ready BEFORE the user clicks.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).

A click that needs the model used to pay, in one go and with the user watching:
the model load, the imagery read around the click, and the encode of it. All
three can be paid earlier, on time the user is spending anyway, because the user
announces where they are about to click by moving the cursor there:

- Arriving on the Correct step with AI armed says a polygon is about to be
  picked. That is when the model starts loading.
- Switching to Manual says the same thing more plainly: it is the only mode
  that runs on the local model. That is when the subprocess is pre-started.
- Resting the cursor on a polygon, in the review or on the map in Manual, says
  where the click is going. That is when the imagery there is read and encoded,
  silently.

A warm-up that guesses wrong costs nothing but the work itself: the click starts
its own crop, and imagery already encoded stays in the model's crop cache for
whenever the user does reach that neighbourhood.

What it must never cost is a gesture. A crop warm-up holds the predictor pipe
for several seconds with no cursor and no status line, and Save and Undo both
stand back from a busy pipe, so an invisible warm-up could swallow either one
with nothing on screen to explain it. Two rules keep that from happening: one
never starts while an object is open (`_manual_warm_allowed`), and one already
running gives the pipe up as soon as a gesture arrives for an object
(`_abandon_speculative_manual_crop`).

Online basemaps are deliberately never warmed. Their imagery is not read from a
file: it is pulled through the live layer provider, which the read has to switch
to bilinear resampling and tell to drop its cached tiles. Doing that behind a
cursor movement would reload the user's basemap while they are looking at it.
"""
from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsMessageLog,
    QgsPointXY,
    QgsProject,
)

from .shared import _debounce_timer

# How long the cursor must rest on one polygon before its crop is read. Long
# enough that sweeping the mouse across a dense review reads nothing, short
# enough that a deliberate move onto a polygon is warm by the time it is
# clicked.
CORRECT_HOVER_WARM_MS = 200

# Same idea on the Manual map, a little longer: there is no object under the
# cursor to aim at, so a moving mouse crosses many candidate crops.
MANUAL_HOVER_WARM_MS = 300


class LocalAiWarmMixin:
    """Model load and crop warm-up for the Correct step's AI method."""

    def _correct_ai_warm_allowed(self) -> bool:
        """Is a speculative warm-up appropriate right now?

        Everything here is re-checked at fire time, because a debounced timer
        outlives the state that armed it."""
        if getattr(self, "_headless", False):
            return False
        if getattr(self, "_correct_method", "ai") != "ai":
            return False
        if self._auto_review is None or getattr(self, "_auto_review_step", 0) != 1:
            return False
        # A live fix session runs its own crop, and a live run owns the machine.
        if getattr(self, "_refine_handoff_active", False):
            return False
        if getattr(self, "_qgis_bridge_active", False):
            return False
        return getattr(self, "_auto_worker", None) is None

    def _warm_local_ai_for_correct(self) -> None:
        """Entering Correct with the AI method armed: get the local model up NOW.

        Two levels, because the model can be up in two different senses. With no
        predictor object the subprocess cannot even be started, and that load is
        the longest part of a first pick, so it is kicked off in the background
        here. With one already in hand, the subprocess itself is pre-started, and
        it warms its own kernels before reporting ready.

        Silent and RAM-conscious: it never installs anything and never touches a
        machine whose environment is not already complete, so an Automatic-only
        user who will never open a polygon pays nothing."""
        if not self._correct_ai_warm_allowed():
            return
        predictor = getattr(self, "predictor", None)
        if predictor is None:
            self._start_local_ai_load_for_correct()
            return
        if getattr(self, "_encoding_in_progress", False):
            return  # a worker owns the pipe; warm_up would race the spawn
        try:
            predictor.warm_up()
        except Exception:  # noqa: BLE001 - a prewarm must never break navigation
            pass  # nosec B110

    def _start_local_ai_load_for_correct(self) -> None:
        """Begin loading the local model in the background, if and only if this
        machine is already set up for it.

        Deliberately NOT the click path's env gate: that one offers to install
        what is missing, which is a question, and nobody asked one here. So an
        incomplete environment is left alone and the click keeps its offer.
        `_load_predictor` ends in `_on_predictor_loaded`, which pre-starts the
        subprocess when `_warm_predictor_on_ready` is set."""
        for attr in ("deps_install_worker", "_verify_worker", "download_worker",
                     "_predictor_worker", "_startup_check_worker"):
            worker = getattr(self, attr, None)
            try:
                if worker is not None and worker.isRunning():
                    return  # a load is already on its way
            except RuntimeError:
                continue
        try:
            from ...core.checkpoint_manager import checkpoint_exists
            from ...core.venv_manager import get_venv_status
            if not checkpoint_exists():
                return
            # No subprocess probe: this runs on a navigation click, and the
            # cold torch import behind the real verification would freeze the
            # window. Packages on disk are enough to try the load.
            ready, _msg = get_venv_status(allow_subprocess_probe=False)
            if not ready:
                return
        except Exception:  # noqa: BLE001 - a warm-up never diagnoses anything
            return
        QgsMessageLog.logMessage(
            "Correct step armed on AI: loading the local model ahead of the pick",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        self._warm_predictor_on_ready = True
        self._load_predictor()

    def _warm_local_ai_for_manual(self) -> None:
        """Manual mode is on screen: get the local model up NOW.

        Choosing Manual is the plainest statement a user makes about the local
        model, since it is the only mode that runs on it. Start already calls
        warm_up(), but by then the user is one aim away from clicking, so the
        subprocess spawn, the model load and the kernel warm-up all queue ahead
        of the first crop. Doing it on the mode itself moves them off the click.

        Two levels, like the Correct step. With a predictor in hand the
        subprocess is pre-started here. With none, the environment check that
        follows owns the load and already refuses on an incomplete machine, so
        this only asks that load to warm the model when it reports ready:
        nothing is installed, nothing is probed and no question is asked, and
        the flag is spent only if a model really loads. An Automatic-only user
        never selects Manual, so they never pay the model's RAM."""
        if getattr(self, "_headless", False):
            return
        predictor = getattr(self, "predictor", None)
        if predictor is None:
            self._warm_predictor_on_ready = True
            return
        if getattr(self, "_encoding_in_progress", False):
            return  # a worker owns the pipe; warm_up would race the spawn
        try:
            predictor.warm_up()
        except Exception:  # noqa: BLE001 - a prewarm must never break navigation
            pass  # nosec B110

    # ------------------------------------------------------------------
    # Crop warm-up on hover
    # ------------------------------------------------------------------

    def _schedule_correct_hover_warm(self, idx) -> None:
        """The cursor moved onto polygon ``idx`` on the resting Correct step:
        arm the crop warm-up. Cheap enough to call on every hover change."""
        if idx is None or not self._correct_ai_warm_allowed():
            return
        if self.dock_widget is None:
            return
        if getattr(self, "predictor", None) is None:
            return  # nothing to encode with yet; the model load is under way
        self._correct_hover_warm_idx = idx
        _debounce_timer(self, "_correct_hover_warm_timer", self.dock_widget,
                        CORRECT_HOVER_WARM_MS, self._warm_hovered_correct_crop)

    def _warm_hovered_correct_crop(self) -> None:
        """Read and encode the crop of the polygon the cursor is resting on.

        Silent throughout: no busy cursor, no error dialog, no session state
        touched. The only lasting effect is that the imagery is encoded, which
        is what the pick was going to have to wait for."""
        idx = getattr(self, "_correct_hover_warm_idx", None)
        if idx is None or not self._correct_ai_warm_allowed():
            return
        if getattr(self, "_encoding_in_progress", False):
            return  # never queue speculative work ahead of a real gesture
        objects = getattr(self, "_auto_objects", None) or []
        if idx < 0 or idx >= len(objects):
            return
        geom = objects[idx][0]
        if geom is None or geom.isEmpty():
            return
        if not self._bind_correct_crop_context():
            return
        from ...core.crop_window import crop_window_key, neighborhood_crop_window
        bb = geom.boundingBox()
        # The grid window, with NO reuse of a crop already in hand, because that
        # is what the pick will compute: opening a polygon starts a fix session,
        # and starting one clears the record of which crop is loaded. Warming a
        # reused window instead would prepare imagery the pick then asks for
        # under a different name, and the wait would come back in full.
        cx, cy, scale = neighborhood_crop_window(
            (bb.xMinimum(), bb.yMinimum(), bb.xMaximum(), bb.yMaximum()),
            self._get_native_pixel_size())
        if crop_window_key(cx, cy, scale) in (
                getattr(self, "_encoded_crop_window", None),
                getattr(self, "_inflight_crop_window", None)):
            return  # this neighbourhood is already encoded, or on its way
        self._extract_and_encode_crop(
            QgsPointXY(cx, cy), mupp_override=scale, show_busy=False, quiet=True)

    # ------------------------------------------------------------------
    # Manual mode: warm where the cursor is resting
    # ------------------------------------------------------------------

    def _manual_warm_allowed(self) -> bool:
        """Is warming appropriate for the live Manual session right now?

        Only BETWEEN objects, and only while no gesture is pending on one. A
        session with a shape in progress must keep the crop it is working on:
        swapping it behind the click path would leave the accumulated points
        describing pixels in another image. And an object on screen can be
        saved or undone at any moment, both of which stand back from a busy
        predictor pipe, so a silent warm-up started under one would swallow the
        gesture with nothing on screen to explain it.

        Saved polygons do NOT close the door. They last the whole session, and
        moving on to the next object right after a save is the Manual loop this
        warm-up exists for. The gestures they still answer (undo back into the
        last saved polygon, undo a delete) are covered the other way round, by
        _abandon_speculative_manual_crop at the gesture itself.
        """
        if getattr(self, "_headless", False):
            return False
        if getattr(self, "_refine_handoff_active", False):
            return False  # the review has its own warm-up
        if getattr(self, "predictor", None) is None:
            return False
        if self._current_layer is None or not self._current_raster_path:
            return False
        if getattr(self, "_is_online_layer", False):
            return False
        if getattr(self, "_encoding_in_progress", False):
            return False
        if getattr(self, "_pending_manual_click", None) is not None:
            return False  # a click is waiting to replay; the next crop is its own
        if self.current_mask is not None:
            return False
        if getattr(self, "_is_refining_saved_object", False):
            return False
        # An object with no mask of its own is still open: an unfrozen crop
        # session, or a saved polygon re-opened from a save that carried
        # geometry only. Its next click, its Save and its Undo all belong to
        # the crop already on screen.
        if getattr(self, "_unfrozen_display_polygon", None) is not None:
            return False
        if self._frozen_sessions or self._active_crop_points_positive:
            return False
        if self._active_crop_points_negative:
            return False
        # Points with no active crop behind them: re-opening a saved polygon
        # restores its prompts without refilling the per-crop lists.
        prompts = getattr(self, "prompts", None)
        if prompts is None:
            return True
        try:
            return sum(prompts.point_count) == 0
        except (AttributeError, TypeError):
            return True

    def _begin_speculative_manual_crop(self, center_point, scale) -> None:
        """Start a Manual warm-up crop and record that the pipe it takes
        belongs to nobody. The flag rides the START, so a call that started
        nothing (an online layer, a busy pipe) leaves it down."""
        self._speculative_manual_crop = bool(self._extract_and_encode_crop(
            center_point, mupp_override=scale, show_busy=False, quiet=True))

    def _abandon_speculative_manual_crop(self) -> bool:
        """Give up a warm-up holding the predictor pipe so a gesture the user
        just made can run instead.

        True when the pipe was held by a cursor-less warm-up with no click
        waiting behind it, and that warm-up has been abandoned: the caller goes
        ahead. False when the pipe belongs to work somebody asked for, which
        the caller must still stand back from. The abandoned crop costs a
        re-read later, and that is the trade: work is cheap, a lost gesture is
        not.

        Everything is re-checked here, because the warm-up that raised the flag
        may already be over and the pipe may have passed to a real crop since.
        """
        if not getattr(self, "_speculative_manual_crop", False):
            return False
        if not getattr(self, "_encoding_in_progress", False):
            self._speculative_manual_crop = False
            return False  # nothing holds the pipe; the caller was not blocked
        if getattr(self, "_encode_cursor_set", True):
            # The pipe wears a busy cursor, so this crop was asked for by name
            # (a click, a fix-session open) and the flag is stale.
            self._speculative_manual_crop = False
            return False
        if getattr(self, "_pending_manual_click", None) is not None:
            # A real click is already waiting on this crop. Dropping the crop
            # drops the click with it, which is the thing this method exists to
            # prevent; the click lands in a moment and the gesture can follow.
            return False
        self._speculative_manual_crop = False
        self._invalidate_manual_encode()
        # Abandoning drops the RESULT, not the work: the worker may already have
        # loaded that crop into the predictor, and its completion is thrown away
        # without recording it. So the crop the predictor holds is now unknown,
        # and the two facts that claim to know it have to go with it. Leaving
        # _current_crop_info standing let the next click in the OLD crop pass
        # _check_crop_status as "ok" and predict against the abandoned imagery
        # through the old crop's transform, i.e. a mask in the wrong place.
        # The cost is one re-read, and a re-read is cheap next to a wrong shape.
        self._current_crop_info = None
        self._encoded_crop_window = None
        QgsMessageLog.logMessage(
            "Dropped a warm-up crop: the user asked for something else",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        return True

    def _schedule_manual_hover_warm(self, canvas_point) -> None:
        """The cursor moved over the map during a Manual session: arm the warm-up
        for where it is resting. Called on every move, so it stays cheap."""
        if self.dock_widget is None or not self._manual_warm_allowed():
            return
        self._manual_warm_canvas_point = canvas_point
        _debounce_timer(self, "_manual_hover_warm_timer", self.dock_widget,
                        MANUAL_HOVER_WARM_MS, self._warm_hovered_manual_crop)

    def _warm_hovered_manual_crop(self) -> None:
        """Read and encode the crop around the resting cursor, if a click there
        would otherwise have to wait for it.

        The centre is snapped to the shared grid rather than sitting exactly under
        the cursor, so coming back to a neighbourhood asks the model for a crop it
        already has. The click itself still lands well inside that crop, since one
        grid cell is a quarter of the crop's own width.
        """
        point = getattr(self, "_manual_warm_canvas_point", None)
        if point is None or not self._manual_warm_allowed():
            return
        try:
            raster_pt = self._transform_to_raster_crs(point)
        except (RuntimeError, AttributeError):
            return
        if raster_pt is None or not self._is_point_in_raster_extent(raster_pt):
            return
        # A click here would already be answered from the crop in hand: nothing
        # to warm, and re-reading would only take the crop away from it.
        if self._check_crop_status(raster_pt) == "ok":
            return
        from ...core.crop_window import crop_window_key, snap_center_to_grid
        scale = self._compute_initial_scale_factor()
        cx, cy = snap_center_to_grid(
            raster_pt.x(), raster_pt.y(), scale or 1.0,
            self._get_native_pixel_size())
        if crop_window_key(cx, cy, scale or 1.0) in (
                getattr(self, "_encoded_crop_window", None),
                getattr(self, "_inflight_crop_window", None)):
            return
        QgsMessageLog.logMessage(
            "Manual: warming the crop under the resting cursor",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        self._begin_speculative_manual_crop(QgsPointXY(cx, cy), scale)

    def _bind_correct_crop_context(self) -> bool:
        """Point the crop reader at the raster the run segmented.

        The crop reader needs to know which raster and how it maps to ground,
        which a fix session sets up when it starts. A warm-up happens before any
        session exists, so it binds that much and nothing else: no session
        reset, no map tool, no dock change. Starting the real session later
        re-binds the same values and then does the rest.

        False when there is nothing safe to read: no resolvable raster, or an
        online one (its crop comes off the tile network, which is not something
        to do behind a cursor movement)."""
        layer = self._resolve_auto_source_layer()
        if layer is None:
            return False
        try:
            if not self._is_layer_valid(layer):
                return False
            if self._is_online_provider(layer):
                return False
            source = layer.source()
        except (RuntimeError, AttributeError):
            return False
        if self._current_layer is layer and self._current_raster_path:
            return True
        self._current_layer = layer
        self._current_layer_name = layer.name().replace(" ", "_")
        self._current_raster_path = source
        self._is_online_layer = False
        self._is_non_georeferenced_mode = not self._is_layer_georeferenced(layer)
        self._bind_correct_crop_transforms(layer)
        return True

    def _bind_correct_crop_transforms(self, layer) -> None:
        """Canvas <-> raster transforms for the bound raster, or None when both
        sides share a CRS. Mirrors what a session start installs."""
        self._canvas_to_raster_xform = None
        self._raster_to_canvas_xform = None
        try:
            canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
            raster_crs = layer.crs()
            if not (raster_crs.isValid() and canvas_crs.isValid()):
                return
            if canvas_crs == raster_crs:
                return
            project = QgsProject.instance()
            self._canvas_to_raster_xform = QgsCoordinateTransform(
                canvas_crs, raster_crs, project)
            self._raster_to_canvas_xform = QgsCoordinateTransform(
                raster_crs, canvas_crs, project)
        except (RuntimeError, AttributeError):
            self._canvas_to_raster_xform = None
            self._raster_to_canvas_xform = None
