"""Get the local model and the imagery ready BEFORE the user clicks.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).

A click that needs the model used to pay, in one go and with the user watching:
the model load, the imagery read around the click, and the encode of it. All
three can be paid earlier, on time the user is spending anyway, because the user
announces where they are about to click by moving the cursor there:

- Opening the review with the AI fix method armed says a polygon may well be
  picked in a moment. That is when the model starts loading, so the reading
  the user does on the Keep step pays for the load.
- Arriving on the Correct step with AI armed says a polygon is about to be
  picked. That is when the model starts loading, for anyone who got there
  without a load already running.
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

A basemap is warmed like any other layer, but never through the layer itself.
Its imagery is not read from a file: it is pulled through a layer provider,
which the read switches to bilinear resampling and tells to drop its cached
tiles. So a warm-up reads through a private copy of the layer instead
(`core/online_layer_twin.py`), and the layer on screen is left untouched. No
copy, no warm-up: the click then pays for its own imagery, as it always did.
"""
from __future__ import annotations

import os

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
# clicked. Shipped value, and the fallback the getter below returns.
CORRECT_HOVER_WARM_MS = 200

# Same idea on the Manual map, a little longer: there is no object under the
# cursor to aim at, so a moving mouse crosses many candidate crops.
MANUAL_HOVER_WARM_MS = 300

# The same rest when the imagery is a file on this machine. Reading one crop
# off a local raster costs a few milliseconds, so the rest IS the delay, and a
# crop the cursor turns out not to want costs almost nothing to have read.
MANUAL_LOCAL_HOVER_WARM_MS = 120

# Ceiling on either delay. Past a few seconds the cursor has moved on and the
# warm-up prepares a crop nobody is aiming at any more.
_MAX_HOVER_WARM_MS = 5_000


def local_ai_warmup_enabled() -> bool:
    """Whether any speculative warm-up may start.

    The delay dials cannot answer this: zero means warm on the next pass of the
    event loop, not never. Off, nothing is loaded, read or encoded before a
    click, and every gesture pays for its own crop. Fail-open, so an absent or
    damaged configuration warms exactly as shipped.
    """
    try:
        from ...core.server_dials import feature_enabled

        return feature_enabled("local_ai_warmup")
    except Exception:  # noqa: BLE001 -- a kill switch is best-effort  # nosec B110
        return True


# Shortest rest a served value may ask for. Below this the debounce fires on the
# next pass of the event loop, so a mouse crossing the map reads and encodes a
# crop per polygon it passes instead of only where the user stops. Still
# immediate to a human, and it keeps one JSON edit from freezing the fleet.
_MIN_HOVER_WARM_MS = 50


def correct_hover_warm_ms() -> int:
    """Rest time before the hovered polygon's crop is read, in milliseconds.

    Floored, not zeroable: see _MIN_HOVER_WARM_MS. Cache-only and never raises,
    so it is safe on the hover path it is read from.
    """
    try:
        from ...core.server_dials import dial_in_range

        return int(dial_in_range("ui.warm_hover_ms.correct", CORRECT_HOVER_WARM_MS,
                                 _MIN_HOVER_WARM_MS, _MAX_HOVER_WARM_MS))
    except Exception:  # noqa: BLE001 -- a warm-up delay is best-effort  # nosec B110
        return CORRECT_HOVER_WARM_MS


def manual_hover_warm_ms() -> int:
    """Same rest time on the Manual map, in milliseconds. Same floor."""
    try:
        from ...core.server_dials import dial_in_range

        return int(dial_in_range("ui.warm_hover_ms.manual", MANUAL_HOVER_WARM_MS,
                                 _MIN_HOVER_WARM_MS, _MAX_HOVER_WARM_MS))
    except Exception:  # noqa: BLE001 -- a warm-up delay is best-effort  # nosec B110
        return MANUAL_HOVER_WARM_MS


def manual_local_hover_warm_ms() -> int:
    """The Manual rest time when the crop comes off a file on this machine.

    Its own dial, so the two cases can be tuned apart: a rendered or online
    layer pays a network fetch for a crop nobody asked for, a file pays a
    windowed read. Same floor and same ceiling as the others.
    """
    try:
        from ...core.server_dials import dial_in_range

        return int(dial_in_range("ui.warm_hover_ms.manual_local",
                                 MANUAL_LOCAL_HOVER_WARM_MS,
                                 _MIN_HOVER_WARM_MS, _MAX_HOVER_WARM_MS))
    except Exception:  # noqa: BLE001 -- a warm-up delay is best-effort  # nosec B110
        return MANUAL_LOCAL_HOVER_WARM_MS


class LocalAiWarmMixin:
    """Model load and crop warm-up for the review's AI method and for Manual."""

    def _correct_step_resting_on_ai(self) -> bool:
        """The Correct step is on screen, armed on AI, with nothing running.

        The shared half of the warm-up guards below. Everything is re-checked
        at fire time, because a debounced timer outlives the state that armed
        it."""
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

    def _correct_ai_warm_allowed(self) -> bool:
        """May the LOCAL MODEL be loaded or pre-started right now?"""
        if not local_ai_warmup_enabled():
            return False
        # Nothing to warm when the fix is answered off the machine: loading a
        # model that will not be asked anything costs the user memory and time.
        if self._correct_ai_route_is_remote():
            return False
        return self._correct_step_resting_on_ai()

    def _correct_crop_warm_allowed(self) -> bool:
        """May the hovered polygon's CROP be read and encoded ahead of the pick?

        Wider than the model guard on purpose: the remote route has no model to
        load, and reading the crop ahead is exactly what makes its pick answer
        in one round trip instead of paying the read, the upload and the far
        side's encode at the click. The far side expects and rate-limits these
        warm-ups, and a refused one costs nothing: the pick sends the pixels
        itself, as it always has."""
        if not local_ai_warmup_enabled():
            return False
        if not self._correct_step_resting_on_ai():
            return False
        if self._correct_ai_route_is_remote():
            return True
        return self._correct_ai_warm_allowed()

    def _review_ai_warm_allowed(self) -> bool:
        """Same question for the review as a whole, one step earlier.

        Everything the Correct-step guard asks except the step itself: the
        review is open and its fix method is AI, wherever the user is standing.
        The step is what this one drops on purpose. Keep is where the user
        reads the result, and that reading is the last free time the session
        has, so the model load starts there rather than on arrival at Correct.
        Re-checked at fire time like the other guard."""
        if not local_ai_warmup_enabled():
            return False
        # Nothing to warm when the fix is answered off the machine: loading a
        # model that will not be asked anything costs the user memory and time.
        if self._correct_ai_route_is_remote():
            return False
        if getattr(self, "_headless", False):
            return False
        if getattr(self, "_correct_method", "ai") != "ai":
            return False
        if self._auto_review is None:
            return False
        # A live fix session runs its own crop, and a live run owns the machine.
        if getattr(self, "_refine_handoff_active", False):
            return False
        if getattr(self, "_qgis_bridge_active", False):
            return False
        return getattr(self, "_auto_worker", None) is None

    def _start_or_warm_local_ai(self) -> None:
        """Get the local model up, at whichever of its two levels applies.

        The model can be up in two different senses. With no predictor object
        the subprocess cannot even be started, and that load is the longest part
        of a first pick, so it is kicked off in the background here. With one
        already in hand, the subprocess itself is pre-started, and it warms its
        own kernels before reporting ready.

        Silent and RAM-conscious: it never installs anything and never touches a
        machine whose environment is not already complete, so an Automatic-only
        user who will never open a polygon pays nothing. Callers own the guard,
        so nothing here decides whether the moment is right."""
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

    def _warm_local_ai_for_review(self) -> None:
        """The review just opened: start the local model now.

        The user is about to read the Keep step, and reading takes seconds the
        model load can hide behind. Waiting for the Correct step spends those
        seconds and starts the same load with the user already aiming at a
        polygon."""
        if self._review_ai_warm_allowed():
            self._start_or_warm_local_ai()

    def _warm_local_ai_for_correct(self) -> None:
        """Entering Correct with the AI method armed: get the local model up NOW.

        Usually the review opening already started this, and the second call
        costs nothing. It still matters for the user who armed AI after the
        review opened, or who came back to Correct with the load abandoned."""
        if self._correct_ai_warm_allowed():
            self._start_or_warm_local_ai()

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
            "Review armed on the AI fix method: loading the local model "
            "ahead of the pick",
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
        if idx is None or not self._correct_crop_warm_allowed():
            return
        if self.dock_widget is None:
            return
        # Recorded even with no predictor yet. Hovering while the model loads is
        # the commonest thing a user does on a first visit, and dropping that
        # hover meant the click that followed still paid a full read and encode
        # on imagery the cursor had been sitting on for half a minute.
        # _replay_hover_warm_when_ready picks it up when the model arrives.
        self._correct_hover_warm_idx = idx
        if (getattr(self, "predictor", None) is None
                and not self._correct_ai_route_is_remote()):
            # The remote route builds its predictor at fire time, so an empty
            # slot is no reason to stand down there.
            return
        _debounce_timer(self, "_correct_hover_warm_timer", self.dock_widget,
                        correct_hover_warm_ms(), self._warm_hovered_correct_crop)

    def _warm_hovered_correct_crop(self) -> None:
        """Read and encode the crop of the polygon the cursor is resting on.

        Silent throughout: no busy cursor, no error dialog, no session state
        touched. The only lasting effect is that the imagery is encoded, which
        is what the pick was going to have to wait for."""
        idx = getattr(self, "_correct_hover_warm_idx", None)
        if idx is None or not self._correct_crop_warm_allowed():
            return
        if getattr(self, "_encoding_in_progress", False):
            return  # never queue speculative work ahead of a real gesture
        objects = getattr(self, "_auto_objects", None) or []
        if idx < 0 or idx >= len(objects):
            return
        geom = objects[idx][0]
        if geom is None or geom.isEmpty():
            return
        if self._correct_ai_route_is_remote():
            # The warm-up must feed the predictor the pick will use. With the
            # remote one in place the encode is a windowed read and one small
            # request; failing to put it there is not an error, it just means
            # the pick pays its own way, as it always could.
            if not self._ensure_cloud_correct_predictor():
                return
        elif getattr(self, "predictor", None) is None:
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

        A layer QGIS renders is warmed only through a private copy of it, so
        the extra question it answers is whether that copy can be had.
        """
        if not local_ai_warmup_enabled():
            return False  # off means nothing is read or encoded before a click
        if getattr(self, "_headless", False):
            return False
        if getattr(self, "_refine_handoff_active", False):
            return False  # the review has its own warm-up
        if getattr(self, "predictor", None) is None:
            return False
        if self._current_layer is None or not self._current_raster_path:
            return False
        if getattr(self, "_is_online_layer", False) and not self._online_warm_layer_ready():
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

    def _online_warm_layer_ready(self) -> bool:
        """May a warm-up read the rendered layer of this session, and through
        what? True only when the served switch is on and a private copy of the
        layer builds, because reading the user's own layer would reload the
        basemap under the cursor. The copy is cached, so answering this on every
        mouse move is cheap."""
        from ...core.online_layer_twin import online_layer_twin, online_prewarm_enabled

        if not online_prewarm_enabled():
            return False
        return online_layer_twin(self._current_layer) is not None

    def _begin_speculative_manual_crop(self, center_point, scale) -> None:
        """Start a Manual warm-up crop and record that the pipe it takes
        belongs to nobody. The flag rides the START, so a call that started
        nothing (a busy pipe, a rendered layer with no private copy to read
        through) leaves it down."""
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
        if self.dock_widget is None:
            return
        # Kept even when the guard refuses, for the same reason as the Correct
        # step: the commonest refusal is a model still loading, and the point
        # the cursor rested on during that load is exactly the crop the first
        # click will want (_replay_hover_warm_when_ready).
        self._manual_warm_canvas_point = canvas_point
        if not self._manual_warm_allowed():
            return
        rest_ms = (manual_local_hover_warm_ms() if self._manual_warm_reads_a_file()
                   else manual_hover_warm_ms())
        _debounce_timer(self, "_manual_hover_warm_timer", self.dock_widget,
                        rest_ms, self._warm_hovered_manual_crop)

    def _manual_warm_reads_a_file(self) -> bool:
        """Whether the crop under the cursor comes off a raster file on disk.

        The imagery of the live Manual session, not the layer the user has
        selected somewhere: this decides how long the warm-up waits before
        reading it. False for anything QGIS renders and for a GDAL source that
        is not a file here (a URL, a service), because those pay a fetch."""
        if getattr(self, "_is_online_layer", False):
            return False
        source = getattr(self, "_current_raster_path", None)
        if not source:
            return False
        # Answered once per raster and remembered: this runs on every mouse
        # move over the map, and a disk stat per move is a stat per pixel of
        # travel, on a network share as easily as on a local file.
        memo = getattr(self, "_manual_warm_file_memo", None)
        if memo is not None and memo[0] == source:
            return memo[1]
        try:
            provider = self._current_layer.dataProvider()
            if provider is None or provider.name() != "gdal":
                reads_a_file = False
            else:
                reads_a_file = os.path.isfile(source)
        except (RuntimeError, AttributeError, OSError, ValueError):
            return False
        self._manual_warm_file_memo = (source, reads_a_file)
        return reads_a_file

    def _replay_hover_warm_when_ready(self) -> None:
        """The model just arrived: warm the crop the cursor has been resting on.

        Called from the predictor-loaded slot. Both lanes are tried and both
        re-check their own guards, so at most one of them has anything to do
        and neither can fire in a state that does not want it. Nothing rests
        under the cursor means nothing happens, which is the common case."""
        if getattr(self, "_headless", False):
            return
        if getattr(self, "predictor", None) is None:
            return
        if getattr(self, "_correct_hover_warm_idx", None) is not None:
            self._warm_hovered_correct_crop()
        if getattr(self, "_manual_warm_canvas_point", None) is not None:
            self._warm_hovered_manual_crop()

    def _warm_hovered_manual_crop(self) -> None:
        """Read and encode the crop around the resting cursor, if a click there
        would otherwise have to wait for it.

        The centre is snapped to the shared grid rather than sitting exactly under
        the cursor, so coming back to a neighbourhood asks the model for a crop it
        already has. The click itself still lands well inside that crop, since one
        grid cell is a quarter of the crop's own width.

        A layer QGIS renders measures that grid in ground per pixel instead of a
        zoom-out factor on native pixels, so it asks _online_grid_window for the
        same window its own fetch would cut, and the read then goes through the
        private copy of the layer.
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
        if getattr(self, "_is_online_layer", False):
            window = self._online_grid_window(raster_pt)
            if window is None:
                return
            center, scale = window
        else:
            scale = self._compute_initial_scale_factor()
            cx, cy = snap_center_to_grid(
                raster_pt.x(), raster_pt.y(), scale or 1.0,
                self._get_native_pixel_size())
            center = QgsPointXY(cx, cy)
        if crop_window_key(center.x(), center.y(), scale or 1.0) in (
                getattr(self, "_encoded_crop_window", None),
                getattr(self, "_inflight_crop_window", None)):
            return
        QgsMessageLog.logMessage(
            "Manual: warming the crop under the resting cursor",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        self._begin_speculative_manual_crop(center, scale)

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
            if self._needs_canvas_render(layer):
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
