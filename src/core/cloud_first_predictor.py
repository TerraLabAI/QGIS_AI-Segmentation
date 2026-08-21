"""A predictor that asks the network first and the machine second.

It wears the same six members the click path reads off a predictor
(``is_image_set``, ``set_image``, ``predict``, ``reset_image``, ``cleanup``,
``warm_up``), so the worker, the transport lock and the click code cannot tell
it apart from either of the two it holds.

The point is the second half. A click that the network cannot answer, for any
reason at all, is answered on the machine instead, with the crop this object
kept, and the user gets their outline. Nothing is raised at them and nothing
pops up. When there is no on-device predictor to fall back to, the original
failure travels on to the click path, which already knows how to report one.

The crop is kept here on purpose. Both sides need the same pixels, and the
caller hands them over once, on its own worker thread, well before the click.
"""
from __future__ import annotations

import numpy as np
from qgis.core import Qgis, QgsMessageLog

from .log_scrub import scrub_sensitive


def _log(message: str, level=Qgis.MessageLevel.Info) -> None:
    QgsMessageLog.logMessage(message, "AI Segmentation", level=level)


def _safe_to_log(err: Exception) -> str:
    """What a failure says, safe to write down.

    Every failure below travels from the far side, so its text can carry the
    address it came from. The telemetry copy is scrubbed already; the log copy
    goes through the same door.
    """
    return scrub_sensitive(str(err))


def _resize_nearest(arr: np.ndarray, side: int) -> np.ndarray:
    """Square-resize a 2D array by nearest neighbour, without scipy.

    The macOS QGIS python cannot load scipy at all, so every resize in the
    click path is written out like this one.
    """
    h, w = arr.shape
    rows = (np.arange(side) * h // side).clip(0, h - 1)
    cols = (np.arange(side) * w // side).clip(0, w - 1)
    return arr[rows[:, None], cols[None, :]]


def _answer_was_superseded(err: Exception) -> bool:
    """Did the remote side stop because the crop moved on, rather than fail?

    A failure is what the machine answers instead. A crop that moved on is not:
    the click it belonged to has no picture left to be answered on.
    """
    try:
        from .cloud_sam_predictor import RefineSupersededError

        return isinstance(err, RefineSupersededError)
    except Exception:  # noqa: BLE001 -- an unimportable class is not a match
        return False


class CloudFirstPredictor:
    """Holds the remote predictor, and reaches for a local one when it fails."""

    def __init__(self, remote, local_source=None, on_fallback=None,
                 on_remote_answer=None) -> None:
        """``local_source`` is read at the moment of the failure, not now: the
        on-device predictor often arrives after this object is built, and a
        reference taken here would freeze that answer at None.

        ``on_remote_answer`` fires once per click the network really answered.
        It is what tells the session that this object owes a credit, so it must
        never fire for a click the machine answered instead.
        """
        self._remote = remote
        self._local_source = local_source
        self._on_fallback = on_fallback
        self._on_remote_answer = on_remote_answer
        self._crop: np.ndarray | None = None
        # Which local predictor already holds our crop, so the fallback pays
        # the encode once per crop instead of once per click.
        self._local_holding_crop = None
        # Which crop the session is on. The remote wait runs the event loop, so
        # a new crop can land inside one click; the fallback must not then
        # answer the old click with the new pixels.
        self._generation = 0
        # Only the on-device path ever fills this. Kept so a caller reading it
        # off whichever predictor it has gets an answer, not an AttributeError.
        self.input_size = None
        # Which side answered the last click. Read by anything whose behaviour
        # is calibrated on one model and not the other, the unsure hint first.
        # None until a click has been answered at all.
        self.last_answer_was_remote: bool | None = None

    # -- the shape every predictor in the click path has --------------------

    @property
    def is_image_set(self) -> bool:
        return self._crop is not None

    @property
    def original_size(self) -> tuple[int, int] | None:
        if self._crop is None:
            return None
        return (int(self._crop.shape[0]), int(self._crop.shape[1]))

    @property
    def low_res_side(self) -> int | None:
        """The side a hand-built mask seed has to be, for whoever answers next.

        The remote's, once an answer has said what it is. Before that there is
        nothing to report: guessing here is what sends a seed the far side
        refuses, and a refused seed used to become a silent fall back to the
        machine for that one click.
        """
        return getattr(self._remote, "low_res_side", None)

    def warm_up(self) -> bool:
        """Nothing to start on the remote side. The on-device one warms itself
        when the plugin loads it."""
        return True

    def hover_preview_handle(self):
        """The remote side's name for the crop in hand, or None.

        Only the network answers a preview, so there is nothing to fall back
        to: a session on the machine simply has no preview to give.
        """
        getter = getattr(self._remote, "hover_preview_handle", None)
        if getter is None:
            return None
        try:
            return getter()
        except Exception:  # noqa: BLE001 -- an unnamed crop is no preview
            return None

    def session_generation(self) -> int:
        """Which crop this predictor is on. Moves whenever the pixels change or
        the session ends."""
        return self._generation

    def reset_image(self) -> None:
        self._generation += 1
        self._crop = None
        self._local_holding_crop = None
        try:
            self._remote.reset_image()
        except Exception as err:  # noqa: BLE001 -- a reset must never raise
            _log(f"Remote click route: reset ignored ({_safe_to_log(err)})")

    def cleanup(self) -> None:
        """Drop the remote side only. The on-device predictor belongs to the
        plugin, which reuses it for the next session."""
        self._generation += 1
        self._crop = None
        self._local_holding_crop = None
        try:
            self._remote.cleanup()
        except Exception as err:  # noqa: BLE001 -- teardown must never raise
            _log(f"Remote click route: cleanup ignored ({_safe_to_log(err)})")

    def set_image(self, image_np: np.ndarray) -> None:
        """Take the crop, then let the remote side send it ahead of the click.

        Runs on the caller's worker thread. A remote failure here is not an
        error to report: the crop is held, so the first click still has pixels
        for whichever side answers it.
        """
        if image_np is not self._crop:
            self._generation += 1
        self._crop = image_np
        self._local_holding_crop = None
        try:
            self._remote.set_image(image_np)
        except Exception as err:  # noqa: BLE001 -- the click carries the crop
            _log("Remote click route: the crop did not go out ahead of the "
                 f"click, it will travel with it ({_safe_to_log(err)})")

    def predict(self, *args, **kwargs):
        if self._crop is None:
            raise RuntimeError("Image has not been set. Call set_image first.")
        generation = self._generation
        try:
            answer = self._remote.predict(*args, **kwargs)
        except Exception as err:  # noqa: BLE001 -- the machine answers instead
            if generation != self._generation or _answer_was_superseded(err):
                # The crop changed while the click was out. The pixels held
                # here are the NEW ones, so answering this click on them would
                # put the outline of one picture on another. The click ends.
                raise
            return self._predict_on_device(err, *args, **kwargs)
        # The network answered. Noted after the call returns, so a click that
        # raised on its way is never counted as one the remote side served.
        self.last_answer_was_remote = True
        if self._on_remote_answer is not None:
            try:
                self._on_remote_answer()
            except Exception:  # nosec B110 -- a note never costs a click
                pass
        return answer

    # -- the second half ----------------------------------------------------

    # predict(point_coords, point_labels, box, mask_input, ...): the seed is the
    # fourth positional, and the click path passes it by name. Both are handled.
    _MASK_INPUT_ARG = 3

    def _seed_for(self, local, args, kwargs):
        """Give the on-device model a seed at the side IT reads.

        The two engines do not work in the same low-res space, and each refuses
        the other's side outright. A seed is whatever the last answer returned,
        so a session that starts on the network and falls back mid-object
        carries the far side's seed into the on-device model. That is every
        correcting click after a network drop, and the polygon-seeded ones too.

        Resizing here rather than at the caller is deliberate: the caller
        cannot know which side will answer, and this is the one place that
        does.
        """
        want = getattr(local, "low_res_side", None)
        try:
            want = int(want)
        except (TypeError, ValueError):
            return args, kwargs
        if want <= 0:
            return args, kwargs

        by_name = "mask_input" in kwargs
        if by_name:
            seed = kwargs["mask_input"]
        elif len(args) > self._MASK_INPUT_ARG:
            seed = args[self._MASK_INPUT_ARG]
        else:
            return args, kwargs
        if seed is None or getattr(seed, "ndim", 0) != 3:
            return args, kwargs
        if seed.shape[1] == want and seed.shape[2] == want:
            return args, kwargs

        _log(f"Click seed resized {seed.shape[1]}x{seed.shape[2]} to {want} for "
             "the model on this computer")
        resized = np.stack([
            _resize_nearest(np.asarray(plane, dtype=np.float32), want)
            for plane in seed
        ])
        if by_name:
            kwargs = dict(kwargs, mask_input=resized)
        else:
            args = list(args)
            args[self._MASK_INPUT_ARG] = resized
            args = tuple(args)
        return args, kwargs

    def _predict_on_device(self, remote_error: Exception, *args, **kwargs):
        """Answer this one click on the machine. Re-raises when it cannot.

        The remote failure is deliberately not re-wrapped: when there is no
        on-device predictor, the click path receives exactly the error it would
        have received without this class, and reports it the way it always has.
        """
        local = None
        if self._local_source is not None:
            try:
                local = self._local_source()
            except Exception:  # nosec B110 -- no local predictor, then
                local = None
        if local is None:
            raise remote_error
        _log("Click answered on this computer, the network could not: "
             f"{_safe_to_log(remote_error)}", Qgis.MessageLevel.Warning)
        if self._local_holding_crop is not local:
            local.set_image(self._crop)
            self._local_holding_crop = local
        args, kwargs = self._seed_for(local, args, kwargs)
        answer = local.predict(*args, **kwargs)
        self.last_answer_was_remote = False
        if self._on_fallback is not None:
            try:
                self._on_fallback()
            except Exception:  # nosec B110 -- a status note never costs a click
                pass
        return answer
