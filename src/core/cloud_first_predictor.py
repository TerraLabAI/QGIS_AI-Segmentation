















from __future__ import annotations

import numpy as np
from qgis.core import Qgis, QgsMessageLog

from .log_scrub import scrub_sensitive


def _log(message: str, level=Qgis.MessageLevel.Info) -> None:
    QgsMessageLog.logMessage(message, "AI Segmentation", level=level)


def _safe_to_log(err: Exception) -> str:






    return scrub_sensitive(str(err))


def _resize_nearest(arr: np.ndarray, side: int) -> np.ndarray:





    h, w = arr.shape
    rows = (np.arange(side) * h // side).clip(0, h - 1)
    cols = (np.arange(side) * w // side).clip(0, w - 1)
    return arr[rows[:, None], cols[None, :]]


def _answer_was_superseded(err: Exception) -> bool:





    try:
        from .cloud_sam_predictor import RefineSupersededError

        return isinstance(err, RefineSupersededError)
    except Exception:  # noqa: BLE001
        return False


class CloudFirstPredictor:


    def __init__(self, remote, local_source=None, on_fallback=None,
                 on_remote_answer=None) -> None:








        self._remote = remote
        self._local_source = local_source
        self._on_fallback = on_fallback
        self._on_remote_answer = on_remote_answer
        self._crop: np.ndarray | None = None


        self._local_holding_crop = None



        self._generation = 0


        self.input_size = None



        self.last_answer_was_remote: bool | None = None



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






        return getattr(self._remote, "low_res_side", None)

    def warm_up(self) -> bool:


        return True

    def hover_preview_handle(self):





        getter = getattr(self._remote, "hover_preview_handle", None)
        if getter is None:
            return None
        try:
            return getter()
        except Exception:  # noqa: BLE001
            return None

    def session_generation(self) -> int:


        return self._generation

    def set_session_id(self, session_id: str | None) -> None:



        setter = getattr(self._remote, "set_session_id", None)
        if setter is not None:
            try:
                setter(session_id)
            except Exception:  # nosec B110
                pass

    def reset_image(self) -> None:
        self._generation += 1
        self._crop = None
        self._local_holding_crop = None
        self.last_answer_was_remote = None
        self.input_size = None
        try:
            self._remote.reset_image()
        except Exception as err:  # noqa: BLE001
            _log(f"Remote click route: reset ignored ({_safe_to_log(err)})")

    def cleanup(self) -> None:


        self._generation += 1
        self._crop = None
        self._local_holding_crop = None
        self.last_answer_was_remote = None
        self.input_size = None
        try:
            self._remote.cleanup()
        except Exception as err:  # noqa: BLE001
            _log(f"Remote click route: cleanup ignored ({_safe_to_log(err)})")

    def set_image(self, image_np: np.ndarray) -> None:






        if image_np is not self._crop:
            self._generation += 1
        self._crop = image_np
        self._local_holding_crop = None
        try:
            self._remote.set_image(image_np)
        except Exception as err:  # noqa: BLE001
            _log("Remote click route: the crop did not go out ahead of the "
                 f"click, it will travel with it ({_safe_to_log(err)})")

    def predict(self, *args, **kwargs):
        if self._crop is None:
            raise RuntimeError("Image has not been set. Call set_image first.")
        generation = self._generation
        try:
            answer = self._remote.predict(*args, **kwargs)
        except Exception as err:  # noqa: BLE001
            if generation != self._generation or _answer_was_superseded(err):



                raise
            return self._predict_on_device(err, *args, **kwargs)
        if generation != self._generation:
            from .cloud_sam_predictor import RefineSupersededError

            raise RefineSupersededError("The crop changed while its answer was on the way")


        self.last_answer_was_remote = True
        if self._on_remote_answer is not None:
            try:
                self._on_remote_answer()
            except Exception:  # nosec B110
                pass
        return answer





    _MASK_INPUT_ARG = 3

    def _seed_for(self, local, args, kwargs):












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






        generation = self._generation
        crop = self._crop
        local = None
        if self._local_source is not None:
            try:
                local = self._local_source()
            except Exception:  # nosec B110
                local = None
        if local is None:
            raise remote_error
        _log("Click answered on this computer, the network could not: "
             f"{_safe_to_log(remote_error)}", Qgis.MessageLevel.Warning)
        if self._local_holding_crop is not local:
            local.set_image(crop)
            if generation != self._generation:
                raise remote_error
            self._local_holding_crop = local
        args, kwargs = self._seed_for(local, args, kwargs)
        answer = local.predict(*args, **kwargs)
        if generation != self._generation:
            from .cloud_sam_predictor import RefineSupersededError

            raise RefineSupersededError("The crop changed while its local answer was being computed")
        self.input_size = getattr(local, "input_size", None)
        self.last_answer_was_remote = False
        if self._on_fallback is not None:
            try:
                self._on_fallback()
            except Exception:  # nosec B110
                pass
        return answer
