from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess  # nosec B404
import sys
import tempfile
import threading
import time
from collections import OrderedDict

import numpy as np
from qgis.core import Qgis, QgsMessageLog

from .pip_diagnostics import get_app_control_help, is_antivirus_error, is_app_control_error
from .subprocess_utils import get_clean_env_for_venv, get_subprocess_kwargs  # nosec B404



_WORKER_CROP_CACHE_SIZE = 4




_READER_JOIN_S = 0.2


class SamWorkerError(RuntimeError):
    pass






def build_sam_predictor_config(checkpoint: str | None = None):
    from .venv_manager import get_venv_dir, get_venv_python_path

    plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_python = get_venv_python_path(get_venv_dir())
    worker_script = os.path.join(plugin_dir, "workers", "prediction_worker.py")

    if not os.path.exists(venv_python):
        raise FileNotFoundError(f"Virtual environment Python not found: {venv_python}")

    if not os.path.exists(worker_script):
        raise FileNotFoundError(f"Worker script not found: {worker_script}")

    return {
        "venv_python": venv_python,
        "worker_script": worker_script,
        "checkpoint": checkpoint
    }


def _windows_kernel32():
    import ctypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    return kernel32


def open_windows_worker_handle(pid) -> int | None:






    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    try:
        synchronize_and_terminate = 0x00100000 | 0x0001
        return _windows_kernel32().OpenProcess(
            synchronize_and_terminate, False, pid) or None
    except Exception:  # noqa: BLE001
        return None


def end_windows_worker_process(handle, wait_s: float = 2.0) -> None:








    try:
        kernel32 = _windows_kernel32()
        try:
            wait_object_timeout = 0x102
            if kernel32.WaitForSingleObject(handle, int(wait_s * 1000)) == wait_object_timeout:
                kernel32.TerminateProcess(handle, 1)
                kernel32.WaitForSingleObject(handle, 1000 if wait_s > 0 else 0)
        finally:
            kernel32.CloseHandle(handle)
    except Exception:  # noqa: BLE001
        pass  # nosec B110


class SamPredictor:



    _TIMEOUT_INIT = 240
    _TIMEOUT_RESET = 30
    _TIMEOUT_SET_IMAGE = 180
    _TIMEOUT_PREDICT = 120


    _MAX_FOREIGN_LINES = 4

    def __init__(self, sam_config: dict, device: str | None = None) -> None:



        self._cleanup_lock = threading.Lock()
        from .server_dials import dial_in_range
        self._timeout_init = dial_in_range(
            "tuning.processing.sam_timeout_init_s", self._TIMEOUT_INIT, 30, 600)
        self._timeout_reset = dial_in_range(
            "tuning.processing.sam_timeout_reset_s", self._TIMEOUT_RESET, 5, 120)
        self._timeout_set_image = dial_in_range(
            "tuning.processing.sam_timeout_set_image_s", self._TIMEOUT_SET_IMAGE, 10, 600)
        self._timeout_predict = dial_in_range(
            "tuning.processing.sam_timeout_predict_s", self._TIMEOUT_PREDICT, 10, 600)
        self.venv_python = sam_config["venv_python"]
        self.worker_script = sam_config["worker_script"]
        self.checkpoint = sam_config["checkpoint"]
        self.process = None
        self._stderr_file = None



        self._worker_handle = None



        self._worker_crop_keys = OrderedDict()
        self._warming_up = False
        self._last_worker_error = None
        self.is_image_set = False
        self.original_size = None
        self.input_size = None




        self.low_res_side = 256


        self.last_answer_was_remote = False

        QgsMessageLog.logMessage(
            "SAM Predictor initialized (subprocess mode)",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )

    def _read_stderr(self) -> str:

        if self._stderr_file is None:
            return ""
        try:
            if self.process is not None:
                try:
                    self.process.wait(timeout=2)
                except (subprocess.TimeoutExpired, OSError):
                    pass
            self._stderr_file.seek(0)
            return self._stderr_file.read()
        except Exception:
            return ""

    def _read_response(self, timeout_seconds: int) -> str:














        if self.process is None:
            raise RuntimeError("Worker process is not running")

        result = [None]
        error = [None]

        def _reader():
            try:
                result[0] = self.process.stdout.readline()
            except Exception as e:
                error[0] = e

        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()
        reader_thread.join(timeout=timeout_seconds)

        if reader_thread.is_alive():
            self.cleanup()



            reader_thread.join(timeout=_READER_JOIN_S)
            raise TimeoutError(
                f"Worker did not respond within {timeout_seconds}s"
            )

        if error[0] is not None:
            raise error[0]

        line = result[0]
        if not line:
            exit_code = self.process.poll() if self.process else None
            stderr_output = self._read_stderr()
            if stderr_output and is_antivirus_error(stderr_output):
                if is_app_control_error(stderr_output):
                    from .cache_paths import PLUGIN_CACHE_DIR
                    raise RuntimeError(get_app_control_help(PLUGIN_CACHE_DIR))
                raise RuntimeError(
                    "A security policy is blocking the AI engine.\n\n"
                    "Ask your IT administrator to whitelist "
                    "this folder:\n"
                    f"  {os.path.dirname(self.venv_python)}\n\n"
                    "Then restart QGIS.")
            raise RuntimeError(self._worker_died_message(
                "Worker process closed stdout unexpectedly",
                exit_code, stderr_output))

        return line

    def _worker_died_message(
        self, prefix: str, exit_code=None, stderr_output: str | None = None
    ) -> str:

        if exit_code is None and self.process is not None:
            exit_code = self.process.poll()
        if stderr_output is None:
            stderr_output = self._read_stderr()
        msg = prefix
        if exit_code is not None:
            msg = f"{msg} (exit code {exit_code})"
        if stderr_output:
            msg = f"{msg}\nWorker stderr: {stderr_output[:500]}"
            QgsMessageLog.logMessage(
                f"Prediction worker stderr:\n{stderr_output[:1000]}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical
            )
        return msg

    def _read_typed_response(
        self,
        timeout_seconds: int,
        expected: tuple[str, ...],
        what: str,
        error_prefix: str,
    ) -> dict:















        deadline = time.monotonic() + timeout_seconds
        for _ in range(self._MAX_FOREIGN_LINES + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.cleanup()
                raise TimeoutError(
                    f"Worker did not respond within {timeout_seconds}s")
            stripped = self._read_response(max(1, int(remaining))).strip()
            if not stripped:
                continue
            try:
                response = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):




                if self.process is None or self.process.poll() is not None:
                    raise RuntimeError(self._worker_died_message(
                        "The AI engine stopped while sending its answer. "
                        "It most likely ran out of memory: close other "
                        "applications, or work on a smaller area, then try "
                        "again.")) from None
                QgsMessageLog.logMessage(
                    f"Skipping non-JSON worker output: {stripped[:200]}",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Warning
                )
                continue
            if not isinstance(response, dict):
                continue
            response_type = response.get("type")
            if response_type in expected:
                return response
            if response_type == "error":
                raise SamWorkerError(
                    f"{error_prefix}: {response.get('message', 'Unknown error')}")
            QgsMessageLog.logMessage(
                f"Skipping stale worker response while awaiting {what}: "
                f"{response_type}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )
        raise RuntimeError(
            f"The AI engine kept answering something other than the {what}. "
            "Click again to restart it.")

    def __del__(self):
        self._drop_process_without_waiting()

    def _drop_process_without_waiting(self):













        lock = getattr(self, "_cleanup_lock", None)
        if lock is None or not lock.acquire(blocking=False):
            return
        try:
            proc = getattr(self, "process", None)
            self.process = None
            if proc is not None:
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass  # nosec B110
            handle = getattr(self, "_worker_handle", None)
            self._worker_handle = None
            if handle is not None:
                end_windows_worker_process(handle, wait_s=0)
            stderr_file = getattr(self, "_stderr_file", None)
            self._stderr_file = None
            if stderr_file is not None:
                try:
                    stderr_file.close()
                except Exception:  # noqa: BLE001
                    pass  # nosec B110
            self._warming_up = False
            self.is_image_set = False
        finally:
            lock.release()

    def _launch_process(self) -> bool:

        try:
            QgsMessageLog.logMessage(
                f"Starting prediction worker: {self.venv_python}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info
            )

            cmd = [self.venv_python, self.worker_script]
            self._worker_crop_keys.clear()

            env = get_clean_env_for_venv()
            subprocess_kwargs = get_subprocess_kwargs()


            try:





                self._stderr_file = tempfile.TemporaryFile(
                    mode="w+", encoding="utf-8", errors="replace"
                )
            except Exception:
                self._stderr_file = None

            self.process = subprocess.Popen(  # nosec B603
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=(
                    self._stderr_file if self._stderr_file is not None
                    else subprocess.DEVNULL
                ),
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                **subprocess_kwargs
            )

            init_request = {
                "action": "init",
                "checkpoint_path": self.checkpoint,


                "parent_pid": os.getpid(),
            }

            self.process.stdin.write(json.dumps(init_request) + "\n")
            self.process.stdin.flush()
            return True

        except Exception as e:
            import traceback
            error_msg = f"Failed to launch prediction worker: {str(e)}\n{traceback.format_exc()}"
            QgsMessageLog.logMessage(error_msg, "AI Segmentation", level=Qgis.MessageLevel.Critical)
            self.cleanup()
            return False

    def _wait_for_ready(self) -> bool:




        try:
            deadline = time.monotonic() + self._timeout_init
            skipped = 0
            max_skipped = 50

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Worker did not send ready within {self._timeout_init}s. "
                        "The SAM model may be too slow to load on this machine. "
                        "Try restarting QGIS and ensure no other heavy "
                        "processes are running.")


                if self.process is not None and self.process.poll() is not None:
                    exit_code = self.process.poll()
                    stderr_output = self._read_stderr()
                    if stderr_output and is_antivirus_error(stderr_output):
                        if is_app_control_error(stderr_output):
                            from .cache_paths import PLUGIN_CACHE_DIR
                            raise RuntimeError(get_app_control_help(PLUGIN_CACHE_DIR))
                        raise RuntimeError(
                            "A security policy is blocking the AI engine.\n\n"
                            "Ask your IT administrator to whitelist "
                            "this folder:\n"
                            f"  {os.path.dirname(self.venv_python)}\n\n"
                            "Then restart QGIS.")
                    raise RuntimeError(
                        "Worker process died during initialization "
                        "(exit code {}){}".format(
                            exit_code,
                            "\nWorker stderr: " + stderr_output[:500]
                            if stderr_output else ""))

                response_line = self._read_response(
                    max(1, int(remaining)))
                stripped = response_line.strip()

                if not stripped:
                    skipped += 1
                    if skipped >= max_skipped:
                        raise RuntimeError(
                            f"Skipped {max_skipped} blank lines without valid JSON")
                    continue

                try:
                    response = json.loads(stripped)
                except (json.JSONDecodeError, ValueError):
                    skipped += 1
                    QgsMessageLog.logMessage(
                        f"Skipping non-JSON worker output: {stripped[:200]}",
                        "AI Segmentation",
                        level=Qgis.MessageLevel.Warning
                    )
                    if skipped >= max_skipped:
                        raise RuntimeError(
                            f"Skipped {max_skipped} non-JSON lines without valid response") from None
                    continue

                if response.get("type") == "ready":
                    if sys.platform == "win32" and self._worker_handle is None:
                        self._worker_handle = open_windows_worker_handle(
                            response.get("pid"))
                    QgsMessageLog.logMessage(
                        "Prediction worker ready",
                        "AI Segmentation",
                        level=Qgis.MessageLevel.Success
                    )
                    return True
                if response.get("type") == "error":
                    error_msg = response.get("message", "Unknown error")
                    self._last_worker_error = error_msg
                    QgsMessageLog.logMessage(
                        f"Worker initialization error: {error_msg}",
                        "AI Segmentation",
                        level=Qgis.MessageLevel.Critical
                    )
                    self.cleanup()
                    return False
                QgsMessageLog.logMessage(
                    f"Unexpected response from worker: {response}",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Critical
                )
                self.cleanup()
                return False

        except Exception as e:
            import traceback
            error_msg = f"Failed waiting for worker ready: {str(e)}\n{traceback.format_exc()}"
            QgsMessageLog.logMessage(
                error_msg, "AI Segmentation",
                level=Qgis.MessageLevel.Critical)
            self.cleanup()
            return False

    def _start_worker(self) -> bool:

        if self._warming_up:
            self._warming_up = False
            if self.process is not None and self.process.poll() is None:
                return self._wait_for_ready()

            self.cleanup()

        if self.process is not None:
            return True

        if not self._launch_process():
            return False
        return self._wait_for_ready()

    def warm_up(self) -> bool:









        if self.process is not None:
            return True
        if not self._launch_process():
            return False
        self._warming_up = True
        return True

    def cleanup(self) -> None:
        with self._cleanup_lock:
            if self.process is not None:
                proc = self.process
                self.process = None
                try:
                    if proc.poll() is None:
                        try:
                            proc.stdin.write(json.dumps({"action": "quit"}) + "\n")
                            proc.stdin.flush()
                            proc.wait(timeout=2)
                        except (subprocess.TimeoutExpired, BrokenPipeError, OSError):









                            proc.terminate()
                            try:
                                proc.wait(timeout=2)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                try:
                                    proc.wait(timeout=1)
                                except Exception:
                                    pass  # nosec B110
                            try:
                                if proc.stdout:
                                    proc.stdout.close()
                            except Exception:
                                pass  # nosec B110
                except Exception as e:
                    QgsMessageLog.logMessage(
                        f"Warning during predictor cleanup: {str(e)}",
                        "AI Segmentation",
                        level=Qgis.MessageLevel.Warning
                    )
            if self._worker_handle is not None:
                end_windows_worker_process(self._worker_handle)
                self._worker_handle = None
            self._worker_crop_keys.clear()


            if self._stderr_file is not None:
                try:
                    self._stderr_file.close()
                except Exception:
                    pass  # nosec B110
                self._stderr_file = None

            self._warming_up = False
            self.is_image_set = False

    def reset_image(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                request = {"action": "reset"}
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()


                self._worker_crop_keys.clear()




                self._read_typed_response(
                    self._timeout_reset, ("reset_done",), "reset confirmation",
                    "Worker error resetting image")
            except Exception as e:
                QgsMessageLog.logMessage(
                    f"Error resetting image: {str(e)}",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Warning
                )

        self.is_image_set = False
        self.original_size = None
        self.input_size = None

    def _send_request_early(self, request: dict) -> bool:












        if self.process is None or not self._warming_up:
            return False
        try:
            if self.process.poll() is not None:
                return False
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError, AttributeError):
            return False
        return True

    def _write_request(self, request: dict) -> None:
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()

    def _remember_worker_crop(self, digest: str, worker_key) -> None:
        if not isinstance(worker_key, str) or not worker_key:
            return
        self._worker_crop_keys[digest] = worker_key
        self._worker_crop_keys.move_to_end(digest)
        while len(self._worker_crop_keys) > _WORKER_CROP_CACHE_SIZE:
            self._worker_crop_keys.popitem(last=False)

    def set_image(self, image_np: np.ndarray) -> None:









        if not isinstance(image_np, np.ndarray) or image_np.ndim != 3 or image_np.shape[2] != 3:
            shape = getattr(image_np, "shape", None)
            raise SamWorkerError(
                f"Invalid image for encoding: expected (H, W, 3), got shape {shape}")
        if image_np.shape[0] == 0 or image_np.shape[1] == 0:
            raise SamWorkerError(
                f"Invalid image for encoding: empty crop {image_np.shape}")
        if image_np.dtype != np.uint8:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        image_np = np.ascontiguousarray(image_np)

        digest = hashlib.blake2b(image_np, digest_size=16).hexdigest()
        digest = f"{digest}:{image_np.shape}"
        worker_key = (self._worker_crop_keys.get(digest)
                      if self.process is not None and not self._warming_up
                      else None)

        def full_request() -> dict:
            return {
                "action": "set_image",
                "image": base64.b64encode(image_np).decode("ascii"),
                "image_shape": list(image_np.shape),
                "image_dtype": str(image_np.dtype),
            }

        request = full_request() if worker_key is None else {
            "action": "set_image",
            "image_key": worker_key,
            "image_shape": list(image_np.shape),
            "image_dtype": str(image_np.dtype),
        }



        sent_while_loading = self._send_request_early(request)

        if not self._start_worker():
            error = self._last_worker_error or "Failed to start prediction worker"
            self._last_worker_error = None
            raise RuntimeError(error)

        try:
            if not sent_while_loading:
                self._write_request(request)

            response = self._read_typed_response(
                self._timeout_set_image, ("image_set", "image_missing"),
                "encoded image", "Worker error encoding image")
            if response.get("type") == "image_missing":

                self._worker_crop_keys.pop(digest, None)
                self._write_request(full_request())
                response = self._read_typed_response(
                    self._timeout_set_image, ("image_set",), "encoded image",
                    "Worker error encoding image")
            self._remember_worker_crop(digest, response.get("image_key"))

            self.original_size = tuple(response["original_size"])
            if "input_size" in response:
                self.input_size = tuple(response["input_size"])
            else:
                self.input_size = None
            self.is_image_set = True

            QgsMessageLog.logMessage(
                f"Set image: original_size={self.original_size}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info
            )

        except SamWorkerError as e:

            QgsMessageLog.logMessage(
                f"Failed to encode image: {str(e)}",
                "AI Segmentation", level=Qgis.MessageLevel.Critical)
            raise
        except Exception as e:
            import traceback
            error_msg = f"Failed to encode image: {str(e)}\n{traceback.format_exc()}"
            QgsMessageLog.logMessage(
                error_msg, "AI Segmentation", level=Qgis.MessageLevel.Critical)
            self.cleanup()
            raise

    def predict(
        self,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        mask_input: np.ndarray | None = None,
        multimask_output: bool = False,
        return_logits: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_image_set:
            raise RuntimeError("Image has not been set. Call set_image first.")

        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Prediction worker is not running")

        try:
            request = {
                "action": "predict",
                "point_coords": point_coords.tolist() if point_coords is not None else None,
                "point_labels": point_labels.tolist() if point_labels is not None else None,
                "multimask_output": multimask_output,


                "masks_packed": True,
            }


            if mask_input is not None:



                if mask_input.ndim != 3 or mask_input.shape[0] != 1:
                    raise ValueError(
                        "Invalid mask seed for prediction: expected (1, H, W), "
                        f"got shape {tuple(mask_input.shape)}")
                request["mask_input"] = base64.b64encode(mask_input.tobytes()).decode("utf-8")
                request["mask_input_shape"] = list(mask_input.shape)
                request["mask_input_dtype"] = str(mask_input.dtype)

            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()

            response = self._read_typed_response(
                self._timeout_predict, ("prediction",), "prediction",
                "Worker prediction error")

            masks_b64 = response["masks"]
            masks_shape = response["masks_shape"]
            masks_dtype = response["masks_dtype"]

            masks_bytes = base64.b64decode(masks_b64)
            if response.get("masks_packed"):
                masks = np.unpackbits(
                    np.frombuffer(masks_bytes, dtype=np.uint8),
                    count=int(np.prod(masks_shape)),
                ).reshape(masks_shape)
            else:
                masks = np.frombuffer(masks_bytes, dtype=masks_dtype).reshape(masks_shape)

            scores = np.array(response["scores"])

            low_res_masks_b64 = response["low_res_masks"]
            low_res_masks_shape = response["low_res_masks_shape"]
            low_res_masks_dtype = response["low_res_masks_dtype"]

            low_res_masks_bytes = base64.b64decode(low_res_masks_b64.encode("utf-8"))
            low_res_masks = np.frombuffer(
                low_res_masks_bytes, dtype=low_res_masks_dtype
            ).reshape(low_res_masks_shape)

            return masks, scores, low_res_masks

        except SamWorkerError as e:


            QgsMessageLog.logMessage(
                f"Prediction failed: {str(e)}",
                "AI Segmentation", level=Qgis.MessageLevel.Critical)
            raise
        except Exception as e:
            import traceback
            error_msg = f"Prediction failed: {str(e)}\n{traceback.format_exc()}"
            QgsMessageLog.logMessage(error_msg, "AI Segmentation", level=Qgis.MessageLevel.Critical)
            self.cleanup()
            raise
