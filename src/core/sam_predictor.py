from __future__ import annotations

import base64
import json
import os
import subprocess  # nosec B404
import tempfile
import threading
import time

import numpy as np
from qgis.core import Qgis, QgsMessageLog

from .pip_diagnostics import get_app_control_help, is_antivirus_error, is_app_control_error
from .subprocess_utils import get_clean_env_for_venv, get_subprocess_kwargs  # nosec B404


class SamWorkerError(RuntimeError):
    """Error reported by the worker over the protocol.

    The subprocess is still alive and usable: callers must NOT tear it
    down, or every later click pays a full model reload.
    """


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


class SamPredictor:
    # Per-operation timeouts (seconds)
    # Model loading, plus the worker's warm-up pass at the real crop size (it
    # runs before the ready line so no click can queue behind it).
    _TIMEOUT_INIT = 240
    _TIMEOUT_RESET = 30
    _TIMEOUT_SET_IMAGE = 180  # Encoding 1 crop on slow CPU can take ~60s
    _TIMEOUT_PREDICT = 120
    # How many lines that are not the awaited answer _read_typed_response
    # steps over before it gives up on resynchronizing the pipe.
    _MAX_FOREIGN_LINES = 4

    def __init__(self, sam_config: dict, device: str | None = None) -> None:
        self.venv_python = sam_config["venv_python"]
        self.worker_script = sam_config["worker_script"]
        self.checkpoint = sam_config["checkpoint"]
        self.process = None
        self._stderr_file = None
        self._warming_up = False  # True when init sent but not yet confirmed
        self._last_worker_error = None  # Store error from worker for propagation
        self.is_image_set = False
        self.original_size = None
        self.input_size = None  # Only set by SAM1 path
        self._cleanup_lock = threading.Lock()

        QgsMessageLog.logMessage(
            "SAM Predictor initialized (subprocess mode)",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )

    def _read_stderr(self) -> str:
        """Read captured stderr from the worker subprocess."""
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
        """Read a line from the worker stdout with a timeout.

        Uses a daemon thread to perform the blocking readline so
        the main thread is not blocked indefinitely.

        A process that has already exited is NOT refused here. Its stdout may
        still hold the answer it managed to write before dying, and reading it
        is what tells a plain crash apart from a security block or an
        out-of-memory kill. At EOF readline returns "" straight away and the
        branch below turns that into a message carrying the exit code and the
        stderr tail, which beats the bare "Worker process is not running" the
        caller used to get. Callers that must not write to a dead worker
        already check poll() themselves before sending their request.
        """
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
        """Diagnostic text for a worker that stopped mid-exchange."""
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
        """Read the worker's answer, skipping anything that is not it.

        A single foreign line used to be fatal. Anything that was not the
        awaited answer (a stale reply from a request that timed out earlier, a
        library writing to stdout) raised "Unexpected response: {...}" and the
        generic handler tore the worker down, so the user lost the loaded
        model and read an internal protocol type. A half-written line was
        worse: json.loads surfaced its raw "Unterminated string starting at
        char 32", which names nothing the user can act on.

        Foreign lines are skipped instead, bounded in count AND inside the
        original time budget, so a silent worker still fails at
        `timeout_seconds`. This is what the init handshake already does.
        A `type: error` answer keeps the subprocess alive (SamWorkerError).
        """
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
                # A truncated line means the worker died while writing its
                # answer. An out-of-memory kill mid-way through a large mask
                # payload looks exactly like this, so say that rather than
                # repeating the parser's complaint.
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
        """Ensure subprocess is cleaned up on garbage collection."""
        self.cleanup()

    def _launch_process(self) -> bool:
        """Launch the subprocess and send init, but do NOT wait for response."""
        try:
            QgsMessageLog.logMessage(
                f"Starting prediction worker: {self.venv_python}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info
            )

            cmd = [self.venv_python, self.worker_script]

            env = get_clean_env_for_venv()
            subprocess_kwargs = get_subprocess_kwargs()

            # Capture stderr to temp file for crash diagnostics
            try:
                self._stderr_file = tempfile.TemporaryFile(
                    mode="w+", encoding="utf-8"
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
                "checkpoint_path": self.checkpoint
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
        """Wait for the worker to finish model loading (the 'ready' response).

        Skips non-JSON lines that may leak from library imports to stdout.
        """
        try:
            deadline = time.monotonic() + self._TIMEOUT_INIT
            skipped = 0
            max_skipped = 50

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Worker did not send ready within {self._TIMEOUT_INIT}s. "
                        "The SAM model may be too slow to load on this machine. "
                        "Try restarting QGIS and ensure no other heavy "
                        "processes are running.")

                # Check subprocess is still alive before blocking on read
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
        # If warm_up() already launched the process, just wait for ready
        if self._warming_up:
            self._warming_up = False
            if self.process is not None and self.process.poll() is None:
                return self._wait_for_ready()
            # Process died during warm-up, fall through to full start
            self.cleanup()

        if self.process is not None:
            return True

        if not self._launch_process():
            return False
        return self._wait_for_ready()

    def warm_up(self) -> bool:
        """Pre-start the worker subprocess and begin loading the SAM model.

        Call this when the user clicks 'Start AI Segmentation'.
        Launches the subprocess and sends the init command, but returns
        immediately without waiting for model loading to finish.
        The model loads in the background while the user positions their
        first click. The next call to set_image() will wait for ready
        if needed.
        """
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
                self.process = None  # Prevent re-entrant access
                try:
                    if proc.poll() is None:
                        try:
                            proc.stdin.write(json.dumps({"action": "quit"}) + "\n")
                            proc.stdin.flush()
                            proc.wait(timeout=2)
                        except (subprocess.TimeoutExpired, BrokenPipeError, OSError):
                            # Kill FIRST, then close stdout. A daemon reader
                            # thread may be mid readline() on proc.stdout (the
                            # read-timeout path in _read_response), and closing
                            # that pipe blocks behind the buffered-read lock
                            # until the child exits on its own, so a truly hung
                            # child would keep this thread stuck for its whole
                            # lifetime. Killing the child makes the pending
                            # readline hit EOF, which frees the lock; only then
                            # is closing the pipe instant.
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

            # Always close stderr file, even if process was already None
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

                # Reset is where a pipe left out of step by an earlier
                # timeout gets back in step, so it reads through stale
                # answers rather than warning and leaving them queued for
                # the next click to trip over.
                self._read_typed_response(
                    self._TIMEOUT_RESET, ("reset_done",), "reset confirmation",
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
        """Hand a request to a worker that is still loading its model.

        The worker reads its stdin from the moment it starts, so a crop sent
        here is in its hands while it loads, and it drops its own throwaway
        warm-up encode in favour of this one. Waiting for ready first, which is
        what set_image used to do, meant a user who clicked during the load
        paid that throwaway encode in full before their own.

        False when there is nothing to send it to, or the pipe refused. The
        caller then goes the ordinary way round, which reports the real reason
        the worker is not there.
        """
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

    def set_image(self, image_np: np.ndarray) -> None:
        """Send a raw image crop (H,W,3 uint8) to the worker for encoding.

        The worker calls SAM2ImagePredictor.set_image() which computes
        image embeddings for subsequent predict() calls.
        """
        # Validate the crop before shipping it to the worker. A malformed array
        # (wrong rank, zero-size, non-uint8) otherwise reaches SAM and throws an
        # opaque tensor-shape traceback that surfaces as a bare encoding_error.
        # Coerce the cheap cases (dtype, contiguity) and fail the rest clearly.
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

        request = {
            "action": "set_image",
            "image": base64.b64encode(image_np.tobytes()).decode("utf-8"),
            "image_shape": list(image_np.shape),
            "image_dtype": str(image_np.dtype),
        }
        # Overlap the crop with the model load when the two can overlap: send
        # first, THEN wait for ready. The worker answers in order, so the ready
        # line still arrives before the image_set line read below.
        sent_while_loading = self._send_request_early(request)

        if not self._start_worker():
            error = self._last_worker_error or "Failed to start prediction worker"
            self._last_worker_error = None
            raise RuntimeError(error)

        try:
            if not sent_while_loading:
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()

            response = self._read_typed_response(
                self._TIMEOUT_SET_IMAGE, ("image_set",), "encoded image",
                "Worker error encoding image")

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
            # Worker is still healthy: keep it alive for the next attempt.
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
            }

            # Add mask_input if provided (for iterative refinement with negative points)
            if mask_input is not None:
                request["mask_input"] = base64.b64encode(mask_input.tobytes()).decode("utf-8")
                request["mask_input_shape"] = list(mask_input.shape)
                request["mask_input_dtype"] = str(mask_input.dtype)

            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()

            response = self._read_typed_response(
                self._TIMEOUT_PREDICT, ("prediction",), "prediction",
                "Worker prediction error")

            masks_b64 = response["masks"]
            masks_shape = response["masks_shape"]
            masks_dtype = response["masks_dtype"]

            masks_bytes = base64.b64decode(masks_b64.encode("utf-8"))
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
            # Worker is still healthy: keep it alive (and the encoded image)
            # so the user can simply click again.
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
