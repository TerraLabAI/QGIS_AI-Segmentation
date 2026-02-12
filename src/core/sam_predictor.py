from typing import Tuple, Optional
import numpy as np
import os
import json
import subprocess
import threading
import tempfile
import base64

from qgis.core import QgsMessageLog, Qgis

from .subprocess_utils import get_clean_env_for_venv, get_subprocess_kwargs


def build_sam_vit_b_no_encoder(checkpoint: Optional[str] = None):
    from .venv_manager import get_venv_python_path, get_venv_dir

    plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_python = get_venv_python_path(get_venv_dir())
    worker_script = os.path.join(plugin_dir, 'workers', 'prediction_worker.py')

    if not os.path.exists(venv_python):
        raise FileNotFoundError(f"Virtual environment Python not found: {venv_python}")

    if not os.path.exists(worker_script):
        raise FileNotFoundError(f"Worker script not found: {worker_script}")

    return {
        'venv_python': venv_python,
        'worker_script': worker_script,
        'checkpoint': checkpoint
    }


class SamPredictorNoImgEncoder:
    # Per-operation timeouts (seconds)
    _TIMEOUT_INIT = 120       # Model loading
    _TIMEOUT_RESET = 30
    _TIMEOUT_SET_FEATURES = 60
    _TIMEOUT_PREDICT = 120

    def __init__(self, sam_config: dict, device: Optional[str] = None) -> None:
        self.venv_python = sam_config['venv_python']
        self.worker_script = sam_config['worker_script']
        self.checkpoint = sam_config['checkpoint']
        self.process = None
        self._stderr_file = None
        self.is_image_set = False
        self.original_size = None
        self.input_size = None

        class FakeModel:
            class FakeEncoder:
                img_size = 1024
            image_encoder = FakeEncoder()

        self.model = FakeModel()

        QgsMessageLog.logMessage(
            "SAM Predictor initialized (subprocess mode)",
            "AI Segmentation",
            level=Qgis.Info
        )

    def _read_stderr(self) -> str:
        """Read captured stderr from the worker subprocess."""
        if self._stderr_file is None:
            return ""
        try:
            self._stderr_file.seek(0)
            return self._stderr_file.read()
        except Exception:
            return ""

    def _read_response(self, timeout_seconds: int) -> str:
        """Read a line from the worker stdout with a timeout.

        Uses a daemon thread to perform the blocking readline so
        the main thread is not blocked indefinitely.
        """
        if self.process is None or self.process.poll() is not None:
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
            # Thread is still blocking - worker is hung
            raise TimeoutError(
                f"Worker did not respond within {timeout_seconds}s"
            )

        if error[0] is not None:
            raise error[0]

        line = result[0]
        if not line:
            stderr_output = self._read_stderr()
            msg = "Worker process closed stdout unexpectedly"
            if stderr_output:
                msg = "{}\nWorker stderr: {}".format(
                    msg, stderr_output[:500])
                QgsMessageLog.logMessage(
                    "Prediction worker stderr:\n{}".format(
                        stderr_output[:1000]),
                    "AI Segmentation",
                    level=Qgis.Critical
                )
            raise RuntimeError(msg)

        return line

    def __del__(self):
        """Ensure subprocess is cleaned up on garbage collection."""
        self.cleanup()

    def _start_worker(self) -> bool:
        if self.process is not None:
            return True

        try:
            QgsMessageLog.logMessage(
                f"Starting prediction worker: {self.venv_python}",
                "AI Segmentation",
                level=Qgis.Info
            )

            cmd = [self.venv_python, self.worker_script]

            env = get_clean_env_for_venv()
            subprocess_kwargs = get_subprocess_kwargs()

            # Capture stderr to temp file for crash diagnostics
            try:
                self._stderr_file = tempfile.TemporaryFile(
                    mode='w+', encoding='utf-8'
                )
            except Exception:
                self._stderr_file = None

            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=(
                    self._stderr_file if self._stderr_file is not None
                    else subprocess.DEVNULL
                ),
                text=True,
                bufsize=1,
                env=env,
                **subprocess_kwargs
            )

            init_request = {
                "action": "init",
                "checkpoint_path": self.checkpoint
            }

            self.process.stdin.write(json.dumps(init_request) + '\n')
            self.process.stdin.flush()

            response_line = self._read_response(self._TIMEOUT_INIT)
            response = json.loads(response_line.strip())

            if response.get("type") == "ready":
                QgsMessageLog.logMessage(
                    "Prediction worker ready",
                    "AI Segmentation",
                    level=Qgis.Success
                )
                return True
            elif response.get("type") == "error":
                error_msg = response.get("message", "Unknown error")
                QgsMessageLog.logMessage(
                    f"Worker initialization error: {error_msg}",
                    "AI Segmentation",
                    level=Qgis.Critical
                )
                self.cleanup()
                return False
            else:
                QgsMessageLog.logMessage(
                    f"Unexpected response from worker: {response}",
                    "AI Segmentation",
                    level=Qgis.Critical
                )
                self.cleanup()
                return False

        except Exception as e:
            import traceback
            error_msg = f"Failed to start prediction worker: {str(e)}\n{traceback.format_exc()}"
            QgsMessageLog.logMessage(error_msg, "AI Segmentation", level=Qgis.Critical)
            self.cleanup()
            return False

    def cleanup(self) -> None:
        if self.process is not None:
            try:
                if self.process.poll() is None:
                    try:
                        self.process.stdin.write(json.dumps({"action": "quit"}) + '\n')
                        self.process.stdin.flush()
                        self.process.wait(timeout=2)
                    except (subprocess.TimeoutExpired, BrokenPipeError, OSError):
                        # Close stdout to unblock any daemon thread stuck on readline()
                        try:
                            if self.process.stdout:
                                self.process.stdout.close()
                        except Exception:
                            pass
                        self.process.terminate()
                        try:
                            self.process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            self.process.kill()
                            self.process.wait(timeout=1)
            except Exception as e:
                QgsMessageLog.logMessage(
                    f"Warning during predictor cleanup: {str(e)}",
                    "AI Segmentation",
                    level=Qgis.Warning
                )
            finally:
                self.process = None
                if self._stderr_file is not None:
                    try:
                        self._stderr_file.close()
                    except Exception:
                        pass
                    self._stderr_file = None

        self.is_image_set = False

    def reset_image(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                request = {"action": "reset"}
                self.process.stdin.write(json.dumps(request) + '\n')
                self.process.stdin.flush()

                response_line = self._read_response(self._TIMEOUT_RESET)
                response = json.loads(response_line.strip())

                if response.get("type") != "reset_done":
                    QgsMessageLog.logMessage(
                        f"Unexpected reset response: {response}",
                        "AI Segmentation",
                        level=Qgis.Warning
                    )
            except Exception as e:
                QgsMessageLog.logMessage(
                    f"Error resetting image: {str(e)}",
                    "AI Segmentation",
                    level=Qgis.Warning
                )

        self.is_image_set = False
        self.original_size = None
        self.input_size = None

    def set_image_feature(
        self,
        img_features: np.ndarray,
        img_size: Tuple[int, int],
        input_size: Optional[Tuple[int, int]] = None
    ) -> None:
        if not self._start_worker():
            raise RuntimeError("Failed to start prediction worker")

        try:
            features_b64 = base64.b64encode(img_features.tobytes()).decode('utf-8')

            request = {
                "action": "set_features",
                "features": features_b64,
                "features_shape": list(img_features.shape),
                "features_dtype": str(img_features.dtype),
                "img_size": list(img_size),
                "input_size": list(input_size) if input_size else None
            }

            self.process.stdin.write(json.dumps(request) + '\n')
            self.process.stdin.flush()

            response_line = self._read_response(self._TIMEOUT_SET_FEATURES)
            response = json.loads(response_line.strip())

            if response.get("type") == "features_set":
                self.original_size = img_size
                self.input_size = input_size if input_size else img_size
                self.is_image_set = True

                QgsMessageLog.logMessage(
                    f"Set image features: shape={img_features.shape}, "
                    f"original_size={self.original_size}, input_size={self.input_size}",
                    "AI Segmentation",
                    level=Qgis.Info
                )
            elif response.get("type") == "error":
                error_msg = response.get("message", "Unknown error")
                raise RuntimeError(f"Worker error setting features: {error_msg}")
            else:
                raise RuntimeError(f"Unexpected response: {response}")

        except Exception as e:
            import traceback
            error_msg = f"Failed to set image features: {str(e)}\n{traceback.format_exc()}"
            QgsMessageLog.logMessage(error_msg, "AI Segmentation", level=Qgis.Critical)
            self.cleanup()
            raise

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = False,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_image_set:
            raise RuntimeError("Features have not been set. Call set_image_feature first.")

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
                request["mask_input"] = base64.b64encode(mask_input.tobytes()).decode('utf-8')
                request["mask_input_shape"] = list(mask_input.shape)
                request["mask_input_dtype"] = str(mask_input.dtype)

            self.process.stdin.write(json.dumps(request) + '\n')
            self.process.stdin.flush()

            response_line = self._read_response(self._TIMEOUT_PREDICT)
            response = json.loads(response_line.strip())

            if response.get("type") == "prediction":
                masks_b64 = response["masks"]
                masks_shape = response["masks_shape"]
                masks_dtype = response["masks_dtype"]

                masks_bytes = base64.b64decode(masks_b64.encode('utf-8'))
                masks = np.frombuffer(masks_bytes, dtype=masks_dtype).reshape(masks_shape)

                scores = np.array(response["scores"])

                low_res_masks_b64 = response["low_res_masks"]
                low_res_masks_shape = response["low_res_masks_shape"]
                low_res_masks_dtype = response["low_res_masks_dtype"]

                low_res_masks_bytes = base64.b64decode(low_res_masks_b64.encode('utf-8'))
                low_res_masks = np.frombuffer(low_res_masks_bytes, dtype=low_res_masks_dtype).reshape(low_res_masks_shape)

                return masks, scores, low_res_masks

            elif response.get("type") == "error":
                error_msg = response.get("message", "Unknown error")
                raise RuntimeError(f"Worker prediction error: {error_msg}")
            else:
                raise RuntimeError(f"Unexpected response: {response}")

        except Exception as e:
            import traceback
            error_msg = f"Prediction failed: {str(e)}\n{traceback.format_exc()}"
            QgsMessageLog.logMessage(error_msg, "AI Segmentation", level=Qgis.Critical)
            raise
