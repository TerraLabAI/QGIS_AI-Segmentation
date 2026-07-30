#!/usr/bin/env python3
import base64
import hashlib
import json
import os
import queue
import sys
import threading
from collections import OrderedDict

# Plugin is CPU-only on NVIDIA; hide devices before torch imports (#31).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Redirect stdout to stderr during library imports to prevent
# torch/sam2 print() calls from corrupting the JSON protocol.
_real_stdout = sys.stdout
sys.stdout = sys.stderr

# On Windows, register torch DLL directories in this subprocess.
# os.add_dll_directory() calls from the parent process do not propagate,
# so the subprocess must register them itself for c10.dll etc. to load.
if sys.platform == "win32":
    _site_packages = None
    for p in sys.path:
        if p.endswith("site-packages") and os.path.isdir(p):
            _site_packages = p
            break
    if _site_packages:
        for _subdir in ("torch\\lib", "torch\\bin", "torchvision"):
            _dll_dir = os.path.join(_site_packages, _subdir)
            if os.path.isdir(_dll_dir):
                try:
                    os.add_dll_directory(_dll_dir)
                except OSError:
                    pass

try:
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
except ImportError as e:
    sys.stdout = _real_stdout
    error_msg = {
        "type": "error",
        "message": f"Failed to import dependencies: {str(e)}. "
                   "Please reinstall dependencies."
    }
    print(json.dumps(error_msg), flush=True)
    sys.exit(1)
except OSError as e:
    sys.stdout = _real_stdout
    err_str = str(e)
    err_lower = err_str.lower()
    # Detect Application Control / AppLocker blocking DLL loading. This
    # worker runs standalone in the venv (no plugin imports), so the
    # pattern list mirrors pip_diagnostics._APP_CONTROL_PATTERNS: the
    # locale-independent "4551" code plus localized Windows wordings.
    if any(m in err_lower for m in (
        "application control", "applocker", "blocked by your organization",
        "blocked by group policy", "winerror 4551", "os error 4551",
        "control de aplicaciones",                # es
        "strategie de controle d'application",    # fr (accents stripped)
        "stratégie de contrôle d'application",    # fr
        "beleid voor toepassingsbeheer",          # nl
    )):
        error_msg = {
            "type": "error",
            "message": (
                "Your organization's security policy is blocking the "
                "AI engine.\n\n"
                "Ask your IT administrator to add a path-based allow rule "
                "for the plugin's environment folder "
                "(~/.qgis_ai_segmentation). One rule keeps working across "
                "updates. Then restart QGIS."
            ),
        }
    # Catch Windows DLL loading errors (shm.dll, etc.)
    elif "shm.dll" in err_str or "DLL" in err_str.upper():
        error_msg = {
            "type": "error",
            "message": f"PyTorch DLL error (Windows): {err_str}. "
                       "Try: 1) Install Visual C++ Redistributables from "
                       "https://aka.ms/vs/17/release/vc_redist.x64.exe "
                       "2) If already installed, reinstall the plugin dependencies "
                       "(Settings > Reinstall dependencies)."
        }
    else:
        error_msg = {
            "type": "error",
            "message": f"Failed to load PyTorch: {err_str}"
        }
    print(json.dumps(error_msg), flush=True)
    sys.exit(1)

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    _USE_SAM2 = True
except ImportError:
    _USE_SAM2 = False

# Restore real stdout now that imports are done
sys.stdout = _real_stdout

SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


def build_sam2_model(checkpoint, device):
    """Build SAM 2.1 Base Plus model."""
    return build_sam2(
        SAM2_MODEL_CFG, checkpoint,
        device=str(device), mode="eval"
    )


def build_sam1_model(checkpoint, device):
    """Build SAM ViT-B model (Python 3.9 fallback)."""
    from segment_anything import sam_model_registry
    model = sam_model_registry["vit_b"](checkpoint=checkpoint)
    model.to(device)
    model.eval()
    return model


def resolve_venv_torch_device():
    if sys.platform == "darwin":
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                test = torch.zeros(1, device="mps")
                _ = test + 1
                torch.mps.synchronize()
                del test
                os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
                return torch.device("mps")
        except Exception:
            pass  # nosec B110

    num_cores = os.cpu_count() or 4
    # Leave the machine a core for QGIS: the image encode runs for seconds and
    # a fully saturated CPU freezes the canvas, the event loop and the plugin's
    # own array work. Torch still keeps at least half the cores, so the encode
    # stays close to full speed.
    optimal_threads = max(1, min(num_cores - 1, max(2, num_cores // 2)))
    torch.set_num_threads(optimal_threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(max(2, optimal_threads // 2))
        except RuntimeError:
            pass
    return torch.device("cpu")


# Encoded crops kept in memory, most-recently-used last. The image encoder is
# the entire cost of set_image (seconds); the mask decoder that answers a click
# is milliseconds. A fix pass walks the same neighbourhood over and over, so the
# same crop comes back, and a repeat has to be a dict lookup instead of a second
# forward pass. Content-addressed on the crop's own bytes, so a caller re-sending
# the same pixels needs to know nothing about this. Each entry holds the encoder
# feature maps for one crop, so the ceiling is a memory budget, not a count.
ENCODED_CROP_CACHE_MAX = 4
_encoded_crop_cache = OrderedDict()


def crop_cache_key(image_bytes, shape):
    """Content address for one crop: its bytes plus its shape."""
    digest = hashlib.blake2b(image_bytes, digest_size=16).hexdigest()
    return f"{digest}:{'x'.join(str(n) for n in shape)}"


def capture_encoded_crop(predictor):
    """The predictor's post-encode state for the crop just encoded."""
    if _USE_SAM2:
        return {
            "_features": predictor._features,
            "_orig_hw": predictor._orig_hw,
        }
    return {
        "features": predictor.features,
        "original_size": predictor.original_size,
        "input_size": predictor.input_size,
    }


def restore_encoded_crop(predictor, state):
    """Put a cached crop back into the predictor, skipping the encoder."""
    if _USE_SAM2:
        predictor._features = state["_features"]
        predictor._orig_hw = state["_orig_hw"]
        predictor._is_batch = False
        predictor._is_image_set = True
        return
    predictor.features = state["features"]
    predictor.original_size = state["original_size"]
    predictor.input_size = state["input_size"]
    predictor.is_image_set = True


def encode_or_reuse_crop(predictor, request):
    """Make one crop the predictor's current image, and answer for it.

    Runs the image encoder only for a crop this process has not seen lately.
    Returns the ``image_set`` payload; ``cached`` says which of the two happened,
    which is advisory (older callers ignore the field).
    """
    image_shape = request["image_shape"]
    image_bytes = base64.b64decode(request["image"].encode("utf-8"))
    key = crop_cache_key(image_bytes, image_shape)

    cached = _encoded_crop_cache.get(key)
    if cached is not None:
        _encoded_crop_cache.move_to_end(key)
        restore_encoded_crop(predictor, cached)
        original_size = tuple(image_shape[:2])
    else:
        image_np = np.frombuffer(
            image_bytes, dtype=request["image_dtype"]).reshape(image_shape)
        with torch.inference_mode():
            predictor.set_image(image_np)
        original_size = image_np.shape[:2]
        if ENCODED_CROP_CACHE_MAX > 0:
            _encoded_crop_cache[key] = capture_encoded_crop(predictor)
            while len(_encoded_crop_cache) > ENCODED_CROP_CACHE_MAX:
                _encoded_crop_cache.popitem(last=False)

    response_data = {
        "original_size": list(original_size),
        "cached": cached is not None,
    }
    # SAM1 exposes the resized input size through its transform.
    if not _USE_SAM2 and hasattr(predictor, "input_size"):
        response_data["input_size"] = list(predictor.input_size)
    return response_data


def warm_up_kernels(predictor, device):
    """Run one throwaway encode + one throwaway click at the real crop size.

    The first forward pass of a process pays for kernel compilation and
    allocator growth on top of the actual work, so the user's first click was
    the slowest one of the session. Doing it here, while nobody is waiting,
    hands the first real click a steady-state model. Never fatal, never
    cached, and the predictor is reset afterwards so it holds no stale image.
    """
    try:
        blank = np.full((1024, 1024, 3), 128, dtype=np.uint8)
        with torch.inference_mode():
            predictor.set_image(blank)
            predictor.predict(
                point_coords=np.array([[512, 512]]),
                point_labels=np.array([1]),
                multimask_output=False,
            )
        if _USE_SAM2:
            predictor.reset_predictor()
        else:
            predictor.reset_image()
        if device.type == "mps":
            torch.mps.synchronize()
        return True
    except Exception as e:  # noqa: BLE001 - a cold model still works, just slower
        sys.stderr.write(f"[prediction_worker] warm-up skipped: {e}\n")
        sys.stderr.flush()
        return False


def send_response(response_type, data):
    response = {"type": response_type, **data}
    _real_stdout.write(json.dumps(response) + "\n")
    _real_stdout.flush()


def send_error(error_message):
    send_response("error", {"message": error_message})


def send_ready():
    send_response("ready", {})


def encode_numpy_array(arr):
    return base64.b64encode(arr.tobytes()).decode("utf-8")


def decode_numpy_array(b64_string, shape, dtype):
    bytes_data = base64.b64decode(b64_string.encode("utf-8"))
    arr = np.frombuffer(bytes_data, dtype=dtype)
    return arr.reshape(shape)


MAX_LINE_LENGTH = 50 * 1024 * 1024  # 50 MB max JSON line

# One daemon thread owns stdin and every read goes through its queue, the init
# request included, so no line can be lost between two readers. What that buys
# is the question `_stdin_request_waiting` answers: is somebody already waiting
# on this worker, asked without blocking on a read that may never come.
#
# The queue is bounded because the length limit above is a memory guard. The
# parent keeps one request in flight, so a queue this short only fills when
# something upstream misbehaves, and then the extra lines wait in the pipe
# rather than in this process.
STDIN_QUEUE_MAX = 4
_stdin_lines = queue.Queue(maxsize=STDIN_QUEUE_MAX)
_stdin_reader_started = False
_stdin_at_eof = False


def _pump_stdin_lines():
    """Move stdin into the queue, line by line, until the input ends.

    The empty string is the end marker, which is what a plain readline returns
    at EOF, so the consumer sees the same thing either way. A read that fails
    ends the input the same way rather than leaving the consumer waiting."""
    while True:
        try:
            line = sys.stdin.readline()
        except Exception:  # noqa: BLE001 - a broken pipe is the end of input
            line = ""
        _stdin_lines.put(line)
        if not line:
            return


def _start_stdin_reader():
    """Start the one stdin reader, before the model load, so a request sent
    during that load is already in hand when it ends."""
    global _stdin_reader_started
    thread = threading.Thread(
        target=_pump_stdin_lines, name="stdin-reader", daemon=True)
    thread.start()
    _stdin_reader_started = True
    return thread


def _stdin_request_waiting():
    """Is a request already queued, i.e. is somebody waiting on this worker?"""
    return not _stdin_lines.empty()


def _safe_readline():
    """Next line of stdin, taken from the reader thread.

    Same contract as the plain readline it replaces: it blocks until a line
    arrives, returns "" once the input has ended, and refuses a line over
    MAX_LINE_LENGTH so one send cannot exhaust memory. With no reader running
    it reads stdin itself, so a thread that never started costs the skip
    decision, not the protocol."""
    global _stdin_at_eof
    if _stdin_at_eof:
        return ""
    line = _stdin_lines.get() if _stdin_reader_started else sys.stdin.readline()
    if not line:
        _stdin_at_eof = True
        return ""
    if len(line) > MAX_LINE_LENGTH:
        raise ValueError(
            f"Input line exceeds maximum length ({MAX_LINE_LENGTH} bytes)")
    return line


def main():
    try:
        # Read stdin from here on, so a request sent while the model loads lands
        # in the queue instead of sitting unseen in the pipe.
        _start_stdin_reader()

        init_request = json.loads(_safe_readline())

        if init_request.get("action") != "init":
            send_error("First request must be 'init'")
            sys.exit(1)

        checkpoint_path = init_request.get("checkpoint_path")
        if not checkpoint_path or not isinstance(checkpoint_path, str):
            send_error("Invalid or missing checkpoint_path")
            sys.exit(1)
        checkpoint_path = os.path.normpath(os.path.abspath(checkpoint_path))
        if not os.path.isfile(checkpoint_path):
            send_error(f"Checkpoint file not found: {checkpoint_path}")
            sys.exit(1)

        device = resolve_venv_torch_device()

        if _USE_SAM2:
            sam_model = build_sam2_model(checkpoint_path, device)
            predictor = SAM2ImagePredictor(sam_model)
            model_label = "SAM2.1"
        else:
            sam_model = build_sam1_model(checkpoint_path, device)
            from segment_anything import SamPredictor as Sam1Predictor
            predictor = Sam1Predictor(sam_model)
            model_label = "SAM1-ViT-B"

        device_label = str(device)
        if device.type == "cpu":
            device_label = f"cpu ({torch.get_num_threads()}t)"
        sys.stderr.write(
            f"[prediction_worker] {model_label}, PyTorch={torch.__version__}, "
            f"device={device_label}, "
            f"Python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n"
        )
        sys.stderr.flush()

        # Before announcing ready, not after: the caller does not block on this
        # spawn (warm_up returns as soon as init is sent), so the seconds spent
        # here are seconds nobody is waiting on. Announcing first would let a
        # real crop queue up behind the throwaway one.
        #
        # Unless somebody is already waiting. A request that arrived during the
        # model load means a caller is blocked on this process right now, and a
        # throwaway encode ahead of it would be added to that wait. Their own
        # first encode pays for the kernel compilation instead: this warm-up
        # moves that cost, it never removes it.
        if _stdin_request_waiting():
            sys.stderr.write(
                "[prediction_worker] warm-up skipped: a request is already "
                "waiting\n")
            sys.stderr.flush()
        else:
            import time as _time
            _t_warm = _time.monotonic()
            if warm_up_kernels(predictor, device):
                sys.stderr.write(
                    "[prediction_worker] warmed in "
                    f"{(_time.monotonic() - _t_warm) * 1000:.0f} ms\n")
                sys.stderr.flush()

        send_ready()

        while True:
            line = _safe_readline()
            if not line:
                break

            try:
                request = json.loads(line)
                action = request.get("action")

                if action == "set_image":
                    send_response("image_set",
                                  encode_or_reuse_crop(predictor, request))

                elif action == "predict":
                    point_coords = np.array(
                        request["point_coords"]) if request.get("point_coords") else None
                    point_labels = np.array(
                        request["point_labels"]) if request.get("point_labels") else None
                    multimask_output = request.get("multimask_output", False)

                    # Decode mask_input if provided (for iterative refinement)
                    mask_input = None
                    if request.get("mask_input"):
                        mask_input = decode_numpy_array(
                            request["mask_input"],
                            request["mask_input_shape"],
                            request["mask_input_dtype"]
                        )

                    # Auto-select best mask: when caller requests single mask
                    # and no mask_input (first click), use multimask internally
                    # and pick the highest-scoring one for better accuracy.
                    auto_best = (not multimask_output and mask_input is None)
                    effective_multimask = True if auto_best else multimask_output

                    predict_kwargs = {
                        "point_coords": point_coords,
                        "point_labels": point_labels,
                        "mask_input": mask_input,
                        "multimask_output": effective_multimask,
                    }
                    if _USE_SAM2:
                        predict_kwargs["normalize_coords"] = True

                    with torch.inference_mode():
                        masks, scores, low_res_masks = predictor.predict(
                            **predict_kwargs)

                    # When auto-selecting best mask, prefer non-empty masks
                    # below 80% of the crop (same convention as the caller's
                    # first-click pick) so a whole-crop or empty candidate
                    # never wins on score alone.
                    if auto_best and masks.shape[0] > 1:
                        total = masks.shape[1] * masks.shape[2]
                        areas = [int(m.sum()) for m in masks]
                        candidates = [
                            i for i in range(len(scores))
                            if 0 < areas[i] < 0.8 * total
                        ]
                        if candidates:
                            best_idx = max(
                                candidates, key=lambda i: float(scores[i]))
                        else:
                            best_idx = int(np.argmax(scores))
                        masks = masks[best_idx:best_idx + 1]
                        scores = scores[best_idx:best_idx + 1]
                        low_res_masks = low_res_masks[best_idx:best_idx + 1]

                    # Empty masks are a normal result ("nothing here"), not an
                    # error: the caller reverts the click with a friendly
                    # message. Reporting an error here used to kill the whole
                    # worker session.

                    send_response("prediction", {
                        "masks": encode_numpy_array(masks),
                        "masks_shape": list(masks.shape),
                        "masks_dtype": str(masks.dtype),
                        "scores": scores.tolist(),
                        "low_res_masks": encode_numpy_array(low_res_masks),
                        "low_res_masks_shape": list(low_res_masks.shape),
                        "low_res_masks_dtype": str(low_res_masks.dtype),
                    })

                elif action == "reset":
                    if _USE_SAM2:
                        predictor.reset_predictor()
                    else:
                        predictor.reset_image()
                    send_response("reset_done", {})

                elif action == "quit":
                    break

                else:
                    send_error(f"Unknown action: {action}")

            except Exception as e:
                import traceback
                send_error(f"Error processing request: {str(e)}\n{traceback.format_exc()}")

    except Exception as e:
        import traceback
        send_error(f"Worker initialization failed: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
