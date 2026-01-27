#!/usr/bin/env python3
import base64
import hashlib
import json
import os
import queue
import sys
import threading
from collections import OrderedDict


os.environ["CUDA_VISIBLE_DEVICES"] = ""

if sys.platform == "win32":






    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")





    os.environ.setdefault("KMP_BLOCKTIME", "20")



    try:
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    except Exception:  # noqa: BLE001
        pass  # nosec B110


def opt_out_of_windows_power_throttling():









    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class _PowerThrottlingState(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG),
                        ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessInformation.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        process_power_throttling = 4
        execution_speed = 0x1

        state = _PowerThrottlingState(1, execution_speed, 0)
        return bool(kernel32.SetProcessInformation(
            kernel32.GetCurrentProcess(), process_power_throttling,
            ctypes.byref(state), ctypes.sizeof(state)))
    except Exception:  # noqa: BLE001
        return False



opt_out_of_windows_power_throttling()



_real_stdout = sys.stdout
sys.stdout = sys.stderr




_dll_directory_handles = []
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
                    _dll_directory_handles.append(os.add_dll_directory(_dll_dir))
                except OSError:
                    pass


def _emit_error(msg: dict) -> None:

    print(json.dumps(msg), flush=True)


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
    _emit_error(error_msg)
    sys.exit(1)
except OSError as e:
    sys.stdout = _real_stdout
    err_str = str(e)
    err_lower = err_str.lower()




    if any(m in err_lower for m in (
        "application control", "applocker", "blocked by your organization",
        "blocked by group policy", "winerror 4551", "os error 4551",
        "control de aplicaciones",
        "strategie de controle d'application",
        "stratégie de contrôle d'application",
        "beleid voor toepassingsbeheer",
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

    elif "shm.dll" in err_str or "DLL" in err_str.upper():
        error_msg = {
            "type": "error",
            "message": f"PyTorch DLL error (Windows): {err_str}. "
                       "Try: 1) Install Visual C++ Redistributables from "
                       "https://aka.ms/vs/17/release/vc_redist.x64.exe "
                       "2) If already installed, open the AI Segmentation "
                       "panel and click Install."
        }
    else:
        error_msg = {
            "type": "error",
            "message": f"Failed to load PyTorch: {err_str}"
        }
    _emit_error(error_msg)
    sys.exit(1)

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    _USE_SAM2 = True
    _SAM2_IMPORT_ERROR = None
except ImportError as e:




    _USE_SAM2 = False
    _SAM2_IMPORT_ERROR = str(e)


sys.stdout = _real_stdout

SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


def build_sam2_model(checkpoint, device):

    return build_sam2(
        SAM2_MODEL_CFG, checkpoint,
        device=str(device), mode="eval"
    )


def build_sam1_model(checkpoint, device):







    try:
        from segment_anything import sam_model_registry
    except ImportError as e:
        detail = (f"sam2 did not import either ({_SAM2_IMPORT_ERROR}). "
                  if _SAM2_IMPORT_ERROR else "")
        raise RuntimeError(
            f"The on-device AI is not installed in this environment: "
            f"segment_anything is missing ({e}). {detail}"
            f"Reinstall the AI components from the plugin panel."
        ) from e
    model = sam_model_registry["vit_b"](checkpoint=checkpoint)
    model.to(device)
    model.eval()
    return model


def windows_torch_thread_count(logical_cores):








    physical = max(1, torch.get_num_threads())
    return max(1, min(physical, logical_cores - 1, max(3, logical_cores // 2)))


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




    optimal_threads = max(1, min(num_cores - 1, max(2, num_cores // 2)))
    if sys.platform == "win32":
        optimal_threads = windows_torch_thread_count(num_cores)
    torch.set_num_threads(optimal_threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(max(2, optimal_threads // 2))
        except RuntimeError:
            pass
    return torch.device("cpu")









ENCODED_CROP_CACHE_MAX = 4
_encoded_crop_cache = OrderedDict()


def crop_cache_key(image_bytes, shape):

    digest = hashlib.blake2b(image_bytes, digest_size=16).hexdigest()
    return f"{digest}:{'x'.join(str(n) for n in shape)}"


def capture_encoded_crop(predictor):

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






    image_shape = request["image_shape"]
    if not request.get("image"):



        key = request.get("image_key")
        cached = _encoded_crop_cache.get(key) if key else None
        if cached is None:
            return None
    else:
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


        "image_key": key,
    }

    if not _USE_SAM2 and hasattr(predictor, "input_size"):
        response_data["input_size"] = list(predictor.input_size)
    return response_data


def warm_up_decoder_only(predictor):









    try:
        decoder = predictor.model.sam_mask_decoder
        channels = (decoder.conv_s0.out_channels,
                    decoder.conv_s1.out_channels,
                    predictor.model.hidden_dim)
        sizes = predictor._bb_feat_sizes
        feats = [torch.zeros(1, c, h, w) for c, (h, w) in zip(channels, sizes)]
        predictor.reset_predictor()
        predictor._features = {"image_embed": feats[-1],
                               "high_res_feats": feats[:-1]}
        predictor._orig_hw = [(1024, 1024)]
        predictor._is_image_set = True
        with torch.inference_mode():
            predictor.predict(
                point_coords=np.array([[512, 512]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
        return True
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[prediction_worker] decoder warm-up skipped: {e}\n")
        sys.stderr.flush()
        return False
    finally:
        predictor.reset_predictor()


def warm_up_kernels(predictor, device):








    if sys.platform == "win32" and _USE_SAM2 and device.type == "cpu":
        if warm_up_decoder_only(predictor):
            return True
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
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[prediction_worker] warm-up skipped: {e}\n")
        sys.stderr.flush()
        return False


def send_response(response_type, data):
    response = {"type": response_type, **data}
    _real_stdout.write(json.dumps(response) + "\n")
    _real_stdout.flush()


def send_error(error_message):
    send_response("error", {"message": error_message})


def send_ready(pid=None):



    send_response("ready", {} if pid is None else {"pid": pid})


def encode_numpy_array(arr):
    return base64.b64encode(arr.tobytes()).decode("utf-8")


def decode_numpy_array(b64_string, shape, dtype):
    bytes_data = base64.b64decode(b64_string.encode("utf-8"), validate=True)
    arr = np.frombuffer(bytes_data, dtype=dtype)
    return arr.reshape(shape)


MAX_LINE_LENGTH = 50 * 1024 * 1024










STDIN_QUEUE_MAX = 4
_stdin_lines = queue.Queue(maxsize=STDIN_QUEUE_MAX)
_stdin_reader_started = False
_stdin_state = {"at_eof": False}


def _read_bounded_stdin_line():

    line = sys.stdin.readline(MAX_LINE_LENGTH + 1)
    if len(line) > MAX_LINE_LENGTH:
        while line and not line.endswith("\n"):
            line = sys.stdin.readline(64 * 1024)
        raise ValueError(
            f"Input line exceeds maximum length ({MAX_LINE_LENGTH} bytes)")
    return line


def _pump_stdin_lines():





    while True:
        try:
            line = _read_bounded_stdin_line()
        except ValueError as exc:
            _stdin_lines.put(exc)
            continue
        except Exception:  # noqa: BLE001
            line = ""
        _stdin_lines.put(line)
        if not line:
            return


def _start_stdin_reader():


    global _stdin_reader_started
    thread = threading.Thread(
        target=_pump_stdin_lines, name="stdin-reader", daemon=True)
    thread.start()
    _stdin_reader_started = True
    return thread


def _stdin_request_waiting():

    return not _stdin_lines.empty()


def _safe_readline():







    if _stdin_state["at_eof"]:
        return ""
    line = (_stdin_lines.get() if _stdin_reader_started
            else _read_bounded_stdin_line())
    if isinstance(line, ValueError):
        raise line
    if not line:
        _stdin_state["at_eof"] = True
        return ""
    if len(line) > MAX_LINE_LENGTH:
        raise ValueError(
            f"Input line exceeds maximum length ({MAX_LINE_LENGTH} bytes)")
    return line


def exit_when_parent_ends(parent_pid):








    if sys.platform != "win32" or not isinstance(parent_pid, int) or parent_pid <= 0:
        return
    try:
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.WaitForSingleObject.restype = ctypes.c_uint32
        synchronize = 0x00100000
        handle = kernel32.OpenProcess(synchronize, False, parent_pid)
        if not handle:
            return
    except Exception:  # noqa: BLE001
        return

    def _wait():
        infinite = 0xFFFFFFFF
        if kernel32.WaitForSingleObject(handle, infinite) == 0:
            os._exit(0)

    threading.Thread(target=_wait, name="parent-watch", daemon=True).start()


def main():
    try:


        _start_stdin_reader()

        init_request = json.loads(_safe_readline())

        if init_request.get("action") != "init":
            send_error("First request must be 'init'")
            sys.exit(1)
        exit_when_parent_ends(init_request.get("parent_pid"))

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

        send_ready(os.getpid())

        while True:
            try:
                line = _safe_readline()
            except ValueError as e:





                send_error(f"Error processing request: {e}")
                continue
            if not line:
                break

            try:
                request = json.loads(line)
                action = request.get("action")

                if action in ("set_image", "predict"):



                    opt_out_of_windows_power_throttling()

                if action == "set_image":
                    answer = encode_or_reuse_crop(predictor, request)
                    if answer is None:
                        send_response("image_missing", {})
                    else:
                        send_response("image_set", answer)

                elif action == "predict":
                    point_coords = np.array(
                        request["point_coords"]) if request.get("point_coords") else None
                    point_labels = np.array(
                        request["point_labels"]) if request.get("point_labels") else None
                    multimask_output = request.get("multimask_output", False)


                    mask_input = None
                    if request.get("mask_input"):
                        mask_input = decode_numpy_array(
                            request["mask_input"],
                            request["mask_input_shape"],
                            request["mask_input_dtype"]
                        )




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





                    if auto_best and masks.shape[0] > 1:
                        total = masks.shape[1] * masks.shape[2]
                        areas = [int(np.count_nonzero(m)) for m in masks]
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






                    if request.get("masks_packed"):



                        masks_payload = {
                            "masks": encode_numpy_array(
                                np.packbits(np.asarray(masks) > 0)),
                            "masks_shape": list(masks.shape),
                            "masks_dtype": "uint8",
                            "masks_packed": True,
                        }
                    else:
                        masks_payload = {
                            "masks": encode_numpy_array(masks),
                            "masks_shape": list(masks.shape),
                            "masks_dtype": str(masks.dtype),
                        }
                    send_response("prediction", {
                        **masks_payload,
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


                    _encoded_crop_cache.clear()
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


def end_worker_process():







    if sys.platform != "win32":
        return
    try:
        _real_stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
    end_worker_process()
