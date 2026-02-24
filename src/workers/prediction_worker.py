#!/usr/bin/env python3
import sys
import os
import json
import base64

try:
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
except ImportError as e:
    error_msg = {
        "type": "error",
        "message": "Failed to import dependencies: {}. "
                   "Please reinstall dependencies.".format(str(e))
    }
    print(json.dumps(error_msg), flush=True)
    sys.exit(1)
except OSError as e:
    # Catch Windows DLL loading errors (shm.dll, etc.)
    if "shm.dll" in str(e) or "DLL" in str(e).upper():
        error_msg = {
            "type": "error",
            "message": "PyTorch DLL error (Windows): {}. "
                       "This usually means Visual C++ Redistributables are missing. "
                       "Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe".format(str(e))
        }
    else:
        error_msg = {
            "type": "error",
            "message": "Failed to load PyTorch: {}".format(str(e))
        }
    print(json.dumps(error_msg), flush=True)
    sys.exit(1)

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    _USE_SAM2 = True
except ImportError:
    _USE_SAM2 = False

SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


def build_sam2_model(checkpoint, device):
    """Build SAM 2.1 Base Plus model."""
    model = build_sam2(
        SAM2_MODEL_CFG, checkpoint,
        device=str(device), mode="eval"
    )
    return model


def build_sam1_model(checkpoint, device):
    """Build SAM ViT-B model (Python 3.9 fallback)."""
    from segment_anything import sam_model_registry
    model = sam_model_registry["vit_b"](checkpoint=checkpoint)
    model.to(device)
    model.eval()
    return model


def get_optimal_device():
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
            pass

    num_cores = os.cpu_count() or 4
    optimal_threads = max(4, num_cores // 2) if sys.platform == "darwin" else num_cores
    torch.set_num_threads(optimal_threads)
    if hasattr(torch, 'set_num_interop_threads'):
        try:
            torch.set_num_interop_threads(max(2, optimal_threads // 2))
        except RuntimeError:
            pass
    return torch.device("cpu")


def send_response(response_type, data):
    response = {"type": response_type, **data}
    print(json.dumps(response), flush=True)


def send_error(error_message):
    send_response("error", {"message": error_message})


def send_ready():
    send_response("ready", {})


def encode_numpy_array(arr):
    return base64.b64encode(arr.tobytes()).decode('utf-8')


def decode_numpy_array(b64_string, shape, dtype):
    bytes_data = base64.b64decode(b64_string.encode('utf-8'))
    arr = np.frombuffer(bytes_data, dtype=dtype)
    return arr.reshape(shape)


MAX_LINE_LENGTH = 50 * 1024 * 1024  # 50 MB max JSON line


def _safe_readline():
    """Read a line from stdin with size limit to prevent memory exhaustion."""
    line = sys.stdin.readline()
    if len(line) > MAX_LINE_LENGTH:
        raise ValueError(
            "Input line exceeds maximum length ({} bytes)".format(
                MAX_LINE_LENGTH))
    return line


def main():
    try:
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
            send_error("Checkpoint file not found: {}".format(checkpoint_path))
            sys.exit(1)

        device = get_optimal_device()

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
            device_label = "cpu ({}t)".format(torch.get_num_threads())
        sys.stderr.write(
            "[prediction_worker] {}, PyTorch={}, device={}, Python={}.{}.{}\n".format(
                model_label, torch.__version__, device_label,
                sys.version_info.major, sys.version_info.minor,
                sys.version_info.micro
            )
        )
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
                    image_b64 = request["image"]
                    image_shape = request["image_shape"]
                    image_dtype = request["image_dtype"]

                    image_np = decode_numpy_array(
                        image_b64, image_shape, image_dtype)

                    with torch.inference_mode():
                        predictor.set_image(image_np)
                    original_size = image_np.shape[:2]

                    response_data = {
                        "original_size": list(original_size),
                    }
                    # SAM1 predictor exposes input_size via transform
                    if not _USE_SAM2 and hasattr(predictor, 'input_size'):
                        response_data["input_size"] = list(predictor.input_size)

                    send_response("image_set", response_data)

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

                    predict_kwargs = dict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        mask_input=mask_input,
                        multimask_output=effective_multimask,
                    )
                    if _USE_SAM2:
                        predict_kwargs["normalize_coords"] = True

                    with torch.inference_mode():
                        masks, scores, low_res_masks = predictor.predict(
                            **predict_kwargs)

                    # When auto-selecting best mask, pick the highest score
                    if auto_best and masks.shape[0] > 1:
                        best_idx = int(np.argmax(scores))
                        masks = masks[best_idx:best_idx + 1]
                        scores = scores[best_idx:best_idx + 1]
                        low_res_masks = low_res_masks[best_idx:best_idx + 1]

                    # Discard empty masks (all zeros with misleading scores)
                    if masks.shape[0] == 1 and masks[0].sum() == 0:
                        send_error("Segmentation produced an empty mask. "
                                   "Try clicking closer to the target.")
                        continue

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
                    send_error("Unknown action: {}".format(action))

            except Exception as e:
                import traceback
                send_error("Error processing request: {}\n{}".format(
                    str(e), traceback.format_exc()))

    except Exception as e:
        import traceback
        send_error("Worker initialization failed: {}\n{}".format(
            str(e), traceback.format_exc()))
        sys.exit(1)


if __name__ == "__main__":
    main()
