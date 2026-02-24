#!/usr/bin/env python3
import sys
import os
import gc
import json
import base64

# Ensure consistent GPU ordering on multi-GPU systems
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

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
    try:
        if torch.cuda.is_available():
            best_idx = -1
            best_mem = 0
            count = torch.cuda.device_count()
            for i in range(count):
                try:
                    mem = torch.cuda.get_device_properties(i).total_memory
                    if mem >= 2 * 1024 ** 3 and mem > best_mem:
                        best_mem = mem
                        best_idx = i
                except Exception:
                    continue
            if best_idx < 0:
                return torch.device("cpu")
            # Verify CUDA kernels actually work
            cuda_dev = "cuda:{}".format(best_idx)
            t = torch.zeros(1, device=cuda_dev)
            _ = t + 1
            torch.cuda.synchronize(best_idx)
            del t
            torch.cuda.empty_cache()
            return torch.device(cuda_dev)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Prevent MPS OOM by disabling memory pool upper limit
            os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
            # Verify MPS actually works
            try:
                t = torch.zeros(1, device="mps")
                _ = t + 1
                torch.mps.synchronize()
                del t
            except Exception:
                return torch.device("cpu")
            return torch.device("mps")
        else:
            return torch.device("cpu")
    except Exception:
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


def main():
    try:
        init_request = json.loads(sys.stdin.readline())

        if init_request.get("action") != "init":
            send_error("First request must be 'init'")
            sys.exit(1)

        checkpoint_path = init_request.get("checkpoint_path")

        device = get_optimal_device()

        # Free memory before loading model
        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        elif device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
            gc.collect()
            torch.mps.empty_cache()

        if _USE_SAM2:
            sam_model = build_sam2_model(checkpoint_path, device)
            predictor = SAM2ImagePredictor(sam_model)
            model_label = "SAM2.1"
        else:
            sam_model = build_sam1_model(checkpoint_path, device)
            from segment_anything import SamPredictor as Sam1Predictor
            predictor = Sam1Predictor(sam_model)
            model_label = "SAM1-ViT-B"

        # Log environment info for diagnostics (no personal paths)
        device_name = str(device)
        if device.type == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(device)
                gpu_mem = torch.cuda.get_device_properties(
                    device).total_memory / (1024**3)
                device_name = "cuda ({}, {:.1f}GB)".format(gpu_name, gpu_mem)
            except Exception:
                pass
        sys.stderr.write(
            "[prediction_worker] {}, PyTorch={}, device={}, Python={}.{}.{}\n".format(
                model_label, torch.__version__, device_name,
                sys.version_info.major, sys.version_info.minor,
                sys.version_info.micro
            )
        )
        sys.stderr.flush()

        send_ready()

        while True:
            line = sys.stdin.readline()
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

                    try:
                        with torch.inference_mode():
                            predictor.set_image(image_np)
                        original_size = image_np.shape[:2]
                    except RuntimeError:
                        if device.type != "cpu":
                            # GPU error - fall back to CPU and retry
                            try:
                                if device.type == "cuda":
                                    torch.cuda.empty_cache()
                            except Exception:
                                pass
                            predictor.model.to(torch.device("cpu"))
                            try:
                                with torch.inference_mode():
                                    predictor.set_image(image_np)
                                original_size = image_np.shape[:2]
                            except Exception as cpu_err:
                                send_error(
                                    "CPU retry also failed: {}".format(
                                        cpu_err))
                                continue
                        else:
                            raise

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

                    predict_kwargs = dict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        mask_input=mask_input,
                        multimask_output=multimask_output,
                    )
                    if _USE_SAM2:
                        predict_kwargs["normalize_coords"] = True

                    try:
                        with torch.inference_mode():
                            masks, scores, low_res_masks = predictor.predict(
                                **predict_kwargs)
                    except RuntimeError:
                        if device.type != "cpu":
                            try:
                                if device.type == "cuda":
                                    torch.cuda.empty_cache()
                            except Exception:
                                pass
                            predictor.model.to(torch.device("cpu"))
                            try:
                                with torch.inference_mode():
                                    masks, scores, low_res_masks = predictor.predict(
                                        **predict_kwargs)
                            except Exception as cpu_err:
                                send_error(
                                    "CPU retry also failed: {}".format(cpu_err))
                                continue
                        else:
                            raise

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
