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
    from typing import Tuple, Optional  # noqa: E402
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


def build_sam_vit_b(checkpoint: Optional[str] = None):
    """Build full SAM ViT-B model with real image encoder."""
    from segment_anything import sam_model_registry

    sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
    sam.eval()
    return sam


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


class SamPredictor:
    def __init__(self, sam_model, device: Optional[torch.device] = None) -> None:
        self.model = sam_model
        self.device = device if device is not None else get_optimal_device()
        # Free memory before loading model onto GPU
        if self.device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        elif self.device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
            gc.collect()
            torch.mps.empty_cache()
        self.model.to(self.device)

        # Verify GPU kernels are available
        if self.device.type == "cuda":
            try:
                test = torch.zeros(1, device=self.device)
                _ = test + 1
                torch.cuda.synchronize()
                del test
            except RuntimeError:
                self.device = torch.device("cpu")
                self.model.to(self.device)
        elif self.device.type == "mps":
            try:
                test = torch.zeros(1, device=self.device)
                _ = test + 1
                torch.mps.synchronize()
                del test
            except RuntimeError:
                self.device = torch.device("cpu")
                self.model.to(self.device)

        self.reset_image()

    def reset_image(self) -> None:
        self.features = None
        self.original_size = None
        self.input_size = None
        self.is_image_set = False

    @property
    def transform(self):
        from segment_anything.utils.transforms import ResizeLongestSide
        return ResizeLongestSide(self.model.image_encoder.img_size)

    def set_image_feature(
        self,
        img_features: np.ndarray,
        img_size: Tuple[int, int],
        input_size: Optional[Tuple[int, int]] = None
    ) -> None:
        self.features = torch.as_tensor(img_features, dtype=torch.float32, device=self.device)
        self.original_size = img_size
        self.input_size = input_size if input_size else img_size
        self.is_image_set = True

    @torch.no_grad()
    def set_image(self, image_np: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Run full SAM encoding on an image crop (H, W, 3 uint8).

        Returns (original_size, input_size) tuple.
        """
        from segment_anything.utils.transforms import ResizeLongestSide

        original_h, original_w = image_np.shape[:2]
        transform_obj = ResizeLongestSide(self.model.image_encoder.img_size)
        resized = transform_obj.apply_image(image_np)
        input_h, input_w = resized.shape[:2]

        # Convert HWC -> CHW, add batch dim, preprocess
        tensor = torch.as_tensor(
            resized.transpose(2, 0, 1), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        tensor = self.model.preprocess(tensor)

        features = self.model.image_encoder(tensor)
        self.features = features
        self.original_size = (original_h, original_w)
        self.input_size = (input_h, input_w)
        self.is_image_set = True

        # Free intermediate tensors
        del tensor
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

        return self.original_size, self.input_size

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

        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None

        if point_coords is not None:
            point_coords = self.transform.apply_coords(
                point_coords, self.original_size
            )
            coords_torch = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            labels_torch = torch.as_tensor(
                point_labels, dtype=torch.int, device=self.device
            )
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]

        if mask_input is not None:
            mask_input_torch = torch.as_tensor(
                mask_input, dtype=torch.float, device=self.device
            )
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        masks_np = masks[0].detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()

        return masks_np, iou_predictions_np, low_res_masks_np

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_image_set:
            raise RuntimeError("Features have not been set.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        masks = self.model.postprocess_masks(
            low_res_masks,
            input_size=self.input_size,
            original_size=self.original_size,
        )

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks


def send_response(response_type: str, data: dict):
    response = {"type": response_type, **data}
    print(json.dumps(response), flush=True)


def send_error(error_message: str):
    send_response("error", {"message": error_message})


def send_ready():
    send_response("ready", {})


def encode_numpy_array(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode('utf-8')


def decode_numpy_array(b64_string: str, shape: list, dtype: str) -> np.ndarray:
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

        sam = build_sam_vit_b(checkpoint=checkpoint_path)
        predictor = SamPredictor(sam)

        # Log environment info for diagnostics (no personal paths)
        device_name = str(predictor.device)
        if predictor.device.type == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(predictor.device)
                gpu_mem = torch.cuda.get_device_properties(
                    predictor.device).total_memory / (1024**3)
                device_name = "cuda ({}, {:.1f}GB)".format(gpu_name, gpu_mem)
            except Exception:
                pass
        sys.stderr.write(
            "[prediction_worker] PyTorch={}, device={}, Python={}.{}.{}\n".format(
                torch.__version__, device_name,
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
                        orig_size, inp_size = predictor.set_image(image_np)
                    except RuntimeError:
                        if predictor.device.type != "cpu":
                            # GPU error - fall back to CPU and retry
                            try:
                                if predictor.device.type == "cuda":
                                    torch.cuda.empty_cache()
                            except Exception:
                                pass
                            predictor.device = torch.device("cpu")
                            predictor.model.to(predictor.device)
                            try:
                                orig_size, inp_size = predictor.set_image(
                                    image_np)
                            except Exception as cpu_err:
                                send_error(
                                    "CPU retry also failed: {}".format(
                                        cpu_err))
                                continue
                        else:
                            raise

                    send_response("image_set", {
                        "original_size": list(orig_size),
                        "input_size": list(inp_size),
                    })

                elif action == "set_features":
                    features_b64 = request["features"]
                    features_shape = request["features_shape"]
                    features_dtype = request["features_dtype"]
                    img_size = tuple(request["img_size"])
                    input_size = tuple(request["input_size"]) if request.get("input_size") else None

                    features_np = decode_numpy_array(features_b64, features_shape, features_dtype)
                    predictor.set_image_feature(features_np, img_size, input_size)

                    send_response("features_set", {})

                elif action == "predict":
                    point_coords = np.array(request["point_coords"]) if request.get("point_coords") else None
                    point_labels = np.array(request["point_labels"]) if request.get("point_labels") else None
                    multimask_output = request.get("multimask_output", False)

                    # Decode mask_input if provided (for iterative refinement)
                    mask_input = None
                    if request.get("mask_input"):
                        mask_input = decode_numpy_array(
                            request["mask_input"],
                            request["mask_input_shape"],
                            request["mask_input_dtype"]
                        )

                    try:
                        masks, scores, low_res_masks = predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            mask_input=mask_input,
                            multimask_output=multimask_output,
                        )
                    except RuntimeError:
                        if predictor.device.type != "cpu":
                            # GPU error (OOM, illegal access, no kernel image, etc.)
                            # Fall back to CPU and retry
                            try:
                                if predictor.device.type == "cuda":
                                    torch.cuda.empty_cache()
                            except Exception:
                                pass
                            predictor.device = torch.device("cpu")
                            predictor.model.to(predictor.device)
                            if predictor.features is not None:
                                predictor.features = predictor.features.to(predictor.device)
                            try:
                                masks, scores, low_res_masks = predictor.predict(
                                    point_coords=point_coords,
                                    point_labels=point_labels,
                                    mask_input=mask_input,
                                    multimask_output=multimask_output,
                                )
                            except Exception as cpu_err:
                                send_error("CPU retry also failed: {}".format(cpu_err))
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
