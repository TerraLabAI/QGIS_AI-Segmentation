#!/usr/bin/env python3
import sys
import os
import gc
import json
import base64

# Ensure consistent GPU ordering on multi-GPU systems
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from typing import Tuple, Optional  # noqa: E402


class FakeImageEncoderViT(nn.Module):
    def __init__(self, img_size: int = 1024) -> None:
        super().__init__()
        self.img_size = img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def build_sam_no_encoder(checkpoint: Optional[str] = None, model_id: str = "sam_vit_b"):
    """Build a SAM1 model with a fake encoder (decoder-only for prediction).

    Works for both ViT-B and ViT-L since they share the same decoder architecture
    (prompt_embed_dim=256, vit_patch_size=16, image_embedding_size=64).
    """
    from segment_anything.modeling import MaskDecoder, PromptEncoder, Sam, TwoWayTransformer

    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    sam = Sam(
        image_encoder=FakeImageEncoderViT(img_size=image_size),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    sam.eval()

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device('cpu'), weights_only=True)
        sam.load_state_dict(state_dict, strict=False)

    return sam


# Keep old name as alias for backward compatibility
build_sam_vit_b_no_encoder = build_sam_no_encoder


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
            return torch.device("mps")
        else:
            return torch.device("cpu")
    except Exception:
        return torch.device("cpu")


class SamPredictorNoImgEncoder:
    def __init__(self, sam_model, device: Optional[torch.device] = None) -> None:
        self.model = sam_model
        self.device = device if device is not None else get_optimal_device()
        # Free memory before loading model onto GPU
        if self.device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        self.model.to(self.device)

        # Verify CUDA kernels are available for this GPU
        if self.device.type == "cuda":
            try:
                test = torch.zeros(1, device=self.device)
                _ = test + 1  # Force a kernel execution
                torch.cuda.synchronize()
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


def _build_sam2_predictor(init_request):
    """Build a SAM2 predictor for prediction."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint_path = init_request.get("checkpoint_path")
    sam2_model_cfg = init_request.get(
        "sam2_model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml")

    device = get_optimal_device()
    model = build_sam2(sam2_model_cfg, checkpoint_path, device=str(device))
    return SAM2ImagePredictor(model), device


def _set_sam2_features(
    sam2_predictor, features_np, img_size, device,
    high_res_feats_np=None
):
    """Manually set pre-encoded features on a SAM2ImagePredictor.

    Sets the internal _features dict with image_embed and high_res_feats
    from the cached encoding.
    """
    features_torch = torch.as_tensor(
        features_np, dtype=torch.float32, device=device)

    if high_res_feats_np is not None:
        hr_tensors = [
            torch.as_tensor(
                hr.astype(np.float32),
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)
            for hr in high_res_feats_np
        ]
    else:
        hr_tensors = []

    sam2_predictor._features = {
        "image_embed": features_torch.unsqueeze(0),
        "high_res_feats": hr_tensors,
    }
    sam2_predictor._is_image_set = True
    sam2_predictor._orig_hw = [img_size]


def main():
    try:
        init_request = json.loads(sys.stdin.readline())

        if init_request.get("action") != "init":
            send_error("First request must be 'init'")
            sys.exit(1)

        checkpoint_path = init_request.get("checkpoint_path")
        model_id = init_request.get("model_id", "sam_vit_b")
        model_family = init_request.get("model_family", "sam1")

        if model_family == "sam2":
            sam2_predictor, sam2_device = _build_sam2_predictor(init_request)
            predictor = None
        else:
            sam = build_sam_no_encoder(
                checkpoint=checkpoint_path, model_id=model_id)
            predictor = SamPredictorNoImgEncoder(sam)
            sam2_predictor = None
            sam2_device = None

        send_ready()

        # State for SAM2 CPU fallback (store last features to re-set after rebuild)
        sam2_last_features_np = None
        sam2_last_img_size = None
        sam2_last_hr_feats_np = None

        while True:
            line = sys.stdin.readline()
            if not line:
                break

            try:
                request = json.loads(line)
                action = request.get("action")

                if action == "set_features":
                    features_b64 = request["features"]
                    features_shape = request["features_shape"]
                    features_dtype = request["features_dtype"]
                    img_size = tuple(request["img_size"])

                    features_np = decode_numpy_array(
                        features_b64, features_shape, features_dtype)

                    if model_family == "sam2":
                        # SAM2: manually set internal predictor state
                        sam2_last_features_np = features_np.copy()
                        sam2_last_img_size = img_size

                        # Deserialize high_res_feats if provided
                        hr_feats_np = None
                        if request.get("high_res_feats"):
                            hr_feats_np = []
                            for hr_item in request["high_res_feats"]:
                                hr_arr = decode_numpy_array(
                                    hr_item["data"],
                                    hr_item["shape"],
                                    hr_item["dtype"])
                                hr_feats_np.append(hr_arr)
                        sam2_last_hr_feats_np = hr_feats_np

                        _set_sam2_features(
                            sam2_predictor, features_np,
                            img_size, sam2_device,
                            high_res_feats_np=hr_feats_np)
                    else:
                        # SAM1: use dedicated predictor method
                        input_size = (
                            tuple(request["input_size"])
                            if request.get("input_size") else None
                        )
                        predictor.set_image_feature(
                            features_np, img_size, input_size)

                    send_response("features_set", {})

                elif action == "predict":
                    point_coords = (
                        np.array(request["point_coords"])
                        if request.get("point_coords") else None
                    )
                    point_labels = (
                        np.array(request["point_labels"])
                        if request.get("point_labels") else None
                    )
                    multimask_output = request.get("multimask_output", False)

                    # Decode mask_input if provided (for iterative refinement)
                    mask_input = None
                    if request.get("mask_input"):
                        mask_input = decode_numpy_array(
                            request["mask_input"],
                            request["mask_input_shape"],
                            request["mask_input_dtype"]
                        )

                    if model_family == "sam2":
                        try:
                            with torch.no_grad():
                                masks, scores, logits = sam2_predictor.predict(
                                    point_coords=point_coords,
                                    point_labels=point_labels,
                                    mask_input=mask_input,
                                    multimask_output=multimask_output,
                                )
                        except RuntimeError:
                            if sam2_device.type != "cpu":
                                try:
                                    if sam2_device.type == "cuda":
                                        torch.cuda.empty_cache()
                                except Exception:
                                    pass
                                sam2_device = torch.device("cpu")
                                sam2_predictor, sam2_device = (
                                    _build_sam2_predictor(init_request))
                                # Re-set features on rebuilt predictor
                                if sam2_last_features_np is not None:
                                    _set_sam2_features(
                                        sam2_predictor,
                                        sam2_last_features_np,
                                        sam2_last_img_size,
                                        sam2_device,
                                        high_res_feats_np=sam2_last_hr_feats_np)
                                try:
                                    with torch.no_grad():
                                        masks, scores, logits = (
                                            sam2_predictor.predict(
                                                point_coords=point_coords,
                                                point_labels=point_labels,
                                                mask_input=mask_input,
                                                multimask_output=multimask_output,
                                            ))
                                except Exception as cpu_err:
                                    send_error(
                                        "CPU retry also failed: {}".format(
                                            cpu_err))
                                    continue
                            else:
                                raise

                        send_response("prediction", {
                            "masks": encode_numpy_array(masks),
                            "masks_shape": list(masks.shape),
                            "masks_dtype": str(masks.dtype),
                            "scores": scores.tolist(),
                            "low_res_masks": encode_numpy_array(logits),
                            "low_res_masks_shape": list(logits.shape),
                            "low_res_masks_dtype": str(logits.dtype),
                        })

                    else:
                        # SAM1 prediction path
                        try:
                            masks, scores, low_res_masks = predictor.predict(
                                point_coords=point_coords,
                                point_labels=point_labels,
                                mask_input=mask_input,
                                multimask_output=multimask_output,
                            )
                        except RuntimeError:
                            if predictor.device.type != "cpu":
                                try:
                                    if predictor.device.type == "cuda":
                                        torch.cuda.empty_cache()
                                except Exception:
                                    pass
                                predictor.device = torch.device("cpu")
                                predictor.model.to(predictor.device)
                                if predictor.features is not None:
                                    predictor.features = (
                                        predictor.features.to(
                                            predictor.device))
                                try:
                                    masks, scores, low_res_masks = (
                                        predictor.predict(
                                            point_coords=point_coords,
                                            point_labels=point_labels,
                                            mask_input=mask_input,
                                            multimask_output=multimask_output,
                                        ))
                                except Exception as cpu_err:
                                    send_error(
                                        "CPU retry also failed: {}".format(
                                            cpu_err))
                                    continue
                            else:
                                raise

                        send_response("prediction", {
                            "masks": encode_numpy_array(masks),
                            "masks_shape": list(masks.shape),
                            "masks_dtype": str(masks.dtype),
                            "scores": scores.tolist(),
                            "low_res_masks": encode_numpy_array(
                                low_res_masks),
                            "low_res_masks_shape": list(
                                low_res_masks.shape),
                            "low_res_masks_dtype": str(
                                low_res_masks.dtype),
                        })

                elif action == "reset":
                    if model_family == "sam2":
                        sam2_predictor.reset_predictor()
                        sam2_last_features_np = None
                        sam2_last_img_size = None
                        sam2_last_hr_feats_np = None
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
        send_error(f"Worker initialization failed: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
