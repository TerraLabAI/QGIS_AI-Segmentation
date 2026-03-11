"""FastAPI server for SAM3 inference.

Exposes the same protocol as the SAM2 server but with SAM3 model.
Endpoints: /health, /set_image, /predict, /reset

Point-based prediction uses SAM3InteractiveImagePredictor (SAM1-style API).
Text-based prediction uses Sam3Processor.set_text_prompt().
"""
import os
import uuid
import time
import base64
import threading

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional, List

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3.model.sam3_image_processor import Sam3Processor

CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH", "/app/checkpoints/sam3.pt"
)
API_KEY = os.environ.get("API_KEY", "")
SESSION_TTL = int(os.environ.get("SESSION_TTL", "600"))

app = FastAPI(title="SAM3 Inference API", version="1.0.0")

_predictor_lock = threading.Lock()
_sessions = {}
_sam_model = None
_processor = None
_device = None


class SetImageRequest(BaseModel):
    image_b64: str
    image_shape: List[int]
    image_dtype: str = "uint8"


class PredictRequest(BaseModel):
    session_id: str
    point_coords: List[List[float]] = []
    point_labels: List[int] = []
    text_prompt: Optional[str] = None
    multimask_output: bool = False
    mask_input: Optional[str] = None
    mask_input_shape: Optional[List[int]] = None
    mask_input_dtype: Optional[str] = None


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    num_cores = os.cpu_count() or 4
    torch.set_num_threads(num_cores)
    return torch.device("cpu")


def check_api_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def cleanup_expired_sessions():
    now = time.time()
    expired = [
        sid for sid, s in _sessions.items()
        if now - s["last_used"] > SESSION_TTL
    ]
    for sid in expired:
        session = _sessions.pop(sid, None)
        if session and session.get("predictor"):
            session["predictor"].reset_predictor()


@app.on_event("startup")
def startup():
    global _sam_model, _processor, _device
    _device = get_device()
    print("Loading SAM3 model on {}...".format(_device))
    _sam_model = build_sam3_image_model(
        checkpoint_path=CHECKPOINT_PATH,
        device=str(_device),
        load_from_HF=False,
        enable_inst_interactivity=True,
    )
    # Sam3Image lacks image_size attr that SAM3InteractiveImagePredictor needs
    if not hasattr(_sam_model, "image_size"):
        _sam_model.image_size = 1008
    _processor = Sam3Processor(_sam_model, device=str(_device))
    print("SAM3 model loaded.")

    # Self-test: verify predictor works with a small dummy image
    print("Running startup self-test...")
    try:
        test_predictor = SAM3InteractiveImagePredictor(_sam_model)
        test_img = np.zeros((64, 64, 3), dtype=np.uint8)
        with torch.inference_mode():
            test_predictor.set_image(test_img)
        test_predictor.reset_predictor()
        print("Self-test passed: predictor works.")
    except Exception as e:
        print("WARNING: Self-test failed: {}".format(e))
        import traceback
        traceback.print_exc()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "SAM 3",
        "device": str(_device),
        "active_sessions": len(_sessions),
    }


@app.post("/set_image")
def set_image(req: SetImageRequest, x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)
    cleanup_expired_sessions()

    try:
        image_bytes = base64.b64decode(req.image_b64.encode("utf-8"))
        image_np = np.frombuffer(
            image_bytes, dtype=req.image_dtype
        ).reshape(req.image_shape)

        session_id = str(uuid.uuid4())
        predictor = SAM3InteractiveImagePredictor(_sam_model)

        with torch.inference_mode():
            predictor.set_image(image_np)

        text_state = None
        with torch.inference_mode():
            pil_image = Image.fromarray(image_np)
            text_state = _processor.set_image(pil_image)

        original_size = list(image_np.shape[:2])

        _sessions[session_id] = {
            "predictor": predictor,
            "text_state": text_state,
            "last_used": time.time(),
            "original_size": original_size,
        }

        return {
            "session_id": session_id,
            "original_size": original_size,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="set_image failed: {}".format(str(e))
        )


@app.post("/predict")
def predict(req: PredictRequest, x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)

    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(
            status_code=404, detail="Session not found or expired"
        )

    session["last_used"] = time.time()

    has_points = len(req.point_coords) > 0
    has_text = req.text_prompt is not None and req.text_prompt.strip() != ""

    if not has_points and not has_text:
        raise HTTPException(
            status_code=422,
            detail="At least one of point_coords or text_prompt is required"
        )

    try:
        if has_text and not has_points:
            return _predict_text(req, session)
        return _predict_points(req, session, has_points)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="predict failed: {}".format(str(e))
        )


def _predict_text(req, session):
    """Text-only prediction via Sam3Processor."""
    text_state = session["text_state"]
    original_size = session["original_size"]

    with torch.inference_mode():
        output = _processor.set_text_prompt(
            prompt=req.text_prompt.strip(),
            state=text_state,
        )

    out_masks = output.get("masks")
    out_scores = output.get("scores")

    if out_masks is None or len(out_masks) == 0:
        raise HTTPException(
            status_code=422,
            detail="Segmentation produced no results. "
                   "Try a different text prompt."
        )

    if isinstance(out_masks, torch.Tensor):
        out_masks = out_masks.cpu().numpy()
    if isinstance(out_scores, torch.Tensor):
        out_scores = out_scores.cpu().numpy()

    out_masks = np.asarray(out_masks, dtype=bool)
    out_scores = np.asarray(out_scores, dtype=np.float32)

    if out_masks.ndim == 2:
        out_masks = out_masks[np.newaxis]
    if out_masks.ndim == 4:
        out_masks = out_masks.squeeze(1)

    h, w = original_size
    if out_masks.shape[-2:] != (h, w):
        from PIL import Image as PILImage
        resized = []
        for m in out_masks:
            img = PILImage.fromarray(m.astype(np.uint8) * 255)
            img = img.resize((w, h), PILImage.NEAREST)
            resized.append(np.array(img) > 127)
        out_masks = np.stack(resized)

    if not req.multimask_output and out_masks.shape[0] > 1:
        best_idx = int(np.argmax(out_scores))
        out_masks = out_masks[best_idx:best_idx + 1]
        out_scores = out_scores[best_idx:best_idx + 1]

    if out_masks.shape[0] == 1 and out_masks[0].sum() == 0:
        raise HTTPException(
            status_code=422,
            detail="Segmentation produced an empty mask. "
                   "Try a different text prompt."
        )

    n = out_masks.shape[0]
    low_res_masks = np.zeros((n, 256, 256), dtype=np.float32)

    return {
        "masks": base64.b64encode(out_masks.tobytes()).decode("utf-8"),
        "masks_shape": list(out_masks.shape),
        "masks_dtype": str(out_masks.dtype),
        "scores": out_scores.tolist(),
        "low_res_masks": base64.b64encode(
            low_res_masks.tobytes()
        ).decode("utf-8"),
        "low_res_masks_shape": list(low_res_masks.shape),
        "low_res_masks_dtype": str(low_res_masks.dtype),
    }


def _predict_points(req, session, has_points):
    """Point-based prediction via SAM3InteractiveImagePredictor."""
    predictor = session["predictor"]

    point_coords = np.array(req.point_coords) if has_points else None
    point_labels = np.array(req.point_labels) if has_points else None

    mask_input = None
    if req.mask_input:
        mask_bytes = base64.b64decode(req.mask_input.encode("utf-8"))
        mask_input = np.frombuffer(
            mask_bytes, dtype=req.mask_input_dtype
        ).reshape(req.mask_input_shape)

    auto_best = (not req.multimask_output and mask_input is None)
    effective_multimask = True if auto_best else req.multimask_output

    predict_kwargs = {
        "multimask_output": effective_multimask,
    }
    if point_coords is not None:
        predict_kwargs["point_coords"] = point_coords
        predict_kwargs["point_labels"] = point_labels
        predict_kwargs["normalize_coords"] = True
    if mask_input is not None:
        predict_kwargs["mask_input"] = mask_input

    with torch.inference_mode():
        masks, scores, low_res_masks = predictor.predict(**predict_kwargs)

    if auto_best and masks.shape[0] > 1:
        best_idx = int(np.argmax(scores))
        masks = masks[best_idx:best_idx + 1]
        scores = scores[best_idx:best_idx + 1]
        low_res_masks = low_res_masks[best_idx:best_idx + 1]

    if masks.shape[0] == 1 and masks[0].sum() == 0:
        raise HTTPException(
            status_code=422,
            detail="Segmentation produced an empty mask. "
                   "Try a different prompt or click closer to the target."
        )

    return {
        "masks": base64.b64encode(masks.tobytes()).decode("utf-8"),
        "masks_shape": list(masks.shape),
        "masks_dtype": str(masks.dtype),
        "scores": scores.tolist(),
        "low_res_masks": base64.b64encode(
            low_res_masks.tobytes()
        ).decode("utf-8"),
        "low_res_masks_shape": list(low_res_masks.shape),
        "low_res_masks_dtype": str(low_res_masks.dtype),
    }


@app.post("/reset")
def reset(session_id: str, x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)

    session = _sessions.pop(session_id, None)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session["predictor"].reset_predictor()
    return {"status": "reset_done"}
