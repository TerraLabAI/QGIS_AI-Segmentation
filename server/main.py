"""FastAPI server for SAM2 inference.

Exposes the same protocol as prediction_worker.py but over HTTP.
Endpoints: /health, /set_image, /predict, /reset
"""
import os
import uuid
import time
import base64
import threading

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional, List

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH", "/app/checkpoints/sam2.1_hiera_base_plus.pt"
)
API_KEY = os.environ.get("API_KEY", "")
SESSION_TTL = int(os.environ.get("SESSION_TTL", "600"))  # 10 min default

app = FastAPI(title="SAM2 Inference API", version="1.0.0")

# Global state
_predictor_lock = threading.Lock()
_sessions = {}  # session_id -> {predictor, last_used, original_size}
_sam_model = None
_device = None


class SetImageRequest(BaseModel):
    image_b64: str
    image_shape: List[int]  # [H, W, 3]
    image_dtype: str = "uint8"


class PredictRequest(BaseModel):
    session_id: str
    point_coords: List[List[float]]
    point_labels: List[int]
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
    global _sam_model, _device
    _device = get_device()
    print("Loading SAM2 model on {}...".format(_device))
    _sam_model = build_sam2(
        SAM2_MODEL_CFG, CHECKPOINT_PATH,
        device=str(_device), mode="eval"
    )
    print("SAM2 model loaded.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "SAM2.1 Base Plus",
        "device": str(_device),
        "active_sessions": len(_sessions),
    }


@app.post("/set_image")
def set_image(req: SetImageRequest, x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)
    cleanup_expired_sessions()

    image_bytes = base64.b64decode(req.image_b64.encode("utf-8"))
    image_np = np.frombuffer(image_bytes, dtype=req.image_dtype).reshape(req.image_shape)

    session_id = str(uuid.uuid4())
    predictor = SAM2ImagePredictor(_sam_model)

    with torch.inference_mode():
        predictor.set_image(image_np)

    original_size = list(image_np.shape[:2])

    _sessions[session_id] = {
        "predictor": predictor,
        "last_used": time.time(),
        "original_size": original_size,
    }

    return {
        "session_id": session_id,
        "original_size": original_size,
    }


@app.post("/predict")
def predict(req: PredictRequest, x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)

    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    session["last_used"] = time.time()
    predictor = session["predictor"]

    point_coords = np.array(req.point_coords)
    point_labels = np.array(req.point_labels)

    mask_input = None
    if req.mask_input:
        mask_bytes = base64.b64decode(req.mask_input.encode("utf-8"))
        mask_input = np.frombuffer(
            mask_bytes, dtype=req.mask_input_dtype
        ).reshape(req.mask_input_shape)

    # Auto-select best mask on first click (same logic as prediction_worker)
    auto_best = (not req.multimask_output and mask_input is None)
    effective_multimask = True if auto_best else req.multimask_output

    with torch.inference_mode():
        masks, scores, low_res_masks = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            multimask_output=effective_multimask,
            normalize_coords=True,
        )

    if auto_best and masks.shape[0] > 1:
        best_idx = int(np.argmax(scores))
        masks = masks[best_idx:best_idx + 1]
        scores = scores[best_idx:best_idx + 1]
        low_res_masks = low_res_masks[best_idx:best_idx + 1]

    if masks.shape[0] == 1 and masks[0].sum() == 0:
        raise HTTPException(
            status_code=422,
            detail="Segmentation produced an empty mask. Try clicking closer to the target."
        )

    return {
        "masks": base64.b64encode(masks.tobytes()).decode("utf-8"),
        "masks_shape": list(masks.shape),
        "masks_dtype": str(masks.dtype),
        "scores": scores.tolist(),
        "low_res_masks": base64.b64encode(low_res_masks.tobytes()).decode("utf-8"),
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
