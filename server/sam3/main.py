"""FastAPI server for SAM3 inference.

Exposes the same protocol as the SAM2 server but with SAM3 model.
Endpoints: /health, /set_image, /predict, /reset

Point-based prediction uses SAM3InteractiveImagePredictor wrapping the
internal Sam3TrackerPredictor (extracted from Sam3Image.inst_interactive_predictor).
Text-based prediction uses Sam3Processor.set_text_prompt().
"""
import os
import uuid
import time
import base64
import zlib
import threading

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, Header
from starlette.middleware.gzip import GZipMiddleware
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
app.add_middleware(GZipMiddleware, minimum_size=1000)

_predictor_lock = threading.Lock()
_sessions = {}
_sam_model = None
_tracker_model = None
_processor = None
_device = None


class SetImageRequest(BaseModel):
    image_b64: str
    image_shape: List[int]
    image_dtype: str = "uint8"
    image_format: Optional[str] = None  # "jpeg" or None (raw numpy)


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
        if session:
            predictor = session.get("predictor")
            if predictor:
                try:
                    predictor.reset_predictor()
                except Exception:
                    pass


@app.on_event("startup")
def startup():
    global _sam_model, _tracker_model, _processor, _device
    _device = get_device()
    print("Loading SAM3 model on {}...".format(_device))
    _sam_model = build_sam3_image_model(
        checkpoint_path=CHECKPOINT_PATH,
        device=str(_device),
        load_from_HF=False,
        enable_inst_interactivity=True,
    )
    _processor = Sam3Processor(_sam_model, device=str(_device))

    # Extract the internal Sam3TrackerPredictor from the Sam3Image model.
    # Sam3Image itself does NOT have forward_image(); only the tracker does.
    inst_pred = getattr(_sam_model, "inst_interactive_predictor", None)
    if inst_pred is None:
        raise RuntimeError(
            "Sam3Image has no inst_interactive_predictor. "
            "Was enable_inst_interactivity=True passed to build?"
        )
    _tracker_model = inst_pred.model
    # The tracker's backbone is None after build_tracker(); share it from
    # the parent Sam3Image so forward_image() works.
    if getattr(_tracker_model, "backbone", None) is None:
        _tracker_model.backbone = _sam_model.backbone
        print("Shared backbone from Sam3Image to tracker.")

    # Performance: enable cudnn autotuner
    # fp16 is handled by torch.amp.autocast at inference time (not .half())
    # to avoid dtype mismatches between model components (tracker vs text encoder)
    torch.backends.cudnn.benchmark = True

    # torch.compile for point prediction (fixed 1024x1024 input)
    if hasattr(torch, "compile") and _device.type == "cuda":
        try:
            _tracker_model = torch.compile(
                _tracker_model, mode="default"
            )
            print("Tracker model compiled with torch.compile.")
        except Exception as e:
            print("torch.compile failed, using eager mode: {}".format(e))

    print("SAM3 model loaded (tracker: {}).".format(type(_tracker_model).__name__))

    # Full warmup: run complete predict cycles so torch.compile traces
    # all graphs before the first real request (avoids 10-30s penalty).
    print("Running startup warmup (compiling graphs)...")
    try:
        test_predictor = SAM3InteractiveImagePredictor(_tracker_model)
        test_img = np.zeros((1024, 1024, 3), dtype=np.uint8)
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                test_predictor.set_image(test_img)
                test_predictor.predict(
                    point_coords=np.array([[512.0, 512.0]]),
                    point_labels=np.array([1]),
                    multimask_output=True,
                    normalize_coords=True,
                )
        test_predictor.reset_predictor()
        print("Warmup done: point prediction graph compiled.")
    except Exception as e:
        print("WARNING: Warmup failed: {}".format(e))
        import traceback
        traceback.print_exc()

    # Text processor warmup
    try:
        test_pil = Image.fromarray(
            np.zeros((1024, 1024, 3), dtype=np.uint8)
        )
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                state = _processor.set_image(test_pil)
                _processor.set_text_prompt(prompt="test", state=state)
        print("Warmup done: text prediction graph compiled.")
    except Exception as e:
        print("WARNING: Text warmup failed: {}".format(e))
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
        if req.image_format == "jpeg":
            import io
            pil_image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(pil_image)
        else:
            image_np = np.frombuffer(
                image_bytes, dtype=req.image_dtype
            ).reshape(req.image_shape)
            pil_image = Image.fromarray(image_np)

        session_id = str(uuid.uuid4())
        predictor = SAM3InteractiveImagePredictor(_tracker_model)

        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                predictor.set_image(image_np)

        # Defer text processor encoding to first text prediction
        original_size = list(image_np.shape[:2])

        _sessions[session_id] = {
            "predictor": predictor,
            "text_state": None,
            "pil_image": pil_image,
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
    original_size = session["original_size"]

    # Lazy init: encode image for text processor on first text prediction
    if session["text_state"] is None:
        pil_image = session.get("pil_image")
        if pil_image is None:
            raise HTTPException(
                status_code=500,
                detail="No image available for text processor init"
            )
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                session["text_state"] = _processor.set_image(pil_image)
        session["pil_image"] = None  # free memory

    text_state = session["text_state"]

    with torch.inference_mode():
        with torch.amp.autocast("cuda", dtype=torch.float16):
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
        import torch.nn.functional as F
        masks_tensor = torch.from_numpy(
            out_masks.astype(np.float32)
        ).unsqueeze(1).to(_device)
        masks_resized = F.interpolate(
            masks_tensor, size=(h, w), mode="nearest"
        )
        out_masks = (
            masks_resized.squeeze(1).cpu().numpy() > 0.5
        ).astype(bool)

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

    masks_compressed = zlib.compress(out_masks.tobytes(), level=1)
    lr_compressed = zlib.compress(low_res_masks.tobytes(), level=1)

    return {
        "masks": base64.b64encode(masks_compressed).decode("utf-8"),
        "masks_shape": list(out_masks.shape),
        "masks_dtype": str(out_masks.dtype),
        "masks_compressed": True,
        "scores": out_scores.tolist(),
        "low_res_masks": base64.b64encode(
            lr_compressed
        ).decode("utf-8"),
        "low_res_masks_shape": list(low_res_masks.shape),
        "low_res_masks_dtype": str(low_res_masks.dtype),
        "low_res_masks_compressed": True,
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
        with torch.amp.autocast("cuda", dtype=torch.float16):
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

    masks_compressed = zlib.compress(masks.tobytes(), level=1)
    lr_compressed = zlib.compress(low_res_masks.tobytes(), level=1)

    return {
        "masks": base64.b64encode(masks_compressed).decode("utf-8"),
        "masks_shape": list(masks.shape),
        "masks_dtype": str(masks.dtype),
        "masks_compressed": True,
        "scores": scores.tolist(),
        "low_res_masks": base64.b64encode(
            lr_compressed
        ).decode("utf-8"),
        "low_res_masks_shape": list(low_res_masks.shape),
        "low_res_masks_dtype": str(low_res_masks.dtype),
        "low_res_masks_compressed": True,
    }


@app.post("/reset")
def reset(session_id: str, x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)

    session = _sessions.pop(session_id, None)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    predictor = session.get("predictor")
    if predictor:
        try:
            predictor.reset_predictor()
        except Exception:
            pass
    return {"status": "reset_done"}
