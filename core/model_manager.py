"""
Model Manager for AI Segmentation

Handles automatic downloading and management of SAM ONNX models.
Models are downloaded from HuggingFace Hub on first run.
"""

import os
from pathlib import Path
from typing import Tuple, Callable, Optional
import urllib.request
import urllib.error

from qgis.core import QgsMessageLog, Qgis


# HuggingFace model repository
HUGGINGFACE_REPO = "visheratin/segment-anything-vit-b"
HUGGINGFACE_BASE_URL = f"https://huggingface.co/{HUGGINGFACE_REPO}/resolve/main"

# Model files - using quantized versions for smaller download
# Full versions available: encoder.onnx (359MB), decoder.onnx (16.5MB)
MODEL_FILES = {
    "encoder": {
        "filename": "encoder-quant.onnx",
        "url": f"{HUGGINGFACE_BASE_URL}/encoder-quant.onnx",
        "size_mb": 100,  # Approximate size for progress calculation
    },
    "decoder": {
        "filename": "decoder-quant.onnx",
        "url": f"{HUGGINGFACE_BASE_URL}/decoder-quant.onnx",
        "size_mb": 9,
    },
}

# Total download size (approximate)
TOTAL_DOWNLOAD_SIZE_MB = sum(m["size_mb"] for m in MODEL_FILES.values())


def get_plugin_dir() -> Path:
    """Get the plugin directory path."""
    return Path(__file__).parent.parent


def get_models_dir() -> Path:
    """
    Get the models directory path.
    Creates the directory if it doesn't exist.

    Returns:
        Path to the models directory
    """
    models_dir = get_plugin_dir() / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


def get_encoder_path() -> Path:
    """Get the path to the encoder model file."""
    return get_models_dir() / MODEL_FILES["encoder"]["filename"]


def get_decoder_path() -> Path:
    """Get the path to the decoder model file."""
    return get_models_dir() / MODEL_FILES["decoder"]["filename"]


def models_exist() -> bool:
    """
    Check if both encoder and decoder models exist.

    Returns:
        True if both models are present
    """
    return get_encoder_path().exists() and get_decoder_path().exists()


def get_missing_models() -> list:
    """
    Get list of missing model files.

    Returns:
        List of missing model names ('encoder', 'decoder')
    """
    missing = []
    if not get_encoder_path().exists():
        missing.append("encoder")
    if not get_decoder_path().exists():
        missing.append("decoder")
    return missing


def download_file(
    url: str,
    destination: Path,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> bool:
    """
    Download a file from URL with progress reporting.

    Args:
        url: URL to download from
        destination: Local path to save the file
        progress_callback: Optional callback(downloaded_bytes, total_bytes, status_message)

    Returns:
        True if download successful
    """
    try:
        # Create a request with a user agent (some servers require this)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "QGIS-AI-Segmentation/1.0"}
        )

        with urllib.request.urlopen(request, timeout=30) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 8192 * 4  # 32KB chunks

            # Create parent directory if needed
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Download to temp file first, then rename
            temp_path = destination.with_suffix(".download")

            with open(temp_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if progress_callback:
                        progress_callback(
                            downloaded,
                            total_size,
                            f"Downloading... {downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB"
                        )

            # Rename temp file to final destination
            temp_path.rename(destination)

            QgsMessageLog.logMessage(
                f"Downloaded: {destination.name}",
                "AI Segmentation",
                level=Qgis.Info
            )
            return True

    except urllib.error.URLError as e:
        QgsMessageLog.logMessage(
            f"Download failed (network error): {str(e)}",
            "AI Segmentation",
            level=Qgis.Critical
        )
        return False
    except Exception as e:
        QgsMessageLog.logMessage(
            f"Download failed: {str(e)}",
            "AI Segmentation",
            level=Qgis.Critical
        )
        # Clean up partial download
        temp_path = destination.with_suffix(".download")
        if temp_path.exists():
            temp_path.unlink()
        return False


def download_models(
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Tuple[bool, str]:
    """
    Download all required SAM models from HuggingFace.

    Args:
        progress_callback: Optional callback(percent, status_message)

    Returns:
        Tuple of (success, message)
    """
    missing = get_missing_models()

    if not missing:
        return True, "Models already downloaded"

    total_size = sum(MODEL_FILES[m]["size_mb"] for m in missing) * 1024 * 1024
    downloaded_total = 0

    for model_name in missing:
        model_info = MODEL_FILES[model_name]
        url = model_info["url"]

        if model_name == "encoder":
            destination = get_encoder_path()
        else:
            destination = get_decoder_path()

        if progress_callback:
            progress_callback(
                int(downloaded_total / total_size * 100),
                f"Downloading {model_name} model..."
            )

        def file_progress(downloaded, file_total, msg):
            if progress_callback:
                current_total = downloaded_total + downloaded
                percent = int(current_total / total_size * 100) if total_size > 0 else 0
                progress_callback(percent, f"Downloading {model_name}: {msg}")

        success = download_file(url, destination, file_progress)

        if not success:
            return False, f"Failed to download {model_name} model"

        downloaded_total += model_info["size_mb"] * 1024 * 1024

    if progress_callback:
        progress_callback(100, "All models downloaded!")

    return True, "Models downloaded successfully"


def verify_models() -> Tuple[bool, str]:
    """
    Verify that downloaded models are valid ONNX files.

    Returns:
        Tuple of (valid, message)
    """
    try:
        import onnxruntime as ort

        encoder_path = get_encoder_path()
        decoder_path = get_decoder_path()

        if not encoder_path.exists():
            return False, "Encoder model not found"
        if not decoder_path.exists():
            return False, "Decoder model not found"

        # Try to load models (this validates them)
        try:
            ort.InferenceSession(str(encoder_path), providers=['CPUExecutionProvider'])
        except Exception as e:
            return False, f"Invalid encoder model: {str(e)}"

        try:
            ort.InferenceSession(str(decoder_path), providers=['CPUExecutionProvider'])
        except Exception as e:
            return False, f"Invalid decoder model: {str(e)}"

        return True, "Models verified successfully"

    except ImportError:
        return False, "onnxruntime not installed"
    except Exception as e:
        return False, f"Verification failed: {str(e)}"


def get_model_info() -> dict:
    """
    Get information about the current model setup.

    Returns:
        Dictionary with model information
    """
    return {
        "repository": HUGGINGFACE_REPO,
        "encoder": {
            "path": str(get_encoder_path()),
            "exists": get_encoder_path().exists(),
            "filename": MODEL_FILES["encoder"]["filename"],
        },
        "decoder": {
            "path": str(get_decoder_path()),
            "exists": get_decoder_path().exists(),
            "filename": MODEL_FILES["decoder"]["filename"],
        },
        "total_size_mb": TOTAL_DOWNLOAD_SIZE_MB,
        "ready": models_exist(),
    }


def delete_models() -> bool:
    """
    Delete downloaded models (for troubleshooting/re-download).

    Returns:
        True if deletion successful
    """
    try:
        encoder_path = get_encoder_path()
        decoder_path = get_decoder_path()

        if encoder_path.exists():
            encoder_path.unlink()
        if decoder_path.exists():
            decoder_path.unlink()

        QgsMessageLog.logMessage(
            "Models deleted",
            "AI Segmentation",
            level=Qgis.Info
        )
        return True
    except Exception as e:
        QgsMessageLog.logMessage(
            f"Failed to delete models: {str(e)}",
            "AI Segmentation",
            level=Qgis.Warning
        )
        return False
