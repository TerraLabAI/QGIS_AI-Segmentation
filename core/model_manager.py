"""
Model Manager for AI Segmentation

Handles automatic downloading and management of SAM/SAM2 ONNX models.
Supports multiple model variants stored in per-model subdirectories.

Directory Structure:
    models/
    ├── sam_vit_b/
    │   ├── encoder.onnx
    │   └── decoder.onnx
    ├── sam2_base_plus/
    │   ├── encoder.onnx
    │   └── decoder.onnx
    └── sam2_large/
        ├── encoder.onnx
        └── decoder.onnx
"""

import os
from pathlib import Path
from typing import Tuple, Callable, Optional, List
import urllib.request
import urllib.error

from qgis.core import QgsMessageLog, Qgis

from .model_registry import (
    MODEL_REGISTRY,
    MODEL_ORDER,
    DEFAULT_MODEL_ID,
    get_model_config,
    ModelConfig,
)


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


def get_model_dir(model_id: str) -> Path:
    """
    Get the directory for a specific model.
    Creates the directory if it doesn't exist.

    Args:
        model_id: Model identifier (e.g., "sam_vit_b")

    Returns:
        Path to the model's directory
    """
    model_dir = get_models_dir() / model_id
    model_dir.mkdir(exist_ok=True)
    return model_dir


def get_encoder_path(model_id: str = None) -> Path:
    """
    Get the path to the encoder model file.

    Args:
        model_id: Model identifier. If None, uses DEFAULT_MODEL_ID.

    Returns:
        Path to the encoder ONNX file
    """
    if model_id is None:
        model_id = DEFAULT_MODEL_ID
    return get_model_dir(model_id) / "encoder.onnx"


def get_decoder_path(model_id: str = None) -> Path:
    """
    Get the path to the decoder model file.

    Args:
        model_id: Model identifier. If None, uses DEFAULT_MODEL_ID.

    Returns:
        Path to the decoder ONNX file
    """
    if model_id is None:
        model_id = DEFAULT_MODEL_ID
    return get_model_dir(model_id) / "decoder.onnx"


def model_exists(model_id: str) -> bool:
    """
    Check if a specific model's encoder and decoder exist.

    Args:
        model_id: Model identifier (e.g., "sam_vit_b")

    Returns:
        True if both encoder and decoder are present
    """
    return get_encoder_path(model_id).exists() and get_decoder_path(model_id).exists()


def models_exist(model_id: str = None) -> bool:
    """
    Check if models exist for the specified or default model.
    Backward-compatible wrapper.

    Args:
        model_id: Model identifier. If None, checks if ANY model is installed.

    Returns:
        True if models are present
    """
    if model_id is not None:
        return model_exists(model_id)
    # If no model_id specified, check if any model is installed
    return len(get_installed_models()) > 0


def get_installed_models() -> List[str]:
    """
    Get list of installed model IDs.

    Returns:
        List of model IDs that are fully installed
    """
    installed = []
    for model_id in MODEL_REGISTRY.keys():
        if model_exists(model_id):
            installed.append(model_id)
    return installed


def get_first_installed_model() -> Optional[str]:
    """
    Get the first installed model in preference order.

    Returns:
        Model ID of the first installed model, or None if none installed
    """
    for model_id in MODEL_ORDER:
        if model_exists(model_id):
            return model_id
    return None


def get_missing_models() -> list:
    """
    Get list of missing model files for the default model.
    Backward-compatible wrapper.

    Returns:
        List of missing model names ('encoder', 'decoder')
    """
    return get_missing_model_files(DEFAULT_MODEL_ID)


def get_missing_model_files(model_id: str) -> list:
    """
    Get list of missing model files for a specific model.

    Args:
        model_id: Model identifier

    Returns:
        List of missing model names ('encoder', 'decoder')
    """
    missing = []
    if not get_encoder_path(model_id).exists():
        missing.append("encoder")
    if not get_decoder_path(model_id).exists():
        missing.append("decoder")
    return missing


def migrate_legacy_models():
    """
    Migrate old model files to the new per-model directory structure.

    Old structure:
        models/
        ├── encoder-quant.onnx
        └── decoder-quant.onnx

    New structure:
        models/
        └── sam_vit_b/
            ├── encoder.onnx
            └── decoder.onnx
    """
    models_dir = get_models_dir()
    old_encoder = models_dir / "encoder-quant.onnx"
    old_decoder = models_dir / "decoder-quant.onnx"

    if old_encoder.exists() and old_decoder.exists():
        QgsMessageLog.logMessage(
            "Migrating legacy model files to new structure...",
            "AI Segmentation",
            level=Qgis.Info
        )

        # Create new directory
        new_dir = get_model_dir("sam_vit_b")

        # Move and rename files
        new_encoder = new_dir / "encoder.onnx"
        new_decoder = new_dir / "decoder.onnx"

        try:
            old_encoder.rename(new_encoder)
            old_decoder.rename(new_decoder)
            QgsMessageLog.logMessage(
                "Successfully migrated legacy models to sam_vit_b/",
                "AI Segmentation",
                level=Qgis.Info
            )
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Failed to migrate legacy models: {e}",
                "AI Segmentation",
                level=Qgis.Warning
            )
    elif old_encoder.exists() or old_decoder.exists():
        # Partial old installation - clean up
        QgsMessageLog.logMessage(
            "Found partial legacy model files, cleaning up...",
            "AI Segmentation",
            level=Qgis.Info
        )
        if old_encoder.exists():
            old_encoder.unlink()
        if old_decoder.exists():
            old_decoder.unlink()


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


def download_model(
    model_id: str,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Tuple[bool, str]:
    """
    Download a specific model from HuggingFace.

    Args:
        model_id: Model identifier (e.g., "sam_vit_b")
        progress_callback: Optional callback(percent, status_message)

    Returns:
        Tuple of (success, message)
    """
    try:
        config = get_model_config(model_id)
    except KeyError as e:
        return False, str(e)

    # Check if already downloaded
    if model_exists(model_id):
        return True, f"{config.display_name} already downloaded"

    # Build URLs
    base_url = f"https://huggingface.co/{config.huggingface_repo}/resolve/main"
    encoder_url = f"{base_url}/{config.encoder_file}"
    decoder_url = f"{base_url}/{config.decoder_file}"

    # Calculate total size
    total_size = (config.encoder_size_mb + config.decoder_size_mb) * 1024 * 1024
    downloaded_total = 0

    # Download encoder
    missing = get_missing_model_files(model_id)

    for part in ["encoder", "decoder"]:
        if part not in missing:
            continue

        if part == "encoder":
            url = encoder_url
            destination = get_encoder_path(model_id)
            size_mb = config.encoder_size_mb
        else:
            url = decoder_url
            destination = get_decoder_path(model_id)
            size_mb = config.decoder_size_mb

        if progress_callback:
            progress_callback(
                int(downloaded_total / total_size * 100),
                f"Downloading {part}..."
            )

        def file_progress(downloaded, file_total, msg):
            if progress_callback:
                current_total = downloaded_total + downloaded
                percent = int(current_total / total_size * 100) if total_size > 0 else 0
                progress_callback(percent, f"Downloading {part}: {msg}")

        success = download_file(url, destination, file_progress)

        if not success:
            return False, f"Failed to download {part} for {config.display_name}"

        downloaded_total += size_mb * 1024 * 1024

    if progress_callback:
        progress_callback(100, f"{config.display_name} downloaded!")

    return True, f"{config.display_name} downloaded successfully"


def download_models(
    progress_callback: Optional[Callable[[int, str], None]] = None,
    model_id: str = None
) -> Tuple[bool, str]:
    """
    Download models. Backward-compatible wrapper.

    Args:
        progress_callback: Optional callback(percent, status_message)
        model_id: Optional model ID. If None, downloads DEFAULT_MODEL_ID.

    Returns:
        Tuple of (success, message)
    """
    if model_id is None:
        model_id = DEFAULT_MODEL_ID
    return download_model(model_id, progress_callback)


def verify_model(model_id: str) -> Tuple[bool, str]:
    """
    Verify that a specific model's ONNX files are valid.

    Args:
        model_id: Model identifier

    Returns:
        Tuple of (valid, message)
    """
    try:
        import onnxruntime as ort

        encoder_path = get_encoder_path(model_id)
        decoder_path = get_decoder_path(model_id)

        if not encoder_path.exists():
            return False, f"Encoder model not found for {model_id}"
        if not decoder_path.exists():
            return False, f"Decoder model not found for {model_id}"

        # Try to load models (this validates them)
        try:
            ort.InferenceSession(str(encoder_path), providers=['CPUExecutionProvider'])
        except Exception as e:
            return False, f"Invalid encoder model: {str(e)}"

        try:
            ort.InferenceSession(str(decoder_path), providers=['CPUExecutionProvider'])
        except Exception as e:
            return False, f"Invalid decoder model: {str(e)}"

        return True, f"{model_id} verified successfully"

    except ImportError:
        return False, "onnxruntime not installed"
    except Exception as e:
        return False, f"Verification failed: {str(e)}"


def verify_models() -> Tuple[bool, str]:
    """
    Verify the default model. Backward-compatible wrapper.

    Returns:
        Tuple of (valid, message)
    """
    return verify_model(DEFAULT_MODEL_ID)


def get_model_info(model_id: str = None) -> dict:
    """
    Get information about a model setup.

    Args:
        model_id: Model identifier. If None, uses DEFAULT_MODEL_ID.

    Returns:
        Dictionary with model information
    """
    if model_id is None:
        model_id = DEFAULT_MODEL_ID

    try:
        config = get_model_config(model_id)
    except KeyError:
        return {"error": f"Unknown model: {model_id}"}

    return {
        "model_id": model_id,
        "display_name": config.display_name,
        "description": config.description,
        "repository": config.huggingface_repo,
        "encoder": {
            "path": str(get_encoder_path(model_id)),
            "exists": get_encoder_path(model_id).exists(),
            "remote_filename": config.encoder_file,
            "size_mb": config.encoder_size_mb,
        },
        "decoder": {
            "path": str(get_decoder_path(model_id)),
            "exists": get_decoder_path(model_id).exists(),
            "remote_filename": config.decoder_file,
            "size_mb": config.decoder_size_mb,
        },
        "total_size_mb": config.total_size_mb,
        "ready": model_exists(model_id),
    }


def get_all_models_info() -> List[dict]:
    """
    Get information about all available models.

    Returns:
        List of model info dictionaries
    """
    return [get_model_info(model_id) for model_id in MODEL_ORDER]


def delete_model(model_id: str) -> bool:
    """
    Delete a specific model's files.

    Args:
        model_id: Model identifier

    Returns:
        True if deletion successful
    """
    try:
        encoder_path = get_encoder_path(model_id)
        decoder_path = get_decoder_path(model_id)

        if encoder_path.exists():
            encoder_path.unlink()
        if decoder_path.exists():
            decoder_path.unlink()

        # Try to remove the model directory if empty
        model_dir = get_model_dir(model_id)
        try:
            model_dir.rmdir()
        except OSError:
            pass  # Directory not empty or other issue

        QgsMessageLog.logMessage(
            f"Model {model_id} deleted",
            "AI Segmentation",
            level=Qgis.Info
        )
        return True
    except Exception as e:
        QgsMessageLog.logMessage(
            f"Failed to delete model {model_id}: {str(e)}",
            "AI Segmentation",
            level=Qgis.Warning
        )
        return False


def delete_models() -> bool:
    """
    Delete the default model. Backward-compatible wrapper.

    Returns:
        True if deletion successful
    """
    return delete_model(DEFAULT_MODEL_ID)
