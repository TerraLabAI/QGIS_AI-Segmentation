


















from __future__ import annotations


def correct_ai_method_enabled() -> bool:







    try:
        from ...core.server_dials import feature_enabled

        return feature_enabled("correct_ai_method")
    except Exception:  # noqa: BLE001
        return True


def correct_ai_ready_without_download() -> bool:














    try:
        from ...core.activation_manager import get_auth_token
        from ...core.server_dials import correct_ai_cloud_enabled

        if correct_ai_cloud_enabled() and get_auth_token():
            return True
    except Exception:  # noqa: BLE001
        pass  # nosec B110
    try:
        from ...core.checkpoint_manager import checkpoint_exists
        from ...core.venv_manager import local_model_ready

        packages_ready, _why = local_model_ready()
        return bool(packages_ready) and checkpoint_exists()
    except Exception:  # noqa: BLE001
        return False


def correct_default_method() -> str:

    if not correct_ai_method_enabled():
        return "manual"
    try:
        from ...core.detection_policy import (
            review_correct_default_method,
            review_correct_default_method_ready,
        )

        if correct_ai_ready_without_download():
            return review_correct_default_method_ready()
        return review_correct_default_method()
    except Exception:  # noqa: BLE001
        return "manual"
