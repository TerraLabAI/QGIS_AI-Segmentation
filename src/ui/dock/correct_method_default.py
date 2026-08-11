"""Which half of the Correct step's AI | Manual switch a review opens on.

Three questions, kept apart on purpose:

- Is the AI method offered at all? A server kill switch, fail-open.
- Would arming it right now cost the user a download? A local capability
  answer, read off the machine, fail-closed.
- Which method does the server prefer? Two served keys, one for each answer to
  the question above.

The order matters. The AI method is worth opening on only when it draws the
first outline at once. When it would start an install instead, the step opens
on hand editing, whatever the server prefers, because a user who lands on
Correct wants a tool and not a wait.

Read at dock build AND at every review open (`reset_review_steps`): the dock is
built before the first configuration fetch lands, and a sign-in or a finished
install between two reviews changes the answer.
"""
from __future__ import annotations


def correct_ai_method_enabled() -> bool:
    """Whether the review offers the AI fix method at all.

    A kill switch, not a preference: picking a polygon with AI armed can start
    the on-device install, and that holds the whole review until it finishes.
    Off, every fix is a Manual one. Fail-open, so an absent or damaged
    configuration leaves both methods exactly as shipped.
    """
    try:
        from ...core.server_dials import feature_enabled

        return feature_enabled("correct_ai_method")
    except Exception:  # noqa: BLE001 -- a kill switch is best-effort
        return True


def correct_ai_ready_without_download() -> bool:
    """Whether arming the AI fix right now would download or install nothing.

    Two ways to be ready, and either one is enough:

    - the off-machine route is switched on and the user is signed in, so no
      local piece is ever needed;
    - the on-device pieces are already on disk, packages and model file both.

    Anything else means the first click would open the one-time setup, so the
    caller must not open the step on this method. Fails closed: an answer that
    cannot be read counts as not ready.

    Filesystem checks only, cheap enough for a click path and for dock build.
    """
    try:
        from ...core.activation_manager import get_auth_token
        from ...core.server_dials import correct_ai_cloud_enabled

        if correct_ai_cloud_enabled() and get_auth_token():
            return True
    except Exception:  # noqa: BLE001 -- an unreadable route is not a ready one
        pass  # nosec B110
    try:
        from ...core.checkpoint_manager import checkpoint_exists
        from ...core.venv_manager import local_model_ready

        packages_ready, _why = local_model_ready()
        return bool(packages_ready) and checkpoint_exists()
    except Exception:  # noqa: BLE001 -- a probe that fails is not a ready one
        return False


def correct_default_method() -> str:
    """Which fix method the Correct step opens on: "ai" or "manual"."""
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
    except Exception:  # noqa: BLE001 -- a served default is best-effort
        return "manual"
