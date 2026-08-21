"""The Semi-Auto option that answers clicks on TerraLab's servers.

Two questions, and nothing else: does the user want it, and is it offered at
all. Whoever swaps a predictor in asks both first.

A plugin that never hears from a server still behaves exactly as it did before
this file existed. That is carried by the second question, which is fail-closed
(see ``manual_cloud_route_offered``), not by the first, which answers yes until
the user says otherwise: the cloud engine is the one the product recommends, and
a user who has never touched the cards gets it.

That means the FIRST click on the cloud half sends a crop. The disclosure is
the engine card, which names the two halves and says where each one runs, plus
a line under it naming the destination. It is visible before the click rather
than a modal blocking it. Disclosure, not a gate: ``cloud_notice_seen`` decides
whether that line is drawn and nothing else, so do not reach for it here.

The choice lasts for the session and no longer. It used to sit in the user's
QGIS profile under ``AISegmentation/manual_cloud_route``, which held a machine
on one engine long after the afternoon that picked it. That key is now
abandoned on purpose: the profiles that carry it keep it, and nothing reads it.
Picking My computer is a step the user takes when they want it, each time.
"""
from __future__ import annotations

# The name the fleet-wide switch is published under, so the option can be
# withdrawn from everyone with one deploy and no plugin release.
MANUAL_CLOUD_ROUTE_FEATURE = "manual_cloud_route"

# Session state, not a setting. Module-level, so a plugin reload clears it along
# with the module and the cards come back up on Cloud AI.
_cloud_route_picked = True


def manual_cloud_route_enabled() -> bool:
    """Whether the clicks should be answered off the machine.

    True at every start: the engine cards on the Semi-Auto page open on Cloud
    AI. A user who picks My computer holds it until QGIS or the plugin
    restarts. No imagery travels on this alone: the data notice is a separate
    answer, and every sender checks both.
    """
    return _cloud_route_picked


def set_manual_cloud_route_enabled(enabled: bool) -> None:
    """Record the user's choice for the rest of this session."""
    global _cloud_route_picked
    _cloud_route_picked = bool(enabled)


def manual_cloud_route_offered() -> bool:
    """Whether the option is shown at all.

    Fail-CLOSED, which is the opposite of nearly every switch here and is the
    point. The others withdraw a feature the plugin can perform on its own;
    this one offers a route that only exists if a server answers. Today no
    deployed server does. Offered by default, a user who picks Cloud AI to skip
    the on-device install has no model at all behind it, and every click ends
    in a failure dialog. Off by default, that user keeps the plugin they
    already had.

    So the option appears the day the blob says ``features.manual_cloud_route:
    true``, and not before. Same shape as ``correct_ai_cloud_enabled``, for the
    same reason.
    """
    if _dev_cloud_route_opt_in():
        return True
    try:
        from .server_dials import dial_bool

        return dial_bool(f"features.{MANUAL_CLOUD_ROUTE_FEATURE}", False)
    except Exception:  # nosec B110 -- no configuration means no cloud route
        return False


# Dev opt-in, same channel and same rules as TERRALAB_DETECTION_URL: a line in
# the gitignored .env.local, absent from every user's checkout, so this reads
# False everywhere but a working tree that asked for it. It exists so the
# route can be exercised from a development checkout before it is turned on
# for anyone.
_MANUAL_CLOUD_ENV = "TERRALAB_MANUAL_CLOUD"


def _dev_cloud_route_opt_in() -> bool:
    """Whether this working tree asked for the cloud engine by hand."""
    import os

    plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    env_path = os.path.join(plugin_dir, ".env.local")
    try:
        if not os.path.isfile(env_path):
            return False
        with open(env_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line.startswith(f"{_MANUAL_CLOUD_ENV}="):
                    value = line.split("=", 1)[1].strip().strip('"').strip("'")
                    return value.lower() in ("1", "true", "yes", "on")
    except Exception:  # noqa: BLE001 -- an unreadable file is no opt-in  # nosec B110
        return False
    return False
