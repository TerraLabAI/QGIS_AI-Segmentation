"""The Semi-Auto option that answers clicks on TerraLab's servers.

Two questions, and nothing else: does the user want it, and is it offered at
all. Whoever swaps a predictor in asks both first.

A plugin that never hears from a server still behaves exactly as it did before
this file existed. That is carried by the second question, which is fail-closed
(see ``manual_cloud_route_offered``), not by the first, which now reads an
unanswered store as yes: the cloud engine is the one the product recommends,
and a user who has never touched the box gets it. Nothing travels on that alone.
Imagery leaves the machine only once the data notice has been accepted, which
the click path checks for itself.
"""
from __future__ import annotations

from qgis.core import QgsSettings

# Frozen literal. It sits in the user's QGIS profile, so a rename would lose
# the choice of every install that already made one.
MANUAL_CLOUD_ROUTE_KEY = "AISegmentation/manual_cloud_route"

# The name the fleet-wide switch is published under, so the option can be
# withdrawn from everyone with one deploy and no plugin release.
MANUAL_CLOUD_ROUTE_FEATURE = "manual_cloud_route"


def manual_cloud_route_enabled(settings=None) -> bool:
    """Whether the clicks should be answered off the machine.

    True when nothing has been stored, because the engine cards on the
    Semi-Auto page open on Cloud AI. A user who picked My computer has False
    written, and keeps it. No imagery travels on this alone: the data notice is
    a separate answer, and every sender checks both.

    False on anything unreadable: the on-device path is the one that always
    works, so it is what a broken store means.
    """
    try:
        store = settings or QgsSettings()
        return bool(store.value(MANUAL_CLOUD_ROUTE_KEY, True, type=bool))
    except Exception:  # nosec B110 -- an unreadable setting means off
        return False


def set_manual_cloud_route_enabled(enabled: bool, settings=None) -> None:
    """Persist the user's choice. Never raises: a settings store that refuses a
    write must not take the toggle down with it."""
    try:
        store = settings or QgsSettings()
        store.setValue(MANUAL_CLOUD_ROUTE_KEY, bool(enabled))
    except Exception:  # nosec B110 -- the choice is lost, the session is not
        pass


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
# False everywhere but a working tree that asked for it. It exists because the
# route is testable long before it is servable: the tagged revision answers
# /refine today while production traffic still runs an image without it, and
# an .env.local already points this checkout at that revision.
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
