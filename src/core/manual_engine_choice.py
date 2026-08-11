"""Whether the user has agreed to what a cloud Semi-Auto session sends.

``manual_cloud_route.py`` holds where the clicks are answered. This file holds
the one fact that decides whether they may travel at all: has the user read
what leaves the machine, and said yes.

Asked once, at the first cloud Start, never at sign-in. Consent belongs to the
moment imagery is about to leave the machine, not to an account form.

It defaults to False, and so does an unreadable settings store: asking twice
costs a click, assuming an answer nobody gave costs trust. Everything that
sends a crop checks this for itself, which is what makes an engine card
that ships on Cloud AI safe. A headless caller or a script can open no dialog, so it never
sends one.

An earlier build also stored whether the user had been SHOWN a chooser screen,
under ``AISegmentation/manual_engine_chosen``. That screen is gone, replaced by
the checkbox on the Semi-Auto page, so nothing reads the key any more. It is
left where it lies in the profiles that carry it.
"""
from __future__ import annotations

from qgis.core import QgsSettings

# Frozen literal. It sits in the user's QGIS profile, so a rename silently
# re-asks every install that has already answered.
MANUAL_CLOUD_CONSENT_KEY = "AISegmentation/manual_cloud_consent"


def cloud_consent_given() -> bool:
    """Whether the user has accepted what a cloud session sends."""
    try:
        return bool(QgsSettings().value(
            MANUAL_CLOUD_CONSENT_KEY, False, type=bool))
    except Exception:  # noqa: BLE001 -- an unreadable setting means "not yet"  # nosec B110
        return False


def set_cloud_consent(accepted: bool) -> None:
    """Record the answer to the cloud data notice.

    A refusal is stored as a refusal rather than left blank: the dialog then
    comes back the next time cloud is picked, which is the point, and nothing
    reads the flag as "already agreed".
    """
    try:
        QgsSettings().setValue(MANUAL_CLOUD_CONSENT_KEY, bool(accepted))
    except Exception:  # noqa: BLE001 -- the answer is lost, the session is not  # nosec B110
        pass
