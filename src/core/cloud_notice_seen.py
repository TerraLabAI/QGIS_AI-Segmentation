"""Whether the cloud disclosure line has already been read on this machine.

VISIBILITY ONLY. This flag hides a sentence and does nothing else. It must
never gate a run, a route, a request or a permission. So
``mark_cloud_notice_seen`` takes no argument and can store no answer, and
``cloud_notice_seen`` is only ever read inside a ``setVisible`` call. A caller
asking whether something is ALLOWED is in the wrong file.

One flag for the whole plugin, not one per mode. A Semi-Auto click and an
Automatic tile reach the same servers, so whichever mode the user runs first
answers for both.

Write-once, never cleared, per machine. Account Settings keeps the Terms and
Privacy links reachable for as long as the plugin is installed, which is what
makes retiring the line safe.

Not ``MANUAL_CLOUD_CONSENT_KEY``: that key meant consent, it is dormant, and
reusing it would read an old yes or no as an answer to a question nobody is
asking any more.
"""
from __future__ import annotations

from qgis.core import QgsSettings

# Frozen literal. It sits in the user's QGIS profile, so a rename shows the
# line again on every install that has already retired it.
CLOUD_NOTICE_SEEN_KEY = "AISegmentation/cloud_notice_seen"


def cloud_notice_seen() -> bool:
    """Whether a cloud run has already completed on this machine.

    False on an unreadable store: showing the line once more costs a line of
    text, hiding it on a guess costs the disclosure.
    """
    try:
        return bool(QgsSettings().value(
            CLOUD_NOTICE_SEEN_KEY, False, type=bool))
    except Exception:  # noqa: BLE001 -- an unreadable setting means "not yet"  # nosec B110
        return False


def mark_cloud_notice_seen() -> None:
    """Record that a cloud run completed. Takes no answer, so it cannot store one."""
    try:
        QgsSettings().setValue(CLOUD_NOTICE_SEEN_KEY, True)
    except Exception:  # noqa: BLE001 -- a lost write shows the line once more  # nosec B110
        pass
