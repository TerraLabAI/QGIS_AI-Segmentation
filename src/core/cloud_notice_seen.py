



















from __future__ import annotations

from qgis.core import QgsSettings



CLOUD_NOTICE_SEEN_KEY = "AISegmentation/cloud_notice_seen"


def cloud_notice_seen() -> bool:





    try:
        return bool(QgsSettings().value(
            CLOUD_NOTICE_SEEN_KEY, False, type=bool))
    except Exception:  # noqa: BLE001  # nosec B110
        return False


def mark_cloud_notice_seen() -> None:

    try:
        QgsSettings().setValue(CLOUD_NOTICE_SEEN_KEY, True)
    except Exception:  # noqa: BLE001  # nosec B110
        pass
