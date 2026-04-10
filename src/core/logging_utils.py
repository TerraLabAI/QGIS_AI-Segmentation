

from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog

LOG_TAG = "AI Segmentation"


def log(message: str, level=Qgis.MessageLevel.Info):

    QgsMessageLog.logMessage(message, LOG_TAG, level=level)
