"""Shared logging utility for the AI Segmentation plugin."""

from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog

LOG_TAG = "AI Segmentation"


def log(message: str, level=Qgis.MessageLevel.Info):
    """Log a message to the QGIS message log under the AI Segmentation tag."""
    QgsMessageLog.logMessage(message, LOG_TAG, level=level)
