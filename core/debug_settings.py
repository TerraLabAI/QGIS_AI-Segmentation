
from dataclasses import dataclass, field
from typing import Callable, List
from qgis.core import QgsMessageLog, Qgis


@dataclass
class DebugSettings:
    mask_threshold: float = 0.0
    force_cpu: bool = True
    max_image_size: int = 2048
    show_confidence_scores: bool = True
    show_timing_info: bool = True
    verbose_logging: bool = False

    _listeners: List[Callable] = field(default_factory=list, repr=False)

    def add_listener(self, callback: Callable):
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable):
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify_listeners(self):
        for listener in self._listeners:
            try:
                listener(self)
            except Exception as e:
                QgsMessageLog.logMessage(
                    f"Settings listener error: {e}",
                    "AI Segmentation",
                    level=Qgis.Warning
                )

    def set_mask_threshold(self, value: float):
        self.mask_threshold = max(-5.0, min(5.0, value))
        QgsMessageLog.logMessage(
            f"[SETTINGS] mask_threshold = {self.mask_threshold}",
            "AI Segmentation",
            level=Qgis.Info
        )
        self._notify_listeners()

    def set_force_cpu(self, value: bool):
        self.force_cpu = value
        QgsMessageLog.logMessage(
            f"[SETTINGS] force_cpu = {self.force_cpu}",
            "AI Segmentation",
            level=Qgis.Info
        )
        self._notify_listeners()

    def set_max_image_size(self, value: int):
        self.max_image_size = max(512, min(4096, value))
        QgsMessageLog.logMessage(
            f"[SETTINGS] max_image_size = {self.max_image_size}",
            "AI Segmentation",
            level=Qgis.Info
        )
        self._notify_listeners()

    def set_show_confidence_scores(self, value: bool):
        self.show_confidence_scores = value
        self._notify_listeners()

    def set_show_timing_info(self, value: bool):
        self.show_timing_info = value
        self._notify_listeners()

    def set_verbose_logging(self, value: bool):
        self.verbose_logging = value
        QgsMessageLog.logMessage(
            f"[SETTINGS] verbose_logging = {self.verbose_logging}",
            "AI Segmentation",
            level=Qgis.Info
        )
        self._notify_listeners()

    def to_dict(self) -> dict:
        return {
            "mask_threshold": self.mask_threshold,
            "force_cpu": self.force_cpu,
            "max_image_size": self.max_image_size,
            "show_confidence_scores": self.show_confidence_scores,
            "show_timing_info": self.show_timing_info,
            "verbose_logging": self.verbose_logging,
        }

    def from_dict(self, data: dict):
        if "mask_threshold" in data:
            self.mask_threshold = data["mask_threshold"]
        if "force_cpu" in data:
            self.force_cpu = data["force_cpu"]
        if "max_image_size" in data:
            self.max_image_size = data["max_image_size"]
        if "show_confidence_scores" in data:
            self.show_confidence_scores = data["show_confidence_scores"]
        if "show_timing_info" in data:
            self.show_timing_info = data["show_timing_info"]
        if "verbose_logging" in data:
            self.verbose_logging = data["verbose_logging"]


_global_settings = DebugSettings()


def get_settings() -> DebugSettings:
    return _global_settings


def reset_settings():
    global _global_settings
    listeners = _global_settings._listeners.copy()
    _global_settings = DebugSettings()
    _global_settings._listeners = listeners
    _global_settings._notify_listeners()
