








from __future__ import annotations

from qgis.PyQt.QtCore import QSettings

from ...core.i18n import tr
from ...core.server_dials import dial_in_range

UPDATE_TRIGGER_PLUGIN_REGISTRY = "plugin_registry"
UPDATE_TRIGGER_SERVED_LATEST = "served_latest_version"
UPDATE_TRIGGER_SERVED_MIN = "served_min_recommended"



_PROMPT_SHOWN_KEY = "AISegmentation/telemetry/update_prompt_shown_version"
_UNKNOWN_VERSION = "unknown"

_UPDATE_NOTE_SECONDS = 8


def resolve_update_offer(installed_version: str) -> tuple[str | None, str]:





    from ...core.activation_manager import (
        get_latest_version,
        is_update_available,
        is_update_recommended,
    )

    if is_update_available(installed_version):
        return get_latest_version(), UPDATE_TRIGGER_SERVED_LATEST
    if is_update_recommended(installed_version):
        return get_latest_version() or _UNKNOWN_VERSION, UPDATE_TRIGGER_SERVED_MIN
    return None, ""


class DockServerSwitchesMixin:


    def _setup_update_recommendation(self):

        from .guidance import BLUE_TINT, HINT_UPDATE_RECOMMENDED, DismissibleHint

        self.update_recommended_hint = DismissibleHint(
            HINT_UPDATE_RECOMMENDED,
            tr("A newer version of AI Segmentation is available with the "
               "latest fixes."),
            tint=BLUE_TINT,
            action_text=tr("Update now"),
            visibility_gate=self._should_show_update_recommendation,
            parent=self,
        )
        self.update_recommended_hint.action.connect(self._on_update_action_clicked)
        self.update_recommended_hint.dismissed.connect(self._on_update_hint_dismissed)
        self.update_recommended_hint.setVisible(False)
        self.main_layout.addWidget(self.update_recommended_hint)

    def _installed_version(self) -> str:
        try:
            from ...core.request_context import plugin_version

            return plugin_version() or ""
        except Exception:  # noqa: BLE001
            return ""

    def _offered_update_version(self) -> str:

        installed = self._installed_version()
        if not installed:
            return ""
        try:
            version, _trigger = resolve_update_offer(installed)
        except Exception:  # noqa: BLE001
            return ""
        return version or ""

    def _should_show_update_recommendation(self) -> bool:














        return False

    def _update_banner_text(self, version: str) -> str:





        from ...core.activation_manager import get_release_notes_line

        if not version or version == _UNKNOWN_VERSION:
            return tr("A newer version of AI Segmentation is available with the "
                      "latest fixes.")
        lines = [tr("AI Segmentation {version} is available.").format(version=version)]
        try:
            notes = get_release_notes_line()
        except Exception:  # noqa: BLE001
            notes = None
        if notes:
            lines.append(notes)
        return "\n".join(lines)

    def refresh_update_recommendation(self) -> None:

        hint = getattr(self, "update_recommended_hint", None)
        if hint is None:
            return
        try:
            from .guidance import HINT_UPDATE_RECOMMENDED, is_hint_dismissed_for_version

            version = self._offered_update_version()
            show = bool(version) and self._should_show_update_recommendation()
            if show and is_hint_dismissed_for_version(
                    HINT_UPDATE_RECOMMENDED, version, self._installed_version()):
                show = False
            if show:
                hint.set_body_text(self._update_banner_text(version))
                _, trigger = resolve_update_offer(self._installed_version())
                self._track_update_prompt_shown(
                    version, trigger or UPDATE_TRIGGER_SERVED_LATEST)
            hint.setVisible(show)
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _track_update_prompt_shown(self, version: str, trigger: str) -> None:

        try:
            settings = QSettings()
            if settings.value(_PROMPT_SHOWN_KEY, "", type=str) == version:
                return
            settings.setValue(_PROMPT_SHOWN_KEY, version)
            from ...core import telemetry_events as ev
            from ...core.telemetry import track

            track(ev.PLUGIN_UPDATE_PROMPT_SHOWN, {
                "offered_version": version,
                "trigger": trigger,
            })
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _track_update_prompt_clicked(self, version: str, action: str) -> None:
        try:
            from ...core import telemetry_events as ev
            from ...core.telemetry import track

            track(ev.PLUGIN_UPDATE_PROMPT_CLICKED, {
                "offered_version": version or _UNKNOWN_VERSION,
                "action": action,
            })
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _on_update_hint_dismissed(self) -> None:

        version = self._offered_update_version()
        try:
            from .guidance import HINT_UPDATE_RECOMMENDED, dismiss_hint_for_version

            if version:
                dismiss_hint_for_version(HINT_UPDATE_RECOMMENDED, version)
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        self._track_update_prompt_clicked(version, "dismissed")

    def _on_update_action_clicked(self) -> None:

        version = self._offered_update_version()
        landed = open_plugin_manager_or_marketplace()
        self._track_update_prompt_clicked(
            version, "plugin_manager" if landed else "marketplace_page")

    def apply_server_feature_switches(self) -> None:






        try:
            from ...core.server_dials import feature_enabled

            library_btn = getattr(self, "auto_library_btn", None)
            if library_btn is not None:
                library_btn.setVisible(feature_enabled("library"))
        except (RuntimeError, AttributeError):
            pass  # nosec B110
        self.refresh_update_recommendation()


def open_plugin_manager_or_marketplace() -> bool:




    from ..terralab_menu import open_plugin_manager_updates

    landed = open_plugin_manager_updates(fallback_url=_marketplace_url())
    if landed:
        _note_update_applies_on_reload()
    return landed


def _marketplace_url() -> str:
    from ...core.activation_manager import get_marketplace_url

    return get_marketplace_url()


def _note_update_applies_on_reload() -> None:

    try:
        from qgis.core import Qgis
        from qgis.utils import iface

        iface.messageBar().pushMessage(
            "AI Segmentation",
            tr("The update applies once QGIS reloads the plugin. Restart QGIS "
               "if the panel misbehaves after it."),
            level=Qgis.MessageLevel.Info,
            duration=dial_in_range(
                "tuning.notify.update_note_seconds", _UPDATE_NOTE_SECONDS,
                3, 60))
    except Exception:  # noqa: BLE001  # nosec B110
        pass
