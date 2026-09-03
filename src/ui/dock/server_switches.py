"""Server feature switches and the update recommendation, applied to the dock.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py); split out
so agents and humans work on one concern per file. Methods are plain mixin
members: widgets and signals live on the dock instance.

Everything here fails open. When no configuration has arrived the dock looks
and behaves exactly as it ships, so an offline start loses nothing.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QSettings

from ...core.i18n import tr

UPDATE_TRIGGER_PLUGIN_REGISTRY = "plugin_registry"
UPDATE_TRIGGER_SERVED_LATEST = "served_latest_version"
UPDATE_TRIGGER_SERVED_MIN = "served_min_recommended"
# The version telemetry already announced, so the prompt is counted once per
# version per install. Outside the hint prefix on purpose: a guidance reset
# re-shows the card, it does not make the release new again.
_PROMPT_SHOWN_KEY = "AISegmentation/telemetry/update_prompt_shown_version"
_UNKNOWN_VERSION = "unknown"


def resolve_update_offer(installed_version: str) -> tuple[str | None, str]:
    """The version the server offers above this build, and what knew about it.

    Best source first: a served ``latest_version`` names the release, a served
    ``min_recommended_version`` only says this build is behind.
    """
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
    """Applies the server switches to the widgets they gate."""

    def _setup_update_recommendation(self):
        """Build the update banner. Hidden until the server asks for it."""
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
        except Exception:  # noqa: BLE001 -- guidance is best-effort
            return ""

    def _offered_update_version(self) -> str:
        """The version the banner is currently offering, or an empty string."""
        installed = self._installed_version()
        if not installed:
            return ""
        try:
            version, _trigger = resolve_update_offer(installed)
        except Exception:  # noqa: BLE001 -- guidance is best-effort
            return ""
        return version or ""

    def _should_show_update_recommendation(self) -> bool:
        """Always False. The served version no longer offers anything by itself.

        It used to: a served ``latest_version`` above this build put a banner
        up on its own. QGIS has usually not fetched the plugin repository at
        that point, so Update now opened a Plugin Manager whose Upgradeable tab
        was empty and the user could do nothing for ten minutes. The offer is
        now made by one surface only, the update card in about.py, and only on
        the installer's own verdict.

        The served version keeps its job: it is the hint that a background
        repository refresh is worth making (see _maybe_refresh_plugin_repository).
        The hint widget itself stays built so a guidance reset and the existing
        dismissal memory keep working.
        """
        return False

    def _update_banner_text(self, version: str) -> str:
        """The banner sentence for one offered version.

        Plain text throughout: the card renders as PlainText, so a served line
        is shown as written and can never open a tag.
        """
        from ...core.activation_manager import get_release_notes_line

        if not version or version == _UNKNOWN_VERSION:
            return tr("A newer version of AI Segmentation is available with the "
                      "latest fixes.")
        lines = [tr("AI Segmentation {version} is available.").format(version=version)]
        try:
            notes = get_release_notes_line()
        except Exception:  # noqa: BLE001 -- copy is best-effort
            notes = None
        if notes:
            lines.append(notes)
        return "\n".join(lines)

    def refresh_update_recommendation(self) -> None:
        """Show or hide the update banner for the configuration in hand."""
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
            pass  # nosec B110 -- the widget can be gone during teardown

    def _track_update_prompt_shown(self, version: str, trigger: str) -> None:
        """Count the prompt once per offered version per install."""
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
        except Exception:  # noqa: BLE001 -- telemetry is best-effort  # nosec B110
            pass

    def _track_update_prompt_clicked(self, version: str, action: str) -> None:
        try:
            from ...core import telemetry_events as ev
            from ...core.telemetry import track

            track(ev.PLUGIN_UPDATE_PROMPT_CLICKED, {
                "offered_version": version or _UNKNOWN_VERSION,
                "action": action,
            })
        except Exception:  # noqa: BLE001 -- telemetry is best-effort  # nosec B110
            pass

    def _on_update_hint_dismissed(self) -> None:
        """Remember the refusal for this version only, so the next one asks."""
        version = self._offered_update_version()
        try:
            from .guidance import HINT_UPDATE_RECOMMENDED, dismiss_hint_for_version

            if version:
                dismiss_hint_for_version(HINT_UPDATE_RECOMMENDED, version)
        except Exception:  # noqa: BLE001 -- a memory is best-effort  # nosec B110
            pass
        self._track_update_prompt_clicked(version, "dismissed")

    def _on_update_action_clicked(self) -> None:
        """Send the user to the Plugin Manager, and say what happens next."""
        version = self._offered_update_version()
        landed = open_plugin_manager_or_marketplace()
        self._track_update_prompt_clicked(
            version, "plugin_manager" if landed else "marketplace_page")

    def apply_server_feature_switches(self) -> None:
        """Apply every server switch the dock owns.

        Safe to call again: the dock calls it once at build time (a
        configuration left by an earlier session is already there) and again
        each time a fresh one lands.
        """
        try:
            from ...core.server_dials import feature_enabled

            library_btn = getattr(self, "auto_library_btn", None)
            if library_btn is not None:
                library_btn.setVisible(feature_enabled("library"))
        except (RuntimeError, AttributeError):
            pass  # nosec B110 -- the widget can be gone during teardown
        self.refresh_update_recommendation()


def open_plugin_manager_or_marketplace() -> bool:
    """Open the Plugin Manager, or the marketplace page when it will not open.

    True when the Plugin Manager took the user, False when the browser did.
    """
    from ..terralab_menu import open_plugin_manager_updates

    landed = open_plugin_manager_updates(fallback_url=_marketplace_url())
    if landed:
        _note_update_applies_on_reload()
    return landed


def _marketplace_url() -> str:
    from ...core.activation_manager import get_marketplace_url

    return get_marketplace_url()


def _note_update_applies_on_reload() -> None:
    """One calm line under the Plugin Manager, so nobody hunts for the change."""
    try:
        from qgis.core import Qgis
        from qgis.utils import iface

        iface.messageBar().pushMessage(
            "AI Segmentation",
            tr("The update applies once QGIS reloads the plugin. Restart QGIS "
               "if the panel misbehaves after it."),
            level=Qgis.MessageLevel.Info,
            duration=8)
    except Exception:  # noqa: BLE001 -- a note is best-effort  # nosec B110
        pass
