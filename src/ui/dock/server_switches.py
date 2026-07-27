"""Server feature switches and the update recommendation, applied to the dock.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py); split out
so agents and humans work on one concern per file. Methods are plain mixin
members: widgets and signals live on the dock instance.

Everything here fails open. When no configuration has arrived the dock looks
and behaves exactly as it ships, so an offline start loses nothing.
"""
from __future__ import annotations

from ...core.i18n import tr


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
        self.update_recommended_hint.action.connect(self._on_open_plugin_manager)
        self.update_recommended_hint.setVisible(False)
        self.main_layout.addWidget(self.update_recommended_hint)

    def _should_show_update_recommendation(self) -> bool:
        """Whether this build is below the version the server recommends.

        False whenever anything is unknown: no version on either side, or the
        plugin repository already shows its own update notice, which says the
        same thing with a real version number.
        """
        try:
            repo_notice = getattr(self, "update_notification_widget", None)
            if repo_notice is not None and repo_notice.isVisible():
                return False
            from ...core.activation_manager import is_update_recommended
            from ...core.request_context import plugin_version

            version = plugin_version()
            return bool(version) and is_update_recommended(version)
        except Exception:  # noqa: BLE001 -- guidance is best-effort  # nosec B110
            return False

    def refresh_update_recommendation(self) -> None:
        """Show or hide the update banner for the configuration in hand."""
        hint = getattr(self, "update_recommended_hint", None)
        if hint is None:
            return
        try:
            from .guidance import HINT_UPDATE_RECOMMENDED, is_hint_dismissed

            show = (
                not is_hint_dismissed(HINT_UPDATE_RECOMMENDED) and self._should_show_update_recommendation()
            )
            hint.setVisible(show)
        except (RuntimeError, AttributeError):
            pass  # nosec B110 -- the widget can be gone during teardown

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
