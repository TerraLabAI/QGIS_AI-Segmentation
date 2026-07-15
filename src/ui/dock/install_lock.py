


















from __future__ import annotations

from ...core.i18n import tr




_INSTALL_LOCKED_WIDGETS = (
    "auto_step_next_btn",
    "auto_export_btn",
    "auto_retry_btn",
    "auto_review_exit_btn",
    "auto_correct_method_switch",
    "auto_add_lane_card",
    "auto_add_lane_btn",
    "auto_add_lane_keep_btn",
    "auto_correct_select_card",
    "auto_shape_merge_btn",
    "auto_correct_undo_btn",
    "auto_correct_clear_btn",
    "mode_switch",
)





_OWNED_LOCKED_WIDGETS = frozenset({
    "auto_export_btn",
    "auto_shape_merge_btn",
})


class DockInstallLockMixin:


    def set_auto_review_installing(self, active: bool) -> None:






        active = bool(active)
        self._auto_review_installing = active
        try:
            self.auto_review_install_banner.setVisible(active)
            if active:
                self.auto_review_install_progress.setValue(0)
                self.auto_review_install_label.setText(tr(
                    "Setting up the on-device AI. This runs once and takes a "
                    "few minutes. The review waits here until it is done."))
        except (RuntimeError, AttributeError):

            pass
        self._apply_review_install_lock(active)

    def review_install_locked(self) -> bool:

        return bool(getattr(self, "_auto_review_installing", False))

    def _apply_review_install_lock(self, active: bool) -> None:







        for name in _INSTALL_LOCKED_WIDGETS:
            if not active and name in _OWNED_LOCKED_WIDGETS:
                continue
            widget = getattr(self, name, None)
            if widget is None:
                continue
            try:
                widget.setEnabled(not active)
            except (RuntimeError, AttributeError):
                continue
        try:
            if active:
                self.auto_step_next_btn.setToolTip(tr(
                    "Wait for the on-device AI to finish installing."))
            else:
                self.auto_step_next_btn.setToolTip("")
        except (RuntimeError, AttributeError):
            pass
        try:
            self._set_review_dials_locked(
                active, int(getattr(self, "_auto_review_step", 0)))
        except (RuntimeError, AttributeError):

            pass
        if not active:
            try:
                self.mode_switch.setEnabled(
                    not bool(getattr(self, "_auto_review_active", False)))
            except (RuntimeError, AttributeError):

                pass
            self._release_owned_install_locks()

    def _release_owned_install_locks(self) -> None:






        try:
            self.auto_export_btn.setEnabled(
                int(getattr(self, "_auto_review_visible_count", 0)) > 0)
        except (RuntimeError, AttributeError, TypeError, ValueError):

            pass
        try:
            self._set_bridge_merge_enabled(True)
            self._apply_merge_tile()
        except (RuntimeError, AttributeError):
            pass
