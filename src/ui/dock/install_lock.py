"""The review is held still while the local AI installs.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py); split
out so agents and humans work on one concern per file. Methods are plain
mixin members: widgets and signals live on the dock instance.

Clicking the AI fix without the on-device model installed starts a one-time
setup that runs for minutes. That setup used to run behind a review the user
could still walk around, and walking away broke it in three ways at once: the
banner lives on the Correct step, so any other step hid the only progress on
screen; the method switch and the ladder each drop the pending intent, so the
install finished with nobody left to hand the model to; and Finish tore the
review down under it.

So the setup takes the review for as long as it runs. The dials, the step
primary, Export, the escape links, the AI | Manual switch, the Add lane and
the mode switch all go dead, and the banner is the one live thing left. The
way out is its Cancel, which stops the install and gives the review back.
"""
from __future__ import annotations

from ...core.i18n import tr

# Everything that moves the user off this step, changes the method under the
# pending install, or ends the review. Named rather than walked, so a control
# added later is locked on purpose and not by accident.
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


class DockInstallLockMixin:
    """The review's on/off for a local-AI install that owns the panel."""

    def set_auto_review_installing(self, active: bool) -> None:
        """Reflect a local-AI install kicked off from the review.

        Shows the banner (progress fed by set_install_progress) and holds the
        review still around it. The plugin re-opens the fix or the Add lane by
        itself once the model is ready.
        """
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
        """True while an install from the review owns the panel."""
        return bool(getattr(self, "_auto_review_installing", False))

    def _apply_review_install_lock(self, active: bool) -> None:
        """Freeze or release the controls around the banner.

        Releasing restores the mode switch to what the review itself asks for
        (locked while a review is open), never to plain enabled: the install
        ends inside the review, and handing the mode switch back there would
        open the destructive door the review keeps shut.
        """
        for name in _INSTALL_LOCKED_WIDGETS:
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
