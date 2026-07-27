"""Inert compatibility shim for the retired Refine-in-Manual handoff screen.

The separate handoff page (its own title, state card, click legend, per-state
buttons and the "back to Automatic review" footer) and the visible mode switch
to Manual are GONE. Fixing a detection with the on-device AI is now the Correct
step's in-place "Refine with AI" sub-state (see auto_correct_build.py and the
enter/leave_ai_reshape_state setters in auto_state.py); it never leaves the
review.

What survives here are thin no-op setters, because the reused SAM edit machine
(manual_handoff.py, manual_predict.py) still calls them as it opens, saves and
closes an object. They intentionally do nothing: the reshape sub-state is driven
by enter/leave_ai_reshape_state, not by these. The setup stubs keep the dock
build call sites (build.py) untouched, building nothing.
"""
from __future__ import annotations

from qgis.PyQt.QtWidgets import QWidget


class DockHandoffMixin:
    """No-op shim for the retired handoff screen (see the module docstring)."""

    # ------------------------------------------------------------------
    # Build stubs (called by build.py): build nothing, so the page is gone.
    # ------------------------------------------------------------------

    def _setup_handoff_header(self, layout) -> None:
        """Kept only so other modules that toggle ``refine_handoff_banner`` by
        name never hit a missing attribute; it is a hidden zero-height widget."""
        self.refine_handoff_banner = QWidget()
        self.refine_handoff_banner.setFixedHeight(0)
        self.refine_handoff_banner.setVisible(False)
        layout.addWidget(self.refine_handoff_banner)

    def _setup_handoff_state_card(self, layout) -> None:
        """The handoff state card is gone; reshape lives in the Correct card."""
        return

    def _setup_handoff_footer(self, layout) -> None:
        """The "back to Automatic review" footer is gone; reshape Done folds
        back in place without leaving the review."""
        return

    # ------------------------------------------------------------------
    # No-op setters the reused SAM edit machine still calls.
    # ------------------------------------------------------------------

    def set_refine_handoff_preparing(self, preparing: bool) -> None:
        return

    def update_handoff_progress(self, kept: int) -> None:
        return

    def set_handoff_selected(self, count: int) -> None:
        return

    def set_handoff_editing(self, editing: bool) -> None:
        return

    def note_handoff_shape_edited(self) -> None:
        return

    def note_handoff_shape_removed(self, count: int = 1) -> None:
        return

    def _reset_handoff_counters(self) -> None:
        return

    def end_refine_handoff(self, target_mode=None) -> None:
        """Retained for the project-teardown path (auto_zone), which may fire
        while a reshape is open: just leave the in-place reshape sub-state.
        No mode switch happens any more."""
        _ = target_mode
        try:
            self.leave_ai_reshape_state()
        except (RuntimeError, AttributeError):
            pass
