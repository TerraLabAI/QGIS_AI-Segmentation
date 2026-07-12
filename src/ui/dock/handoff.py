"""Refine-in-Manual handoff UI: header, morphing state card, and drivers.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.

The handoff is a TEMPORARY stop: every detection from the Automatic run
is already part of the result, the user only drops in to fix a few
shapes or remove false positives, then returns to the review to export.
There is deliberately NO keep/validate concept here (nothing to tick
off, no progress to complete): the page is click a detection, then
Edit shape or Remove, and one unmissable exit back to the review.
Actions are buttons; keyboard shortcuts live in tooltips only.
"""
from __future__ import annotations


from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


from ...core.i18n import tr
from .styles import (
    BRAND_RED,
    _BTN_BLUE_OUTLINE,
    _BTN_GHOST,
    _BTN_GREEN,
    _BTN_RED_OUTLINE,
    _CARD_MARGINS,
    _CARD_QSS,
    _RECAP_CARD_QSS,
    _micro_header,
    _msg_label_qss,
    _msg_text,
    _sign_badge,
)
from .widgets import Mode


class DockHandoffMixin:
    """Refine-in-Manual handoff header + state card and their state drivers."""

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _setup_handoff_header(self, layout) -> None:
        """Kept for API stability: other modules toggle
        ``refine_handoff_banner`` by name, but the page no longer has a
        separate header (naked text outside a card is banned). The whole
        handoff lives in the single state card; this is an empty stub."""
        self.refine_handoff_banner = QWidget()
        self.refine_handoff_banner.setFixedHeight(0)
        self.refine_handoff_banner.setVisible(False)
        layout.addWidget(self.refine_handoff_banner)

    def _setup_handoff_state_card(self, layout) -> None:
        """THE one framed block of the handoff page: a standard card holding
        the page title, the current instruction, the editing click legend and
        the per-state action buttons (hidden, never greyed). No colored edge
        ornament (Yvann's call 2026-07-11). Buttons fire the same plugin
        handlers the keyboard shortcuts use; the shortcuts live in tooltips."""
        self.handoff_state_card = QWidget()
        self.handoff_state_card.setObjectName("handoffStateCard")
        self.handoff_state_card.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.handoff_state_card.setStyleSheet(
            _CARD_QSS.format(name="handoffStateCard") + "QLabel { background: transparent; border: none; }"
        )
        col = QVBoxLayout(self.handoff_state_card)
        col.setContentsMargins(*_CARD_MARGINS)
        col.setSpacing(6)

        # Page title: the card's constant first line, so the page always
        # names itself inside the frame (no naked header above).
        self._handoff_page_title = QLabel(tr("Edit your detections"))
        self._handoff_page_title.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: palette(text);")
        col.addWidget(self._handoff_page_title)

        # One instruction line per state (armed while the map waits for a
        # click, neutral while the model loads). A taxonomy message label, so
        # it never reads as bare floating text.
        self._handoff_instruction = QLabel("")
        self._handoff_instruction.setWordWrap(True)
        col.addWidget(self._handoff_instruction)

        # Editing header (quiet uppercase label) leads the actions while a
        # shape is open; the plain title below carries the selected state.
        self._handoff_micro_header = _micro_header(tr("Editing this shape"))
        col.addWidget(self._handoff_micro_header)

        self._handoff_state_title = QLabel("")
        self._handoff_state_title.setWordWrap(True)
        self._handoff_state_title.setStyleSheet(
            "font-weight: bold; font-size: 12px; color: palette(text);")
        col.addWidget(self._handoff_state_title)

        self._handoff_state_sub = QLabel("")
        self._handoff_state_sub.setWordWrap(True)
        self._handoff_state_sub.setStyleSheet(
            "font-size: 11px; color: rgba(128,128,128,0.95);")
        col.addWidget(self._handoff_state_sub)

        # Editing click legend: the one place a mouse gesture must be spelled
        # out, because left/right click are the whole editing model.
        self._handoff_click_legend = QWidget()
        legend_col = QVBoxLayout(self._handoff_click_legend)
        legend_col.setContentsMargins(0, 0, 0, 0)
        legend_col.setSpacing(4)
        for badge_text, badge_color, row_text in (
            ("+", "rgba(67,160,71,0.9)", tr("Left-click adds what you click to the shape")),
            ("−", BRAND_RED, tr("Right-click removes it from the shape")),
        ):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(7)
            row.addWidget(_sign_badge(badge_text, badge_color))
            lbl = QLabel(row_text)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("font-size: 11px; color: palette(text);")
            row.addWidget(lbl, 1)
            legend_col.addLayout(row)
        col.addWidget(self._handoff_click_legend)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 2, 0, 0)
        btn_row.setSpacing(6)
        # Blue outline = "still editing / temporary", the quiet secondary next
        # to the page's one filled primary (Done, back to review).
        self.handoff_edit_btn = QPushButton(tr("Edit shape"))
        self.handoff_edit_btn.setStyleSheet(_BTN_BLUE_OUTLINE)
        self.handoff_edit_btn.setToolTip(
            tr("Opens the shape so clicks can extend or trim it. "
               "Key: E, or double-click it on the map"))
        self.handoff_save_btn = QPushButton(tr("Save shape"))
        self.handoff_save_btn.setStyleSheet(_BTN_GREEN)
        self.handoff_save_btn.setToolTip(
            tr("Saves this shape and closes the edit. Key: S"))
        self.handoff_undo_btn = QPushButton(tr("Undo click"))
        self.handoff_undo_btn.setStyleSheet(_BTN_GHOST)
        self.handoff_undo_btn.setToolTip(
            tr("Undoes the last change to this shape. Key: Ctrl+Z"))
        self.handoff_delete_btn = QPushButton(tr("Remove"))
        self.handoff_delete_btn.setStyleSheet(_BTN_RED_OUTLINE)
        self.handoff_delete_btn.setToolTip(
            tr("Removes it from the results. Key: Delete (Ctrl+Z restores it)"))
        for btn in (self.handoff_edit_btn, self.handoff_save_btn,
                    self.handoff_undo_btn, self.handoff_delete_btn):
            btn.setMinimumHeight(30)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.handoff_edit_btn.clicked.connect(self.handoff_edit_requested.emit)
        self.handoff_save_btn.clicked.connect(self.save_polygon_requested.emit)
        self.handoff_undo_btn.clicked.connect(self.undo_requested.emit)
        self.handoff_delete_btn.clicked.connect(self.handoff_delete_requested.emit)
        btn_row.addWidget(self.handoff_edit_btn, 1)
        btn_row.addWidget(self.handoff_save_btn, 1)
        btn_row.addWidget(self.handoff_undo_btn, 1)
        btn_row.addWidget(self.handoff_delete_btn, 1)
        self._handoff_btn_row = QWidget()
        self._handoff_btn_row.setLayout(btn_row)
        col.addWidget(self._handoff_btn_row)

        self.handoff_state_card.setVisible(False)
        layout.addWidget(self.handoff_state_card)

    def _setup_handoff_footer(self, layout) -> None:
        """The single, unmissable exit: a full-width primary button back to
        the Automatic review. It reads as the page's 'Done' so the handoff
        never feels like a place you could get stuck in. Kept clickable even
        while the local model is still loading (leaving needs no model)."""
        # Names the destination in full ("back to review" alone did not say
        # WHICH review, so the return to Automatic read as unclear).
        self.back_to_review_btn = QPushButton(
            tr("Done, back to Automatic review"))
        self.back_to_review_btn.setStyleSheet(_BTN_GREEN)
        self.back_to_review_btn.setMinimumHeight(36)
        self.back_to_review_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.back_to_review_btn.setToolTip(
            tr("Returns to the Automatic review with your edits. "
               "The export happens there."))
        self.back_to_review_btn.clicked.connect(self.back_to_review_requested.emit)
        self.back_to_review_btn.setVisible(False)
        layout.addWidget(self.back_to_review_btn)

        # One quiet instrument-voice line under the exit: the measured tally of
        # what this handoff changed. A recap card (never naked text); shown
        # only when there is something to report (one info per state).
        self._handoff_footer_recap = QLabel("")
        self._handoff_footer_recap.setWordWrap(True)
        self._handoff_footer_recap.setStyleSheet(_RECAP_CARD_QSS)
        self._handoff_footer_recap.setVisible(False)
        layout.addWidget(self._handoff_footer_recap)

    # ------------------------------------------------------------------
    # Enter / leave the handoff
    # ------------------------------------------------------------------

    def begin_refine_handoff(self, seed_total: int = 0) -> None:
        """Enter Manual mode to refine the Automatic results, preserving state.

        Drives the mode change by emitting mode_changed directly instead of going
        through the segmented control's mode_selected: a Manual session IS active
        during the round trip, so _on_mode_selected's run-active guard would
        otherwise refuse the switch. The plugin's _on_mode_changed handler sees the
        handoff flag and skips its usual destructive reset. ``seed_total`` is
        kept for API stability (the header no longer shows a counter).
        """
        self._refine_handoff = True
        self._handoff_seed_total = int(seed_total)
        self._handoff_selected = 0
        self._handoff_editing = False
        self._handoff_preparing = False
        self._reset_handoff_counters()
        self.refine_handoff_banner.setVisible(True)
        self.back_to_review_btn.setVisible(True)
        # The shared refine panel becomes the open shape's "Shape settings",
        # collapsed by default and shown only while an edit is open.
        self.set_refine_panel_title(tr("Shape settings"))
        self.set_refine_collapsed(True)
        self.mode_switch.setEnabled(False)
        self.mode_switch.setToolTip(
            tr("Go back to the Automatic review to switch modes."))
        self._mode = Mode.INTERACTIVE
        self.mode_switch.blockSignals(True)
        self.mode_switch.set_mode(Mode.INTERACTIVE)
        self.mode_switch.blockSignals(False)
        self.mode_changed.emit(Mode.INTERACTIVE)
        self._update_full_ui()

    def end_refine_handoff(self, target_mode: Mode) -> None:
        """Leave the refine handoff, returning to target_mode (usually Automatic)."""
        self._refine_handoff = False
        self._handoff_selected = 0
        self._handoff_editing = False
        self.set_refine_handoff_preparing(False)
        self._reset_handoff_counters()
        self.refine_handoff_banner.setVisible(False)
        self.back_to_review_btn.setVisible(False)
        self.handoff_state_card.setVisible(False)
        # Restore the base-Manual refine panel (its own title, expanded).
        self.set_refine_panel_title(tr("Refine selection"))
        self.set_refine_collapsed(False)
        self.mode_switch.setEnabled(True)
        self.mode_switch.setToolTip("")
        self._mode = target_mode
        self.mode_switch.blockSignals(True)
        self.mode_switch.set_mode(target_mode)
        self.mode_switch.blockSignals(False)
        self.mode_changed.emit(target_mode)
        self._update_full_ui()

    def set_refine_handoff_preparing(self, preparing: bool) -> None:
        """Reflect that the local model is still loading after a Refine click.

        The predictor loads asynchronously; while it comes up there is nothing
        to edit yet, so the header says so and the state card stays hidden.
        The Done button stays clickable (it only needs
        _restore_auto_review_after_handoff, no predictor), so a stuck or slow
        model load can never trap the user in the handoff.
        """
        self._handoff_preparing = bool(preparing)
        try:
            self._update_handoff_card()
        except (RuntimeError, AttributeError):
            pass

    # ------------------------------------------------------------------
    # State drivers (called by the plugin)
    # ------------------------------------------------------------------

    def update_handoff_progress(self, kept: int) -> None:
        """Kept for API stability: the handoff no longer surfaces a kept
        counter (every detection on this page is already part of the result)."""
        self._handoff_kept = int(kept)

    def set_handoff_selected(self, count: int) -> None:
        """Track how many detections are selected in the handoff review; the
        state card swaps to the "N selected" actions when > 0."""
        self._handoff_selected = int(count)
        self._update_instructions()
        self._update_refine_panel_visibility()

    def _reset_handoff_counters(self) -> None:
        """Zero the edited/removed tallies and hide their footer recap. Called
        on entering and leaving a handoff (one clean slate per visit)."""
        self._handoff_edited_count = 0
        self._handoff_removed_count = 0
        self._refresh_handoff_footer_recap()

    def note_handoff_shape_edited(self) -> None:
        """One shape was edited and saved in the handoff; bump the footer tally."""
        self._handoff_edited_count = getattr(self, "_handoff_edited_count", 0) + 1
        self._refresh_handoff_footer_recap()

    def note_handoff_shape_removed(self, count: int = 1) -> None:
        """``count`` shapes were removed in the handoff; bump the footer tally."""
        self._handoff_removed_count = (
            getattr(self, "_handoff_removed_count", 0) + max(0, int(count)))
        self._refresh_handoff_footer_recap()

    def _refresh_handoff_footer_recap(self) -> None:
        """Render the quiet 'N shapes edited . M removed' line under the exit,
        shown only when there is something to report."""
        recap = getattr(self, "_handoff_footer_recap", None)
        if recap is None:
            return
        edited = getattr(self, "_handoff_edited_count", 0)
        removed = getattr(self, "_handoff_removed_count", 0)
        if not self._refine_handoff or getattr(self, "_handoff_editing", False) or (edited <= 0 and removed <= 0):
            # One info per state: while a shape is open the card carries the
            # editing actions, so the tally waits until the edit closes.
            recap.setVisible(False)
            return
        parts = []
        if edited > 0:
            parts.append(
                tr("1 shape edited") if edited == 1
                else tr("{n} shapes edited").format(n=edited))
        if removed > 0:
            parts.append(
                tr("1 removed") if removed == 1
                else tr("{n} removed").format(n=removed))
        recap.setText(" · ".join(parts))
        recap.setVisible(True)

    def set_handoff_editing(self, editing: bool) -> None:
        """Track whether a detection is OPEN for editing in the handoff. The
        edit session is geometry-based (no SAM prompt points), so point counts
        can no longer signal the editing state to the state card."""
        self._handoff_editing = bool(editing)
        self._update_instructions()
        self._update_refine_panel_visibility()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _update_handoff_card(self) -> None:
        """Render the single handoff card from the current sub-state (the
        card is the whole page frame: it also carries the preparing state)."""
        card = getattr(self, "handoff_state_card", None)
        if card is None:
            return
        if not self._refine_handoff:
            card.setVisible(False)
            return

        editing = getattr(self, "_handoff_editing", False)
        selected = getattr(self, "_handoff_selected", 0)
        instruction = None  # (kind, text) taxonomy message, one per state
        show_legend = False
        show_edit = show_save = show_undo = show_delete = False
        self._handoff_state_title.setText("")
        self._handoff_state_sub.setText("")

        if getattr(self, "_handoff_preparing", False):
            instruction = ("neutral",
                           tr("Preparing Manual mode, loading the local model..."))
            editing = False
        elif editing:
            # The micro-header ("EDITING THIS SHAPE") leads the click legend
            # and the actions; nothing else competes.
            show_legend = True
            show_save = show_undo = show_delete = True
        elif selected > 0:
            self._handoff_state_title.setText(
                tr("1 detection selected") if selected == 1
                else tr("{n} detections selected").format(n=selected))
            self._handoff_state_sub.setText(
                tr("Click an empty spot to deselect."))
            show_edit = selected == 1
            show_delete = True
        else:
            # Resting: the map tool waits for a click, so the one instruction
            # is the armed message.
            instruction = ("armed", tr("Click a detection on the map"))

        if instruction is not None:
            kind, text = instruction
            self._handoff_instruction.setStyleSheet(_msg_label_qss(kind))
            self._handoff_instruction.setText(_msg_text(kind, text))
        self._handoff_instruction.setVisible(instruction is not None)
        self._handoff_micro_header.setVisible(editing)
        self._handoff_state_title.setVisible(bool(self._handoff_state_title.text()))
        self._handoff_state_sub.setVisible(bool(self._handoff_state_sub.text()))
        self._handoff_click_legend.setVisible(show_legend)
        self.handoff_edit_btn.setVisible(show_edit)
        self.handoff_save_btn.setVisible(show_save)
        self.handoff_undo_btn.setVisible(show_undo)
        self.handoff_delete_btn.setVisible(show_delete)
        self._handoff_btn_row.setVisible(
            show_edit or show_save or show_undo or show_delete)
        # While a shape is open, Save shape is the screen's one filled
        # primary: the green exit steps aside (Esc/Save close the edit first)
        # so two green CTAs never compete.
        try:
            self.back_to_review_btn.setVisible(not editing)
        except (RuntimeError, AttributeError):
            pass
        self._refresh_handoff_footer_recap()
        card.setVisible(True)
