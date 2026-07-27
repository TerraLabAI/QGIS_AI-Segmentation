"""Automatic post-run review panel construction: the linear 3-step review.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out of auto_build.py so agents and humans work on one concern per
file. Methods are plain mixin members: widgets/signals live on the dock
instance.

Layout: ONE review card (single instrument surface) holding the title, the
step-dials row (Keep / Correct / Shapes), then a step stack whose pages are
Keep (this file), Correct (auto_correct_build.py) and Shapes
(auto_review_steps.py). Below the card: the dynamic step primary, the Export
primary (Shapes step only) and the quiet links row. The View-as block is NOT
in the card: it is built by _setup_auto_review_view_block and pinned to the
dock bottom, just above the footer credits row (about.py). Store-only setters
live in auto_state.py.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
from .auto_correct_build import _REVIEW_HEADING_NOTE_QSS, _REVIEW_HEADING_QSS
from .guidance import (
    BLUE_TINT,
    HINT_REVIEW_CONFIDENCE,
    DismissibleHint,
)
from .styles import (
    _BTN_GREEN,
    _BTN_LINK_MUTED,
    _CARD_MARGINS,
    _CARD_QSS,
    _COMBO_THEME_QSS,
    _REVIEW_CONF_MAX,
    _REVIEW_CONF_MIN,
    _REVIEW_CONF_SPIN_MIN,
    _REVIEW_CONF_STEP,
    _SLIDER_QSS,
    ERROR_TEXT,
    _card_divider,
    _step_dial,
)

# Swatch colours for the Display-colors legend, each SAMPLED FROM THE REAL
# renderer (auto_results): the review blue fill, the outline-mode red stroke,
# the Viridis ramp ends of the confidence heatmap, and four color_hsla(h,78,55)
# hues from the random-per-object expression. Map identity shown inline, not a
# new UI tint: the dot next to the words is exactly the colour on the canvas.
_LEGEND_RANDOM_DOTS = ("#e65133", "#b3e633", "#33e67a", "#3389e6")

# Two-stage retry guard (D11): with a non-empty correction journal the muted
# escape link swaps to this error-warm confirm look on the first click; the
# state machine is plugin-side, the dock only swaps the look
# (set_retry_confirm_pending).
_BTN_LINK_CONFIRM = (
    "QPushButton { background: transparent; border: none;"  # ui-ok: confirm-stage variant of _BTN_LINK_MUTED
    f" color: {ERROR_TEXT}; font-size: 11px; padding: 4px 8px;"
    " text-decoration: underline; }"
)


class _CurrentPageStack(QWidget):
    """Page stack that sizes to the CURRENT page only (QStackedWidget API).

    A real QStackedWidget always sizes to its TALLEST page: sizeHint takes
    the max even over Ignored-policy pages, and worse, the parent card's
    layout sizes by heightForWidth (its labels word-wrap), for which
    QWidgetItem bypasses every widget-level virtual and asks the internal
    QStackedLayout directly, again the max over ALL pages. So a short step
    (Keep) inherited the Shapes page's wrapped height as
    dead space between its content and the primary button. Here the layout
    only ever CONTAINS the current page (swapped on setCurrentIndex), so
    every size channel naturally reports the current page alone."""

    currentChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pages: list[QWidget] = []
        self._current = -1
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

    def addWidget(self, page: QWidget) -> int:
        self._pages.append(page)
        page.setParent(self)
        page.hide()
        if self._current < 0:
            self.setCurrentIndex(0)
        return len(self._pages) - 1

    def count(self) -> int:
        return len(self._pages)

    def widget(self, index: int):
        if 0 <= index < len(self._pages):
            return self._pages[index]
        return None

    def currentIndex(self) -> int:
        return self._current

    def currentWidget(self):
        return self.widget(self._current)

    def setCurrentIndex(self, index: int) -> None:
        if index == self._current or not 0 <= index < len(self._pages):
            return
        lay = self.layout()
        old = self.currentWidget()
        if old is not None:
            lay.removeWidget(old)
            old.hide()
        self._current = index
        page = self._pages[index]
        lay.addWidget(page)
        page.show()
        self.updateGeometry()
        self.currentChanged.emit(index)


def _legend_dots(colors, glyph: str = "●") -> str:
    """Inline swatch run for the legend line (rich-text colored dots)."""
    return "".join(
        f'<span style="color:{c};">{glyph}</span>'
        for c in colors
    )


def display_legend_html(mode: str) -> str:
    """Legend line under the View-detections-as combo: swatch dots in the real
    renderer colours + one short line saying what they mean. The dots carry
    the colour so the words never have to name it. Shared with the plugin
    side, which re-sets the line on every mode switch."""
    if mode == "outline":
        return "{d}&nbsp; {t}".format(
            d=_legend_dots(("#e31a1c",), glyph="○"),
            t=tr("Outlines only - check boundaries against the imagery"))
    if mode == "confidence":
        return "{y}&nbsp;{cf}&nbsp; &middot; &nbsp;{p}&nbsp;{un}".format(
            y=_legend_dots(("#fde725",)), cf=tr("confident"),
            p=_legend_dots(("#440154",)), un=tr("uncertain"))
    if mode == "random":
        return "{d}&nbsp; {t}".format(
            d=_legend_dots(_LEGEND_RANDOM_DOTS),
            t=tr("One color per object - check neighbors are separated"))
    return "{d}&nbsp; {t}".format(
        d=_legend_dots(("#0078ff",)),
        t=tr("Detected object"))


def _export_btn_label(n: int) -> str:
    """Export-button label with a singular branch, so a single detection reads
    'Export 1 polygon' and never the ungrammatical 'Export 1 polygons'."""
    if n == 1:
        return tr("Export 1 polygon")
    return tr("Export {n} polygons").format(n=n)


def _dial_label_qss(active: bool) -> str:
    """10px label under a step dial: the active step's label is bolder and in
    the full text colour, the others stay muted."""
    if active:
        return ("font-size: 10px; font-weight: 600; color: palette(text);"
                " background: transparent; border: none;")
    return ("font-size: 10px; color: rgba(128,128,128,0.95);"
            " background: transparent; border: none;")


class _StepDialColumn(QWidget):
    """One clickable dial column of the review ladder.

    The dial row doubles as free navigation: click Keep to re-tune the
    cutoff, come back to Correct, jump ahead to Shapes. Navigability is driven
    by _set_review_dials_locked, so the pointer cursor never invites a click
    the plugin would refuse (only the already-current step is non-navigable)."""

    clicked = pyqtSignal(int)

    def __init__(self, step: int, parent=None):
        super().__init__(parent)
        self._step = int(step)
        self._navigable = False

    def set_navigable(self, on: bool) -> None:
        self._navigable = bool(on)
        self.setCursor(Qt.CursorShape.PointingHandCursor if self._navigable
                       else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt virtual
        # Qt6 mouse events carry position(); Qt5 carries localPos() (same
        # QPointF contract). Resolved BY NAME so the static Qt6 checker never
        # sees a flat deprecated access on either branch.
        getter = (getattr(event, "position", None) or getattr(event, "localPos", None))
        hit = getter is not None and self._navigable and event.button() == Qt.MouseButton.LeftButton
        if hit and self.rect().contains(getter().toPoint()):
            self.clicked.emit(self._step)
        super().mouseReleaseEvent(event)


class DockAutoReviewBuildMixin:
    """Automatic post-run review panel construction (linear 3-step review)."""

    def _setup_auto_review_panel(self, parent_layout):
        """Post-run review panel (hidden until a run finishes successfully).

        ONE titled review card (single instrument surface: title, step dials,
        step stack), then the dynamic step primary, the Export primary (Shapes
        step only) and the quiet links row. The View-as block lives at the dock
        bottom, not here (_setup_auto_review_view_block)."""
        self.auto_review_panel = QWidget()
        self.auto_review_panel.setVisible(False)
        _review_layout = QVBoxLayout(self.auto_review_panel)
        _review_layout.setContentsMargins(0, 0, 0, 0)
        # 8px between the zones (review card / primaries / links line) keeps
        # a calm, clearly hierarchised review area.
        _review_layout.setSpacing(8)

        # Linear-review state read by the auto_state setters. Initialized
        # with the widgets they describe (the dock __init__ stays signals +
        # assembly only).
        self._auto_review_step = 0
        self._auto_zero_entry = False

        # THE review card: the whole edit phase in one card. The page names
        # itself inside this first card (no frameless header); the checkbox
        # indicator QSS rides the card stylesheet so the Refine checkboxes
        # inside stay visible on dark themes.
        from .widgets import checkbox_indicator_qss
        _card = QWidget()
        _card.setObjectName("autoReviewCard")
        _card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        _card_qss = _CARD_QSS.format(name="autoReviewCard")
        _card_qss += "QLabel { background: transparent; border: none; }"
        _card_qss += _SLIDER_QSS
        _card_qss += checkbox_indicator_qss(self)
        _card.setStyleSheet(_card_qss)
        _card_layout = QVBoxLayout(_card)
        _card_layout.setContentsMargins(*_CARD_MARGINS)
        _card_layout.setSpacing(6)

        # No card title: the step dials below name the review, and each step
        # page heads itself (_review_step_heading), so a static "Review
        # detections" line only spent height repeating them.
        # Step dials row: three equal-width columns, the 10px label BELOW
        # each dial (i18n width rule). Clickable for free navigation; only the
        # already-current step is non-navigable. Kept as a handle so the QGIS
        # bridge can hide the ladder while leaving the card (and the View-as
        # control) in place.
        self._auto_review_dials_row = self._build_review_dials()
        _card_layout.addWidget(self._auto_review_dials_row)
        # Breathing room between the ladder and the step page: the dial labels
        # otherwise sit flush on the next line and the two zones read as one
        # block.
        _card_layout.addSpacing(8)

        # Step stack, in build order: 0 Keep / 1 Correct / 2 Shapes.
        # Mirrors the auto_steps pattern; navigation happens only through the
        # auto_state setters, driven by the plugin.
        from .auto_review_steps import build_shapes_page
        self.auto_review_step_stack = _CurrentPageStack()
        self.auto_review_step_stack.addWidget(self._build_auto_keep_page())
        self.auto_review_step_stack.addWidget(self._build_auto_correct_page())
        self.auto_review_step_stack.addWidget(build_shapes_page(self))
        self.auto_review_step_stack.currentChanged.connect(
            self._sync_review_stack_height)
        self._sync_review_stack_height()
        # Never let the stack absorb surplus vertical space: a QStackedWidget
        # defaults to Expanding, and any surplus the page layout hands the
        # card would balloon the stack into dead space between the step
        # content and the primary button.
        self.auto_review_step_stack.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        _card_layout.addWidget(self.auto_review_step_stack)

        # Kept as an instance handle so the QGIS-bridge banner can hide the
        # whole review card while the user edits in QGIS's own tools.
        self._auto_review_card = _card
        _review_layout.addWidget(_card)

        # The two green primaries belong to the review flow: directly under the
        # card, above the quiet links. They are also what a user who does not
        # want to correct anything reaches for, so _keep_review_primary_in_view
        # scrolls them into view on every step change rather than pinning them
        # somewhere else on the dock.

        # Dynamic step primary: ONE green "Next: ..." button whose label the
        # setters swap per step (set_auto_review_step). Hidden on the Shapes
        # step (Export is that step's primary) and on the zero-detection empty
        # state, where there is nothing to advance to.
        self.auto_step_next_btn = QPushButton("")
        self.auto_step_next_btn.setStyleSheet(_BTN_GREEN)
        self.auto_step_next_btn.setMinimumHeight(40)
        self.auto_step_next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_step_next_btn.clicked.connect(
            self._on_auto_step_next_clicked)
        _review_layout.addWidget(self.auto_step_next_btn)

        # Export: the Shapes step's primary, final commit. Names the outcome
        # (a saved layer) instead of the vague "Finish"; taller than every
        # other control so the finish line is unmistakable.
        self.auto_export_btn = QPushButton(_export_btn_label(0))
        self.auto_export_btn.setStyleSheet(_BTN_GREEN)
        self.auto_export_btn.setMinimumHeight(44)
        self.auto_export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_export_btn.clicked.connect(self.auto_export_requested.emit)
        self.auto_export_btn.setVisible(False)
        _review_layout.addWidget(self.auto_export_btn)

        # Quiet links row: "Re-run the whole zone" + Exit on every step. Retry
        # keeps the zone, references and settings (non-destructive re-run, new
        # credits); Exit discards. The dials handle back-navigation, so there
        # is no back-link.
        self.auto_retry_btn = QPushButton("↻  " + tr("Re-run the whole zone"))
        self.auto_retry_btn.setStyleSheet(_BTN_LINK_MUTED)
        self.auto_retry_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_retry_btn.setToolTip(tr(
            "Go back to your zone, references and settings, then detect the "
            "whole zone again. Nothing is saved."))
        self.auto_retry_btn.clicked.connect(self.auto_retry_requested.emit)
        self.auto_review_exit_btn = QPushButton(tr("Exit"))
        self.auto_review_exit_btn.setStyleSheet(_BTN_LINK_MUTED)
        self.auto_review_exit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_review_exit_btn.clicked.connect(
            self.auto_review_exit_requested.emit)
        _review_actions_row = QHBoxLayout()
        _review_actions_row.setContentsMargins(0, 0, 0, 0)
        _review_actions_row.setSpacing(2)
        self._auto_review_links_sep = QLabel("·")
        self._auto_review_links_sep.setStyleSheet(
            "font-size: 11px; color: rgba(128,128,128,0.9);"
            " background: transparent; border: none;")
        _review_actions_row.addStretch(1)
        _review_actions_row.addWidget(self.auto_retry_btn)
        _review_actions_row.addWidget(self._auto_review_links_sep)
        _review_actions_row.addWidget(self.auto_review_exit_btn)
        _review_actions_row.addStretch(1)
        _review_layout.addLayout(_review_actions_row)

        parent_layout.addWidget(self.auto_review_panel)

    def _setup_auto_review_view_block(self, parent_layout):
        """View-detections-as block, pinned to the DOCK BOTTOM.

        It is a way to look at the results, not a step of the review, so it
        sits under the whole flow (just above the footer credits row, added by
        _setup_about_section) instead of leading the review card. Always there
        while a review is live, on every step, and never the first thing the
        eye lands on.

        Visibility follows the review: set_auto_review_active shows and hides
        it, and the teardown paths that skip that call hide it themselves."""
        self.auto_review_view_row = QWidget()
        self.auto_review_view_row.setObjectName("autoReviewViewBlock")
        self.auto_review_view_row.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.auto_review_view_row.setStyleSheet(
            _CARD_QSS.format(name="autoReviewViewBlock") + "QLabel { background: transparent; border: none; }")
        self.auto_review_view_row.setVisible(False)
        _view_col = QVBoxLayout(self.auto_review_view_row)
        _view_col.setContentsMargins(*_CARD_MARGINS)
        _view_col.setSpacing(2)
        _display_row = QHBoxLayout()
        _display_lbl = QLabel(tr("View detections as:"))
        _display_lbl.setStyleSheet("font-size: 11px;")
        _display_row.addWidget(_display_lbl)
        _display_row.addStretch()
        self.auto_display_combo = _build_display_combo(self)
        _display_row.addWidget(self.auto_display_combo)
        _view_col.addLayout(_display_row)

        self.auto_display_legend = QLabel(display_legend_html("random"))
        self.auto_display_legend.setWordWrap(True)
        self.auto_display_legend.setTextFormat(Qt.TextFormat.RichText)
        self.auto_display_legend.setStyleSheet(
            "font-size: 11px; color: rgba(128,128,128,0.95);")
        _view_col.addWidget(self.auto_display_legend)
        parent_layout.addWidget(self.auto_review_view_row)

    # -- Keep page (step 0) --------------------------------------------------

    def _build_auto_keep_page(self) -> QWidget:
        """Keep page: everything that decides WHICH objects survive, and
        nothing that changes how they look. Confidence leads, the two size
        filters follow. All three re-filter the stored objects live (the run
        keeps every mask above a low recall floor), so no re-detection and no
        credits. Confidence fires on release (not every tick) so the re-merge
        runs once per adjustment.

        The step has no heading of its own: the dial above already names it,
        and each group carries its own title in the shared heading style. What
        the confidence number MEANS is the part that was missing, and it lives
        in the tip under the slider."""
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        # This label IS the step's head, so it wears the shared heading style
        # and lines up with the titles on the other two steps.
        _conf_hdr = QHBoxLayout()
        _conf_review_lbl = QLabel(tr("Confidence"))
        _conf_review_lbl.setStyleSheet(_REVIEW_HEADING_QSS)
        _conf_hdr.addWidget(_conf_review_lbl)
        _conf_hdr.addStretch()
        # Editable number so the user can dial an exact cutoff instead of having
        # to nudge the slider. Slider and spinbox stay in sync; either one drives
        # the (debounced) re-filter.
        self.auto_review_confidence_spin = QSpinBox()
        self.auto_review_confidence_spin.setRange(
            _REVIEW_CONF_SPIN_MIN, _REVIEW_CONF_MAX)
        # Free 1% precision in the number box: the spinbox is the exact-cutoff
        # control; only the slider snaps to _REVIEW_CONF_STEP.
        self.auto_review_confidence_spin.setSingleStep(1)
        self.auto_review_confidence_spin.setValue(30)
        self.auto_review_confidence_spin.setSuffix("%")
        self.auto_review_confidence_spin.setMinimumWidth(62)
        self.auto_review_confidence_spin.setMaximumWidth(78)
        _conf_hdr.addWidget(self.auto_review_confidence_spin)
        lay.addLayout(_conf_hdr)

        # Live readout of the cutoff, INSIDE the group it describes: ONE compact
        # line ("✓ 158 of 352 shown · 194 below 65%", built by
        # _format_auto_review_count) right under the Confidence header, so the
        # counts sit with the control that changes them and update under the
        # eye during a drag.
        self._auto_review_count_label = QLabel("")
        self._auto_review_count_label.setWordWrap(True)
        self._auto_review_count_label.setStyleSheet(
            "font-size: 11px; color: palette(text);")
        lay.addWidget(self._auto_review_count_label)

        # Score distribution strip above the slider: bars right of the cutoff are
        # bright (kept), left are dimmed (filtered out). Visual only.
        from ..confidence_histogram import ConfidenceHistogram
        self.auto_conf_histogram = ConfidenceHistogram()
        self.auto_conf_histogram.setToolTip(
            tr("How many objects sit at each confidence level."))
        lay.addWidget(self.auto_conf_histogram)
        self.auto_review_confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.auto_review_confidence_slider.setRange(
            _REVIEW_CONF_MIN, _REVIEW_CONF_MAX)
        self.auto_review_confidence_slider.setValue(30)
        self.auto_review_confidence_slider.setSingleStep(_REVIEW_CONF_STEP)
        self.auto_review_confidence_slider.setPageStep(_REVIEW_CONF_STEP)
        self.auto_review_confidence_slider.setMinimumHeight(26)
        self.auto_review_confidence_slider.setToolTip(tr(
            "Filter detections by confidence. Lower shows more (and noisier),"
            " higher keeps only the strongest. Free and instant."))
        # Dragging the slider mirrors into the spinbox live (cheap); the heavy
        # re-merge runs once via the debounce timer on release / spinbox edit.
        self.auto_review_confidence_slider.valueChanged.connect(
            self._on_conf_slider_moved)
        self.auto_review_confidence_slider.sliderReleased.connect(
            self._schedule_conf_refilter)
        self.auto_review_confidence_spin.valueChanged.connect(
            self._on_conf_spin_changed)
        lay.addWidget(self.auto_review_confidence_slider)

        # End labels so the slider direction reads at a glance (mirrors the
        # detail slider's Coarse/Fine ends).
        _conf_ends = QHBoxLayout()
        _conf_ends.setContentsMargins(2, 0, 2, 0)
        _conf_left = QLabel(tr("More objects"))
        _conf_left.setStyleSheet(
            "font-size: 10px; color: rgba(128,128,128,0.95);")
        _conf_right = QLabel(tr("Only confident"))
        _conf_right.setStyleSheet(
            "font-size: 10px; color: rgba(128,128,128,0.95);")
        _conf_ends.addWidget(_conf_left)
        _conf_ends.addStretch()
        _conf_ends.addWidget(_conf_right)
        lay.addLayout(_conf_ends)

        # What the number on the spinbox actually is. Nothing on the step said
        # it: a user reading "30%" had no way to know it is the AI rating its
        # own work. One line in the blue tip box, under the control it
        # explains, closed for good with the x once read.
        self.auto_confidence_hint = DismissibleHint(
            HINT_REVIEW_CONFIDENCE,
            tr("How sure the AI is about each object. Lower shows more, "
               "higher keeps only the sure ones."),
            tint=BLUE_TINT,
        )
        lay.addWidget(self.auto_confidence_hint)

        # Note shown when the review auto-lowered its starting cutoff because
        # nothing scored above 30%. Hidden by default; cleared on the first
        # user-initiated slider move.
        self.auto_conf_lowered_note = QLabel("")
        self.auto_conf_lowered_note.setWordWrap(True)
        self.auto_conf_lowered_note.setStyleSheet(
            "font-size: 10px; color: rgba(128,128,128,0.95);")
        self.auto_conf_lowered_note.setVisible(False)
        lay.addWidget(self.auto_conf_lowered_note)

        lay.addWidget(self._build_size_filter_block())
        lay.addWidget(self._build_boundary_snap_block())

        lay.addStretch(1)
        return page

    def _build_size_filter_block(self) -> QWidget:
        """Min / Max size: the second way an object is kept or dropped.

        Both hide detections client-side by true ground area (free, instant;
        0 = off / no limit), so they belong beside Confidence and not with the
        outline controls they used to sit under. The count line above names
        this filter when it, rather than Confidence, is what hides everything:
        the control it names now sits on the same step.
        """
        from qgis.PyQt.QtWidgets import QDoubleSpinBox

        block = QWidget()
        col = QVBoxLayout(block)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)
        col.addWidget(_card_divider())
        # Same header shape as the Confidence group above it: the two filters
        # on this step read as siblings, and the step still heads itself
        # through its groups rather than a bare title row.
        hdr = QHBoxLayout()
        size_lbl = QLabel(tr("Size"))
        size_lbl.setStyleSheet(_REVIEW_HEADING_QSS)
        size_note = QLabel(tr("hide anything outside this range"))
        size_note.setWordWrap(True)
        size_note.setStyleSheet(_REVIEW_HEADING_NOTE_QSS)
        hdr.addWidget(size_lbl)
        hdr.addWidget(size_note, 1)
        col.addLayout(hdr)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        min_lbl = QLabel(tr("Minimum"))
        min_lbl.setStyleSheet("font-size: 11px;")
        self.auto_min_size_spin = QDoubleSpinBox()
        self.auto_min_size_spin.setRange(0.0, 1_000_000.0)
        self.auto_min_size_spin.setDecimals(1)
        self.auto_min_size_spin.setValue(0.0)
        self.auto_min_size_spin.setSuffix(" m²")
        self.auto_min_size_spin.setSpecialValueText(tr("Off"))
        self.auto_min_size_spin.setMinimumWidth(78)
        self.auto_min_size_spin.setMaximumWidth(110)
        self.auto_min_size_spin.setToolTip(tr(
            "Hide detections smaller than this ground area. Use it to drop tiny "
            "noise blobs. 0 = keep all."))
        max_lbl = QLabel(tr("Maximum"))
        max_lbl.setStyleSheet("font-size: 11px;")
        self.auto_max_size_spin = QDoubleSpinBox()
        self.auto_max_size_spin.setRange(0.0, 10_000_000.0)
        self.auto_max_size_spin.setDecimals(1)
        self.auto_max_size_spin.setValue(0.0)
        self.auto_max_size_spin.setSuffix(" m²")
        self.auto_max_size_spin.setSpecialValueText(tr("No limit"))
        self.auto_max_size_spin.setMinimumWidth(78)
        self.auto_max_size_spin.setMaximumWidth(110)
        self.auto_max_size_spin.setToolTip(tr(
            "Hide detections larger than this ground area. 0 = no limit."))
        row.addWidget(min_lbl)
        row.addWidget(self.auto_min_size_spin)
        row.addStretch()
        row.addWidget(max_lbl)
        row.addWidget(self.auto_max_size_spin)
        col.addLayout(row)

        # Re-derive the visible set on any change (confidence has its own
        # debounced re-filter path).
        self.auto_min_size_spin.valueChanged.connect(
            lambda _v: self.auto_refine_changed.emit())
        self.auto_max_size_spin.valueChanged.connect(
            lambda _v: self.auto_refine_changed.emit())
        return block

    def _build_boundary_snap_block(self) -> QWidget:
        """Shared borders: one checkbox plus its one-line tip.

        A land-cover map is one surface cut into classes, so neighbouring
        shapes are meant to meet exactly; a detector returns each shape on its
        own, leaving a hairline gap or overlap. This closes them, free and
        instantly, on the shapes currently shown.

        The whole block is HIDDEN unless the run looks like land cover (see
        set_boundary_snap_offered): between two buildings or two cars the gap
        is real data, so there the option must not exist at all.
        """
        from qgis.PyQt.QtWidgets import QCheckBox

        from ...core.boundary_snap import snap_default_enabled
        from .guidance import HINT_REVIEW_SHARED_BORDERS

        self._auto_boundary_snap_offered = False
        block = QWidget()
        col = QVBoxLayout(block)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)
        col.addWidget(_card_divider())

        row = QHBoxLayout()
        label = QLabel(tr("Shared borders:"))
        label.setStyleSheet("font-size: 11px;")
        tip = tr(
            "Give neighbouring shapes one exact border instead of a hairline "
            "gap or overlap. For land cover, where the map is one surface.")
        label.setToolTip(tip)
        self.auto_boundary_snap_check = QCheckBox()
        self.auto_boundary_snap_check.setChecked(snap_default_enabled())
        self.auto_boundary_snap_check.setToolTip(tip)
        self.auto_boundary_snap_check.stateChanged.connect(
            lambda s: self._on_shape_control_changed(
                "shared_boundaries", bool(s)))
        row.addWidget(label)
        row.addStretch()
        row.addWidget(self.auto_boundary_snap_check)
        col.addLayout(row)

        self.auto_boundary_snap_hint = DismissibleHint(
            HINT_REVIEW_SHARED_BORDERS,
            tr("Closes the hairline gaps between neighbouring shapes, for "
               "land cover maps."),
            tint=BLUE_TINT,
        )
        col.addWidget(self.auto_boundary_snap_hint)

        self.auto_boundary_snap_row = block
        block.setVisible(False)
        return block

    # -- Step dials ---------------------------------------------------------

    def _build_review_dials(self) -> QWidget:
        """The three-step dials row: equal-width CLICKABLE columns (stretch 1
        each), dial above, 10px label below. Clicking a dial navigates the
        ladder both ways; state refreshed by _set_review_dial."""
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._auto_review_dials = []
        for i, name in enumerate(
                (tr("Keep"), tr("Correct"), tr("Shapes"))):
            col = _StepDialColumn(i)
            col.setToolTip(tr("Click to open this step"))
            col.clicked.connect(self._on_review_dial_clicked)
            v = QVBoxLayout(col)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(2)
            dial = _step_dial(i + 1, "active" if i == 0 else "todo")
            v.addWidget(dial, 0, Qt.AlignmentFlag.AlignHCenter)
            lab = QLabel(name)
            lab.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            lab.setStyleSheet(_dial_label_qss(i == 0))
            v.addWidget(lab)
            lay.addWidget(col, 1)
            self._auto_review_dials.append((col, dial, lab))
        self.auto_review_dials_row = row
        return row

    def _on_review_dial_clicked(self, step: int) -> None:
        """A step dial was clicked: free back/forward ladder navigation,
        ignored only on the current step. Every step is always reachable now
        (the Manual lock is gone)."""
        try:
            if step == getattr(self, "_auto_review_step", 0):
                return
            self.auto_review_step_requested.emit(int(step))
        except (RuntimeError, AttributeError):
            pass

    def _set_review_dial(self, idx: int, state: str) -> None:
        """Restyle one dial in place (todo/active/done). Borrows the exact
        text + QSS from a throwaway _step_dial so the look can never drift
        from the styles.py helper."""
        try:
            _col, dial, lab = self._auto_review_dials[idx]
        except (AttributeError, IndexError):
            return
        tmp = _step_dial(idx + 1, state)
        dial.setText(tmp.text())
        dial.setStyleSheet(tmp.styleSheet())
        tmp.deleteLater()
        lab.setStyleSheet(_dial_label_qss(state == "active"))

    def _set_review_dials_locked(self, locked: bool, current: int) -> None:
        """Drive each dial column's click affordance: only a non-current step
        is navigable and shows the pointer cursor plus the "click to open"
        tooltip; the current step gets neither, so the tooltip never invites a
        click the ladder ignores. ``locked`` is retained for API stability
        (the batch/Manual-lock states that dimmed the whole row are gone); a
        True value still dims every non-current column to 45% opacity."""
        try:
            dials = self._auto_review_dials
        except AttributeError:
            return
        for i, (col, _dial, _lab) in enumerate(dials):
            if locked and i != current:
                dim = QGraphicsOpacityEffect(col)
                dim.setOpacity(0.45)
                col.setGraphicsEffect(dim)
            else:
                col.setGraphicsEffect(None)
            navigable = not locked and i != current
            try:
                col.set_navigable(navigable)
                col.setToolTip(tr("Click to open this step") if navigable
                               else "")
            except AttributeError:
                pass

    def _sync_review_stack_height(self, _index: int = -1) -> None:
        """Nudge the parent layouts to re-read the stack's hint on a page
        switch. The snug-per-step sizing itself lives in _CurrentPageStack
        (its layout only contains the current page), so a plain
        updateGeometry() is enough to re-flow the height.

        No adjustSize() here: it resized the current page to its preferred
        sizeHint, whose WIDTH can exceed the docked panel width (a long
        word-wrapped line reports its full one-line width as preferred).
        Inside the setWidgetResizable scroll area that transiently made the
        content wider than the viewport, and the main window then grew the
        dock to that width and never shrank it back, so switching steps (e.g.
        Correct back to Keep) sometimes ballooned the whole panel
        sideways. Letting the layout own the geometry keeps the width put."""
        try:
            self.auto_review_step_stack.updateGeometry()
        except (RuntimeError, AttributeError):
            pass

    # The shape-and-size section no longer folds: it owns the Shapes step, so
    # its controls always show (the collapsible header and its setter are
    # gone).

    # The empty-run guidance box is gone: a run that finds nothing enters the
    # review directly on the Correct step with the zero-detection welcome
    # line (D8, set_zero_detection_entry).


def _build_display_combo(dock):
    """Build the View-detections-as combo (theme-safe inside the styled
    card). Split out only to keep _setup_auto_review_panel readable."""
    from qgis.PyQt.QtWidgets import QComboBox

    combo = QComboBox()
    # Theme-safe colors: inside the styled card the native combo loses the
    # app palette on the dark QGIS theme (its text painted black on dark).
    combo.setStyleSheet(_COMBO_THEME_QSS)
    combo.addItem(tr("Normal"), "normal")
    combo.addItem(tr("Outline"), "outline")
    combo.addItem(tr("Confidence"), "confidence")
    # Label "Distinct"; the stored mode key stays "random" so the whole
    # pipeline (state, telemetry, renderer) is unchanged.
    combo.addItem(tr("Distinct"), "random")
    # Distinct by default: a fresh review opens with one colour per object so
    # instances read apart (re-seeded per review by the plugin). Each object
    # keeps its colour across recomputes; only a new detection run re-assigns.
    combo.setCurrentIndex(max(0, combo.findData("random")))
    combo.setToolTip(tr(
        "How detections are coloured on the map (visual only): Normal fill, "
        "Outline, Confidence heatmap, or a distinct colour per object to tell "
        "them apart."))
    combo.currentIndexChanged.connect(
        lambda _i: dock.auto_display_mode_changed.emit(
            combo.currentData() or "confidence"))
    return combo
