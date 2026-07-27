"""Dismissible in-UI guidance tips (ported from AI Edit's onboarding pattern).

A GIS user who is new to AI segmentation benefits from a short, plain-words
nudge at each step. These tips are non-blocking inline callouts: shown once,
closed for good with the small x, and re-enabled from Account Settings
("Show guidance tips again"). One tip per screen at most, so guidance never
becomes clutter.

Pattern (how mature apps do onboarding without burdening the UI):
  - inline callout, never a modal that blocks work,
  - dismissible and remembered (QSettings), not nagging,
  - re-showable on demand from settings (live, if the dock is open),
  - short, action-oriented copy.

Colors follow the mode: a green tint on Manual contexts, a blue tint on
Automatic contexts (matching dock/styles.py brand accents). Everything uses
palette(text) so the tips read on light and dark QGIS themes.
"""

from __future__ import annotations

import weakref

from qgis.PyQt.QtCore import QSettings, Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
from .styles import _SUBCARD_MARGINS, _btn_hint_action_qss, _msg_text

_SETTINGS_PREFIX = "AISegmentation/hints/"

# Hint ids. Listed here so settings can reset them all at once.
HINT_START_MANUAL = "start_manual_info"
HINT_START_AUTO = "start_auto_info"
HINT_TRY_AUTOMATIC = "try_automatic"
# Tutorial-discovery nudges: a post-sign-in first-steps banner and
# a zero-result friction banner, both pointing at the step-by-step guide.
HINT_TUTORIAL_FIRST_STEPS = "tutorial_first_steps"
HINT_TUTORIAL_ZERO_RESULTS = "tutorial_zero_results"
HINT_EXEMPLAR_TIP = "exemplar_tip"
# The next Detect would repeat the last run exactly. Advisory, never a gate,
# so it obeys the same x-to-close rule as every other blue tip.
HINT_RERUN_SAME_SETUP = "rerun_same_setup"
# What the Confidence number is (Keep step): the AI's own score for each
# object. Heads no step and repeats no label, so it is a tip, not a title.
HINT_REVIEW_CONFIDENCE = "review_confidence"
# What the review's shared-borders control does (Keep step, land cover
# only). Shown next to the control it explains.
HINT_REVIEW_SHARED_BORDERS = "review_shared_borders"
# Contextual Correct-step guidance. These stay off the canvas until the user
# reaches the matching gesture and can be dismissed permanently. The
# "click a detection" prompt is deliberately NOT here: it became the step's
# resting hero, which must never be dismissible.
HINT_REVIEW_QGIS_EDIT = "review_qgis_edit"
HINT_REVIEW_RESHAPE_GESTURES = "review_reshape_gestures"
# Which shape the selected-detection actions act on. One line under the action
# grid, gone for good once the user has read it.
HINT_REVIEW_CORRECT_TARGET = "review_correct_target"
# The server asks builds older than a floor version to update. Discreet and
# dismissible: it is guidance, never a wall.
HINT_UPDATE_RECOMMENDED = "update_recommended"
# Advisories under the Automatic prompt box. They share one line, so they need
# one id EACH: closing the tree-versus-forest heads-up must not also silence
# the one-object-per-run note. The blocking guard (the prompt is off the rails)
# has no id on purpose, it must always be read.
HINT_PROMPT_TREE_OR_FOREST = "prompt_tree_or_forest"
HINT_PROMPT_ONE_OBJECT_PER_RUN = "prompt_one_object_per_run"
HINT_PROMPT_EXEMPLAR_BOOST = "prompt_exemplar_boost"
HINT_PROMPT_UNKNOWN_OBJECT = "prompt_unknown_object"
HINT_PROMPT_SILENT_SWAP = "prompt_silent_swap"
HINT_PROMPT_EXAMPLES_DRIVE = "prompt_examples_drive"
HINT_PROMPT_RUN_PLAN = "prompt_run_plan"
# The two armed example-draw instructions, on the example card's message line.
# The amber too-small warning shares that line and stays non-dismissible.
HINT_EXEMPLAR_DRAW_BOX = "exemplar_draw_box"
HINT_EXEMPLAR_EXCLUDE_BOX = "exemplar_exclude_box"
ALL_HINTS = [
    HINT_START_MANUAL,
    HINT_START_AUTO,
    HINT_TRY_AUTOMATIC,
    HINT_TUTORIAL_FIRST_STEPS,
    HINT_TUTORIAL_ZERO_RESULTS,
    HINT_EXEMPLAR_TIP,
    HINT_RERUN_SAME_SETUP,
    HINT_REVIEW_CONFIDENCE,
    HINT_REVIEW_SHARED_BORDERS,
    HINT_REVIEW_QGIS_EDIT,
    HINT_REVIEW_RESHAPE_GESTURES,
    HINT_REVIEW_CORRECT_TARGET,
    HINT_UPDATE_RECOMMENDED,
    HINT_PROMPT_TREE_OR_FOREST,
    HINT_PROMPT_ONE_OBJECT_PER_RUN,
    HINT_PROMPT_EXEMPLAR_BOOST,
    HINT_PROMPT_UNKNOWN_OBJECT,
    HINT_PROMPT_SILENT_SWAP,
    HINT_PROMPT_EXAMPLES_DRIVE,
    HINT_PROMPT_RUN_PLAN,
    HINT_EXEMPLAR_DRAW_BOX,
    HINT_EXEMPLAR_EXCLUDE_BOX,
]


# -- what the server may change about a hint -------------------------------
# Guidance is the highest-churn copy in the plugin and it is release-blocked
# today. Three levers, all fail-open to what shipped:
#   copy.guidance.<hint_id>   the body, one sentence, replaced whole
#   guidance.suppressed       ids that must stop showing
#   guidance.extra           ids the server knows about but this build does not
#
# Suppression is the one place a served list takes something away instead of
# adding to it. It is safe here and only here: a suppressed hint shows LESS
# guidance, never more permission, and a tip that turns out to annoy people has
# to be removable on the day it is reported rather than at the next release.
# An id nobody triggers is simply never built, so an unknown id costs nothing.


def hint_body(hint_id: str, fallback: str) -> str:
    """The served sentence for one hint, or the shipped one.

    Server copy arrives already in the user's language, so it carries no
    ``tr()``; the fallback is the shipped translated string.
    """
    from ...core.server_dials import dial_copy

    return dial_copy(f"guidance.{hint_id}", fallback)


def hint_suppressed(hint_id: str) -> bool:
    """Whether the server asked for this hint to stop showing."""
    from ...core.server_dials import dial_list

    return hint_id in dial_list("guidance.suppressed", ())


def all_hint_ids() -> list[str]:
    """Shipped hint ids plus any extra id the server names.

    Only used for bookkeeping (the guidance reset clears them all), so an id
    with no matching trigger in this build is harmless.
    """
    from ...core.server_dials import dial_list

    extra = dial_list("guidance.extra", ())
    return ALL_HINTS + sorted(extra - set(ALL_HINTS))


# Step-by-step guide. Defined once here; every touchpoint derives its own
# variant via guide_url(<utm_content>) so the base + UTM stem never gets
# copy-pasted (matches the UTM pattern in activation_manager / build.py).
GUIDE_URL_BASE = "https://terra-lab.ai/blog/ai-segmentation-complete-guide"


def guide_url(content: str) -> str:
    """Guide URL with the shared UTM stem and a per-touchpoint utm_content."""
    return (
        f"{GUIDE_URL_BASE}"
        "?utm_source=qgis&utm_medium=plugin&utm_campaign=ai-segmentation"
        f"&utm_content={content}"
    )


def open_guide(content: str) -> None:
    """Open the guide in the system browser and record the tutorial open.

    ``content`` is the touchpoint id (also the utm_content and telemetry
    source), one of: footer_tutorial, post_signin, zero_results.
    """
    from qgis.PyQt.QtCore import QUrl
    from qgis.PyQt.QtGui import QDesktopServices

    QDesktopServices.openUrl(QUrl(guide_url(content)))
    try:
        from ...core import telemetry
        telemetry.track_tutorial_opened(content)
    except Exception:
        pass  # nosec B110  Telemetry is best-effort, never blocks the open.


# Mode tints as RGB components (match the Start-caption colors in styles.py):
# green for Manual contexts, blue for Automatic contexts.
GREEN_TINT = (67, 160, 71)
BLUE_TINT = (30, 136, 229)
# TerraLab brand leaf green (#8bac27, the credit-ring color).
LEAF_TINT = (139, 172, 39)
# Discreet neutral grey for low-key guidance (the tutorial tip): a quiet card
# that does not shout for attention.
NEUTRAL_TINT = (128, 128, 128)

# Live hint widgets, so "Show guidance tips again" can re-show them without a
# dock rebuild. Weak refs: closing/destroying a hint drops out on its own.
_LIVE_HINTS: list[weakref.ref[DismissibleHint]] = []


def is_hint_dismissed(hint_id: str) -> bool:
    return bool(QSettings().value(_SETTINGS_PREFIX + hint_id, False, type=bool))


def dismiss_hint(hint_id: str) -> None:
    QSettings().setValue(_SETTINGS_PREFIX + hint_id, True)


def reset_hints() -> None:
    """Re-enable every hint so the user sees the guidance again.

    Also re-shows any hint widget currently alive (a dock open behind the
    account dialog), so the change is visible immediately, not only next open.
    """
    s = QSettings()
    for hint_id in all_hint_ids():
        s.remove(_SETTINGS_PREFIX + hint_id)
    for ref in list(_LIVE_HINTS):
        widget = ref()
        if widget is None:
            _LIVE_HINTS.remove(ref)
            continue
        widget.reshow()


def _card_qss(tint: tuple[int, int, int], dense: bool = False) -> str:
    """Fill and border for one tinted callout.

    ``dense`` is the taxonomy's armed pair (0.12 / 0.40) instead of the resting
    guidance pair: a map tool is armed and waiting for a draw, which the design
    system asks to read as a stronger blue than a tip sitting beside content.
    """
    r, g, b = tint
    fill, border = ("0.12", "0.40") if dense else ("0.07", "0.30")
    return (
        f"QFrame#hintCard {{ background-color: rgba({r},{g},{b},{fill});"
        f" border: 1px solid rgba({r},{g},{b},{border}); border-radius: 6px; }}"
    )


_BODY_STYLE = (
    "color: palette(text); font-size: 11px; background: transparent; border: none;"
)
_CLOSE_STYLE = (
    "QToolButton { background: transparent; color: rgba(128,128,128,0.95);"
    " border: none; font-size: 14px; font-weight: 700; }"
    "QToolButton:hover { color: palette(text); }"
)


class DismissibleHint(QWidget):
    """A small tinted inline guidance callout with a tiny light x close button.

    Closing the card stores the dismissal so it stays hidden until the user
    resets guidance from Account Settings. ``action_text`` (optional) renders a
    small link-style button under the body, whose click emits ``action``.

    The body passed in is the shipped sentence. The server may replace it and
    may suppress the hint outright (see ``hint_body`` / ``hint_suppressed``);
    both are resolved here so every call site gets them without changing.
    """

    dismissed = pyqtSignal()
    action = pyqtSignal()

    def __init__(
        self,
        hint_id: str,
        body: str,
        tint: tuple[int, int, int] = GREEN_TINT,
        action_text: str | None = None,
        visibility_gate=None,
        action_color: tuple[int, int, int] | None = None,
        show_glyph: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._hint_id = hint_id
        self._show_glyph = bool(show_glyph)
        body = hint_body(hint_id, body)
        # Optional callable -> bool. When set, a guidance reset only re-shows the
        # hint if the gate allows it. Banners pinned to the dock bottom (always
        # mounted, no hidden parent) use this so "Show guidance again" can never
        # reveal them in the wrong mode/step.
        self._visibility_gate = visibility_gate

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        card = QFrame(self)
        card.setObjectName("hintCard")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setStyleSheet(_card_qss(tint))
        outer.addWidget(card)
        # Kept so set_hint can repaint the card when one line carries messages
        # of two kinds (a blue tip and a quiet neutral note).
        self._card = card
        self._card_tint = tint
        self._card_dense = False

        col = QVBoxLayout(card)
        col.setContentsMargins(*_SUBCARD_MARGINS)
        col.setSpacing(4)

        close_btn = QToolButton(card)
        close_btn.setText("✕")  # x glyph
        close_btn.setToolTip(tr("Got it - hide this tip"))
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(_CLOSE_STYLE)
        close_btn.setFixedSize(20, 20)
        close_btn.clicked.connect(self._on_close)

        # Tip prefix from the taxonomy (the lightbulb, the one emoji kind).
        # show_glyph=False for mode DESCRIPTIONS (what Manual/Automatic do):
        # they state a fact, they do not tip.
        body_lbl = QLabel(_msg_text("info", body) if show_glyph else body)
        body_lbl.setWordWrap(True)
        body_lbl.setStyleSheet(_BODY_STYLE)
        self.body_label = body_lbl

        # One compact row: body text, then the action link, then the tiny x.
        # Everything inline keeps the card a thin single band instead of a
        # tall block with an empty second line.
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        head.addWidget(body_lbl, 1, Qt.AlignmentFlag.AlignVCenter)
        if action_text:
            r, g, b = action_color or tint
            act_btn = QToolButton(card)
            act_btn.setText(action_text)
            act_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            # Filled pill in a color that contrasts with the card (blue on the
            # green guide tip): unmistakably a button, not a hanging bit of
            # text. Black text on the fill, the same AA-safe convention as
            # every other filled button in the design system.
            act_btn.setStyleSheet(_btn_hint_action_qss((r, g, b)))
            act_btn.clicked.connect(self.action.emit)
            head.addWidget(act_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        head.addWidget(close_btn, 0, Qt.AlignmentFlag.AlignTop)
        col.addLayout(head)

        self.setVisible(not is_hint_dismissed(hint_id))
        _LIVE_HINTS.append(weakref.ref(self))

    def setVisible(self, visible: bool) -> None:  # noqa: N802 -- Qt override
        """Show the card, unless the server asked for this hint to stop.

        The one choke point for suppression: owners show and hide their hints
        from a dozen places, and Qt itself routes ``show()`` through here, so a
        suppressed hint can never reappear by any of those paths. Never raises,
        because this runs inside layout.
        """
        if visible:
            try:
                # getattr, not the attribute: Qt may reach this from C++ before
                # __init__ has run its body, and a paint-time AttributeError
                # here would take QGIS down with it.
                hint_id = getattr(self, "_hint_id", None)
                if hint_id and hint_suppressed(hint_id):
                    visible = False
            except Exception:  # noqa: BLE001 -- a bad config must not break layout  # nosec B110
                pass
        super().setVisible(visible)

    def set_body_text(self, body: str, copy_id: str | None = None) -> None:
        """Update a contextual hint while preserving its dismissal state.

        ``copy_id`` names the served string for THIS variant. A hint whose body
        swaps with context has one shipped sentence per variant, so they need
        one served id each; without it the body is left as passed, because a
        single served string would flatten every variant into one.
        """
        if copy_id:
            body = hint_body(copy_id, body)
        text = _msg_text("info", body) if self._show_glyph else body
        self.body_label.setText(text)

    def set_hint(self, hint_id: str, body: str,
                 tint: tuple[int, int, int] | None = None,
                 show_glyph: bool | None = None,
                 dense: bool = False) -> bool:
        """Point this one card at another tip, and say whether it may show.

        A line that carries several mutually exclusive advisories needs one
        dismissal id PER message, or closing one would silence the rest. The
        caller owns visibility: False means the user already closed THIS tip,
        so the line stays empty.

        The body renders as plain text, so a served sentence or a word the user
        typed can never inject markup into the card.
        """
        self._hint_id = hint_id
        if show_glyph is not None:
            self._show_glyph = bool(show_glyph)
        tint = tint or self._card_tint
        if (tint, dense) != (self._card_tint, self._card_dense):
            self._card_tint, self._card_dense = tint, dense
            self._card.setStyleSheet(_card_qss(tint, dense))
        self.body_label.setTextFormat(Qt.TextFormat.PlainText)
        self.set_body_text(body, copy_id=hint_id)
        return not is_hint_dismissed(hint_id)

    def reshow(self) -> None:
        """Re-show after a guidance reset, honoring the optional visibility gate.

        A gate that returns False keeps the hint hidden (its owner will show it
        when the relevant screen is next active), so a reset never flashes a
        pinned banner into a mode or step where it does not belong. A hint the
        server suppressed stays hidden too, through the ``setVisible`` gate.
        """
        gate = self._visibility_gate
        if gate is not None:
            try:
                if not gate():
                    return
            except Exception:  # nosec B110 -- a broken gate must not block the reset
                pass
        self.show()

    def _on_close(self) -> None:
        dismiss_hint(self._hint_id)
        self.hide()
        self.dismissed.emit()
