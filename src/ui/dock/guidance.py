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

_SETTINGS_PREFIX = "AISegmentation/hints/"

# Hint ids. Listed here so settings can reset them all at once.
HINT_START_MANUAL = "start_manual_info"
HINT_START_AUTO = "start_auto_info"
HINT_REVIEW_CONFIDENCE = "review_confidence"
HINT_TRY_AUTOMATIC = "try_automatic"
# Tutorial-discovery nudges: a post-sign-in first-steps banner and
# a zero-result friction banner, both pointing at the step-by-step guide.
HINT_TUTORIAL_FIRST_STEPS = "tutorial_first_steps"
HINT_TUTORIAL_ZERO_RESULTS = "tutorial_zero_results"
HINT_EXEMPLAR_TIP = "exemplar_tip"
ALL_HINTS = [
    HINT_START_MANUAL,
    HINT_START_AUTO,
    HINT_REVIEW_CONFIDENCE,
    HINT_TRY_AUTOMATIC,
    HINT_TUTORIAL_FIRST_STEPS,
    HINT_TUTORIAL_ZERO_RESULTS,
    HINT_EXEMPLAR_TIP,
]

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
NEUTRAL_TINT = (128, 132, 138)

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
    for hint_id in ALL_HINTS:
        s.remove(_SETTINGS_PREFIX + hint_id)
    for ref in list(_LIVE_HINTS):
        widget = ref()
        if widget is None:
            _LIVE_HINTS.remove(ref)
            continue
        widget.reshow()


def _card_qss(tint: tuple[int, int, int]) -> str:
    r, g, b = tint
    return (
        "QFrame#hintCard {{ background-color: rgba({r},{g},{b},0.07);"
        " border: 1px solid rgba({r},{g},{b},0.30); border-radius: 6px; }}"
    ).format(r=r, g=g, b=b)


_BODY_STYLE = (
    "color: palette(text); font-size: 11px; background: transparent; border: none;"
)
_CLOSE_STYLE = (
    "QToolButton { background: transparent; color: rgba(120,124,130,0.95);"
    " border: none; font-size: 14px; font-weight: 700; }"
    "QToolButton:hover { color: palette(text); }"
)


class DismissibleHint(QWidget):
    """A small tinted inline guidance callout with a tiny light x close button.

    Closing the card stores the dismissal so it stays hidden until the user
    resets guidance from Account Settings. ``action_text`` (optional) renders a
    small link-style button under the body, whose click emits ``action``.
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

        col = QVBoxLayout(card)
        col.setContentsMargins(10, 7, 8, 7)
        col.setSpacing(4)

        close_btn = QToolButton(card)
        close_btn.setText("✕")  # x glyph
        close_btn.setToolTip(self.tr("Got it - hide this tip"))
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(_CLOSE_STYLE)
        close_btn.setFixedSize(20, 20)
        close_btn.clicked.connect(self._on_close)

        # Tip prefix from the taxonomy (the lightbulb, the one emoji kind).
        # show_glyph=False for mode DESCRIPTIONS (what Manual/Automatic do):
        # they state a fact, they do not tip.
        from .styles import _msg_text
        body_lbl = QLabel(_msg_text("info", body) if show_glyph else body)
        body_lbl.setWordWrap(True)
        body_lbl.setStyleSheet(_BODY_STYLE)

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
            # green guide tip), white text: unmistakably a button, not a
            # hanging bit of text.
            act_btn.setStyleSheet(
                "QToolButton {{ background: rgb({r},{g},{b}); color: white;"
                " border: none; border-radius: 4px; padding: 3px 10px;"
                " font-size: 11px; font-weight: 700; }}"
                "QToolButton:hover {{ background: rgba({r},{g},{b},0.85); }}".format(
                    r=r, g=g, b=b)
            )
            act_btn.clicked.connect(self.action.emit)
            head.addWidget(act_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        head.addWidget(close_btn, 0, Qt.AlignmentFlag.AlignTop)
        col.addLayout(head)

        self.setVisible(not is_hint_dismissed(hint_id))
        _LIVE_HINTS.append(weakref.ref(self))

    def reshow(self) -> None:
        """Re-show after a guidance reset, honoring the optional visibility gate.

        A gate that returns False keeps the hint hidden (its owner will show it
        when the relevant screen is next active), so a reset never flashes a
        pinned banner into a mode or step where it does not belong.
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
