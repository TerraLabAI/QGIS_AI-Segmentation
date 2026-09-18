










from __future__ import annotations

import re

from qgis.PyQt.QtCore import QEvent, QObject, Qt, pyqtSignal
from qgis.PyQt.QtWidgets import QBoxLayout, QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ...core.activation_manager import TUTORIAL_URL_FALLBACK, get_tutorial_url
from ...core.i18n import tr
from ...core.qt_compat import event_pos
from ..dock.font_scale import scale_px_length, scale_qss_font_px
from ..dock.styles import (
    ACCENT_BORDER_SOFT,
    ACCENT_TINT,
    FONT_BASE,
    FONT_BODY,
    INK_2,
    LINE,
    RADIUS_CARD,
    SURFACE,
    category_fill,
)
from ..external_links import open_external_url
from ..sibling_thumbnails import SiblingShot, load_shot_image
from ..siblings_dialog import SiblingCardGrid
from .a11y import FOCUS_RING
from .settings_widgets import SettingsPage, muted_label

GUIDE_THUMBNAIL_URL = "https://terra-lab.ai/blog/ai-segmentation-complete-guide/og.jpg"
_YOUTUBE_ID = re.compile(r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/|shorts/))([\w-]{11})")
_CARD_MIN_W = 240
_KIND_CATEGORIES = {"video": "coral", "guide": "sky"}
_LEARN_CARD_QSS = scale_qss_font_px(
    f"QFrame#learnCard {{ background: {SURFACE}; border: 1px solid {LINE};"
    f" border-radius: {RADIUS_CARD}px; }}"
    f"QFrame#learnCard:hover {{ background: {ACCENT_TINT}; border-color: {ACCENT_BORDER_SOFT}; }}"
    f"QFrame#learnCard:focus {{ border: 2px solid {FOCUS_RING}; }}"
    "QFrame#learnCard QLabel { background: transparent; border: none; }"
    f"QLabel#learnTitle {{ font-size: {FONT_BASE}px; font-weight: 600; color: palette(text); }}"
    f"QLabel#learnNote {{ font-size: {FONT_BODY}px; color: {INK_2}; }}"
)


def youtube_thumbnail_url(url: str) -> str:

    match = _YOUTUBE_ID.search(url or "")
    return f"https://img.youtube.com/vi/{match.group(1)}/maxresdefault.jpg" if match else ""


class TutorialCard(QFrame):


    clicked = pyqtSignal()

    def __init__(self, kind: str, title: str, note: str, thumbnail_url: str, parent=None):
        super().__init__(parent)
        self.setObjectName("learnCard")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(_LEARN_CARD_QSS)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(scale_px_length(_CARD_MIN_W) // 2)

        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setAccessibleName(title)
        self.setAccessibleDescription(note)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(1, 1, 1, 0)
        outer.setSpacing(0)
        self._shot = SiblingShot("", self, category_fill(_KIND_CATEGORIES.get(kind, "sky")))
        outer.addWidget(self._shot)
        words = QVBoxLayout()
        words.setContentsMargins(14, 10, 14, 14)
        words.setSpacing(4)
        for text, name in ((title, "learnTitle"), (note, "learnNote")):
            if not text:
                continue
            label = QLabel(text, self)
            label.setObjectName(name)
            label.setWordWrap(True)
            words.addWidget(label)
        words.addStretch(1)
        outer.addLayout(words, 1)
        self._loader = load_shot_image(self._shot, self, thumbnail_url)

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.clicked.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if (event.button() == Qt.MouseButton.LeftButton
                and self.rect().contains(event_pos(event))):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class _StackWhenNarrow(QObject):


    def __init__(self, layout: QBoxLayout, parent: QWidget):
        super().__init__(parent)
        self._layout = layout

    def eventFilter(self, watched, event):  # noqa: N802





        try:
            if event.type() == QEvent.Type.Resize:
                narrow = event.size().width() < 2 * scale_px_length(_CARD_MIN_W) + 12
                direction = (QBoxLayout.Direction.TopToBottom if narrow
                             else QBoxLayout.Direction.LeftToRight)
                if self._layout.direction() != direction:
                    self._layout.setDirection(direction)
        except RuntimeError:
            pass  # nosec B110
        return False


class LearnPagesMixin:


    def _build_tutorials_page(self) -> SettingsPage:
        page = SettingsPage(tr("Tutorials"), "", self, glyph="play", category="coral")

        video_url = get_tutorial_url() or TUTORIAL_URL_FALLBACK
        row_host = QWidget()
        row = QBoxLayout(QBoxLayout.Direction.LeftToRight, row_host)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        row_host.installEventFilter(_StackWhenNarrow(row, row_host))
        video = TutorialCard(
            "video", tr("Video tutorial"), "", youtube_thumbnail_url(video_url), row_host)
        video.clicked.connect(lambda: self._open_video_tutorial(video_url))
        guide = TutorialCard(
            "guide", tr("Written guide"), "", GUIDE_THUMBNAIL_URL, row_host)
        guide.clicked.connect(self._open_written_guide)
        row.addWidget(video, 1)
        row.addWidget(guide, 1)
        page.add(row_host)
        return page

    def _open_video_tutorial(self, url: str) -> None:
        open_external_url(url, parent=self)
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_tutorial_opened("settings_video")
        except Exception:  # noqa: BLE001
            pass  # nosec B110

    def _open_written_guide(self) -> None:
        from ..dock.guidance import open_guide

        open_guide("settings_tutorials")

    def _build_shortcuts_page(self) -> SettingsPage:
        page = SettingsPage(tr("Keyboard shortcuts"), "", self, glyph="terminal", category="leaf")
        try:
            from .shortcuts_card import build_shortcuts_card

            page.add(build_shortcuts_card(page))
        except Exception as exc:  # noqa: BLE001
            from qgis.core import Qgis

            from ...core.logging_utils import log

            log(f"Keyboard shortcuts not shown: {exc}", Qgis.MessageLevel.Warning)
            page.add(muted_label(tr("The shortcuts could not be listed.")))
        return page

    def _build_plugins_page(self) -> SettingsPage:
        page = SettingsPage(tr("More plugins"), "", self, glyph="puzzle", category="violet")
        self._sibling_grid = SiblingCardGrid()
        page.add(self._sibling_grid)
        return page

    def _refresh_plugins_page(self) -> None:
        grid = getattr(self, "_sibling_grid", None)
        if grid is not None:
            try:
                grid.refresh()
            except RuntimeError:
                pass  # nosec B110


__all__ = ["GUIDE_THUMBNAIL_URL", "LearnPagesMixin", "TutorialCard", "youtube_thumbnail_url"]
