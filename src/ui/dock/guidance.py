



















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
from .font_scale import scale_px_length, scale_qss_font_px
from .styles import (
    ACCENT_BORDER,
    ACCENT_BORDER_SOFT,
    CATEGORY_TILE_GLYPH_PX,
    CATEGORY_TILE_PX,
    FIELD,
    FONT_BODY,
    HOVER,
    HOVER_ON,
    HUE_TIP,
    INK,
    INK_2,
    LINE,
    RADIUS_CARD,
    RADIUS_CONTROL,
    RADIUS_ROW,
    SURFACE,
    _btn_hint_action_qss,
    category_wash,
    category_wash_line,
    paint_category_tile,
)

_SETTINGS_PREFIX = "AISegmentation/hints/"








HINT_START_AUTO = "start_auto_info"


HINT_TUTORIAL_FIRST_STEPS = "tutorial_first_steps"
HINT_TUTORIAL_ZERO_RESULTS = "tutorial_zero_results"
HINT_EXEMPLAR_TIP = "exemplar_tip"




HINT_INPUT_RULE = "input_rule"


HINT_RERUN_SAME_SETUP = "rerun_same_setup"


HINT_REVIEW_CONFIDENCE = "review_confidence"
HINT_REVIEW_CLOSED_CANOPY = "review_closed_canopy"


HINT_REVIEW_SHARED_BORDERS = "review_shared_borders"




HINT_REVIEW_QGIS_EDIT = "review_qgis_edit"
HINT_REVIEW_RESHAPE_GESTURES = "review_reshape_gestures"



HINT_REVIEW_RIGHT_CLICK_DELETE = "review_right_click_delete"


HINT_UPDATE_RECOMMENDED = "update_recommended"




HINT_PROMPT_TREE_OR_FOREST = "prompt_tree_or_forest"
HINT_PROMPT_ONE_OBJECT_PER_RUN = "prompt_one_object_per_run"
HINT_PROMPT_EXEMPLAR_BOOST = "prompt_exemplar_boost"
HINT_PROMPT_UNKNOWN_OBJECT = "prompt_unknown_object"



HINT_PROMPT_STEER_OBJECT = "prompt_steer_object"




HINT_PROMPT_SILENT_SWAP = "prompt_silent_swap"
HINT_PROMPT_EXAMPLES_DRIVE = "prompt_examples_drive"
HINT_PROMPT_RUN_PLAN = "prompt_run_plan"









HINT_EXEMPLAR_DRAW_BOX = "exemplar_draw_polygon"
HINT_EXEMPLAR_EXCLUDE_BOX = "exemplar_exclude_polygon"




HINT_PREVIEW_ZOOM = "preview_zoom_precision"
ALL_HINTS = [
    HINT_START_AUTO,
    HINT_TUTORIAL_FIRST_STEPS,
    HINT_TUTORIAL_ZERO_RESULTS,
    HINT_EXEMPLAR_TIP,
    HINT_INPUT_RULE,
    HINT_RERUN_SAME_SETUP,
    HINT_REVIEW_CONFIDENCE,
    HINT_REVIEW_SHARED_BORDERS,
    HINT_REVIEW_QGIS_EDIT,
    HINT_REVIEW_RESHAPE_GESTURES,
    HINT_REVIEW_RIGHT_CLICK_DELETE,
    HINT_UPDATE_RECOMMENDED,
    HINT_PROMPT_TREE_OR_FOREST,
    HINT_PROMPT_ONE_OBJECT_PER_RUN,
    HINT_PROMPT_EXEMPLAR_BOOST,
    HINT_PROMPT_UNKNOWN_OBJECT,
    HINT_PROMPT_STEER_OBJECT,
    HINT_PROMPT_SILENT_SWAP,
    HINT_PROMPT_EXAMPLES_DRIVE,
    HINT_PROMPT_RUN_PLAN,
    HINT_EXEMPLAR_DRAW_BOX,
    HINT_EXEMPLAR_EXCLUDE_BOX,
    HINT_PREVIEW_ZOOM,
]
















def hint_body(hint_id: str, fallback: str) -> str:





    from ...core.server_dials import dial_copy

    return dial_copy(f"guidance.{hint_id}", fallback)


def hint_suppressed(hint_id: str) -> bool:

    from ...core.server_dials import dial_list

    return hint_id in dial_list("guidance.suppressed", ())


def all_hint_ids() -> list[str]:





    from ...core.server_dials import dial_list

    extra = dial_list("guidance.extra", ())
    return ALL_HINTS + sorted(extra - set(ALL_HINTS))





GUIDE_URL_BASE = "https://terra-lab.ai/blog/ai-segmentation-complete-guide"


def guide_url(content: str) -> str:





    from ...core.surface_dials import guide_url_base

    base = guide_url_base(GUIDE_URL_BASE)
    joiner = "&" if "?" in base else "?"
    return (
        f"{base}{joiner}"
        "utm_source=qgis&utm_medium=plugin&utm_campaign=ai-segmentation"
        f"&utm_content={content}"
    )


def open_guide(content: str) -> None:





    from ..external_links import open_external_url

    open_external_url(guide_url(content))
    try:
        from ...core import telemetry_run_events
        telemetry_run_events.track_tutorial_opened(content)
    except Exception:
        pass  # nosec B110




GREEN_TINT = (67, 160, 71)
BLUE_TINT = (30, 136, 229)

LEAF_TINT = (139, 172, 39)


NEUTRAL_TINT = (128, 128, 128)



_LIVE_HINTS: list[weakref.ref[DismissibleHint]] = []


def is_hint_dismissed(hint_id: str) -> bool:
    return bool(QSettings().value(_SETTINGS_PREFIX + hint_id, False, type=bool))


def dismiss_hint(hint_id: str) -> None:
    QSettings().setValue(_SETTINGS_PREFIX + hint_id, True)


def dismiss_hint_for_version(hint_id: str, version: str) -> None:

    QSettings().setValue(_SETTINGS_PREFIX + hint_id + "/dismissed_version", str(version))


def is_hint_dismissed_for_version(
    hint_id: str, offered_version: str, installed_version: str = "",
) -> bool:





    stored = QSettings().value(_SETTINGS_PREFIX + hint_id + "/dismissed_version", "", type=str)
    if stored:
        return stored.strip() == str(offered_version).strip()
    if not is_hint_dismissed(hint_id):
        return False
    from ...core.server_dials import parse_version

    offered = parse_version(offered_version)
    installed = parse_version(installed_version)
    return offered is not None and installed is not None and offered <= installed


def reset_hints() -> None:





    s = QSettings()
    for hint_id in all_hint_ids():
        s.remove(_SETTINGS_PREFIX + hint_id)
    for ref in list(_LIVE_HINTS):
        widget = ref()
        if widget is None:
            _LIVE_HINTS.remove(ref)
            continue
        widget.reshow()


def _tint_hue(tint: tuple[int, int, int]) -> str | None:






    if tint == NEUTRAL_TINT:
        return None
    if tint == LEAF_TINT:
        return "leaf"
    return HUE_TIP


def _card_qss(hue: str | None, dense: bool = False,
              flat: bool = False) -> str:





    if flat and not dense:

        return "QFrame#hintCard { background: transparent; border: none; }"
    if dense or hue is None:
        border = ACCENT_BORDER_SOFT if dense else LINE
        return (
            f"QFrame#hintCard {{ background-color: {SURFACE};"
            f" border: 1px solid {border};"
            f" border-radius: {RADIUS_CARD}px; }}"
        )
    return (
        f"QFrame#hintCard {{ background-color: {category_wash(hue)};"
        f" border: 1px solid {category_wash_line(hue)};"
        f" border-radius: {RADIUS_CARD}px; }}"
    )



_GLYPH_TILE_PX = CATEGORY_TILE_PX
_GLYPH_PX = CATEGORY_TILE_GLYPH_PX
_CLOSE_GLYPH_PX = 12





_BODY_STYLE = (
    f"color: {INK}; font-size: {FONT_BODY}px; background: transparent; border: none;"
)



_CLOSE_STYLE = (
    "QToolButton { background: transparent; border: none; padding: 0;"
    f" border-radius: {RADIUS_CONTROL}px; }}"
    f"QToolButton:hover {{ background: {HOVER}; }}"
    f"QToolButton:pressed {{ background: {HOVER_ON}; }}"
    f"QToolButton:focus {{ border: 2px solid {ACCENT_BORDER}; }}"
)


class DismissibleHint(QWidget):











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
        closable: bool = True,
        parent=None,
        flat: bool = False,
        hue: str | None = None,
        glyph: str = "lightbulb",
    ):
        super().__init__(parent)
        self._hint_id = hint_id




        self._explicit_hue = hue
        self._hue = hue if hue is not None else _tint_hue(tint)
        self._glyph = glyph

        self._flat = bool(flat)
        self._show_glyph = bool(show_glyph)





        self._closable = bool(closable)
        body = hint_body(hint_id, body)




        self._visibility_gate = visibility_gate

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        card = QFrame(self)
        card.setObjectName("hintCard")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setStyleSheet(_card_qss(self._hue, flat=self._flat))
        outer.addWidget(card)


        self._card = card
        self._card_tint = tint
        self._card_dense = False

        col = QVBoxLayout(card)
        if self._flat:
            col.setContentsMargins(0, 2, 0, 2)
        else:
            col.setContentsMargins(12, 10, 8, 10)
        col.setSpacing(4)
        self._card_col = col

        close_btn = None
        if self._closable:
            from ..icons import icon_for

            close_btn = QToolButton(card)
            close_btn.setToolTip(tr("Got it - hide this tip"))
            close_btn.setAccessibleName(tr("Dismiss"))
            close_btn.setCursor(Qt.CursorShape.PointingHandCursor)

            close_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
            close_btn.setStyleSheet(_CLOSE_STYLE)
            close_btn.setFixedSize(scale_px_length(22), scale_px_length(22))
            from qgis.PyQt.QtCore import QSize
            from qgis.PyQt.QtGui import QColor
            close_btn.setIcon(icon_for(close_btn, "close", _CLOSE_GLYPH_PX, QColor(INK_2)))
            close_btn.setIconSize(QSize(_CLOSE_GLYPH_PX, _CLOSE_GLYPH_PX))
            close_btn.clicked.connect(self._on_close)




        tile_px = scale_px_length(_GLYPH_TILE_PX)
        self._glyph_tile = QLabel(card)
        self._glyph_tile.setFixedSize(tile_px, tile_px)
        self._glyph_tile.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._paint_glyph_tile(tint)
        self._glyph_tile.setVisible(self._show_glyph)

        body_lbl = QLabel(body)


        body_lbl.setTextFormat(Qt.TextFormat.PlainText)
        body_lbl.setWordWrap(True)
        body_lbl.setStyleSheet(scale_qss_font_px(_BODY_STYLE))
        self.body_label = body_lbl




        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(10)
        head.addWidget(self._glyph_tile, 0, Qt.AlignmentFlag.AlignTop)
        head.addWidget(body_lbl, 1, Qt.AlignmentFlag.AlignVCenter)
        if action_text:
            r, g, b = action_color or tint
            act_btn = QToolButton(card)
            act_btn.setText(action_text)
            act_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            act_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)


            act_btn.setStyleSheet(_btn_hint_action_qss((r, g, b)))
            act_btn.clicked.connect(self.action.emit)
            head.addWidget(act_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        if close_btn is not None:
            head.addWidget(close_btn, 0, Qt.AlignmentFlag.AlignTop)
        col.addLayout(head)

        self.setVisible(not self.is_dismissed())


        _LIVE_HINTS[:] = [ref for ref in _LIVE_HINTS if ref() is not None]
        _LIVE_HINTS.append(weakref.ref(self))

    def set_flat(self, flat: bool) -> None:


        flat = bool(flat)
        if flat == self._flat:
            return
        self._flat = flat
        dense = self._card_dense
        self._card.setStyleSheet(_card_qss(self._hue, dense, flat=flat))
        self._card_col.setContentsMargins(
            *((0, 2, 0, 2) if flat and not dense else (12, 10, 8, 10)))

    def is_dismissed(self) -> bool:






        return bool(self._closable and is_hint_dismissed(self._hint_id))

    def setVisible(self, visible: bool) -> None:  # noqa: N802







        if visible:
            try:



                hint_id = getattr(self, "_hint_id", None)
                if hint_id and hint_suppressed(hint_id):
                    visible = False
            except Exception:  # noqa: BLE001  # nosec B110
                pass
        super().setVisible(visible)

    def set_body_text(self, body: str, copy_id: str | None = None) -> None:







        if copy_id:
            body = hint_body(copy_id, body)
        self.body_label.setText(body)

    def _paint_glyph_tile(self, tint: tuple[int, int, int]) -> None:

        del tint
        try:
            tile = self._glyph_tile
            hue = self._hue
            if hue is None:
                from qgis.PyQt.QtGui import QColor

                from ..icons import pixmap_for

                tile.setStyleSheet(
                    f"QLabel {{ background: {FIELD}; border: none;"
                    f" border-radius: {RADIUS_ROW}px; }}")
                tile.setPixmap(pixmap_for(tile, self._glyph, _GLYPH_PX, QColor(INK_2)))
                return
            paint_category_tile(tile, self._glyph, hue, _GLYPH_PX,
                                on_wash=not self._flat)
        except Exception:  # noqa: BLE001
            return

    def set_hint(self, hint_id: str, body: str,
                 tint: tuple[int, int, int] | None = None,
                 show_glyph: bool | None = None,
                 dense: bool = False) -> bool:










        self._hint_id = hint_id
        if show_glyph is not None:
            self._show_glyph = bool(show_glyph)
            self._glyph_tile.setVisible(self._show_glyph)
        tint = tint or self._card_tint
        if (tint, dense) != (self._card_tint, self._card_dense):
            self._card_tint, self._card_dense = tint, dense



            keep = self._explicit_hue is not None and tint != NEUTRAL_TINT
            self._hue = self._explicit_hue if keep else _tint_hue(tint)
            self._card.setStyleSheet(_card_qss(self._hue, dense, flat=self._flat))
            if self._flat:


                self._card_col.setContentsMargins(
                    *((12, 10, 8, 10) if dense else (0, 2, 0, 2)))
            self._paint_glyph_tile(tint)
        self.body_label.setTextFormat(Qt.TextFormat.PlainText)
        self.set_body_text(body, copy_id=hint_id)
        return not self.is_dismissed()

    def reshow(self) -> None:







        gate = self._visibility_gate
        if gate is not None:
            try:
                if not gate():
                    return
            except Exception:  # nosec B110
                pass
        self.show()

    def _on_close(self) -> None:
        dismiss_hint(self._hint_id)
        self.hide()
        self.dismissed.emit()
