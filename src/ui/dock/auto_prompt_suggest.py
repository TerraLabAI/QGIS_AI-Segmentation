


























from __future__ import annotations

import re

from qgis.PyQt.QtCore import QEvent, QItemSelectionModel, QModelIndex, QObject, QSize, Qt
from qgis.PyQt.QtGui import QPainter, QStandardItem, QStandardItemModel
from qgis.PyQt.QtWidgets import QCompleter, QFrame, QListView, QStyledItemDelegate

from ...core.presets.segmentation_presets import (
    catalog_revision,
    fold_search_text,
    pick_label,
    preset_search_haystack,
)
from ...core.qt_compat import event_pos, safe_single_shot
from ...core.surface_dials import (
    prompt_suggest_max_rows,
    prompt_suggest_recent_scan,
    prompt_suggest_synonym_min_chars,
    prompt_suggest_visible_rows,
)
from .auto_flow_look import token_qcolor
from .font_scale import scale_px_length, scale_qss_font_px
from .styles import (
    FONT_BASE,
    HOVER,
    HOVER_ON,
    INK,
    INK_3,
    LINE_STRONG,
    RADIUS_CARD,
    RADIUS_CONTROL,
    SURFACE,
)




_MAX_ROWS = 14







_SYNONYM_MIN_CHARS = 3



_VISIBLE_ROWS = 8




_RECENT_SCAN = 40





_POPUP_QSS = scale_qss_font_px(
    f"QListView {{ background: {SURFACE}; color: {INK};"
    f" font-size: {FONT_BASE}px;"
    f" border: 1px solid {LINE_STRONG}; border-radius: {RADIUS_CARD}px;"
    " padding: 6px; outline: none; }"
    "QListView::item { border: none; background: transparent; }"
    "QListView::item:selected { background: transparent; }"
    "QListView QScrollBar:vertical { background: transparent; width: 8px;"
    " margin: 2px 2px 2px 0; border: none; }"
    f"QListView QScrollBar::handle:vertical {{ background: {LINE_STRONG};"
    " border-radius: 3px; min-height: 24px; }"
    f"QListView QScrollBar::handle:vertical:hover {{ background: {INK_3}; }}"
    "QListView QScrollBar::add-line:vertical, QListView QScrollBar::sub-line:vertical {"
    " height: 0px; width: 0px; border: none; background: none; }"
    "QListView QScrollBar::up-arrow:vertical, QListView QScrollBar::down-arrow:vertical {"
    " height: 0px; width: 0px; image: none; }"
    "QListView QScrollBar::add-page:vertical, QListView QScrollBar::sub-page:vertical {"
    " background: transparent; }"
)







_ROW_HEIGHT = 30


_ROW_TEXT_INSET = 9



_ROW_WASH_CURRENT = HOVER_ON
_ROW_WASH_HOVER = HOVER

_TOKEN_ROLE = Qt.ItemDataRole.UserRole + 1


def word_start_hit(haystack: str, query: str) -> bool:











    if not query:
        return False
    return haystack.startswith(query) or f" {query}" in haystack





_WHOLE_WORD_SCRIPTS = (
    (0x3040, 0x30FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xAC00, 0xD7A3),
    (0xF900, 0xFAFF),
    (0x20000, 0x2FA1F),
)


def writes_a_whole_word(query: str) -> bool:


    return any(
        any(low <= ord(char) <= high for low, high in _WHOLE_WORD_SCRIPTS)
        for char in query
    )


def synonyms_may_answer(query: str) -> bool:








    return (len(query) >= prompt_suggest_synonym_min_chars(_SYNONYM_MIN_CHARS)
            or writes_a_whole_word(query))




_SEARCH_SEPARATORS = re.compile(r"[\W_]+", re.UNICODE)


def fold_for_search(text) -> str:








    return _SEARCH_SEPARATORS.sub(" ", fold_search_text(text)).strip()


def prompt_names_entry(query: str, entry: dict) -> bool:





    return bool(query) and (
        query == entry.get("folded_token")
        or query in (entry.get("folded_labels") or ())
        or query == entry.get("folded_label"))


def prompt_completes_entry(query: str, entry: dict) -> bool:

    return bool(query) and (
        str(entry.get("folded_label") or "").startswith(query)
        or str(entry.get("folded_token") or "").startswith(query))


def prompt_enter_token(query: str, entries: list[dict],
                       current: dict | None = None,
                       picked: bool = False) -> str | None:









    if not query:
        return None
    for entry in entries or ():
        if prompt_names_entry(query, entry):
            return entry.get("token") or None
    if current is not None and (picked or prompt_completes_entry(query, current)):
        return current.get("token") or None
    return None


class _PromptEnterFilter(QObject):









    def __init__(self, on_enter, on_move, parent=None) -> None:
        super().__init__(parent)
        self._on_enter = on_enter
        self._on_move = on_move

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        try:
            if event.type() != QEvent.Type.KeyPress:
                return False
            key = event.key()
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                return bool(self._on_enter())
            if key in (Qt.Key.Key_Up, Qt.Key.Key_Down,
                       Qt.Key.Key_PageUp, Qt.Key.Key_PageDown):
                self._on_move()
        except (RuntimeError, AttributeError):
            return False
        return False


class PromptSuggestList(QListView):








    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.hovered_row = -1




        self.viewport().installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802




        try:
            if obj is self.viewport() and event.type() == QEvent.Type.Leave:
                self._set_hovered_row(-1)
            return super().eventFilter(obj, event)
        except (RuntimeError, AttributeError):
            return False

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        self._set_hovered_row(self.indexAt(event_pos(event)).row())
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._set_hovered_row(-1)
        super().leaveEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802
        self.hovered_row = -1
        super().hideEvent(event)

    def _set_hovered_row(self, row: int) -> None:
        if row != self.hovered_row:
            self.hovered_row = row
            self.viewport().update()


class PromptRowDelegate(QStyledItemDelegate):













    def paint(self, painter: QPainter, option, index) -> None:


        try:
            self._paint_prompt_row(painter, option, index)
        except Exception:  # noqa: BLE001
            return

    def _paint_prompt_row(self, painter: QPainter, option, index) -> None:
        opt = option
        self.initStyleOption(opt, index)
        view = self.parent()
        row = index.row()
        try:
            if view.currentIndex().row() == row:
                tint = token_qcolor(_ROW_WASH_CURRENT)
            elif getattr(view, "hovered_row", -1) == row:
                tint = token_qcolor(_ROW_WASH_HOVER)
            else:
                tint = None
        except (RuntimeError, AttributeError):
            tint = None
        painter.save()
        if tint is not None:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(tint)
            painter.drawRoundedRect(
                opt.rect.adjusted(0, 1, 0, -1), RADIUS_CONTROL, RADIUS_CONTROL)
        painter.setPen(token_qcolor(INK))
        painter.setFont(opt.font)
        text_rect = opt.rect.adjusted(_ROW_TEXT_INSET, 0, -_ROW_TEXT_INSET, 0)

        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            painter.fontMetrics().elidedText(
                opt.text, Qt.TextElideMode.ElideRight, text_rect.width()))
        painter.restore()


class DockAutoPromptSuggestMixin:



    def install_prompt_suggest(self) -> None:


        if getattr(self, "_prompt_suggest_completer", None) is not None:
            return
        self._prompt_suggest_index: list[dict] = []
        self._prompt_suggest_index_revision: str | None = None
        self._prompt_suggest_muted = False

        model = QStandardItemModel(0, 1, self.auto_prompt_input)
        view = PromptSuggestList()
        view.setStyleSheet(_POPUP_QSS)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)


        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)


        view.setUniformItemSizes(True)


        view.setFrameShape(QFrame.Shape.NoFrame)

        view.setSelectionMode(QListView.SelectionMode.SingleSelection)
        view.setSelectionBehavior(QListView.SelectionBehavior.SelectRows)



        view.setMouseTracking(True)



        view.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        view.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)


        view.viewport().setCursor(Qt.CursorShape.PointingHandCursor)

        completer = QCompleter(model, self.auto_prompt_input)
        completer.setPopup(view)


        view.setItemDelegate(PromptRowDelegate(view))


        completer.setCompletionMode(QCompleter.CompletionMode.UnfilteredPopupCompletion)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)



        completer.setCompletionRole(_TOKEN_ROLE)


        completer.setMaxVisibleItems(prompt_suggest_visible_rows(_VISIBLE_ROWS))
        completer.activated[QModelIndex].connect(self._on_prompt_suggest_chosen)



        self._prompt_suggest_picked = False
        self._prompt_suggest_enter_filter = _PromptEnterFilter(
            self._on_prompt_suggest_enter, self._on_prompt_suggest_moved, view)
        view.installEventFilter(self._prompt_suggest_enter_filter)

        self._prompt_suggest_model = model
        self._prompt_suggest_completer = completer
        self.auto_prompt_input.setCompleter(completer)

    def refresh_prompt_suggestions(self, text: str) -> None:






        completer = getattr(self, "_prompt_suggest_completer", None)
        if completer is None or getattr(self, "_prompt_suggest_muted", False):
            return


        self._prompt_suggest_picked = False
        query = fold_for_search(text)



        if not query or self._prompt_suggest_is_settled(query):
            self._prompt_suggest_close(completer)
            return
        rows = self._prompt_suggest_rank(query)
        if not rows:
            self._prompt_suggest_close(completer)
            return
        changed = self._prompt_suggest_fill(rows)
        popup = completer.popup()




        if not popup.isVisible() or changed:
            completer.complete()
            self._prompt_suggest_size_popup(popup, len(rows))




        self._prompt_suggest_arm_first_row(popup, rows[0], query)




        first = rows[0]
        safe_single_shot(0, self, lambda: self._prompt_suggest_arm_first_row(
            popup, first, query))

    def _prompt_suggest_arm_first_row(self, popup, first: dict,
                                      query: str) -> None:










        try:
            if getattr(self, "_prompt_suggest_picked", False):
                return
            if fold_for_search(self.auto_prompt_input.text()) != query:
                return
            index = popup.model().index(0, 0)
            selection = popup.selectionModel()
            if selection is None or not index.isValid():
                return
            if (prompt_names_entry(query, first)
                    or prompt_completes_entry(query, first)):
                if selection.currentIndex().row() != 0:
                    selection.setCurrentIndex(
                        index, QItemSelectionModel.SelectionFlag.NoUpdate)
            elif selection.currentIndex().isValid():
                selection.setCurrentIndex(
                    QModelIndex(), QItemSelectionModel.SelectionFlag.NoUpdate)
            popup.viewport().update()
        except (RuntimeError, AttributeError):
            return

    def _on_prompt_suggest_moved(self) -> None:

        self._prompt_suggest_picked = True

    def _on_prompt_suggest_enter(self) -> bool:






        completer = getattr(self, "_prompt_suggest_completer", None)
        if completer is None:
            return False
        popup = completer.popup()
        if not popup.isVisible():
            return False
        entries = self._prompt_suggest_entries()
        query = fold_for_search(self.auto_prompt_input.text())
        current = None
        index = popup.currentIndex()
        if index.isValid():
            token = str(index.data(_TOKEN_ROLE) or "")
            current = next((e for e in entries if e["token"] == token), None)
        token = prompt_enter_token(
            query, entries, current,
            picked=bool(getattr(self, "_prompt_suggest_picked", False)))
        popup.hide()
        if token:
            self._on_prompt_suggest_chosen_token(token)
        else:
            try:
                self._on_auto_prompt_editing_finished()
            except (RuntimeError, AttributeError):
                pass
        return True

    def _prompt_suggest_close(self, completer) -> None:








        try:
            model = self._prompt_suggest_model
            if model.rowCount():
                model.removeRows(0, model.rowCount())
            completer.popup().hide()
        except (RuntimeError, AttributeError):
            pass



    def _prompt_suggest_rank(self, query: str) -> list[dict]:








        entries = self._prompt_suggest_entries()
        if not entries:
            return []
        recent = self._prompt_suggest_recent_tokens()
        scored: list[tuple[int, int, dict]] = []
        for order, entry in enumerate(entries):
            label = entry["folded_label"]
            token = entry["folded_token"]
            if prompt_names_entry(query, entry):


                scored.append((-1, order, entry))
                continue
            if label.startswith(query) or token.startswith(query):
                band = 1 if entry["top_pick"] else 2
            elif word_start_hit(label, query) or word_start_hit(token, query):
                band = 3
            elif synonyms_may_answer(query) and word_start_hit(entry["haystack"], query):
                band = 4
            elif synonyms_may_answer(query) and word_start_hit(
                    entry["folded_category"], query):



                band = 5
            else:
                continue
            if band < 5 and entry["token"].strip().lower() in recent:
                band = 0
            scored.append((band, order, entry))
        if any(row[0] < 5 for row in scored):
            scored = [row for row in scored if row[0] < 5]
        scored.sort(key=lambda row: (row[0], row[1]))
        max_rows = prompt_suggest_max_rows(_MAX_ROWS)
        out: list[dict] = []
        seen: set[str] = set()
        for _band, _order, entry in scored:
            if entry["token"] in seen:
                continue
            seen.add(entry["token"])
            out.append(entry)
            if len(out) >= max_rows:
                break
        return out

    def _prompt_suggest_is_settled(self, query: str) -> bool:








        return any(
            query == entry["folded_token"]
            for entry in self._prompt_suggest_entries()
        )



    def _prompt_suggest_entries(self) -> list[dict]:






        try:
            revision = catalog_revision()
        except Exception:  # noqa: BLE001
            revision = ""
        cached = getattr(self, "_prompt_suggest_index", None)
        if cached and getattr(self, "_prompt_suggest_index_revision", None) == revision:
            return cached
        try:
            from ...core.presets.segmentation_presets import merged_categories
            from ...core.presets.segmentation_presets_client import cached_or_offline_catalog

            categories = merged_categories(cached_or_offline_catalog()[0])
        except Exception:  # noqa: BLE001
            categories = []
        entries: list[dict] = []
        for category in categories or []:
            if not isinstance(category, dict):
                continue
            category_label = pick_label(category.get("label"), "")
            for preset in category.get("presets", []) or []:
                if not isinstance(preset, dict):
                    continue
                token = str(preset.get("prompt") or "").strip()
                if not token:
                    continue
                label = pick_label(preset.get("label"), token)
                labels = preset.get("label")
                all_labels = (labels.values() if isinstance(labels, dict)
                              else [label])
                entries.append({
                    "token": token,
                    "label": label,
                    "folded_token": fold_for_search(token),
                    "folded_label": fold_for_search(label),
                    "folded_labels": frozenset(
                        fold_for_search(value) for value in all_labels if value),


                    "haystack": fold_for_search(
                        preset_search_haystack(preset, "")),
                    "folded_category": fold_for_search(category_label),
                    "top_pick": bool(preset.get("top_pick")),
                })
        self._prompt_suggest_index = entries
        self._prompt_suggest_index_revision = revision
        return entries

    def forget_prompt_suggest_recent(self) -> None:






        self._prompt_suggest_recent_cache = None

    def _prompt_suggest_recent_tokens(self) -> set[str]:









        cached = getattr(self, "_prompt_suggest_recent_cache", None)
        if cached is not None:
            return cached
        try:
            from ...core.presets.segment_history import get_recent

            recent = get_recent()[:prompt_suggest_recent_scan(_RECENT_SCAN)]
        except Exception:  # noqa: BLE001
            self._prompt_suggest_recent_cache = set()
            return self._prompt_suggest_recent_cache
        tokens: set[str] = set()
        for item in recent:
            prompt = item.get("prompt") if isinstance(item, dict) else item
            text = str(prompt or "").strip().lower()
            if text:
                tokens.add(text)
        self._prompt_suggest_recent_cache = tokens
        return tokens

    def _prompt_suggest_fill(self, rows: list[dict]) -> bool:










        model = self._prompt_suggest_model
        before = model.rowCount()
        for row, entry in enumerate(rows):
            item = model.item(row)
            if item is None:
                item = QStandardItem()
                item.setEditable(False)
                item.setSizeHint(QSize(0, scale_px_length(_ROW_HEIGHT)))
                model.appendRow(item)
            if item.text() != entry["label"]:
                item.setText(entry["label"])
            if item.data(_TOKEN_ROLE) != entry["token"]:
                item.setData(entry["token"], _TOKEN_ROLE)
        extra = model.rowCount() - len(rows)
        if extra > 0:
            model.removeRows(len(rows), extra)
        return model.rowCount() != before

    def _prompt_suggest_size_popup(self, popup, row_count: int) -> None:







        try:
            chrome = popup.height() - popup.viewport().height()
            shown = min(row_count, prompt_suggest_visible_rows(_VISIBLE_ROWS))
            popup.setFixedHeight(
                shown * scale_px_length(_ROW_HEIGHT)
                + max(chrome, 2 * popup.frameWidth()))
        except (RuntimeError, AttributeError):
            pass



    def _on_prompt_suggest_chosen(self, index) -> None:








        token = str(index.data(_TOKEN_ROLE) or "").strip()
        if not token:
            return
        self._on_prompt_suggest_chosen_token(token)

    def _on_prompt_suggest_chosen_token(self, token: str) -> None:

        self._prompt_suggest_muted = True



        self._prompt_from_library = True



        safe_single_shot(0, self, lambda: self._prompt_suggest_settle(token))

    def _prompt_suggest_settle(self, token: str) -> None:
        try:
            self.auto_prompt_input.setText(token)
            self._prompt_suggest_completer.popup().hide()
        except (RuntimeError, AttributeError):
            return
        finally:
            self._prompt_suggest_muted = False
        try:
            self._on_auto_prompt_editing_finished()
        except (RuntimeError, AttributeError):
            pass
