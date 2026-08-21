"""Automatic mode prompt suggestions: the drop-down that opens under the prompt
box on the first letter and offers the catalogue objects in the user's own
language.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py); split out
so agents and humans work on one concern per file. Methods are plain mixin
members: widgets/signals live on the dock instance.

Two facts shape this file.

The catalogue already carries every object's label in twelve languages plus its
extra search words, and every entry already carries the English token the model
is sent. So the list shows the label the user reads and hands the box the
English token, with no dictionary of our own to keep in step. That is the same
lever the prompt box's silent swap already uses; here it happens before the
mistake instead of after it.

The rows say the object and nothing else. No icon, no English gloss beside the
translated word, no badge: a column of coloured pictograms reads as cheap, and
the token that lands in the box is visible in the box a moment later anyway.

Matching runs over every language at once, not just the interface one, because
people mix languages in a search box: a French user on an English QGIS still
types "eolienne". Ranking puts what this user ran before first, then the
objects most people ask for, then everything the query touches through a
synonym or another language.
"""
from __future__ import annotations

import re

from qgis.PyQt.QtCore import QEvent, QItemSelectionModel, QModelIndex, QSize, Qt, QTimer
from qgis.PyQt.QtGui import QColor, QPainter, QStandardItem, QStandardItemModel
from qgis.PyQt.QtWidgets import QCompleter, QFrame, QListView, QStyledItemDelegate

from ...core.presets.segmentation_presets import (
    catalog_revision,
    fold_search_text,
    pick_label,
    preset_search_haystack,
)
from ...core.qt_compat import event_pos
from .styles import BRAND_BLUE

# How many rows the list shows. Past this the popup stops reading as a shortcut
# and starts reading as a catalogue, which the Library button already covers.
_MAX_ROWS = 14

# Shortest Latin-script query that may reach the cross-language synonyms. Under
# three letters, matching twelve languages at once answers "ba" with the German
# "Baum", the Spanish "bodega" and the Portuguese "banco" all at once, which
# buries the objects the user can actually see the name of. Their own language
# and the English token still match from the first letter. Scripts that write a
# whole word in one or two characters are exempt (see synonyms_may_answer).
_SYNONYM_MIN_CHARS = 3

# Rows shown at once. Past this the list scrolls rather than growing over the
# cards under it.
_VISIBLE_ROWS = 8

# Recent objects considered for the top of the list. The store keeps up to 200,
# far more than anyone scans, and a long recent block pushes the curated
# objects off the visible rows.
_RECENT_SCAN = 40

# The list follows the input it hangs under: palette roles so both QGIS themes
# work, the same faint border and radius, and the brand blue for the row under
# the cursor. Nothing is tinted on the edges of a row.
_POPUP_QSS = (
    "QListView { background: palette(base); color: palette(text);"
    " font-size: 12px;"
    " border: 1px solid rgba(128, 128, 128, 0.35); border-radius: 6px;"
    " padding: 3px; outline: none; }"
    "QListView::item { padding: 0px 9px; border-radius: 4px; }"
    "QListView::item:selected { background: rgba(30, 136, 229, 0.35);"
    " color: palette(text); }"
    "QListView::item:hover { background: rgba(30, 136, 229, 0.22); }"
    "QListView::item:selected:hover { background: rgba(30, 136, 229, 0.42); }"
)

# Row height, in pixels, set on every item.
#
# A stylesheet paints an item's padding but does not report it back as the
# item's size, so Qt keeps sizing the rows to the bare font height and the
# popup ends up a few pixels short of its own rows: a scroll bar appears over
# a list that fits. Stating the height on the item is the half Qt reads.
_ROW_HEIGHT = 30

# Left and right breathing room for a row label.
_ROW_TEXT_INSET = 9

# How strongly the brand blue tints a row: the one Enter would take, and
# the one the pointer is on. The lighter answers without competing.
_TINT_SELECTED = 110
_TINT_HOVER = 56

_TOKEN_ROLE = Qt.ItemDataRole.UserRole + 1


def word_start_hit(haystack: str, query: str) -> bool:
    """Does the query open a word of the haystack?

    A plain substring test reads the middle of unrelated words and offers the
    wrong object: "bat" sits inside the Italian "serbatoio" and would put
    Storage tank under someone typing "bâtiment", "arb" sits inside the
    Spanish "embarcadero" and would put Dock under someone typing "árbol".
    Both haystacks are searched in every language at once, so the odds of an
    accidental hit rise with each language shipped. Matching only at a word
    start keeps the real cross-language hits ("bateau", "bundaran") and drops
    the accidents.
    """
    if not query:
        return False
    return haystack.startswith(query) or f" {query}" in haystack


# Character blocks where one or two characters already write a whole word:
# kana, the CJK ideographs and their extension and compatibility blocks, and
# the Hangul syllables. The three-letter floor below does not apply to them.
_WHOLE_WORD_SCRIPTS = (
    (0x3040, 0x30FF),   # hiragana and katakana
    (0x3400, 0x4DBF),   # CJK ideographs, extension A
    (0x4E00, 0x9FFF),   # CJK ideographs
    (0xAC00, 0xD7A3),   # Hangul syllables
    (0xF900, 0xFAFF),   # CJK compatibility ideographs
    (0x20000, 0x2FA1F),  # CJK ideographs, later extensions
)


def writes_a_whole_word(query: str) -> bool:
    """Does the query use a script where one or two characters are already a
    word rather than a fragment?"""
    return any(
        any(low <= ord(char) <= high for low, high in _WHOLE_WORD_SCRIPTS)
        for char in query
    )


def synonyms_may_answer(query: str) -> bool:
    """Is the query long enough to be worth matching against every language?

    The three-letter floor is about alphabets, where two letters are a
    fragment. A Japanese or Chinese word is one or two characters and is
    already whole, so the floor would silence those scripts entirely: the
    catalogue's own Japanese for roof is two characters. Only those scripts are
    exempt; a two-letter Greek or Cyrillic fragment is as thin as a Latin one.
    """
    return len(query) >= _SYNONYM_MIN_CHARS or writes_a_whole_word(query)


# Anything that is not a letter or a digit, in any script. Folded to a space so
# each part of a compound name opens a word.
_SEARCH_SEPARATORS = re.compile(r"[\W_]+", re.UNICODE)


def fold_for_search(text) -> str:
    """Accent-fold, then break every separator into a space so each part of a
    compound name ("rond-point", "plan d'eau", "ouvrage d'art") opens a word the
    search can land on.

    Hyphens and slashes alone were not enough: an apostrophe left "d'art" as one
    word, so "art" opened nothing and the object could not be reached under the
    name its own language gives it.
    """
    return _SEARCH_SEPARATORS.sub(" ", fold_search_text(text)).strip()


class PromptSuggestList(QListView):
    """The pop-up list itself, which keeps track of the row under the pointer.

    Qt does not hand that row to the delegate here. The list is a pop-up, so it
    holds a mouse grab and its hover state never settles, and the stylesheet's
    ``::item:hover`` rule stays dark for the same reason. Following the pointer
    by hand is a few lines and it always paints.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.hovered_row = -1
        # The rows live in the viewport, and so does the pointer: the leave
        # event that ends a hover is delivered THERE, not to the list. Without
        # this filter, moving from a row onto the scroll bar leaves the row it
        # came from lit under a pointer that is no longer on it.
        self.viewport().installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 -- Qt name
        try:
            if obj is self.viewport() and event.type() == QEvent.Type.Leave:
                self._set_hovered_row(-1)
        except (RuntimeError, AttributeError):
            pass
        return super().eventFilter(obj, event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 -- Qt name
        self._set_hovered_row(self.indexAt(event_pos(event)).row())
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 -- Qt name
        self._set_hovered_row(-1)
        super().leaveEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802 -- Qt name
        self.hovered_row = -1
        super().hideEvent(event)

    def _set_hovered_row(self, row: int) -> None:
        if row != self.hovered_row:
            self.hovered_row = row
            self.viewport().update()


class PromptRowDelegate(QStyledItemDelegate):
    """Paints one row of the suggestion list.

    Two highlights, and they mean different things. The strong one marks the
    row Enter would take. The lighter one marks the row the pointer is on, so
    moving the mouse through the list answers without deciding anything: an
    earlier attempt to move the selection under the pointer made Qt copy the
    word into the box and close the list, which is not what pointing at
    something means.

    Both are read from the list rather than from the state Qt reports, which
    carries neither inside a pop-up.
    """

    def paint(self, painter: QPainter, option, index) -> None:
        opt = option
        self.initStyleOption(opt, index)
        view = self.parent()
        row = index.row()
        tint = QColor(BRAND_BLUE)
        try:
            if view.currentIndex().row() == row:
                tint.setAlpha(_TINT_SELECTED)
            elif getattr(view, "hovered_row", -1) == row:
                tint.setAlpha(_TINT_HOVER)
            else:
                tint = None
        except (RuntimeError, AttributeError):
            tint = None
        painter.save()
        if tint is not None:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(tint)
            painter.drawRoundedRect(opt.rect.adjusted(2, 1, -2, -1), 4, 4)
        painter.setPen(opt.palette.color(opt.palette.ColorRole.Text))
        painter.setFont(opt.font)
        painter.drawText(
            opt.rect.adjusted(_ROW_TEXT_INSET, 0, -_ROW_TEXT_INSET, 0),
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            opt.text)
        painter.restore()


class DockAutoPromptSuggestMixin:
    """The drop-down under the Automatic prompt box: catalogue objects in the
    user's language, committed to the box as the English model token."""

    def install_prompt_suggest(self) -> None:
        """Attach the drop-down to the prompt box. Called once, right after
        the box is built, and safe to call again (it rebuilds nothing)."""
        if getattr(self, "_prompt_suggest_completer", None) is not None:
            return
        self._prompt_suggest_index: list[dict] = []
        self._prompt_suggest_index_revision: str | None = None
        self._prompt_suggest_muted = False

        model = QStandardItemModel(0, 1, self.auto_prompt_input)
        view = PromptSuggestList()
        view.setStyleSheet(_POPUP_QSS)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # The list is sized to its rows below, so the bar only ever appears
        # when there are more matches than fit.
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Every row is one line of the same height, so Qt can lay the list out
        # without measuring each row.
        view.setUniformItemSizes(True)
        # Qt frames a popup view itself. Left on, that frame doubles the border
        # the stylesheet draws and squares off the rounded corners.
        view.setFrameShape(QFrame.Shape.NoFrame)
        # One row is picked at a time, and the whole row is the target.
        view.setSelectionMode(QListView.SelectionMode.SingleSelection)
        view.setSelectionBehavior(QListView.SelectionBehavior.SelectRows)
        # Without mouse tracking a QListView never sees the pointer between
        # clicks, so the :hover rule in the stylesheet never fires and the rows
        # sit dead under the mouse.
        view.setMouseTracking(True)
        # Mouse tracking alone is not enough: the :hover rule reads a hover
        # state Qt only produces for widgets that ask for it, so without
        # WA_Hover on the viewport the rows stay dead under the pointer.
        view.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        view.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        # The hand says the row is a target before the user tries it, the same
        # way the Library chip beside the box does.
        view.viewport().setCursor(Qt.CursorShape.PointingHandCursor)

        completer = QCompleter(model, self.auto_prompt_input)
        completer.setPopup(view)
        # After setPopup, never before: it installs a delegate of its own
        # and would throw ours away.
        view.setItemDelegate(PromptRowDelegate(view))
        # The rows are ranked here, not by Qt: the popup shows the model as it
        # stands, and every keystroke rewrites the model.
        completer.setCompletionMode(QCompleter.CompletionMode.UnfilteredPopupCompletion)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        # The row displays the localized label; what lands in the box is the
        # English token stored beside it, because that is the string the model
        # is sent and the string the run is billed under.
        completer.setCompletionRole(_TOKEN_ROLE)
        # _MAX_ROWS caps what is ranked; this caps how much of it is on
        # screen, so a long answer scrolls instead of covering the step below.
        completer.setMaxVisibleItems(_VISIBLE_ROWS)
        completer.activated[QModelIndex].connect(self._on_prompt_suggest_chosen)

        self._prompt_suggest_model = model
        self._prompt_suggest_completer = completer
        self.auto_prompt_input.setCompleter(completer)

    def refresh_prompt_suggestions(self, text: str) -> None:
        """Rewrite and re-open the list for what is in the box right now.

        Called from the prompt box's textChanged handler, so it runs on every
        keystroke and must stay cheap: the catalogue index is built once and
        the per-keystroke work is a folded substring pass over 42 entries.
        """
        completer = getattr(self, "_prompt_suggest_completer", None)
        if completer is None or getattr(self, "_prompt_suggest_muted", False):
            return
        query = fold_for_search(text)
        # The list is a shortcut while a word is being formed, not a commentary
        # on a finished one: a query that already IS an object has nothing left
        # to offer, and a popup over the Detect button at that moment is noise.
        if not query or self._prompt_suggest_is_settled(query):
            self._prompt_suggest_close(completer)
            return
        rows = self._prompt_suggest_rank(query)
        if not rows:
            self._prompt_suggest_close(completer)
            return
        changed = self._prompt_suggest_fill(rows)
        popup = completer.popup()
        # An open list stays open. complete() re-lays the popup out and Qt
        # re-shows it, which reads as the list blinking away and back on every
        # keystroke. So it is called to OPEN the list, and to resize it when
        # the number of rows moves, never just because a letter was typed.
        if not popup.isVisible() or changed:
            completer.complete()
            self._prompt_suggest_size_popup(popup, len(rows))
        # The first row carries the current row, which is what Enter takes and
        # what the strong highlight marks. The pointer gets its own lighter
        # highlight, so the row under the mouse and the row Enter would take
        # are both readable at once (see PromptRowDelegate).
        self._prompt_suggest_arm_first_row(popup)

    def _prompt_suggest_arm_first_row(self, popup) -> None:
        """Make row 0 the current row again when nothing is current.

        Rewriting the rows in place resets the completer's own model, and a
        reset drops the current row: the list then reads as armed, the strong
        highlight is gone, and Enter falls through to the box instead of taking
        a suggestion.

        The row is made current WITHOUT selecting it. A selection makes Qt copy
        that row into the box, which turns pointing at the list into a commit
        nobody asked for.
        """
        try:
            index = popup.model().index(0, 0)
            selection = popup.selectionModel()
            if selection is None or not index.isValid():
                return
            if selection.currentIndex().isValid():
                return
            selection.setCurrentIndex(
                index, QItemSelectionModel.SelectionFlag.NoUpdate)
        except (RuntimeError, AttributeError):
            return

    def _prompt_suggest_close(self, completer) -> None:
        """Empty the list and put it away.

        Emptying it is the half that matters. A QLineEdit drives its completer
        itself on every edit, so Qt re-opens the pop-up whether or not this
        code asks it to, and it shows whatever the model happens to hold. Left
        full, the model still holds the last query's answer, and a letter that
        matches nothing comes back wearing the previous word's suggestions.
        """
        try:
            model = self._prompt_suggest_model
            if model.rowCount():
                model.removeRows(0, model.rowCount())
            completer.popup().hide()
        except (RuntimeError, AttributeError):
            pass

    # --- ranking ------------------------------------------------------

    def _prompt_suggest_rank(self, query: str) -> list[dict]:
        """The rows to show, best first, capped at _MAX_ROWS.

        Four bands, and the reason for the order is what the user is doing:
        someone typing three letters is almost always reaching for a word they
        already used, then for a common object, and only then for something
        they know under another name.
        """
        entries = self._prompt_suggest_entries()
        if not entries:
            return []
        recent = self._prompt_suggest_recent_tokens()
        scored: list[tuple[int, int, dict]] = []
        for order, entry in enumerate(entries):
            label = entry["folded_label"]
            token = entry["folded_token"]
            if label.startswith(query) or token.startswith(query):
                band = 1 if entry["top_pick"] else 2
            elif word_start_hit(label, query) or word_start_hit(token, query):
                band = 3
            elif synonyms_may_answer(query) and word_start_hit(entry["haystack"], query):
                band = 4
            else:
                continue
            if entry["token"].strip().lower() in recent:
                band = 0
            scored.append((band, order, entry))
        scored.sort(key=lambda row: (row[0], row[1]))
        out: list[dict] = []
        seen: set[str] = set()
        for _band, _order, entry in scored:
            if entry["token"] in seen:
                continue
            seen.add(entry["token"])
            out.append(entry)
            if len(out) >= _MAX_ROWS:
                break
        return out

    def _prompt_suggest_is_settled(self, query: str) -> bool:
        """True when the query is already the English token the run is sent.

        The token, and not the label the user reads. A label is exactly what
        the list has something to answer with: someone who typed "piscine" or
        "屋根" in full still needs the row that hands the box the English token,
        and hiding the list there leaves the swap to happen later, out of
        sight, or not at all.
        """
        return any(
            query == entry["folded_token"]
            for entry in self._prompt_suggest_entries()
        )

    # --- data ---------------------------------------------------------

    def _prompt_suggest_entries(self) -> list[dict]:
        """The flattened catalogue, built once per catalogue revision.

        Reads the cached server catalogue when there is one and the shipped
        fallback otherwise. Never touches the network: this runs on the GUI
        thread, on every keystroke.
        """
        try:
            revision = catalog_revision()
        except Exception:  # noqa: BLE001 -- a stamp we cannot read is no stamp
            revision = ""
        cached = getattr(self, "_prompt_suggest_index", None)
        if cached and getattr(self, "_prompt_suggest_index_revision", None) == revision:
            return cached
        try:
            from ...core.presets.segmentation_presets import merged_categories
            from ...core.presets.segmentation_presets_client import cached_or_offline_catalog

            categories = merged_categories(cached_or_offline_catalog()[0])
        except Exception:  # noqa: BLE001 -- an unreadable catalogue means no list, never a crash
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
                entries.append({
                    "token": token,
                    "label": label,
                    "folded_token": fold_for_search(token),
                    "folded_label": fold_for_search(label),
                    "haystack": fold_for_search(
                        preset_search_haystack(preset, category_label)),
                    "top_pick": bool(preset.get("top_pick")),
                })
        self._prompt_suggest_index = entries
        self._prompt_suggest_index_revision = revision
        return entries

    def _prompt_suggest_recent_tokens(self) -> set[str]:
        """The English tokens this user already ran, lowercased for comparison.

        A history entry is a record ({prompt, ts, ...}), not a string: the
        prompt has to be read out of it, or nothing ever matches and the band
        that puts the user's own objects first is dead.
        """
        try:
            from ...core.presets.segment_history import get_recent

            recent = get_recent()[:_RECENT_SCAN]
        except Exception:  # noqa: BLE001 -- no history is a normal first run
            return set()
        tokens: set[str] = set()
        for item in recent:
            prompt = item.get("prompt") if isinstance(item, dict) else item
            text = str(prompt or "").strip().lower()
            if text:
                tokens.add(text)
        return tokens

    def _prompt_suggest_fill(self, rows: list[dict]) -> bool:
        """Write the ranked rows into the popup model, in place.

        Returns True when the number of rows moved, which is the only case
        that needs the popup laid out again.

        Rows are edited rather than rebuilt: clearing the model resets it, and
        a reset empties the list for a frame before the new rows arrive, which
        is the flicker. Editing keeps the same rows on screen and just changes
        what they say.
        """
        model = self._prompt_suggest_model
        before = model.rowCount()
        for row, entry in enumerate(rows):
            item = model.item(row)
            if item is None:
                item = QStandardItem()
                item.setEditable(False)
                item.setSizeHint(QSize(0, _ROW_HEIGHT))
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
        """Make the list exactly as tall as the rows it holds.

        Qt sizes the popup from its own idea of a row height and lands a couple
        of pixels short of the height the rows actually take, which puts a
        scroll bar on a list that fits. The row count is already capped, so the
        honest answer is to state the height rather than let it be guessed.
        """
        try:
            chrome = popup.height() - popup.viewport().height()
            shown = min(row_count, _VISIBLE_ROWS)
            popup.setFixedHeight(shown * _ROW_HEIGHT + max(chrome, 2 * popup.frameWidth()))
        except (RuntimeError, AttributeError):
            pass

    # --- selection ----------------------------------------------------

    def _on_prompt_suggest_chosen(self, index) -> None:
        """A row was picked: put the English token in the box and settle it.

        Qt writes the completion into the line edit around this signal, so the
        box is set on the next turn of the event loop to make sure the token
        is what stays there. Settling it then runs the same path as Enter or
        focus-out, which is what re-seeds the detail level and fires the one
        commit for this prompt.
        """
        token = str(index.data(_TOKEN_ROLE) or "").strip()
        if not token:
            return
        self._prompt_suggest_muted = True
        # The commit telemetry already separates a prompt taken from our
        # curated vocabulary from one typed freehand; the drop-down serves the
        # same vocabulary, so it reports itself the same way.
        self._prompt_from_library = True
        QTimer.singleShot(0, lambda: self._prompt_suggest_settle(token))

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
