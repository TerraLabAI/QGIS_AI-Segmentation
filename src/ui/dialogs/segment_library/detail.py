"""Detail popups: template preview, past-run detail, and run export.

Both detail dialogs share the AI Edit generation-detail layout: the
before/after image fills the left pane at its native aspect ratio with a
floating fullscreen toggle, and a scrollable info panel (badge, title,
prompt + copy, metadata chips) sits on the right above a pinned action
footer. The divider parks at 50/50 and follows the cursor on hover - no
auto-loop animation.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import Qt, QTimer
from qgis.PyQt.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ....core import qt_compat as QtC
from ....core.i18n import tr
from ....core.presets.segmentation_presets import pick_label
from ...before_after_slider import BeforeAfterSlider
from ...template_demo_loader import TemplateDemoLoader
from .common import (
    _ACTION_BTN,
    _AspectBox,
    _BADGE_STYLE,
    _CHIP_CAPTION,
    _CHIP_STYLE,
    _CHIP_VALUE,
    _COPY_BTN,
    _demo_url,
    _DETAIL_STAR_BTN,
    _FS_BTN,
    _GHOST_BTN_QSS,
    _iso_norm,
    _PRIMARY_BTN,
    _PROMPT_STYLE,
    _relative_when,
    _SEARCH_QSS,
    _SECTION_STYLE,
    _SEPARATOR,
    _TITLE_STYLE,
)


class _DetailDialogBase(QDialog):
    """Two-pane detail shell: aspect-true slider left, info + actions right.

    Subclasses call ``super().__init__(parent)``, set their own data
    attributes, then ``_build_shell(title, badge)`` (PyQt forbids touching
    ``self`` before the QDialog base is initialized, so the shell cannot be
    built from this __init__ with subclass data)."""

    def __init__(self, parent=None):
        super().__init__(parent)

    def _build_shell(self, title: str, badge_text: str) -> None:
        self.setWindowTitle(title or tr("Details"))
        self.setMinimumSize(560, 420)
        self.setSizeGripEnabled(True)
        self._fullscreen = False
        self._aspect_locked = False

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # --- left: slider at the image's aspect ratio ----------------------
        # No auto-loop: the divider parks at the middle and follows the cursor
        # on hover / drags, which is the calm, smooth interaction (the old
        # oscillating animation pulled the eye and read as glitchy).
        self.slider = BeforeAfterSlider(None, auto_loop=False, show_badges=True)
        self.slider.set_placeholder_text(tr("Loading..."))
        self._aspect_box = _AspectBox(self.slider, 1.0, self)
        self._aspect_box.setMinimumSize(260, 240)
        self._aspect_box.setSizePolicy(
            QtC.SizePolicyExpanding, QtC.SizePolicyExpanding)
        # Fullscreen toggle floats over the image's bottom-right corner, clear
        # of the BEFORE / AFTER badges along the top edge.
        self._fs_btn = QToolButton(self._aspect_box)
        self._fs_btn.setText("⤢")
        self._fs_btn.setToolTip(tr("Fullscreen"))
        self._fs_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._fs_btn.setStyleSheet(_FS_BTN)
        self._fs_btn.setFixedSize(30, 30)
        self._fs_btn.clicked.connect(self._toggle_fullscreen)
        self._aspect_box.set_overlay(self._fs_btn)
        root.addWidget(self._aspect_box, 1)

        # --- right: scrollable info panel + pinned footer -------------------
        right = QWidget(self)
        right.setMinimumWidth(330)
        right.setMaximumWidth(560)
        right_col = QVBoxLayout(right)
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(10)
        self._info_panel = right

        info_scroll = QScrollArea(right)
        info_scroll.setWidgetResizable(True)
        info_scroll.setFrameShape(QtC.FrameNoFrame)
        info_scroll.setHorizontalScrollBarPolicy(QtC.ScrollBarAlwaysOff)

        info = QWidget()
        info.setMinimumWidth(300)
        self._info_col = QVBoxLayout(info)
        self._info_col.setContentsMargins(4, 2, 10, 2)
        self._info_col.setSpacing(12)

        badge_row = QHBoxLayout()
        badge_row.setContentsMargins(0, 0, 0, 0)
        badge = QLabel(badge_text.upper())
        badge.setStyleSheet(_BADGE_STYLE)
        badge_row.addWidget(badge)
        badge_row.addStretch(1)
        self._info_col.addLayout(badge_row)

        title_lbl = QLabel(title)
        title_lbl.setWordWrap(True)
        title_lbl.setTextFormat(QtC.PlainText)
        title_lbl.setStyleSheet(_TITLE_STYLE)
        self._info_col.addWidget(title_lbl)

        self._build_info()  # subclass content between title and stretch
        self._info_col.addStretch(1)
        info_scroll.setWidget(info)
        right_col.addWidget(info_scroll, 1)

        # Pinned footer: separator + subclass actions. Always on screen, never
        # scrolled away no matter how tall the info content runs.
        footer = QWidget(right)
        footer_col = QVBoxLayout(footer)
        footer_col.setContentsMargins(4, 0, 10, 2)
        footer_col.setSpacing(10)
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(_SEPARATOR)
        footer_col.addWidget(sep)
        self._build_footer(footer_col)
        right_col.addWidget(footer, 0)

        root.addWidget(right, 1)
        self._apply_image_size(1.0)

    # -- subclass hooks -------------------------------------------------------

    def _build_info(self) -> None:
        raise NotImplementedError

    def _build_footer(self, col: QVBoxLayout) -> None:
        raise NotImplementedError

    # -- sizing ----------------------------------------------------------------

    def _lock_aspect_from(self, pixmap) -> None:
        """Adopt the first loaded pixmap's aspect ratio: the slider pane takes
        the image shape (portrait stays portrait) and the window resizes so
        the whole demo is visible without cropping."""
        if self._aspect_locked or pixmap is None or pixmap.isNull():
            return
        if pixmap.height() <= 0:
            return
        self._aspect_locked = True
        ratio = pixmap.width() / pixmap.height()
        self._aspect_box.set_ratio(ratio)
        if not self._fullscreen:
            self._apply_image_size(ratio)

    def _apply_image_size(self, ratio: float) -> None:
        ratio = ratio if ratio > 0 else 1.0
        info_w = 380
        disp_h = 560.0
        disp_w = disp_h * ratio
        max_w = 860.0
        if disp_w > max_w:
            disp_w = max_w
            disp_h = disp_w / ratio
        if disp_h < 380.0:
            disp_h = 380.0
            disp_w = min(disp_h * ratio, max_w)
        self.resize(int(disp_w) + info_w + 12 + 24, int(disp_h) + 24)

    # -- fullscreen -------------------------------------------------------------

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        self._info_panel.setVisible(not self._fullscreen)
        self._fs_btn.setText("⤡" if self._fullscreen else "⤢")
        self._fs_btn.setToolTip(
            tr("Exit fullscreen") if self._fullscreen else tr("Fullscreen"))
        # Maximize rather than true fullscreen: on macOS, showFullScreen() moves
        # the dialog to its own Space so the (modal) library ends up covering
        # it. Maximized stays a normal window we can raise above the library.
        if self._fullscreen:
            self.showMaximized()
        else:
            self.showNormal()
        self.raise_()
        self.activateWindow()

    def keyPressEvent(self, event):  # noqa: N802 - Qt signature
        if event.key() == Qt.Key.Key_Escape and self._fullscreen:
            self._toggle_fullscreen()
            return
        super().keyPressEvent(event)

    # -- shared info blocks -------------------------------------------------------

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text.upper())
        lbl.setStyleSheet(_SECTION_STYLE)
        return lbl

    def _build_prompt_block(self, prompt: str) -> QWidget:
        """PROMPT section: the literal English token the cloud model receives, with a
        one-click Copy that flashes 'Copied'."""
        self._prompt_text = prompt
        wrap = QWidget(self)
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        header.addWidget(self._section_label(tr("Prompt")))
        header.addStretch(1)
        self._copy_btn = QPushButton(tr("Copy"))
        self._copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_btn.setFlat(True)
        self._copy_btn.setToolTip(tr("Copy prompt"))
        self._copy_btn.setStyleSheet(_COPY_BTN)
        self._copy_btn.clicked.connect(self._on_copy_prompt)
        header.addWidget(self._copy_btn)
        v.addLayout(header)
        body = QLabel(prompt)
        body.setWordWrap(True)
        body.setTextFormat(QtC.PlainText)
        body.setTextInteractionFlags(QtC.TextSelectableByMouse)
        body.setStyleSheet(_PROMPT_STYLE)
        v.addWidget(body)
        return wrap

    def _on_copy_prompt(self) -> None:
        text = getattr(self, "_prompt_text", "") or ""
        if not text:
            return
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        self._copy_btn.setText(tr("Copied"))
        QTimer.singleShot(1400, self._reset_copy_btn)

    def _reset_copy_btn(self) -> None:
        try:
            self._copy_btn.setText(tr("Copy"))
        except (RuntimeError, AttributeError):
            pass

    def _chip(self, caption: str, value: str) -> QFrame:
        chip = QFrame(self)
        chip.setStyleSheet(_CHIP_STYLE)
        v = QVBoxLayout(chip)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(2)
        cap = QLabel(caption)
        cap.setStyleSheet(_CHIP_CAPTION)
        val = QLabel(value)
        val.setWordWrap(True)
        val.setTextFormat(QtC.PlainText)
        val.setStyleSheet(_CHIP_VALUE)
        v.addWidget(cap)
        v.addWidget(val)
        return chip

    def _chips_grid(self, chips: list[tuple[str, str]]) -> QWidget:
        host = QWidget(self)
        grid = QGridLayout(host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        for idx, (cap, val) in enumerate(chips):
            r, c = divmod(idx, 3)
            grid.addWidget(self._chip(cap, val), r, c)
        return host


class _PresetDetailDialog(_DetailDialogBase):
    """Template detail: big before/after + the token + 'Use this prompt'."""

    def __init__(self, preset: dict, base: str, parent=None,
                 category_label: str = ""):
        super().__init__(parent)
        self._preset = preset
        self._base = base
        self.chosen = False
        title = pick_label(preset.get("label"), preset.get("prompt", ""))
        self._build_shell(title, category_label or tr("Template"))

        # Own loader (parented to self) so it dies with the popup and never
        # emits into a freed dialog. Bigger preview variant first, card-size
        # image as fallback.
        self._loader = TemplateDemoLoader(self)
        self._loader.loaded.connect(self._on_loaded)
        self._loader.failed.connect(self._on_failed)
        self._loader.request(preset["id"], "before_preview",
                             _demo_url(base, preset, "before", preview=True))
        self._loader.request(preset["id"], "after_preview",
                             _demo_url(base, preset, "after", preview=True))
        self._loader.request(preset["id"], "before", _demo_url(base, preset, "before"))
        self._loader.request(preset["id"], "after", _demo_url(base, preset, "after"))

    def _build_info(self) -> None:
        self._info_col.addWidget(
            self._build_prompt_block(self._preset.get("prompt", "")))
        if self._preset.get("weak"):
            note = QLabel(tr(
                "Fuzzy edges: this one may need cleanup after detection."))
            note.setWordWrap(True)
            note.setStyleSheet(
                "font-size: 11px; color: palette(text); background: transparent;")
            self._info_col.addWidget(note)

    def _build_footer(self, col: QVBoxLayout) -> None:
        use = QPushButton(tr("Use this prompt"))
        use.setStyleSheet(_PRIMARY_BTN)
        use.setMinimumHeight(38)
        use.setCursor(Qt.CursorShape.PointingHandCursor)
        use.clicked.connect(self._on_use)
        # View-only (a run is in flight): inspection stays open, picking does not.
        if getattr(self.parent(), "_view_only", False):
            use.setEnabled(False)
            use.setToolTip(tr("Available when detection finishes"))
        col.addWidget(use)

    def _on_loaded(self, pid: str, which: str, pixmap) -> None:
        if pid != self._preset["id"]:
            return
        if which in ("before", "before_preview"):
            self.slider.set_before(pixmap)
        elif which in ("after", "after_preview"):
            self.slider.set_after(pixmap)
        self._lock_aspect_from(pixmap)

    def _on_failed(self, pid: str, which: str) -> None:
        if pid == self._preset["id"] and not self.slider.has_images():
            self.slider.set_placeholder_text(tr("No preview yet"))

    def _on_use(self) -> None:
        self.chosen = True
        self.accept()


class _RunDetailDialog(_DetailDialogBase):
    """One past run: its preview tile before/after, metadata chips, and the
    actions (Use this prompt / Restore to map / Export / star)."""

    def __init__(self, run: dict, library, parent=None):
        super().__init__(parent or library)
        self._run = run
        self._lib = library
        self.use_requested = False
        prompt = (run.get("prompt") or "").strip() or tr("Older detection")
        self._prompt = prompt
        self._build_shell(prompt, tr("Your detection"))

        if self._lib._plugin is None:
            hint = tr("Open the Library from the Automatic page to use this.")
            for btn in (self.restore_btn, self.export_btn):
                btn.setEnabled(False)
                btn.setToolTip(hint)
        if not run.get("run_id"):
            # Legacy pseudo-run: no server row to star.
            self.star_btn.setEnabled(False)

        # Before/after of the run's preview tile (input vs mask overlay),
        # fetched with the account's auth headers via the image route. Own
        # loader (parented to self) so it dies with the popup.
        self._loader = TemplateDemoLoader(self)
        self._loader.loaded.connect(self._on_loaded)
        self._loader.failed.connect(self._on_failed)
        rid = run.get("preview_request_id") or ""
        if rid:
            for which in ("input", "preview"):
                self._loader.request(
                    rid, which,
                    self._lib._artifact_url(rid, which),
                    headers=self._lib._auth or None)
        else:
            self.slider.set_placeholder_text(tr("No preview"))

    def _build_info(self) -> None:
        run = self._run
        chips: list[tuple[str, str]] = []
        when = _relative_when(_iso_norm(
            run.get("started_at") or run.get("created_at")))
        if when:
            chips.append((tr("DATE"), when))
        chips.append((tr("OBJECTS"), str(run.get("objects") or 0)))
        chips.append((tr("CREDITS"), str(run.get("credits") or 0)))
        chips.append((tr("TILES"), str(run.get("tiles") or 0)))
        try:
            mupp = float(run.get("pixel_size_m") or 0)
            if mupp > 0:
                chips.append((tr("RESOLUTION"), "{:.2f} m/px".format(mupp)))
        except (TypeError, ValueError):
            pass
        if run.get("has_exemplars"):
            chips.append((tr("EXAMPLE"), tr("Used")))
        self._info_col.addWidget(self._build_prompt_block(self._prompt))
        self._info_col.addWidget(self._chips_grid(chips))

    def _build_footer(self, col: QVBoxLayout) -> None:
        # Secondary actions first (bring the run back), then the primary
        # "run it again" path pinned at the very bottom.
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(8)
        self.restore_btn = QPushButton(tr("Restore to map"))
        self.restore_btn.setStyleSheet(_ACTION_BTN)
        self.restore_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.restore_btn.setToolTip(tr(
            "Reopens this run's review at the same place. Free - no credits."))
        self.restore_btn.clicked.connect(
            lambda: self._lib._request_restore(self._run, self))
        actions.addWidget(self.restore_btn, 1)
        self.export_btn = QPushButton(tr("Export..."))
        self.export_btn.setStyleSheet(_ACTION_BTN)
        self.export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_btn.clicked.connect(
            lambda: self._lib._request_export(self._run, self))
        actions.addWidget(self.export_btn, 1)
        col.addLayout(actions)

        primary = QHBoxLayout()
        primary.setContentsMargins(0, 0, 0, 0)
        primary.setSpacing(8)
        self.star_btn = QToolButton(self)
        self.star_btn.setCheckable(True)
        self.star_btn.setFixedSize(38, 38)
        self.star_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.star_btn.setStyleSheet(_DETAIL_STAR_BTN)
        self.set_favorite(bool(self._run.get("is_favorite")))
        self.star_btn.clicked.connect(self._on_star_clicked)
        primary.addWidget(self.star_btn)
        self.use_btn = QPushButton(tr("Use this prompt"))
        self.use_btn.setStyleSheet(_PRIMARY_BTN)
        self.use_btn.setMinimumHeight(38)
        self.use_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.use_btn.setToolTip(tr(
            "Drop this object back into the prompt box for a new detection."))
        self.use_btn.clicked.connect(self._on_use)
        primary.addWidget(self.use_btn, 1)
        col.addLayout(primary)

        # View-only (a run is in flight): keep the run inspectable but disable
        # every action that would pick a prompt or start work.
        if getattr(self._lib, "_view_only", False):
            busy_tip = tr("Available when detection finishes")
            for btn in (self.use_btn, self.restore_btn, self.export_btn):
                btn.setEnabled(False)
                btn.setToolTip(busy_tip)

    def _on_use(self) -> None:
        self.use_requested = True
        self.accept()

    def _on_star_clicked(self, checked: bool) -> None:
        self.set_favorite(checked)  # glyph follows the optimistic flip at once
        self._lib._toggle_favorite(self._run, checked)

    def set_favorite(self, fav: bool) -> None:
        self.star_btn.blockSignals(True)
        self.star_btn.setChecked(fav)
        self.star_btn.setText("★" if fav else "☆")
        self.star_btn.setToolTip(
            tr("Remove from favorites") if fav else tr("Add to favorites"))
        self.star_btn.blockSignals(False)

    def set_busy(self, busy: bool) -> None:
        for btn in (self.restore_btn, self.export_btn):
            btn.setEnabled(not busy and self._lib._plugin is not None)
        if busy:
            self.restore_btn.setText(tr("Loading..."))
        else:
            self.restore_btn.setText(tr("Restore to map"))

    def _on_loaded(self, pid: str, which: str, pixmap) -> None:
        if pid != (self._run.get("preview_request_id") or ""):
            return
        if which == "input":
            self.slider.set_before(pixmap)
        elif which == "preview":
            self.slider.set_after(pixmap)
        self._lock_aspect_from(pixmap)

    def _on_failed(self, pid: str, which: str) -> None:
        if pid == (self._run.get("preview_request_id") or "") and not self.slider.has_images():
            self.slider.set_placeholder_text(tr("No preview"))


class _ExportRunDialog(QDialog):
    """Format + confidence + destination for the direct Export of a past run."""

    _FORMATS = (
        ("GeoPackage", "GPKG"),
        ("GeoJSON", "GeoJSON"),
        ("ESRI Shapefile", "ESRI Shapefile"),
        ("KML", "KML"),
    )

    def __init__(self, run: dict, default_confidence: float, parent=None):
        super().__init__(parent)
        self._run = run
        self.setWindowTitle(tr("Export..."))
        self.setMinimumWidth(420)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 12)
        lay.setSpacing(10)

        form_row = QHBoxLayout()
        fmt_lbl = QLabel(tr("Format:"))
        fmt_lbl.setStyleSheet("color: palette(text); background: transparent;")
        form_row.addWidget(fmt_lbl)
        self.format_combo = QComboBox()
        for label, _driver in self._FORMATS:
            self.format_combo.addItem(label)
        self.format_combo.setToolTip(tr(
            "GeoPackage keeps the embedded style; other formats are saved "
            "without a style."))
        self.format_combo.currentIndexChanged.connect(self._sync_extension)
        form_row.addWidget(self.format_combo, 1)
        lay.addLayout(form_row)

        conf_row = QHBoxLayout()
        conf_lbl = QLabel(tr("Confidence:"))
        conf_lbl.setStyleSheet("color: palette(text); background: transparent;")
        conf_row.addWidget(conf_lbl)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.blockSignals(True)
        self.conf_spin.setValue(default_confidence)
        self.conf_spin.blockSignals(False)
        conf_row.addWidget(self.conf_spin)
        conf_row.addStretch()
        lay.addLayout(conf_row)

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setStyleSheet(_SEARCH_QSS)
        path_row.addWidget(self.path_edit, 1)
        browse = QPushButton(tr("Browse..."))
        browse.setStyleSheet(_GHOST_BTN_QSS)
        browse.setCursor(Qt.CursorShape.PointingHandCursor)
        browse.clicked.connect(self._pick_path)
        path_row.addWidget(browse)
        lay.addLayout(path_row)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel = QPushButton(tr("Cancel"))
        cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel.clicked.connect(self.reject)
        btn_row.addWidget(cancel)
        self.ok_btn = QPushButton(tr("Export..."))
        self.ok_btn.setStyleSheet(_PRIMARY_BTN)
        self.ok_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ok_btn.setEnabled(False)
        self.ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.ok_btn)
        lay.addLayout(btn_row)

    def driver(self) -> str:
        return self._FORMATS[self.format_combo.currentIndex()][1]

    def confidence(self) -> float:
        return float(self.conf_spin.value())

    def path(self) -> str:
        return self.path_edit.text().strip()

    def _default_name(self) -> str:
        token = (self._run.get("prompt") or "detections").strip()
        safe = "".join(c if c.isalnum() else "_" for c in token) or "detections"
        return safe

    def _pick_path(self) -> None:
        from ....core.polygon_exporter import driver_extension
        ext = driver_extension(self.driver())
        label = self._FORMATS[self.format_combo.currentIndex()][0]
        suggested = self._default_name() + ext
        path, _filter = QFileDialog.getSaveFileName(
            self, tr("Export..."), suggested,
            "{} (*{})".format(label, ext))
        if not path:
            return
        if not path.lower().endswith(ext):
            path += ext
        self.path_edit.setText(path)
        self.ok_btn.setEnabled(True)

    def _sync_extension(self, _idx: int) -> None:
        import os

        from ....core.polygon_exporter import driver_extension
        current = self.path_edit.text().strip()
        if not current:
            return
        stem, _old = os.path.splitext(current)
        self.path_edit.setText(stem + driver_extension(self.driver()))
