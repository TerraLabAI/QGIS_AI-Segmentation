









from __future__ import annotations

from qgis.PyQt.QtCore import QSettings, Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QGuiApplication
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
    QProgressBar,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ....core import qt_compat as QtC
from ....core.i18n import tr
from ....core.presets.segmentation_presets import pick_label
from ....core.server_dials import dial_in_range
from ...before_after_slider import BeforeAfterSlider
from ...dock.font_scale import scale_px_length, scale_qss_font_px
from ...dock.styles import (
    _PROGRESS_THIN_QSS,
    BTN_PRIMARY_WIDE_PX,
    FONT_BODY,
    FONT_HINT,
    INK,
    INK_2,
    apply_input_theme_to_tree,
    apply_quiet_scrollbar,
)
from ...template_demo_loader import TemplateDemoLoader
from .common import (
    _ACTION_BTN,
    _BADGE_STYLE,
    _CHIP_CAPTION,
    _CHIP_STYLE,
    _CHIP_VALUE,
    _COPY_BTN,
    _DETAIL_STAR_BTN,
    _FS_BTN,
    _GHOST_BTN_QSS,
    _PRIMARY_BTN,
    _PRIMARY_WIDE_BTN,
    _PROMPT_STYLE,
    _SEARCH_QSS,
    _SECTION_STYLE,
    _SEPARATOR,
    _TITLE_STYLE,
    _apply_library_ground,
    _AspectBox,
    _demo_url,
    _iso_norm,
    _relative_when,
    _set_star_glyph,
    _set_tool_glyph,
)
from .run_summary import run_status_text, run_time_text, run_zone_area_text


_PLAIN_TEXT_QSS = scale_qss_font_px(
    f"color: {INK}; font-size: {FONT_BODY}px; background: transparent;")
_SECTION_NOTE_QSS = scale_qss_font_px(
    f"font-size: {FONT_HINT}px; color: {INK_2}; background: transparent;")



_DETAIL_IMAGE_WIDTH = 768




_EXPORT_DIR_KEY = "AISegmentation/library_export_dir"




_WINDOWS_DEVICE_NAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL"] + [f"COM{i}" for i in range(1, 10)] + [f"LPT{i}" for i in range(1, 10)]
)


class _DetailDialogBase(QDialog):







    def __init__(self, parent=None):
        super().__init__(parent)
        _apply_library_ground(self)

    def _build_shell(self, title: str, badge_text: str) -> None:
        self.setWindowTitle(title or tr("Details"))
        self._apply_screen_floor(560, 420)
        self.setSizeGripEnabled(True)
        self._fullscreen = False
        self._aspect_locked = False

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)





        self.slider = BeforeAfterSlider(None, auto_loop=False, show_badges=True)
        self.slider.set_placeholder_text(tr("Loading..."))
        self._aspect_box = _AspectBox(self.slider, 1.0, self)
        self._aspect_box.setMinimumSize(scale_px_length(260), scale_px_length(240))
        self._aspect_box.setSizePolicy(
            QtC.SizePolicyExpanding, QtC.SizePolicyExpanding)


        self._fs_btn = QToolButton(self._aspect_box)
        _set_tool_glyph(self._fs_btn, "expand", 16, QColor(Qt.GlobalColor.white))
        self._fs_btn.setToolTip(tr("Fullscreen"))
        self._fs_btn.setAccessibleName(tr("Fullscreen"))
        self._fs_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._fs_btn.setStyleSheet(_FS_BTN)
        self._fs_btn.setFixedSize(scale_px_length(30), scale_px_length(30))
        self._fs_btn.clicked.connect(self._toggle_fullscreen)
        self._aspect_box.set_overlay(self._fs_btn)
        root.addWidget(self._aspect_box, 1)


        right = QWidget(self)


        right.setMinimumWidth(scale_px_length(330))
        right.setMaximumWidth(scale_px_length(560))
        right_col = QVBoxLayout(right)
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(10)
        self._info_panel = right

        info_scroll = QScrollArea(right)
        info_scroll.setWidgetResizable(True)
        info_scroll.setFrameShape(QtC.FrameNoFrame)
        apply_quiet_scrollbar(info_scroll)
        info_scroll.setHorizontalScrollBarPolicy(QtC.ScrollBarAlwaysOff)

        info = QWidget()
        info.setMinimumWidth(scale_px_length(300))
        self._info_col = QVBoxLayout(info)
        self._info_col.setContentsMargins(4, 2, 10, 2)
        self._info_col.setSpacing(12)

        badge_row = QHBoxLayout()
        badge_row.setContentsMargins(0, 0, 0, 0)


        badge = QLabel(badge_text.replace("&", "&&"))
        badge.setStyleSheet(_BADGE_STYLE)
        badge_row.addWidget(badge)
        badge_row.addStretch(1)
        self._info_col.addLayout(badge_row)

        title_lbl = QLabel(title)
        title_lbl.setWordWrap(True)
        title_lbl.setTextFormat(QtC.PlainText)
        title_lbl.setStyleSheet(_TITLE_STYLE)
        self._info_col.addWidget(title_lbl)

        self._build_info()
        self._info_col.addStretch(1)
        info_scroll.setWidget(info)
        right_col.addWidget(info_scroll, 1)



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
        from ...dock.font_scale import apply_font_scale_to_tree

        apply_font_scale_to_tree(self)
        self._apply_image_size(1.0)



    def _build_info(self) -> None:
        raise NotImplementedError

    def _build_footer(self, col: QVBoxLayout) -> None:
        raise NotImplementedError



    def _lock_aspect_from(self, pixmap) -> None:



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
        chrome_w = info_w + 12 + 24
        chrome_h = 24





        try:
            avail = self.screen().availableGeometry()
        except (AttributeError, RuntimeError):
            avail = None
        if avail is not None:
            budget_w = avail.width() * 0.96 - chrome_w
            budget_h = avail.height() * 0.92 - chrome_h
            if budget_w > 0 and budget_h > 0:
                shrink = min(budget_w / disp_w, budget_h / disp_h, 1.0)
                disp_w *= shrink
                disp_h *= shrink
        self.resize(int(disp_w) + chrome_w, int(disp_h) + chrome_h)



    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        self._info_panel.setVisible(not self._fullscreen)
        _set_tool_glyph(self._fs_btn, "close" if self._fullscreen else "expand",
                        16, QColor(Qt.GlobalColor.white))
        self._fs_btn.setToolTip(
            tr("Exit fullscreen") if self._fullscreen else tr("Fullscreen"))
        self._fs_btn.setAccessibleName(self._fs_btn.toolTip())



        if self._fullscreen:
            self.showMaximized()
        else:
            self.showNormal()
        self.raise_()
        self.activateWindow()

    def _apply_screen_floor(self, floor_w: int, floor_h: int) -> None:






        try:
            screen = self.screen() or QGuiApplication.primaryScreen()
        except (AttributeError, RuntimeError):
            screen = QGuiApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            floor_w = min(floor_w, int(avail.width() * 0.96))
            floor_h = min(floor_h, int(avail.height() * 0.92))
        self.setMinimumSize(floor_w, floor_h)

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() == Qt.Key.Key_Escape and self._fullscreen:
            self._toggle_fullscreen()
            return
        super().keyPressEvent(event)



    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(_SECTION_STYLE)
        return lbl

    def _build_prompt_block(self, prompt: str) -> QWidget:


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
        self._copy_btn.setAutoDefault(False)
        self._copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_btn.setFlat(True)
        self._copy_btn.setToolTip(tr("Copy prompt"))
        self._copy_btn.setStyleSheet(_COPY_BTN)
        from qgis.PyQt.QtCore import QSize

        from ...icons import icon_for
        self._copy_btn.setIcon(icon_for(self._copy_btn, "copy", 14))
        self._copy_btn.setIconSize(QSize(14, 14))
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
        reset_ms = dial_in_range("tuning.library.copy_feedback_ms", 1400, 500, 6000)

        QtC.safe_single_shot(reset_ms, self, self._reset_copy_btn)

    def _reset_copy_btn(self) -> None:
        try:
            self._copy_btn.setText(tr("Copy"))
        except (RuntimeError, AttributeError):
            pass

    def _wide_primary(self, text: str) -> QPushButton:

        btn = QPushButton(text)
        btn.setStyleSheet(_PRIMARY_WIDE_BTN)
        btn.setFixedHeight(scale_px_length(BTN_PRIMARY_WIDE_PX))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        return btn

    def _unavailable_note(self, col: QVBoxLayout) -> QLabel:



        note = QLabel("")
        note.setWordWrap(True)
        note.setStyleSheet(_SECTION_NOTE_QSS)
        note.setVisible(False)
        col.addWidget(note)
        return note

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


    def __init__(self, preset: dict, base: str, parent=None,
                 category_label: str = ""):
        super().__init__(parent)
        self._preset = preset
        self._base = base
        self.chosen = False
        title = pick_label(preset.get("label"), preset.get("prompt", ""))
        self._build_shell(title, category_label or tr("Template"))




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
            note.setStyleSheet(_SECTION_NOTE_QSS)
            self._info_col.addWidget(note)

    def _build_footer(self, col: QVBoxLayout) -> None:
        use = self._wide_primary(tr("Use this prompt"))
        use.setDefault(True)
        use.clicked.connect(self._on_use)
        col.addWidget(use)
        note = self._unavailable_note(col)


        if getattr(self.parent(), "_view_only", False):
            use.setVisible(False)
            note.setText(tr("Available when detection finishes"))
            note.setVisible(True)

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



    def __init__(self, run: dict, library, parent=None):
        super().__init__(parent or library)
        self._run = run
        self._lib = library
        prompt = (run.get("prompt") or "").strip() or tr("Older detection")
        self._prompt = prompt
        self._build_shell(prompt, tr("Your detection"))




        hint = ""
        if getattr(self._lib, "_view_only", False):
            hint = tr("Available when detection finishes")
        elif self._lib._plugin is None:
            hint = tr("Open the Library from the Automatic page to use this.")
        elif not self._lib._auth:
            hint = tr("Sign in to reopen, export or run this zone again.")
        if hint:
            for btn in (self.restore_btn, self.export_btn, self.rerun_btn):
                btn.setEnabled(False)
                btn.setVisible(False)
            self._actions_note.setText(hint)
            self._actions_note.setVisible(True)
        if not run.get("run_id") or not self._lib._auth:



            self.star_btn.setVisible(False)




        self._loader = TemplateDemoLoader(self)
        self._loader.loaded.connect(self._on_loaded)
        self._loader.failed.connect(self._on_failed)
        rid = run.get("preview_request_id") or ""
        if rid:





            width = dial_in_range(
                "tuning.library.detail_image_width_px", _DETAIL_IMAGE_WIDTH, 256, 2048)
            for which in ("input", "preview"):
                self._loader.request(
                    rid, which,
                    self._lib._artifact_url(rid, which, width),
                    headers=self._lib._auth or None,
                    variant=str(width), immutable=True)
        else:
            self.slider.set_placeholder_text(tr("No preview"))

    def _build_info(self) -> None:




        run = self._run
        chips: list[tuple[str, str]] = []
        when = _relative_when(_iso_norm(
            run.get("started_at") or run.get("created_at")))
        clock = run_time_text(run)
        if when:
            chips.append((tr("Date"), f"{when}, {clock}" if clock else when))
        zone = run_zone_area_text(run)
        if zone:
            chips.append((tr("Zone"), zone))
        chips.append((tr("Objects"), str(run.get("objects") or 0)))
        chips.append((tr("Cloud detections"), str(run.get("tiles") or 0)))
        chips.append((tr("Charged"), str(run.get("credits") or 0)))
        try:
            mupp = float(run.get("pixel_size_m") or 0)
            if mupp > 0:
                chips.append((tr("Resolution"), f"{mupp:.2f} m/px"))
        except (TypeError, ValueError):
            pass
        status = run_status_text(run)
        if status:
            chips.append((tr("Status"), status))
        if run.get("has_exemplars"):
            chips.append((tr("Example"), tr("Used")))
        self._info_col.addWidget(self._build_prompt_block(self._prompt))
        self._info_col.addWidget(self._chips_grid(chips))

    def _build_footer(self, col: QVBoxLayout) -> None:


        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(8)
        self.restore_btn = QPushButton(tr("Restore to map"))
        self.restore_btn.setStyleSheet(_ACTION_BTN)



        self.restore_btn.setAutoDefault(False)
        self.restore_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.restore_btn.setToolTip(tr(
            "Reopens this run's review at the same place, with its imagery. "
            "Free, and it costs no cloud detections."))
        self.restore_btn.clicked.connect(
            lambda: self._lib._request_restore(self._run, self))
        actions.addWidget(self.restore_btn, 1)
        self.export_btn = QPushButton(tr("Export..."))
        self.export_btn.setStyleSheet(_ACTION_BTN)
        self.export_btn.setAutoDefault(False)
        self.export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_btn.clicked.connect(
            lambda: self._lib._request_export(self._run, self))
        actions.addWidget(self.export_btn, 1)



        self.delete_btn = QPushButton(tr("Delete run"))
        self.delete_btn.setStyleSheet(_COPY_BTN)
        self.delete_btn.setAutoDefault(False)
        self.delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_btn.setToolTip(tr(
            "Takes this run out of your history. Its detections stay stored."))
        self.delete_btn.clicked.connect(
            lambda: self._lib._request_delete(self._run, self))
        self.delete_btn.setVisible(
            bool(self._run.get("run_id")) and bool(self._lib._auth))
        actions.addWidget(self.delete_btn)
        col.addLayout(actions)

        primary = QHBoxLayout()
        primary.setContentsMargins(0, 0, 0, 0)
        primary.setSpacing(8)
        self.star_btn = QToolButton(self)
        self.star_btn.setCheckable(True)

        side = scale_px_length(BTN_PRIMARY_WIDE_PX)
        self.star_btn.setFixedSize(side, side)
        self.star_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.star_btn.setStyleSheet(_DETAIL_STAR_BTN)
        self.set_favorite(bool(self._run.get("is_favorite")))
        self.star_btn.clicked.connect(self._on_star_clicked)
        primary.addWidget(self.star_btn)




        self.rerun_btn = self._wide_primary(tr("Run this zone again"))
        self.rerun_btn.setAutoDefault(False)
        self.rerun_btn.setToolTip(tr(
            "Points the map back at this run, ready to detect the same object "
            "again. Nothing is spent until you do."))
        self.rerun_btn.clicked.connect(
            lambda: self._lib._request_rerun(self._run, self))
        primary.addWidget(self.rerun_btn, 1)
        col.addLayout(primary)

        self._actions_note = self._unavailable_note(col)

    def _on_star_clicked(self, checked: bool) -> None:
        self.set_favorite(checked)
        self._lib._toggle_favorite(self._run, checked)

    def set_favorite(self, fav: bool) -> None:
        self.star_btn.blockSignals(True)
        self.star_btn.setChecked(fav)
        _set_star_glyph(self.star_btn, fav, 18)
        tip = tr("Remove from favorites") if fav else tr("Add to favorites")
        self.star_btn.setToolTip(tip)
        self.star_btn.setAccessibleName(tip)
        self.star_btn.blockSignals(False)

    def set_busy(self, busy: bool, actor: str = "restore") -> None:



        usable = (not busy and self._lib._plugin is not None
                  and bool(self._lib._auth)
                  and not getattr(self._lib, "_view_only", False))
        buttons = {
            "restore": (self.restore_btn, tr("Restore to map")),
            "export": (self.export_btn, tr("Export...")),
            "rerun": (self.rerun_btn, tr("Run this zone again")),
        }
        for btn, _label in buttons.values():
            btn.setEnabled(usable)


        self.delete_btn.setEnabled(not busy and bool(self._lib._auth))


        for key, (btn, label) in buttons.items():
            btn.setText(tr("Loading...") if busy and key == actor else label)

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


class _RunProgressDialog(QDialog):








    cancelled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        _apply_library_ground(self)
        self.setWindowTitle(tr("Segment library"))
        self.setMinimumWidth(scale_px_length(360))


        self._done = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 12)
        lay.setSpacing(10)

        self._label = QLabel(tr("Reading this run..."))
        self._label.setStyleSheet(_PLAIN_TEXT_QSS)
        lay.addWidget(self._label)

        self._bar = QProgressBar()
        self._bar.setTextVisible(False)
        self._bar.setStyleSheet(_PROGRESS_THIN_QSS)
        self._bar.setRange(0, 0)
        lay.addWidget(self._bar)

        row = QHBoxLayout()
        row.addStretch()
        cancel = QPushButton(tr("Cancel"))
        cancel.setStyleSheet(_GHOST_BTN_QSS)
        cancel.setAutoDefault(False)
        cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel.clicked.connect(self.reject)
        row.addWidget(cancel)
        lay.addLayout(row)

    def keyPressEvent(self, event):  # noqa: N802



        if event.key() == Qt.Key.Key_Escape and not self._done:
            event.ignore()
            return
        super().keyPressEvent(event)

    def set_step(self, text: str, done: int = 0, total: int = 0) -> None:


        self._label.setText(text)
        if total > 0:
            self._bar.setRange(0, total)
            self._bar.setValue(max(0, min(done, total)))
        else:
            self._bar.setRange(0, 0)

    def finish(self) -> None:

        self._done = True
        self.close()

    def reject(self):  # noqa: N802
        if not self._done:
            self._done = True
            self.cancelled.emit()
        super().reject()


class _ExportRunDialog(QDialog):





    _DRIVER_LABELS = {"GPKG": "GeoPackage"}

    def __init__(self, run: dict, default_confidence: float, parent=None):
        super().__init__(parent)
        _apply_library_ground(self)
        from ....core.polygon_exporter import EXPORT_DRIVERS

        self._run = run


        self._formats = tuple(
            (self._DRIVER_LABELS.get(driver, driver), driver)
            for driver in EXPORT_DRIVERS)
        self.setWindowTitle(tr("Export"))
        self.setMinimumWidth(scale_px_length(420))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 12)
        lay.setSpacing(10)

        form_row = QHBoxLayout()
        fmt_lbl = QLabel(tr("Format:"))
        fmt_lbl.setStyleSheet(_PLAIN_TEXT_QSS)
        form_row.addWidget(fmt_lbl)
        self.format_combo = QComboBox()
        for label, _driver in self._formats:
            self.format_combo.addItem(label)
        self.format_combo.setToolTip(
            tr("GeoPackage keeps the embedded style; other formats are saved "
               "without a style.")
            + "\n"
            + tr("GeoJSON and KML are written in EPSG:4326. Shapefile "
                 "shortens field names to 10 characters."))
        self.format_combo.currentIndexChanged.connect(self._sync_extension)
        form_row.addWidget(self.format_combo, 1)
        lay.addLayout(form_row)

        conf_row = QHBoxLayout()
        conf_lbl = QLabel(tr("Confidence:"))
        conf_lbl.setStyleSheet(_PLAIN_TEXT_QSS)
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



        browse.setAutoDefault(False)
        browse.setCursor(Qt.CursorShape.PointingHandCursor)
        browse.clicked.connect(self._pick_path)
        path_row.addWidget(browse)
        lay.addLayout(path_row)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel = QPushButton(tr("Cancel"))


        cancel.setStyleSheet(_GHOST_BTN_QSS)
        cancel.setAutoDefault(False)
        cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel.clicked.connect(self.reject)
        btn_row.addWidget(cancel)
        self.ok_btn = QPushButton(tr("Export"))
        self.ok_btn.setStyleSheet(_PRIMARY_BTN)
        self.ok_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ok_btn.setDefault(True)
        self.ok_btn.setEnabled(False)
        self.ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.ok_btn)
        lay.addLayout(btn_row)


        apply_input_theme_to_tree(self)
        from ...dock.font_scale import apply_font_scale_to_tree

        apply_font_scale_to_tree(self)

    def driver(self) -> str:
        return self._formats[self.format_combo.currentIndex()][1]

    def confidence(self) -> float:
        return float(self.conf_spin.value())

    def path(self) -> str:
        return self.path_edit.text().strip()

    def _default_name(self) -> str:
        token = (self._run.get("prompt") or "detections").strip()
        stem = "".join(c if c.isalnum() else "_" for c in token) or "detections"
        if stem.upper() in _WINDOWS_DEVICE_NAMES:


            stem += "_export"
        return stem

    def _start_dir(self) -> str:







        import os

        remembered = str(QSettings().value(_EXPORT_DIR_KEY, "", type=str) or "")
        candidates = [remembered]
        try:
            from qgis.core import QgsProject
            candidates.append(QgsProject.instance().homePath() or "")
        except (RuntimeError, ImportError):

            pass
        candidates.append(os.path.expanduser("~"))
        for candidate in candidates:
            if candidate and os.path.isdir(candidate):
                return candidate
        return os.path.expanduser("~")

    def _pick_path(self) -> None:
        import os

        from ....core.polygon_exporter import driver_extension
        ext = driver_extension(self.driver())
        label = self._formats[self.format_combo.currentIndex()][0]
        suggested = os.path.normpath(
            os.path.join(self._start_dir(), self._default_name() + ext))
        path, _filter = QFileDialog.getSaveFileName(
            self, tr("Export"), suggested,
            f"{label} (*{ext})")
        if not path:
            return
        if not path.lower().endswith(ext):
            path += ext


        path = os.path.normpath(path)
        QSettings().setValue(_EXPORT_DIR_KEY, os.path.dirname(path))
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
