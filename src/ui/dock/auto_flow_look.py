










from __future__ import annotations

from qgis.PyQt.QtCore import QEvent, QPointF, QRectF, QSize, Qt
from qgis.PyQt.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from qgis.PyQt.QtWidgets import QFrame, QPushButton, QSizePolicy, QWidget

from .font_scale import scale_point_size, scale_px_length, scale_qss_font_px
from .styles import (
    _BTN_GHOST,
    ACCENT_BORDER,
    BRAND_BLUE,
    BTN_PRIMARY_WIDE_PX,
    FONT_BASE,
    FONT_BODY,
    FONT_HINT,
    FONT_MICRO,
    HOVER,
    HOVER_ON,
    INK,
    INK_2,
    INK_3,
    LINE,
    LINE_SOFT,
    LINE_STRONG,
    ON_ACCENT,
    RADIUS_CARD,
    RADIUS_CHIP,
    RADIUS_CONTROL,
    RADIUS_PANEL,
    SURFACE,
    category_ink,
)


AUTO_COMPOSER_RADIUS = RADIUS_PANEL

TASK_DISC_PX = 22
TASK_ROW_PX = 40


def auto_flow_sheet() -> str:


    return scale_qss_font_px(

        f"QLabel#autoTitle {{ font-size: {FONT_BASE}px; font-weight: 600;"
        f" color: {INK}; background: transparent; border: none; }}"
        f"QLabel#autoText {{ font-size: {FONT_BODY}px; color: {INK};"
        " background: transparent; border: none; }"
        f"QLabel#autoHint {{ font-size: {FONT_HINT}px; color: {INK_2};"
        " background: transparent; border: none; }"
        f"QLabel#autoMicro {{ font-size: {FONT_MICRO}px; color: {INK_3};"
        " background: transparent; border: none; }"
        f"QLabel#autoFieldLabel {{ font-size: {FONT_BODY}px; color: {INK};"
        " background: transparent; border: none; }"


        f"QFrame#autoComposer {{ background: {SURFACE}; border: 1px solid {LINE_STRONG};"
        f" border-radius: {AUTO_COMPOSER_RADIUS}px; }}"
        f'QFrame#autoComposer[focused="true"] {{ border: 1px solid {BRAND_BLUE}; }}'
        "QLineEdit#autoComposerInput { background: transparent; border: none;"
        f" padding: 2px 2px; font-size: {FONT_BASE}px; color: {INK};"
        f" selection-background-color: {BRAND_BLUE}; }}"


        "QWidget#autoDetailCard { background: transparent; border: none; }"

        f"QFrame#autoQuietCard {{ background: {SURFACE}; border: 1px solid {LINE};"
        f" border-radius: {RADIUS_CARD}px; }}"

        f"QFrame#autoTaskList {{ background: {SURFACE}; border: 1px solid {LINE};"
        f" border-radius: {RADIUS_PANEL}px; }}"
        "QWidget#autoTaskRow { background: transparent; border: none;"
        f" border-radius: {RADIUS_CONTROL}px; }}"
        f'QWidget#autoTaskRow[navigable="true"]:hover {{ background: {HOVER}; }}'
        f"QFrame#autoTaskRule {{ background: {LINE_SOFT}; border: none; }}"
        f"QLabel#autoTaskLabel {{ font-size: {FONT_BASE}px; font-weight: 500;"
        f" color: {INK}; background: transparent; border: none; }}"
        f'QLabel#autoTaskLabel[state="todo"] {{ color: {INK_2}; }}'
        f'QLabel#autoTaskLabel[state="active"] {{ font-weight: 600; }}'


        f"QLabel#autoRunVerb {{ font-size: {FONT_BODY}px; font-weight: 500;"
        f" color: {INK}; background: transparent; border: none; }}"
        f"QLabel#autoStatusTime, QLabel#autoStatusFigure {{"
        f" font-size: {FONT_HINT}px; color: {INK_2}; background: transparent; border: none; }}"

        f"QLabel#autoMonoChip {{ font-size: {FONT_HINT}px;"
        f" color: {INK_2}; background: {HOVER}; border: none;"
        f" border-radius: {RADIUS_CHIP}px; padding: 1px 6px; }}"
    )



COMPOSER_INPUT_QSS = scale_qss_font_px(
    "QLineEdit { background: transparent; border: none; padding: 2px 2px;"
    f" font-size: {FONT_BASE}px; color: {INK};"
    f" selection-background-color: {BRAND_BLUE}; }}"
)



_BTN_AUTO_CHIP = scale_qss_font_px(
    f"QPushButton {{ background: transparent; border: 1px solid {LINE};"
    f" border-radius: {RADIUS_CONTROL}px; padding: 3px 10px; min-height: 22px;"
    f" font-size: {FONT_BODY}px; font-weight: 500; color: {INK}; text-align: center; }}"
    f"QPushButton:hover {{ background: {HOVER}; border-color: {LINE_STRONG}; }}"
    f"QPushButton:pressed {{ background: {HOVER_ON}; border-color: {LINE_STRONG}; }}"
    f'QPushButton[active="true"] {{ background: {HOVER_ON}; border-color: {LINE_STRONG}; }}'
    f"QPushButton:disabled {{ color: {INK_3}; border-color: {LINE}; }}"


    f"QPushButton:focus {{ outline: none; border-color: {ACCENT_BORDER}; }}"
)



_BTN_AUTO_FOLD = scale_qss_font_px(
    "QPushButton { background: transparent; border: none; padding: 0;"
    f" border-radius: {RADIUS_CONTROL}px; text-align: left; }}"
    f"QPushButton:hover {{ background: {HOVER}; }}"
    f"QPushButton:pressed {{ background: {HOVER_ON}; }}"
    "QPushButton:disabled { background: transparent; }"
)


_BTN_AUTO_FOLD_TEXT = scale_qss_font_px(
    "QPushButton { background: transparent; border: none; padding: 6px 4px;"
    f" border-radius: {RADIUS_CONTROL}px; text-align: left; color: {INK};"
    f" font-size: {FONT_BODY}px; font-weight: 600; }}"
    f"QPushButton:hover {{ background: {HOVER}; }}"
    f"QPushButton:pressed {{ background: {HOVER_ON}; }}"
    f"QPushButton:disabled {{ color: {INK_3}; background: transparent; }}"
)





_BTN_GHOST_WIDE = _BTN_GHOST + (
    f"QPushButton {{ min-height: {BTN_PRIMARY_WIDE_PX - 2}px; }}")


WARNING_QCOLOR = QColor(245, 166, 35)


_BTN_AUTO_QUIET = scale_qss_font_px(


    "QPushButton { background: transparent; border: 1px solid transparent;"
    f" color: {INK_2}; font-size: {FONT_BODY}px; font-weight: 500; padding: 3px 7px;"
    f" border-radius: {RADIUS_CONTROL}px; }}"
    f"QPushButton:hover {{ color: {INK}; background: {HOVER}; }}"
    f"QPushButton:pressed {{ color: {INK}; background: {HOVER_ON}; }}"
    f"QPushButton:disabled {{ color: {INK_3}; }}"
    f"QPushButton:focus {{ outline: none; border-color: {ACCENT_BORDER}; }}"
)


def repolish_widget(widget) -> None:

    try:
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()
    except (RuntimeError, AttributeError):
        pass


def token_qcolor(token: str) -> QColor:

    text = str(token or "").strip()
    if text.startswith("rgb"):
        parts = [p.strip() for p in text[text.index("(") + 1:text.rindex(")")].split(",")]
        try:
            r, g, b = (int(float(p)) for p in parts[:3])
            alpha = float(parts[3]) if len(parts) > 3 else 1.0
        except (ValueError, IndexError):
            return QColor()
        colour = QColor(r, g, b)
        colour.setAlphaF(max(0.0, min(1.0, alpha)))
        return colour
    return QColor(text)


class ComposerFrame(QFrame):



    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setProperty("focused", False)
        self._watched = None

    def watch_focus(self, field) -> None:
        self._watched = field
        field.installEventFilter(self)

    def eventFilter(self, obj, event):  # noqa: N802
        try:
            if obj is self._watched and event.type() in (
                    QEvent.Type.FocusIn, QEvent.Type.FocusOut):
                focused = event.type() == QEvent.Type.FocusIn
                if bool(self.property("focused")) != focused:
                    self.setProperty("focused", focused)
                    repolish_widget(self)
        except RuntimeError:
            pass
        return False

    def mousePressEvent(self, event):  # noqa: N802

        field = self._watched
        if field is not None:
            try:
                field.setFocus(Qt.FocusReason.MouseFocusReason)
            except RuntimeError:
                pass
        super().mousePressEvent(event)


def name_unlabelled_controls(root: QWidget) -> None:









    import re

    from qgis.PyQt.QtWidgets import (
        QAbstractButton,
        QAbstractSpinBox,
        QComboBox,
        QLineEdit,
        QSlider,
        QToolButton,
    )

    from ...core.i18n import tr

    for widget in root.findChildren(QWidget):
        try:
            if widget.accessibleName().strip():
                continue
            if widget.objectName().startswith("qt_"):
                continue
            if isinstance(widget, QAbstractButton):
                if widget.text().strip():
                    continue
            elif not isinstance(widget, (QAbstractSpinBox, QComboBox, QLineEdit, QSlider)):
                continue
            tip = re.sub(r"<[^>]+>", "", widget.toolTip() or "").strip()
            if tip:
                name = re.split(r"(?<=[.!?])\s", tip, maxsplit=1)[0].rstrip(".")
            elif (isinstance(widget, QToolButton)
                  and isinstance(widget.parentWidget(), QLineEdit)):

                name = tr("Clear")
            else:
                continue
            widget.setAccessibleName(name)
        except RuntimeError:
            continue


def flow_choice_divider(text: str) -> QWidget:



    from qgis.PyQt.QtWidgets import QHBoxLayout, QLabel

    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 2, 0, 2)
    lay.setSpacing(8)
    for index in range(3):
        if index == 1:
            label = QLabel(text)
            label.setStyleSheet(scale_qss_font_px(
                f"font-size: {FONT_HINT}px; color: {INK_2};"
                " background: transparent; border: none;"))
            lay.addWidget(label, 0, Qt.AlignmentFlag.AlignVCenter)
            continue
        line = QFrame()
        line.setFrameShape(QFrame.Shape.NoFrame)
        line.setFixedHeight(1)
        line.setStyleSheet(f"background: {LINE_STRONG}; border: none;")
        lay.addWidget(line, 1, Qt.AlignmentFlag.AlignVCenter)
    return row


class NeverShownWidget(QWidget):



    def setVisible(self, visible: bool) -> None:  # noqa: N802
        super().setVisible(False)


class VisibilityTwin(QWidget):








    def __init__(self, source: QWidget, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setVisible(False)
        source.installEventFilter(self)

    def eventFilter(self, obj, event):  # noqa: N802
        try:
            kind = event.type()
            if kind == QEvent.Type.Show:
                self.setVisible(True)
            elif kind == QEvent.Type.Hide and not obj.isVisibleTo(obj.parentWidget()):
                self.setVisible(False)
        except (RuntimeError, AttributeError):
            pass
        return False


class FitTextButton(QPushButton):










    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._full_text = ""
        self._two_lines = False
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802
        self._full_text = text or ""
        self._apply_fit()

    def text(self) -> str:  # noqa: D401
        return self._full_text

    def _split_text(self) -> str:
        head, sep, tail = self._full_text.partition(" (")
        return f"{head}\n({tail}" if sep else self._full_text

    def _one_line_width(self) -> int:
        metrics = self.fontMetrics()
        return metrics.horizontalAdvance(self._full_text) + scale_px_length(40)

    def sizeHint(self) -> QSize:  # noqa: N802
        hint = super().sizeHint()
        lines = self._split_text().split("\n")
        metrics = self.fontMetrics()
        widest = max(metrics.horizontalAdvance(line) for line in lines)
        return QSize(widest + scale_px_length(40), hint.height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return self.sizeHint()

    def _apply_fit(self) -> None:
        two = self.width() < self._one_line_width() and " (" in self._full_text
        label = self._split_text() if two else self._full_text
        if QPushButton.text(self) != label:
            QPushButton.setText(self, label)
        if two != self._two_lines:
            self._two_lines = two

            self.setMinimumHeight(scale_px_length(46 if two else 36))

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        try:
            self._apply_fit()
        except RuntimeError:
            pass


class TaskDisc(QWidget):





    def __init__(self, number: int, state: str = "todo", parent=None):
        super().__init__(parent)
        self._number = int(number)
        self._state = state
        side = scale_px_length(TASK_DISC_PX)
        self.setFixedSize(side, side)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def disc_state(self) -> str:
        return self._state

    def set_disc_state(self, state: str) -> None:
        if state not in ("done", "active", "todo"):
            state = "todo"
        if state != self._state:
            self._state = state
            self.update()

    def sizeHint(self) -> QSize:  # noqa: N802
        side = scale_px_length(TASK_DISC_PX)
        return QSize(side, side)

    def paintEvent(self, event):  # noqa: N802
        painter = None
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            side = float(min(self.width(), self.height()))
            stroke = max(1.2, side / 16.0)
            inset = 1.0 + stroke / 2.0
            rect = QRectF(inset, inset, side - 2 * inset, side - 2 * inset)
            if self._state == "done":



                leaf = token_qcolor(category_ink("leaf"))
                painter.setPen(QPen(leaf, stroke))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(rect)
                pen = QPen(leaf, max(1.6, side / 12.0))
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                path = QPainterPath()
                path.moveTo(QPointF(side * 0.30, side * 0.52))
                path.lineTo(QPointF(side * 0.44, side * 0.65))
                path.lineTo(QPointF(side * 0.70, side * 0.37))
                painter.drawPath(path)
            else:
                active = self._state == "active"
                if active:
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(token_qcolor(BRAND_BLUE))
                    ink = token_qcolor(ON_ACCENT)
                else:


                    painter.setPen(QPen(token_qcolor(LINE_STRONG), stroke))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    ink = token_qcolor(INK_2)
                painter.drawEllipse(rect)
                font = QFont(self.font())
                font.setPixelSize(scale_point_size(FONT_HINT))
                font.setWeight(QFont.Weight.Bold if active else QFont.Weight.DemiBold)
                painter.setFont(font)
                painter.setPen(ink)
                painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), str(self._number))
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        finally:
            try:
                if painter is not None and painter.isActive():
                    painter.end()
            except Exception:  # noqa: BLE001  # nosec B110
                pass


def toggle_indicator_qss(dock) -> str:





    import os

    from .temp_icon_dirs import make_icon_dir

    try:
        icon_dir = getattr(dock, "_checkbox_icon_dir", None)
        if not icon_dir:
            icon_dir = make_icon_dir("qgis_ai_seg_")
            dock._checkbox_icon_dir = icon_dir
        off_path = os.path.join(icon_dir, "toggle_off.svg").replace("\\", "/")
        on_path = os.path.join(icon_dir, "toggle_on.svg").replace("\\", "/")
        dis_path = os.path.join(icon_dir, "toggle_dis_v2.svg").replace("\\", "/")
        head = ('<svg xmlns="http://www.w3.org/2000/svg" width="30" height="18"'
                ' viewBox="0 0 30 18">')
        files = {
            off_path: (head + '<rect x="0.5" y="0.5" width="29" height="17" rx="8.5"'
                       ' fill="#808080" fill-opacity="0.35"/>'
                       '<circle cx="9" cy="9" r="6.5" fill="#ffffff"/></svg>'),
            on_path: (head + f'<rect x="0.5" y="0.5" width="29" height="17" rx="8.5"'
                      f' fill="{BRAND_BLUE}"/>'
                      '<circle cx="21" cy="9" r="6.5" fill="#ffffff"/></svg>'),


            dis_path: (head + '<rect x="0.5" y="0.5" width="29" height="17" rx="8.5"'
                       ' fill="#808080" fill-opacity="0.12" stroke="#808080"'
                       ' stroke-opacity="0.45"/>'
                       '<circle cx="9" cy="9" r="5.5" fill="#808080"'
                       ' fill-opacity="0.55"/></svg>'),
        }
        for path, body in files.items():
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(body)
    except OSError:
        return ""
    return (
        "QCheckBox { background: transparent; spacing: 8px; }"
        "QCheckBox::indicator { width: 30px; height: 18px; }"
        f'QCheckBox::indicator:unchecked {{ image: url("{off_path}"); }}'
        f'QCheckBox::indicator:checked {{ image: url("{on_path}"); }}'
        f'QCheckBox::indicator:disabled {{ image: url("{dis_path}"); }}'
    )


__all__ = [
    "AUTO_COMPOSER_RADIUS",
    "COMPOSER_INPUT_QSS",
    "ComposerFrame",
    "FitTextButton",
    "TASK_DISC_PX",
    "TASK_ROW_PX",
    "NeverShownWidget",
    "TaskDisc",
    "VisibilityTwin",
    "WARNING_QCOLOR",
    "_BTN_AUTO_CHIP",
    "_BTN_AUTO_FOLD",
    "_BTN_AUTO_FOLD_TEXT",
    "_BTN_AUTO_QUIET",
    "_BTN_GHOST_WIDE",
    "auto_flow_sheet",
    "flow_choice_divider",
    "name_unlabelled_controls",
    "repolish_widget",
    "toggle_indicator_qss",
    "token_qcolor",
]
