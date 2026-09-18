







from __future__ import annotations

from qgis.PyQt.QtCore import Qt

from ...core.i18n import tr
from .guidance import HINT_PREVIEW_ZOOM, is_hint_dismissed
from .styles import (
    _BTN_EXPORT_READY,
    _BTN_GHOST,
    _INSTRUCTIONS_HINT_QSS,
    HUE_HOWTO,
    _msg_label_qss,
    category_label_qss,
    msg_rich,
)











_SIGN_ADD = "\U0001F7E2"
_SIGN_TRIM = "\u274C"
_SIGN_DOT_PX = 10
_SIGN_URLS: dict[str, str] = {}


def _click_sign_html(kind: str) -> str:

    import html
    import os
    import tempfile

    if kind not in _SIGN_URLS:
        from ..canvas_palette import MARKER_NEGATIVE, MARKER_POSITIVE
        colour = (MARKER_POSITIVE if kind == "add" else MARKER_NEGATIVE).name()
        body = (f'<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"'
                f' viewBox="0 0 20 20"><circle cx="10" cy="10" r="8" fill="{colour}"'
                ' stroke="#ffffff" stroke-opacity="0.85" stroke-width="1.5"/></svg>')
        try:
            folder = tempfile.mkdtemp(prefix="qgis_ai_seg_sign_")
            path = os.path.join(folder, f"sign_{kind}.svg")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(body)
            _SIGN_URLS[kind] = path.replace("\\", "/")
        except OSError:
            _SIGN_URLS[kind] = ""
    url = _SIGN_URLS[kind]
    if not url:
        return ""
    return (f'<img src="{html.escape(url, quote=True)}" width="{_SIGN_DOT_PX}"'
            f' height="{_SIGN_DOT_PX}">&nbsp;&nbsp;')


def _legend_line(kind: str, text: str) -> str:
    import html

    return _click_sign_html(kind) + html.escape(text)


class DockInstructionsMixin:





    def _refine_preview_live(self) -> bool:









        try:
            from ...core.hover_preview_client import hover_preview_offered

            return bool(self._manual_cloud_route_picked()
                        and hover_preview_offered()
                        and not self._manual_credits_exhausted())
        except Exception:  # noqa: BLE001
            return False

    def _preview_zoom_tip_wanted(self) -> bool:




        if not self._segmentation_active or self._refine_handoff:
            return False
        if is_hint_dismissed(HINT_PREVIEW_ZOOM):
            return False
        return self._refine_preview_live()

    def _update_refine_panel_visibility(self):



        try:
            self._sync_refine_shape_toggles()
        except (RuntimeError, AttributeError):
            pass  # nosec B110
        if not self._segmentation_active:
            self.refine_group.setVisible(False)
            return

        if self._refine_handoff:






            self.refine_group.setVisible(
                self._has_mask
                or bool(getattr(self, "_auto_correct_session_active", False)))
            return







        self.refine_group.setVisible(True)

    def _update_export_button_style(self):
        count = self._saved_polygon_count
        if count > 1:
            self.export_button.setText(
                tr("Export {count} polygons to a layer").format(count=count)
            )
        else:
            self.export_button.setText(tr("Export polygon to a layer"))






        show = bool(count > 0 and self._segmentation_active
                    and not self._refine_handoff)
        self.export_button.setVisible(show)
        if not show:
            return
        self.export_button.setEnabled(True)




        look = "ghost" if self._manual_save_live() else "primary"
        if getattr(self, "_export_button_look", "primary") != look:
            self._export_button_look = look
            self.export_button.setStyleSheet(
                _BTN_GHOST if look == "ghost" else _BTN_EXPORT_READY)
        self.export_button.setToolTip(
            tr("Writes a GeoPackage layer with your {n} kept polygons.").format(
                n=count))

    def _set_instructions_style(self, style: str) -> None:









        if getattr(self, "_instructions_style", "card") == style:
            return
        self._instructions_style = style
        if style == "waiting":
            self.instructions_label.setStyleSheet(_msg_label_qss("info"))
        else:
            self.instructions_label.setStyleSheet(
                _INSTRUCTIONS_HINT_QSS if style == "compact"
                else category_label_qss(HUE_HOWTO))

        self.instructions_label.setMinimumHeight(0)

    def set_manual_encoding(self, reading: bool, phase: str = "imagery") -> None:







        want = bool(reading)
        phase = phase if want else "imagery"
        if (getattr(self, "_manual_encoding", False) == want
                and getattr(self, "_manual_encoding_phase", "imagery") == phase):
            return
        self._manual_encoding = want
        self._manual_encoding_phase = phase
        if reading:



            self.clear_manual_notice()
        self._update_instructions()

    def _update_instructions(self):

        total = self._positive_count + self._negative_count

        if self.manual_notice_is_live():


            return

        if self._refine_handoff:



            self.instructions_label.setVisible(False)
            return

        if getattr(self, "_manual_encoding", False):






            self._set_instructions_style("waiting")
            phase = getattr(self, "_manual_encoding_phase", "")
            if phase == "remote":
                line = tr("Sending to the AI...")
            elif phase == "encode":
                line = tr("Preparing the imagery for the AI...")
            else:
                line = tr("Reading the imagery around your click...")
            self.instructions_label.setTextFormat(Qt.TextFormat.RichText)
            self.instructions_label.setText(msg_rich("info", line))
            return
        self._set_instructions_style("card")
        if total == 0 and self._saved_polygon_count > 0:

            import html
            text = (
                html.escape(tr(
                    "Polygon saved ({n} total). Click another element, or "
                    "export when done."
                ).format(n=self._saved_polygon_count))
                + "<br>" + _legend_line("add", tr("Left-click to select"))
            )
        elif total == 0:
            import html
            text = (
                html.escape(tr("Click the object you want to segment:"))
                + "<br>" + _legend_line("add", tr("Left-click to select"))
            )
        else:
            text = (
                _legend_line("add", tr("Left-click to add more")) + "<br>"
                + _legend_line("trim", tr("Right-click to exclude from selection"))
            )

        self.instructions_label.setTextFormat(Qt.TextFormat.RichText)
        self.instructions_label.setText(text)
