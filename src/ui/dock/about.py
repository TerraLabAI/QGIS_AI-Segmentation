





from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
from ...core.server_dials import dial_copy
from .font_scale import scale_px_length
from .styles import (
    _BTN_GREEN_STEP,
    _HINT_LINE_QSS,
    _UPDATE_LATER_STYLE,
    BTN_PRIMARY_WIDE_PX,
    CATEGORY_TILE_GLYPH_PX,
    CATEGORY_TILE_PX,
    FONT_BASE,
    FONT_BODY,
    FONT_HINT,
    HUE_HOWTO,
    INK,
    INK_2,
    INK_3,
    LINE,
    RADIUS_CARD,
    RADIUS_CHIP,
    RADIUS_PANEL,
    SURFACE,
    _msg_card_qss,
    category_ink,
    category_line,
    category_tile,
    category_tint,
)


def _web_url_or(candidate: object, fallback: str) -> str:









    from ...core.server_dials import safe_web_url

    return safe_web_url(candidate, fallback)






_UPDATE_ICON_PX = 36



_UPDATE_POLICY_DEFAULT = "recommend"
_UPDATE_POLICIES = ("require", "recommend")

_DISMISSED_UPDATE_VERSIONS: set[str] = set()




def _update_card_qss() -> str:
    return (


        f"QWidget#updateCard {{ background-color: {SURFACE};"
        f" border: 1px solid {LINE}; border-radius: {RADIUS_CARD}px; }}"
        "QWidget#updateCard QLabel { background: transparent; border: none; }"
        f"QLabel#updateTitle {{ font-size: {FONT_BASE}px; font-weight: 600; color: {INK}; }}"
        f"QLabel#updateNote {{ font-size: {FONT_HINT}px; color: {INK_2}; }}"
        f"QLabel#updateBody {{ font-size: {FONT_BODY}px; color: {INK_2}; }}"
        f"QLabel#updateHint {{ font-size: {FONT_HINT}px; color: {INK_3}; }}"
        f"QWidget#updateBadge {{ background: {category_tint('amber')};"
        f" border: 1px solid {category_line('amber')}; border-radius: {RADIUS_CHIP}px; }}"


        f"QLabel#updateBadgeText {{ color: {category_ink('amber')}; font-size: {FONT_HINT}px;"
        " font-weight: 600; }"
    )




_GATE_CARD_MAX_W = 380
_GATE_ICON_PX = 44


_GATE_BADGE_PX = 22


_GATE_TITLE_PX = FONT_BASE + 3


def _update_gate_qss() -> str:
    return (
        f"QFrame#updateGateCard {{ background-color: {SURFACE};"
        f" border: 1px solid {LINE}; border-radius: {RADIUS_PANEL}px; }}"
        "QFrame#updateGateCard QLabel { background: transparent; border: none; }"
        f"QFrame#updateGateBadge {{ background: {category_tint('amber')};"
        f" border: 1px solid {category_line('amber')}; border-radius: {RADIUS_CHIP}px; }}"
        f"QLabel#updateGateBadgeText {{ color: {category_ink('amber')}; font-size: {FONT_HINT}px;"
        " font-weight: 600; }"
        f"QLabel#updateGateTitle {{ font-size: {_GATE_TITLE_PX}px; font-weight: 600;"
        f" color: {INK}; }}"
        f"QLabel#updateGateNote {{ font-size: {FONT_BODY}px; color: {INK}; }}"
        f"QLabel#updateGateBody {{ font-size: {FONT_BODY}px; color: {INK_2}; }}"
        f"QLabel#updateGateHint {{ font-size: {FONT_HINT}px; color: {INK_3}; }}"
    )


class _UpdateGatePage(QWidget):






    def __init__(self, parent=None):
        super().__init__(parent)
        self.card = QFrame(self)
        self.card.setObjectName("updateGateCard")
        self.card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.card.setFixedWidth(_GATE_CARD_MAX_W)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        layout = self.layout()
        if layout is None:
            return
        margins = layout.contentsMargins()
        room = max(200, event.size().width() - margins.left() - margins.right())
        self.card.setFixedWidth(min(_GATE_CARD_MAX_W, room))


def _set_plugin_mark(label: QLabel, side: int) -> None:

    import os

    from qgis.PyQt.QtGui import QPixmap

    from .font_scale import widget_pixel_ratio


    plugin_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    pixmap = QPixmap(os.path.join(plugin_root, "resources", "icons", "icon.png"))
    if pixmap.isNull():
        return
    ratio = widget_pixel_ratio(label)
    physical = max(1, int(round(side * ratio)))
    pixmap = pixmap.scaled(physical, physical, Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
    pixmap.setDevicePixelRatio(ratio)
    label.setPixmap(pixmap)


def _version_above(candidate: object, reference: object) -> bool:

    from ...core.server_dials import parse_version

    left, right = parse_version(candidate), parse_version(reference)
    if left is None or right is None:
        return False
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)) > right + (0,) * (width - len(right))


def _min_supported_version() -> str:

    from ...core.server_dials import parse_version, read_value

    value = read_value("min_supported_version")
    return value.strip() if parse_version(value) is not None else ""


def _served_update_policy() -> str:

    from ...core.server_dials import read_value

    value = read_value("update_policy")
    value = value.strip().lower() if isinstance(value, str) else ""
    return value if value in _UPDATE_POLICIES else _UPDATE_POLICY_DEFAULT


def _dismissed_update_versions() -> set[str]:
    return _DISMISSED_UPDATE_VERSIONS


class DockAboutMixin:


    def _setup_update_notification(self):










        from qgis.PyQt.QtWidgets import QSizePolicy

        from .font_scale import scale_qss_font_px

        self._update_notif_container = QWidget()
        self._update_notif_container.setObjectName("updateCard")
        self._update_notif_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._update_notif_container.setStyleSheet(scale_qss_font_px(_update_card_qss()))


        policy = self._update_notif_container.sizePolicy()
        policy.setVerticalPolicy(QSizePolicy.Policy.Minimum)
        policy.setHeightForWidth(True)
        self._update_notif_container.setSizePolicy(policy)
        card = QVBoxLayout(self._update_notif_container)
        card.setContentsMargins(12, 10, 12, 10)
        card.setSpacing(8)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(10)
        icon_label = QLabel()
        icon_label.setFixedSize(_UPDATE_ICON_PX, _UPDATE_ICON_PX)
        _set_plugin_mark(icon_label, _UPDATE_ICON_PX)
        head.addWidget(icon_label, 0, Qt.AlignmentFlag.AlignTop)

        words = QVBoxLayout()
        words.setContentsMargins(0, 0, 0, 0)
        words.setSpacing(2)
        self._update_badge = QWidget()
        self._update_badge.setObjectName("updateBadge")
        self._update_badge.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        badge_row = QHBoxLayout(self._update_badge)
        badge_row.setContentsMargins(8, 3, 8, 3)
        badge_row.setSpacing(0)
        badge_text = QLabel(tr("Update required"))
        badge_text.setObjectName("updateBadgeText")
        badge_row.addWidget(badge_text)
        self._update_badge.setVisible(False)
        words.addWidget(self._update_badge, 0, Qt.AlignmentFlag.AlignLeft)
        words.addSpacing(2)
        self._update_title_label = QLabel("")
        self._update_title_label.setObjectName("updateTitle")
        self._update_title_label.setWordWrap(True)
        self._update_title_label.setTextFormat(Qt.TextFormat.PlainText)
        words.addWidget(self._update_title_label)



        self.update_notification_label = QLabel("")
        self.update_notification_label.setObjectName("updateNote")
        self.update_notification_label.setWordWrap(True)
        self.update_notification_label.setTextFormat(Qt.TextFormat.PlainText)
        note_policy = self.update_notification_label.sizePolicy()
        note_policy.setHeightForWidth(True)
        self.update_notification_label.setSizePolicy(note_policy)
        words.addWidget(self.update_notification_label)
        head.addLayout(words, 1)
        card.addLayout(head)


        self._update_body_label = QLabel(tr(
            "Update to keep using AI Segmentation. It takes one click in the "
            "QGIS Plugin Manager; the plugin reloads on its own."))
        self._update_body_label.setObjectName("updateBody")
        self._update_body_label.setWordWrap(True)
        self._update_body_label.setVisible(False)
        card.addWidget(self._update_body_label)

        self._update_now_btn = QPushButton(tr("Update now"))
        self._update_now_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_now_btn.setStyleSheet(_BTN_GREEN_STEP)
        self._update_now_btn.setFixedHeight(scale_px_length(BTN_PRIMARY_WIDE_PX))
        self._update_now_btn.setAutoDefault(False)
        self._update_now_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._update_now_btn.clicked.connect(self._on_open_plugin_manager)
        card.addWidget(self._update_now_btn)

        later_row = QHBoxLayout()
        later_row.setContentsMargins(0, 0, 0, 0)
        later_row.addStretch()
        self._update_later_btn = QPushButton(tr("Later"))
        self._update_later_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_later_btn.setStyleSheet(_UPDATE_LATER_STYLE)
        self._update_later_btn.setAutoDefault(False)
        self._update_later_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._update_later_btn.clicked.connect(self._on_dismiss_update_banner)
        later_row.addWidget(self._update_later_btn)


        self._update_hint_label = QLabel("")
        self._update_hint_label.setObjectName("updateHint")
        self._update_hint_label.setVisible(False)
        words.addWidget(self._update_hint_label)
        later_row.addStretch()
        card.addLayout(later_row)

        self._update_notif_container.setVisible(False)
        self.update_notification_widget = self._update_notif_container
        self.main_layout.addWidget(self.update_notification_widget)
        self._build_update_gate_page()

    def _build_update_gate_page(self) -> None:









        from .font_scale import scale_qss_font_px

        page = _UpdateGatePage()
        page.setObjectName("updateGatePage")
        page.setStyleSheet(scale_qss_font_px(_update_gate_qss()))
        outer = QVBoxLayout(page)
        outer.setContentsMargins(16, 24, 16, 24)
        outer.setSpacing(0)
        outer.addStretch(1)
        col = QVBoxLayout(page.card)
        col.setContentsMargins(22, 22, 22, 22)
        col.setSpacing(10)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(12)
        mark = QLabel(page.card)
        mark.setFixedSize(_GATE_ICON_PX, _GATE_ICON_PX)
        _set_plugin_mark(mark, _GATE_ICON_PX)
        head.addWidget(mark, 0, Qt.AlignmentFlag.AlignVCenter)
        badge = QFrame(page.card)
        badge.setObjectName("updateGateBadge")
        badge.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        badge.setFixedHeight(_GATE_BADGE_PX)
        badge_row = QHBoxLayout(badge)
        badge_row.setContentsMargins(10, 0, 10, 0)
        badge_text = QLabel(tr("Update required"), badge)
        badge_text.setObjectName("updateGateBadgeText")
        badge_row.addWidget(badge_text)
        head.addWidget(badge, 0, Qt.AlignmentFlag.AlignVCenter)
        head.addStretch(1)
        col.addLayout(head)

        self._update_gate_title = QLabel("", page.card)
        self._update_gate_title.setObjectName("updateGateTitle")
        self._update_gate_title.setWordWrap(True)
        self._update_gate_title.setTextFormat(Qt.TextFormat.PlainText)
        col.addWidget(self._update_gate_title)

        self._update_gate_note = QLabel("", page.card)
        self._update_gate_note.setObjectName("updateGateNote")
        self._update_gate_note.setWordWrap(True)
        self._update_gate_note.setTextFormat(Qt.TextFormat.PlainText)
        self._update_gate_note.setVisible(False)
        col.addWidget(self._update_gate_note)
        body = QLabel(tr(
            "Update to keep using AI Segmentation. It takes one click in the "
            "QGIS Plugin Manager; the plugin reloads on its own."), page.card)
        body.setObjectName("updateGateBody")
        body.setWordWrap(True)
        col.addWidget(body)

        col.addSpacing(4)
        button = QPushButton(tr("Update now"), page.card)
        button.setObjectName("updateGateButton")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setStyleSheet(_BTN_GREEN_STEP)
        button.setFixedHeight(scale_px_length(BTN_PRIMARY_WIDE_PX))
        button.setAutoDefault(False)
        button.clicked.connect(self._on_open_plugin_manager)
        col.addWidget(button)
        self._update_gate_hint = QLabel("", page.card)
        self._update_gate_hint.setObjectName("updateGateHint")
        self._update_gate_hint.setWordWrap(True)
        self._update_gate_hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        col.addWidget(self._update_gate_hint)

        outer.addWidget(page.card, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addStretch(2)
        page.setVisible(False)
        self.update_gate_page = page

    def _update_gate_waits(self) -> bool:







        return any(bool(getattr(self, name, False)) for name in (
            "_segmentation_active", "_auto_run_active", "_auto_review_active",
            "_auto_finalizing"))

    def _sync_update_gate(self, required: bool) -> bool:

        page = getattr(self, "update_gate_page", None)
        body = getattr(self, "_dock_scroll_area", None)
        if page is None or body is None:
            return False
        shown = bool(required) and not self._update_gate_waits()
        try:
            page.setVisible(shown)
            body.setVisible(not shown)
        except RuntimeError:
            return False
        return shown

    def is_update_offer_shown(self) -> bool:

        try:
            page = getattr(self, "update_gate_page", None)
            return bool(self.update_notification_widget.isVisibleTo(self)
                        or (page is not None and page.isVisibleTo(self)))
        except (RuntimeError, AttributeError):
            return False

    def check_for_updates(self):














        try:
            served, too_old = self._update_offer_state()
            if not served and not too_old:
                self._clear_update_banner()
                return
            version = self._upgradeable_version()
            if not version:
                self._maybe_refresh_plugin_repository()
                return
            required = too_old or _served_update_policy() == "require"
            if required and too_old and _version_above(_min_supported_version(), version):



                from qgis.core import Qgis, QgsMessageLog

                QgsMessageLog.logMessage(
                    f"The plugin repository only offers {version}, below the "
                    f"{_min_supported_version()} the service requires; no update is offered.",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
                self._clear_update_banner()
                return
            self._show_update_banner(version, required=required)
        finally:


            self.refresh_update_recommendation()

    def _update_offer_state(self) -> tuple[str, bool]:

        installed = self._installed_version()
        if not installed:
            return "", False
        too_old = _version_above(_min_supported_version(), installed)
        return self._served_newer_version(), too_old

    def _show_update_banner(self, version: str, required: bool = False) -> None:








        if not required and version in _dismissed_update_versions():
            self._clear_update_banner()
            return
        self._repo_offered_version = version
        self._update_title_label.setText(
            tr("AI Segmentation {version} is out").format(version=version))
        note = self._served_update_line()
        self.update_notification_label.setText(note)
        self.update_notification_label.setVisible(bool(note))
        installed = self._installed_version()
        hint = tr("You have {installed}.").format(installed=installed) if installed else ""
        self._update_badge.setVisible(required)
        self._update_body_label.setVisible(required)
        self._update_later_btn.setVisible(not required)
        self._update_hint_label.setText(hint)
        self._update_hint_label.setVisible(required and bool(installed))


        self._update_gate_title.setText(self._update_title_label.text())
        self._update_gate_note.setText(note)
        self._update_gate_note.setVisible(bool(note))
        self._update_gate_hint.setText(hint)
        self._update_gate_hint.setVisible(bool(installed))
        gated = self._sync_update_gate(required)
        self._update_card_pending = not gated
        self._update_card_required = bool(required)
        self._sync_update_card_for_work()
        from .server_switches import UPDATE_TRIGGER_SERVED_LATEST

        self._track_update_prompt_shown(version, UPDATE_TRIGGER_SERVED_LATEST)

    def _sync_update_card_for_work(self) -> None:









        try:
            pending = bool(getattr(self, "_update_card_pending", False))
            required = bool(getattr(self, "_update_card_required", False))
            shown = pending and (required or not self._update_gate_waits())
            card = self.update_notification_widget
            if card.isHidden() == shown:
                card.setVisible(shown)
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _clear_update_banner(self) -> None:

        self._update_card_pending = False
        try:
            self.update_notification_widget.setVisible(False)
        except (RuntimeError, AttributeError):
            pass  # nosec B110
        self._sync_update_gate(False)

    def _plugin_installer_key(self) -> str:



        import os

        return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))))

    def _upgradeable_version(self) -> str:

        try:
            from pyplugin_installer.installer_data import plugins

            data = plugins.all().get(self._plugin_installer_key())
            if data and data.get("status") == "upgradeable":
                return str(data.get("version_available") or "")
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        return ""

    def _maybe_refresh_plugin_repository(self) -> None:







        if getattr(self, "_update_refresh_requested", False):
            return
        served = self._served_newer_version() or _min_supported_version()
        if not served:
            return
        self._update_refresh_requested = True
        try:
            from pyplugin_installer.installer_data import repositories
        except Exception:  # noqa: BLE001
            self._track_update_prompt_suppressed(served, "installer_unavailable")
            return
        try:
            enabled = list(repositories.allEnabled())
            if not enabled:
                self._track_update_prompt_suppressed(served, "no_repository")
                return
            repositories.checkingDone.connect(self._on_plugin_repository_checked)
            for key in enabled:
                repositories.requestFetching(key, force_reload=True)
        except Exception:  # noqa: BLE001
            self._track_update_prompt_suppressed(served, "refresh_failed")

    def _on_plugin_repository_checked(self) -> None:

        served = self._served_newer_version() or _min_supported_version()
        try:
            from pyplugin_installer.installer_data import plugins, repositories

            try:
                repositories.checkingDone.disconnect(
                    self._on_plugin_repository_checked)
            except (TypeError, RuntimeError):
                pass  # nosec B110
            plugins.rebuild()
        except Exception:  # noqa: BLE001
            if served:
                self._track_update_prompt_suppressed(served, "refresh_failed")
            return
        try:
            if not self._upgradeable_version():
                if served:
                    self._track_update_prompt_suppressed(served, "not_listed")
                return
            self.check_for_updates()
        except RuntimeError:
            pass  # nosec B110

    def _served_newer_version(self) -> str:




        try:
            from ...core.activation_manager import (
                get_latest_version,
                is_update_available,
            )

            installed = self._installed_version()
            if not installed:
                return ""
            latest = get_latest_version()
            if latest and is_update_available(installed):
                return str(latest)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        return ""

    def _track_update_prompt_suppressed(self, served_version: str,
                                        reason: str) -> None:

        seen = getattr(self, "_update_suppressed_reported", None)
        if seen is None:
            seen = set()
            self._update_suppressed_reported = seen
        if served_version in seen:
            return
        seen.add(served_version)
        try:
            from ...core import telemetry_events as ev
            from ...core.telemetry import track

            track(ev.PLUGIN_UPDATE_PROMPT_SUPPRESSED,
                  {"served_version": served_version, "reason": reason})
        except Exception:  # noqa: BLE001
            pass  # nosec B110

    @staticmethod
    def _served_update_line() -> str:

        try:
            from ...core.activation_manager import get_release_notes_line

            return get_release_notes_line() or ""
        except Exception:  # noqa: BLE001
            return ""

    def _on_dismiss_update_banner(self) -> None:

        version = getattr(self, "_repo_offered_version", "") or ""
        self._clear_update_banner()
        if version:
            _dismissed_update_versions().add(version)
        self._track_update_prompt_clicked(version, "dismissed")

    def _on_open_plugin_manager(self, _link=None):

        from .server_switches import open_plugin_manager_or_marketplace

        landed = open_plugin_manager_or_marketplace()
        self._track_update_prompt_clicked(
            getattr(self, "_repo_offered_version", ""),
            "plugin_manager" if landed else "marketplace_page")

    def _setup_about_section(self):




        self.batch_info_widget = QWidget()
        self.batch_info_widget.setObjectName("batchInfoCard")
        self.batch_info_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.batch_info_widget.setStyleSheet(_msg_card_qss("batchInfoCard", "info"))
        batch_info_layout = QHBoxLayout(self.batch_info_widget)


        batch_info_layout.setContentsMargins(12, 10, 12, 10)
        batch_info_layout.setSpacing(10)




        batch_info_icon = category_tile("info", HUE_HOWTO, side=CATEGORY_TILE_PX,
                                        glyph_px=CATEGORY_TILE_GLYPH_PX)
        batch_info_layout.addWidget(batch_info_icon, 0, Qt.AlignmentFlag.AlignTop)

        info_msg = "{}\n{}".format(
            dial_copy("batch_tip.one_element",
                      tr("The AI model works best on one element at a time.")),
            dial_copy("batch_tip.save_before_next",
                      tr("Save your polygon before selecting the next element.")))
        batch_info_text = QLabel(info_msg)
        batch_info_text.setWordWrap(True)
        batch_info_text.setStyleSheet(_HINT_LINE_QSS)
        batch_info_layout.addWidget(batch_info_text, 1)

        self.batch_info_widget.setVisible(False)
        self.main_layout.addWidget(self.batch_info_widget)







        self._setup_auto_review_view_block(self.main_layout)





    def _on_open_guide_footer(self):





        from .guidance import open_guide
        open_guide("footer_tutorial")
