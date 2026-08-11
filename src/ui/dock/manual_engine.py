




























from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ...core.cloud_notice_seen import cloud_notice_seen
from ...core.i18n import tr
from ...core.manual_cloud_route import (
    manual_cloud_route_enabled,
    manual_cloud_route_offered,
    set_manual_cloud_route_enabled,
)
from ...core.server_dials import dial_copy
from .cloud_notice_line import build_cloud_notice_line, cloud_notice_line_html
from .font_scale import scale_qss_font_px
from .manual_local_install_dialog import local_install_minutes
from .pro_nudges import PRO_CARD_MANUAL_LOW, pro_card_dismissed
from .styles import (
    _CARD_CHILD_BTN_RESET_QSS,
    _HINT_LINE_QSS,
    _SUBCARD_MARGINS,
    _SUBCARD_QSS,
    FONT_HINT,
    INK,
)
from .ui_refresh import format_quota_count
from .upsell_card import UpsellCard, keep_working_cta
from .widgets import Mode, _EngineSwitch




_DOT = "·"


def _fill_engine_line(text: str, **parts: str) -> str:







    out = text.replace("{dot}", _DOT)
    for name, value in parts.items():
        out = out.replace("{" + name + "}", value)
    return out


class DockManualEngineMixin:




    def _setup_manual_engine(self) -> None:






        card = QWidget()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)









        engine_header = QLabel(dial_copy(
            "engine.header", tr("Where the segmentation runs:")).rstrip(": "))
        engine_header.setWordWrap(True)
        engine_header.setStyleSheet(_HINT_LINE_QSS)


        layout.addWidget(engine_header)



        self.manual_engine_switch = _EngineSwitch(cloud=True)
        self.manual_engine_switch.engine_selected.connect(
            self._on_manual_engine_picked)
        layout.addWidget(self.manual_engine_switch)










        note_box = QWidget()
        note_box.setObjectName("manualEngineNote")
        note_box.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        note_layout = QVBoxLayout(note_box)
        note_layout.setContentsMargins(*_SUBCARD_MARGINS)
        note_layout.setSpacing(4)

        self.manual_engine_note = QLabel("")
        self.manual_engine_note.setWordWrap(True)
        self.manual_engine_note.setTextFormat(Qt.TextFormat.RichText)
        self.manual_engine_note.setStyleSheet(scale_qss_font_px(
            f"font-size: {FONT_HINT}px; color: {INK}; background: transparent;"))
        note_layout.addWidget(self.manual_engine_note)





        self.manual_engine_privacy_line = build_cloud_notice_line()
        note_layout.addWidget(self.manual_engine_privacy_line)

        self.manual_engine_note_box = note_box
        layout.addWidget(note_box)








        self.manual_engine_low_line = UpsellCard(
            "manualEngineLowCard", "compact",
            on_cta=self._on_manual_low_credit_cta)
        self.manual_engine_low_line.setVisible(False)

        self.manual_engine_low_line.dismissed.connect(
            lambda: self._dismiss_pro_card(
                PRO_CARD_MANUAL_LOW, self.manual_engine_low_line))






        card.setVisible(False)
        self.manual_engine_card = card


        self._manual_engine_card_tinted: str | None = None
















        self.start_container.layout().insertWidget(0, card)



    def _manual_cloud_route_picked(self) -> bool:









        try:
            return bool(
                self._plugin_activated
                and manual_cloud_route_enabled()
                and manual_cloud_route_offered()
            )
        except (RuntimeError, AttributeError):
            return False

    def _manual_engine_offered(self) -> bool:





        try:
            return bool(manual_cloud_route_offered())
        except Exception:  # noqa: BLE001
            return False

    def _manual_engine_local_ready(self) -> bool:

        return bool(getattr(self, "_dependencies_ok", False)
                    and getattr(self, "_checkpoint_ok", False))

    def _manual_install_running(self) -> bool:














        try:
            if not self._progress_timer.isActive():
                return False
        except (RuntimeError, AttributeError):
            return False
        return bool(getattr(self, "_manual_install_wants_model", False)
                    or getattr(self, "_auto_review_installing", False))



    def _manual_engine_card_wanted(self) -> bool:

        return bool(
            self._plugin_activated
            and self._mode == Mode.INTERACTIVE
            and manual_cloud_route_offered()
            and not self._segmentation_active


            and self.layer_combo.count_layers() > 0




            and not (self._manual_cloud_route_picked()
                     and self._manual_credits_exhausted())
        )

    def _sync_manual_engine_card_visibility(self) -> None:







        card = getattr(self, "manual_engine_card", None)
        if card is None:
            return
        try:
            if card.isHidden() == self._manual_engine_card_wanted():
                self._refresh_manual_engine_ui()
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _refresh_manual_engine_ui(self) -> None:

        card = getattr(self, "manual_engine_card", None)
        if card is None:
            return
        try:
            on_start_view = self._manual_engine_card_wanted()
            card.setVisible(on_start_view)
            if on_start_view:
                self._refresh_manual_engine_card_text()
            else:



                self._write_manual_low_credit_line(False)
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _refresh_manual_engine_card_text(self) -> None:

        cloud = self._manual_cloud_route_picked()
        low = cloud and self._manual_engine_credits_low()


        pro_contact = False
        try:


            self.manual_engine_switch.set_cloud(cloud)
            self.manual_engine_switch.set_cloud_gloss(
                self._manual_engine_cloud_gloss())
            self._paint_manual_engine_card()
            note = self._manual_engine_copy(cloud)
            self.manual_engine_note.setText(note)
            self.manual_engine_note.setVisible(bool(note))
            self.manual_engine_privacy_line.setText(cloud_notice_line_html())


            privacy = cloud and not cloud_notice_seen()
            self.manual_engine_privacy_line.setVisible(privacy)


            self.manual_engine_note_box.setVisible(bool(note) or privacy)
            self._write_manual_low_credit_line(low, pro_contact)
        except (RuntimeError, AttributeError):
            pass  # nosec B110
        if low:
            self._note_manual_low_credit_offer()

    def _manual_engine_cloud_gloss(self) -> str:











        return dial_copy("engine.cloud_gloss",
                         tr("Bigger model, more accurate"),
                         max_chars=40)

    def _note_manual_low_credit_offer(self) -> None:





        if getattr(self, "_manual_low_credit_seen", False):
            return
        self._manual_low_credit_seen = True
        try:
            from ...core import telemetry_session_events

            telemetry_session_events.track_pro_upsell_viewed(
                trigger="manual_credits_low")
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _manual_engine_credits_low(self) -> bool:











        try:
            from ...core.credit_gate import (
                low_credit_ceiling,
                low_credit_threshold,
            )

            if getattr(self, "_auto_is_subscriber", False):
                return False
            env = getattr(self, "_quota_envelopes", None)
            if env is not None and env.objects_remaining is not None:


                left, total = env.objects_remaining, env.objects_cap
            else:
                left = getattr(self, "_auto_credits", None)
                total = getattr(self, "_auto_credits_total", None)
            if left is None or not total or int(total) <= 0:
                return False
            if int(left) > low_credit_ceiling():
                return False
            return 0 < int(left) <= int(total) * low_credit_threshold()
        except Exception:  # noqa: BLE001
            return False

    def _manual_pro_ceiling_objects_low(self) -> bool:







        try:
            from ...core.pro_ceiling import (
                pro_ceiling_enabled,
                pro_ceiling_low_fraction,
            )

            if not getattr(self, "_auto_is_subscriber", False):
                return False
            if not pro_ceiling_enabled():
                return False
            env = getattr(self, "_quota_envelopes", None)
            if env is None or not env.has_objects_gauge():
                return False
            left = env.objects_remaining
            if left is None:
                left = max(0, int(env.objects_cap) - int(env.objects_used))
            cap = int(env.objects_cap)
            if cap <= 0:
                return False
            return 0 < int(left) <= cap * pro_ceiling_low_fraction()
        except Exception:  # noqa: BLE001
            return False

    def _paint_manual_engine_card(self) -> None:













        if self._manual_engine_card_tinted == "":
            return
        self._manual_engine_card_tinted = ""
        qss = (_SUBCARD_QSS.format(name="manualEngineNote")
               + "QLabel { background: transparent; border: none; }")
        try:
            self.manual_engine_note_box.setStyleSheet(
                qss + _CARD_CHILD_BTN_RESET_QSS)
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _manual_engine_copy(self, cloud: bool) -> str:













        if cloud:
            return ""
        if not self._manual_engine_local_ready():
            if self._manual_install_running():




                return _fill_engine_line(dial_copy(
                    "engine.installing_line",
                    tr("Setting up on your computer {dot} <b>wait for it "
                       "to finish</b>")))
            if getattr(self, "_manual_install_failed", False):


                return _fill_engine_line(dial_copy(
                    "engine.install_failed_line",
                    tr("The install did not finish {dot} <b>retry it, or "
                       "pick Cloud AI</b>")))
            return self._manual_engine_install_copy()
        return ""

    def _on_manual_low_credit_link(self, url: str) -> None:








        from ...core.pro_page_link import open_pro_page
        open_pro_page("plugin_objects_low_note", "manual_credits_low",
                      parent=self, fallback_url=url)

    def _on_manual_low_credit_cta(self) -> None:






        try:
            url = self._build_upgrade_url("plugin_objects_low_note")
        except (RuntimeError, AttributeError):
            return
        self._on_manual_low_credit_link(url)

    def _on_pro_contact_objects_low(self) -> None:

        line = getattr(self, "manual_engine_low_line", None)
        self._on_pro_contact_clicked(
            "objects_low", getattr(line, "button", None))

    def _write_manual_low_credit_line(self, low: bool,
                                      pro_contact: bool = False) -> None:












        line = getattr(self, "manual_engine_low_line", None)
        if line is None:
            return
        try:
            if not low and not pro_contact:
                line.setVisible(False)
                return
            if pro_contact:
                env = self._quota_envelopes
                objects_left = (env.objects_remaining
                                if env.objects_remaining is not None
                                else max(0, env.objects_cap - env.objects_used))
                title = dial_copy(
                    "pro_ceiling.low_title_objects",
                    tr("{left} of {cap} Semi-Auto objects left this month"))
                title = (title.replace("{left}", format_quota_count(objects_left))
                              .replace("{cap}", format_quota_count(env.objects_cap)))
                body, cta = self._pro_ceiling_copy()
                line.route_cta(self._on_pro_contact_objects_low)
                line.enable_dismiss(False)
                line.set_text(title, body, cta, detail=self._pro_ceiling_detail())
                line.setVisible(True)
                return
            if pro_card_dismissed(PRO_CARD_MANUAL_LOW):
                line.setVisible(False)
                return
            line.route_cta(self._on_manual_low_credit_cta)
            line.enable_dismiss(True)
            reset_day = getattr(self, "_auto_reset_display", "")

            cta = keep_working_cta()
            env = getattr(self, "_quota_envelopes", None)
            if env is not None and env.has_objects_gauge():





                if reset_day:
                    title = dial_copy(
                        "upsell.low_title_objects_reset",
                        tr("{n} of {total} cloud objects left in Semi-Auto, back on "
                           "{date}."))
                else:
                    title = dial_copy(
                        "upsell.low_title_objects",
                        tr("{n} of {total} cloud objects left in Semi-Auto this month."))



                objects_left = (env.objects_remaining
                                if env.objects_remaining is not None
                                else max(0, env.objects_cap - env.objects_used))
                title = (title.replace("{n}", format_quota_count(objects_left))
                              .replace("{total}", format_quota_count(env.objects_cap))
                              .replace("{date}", reset_day))
            else:
                left = int(getattr(self, "_auto_credits", 0) or 0)
                if reset_day:
                    title = dial_copy(
                        "upsell.low_title_detections_reset",
                        tr("{n} cloud detections left, back on {date}."))
                else:
                    title = dial_copy(
                        "upsell.low_title_detections",
                        tr("{n} cloud detections left."))
                title = (title.replace("{n}", format_quota_count(left))
                              .replace("{date}", reset_day))





            line.set_text(title, None, cta)
            line.set_pro_offer("plugin_objects_low_note")
            line.setVisible(True)
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _manual_engine_install_copy(self) -> str:










        minutes = str(local_install_minutes())
        fallback = _fill_engine_line(
            dial_copy("engine.install_line_no_disk",
                      tr("Everything stays on this computer {dot} <b>about "
                         "{n} minutes to install</b>")),
            n=minutes)





        try:
            from ...core.venv_manager import resolved_min_free_gb_full

            gb = float(resolved_min_free_gb_full())
        except Exception:  # noqa: BLE001
            return fallback
        if gb <= 0:
            return fallback
        return _fill_engine_line(
            dial_copy("engine.install_line",
                      tr("Everything stays on this computer {dot} <b>{gb} GB "
                         "and about {n} minutes to install</b>")),
            gb=f"{gb:g}", n=minutes)

    def _refresh_manual_engine_card_enabled(self) -> None:







        try:
            ready = self.layer_combo.currentLayer() is not None
            self._refresh_manual_start_action(ready)
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _refresh_manual_start_action(self, ready: bool) -> None:







        button = getattr(self, "start_button", None)
        if button is None:
            return





        installing = self._manual_install_running()
        needs_install = bool(
            manual_cloud_route_offered()
            and not self._manual_cloud_route_picked()
            and not self._manual_engine_local_ready()
        )
        try:
            if needs_install and not installing:
                button.setText(tr("Install the offline AI"))
                button.setEnabled(ready and not self._segmentation_active)
            else:
                button.setText(tr("Start Semi-Auto AI Segmentation"))







            button.setToolTip("" if button.isEnabled()
                              else self._manual_start_blocked_reason(ready))
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _manual_start_blocked_reason(self, ready: bool) -> str:






        if self._manual_install_running():
            return tr("The offline AI is still downloading.")
        if getattr(self, "_manual_install_failed", False):
            return tr("The install did not finish. Retry it, or pick Cloud AI.")
        if self._segmentation_active:
            return tr("A session is already running.")
        if not ready:
            return tr("Pick a raster layer and accept the Terms to start.")



        if (not self._manual_cloud_route_picked()
                and not self._manual_engine_local_ready()):
            return tr("The offline AI is not installed yet.")
        return ""

    def set_manual_cloud_session_spend(self, saved: int) -> None:






        self._manual_cloud_objects_saved = max(0, int(saved))



    def _track_manual_engine(self, what: str,
                             from_install_gate: bool = False) -> None:

        try:
            from ...core import telemetry_session_events

            telemetry_session_events.track_manual_engine_chosen(
                what,
                local_installed=self._manual_engine_local_ready(),
                from_install_gate=from_install_gate,
            )
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _on_manual_engine_picked(self, cloud: bool) -> None:








        if not cloud and not self._manual_engine_local_ready():
            self._open_manual_local_install()
            return
        self._set_manual_engine_cloud(bool(cloud))

    def _set_manual_engine_cloud(self, on: bool) -> None:



        from_gate = bool(on and not self._manual_engine_local_ready())
        set_manual_cloud_route_enabled(on)
        self._track_manual_engine("cloud" if on else "offline",
                                  from_install_gate=from_gate)





        try:
            self.manual_engine_changed.emit(bool(on))
        except (RuntimeError, AttributeError):
            pass  # nosec B110



        try:
            self._sync_refine_shape_toggles()
        except (RuntimeError, AttributeError):
            pass  # nosec B110
        self._update_full_ui()

    def _on_dock_hidden_reset_engine(self, visible: bool) -> None:

        if not visible:
            self.reset_manual_engine_to_cloud()

    def reset_manual_engine_to_cloud(self) -> None:













        if getattr(self, "_segmentation_active", False):
            return
        try:
            if manual_cloud_route_enabled():
                return
            set_manual_cloud_route_enabled(True)
        except Exception:  # noqa: BLE001
            return
        switch = getattr(self, "manual_engine_switch", None)
        if switch is not None:
            try:
                switch.set_engine_cloud(True)
            except RuntimeError:
                pass  # nosec B110


        for settle in ("_sync_refine_shape_toggles", "_refresh_manual_engine_ui"):
            try:
                getattr(self, settle)()
            except (RuntimeError, AttributeError):
                pass  # nosec B110

    def _manual_engine_gate_start(self) -> bool:










        try:
            if self._mode != Mode.INTERACTIVE or not manual_cloud_route_offered():
                return True
            cloud = self._manual_cloud_route_picked()
        except (RuntimeError, AttributeError):
            return True

        if cloud:
            return True

        if self._manual_engine_local_ready():
            return True



        self._open_manual_local_install()
        return False
