





















from __future__ import annotations

from .i18n import tr






PRO_LOGIN_TARGET = "pricing"




_LOGIN_LINK_WAIT_MS = 6_000




_PENDING: set = set()


def _report_click(telemetry_source: str, checkout_link: str) -> None:
    try:
        from . import telemetry_session_events
        telemetry_session_events.track_pro_upsell_clicked(
            source=telemetry_source, checkout_link=checkout_link)
    except Exception:  # noqa: BLE001
        pass  # nosec B110


def _set_busy(busy: bool) -> None:


    try:
        from qgis.PyQt.QtWidgets import QApplication

        from .qt_compat import WaitCursor
        if busy:
            QApplication.setOverrideCursor(WaitCursor)
        else:
            QApplication.restoreOverrideCursor()
    except Exception:  # noqa: BLE001
        pass  # nosec B110


def open_pro_page(
    cta_source: str,
    telemetry_source: str,
    parent=None,
    fallback_url: str | None = None,
    target: str = PRO_LOGIN_TARGET,
    on_done=None,
) -> None:







    from .activation_manager import (
        get_auth_header,
        get_pro_checkout_url,
        is_plugin_activated,
    )

    plain_url = fallback_url or get_pro_checkout_url(cta_source)

    def finish(url: str, checkout_link: str) -> None:
        _set_busy(False)
        _report_click(telemetry_source, checkout_link)
        try:
            from ..ui.external_links import open_external_url
            open_external_url(url, parent=parent)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        if on_done is not None:
            try:
                on_done(url)
            except Exception:  # noqa: BLE001
                pass  # nosec B110

    auth = get_auth_header() if is_plugin_activated() else {}
    if not auth:


        finish(plain_url, "fallback")
        return

    try:
        from qgis.core import QgsApplication

        from ..api.terralab_client import TerraLabClient
        from ..workers.generic_request_task import GenericRequestTask
        client = TerraLabClient()
        task = GenericRequestTask(
            tr("Opening the plans page"),
            lambda: client.get_plugin_login_link(
                target, cta_source, auth=auth, locale=_ui_locale()),
            hidden=True,
        )
    except Exception:  # noqa: BLE001

        finish(plain_url, "fallback")
        return



    state = {"done": False}

    def settle(url: str, checkout_link: str) -> None:
        if state["done"]:
            return
        state["done"] = True
        _PENDING.discard(task)
        finish(url, checkout_link)

    def on_answer(answer) -> None:
        url = answer.get("url") if isinstance(answer, dict) else None
        if isinstance(url, str) and url.startswith("https://"):
            settle(url, "direct")
        else:


            settle(plain_url, "fallback")

    task.succeeded.connect(on_answer)
    task.failed.connect(lambda *_a: settle(plain_url, "fallback"))

    _set_busy(True)
    try:


        from .qt_compat import safe_single_shot
        from .server_dials import dial_in_range
        wait_ms = dial_in_range(
            "tuning.pairing.login_link_wait_ms", _LOGIN_LINK_WAIT_MS, 2000, 20000)
        safe_single_shot(wait_ms, task,
                         lambda: settle(plain_url, "fallback"))
    except Exception:  # noqa: BLE001
        pass  # nosec B110

    _PENDING.add(task)
    try:
        QgsApplication.taskManager().addTask(task)
    except Exception:  # noqa: BLE001
        settle(plain_url, "fallback")


def _ui_locale() -> str | None:


    try:
        from .i18n import current_locale
        code = current_locale()
        return code.split("_")[0] if code else None
    except Exception:  # noqa: BLE001
        return None
