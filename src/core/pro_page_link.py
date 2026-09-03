"""One door from every Pro button in the plugin to the plans page, signed in.

A Pro button used to open the website in a browser that was signed out, so a
magic link sent to a mailbox stood between the press and the page. The lost
step was never the price.

The plugin already holds an activation key, so it asks the server for a
one-time login link to the pricing page and opens that. The user lands signed
in, reads the plans and the price, and buys from the page that has both. No
mail, no waiting.

Nothing about the old path is removed: on a refusal, an error, a timeout or an
older server, the plain dashboard URL opens exactly as it always did, and the
user is one login from the same place. The request runs on a QgsTask and the
URL opens in its callback, so the GUI thread never waits.

`pro_upsell_clicked` is reported here, once, after the outcome is known, so
every click carries `checkout_link` = direct or fallback and the two paths can
be compared. The property keeps that name because the website event registry
is generated from its own source and the plugin cannot rename a property on
its own; `direct` now means the one-time login link opened.

The ready-made checkout link (`TerraLabClient.get_pro_checkout_link`) is
still there and nothing calls it. It is the second door, for the day a CTA sells a
plan the user has already chosen.
"""
from __future__ import annotations

from .i18n import tr

# What the server should sign the user into. The plans page: they read what
# Pro contains and pick a plan there, rather than meeting one price with no
# way to compare it.
PRO_LOGIN_TARGET = "pricing"

# How long the button may sit there before we give up and open the plain URL.
# The client enforces the same budget on the socket; this is the backstop for a
# task that never reports at all.
_LOGIN_LINK_WAIT_MS = 6_000


def _report_click(telemetry_source: str, checkout_link: str) -> None:
    try:
        from . import telemetry_session_events
        telemetry_session_events.track_pro_upsell_clicked(
            source=telemetry_source, checkout_link=checkout_link)
    except Exception:  # noqa: BLE001 -- telemetry never blocks a click
        pass  # nosec B110


def _set_busy(busy: bool) -> None:
    """Wait cursor while the server answers. Cheap, and it is the only sign the
    press did something during the second it can take."""
    try:
        from qgis.PyQt.QtWidgets import QApplication

        from .qt_compat import WaitCursor
        if busy:
            QApplication.setOverrideCursor(WaitCursor)
        else:
            QApplication.restoreOverrideCursor()
    except Exception:  # noqa: BLE001 -- a cursor is never worth an exception
        pass  # nosec B110


def open_pro_page(
    cta_source: str,
    telemetry_source: str,
    parent=None,
    fallback_url: str | None = None,
    target: str = PRO_LOGIN_TARGET,
    on_done=None,
) -> None:
    """Send the user to the plans page signed in, or to the plain URL.

    ``cta_source`` names the plugin surface for the server and for the URL;
    ``telemetry_source`` is the existing ``pro_upsell_clicked`` source, kept
    unchanged so no dashboard breaks. ``on_done`` is called with the URL that
    was opened, after the browser has been asked.
    """
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
        except Exception:  # noqa: BLE001 -- a dead browser must not raise here
            pass  # nosec B110
        if on_done is not None:
            try:
                on_done(url)
            except Exception:  # noqa: BLE001
                pass  # nosec B110

    auth = get_auth_header() if is_plugin_activated() else {}
    if not auth:
        # No key means the server cannot know who is signing in. This is the
        # free user who never activated, and the plain page is where they go.
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
    except Exception:  # noqa: BLE001 -- QGIS without a task manager, or an
        # import that is not there on this build. Same destination as always.
        finish(plain_url, "fallback")
        return

    # One shot: whichever of the answer, the failure or the backstop arrives
    # first opens a tab, and the rest are ignored.
    state = {"done": False}

    def settle(url: str, checkout_link: str) -> None:
        if state["done"]:
            return
        state["done"] = True
        finish(url, checkout_link)

    def on_answer(answer) -> None:
        url = answer.get("url") if isinstance(answer, dict) else None
        if isinstance(url, str) and url.startswith("https://"):
            settle(url, "direct")
        else:
            # url: null with a reason (key too old, rate limited, sign-in off).
            # The plain page shows the same plans behind one login.
            settle(plain_url, "fallback")

    task.succeeded.connect(on_answer)
    task.failed.connect(lambda *_a: settle(plain_url, "fallback"))

    _set_busy(True)
    try:
        # Parented to the task, so the backstop dies with it and can never fire
        # into a dialog the user has already closed.
        from .qt_compat import safe_single_shot
        safe_single_shot(_LOGIN_LINK_WAIT_MS, task,
                         lambda: settle(plain_url, "fallback"))
    except Exception:  # noqa: BLE001 -- the socket timeout still ends the wait
        pass  # nosec B110

    try:
        QgsApplication.taskManager().addTask(task)
    except Exception:  # noqa: BLE001
        settle(plain_url, "fallback")


def _ui_locale() -> str | None:
    """The two-letter UI language, for the page the link opens. None when it
    cannot be read: the website then follows the browser."""
    try:
        from .i18n import current_locale
        code = current_locale()
        return code.split("_")[0] if code else None
    except Exception:  # noqa: BLE001
        return None
