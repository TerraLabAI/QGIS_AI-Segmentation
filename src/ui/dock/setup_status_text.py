"""Translate the raw setup statuses the startup check reports.

``get_venv_status()`` answers with a short English sentence written for a log
line, and the dock used to print it straight into the setup label, so a French
or Portuguese user read English there and nowhere else. This module turns each
known status into a ``tr()`` sentence that says what to do next. The raw text
stays in the message log and in the label tooltip.

Only a status this module recognises gets replaced. Every other caller of
``set_dependency_status`` already passes a ``tr()`` string, and swapping one of
those for a sentence of our own would throw away eleven translations. A status
the check may grow later is caught by the markers below, which use words no
translated status carries.

The ``tr()`` calls sit inside the functions, not at module level: the locale is
not settled at import time, and the i18n guard only collects a ``tr()`` call
whose first argument is a literal.
"""
from __future__ import annotations

from ...core.i18n import tr

# Words that only the untranslated statuses of the environment check use. They
# catch a status added to the check after this table was written, so it reaches
# the user as a sentence in their language instead of raw English.
_RAW_STATUS_MARKERS = (
    "Virtual environment",
    "Python runtime",
    "Package verification",
    "Old installation",
    "Previous installation",
    "Local model packages",
    "Dependencies not",
    "Dependencies need",
)


# One code per status the environment check can report. Everything outside
# this module branches on a code, never on the wording: the sentences travel
# through a plain string signal, so they cannot carry a code of their own, and
# two places used to read a decision off the English. The table below is the
# single place a reworded status has to be updated.
STATUS_OLD_INSTALL = "old_install"
STATUS_INTERRUPTED = "interrupted"
STATUS_NOT_INSTALLED = "not_installed"
STATUS_NO_VENV = "no_venv"
STATUS_RUNTIME_DAMAGED = "runtime_damaged"
STATUS_NEEDS_UPDATE = "needs_update"
STATUS_VERIFY_FAILED = "verify_failed"
STATUS_NO_LOCAL_MODEL = "no_local_model"
STATUS_INCOMPLETE = "incomplete"
STATUS_CHECK_FAILED = "check_failed"

_EXACT_STATUS_CODES = {
    "Old installation detected. Migration required.": STATUS_OLD_INSTALL,
    "Previous installation was interrupted": STATUS_INTERRUPTED,
    "Dependencies not installed": STATUS_NOT_INSTALLED,
    "Virtual environment not configured": STATUS_NO_VENV,
    "Python runtime is damaged. Reinstall required.": STATUS_RUNTIME_DAMAGED,
    "Dependencies need updating": STATUS_NEEDS_UPDATE,
    "Package verification failed (torch import error)": STATUS_VERIFY_FAILED,
    "Local model packages are not installed": STATUS_NO_LOCAL_MODEL,
}

_PREFIX_STATUS_CODES = {
    "Virtual environment incomplete:": STATUS_INCOMPLETE,
    "Package verification failed": STATUS_CHECK_FAILED,
}


def setup_status_code(message: str) -> str:
    """The stable code for a raw setup status, or "" when it has none."""
    if not message:
        return ""
    text = message.strip()
    exact = _EXACT_STATUS_CODES.get(text)
    if exact is not None:
        return exact
    for prefix, code in _PREFIX_STATUS_CODES.items():
        if text.startswith(prefix):
            return code
    return ""


def _exact_status_sentences() -> dict[str, str]:
    """Each status code the environment check produces, with its sentence.

    The keys are codes, so a reworded status is a one-line change in the table
    above rather than a silent loss of the sentence here.
    """
    return {
        STATUS_OLD_INSTALL:
            tr("An old version is installed. Click Install to replace it."),
        STATUS_INTERRUPTED:
            tr("The last install did not finish. Click Install to start again."),
        STATUS_NOT_INSTALLED:
            tr("The AI components are not installed. Click Install to add them."),
        STATUS_NO_VENV:
            tr("The AI workspace is missing. Click Install to build it."),
        STATUS_RUNTIME_DAMAGED:
            tr("The AI runtime is damaged. Click Install to repair it."),
        STATUS_NEEDS_UPDATE:
            tr("The AI components need an update. The update starts now."),
        STATUS_VERIFY_FAILED:
            tr("The AI components did not load. Click Install to repair them."),
        STATUS_NO_LOCAL_MODEL:
            tr("The on-device AI is not installed. Click Install to add it."),
        STATUS_INCOMPLETE:
            tr("Some AI components are missing. Click Install to complete them."),
        STATUS_CHECK_FAILED:
            tr("The AI components did not pass the check. "
               "Click Install to repair them."),
    }


def unknown_setup_status_sentence() -> str:
    """The sentence for a setup status with no mapping of its own.

    It names the state and the next click, which is what every mapped sentence
    really carries.
    """
    return tr("The AI components are not ready. Click Install to set them up.")


def setup_status_sentence(message: str) -> str | None:
    """The translated sentence for a raw setup status, or None to leave it.

    None means the text did not come from the environment check, so the caller
    shows it unchanged: it is already a translated string from its own caller.
    """
    if not message:
        return None
    code = setup_status_code(message)
    if code:
        return _exact_status_sentences().get(code)
    for marker in _RAW_STATUS_MARKERS:
        if marker in message:
            return unknown_setup_status_sentence()
    return None
