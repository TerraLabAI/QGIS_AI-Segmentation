

















from __future__ import annotations

from ...core.i18n import tr




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





    return tr("The AI components are not ready. Click Install to set them up.")


def setup_status_sentence(message: str) -> str | None:





    if not message:
        return None
    code = setup_status_code(message)
    if code:
        return _exact_status_sentences().get(code)
    for marker in _RAW_STATUS_MARKERS:
        if marker in message:
            return unknown_setup_status_sentence()
    return None
