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

# Status families that carry a technical tail. The reason changes, the sentence
# the user needs does not, and the tail goes to the tooltip.
_STATUS_PREFIXES = (
    "Virtual environment incomplete:",
    "Package verification failed",
)

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


def _exact_status_sentences() -> dict[str, str]:
    """Each exact status the environment check returns, with its sentence.

    Matched on the raw English, which is what the core module produces. It is
    never something the user typed and never changes with the locale.
    """
    return {
        "Old installation detected. Migration required.":
            tr("An old version is installed. Click Install to replace it."),
        "Previous installation was interrupted":
            tr("The last install did not finish. Click Install to start again."),
        "Dependencies not installed":
            tr("The AI components are not installed. Click Install to add them."),
        "Virtual environment not configured":
            tr("The AI workspace is missing. Click Install to build it."),
        "Python runtime is damaged. Reinstall required.":
            tr("The AI runtime is damaged. Click Install to repair it."),
        "Dependencies need updating":
            tr("The AI components need an update. The update starts now."),
        "Package verification failed (torch import error)":
            tr("The AI components did not load. Click Install to repair them."),
        "Local model packages are not installed":
            tr("The on-device AI is not installed. Click Install to add it."),
    }


def _prefix_status_sentence(prefix: str) -> str:
    """The sentence for a status family, whatever technical tail it carries."""
    if prefix == "Virtual environment incomplete:":
        return tr("Some AI components are missing. Click Install to complete them.")
    return tr("The AI components did not pass the check. Click Install to repair them.")


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
    text = message.strip()
    exact = _exact_status_sentences().get(text)
    if exact is not None:
        return exact
    for prefix in _STATUS_PREFIXES:
        if text.startswith(prefix):
            return _prefix_status_sentence(prefix)
    for marker in _RAW_STATUS_MARKERS:
        if marker in text:
            return unknown_setup_status_sentence()
    return None
