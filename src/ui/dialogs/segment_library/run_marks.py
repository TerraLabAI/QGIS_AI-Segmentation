"""What this computer has already done with a past run.

The account keeps no such flag: the history rows say when a run happened and
what it cost, never whether the user brought it back or wrote it to a file. So
the library records it here, in the user's own QGIS profile, and the run cards
read it to print "Restored" or "Exported".

Deliberately local. Two machines on one account keep their own answer, which
is the honest one: what matters to the user is whether THIS project already
has the run in it.

Small and bounded: the newest ``_MAX_MARKS`` runs, oldest dropped first, so a
long history never grows the settings file without limit.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QSettings

from ...core.surface_dials import library_max_marks

# QSettings key, persisted in the user's profile. Renaming it loses every mark
# on every existing install, so it stays as written.
_MARKS_KEY = "AISegmentation/library_run_marks"
_MAX_MARKS = 300
# Exporting a run says more than restoring it (the shapes left QGIS), so it
# wins when a run has been through both.
_RANK = {"restored": 1, "exported": 2}


def _read_marks() -> dict:
    try:
        raw = QSettings().value(_MARKS_KEY, "") or ""
    except (RuntimeError, TypeError):
        return {}
    marks: dict[str, str] = {}
    for chunk in str(raw).split("|"):
        run_id, _, kind = chunk.partition("=")
        if run_id and kind in _RANK:
            marks[run_id] = kind
    return marks


def _write_marks(marks: dict) -> None:
    items = list(marks.items())[-library_max_marks(_MAX_MARKS):]
    try:
        QSettings().setValue(
            _MARKS_KEY, "|".join(f"{k}={v}" for k, v in items))
    except (RuntimeError, TypeError):
        pass  # nosec B110 - a mark is a nicety, never worth an error


def run_mark(run_id: str) -> str:
    """"restored", "exported" or "" for one run."""
    if not run_id:
        return ""
    return _read_marks().get(str(run_id), "")


def set_run_mark(run_id: str, kind: str) -> None:
    """Record that this computer restored or exported a run.

    A weaker mark never overwrites a stronger one: a run exported last week and
    reopened today still reads as exported.
    """
    if not run_id or kind not in _RANK:
        return
    marks = _read_marks()
    current = marks.get(str(run_id), "")
    if _RANK.get(current, 0) >= _RANK[kind]:
        return
    # Re-inserted at the end so the trim drops the runs nobody has touched in
    # a long time rather than the one just used.
    marks.pop(str(run_id), None)
    marks[str(run_id)] = kind
    _write_marks(marks)


def forget_run_mark(run_id: str) -> None:
    """Drop a run's mark, for a run the user deleted."""
    if not run_id:
        return
    marks = _read_marks()
    if marks.pop(str(run_id), None) is not None:
        _write_marks(marks)
