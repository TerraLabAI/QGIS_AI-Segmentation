













from __future__ import annotations

from qgis.PyQt.QtCore import QSettings

from ....core.surface_dials import library_max_marks



_MARKS_KEY = "AISegmentation/library_run_marks"
_MAX_MARKS = 300


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
        pass  # nosec B110


def run_mark(run_id: str) -> str:

    if not run_id:
        return ""
    return _read_marks().get(str(run_id), "")


def set_run_mark(run_id: str, kind: str) -> None:





    if not run_id or kind not in _RANK:
        return
    marks = _read_marks()
    current = marks.get(str(run_id), "")
    if _RANK.get(current, 0) >= _RANK[kind]:
        return


    marks.pop(str(run_id), None)
    marks[str(run_id)] = kind
    _write_marks(marks)


def forget_run_mark(run_id: str) -> None:

    if not run_id:
        return
    marks = _read_marks()
    if marks.pop(str(run_id), None) is not None:
        _write_marks(marks)
