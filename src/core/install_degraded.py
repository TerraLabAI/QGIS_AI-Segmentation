"""Record an install that finished without everything it set out to install.

An install is allowed to end short: a blocked download or a volume with no
room leaves the on-device packages out, and the cloud mode still works. That
outcome used to leave no trace beside the dependency hash, so a later start
read the environment as a full one. This writes the short list next to the
hash, and clears it the moment an install lands complete.
"""
from __future__ import annotations

import os

_MARKER_BASENAME = "degraded.txt"


def _marker_path(venv_dir: str) -> str:
    return os.path.join(venv_dir, _MARKER_BASENAME)


def record_degraded(venv_dir: str, package_names: list[str]) -> None:
    """Write the packages this install could not deliver. Never raises."""
    path = _marker_path(venv_dir)
    if not package_names:
        clear_degraded(venv_dir)
        return
    try:
        os.makedirs(venv_dir, exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(sorted(package_names)))
        os.replace(tmp_path, path)
    except OSError:
        pass  # nosec B110 - a missing marker only costs a log line


def clear_degraded(venv_dir: str) -> None:
    """Drop the marker. Called when an install delivers everything."""
    try:
        os.unlink(_marker_path(venv_dir))
    except OSError:
        pass  # nosec B110 - already gone is the state we want


def degraded_packages(venv_dir: str) -> list[str]:
    """The packages the last install could not deliver, oldest record wins."""
    try:
        with open(_marker_path(venv_dir), encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    except OSError:
        return []
