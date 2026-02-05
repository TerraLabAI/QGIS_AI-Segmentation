







from __future__ import annotations

import os

_MARKER_BASENAME = "degraded.txt"


def _marker_path(venv_dir: str) -> str:
    return os.path.join(venv_dir, _MARKER_BASENAME)


def record_degraded(venv_dir: str, package_names: list[str]) -> None:

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
        pass  # nosec B110


def clear_degraded(venv_dir: str) -> None:

    try:
        os.unlink(_marker_path(venv_dir))
    except OSError:
        pass  # nosec B110


def degraded_packages(venv_dir: str) -> list[str]:

    try:
        with open(_marker_path(venv_dir), encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    except OSError:
        return []
