






from __future__ import annotations

import logging
import threading

from .tile_convert_child import _recv, _send

logger = logging.getLogger(__name__)


def _init_all(procs: list, request, timeout: float) -> list:



    answers: list = [(False, "no answer")] * len(procs)

    def exchange(index: int, proc) -> None:
        answers[index] = _await_ready(proc, timeout, request=request)

    threads = []
    for index, proc in enumerate(procs):
        thread = threading.Thread(target=exchange, args=(index, proc),
                                  daemon=True, name="tileconvinit")
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join(timeout + 2.0)
    return list(answers)


def _await_ready(proc, timeout: float, request=None) -> tuple[bool, str]:










    answer: list = []



    broke: list = []

    def read() -> None:
        step = "read"
        try:
            if request is not None:
                step = "send"
                _send(proc.stdin, request)
                step = "read"
            answer.append(_recv(proc.stdout))
        except Exception as exc:  # noqa: BLE001
            broke.append(f"{step} failed: {type(exc).__name__}")
            answer.append(None)

    thread = threading.Thread(target=read, daemon=True, name="tileconv-handshake")
    thread.start()
    thread.join(timeout)
    if not answer:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        thread.join(timeout=1.0)
        if not thread.is_alive():
            for stream in (proc.stdin, proc.stdout):
                try:
                    stream.close()
                except Exception:  # noqa: BLE001  # nosec B110
                    pass
        return False, f"no answer within {timeout:.0f}s"
    frame = answer[0]
    if not frame:
        return False, _silent_end(proc, broke)
    if frame[0] == "no":
        logger.info("TileConvertProcessPool: child could not start (%s)",
                    frame[1])
        return False, str(frame[1])[:300]
    expected = "init_ok" if request is not None else "ready"
    return frame[0] == expected, ""


def _silent_end(proc, broke: list) -> str:





    try:
        code = proc.wait(timeout=1.0)
    except Exception:  # noqa: BLE001
        code = None
    status = "still running" if code is None else f"exit {code}"
    if broke:
        return f"{broke[0]}, child {status}"
    return f"died before answering ({status})"
