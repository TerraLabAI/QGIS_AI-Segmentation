











from __future__ import annotations

import time
from collections.abc import Callable, Sequence


def request_scan_renders(
    request_render: Callable | None,
    collect_render: Callable | None,
    tiles: Sequence,
    block: Sequence,
    stopped: bool = False,
) -> dict:






    if request_render is None or collect_render is None or stopped:
        return {}
    tokens: dict = {}
    for idx, _qr, _qc in block:
        tx, ty, tw, th = tiles[idx]
        seq = request_render(tx, ty, tw, th)
        if seq is None:
            break
        tokens[idx] = seq
    return tokens


def release_scan_renders(
    collect_render: Callable | None, tokens: dict, extra=None
) -> None:


    if collect_render is None:
        return
    pending = list(tokens.values())
    if extra is not None:
        pending.append(extra)
    for seq in pending:
        try:
            collect_render(seq)
        except Exception:  # nosec B110
            pass


def apply_scan_result(
    worker, block: Sequence, response: dict, group: int,
    min_score: float, min_px: int, stats: dict,
) -> None:







    from ..core import scan_gate

    skip, _keep = scan_gate.classify_block(
        block, response, group, min_score, min_px)
    worker._gate_skip |= skip
    worker._gate_prepaid |= {idx for idx, _qr, _qc in block}
    stats["scans"] += 1


def drain_scan_replies(
    worker, in_flight: dict, wait_flags, group: int,
    min_score: float, min_px: int, stats: dict,
) -> None:







    from qgis.PyQt.QtCore import QCoreApplication

    deadline = time.monotonic() + worker._stop_drain_budget_s
    while in_flight and time.monotonic() < deadline:
        QCoreApplication.processEvents(wait_flags, 100)
        read_replies: list = []
        for reply in [r for r in in_flight if worker._reply_is_finished(r)]:
            block_i, block, _submission = in_flight.pop(reply)
            read_replies.append(reply)
            outcome = worker._classify_submit_response(
                -(block_i + 1), worker._read_reply(-(block_i + 1), reply), {})
            if outcome[0] == "completed_inline":
                apply_scan_result(
                    worker, block, outcome[1], group, min_score, min_px, stats)
            else:
                stats["unscanned"] += 1
        worker._free_read_replies(read_replies)
