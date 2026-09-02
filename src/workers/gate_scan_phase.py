"""Render pipelining for the empty-tile scan phase of an Automatic run.

The per-tile render is the serialized bottleneck of a large run, which is why
the detect phase drives an async prefetch. The scan phase groups neighbouring
tiles into one packed image, so it needs the same treatment: ask for every
member of a block at once, then collect them in order, instead of blocking on
each render with nothing on the wire.

Free functions on purpose. They take the two bridge callables rather than the
worker, so the scan phase keeps its own small surface and the worker module
does not grow another pair of methods.
"""
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
    """Post one render request per block member WITHOUT waiting on any.

    Returns ``{tile index: collect token}``. Empty when the renderer has no
    async API (tests, mocks) or the run is stopping, which leaves the caller on
    its blocking path.
    """
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
    """Collect and drop the renders of a block being abandoned, so the bridge
    hands their images back instead of holding them for the rest of the run."""
    if collect_render is None:
        return
    pending = list(tokens.values())
    if extra is not None:
        pending.append(extra)
    for seq in pending:
        try:
            collect_render(seq)
        except Exception:  # nosec B110 -- releasing a render never fails a run
            pass


def apply_scan_result(
    worker, block: Sequence, response: dict, group: int,
    min_score: float, min_px: int, stats: dict,
) -> None:
    """Fold one settled scan block into the gate decisions.

    Shared by the scan loop and its wind-down. The service bills a request the
    moment it accepts it, and a scan carries its whole block's charge, so a
    reply read late still has to mark its members prepaid: without that they
    submit again on the detect path and the block is paid for twice.
    """
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
    """Read whatever scan replies are still open, on the short stop budget.

    The detect loops drain their in-flight replies for the same reason: an
    accepted request is already charged, so a reply aborted unread is a block
    the user paid for whose members would then be sent again. Bounded, so one
    hung reply can never hold the wind-down open.
    """
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
                stats["unscanned"] += 1  # fail open: its tiles stay kept
        worker._free_read_replies(read_replies)
