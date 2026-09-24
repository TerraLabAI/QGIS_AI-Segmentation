



















from __future__ import annotations

import time


def click_clock_now() -> float:



    return time.perf_counter()


def _ms(seconds: float | None) -> int | None:
    if seconds is None:
        return None
    return max(0, int(round(seconds * 1000.0)))


class ClickPhaseClock:


    __slots__ = (
        "pressed_at", "waited_on_crop", "crop_ready_at", "encode_s",
        "predict_started_at", "answered_at", "drawn_at",
        "round_trips", "wire_s", "server_ms", "sent_bytes", "received_bytes",
    )

    def __init__(self, pressed_at: float | None = None) -> None:
        self.pressed_at = click_clock_now() if pressed_at is None else pressed_at
        self.waited_on_crop = False
        self.crop_ready_at: float | None = None
        self.encode_s: float | None = None
        self.predict_started_at: float | None = None
        self.answered_at: float | None = None
        self.drawn_at: float | None = None
        self.round_trips = 0
        self.wire_s = 0.0
        self.server_ms: float | None = None
        self.sent_bytes = 0
        self.received_bytes = 0



    def note_crop_wait(self) -> None:


        self.waited_on_crop = True
        self.crop_ready_at = None

    def note_crop_ready(self, encode_s: float | None = None) -> None:





        if not self.waited_on_crop or self.crop_ready_at is not None:
            return
        self.crop_ready_at = click_clock_now()
        if encode_s is not None and encode_s >= 0:
            self.encode_s = encode_s

    def note_request(self, sent_at: float, answered_at: float,
                     sent_bytes: int, received_bytes: int) -> None:


        self.round_trips += 1
        self.wire_s += max(0.0, answered_at - sent_at)
        self.sent_bytes += max(0, int(sent_bytes))
        self.received_bytes += max(0, int(received_bytes))

    def note_server_ms(self, value) -> None:

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        if value != value or value < 0:
            return
        self.server_ms = (self.server_ms or 0.0) + float(value)



    def crop_wait_ms(self) -> int:

        if not self.waited_on_crop or self.crop_ready_at is None:
            return 0
        return _ms(self.crop_ready_at - self.pressed_at) or 0

    def phase_properties(self) -> dict:

        total = (None if self.drawn_at is None
                 else _ms(self.drawn_at - self.pressed_at))
        return {
            "new_crop": bool(self.waited_on_crop),
            "crop_wait_ms": self.crop_wait_ms(),
            "encode_ms": _ms(self.encode_s),
            "total_ms": total,
            "wire_ms": _ms(self.wire_s) if self.round_trips else None,
            "server_ms": (None if self.server_ms is None
                          else int(round(self.server_ms))),
            "round_trips": int(self.round_trips),
            "sent_kb": int(round(self.sent_bytes / 1024.0)),
        }

    def summary_line(self) -> str:

        props = self.phase_properties()
        total = props["total_ms"] or 0
        crop = props["crop_wait_ms"]
        started = self.crop_ready_at or self.pressed_at
        before = answer = 0
        if self.predict_started_at is not None:
            before = _ms(self.predict_started_at - started) or 0
            if self.answered_at is not None:
                answer = _ms(self.answered_at - self.predict_started_at) or 0
        parts = [f"Click timing: {total} ms"]
        if crop:
            encode = props["encode_ms"]
            parts.append(f"crop {crop} ms" + (f" (encode {encode} ms)" if encode is not None else ""))
        if self.predict_started_at is not None:
            detail = ""
            if self.round_trips:
                server = props["server_ms"]
                detail = f" (wire {props['wire_ms']} ms" + (
                    f", service {server} ms)" if server is not None else ")")
            parts.append(f"before {before} ms, answer {answer} ms{detail}")
        parts.append(f"after {max(0, total - crop - before - answer)} ms")
        tail = (f"{self.round_trips} request(s), {int(self.sent_bytes)} B up, "
                f"{int(self.received_bytes)} B down" if self.round_trips else "no request")
        return ", ".join(parts) + f"; {tail}"




_current: ClickPhaseClock | None = None


def activate_click_clock(clock: ClickPhaseClock | None) -> None:

    global _current
    _current = clock


def active_click_clock() -> ClickPhaseClock | None:

    return _current
