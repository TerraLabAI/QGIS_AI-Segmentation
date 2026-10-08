



















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
        "crop_from", "crop_offset",
        "req_sent_s", "ttfb_s", "new_conn", "loop_lag_s",
        "mask_parts", "mask_vertices", "worker_start",
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

        self.crop_from: str | None = None

        self.crop_offset: float | None = None


        self.req_sent_s: float | None = None
        self.ttfb_s: float | None = None
        self.new_conn: bool | None = None
        self.loop_lag_s: float | None = None

        self.mask_parts: int | None = None
        self.mask_vertices: int | None = None


        self.worker_start: dict | None = None



    def note_crop_wait(self) -> None:



        self.waited_on_crop = True
        self.crop_ready_at = None
        self.crop_from = "fresh"

    def note_joined_warm(self) -> None:


        if self.waited_on_crop:
            self.crop_from = "warm"

    def note_crop_offset(self, point, bounds) -> None:



        try:
            minx, miny, maxx, maxy = (float(v) for v in bounds)
            half_w, half_h = (maxx - minx) / 2.0, (maxy - miny) / 2.0
            if half_w <= 0 or half_h <= 0:
                return
            dx = abs(float(point[0]) - (minx + half_w)) / half_w
            dy = abs(float(point[1]) - (miny + half_h)) / half_h
            self.crop_offset = round(min(1.0, max(dx, dy)), 2)
        except (TypeError, ValueError, IndexError):
            return

    def note_crop_ready(self, encode_s: float | None = None) -> None:





        if not self.waited_on_crop or self.crop_ready_at is not None:
            return
        self.crop_ready_at = click_clock_now()
        if encode_s is not None and encode_s >= 0:
            self.encode_s = encode_s

    def note_request(self, sent_at: float, answered_at: float,
                     sent_bytes: int, received_bytes: int,
                     wire: dict | None = None) -> None:







        self.round_trips += 1
        self.wire_s += max(0.0, answered_at - sent_at)
        self.sent_bytes += max(0, int(sent_bytes))
        self.received_bytes += max(0, int(received_bytes))
        if not wire:
            return
        uploaded_at = wire.get("uploaded_at")
        if uploaded_at is not None:
            self.req_sent_s = (self.req_sent_s or 0.0) + max(0.0, uploaded_at - sent_at)
        first_byte_at = wire.get("first_byte_at")
        if first_byte_at is not None:
            self.ttfb_s = (self.ttfb_s or 0.0) + max(0.0, first_byte_at - sent_at)
        if "new_conn" in wire:
            self.new_conn = bool(self.new_conn) or bool(wire["new_conn"])
        lag = wire.get("loop_lag_s")
        if lag is not None:
            self.loop_lag_s = max(self.loop_lag_s or 0.0, float(lag))

    def note_drawn_outline(self, parts: int, vertices: int) -> None:

        self.mask_parts = max(0, int(parts))
        self.mask_vertices = max(0, int(vertices))

    def note_worker_start(self, timings: dict | None) -> None:

        if timings:
            self.worker_start = {key: int(timings[key]) for key in (
                "worker_boot_ms", "model_load_ms") if key in timings}

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

    def _answer_ms(self) -> int:

        if self.predict_started_at is None or self.answered_at is None:
            return 0
        return _ms(self.answered_at - self.predict_started_at) or 0

    def phase_properties(self) -> dict:




        total = (None if self.drawn_at is None
                 else _ms(self.drawn_at - self.pressed_at))
        crop = self.crop_wait_ms()
        before = 0
        if self.predict_started_at is not None:
            before = _ms(self.predict_started_at
                         - (self.crop_ready_at or self.pressed_at)) or 0
        props = {
            "new_crop": bool(self.waited_on_crop),
            "crop_wait_ms": crop,
            "encode_ms": _ms(self.encode_s),
            "total_ms": total,
            "wire_ms": _ms(self.wire_s) if self.round_trips else None,
            "server_ms": (None if self.server_ms is None
                          else int(round(self.server_ms))),
            "round_trips": int(self.round_trips),
            "sent_kb": int(round(self.sent_bytes / 1024.0)),
            "received_kb": int(round(self.received_bytes / 1024.0)),
            "before_ms": before,
            "after_ms": (None if total is None
                         else max(0, total - crop - before - self._answer_ms())),
            "crop_offset": self.crop_offset,
            "crop_from": (self.crop_from or "fresh") if self.waited_on_crop else "ready",
            "req_sent_ms": _ms(self.req_sent_s),
            "ttfb_ms": _ms(self.ttfb_s),
            "new_conn": self.new_conn,
            "loop_lag_ms": _ms(self.loop_lag_s),
            "mask_parts": self.mask_parts,
            "mask_vertices": self.mask_vertices,
        }
        props.update(self.worker_start or {})
        return props

    def summary_line(self) -> str:

        props = self.phase_properties()
        total = props["total_ms"] or 0
        crop = props["crop_wait_ms"]
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
            parts.append(f"before {props['before_ms']} ms, answer {self._answer_ms()} ms{detail}")
        parts.append(f"after {props['after_ms'] or 0} ms")
        tail = (f"{self.round_trips} request(s), {int(self.sent_bytes)} B up, "
                f"{int(self.received_bytes)} B down" if self.round_trips else "no request")
        net = [f"{label} {props[key]} ms" for label, key in (
            ("sent at", "req_sent_ms"), ("first byte at", "ttfb_ms"),
            ("loop lag", "loop_lag_ms")) if props[key] is not None]
        if props["new_conn"] is not None:
            net.append("new connection" if props["new_conn"] else "reused connection")
        if net:
            tail += f" ({', '.join(net)})"
        if props["mask_parts"] is not None:
            tail += (f"; outline {props['mask_parts']} part(s), "
                     f"{props['mask_vertices']} vertices")
        if "worker_boot_ms" in props:
            tail += (f"; worker boot {props['worker_boot_ms']} ms, "
                     f"model load {props.get('model_load_ms')} ms")
        crop_at = props["crop_from"] + (
            "" if props["crop_offset"] is None else f" at {props['crop_offset']:.2f}")
        return ", ".join(parts) + f"; {tail}; crop {crop_at}"




_current: ClickPhaseClock | None = None


def activate_click_clock(clock: ClickPhaseClock | None) -> None:

    global _current
    _current = clock


def active_click_clock() -> ClickPhaseClock | None:

    return _current
