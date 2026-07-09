







from __future__ import annotations


DEFAULT_COOLDOWN_CYCLES = 1
DEFAULT_FAILURE_THRESHOLD = 3


class AdaptiveConcurrency:















    def __init__(
        self,
        start: int = 3,
        minimum: int = 1,
        maximum: int = 6,
        cooldown_cycles: int = DEFAULT_COOLDOWN_CYCLES,
    ) -> None:
        self._min = max(1, int(minimum))
        self._max = max(self._min, int(maximum))
        self._cap = min(max(int(start), self._min), self._max)
        self._cooldown_cycles = max(0, int(cooldown_cycles))
        self._cooldown = 0



        self.setbacks = 0

    @property
    def cap(self) -> int:

        return self._cap

    @property
    def maximum(self) -> int:


        return self._max

    def set_maximum(self, maximum: int) -> None:









        self._max = max(self._min, int(maximum))
        self._cap = min(max(self._cap, self._min), self._max)

    def on_clean_cycle(self) -> None:





        if self._cooldown > 0:
            self._cooldown -= 1
            return
        if self._cap < self._max:
            self._cap += 1

    def on_setback(self) -> None:


        self._cap = max(self._min, self._cap // 2)
        self._cooldown = self._cooldown_cycles
        self.setbacks += 1


class OfflineFastFail:



























    HARD_CODES = frozenset({
        "DNS_ERROR", "CONNECTION_REFUSED", "PROXY_ERROR", "NO_INTERNET",
    })

    def __init__(self, threshold: int = DEFAULT_FAILURE_THRESHOLD) -> None:
        self._threshold = max(1, int(threshold))
        self._streak = 0

    @property
    def tripped(self) -> bool:

        return self._streak >= self._threshold

    @property
    def streak(self) -> int:

        return self._streak

    def reset(self) -> None:


        self._streak = 0

    def record(self, code: str) -> bool:





        if code in self.HARD_CODES:
            self._streak += 1
        else:
            self._streak = 0
        return self.tripped
