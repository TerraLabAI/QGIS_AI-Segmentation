










from __future__ import annotations

import os
import time


REPLACE_ATTEMPTS = 5


REPLACE_DELAY_S = 0.2


def replace_file_with_retry(source: str, target: str,
                            attempts: int = REPLACE_ATTEMPTS,
                            delay_s: float = REPLACE_DELAY_S) -> None:







    total = max(1, int(attempts))
    for attempt in range(1, total + 1):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt >= total or os.name != "nt":
                raise
            time.sleep(delay_s)
