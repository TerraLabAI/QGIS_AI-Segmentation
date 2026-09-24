




















from __future__ import annotations

import base64
import binascii
import gzip
import json
import math
import time

from qgis.core import QgsApplication, QgsTask

from .qt_compat import silent_task_flags

WIRE_VERSION = 1
FINALIZE_PATH = "/v1/finalize"





DEV_URL_KEY = "TERRALAB_FINALIZE_URL"



_TIMEOUT_BASE_S = 30.0
_TIMEOUT_PER_1K_FRAGMENTS_S = 3.0
_TIMEOUT_MAX_S = 300.0



FAIL_TIMEOUT = "TIMEOUT"
FAIL_NETWORK = "NETWORK"
FAIL_HTTP_4XX = "HTTP_4XX"
FAIL_HTTP_5XX = "HTTP_5XX"
FAIL_MALFORMED = "MALFORMED"
FAIL_EMPTY = "EMPTY"
FAIL_CLIENT = "CLIENT"



_inflight: set = set()


def _dev_url() -> str:
    try:
        from .env_local import env_local_value

        return env_local_value(DEV_URL_KEY, "").strip()
    except Exception:  # noqa: BLE001
        return ""


def server_finalize_enabled() -> bool:






    if _dev_url():
        return True
    try:
        from .server_dials import dial_bool

        return dial_bool("features.server_finalize", False)
    except Exception:  # noqa: BLE001
        return False


def finalize_url() -> str:





    base = _dev_url()
    if not base:
        try:
            from .server_dials import dial_url

            base = dial_url("finalize.url", "")
        except Exception:  # noqa: BLE001
            base = ""
    base = (base or "").strip().rstrip("/")
    if not base:
        return ""
    return base if base.endswith(FINALIZE_PATH) else base + FINALIZE_PATH


def server_finalize_active() -> bool:

    return server_finalize_enabled() and bool(finalize_url())


def finalize_timeout_ms(fragments: int) -> int:

    try:
        from .server_dials import dial_in_range

        base_s = dial_in_range("finalize.timeout_base_s", _TIMEOUT_BASE_S, 1.0, 900.0)
        per_k = dial_in_range(
            "finalize.timeout_per_1k_fragments_s", _TIMEOUT_PER_1K_FRAGMENTS_S, 0.0, 300.0)
        cap_s = dial_in_range("finalize.timeout_max_s", _TIMEOUT_MAX_S, 1.0, 1800.0)
    except Exception:  # noqa: BLE001
        base_s, per_k, cap_s = _TIMEOUT_BASE_S, _TIMEOUT_PER_1K_FRAGMENTS_S, _TIMEOUT_MAX_S
    seconds = min(cap_s, base_s + per_k * max(0, int(fragments)) / 1000.0)
    return int(max(1.0, seconds) * 1000)


def _json_safe(value):



    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def encode_request(context: dict, fragments: list) -> bytes:








    frag_out = []
    for tile_index, wkb, score in fragments:
        frag_out.append({
            "t": int(tile_index),
            "g": base64.b64encode(bytes(wkb)).decode("ascii"),
            "s": float(score),
        })
    body = {
        "version": WIRE_VERSION,
        "run_id": str(context.get("run_id") or ""),
        "crs_authid": str(context.get("crs_authid") or ""),
        "zone_wkt": str(context.get("zone_wkt") or ""),
        "prompt": str(context.get("prompt") or ""),
        "merge": _json_safe(context.get("merge") or None),
        "align": _json_safe(context.get("align") or None),
        "fragments": frag_out,
    }
    raw = json.dumps(body, allow_nan=False, separators=(",", ":")).encode("utf-8")
    return gzip.compress(raw, 6, mtime=0)


def decode_answer(answer, run_id: str) -> tuple[list | None, str, dict]:







    if not isinstance(answer, dict):
        return None, FAIL_MALFORMED, {}
    version = answer.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version != WIRE_VERSION:
        return None, FAIL_MALFORMED, {}
    answered_run = answer.get("run_id")
    if answered_run is not None and str(answered_run) != str(run_id):
        return None, FAIL_MALFORMED, {}
    objects = answer.get("objects")
    if not isinstance(objects, list):
        return None, FAIL_MALFORMED, {}
    if not objects:
        return None, FAIL_EMPTY, {}
    rows = []
    seen = set()
    for item in objects:
        if not isinstance(item, dict):
            return None, FAIL_MALFORMED, {}
        fid = item.get("fid")
        score = item.get("s")
        text = item.get("g")
        if not isinstance(fid, int) or isinstance(fid, bool) or fid in seen:
            return None, FAIL_MALFORMED, {}
        if (not isinstance(score, (int, float)) or isinstance(score, bool)
                or not math.isfinite(score)):
            return None, FAIL_MALFORMED, {}
        if not isinstance(text, str) or not text:
            return None, FAIL_MALFORMED, {}
        try:
            wkb = base64.b64decode(text, validate=True)
        except (binascii.Error, ValueError):
            return None, FAIL_MALFORMED, {}
        if not wkb:
            return None, FAIL_MALFORMED, {}
        seen.add(fid)
        rows.append((fid, wkb, float(score)))
    stats = answer.get("stats")
    return rows, "", stats if isinstance(stats, dict) else {}


def _failure_from(answer, http_status) -> str:

    code = ""
    if isinstance(answer, dict):
        code = str(answer.get("code") or answer.get("error") or "").upper()
    if http_status is None:
        return FAIL_TIMEOUT if "TIMEOUT" in code else FAIL_NETWORK
    if int(http_status) >= 500:
        return FAIL_HTTP_5XX
    if int(http_status) >= 400:
        return FAIL_HTTP_4XX
    return FAIL_MALFORMED


class ServerFinalizeTask(QgsTask):







    def __init__(self, url: str, context: dict, fragments: list, auth: dict,
                 timeout_ms: int) -> None:
        super().__init__("AI Segmentation finalize", silent_task_flags())
        self._url = url
        self._context = dict(context)
        self._fragments = fragments
        self._auth = dict(auth or {})
        self._timeout_ms = int(timeout_ms)
        self.fragment_count = len(fragments)
        self.done = False
        self.rows: list | None = None
        self.failure = ""
        self.stats: dict = {}
        self.http_status = None
        self.request_bytes = 0
        self.elapsed_ms = 0

    def run(self) -> bool:  # noqa: D102
        started = time.monotonic()
        try:
            self._call()
        except Exception:  # noqa: BLE001
            self.rows = None
            self.failure = self.failure or FAIL_CLIENT
        finally:
            self.elapsed_ms = int((time.monotonic() - started) * 1000)
            self.done = True
        return True

    def _call(self) -> None:
        try:
            body = encode_request(self._context, self._fragments)
        finally:
            self._fragments = []
        self.request_bytes = len(body)
        from ..api.terralab_client import TerraLabClient

        client = TerraLabClient()


        answer, http_status, _was_json = client._request_once(
            "POST", self._url, self._auth, body, True, self._timeout_ms,
            False, True, True)
        self.http_status = http_status
        if http_status != 200 or not isinstance(answer, dict) or "error" in answer:
            self.failure = _failure_from(answer, http_status)
            return
        rows, failure, stats = decode_answer(answer, self._context.get("run_id") or "")
        self.rows = rows
        self.failure = failure
        self.stats = stats

    def finished(self, result: bool) -> None:  # noqa: D102
        _inflight.discard(self)


def start_finalize_task(url: str, context: dict, fragments: list, auth: dict,
                        timeout_ms: int) -> ServerFinalizeTask:

    task = ServerFinalizeTask(url, context, fragments, auth, timeout_ms)
    _inflight.add(task)
    QgsApplication.taskManager().addTask(task)
    return task
