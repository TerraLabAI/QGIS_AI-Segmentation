
















from __future__ import annotations

import bisect
import json
import re
import threading
import time

from .log_scrub import anonymize_paths, scrub_sensitive

LOG_LINES = 300

MAX_BODY_BYTES = 60 * 1024
MAX_LINE_CHARS = 2000
MIN_INTERVAL_S = 600
ROUTE = "/api/plugin/problem-report"
_LAST_SENT_KEY = "AI_Segmentation/problem_report_last_sent_at"
_TIMEOUT_MS = 10_000


SENT = "sent"
THROTTLED = "throttled"
OFF = "off"





_EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


_ANY_URL_RE = re.compile(r"(?i)(?<![a-z0-9+.-])(?P<pre>[0-9+.-]*)[a-z][a-z0-9+.-]*://[^\s'\"<>]*")
_JWT_RE = re.compile(r"\beyJ[\w-]{6,}\.[\w-]{6,}\.[\w-]{6,}")

_PREFIXED_SECRET_RE = re.compile(
    r"\b(?:sk|pk|rk|hf|ghp|gho|ghs|ghu|glpat|xox[abprs])[-_][A-Za-z0-9_-]{12,}"
    r"|\bAKIA[0-9A-Z]{16}\b"
)

_PREFIXED_SECRET_HINT_RE = re.compile(r"(?:sk|pk|rk|hf|gh[opsu]|glpat|xox[abprs])[-_]|AKIA")

_ASSIGNED_SECRET_RE = re.compile(
    r"(?i)\b(?:token|secret|signature|sig|key|apikey|passwd|pwd)\b\s*[:=]\s*[\"']?[^\s\"'&,;]+"
)




_TOKEN_RUN_RE = re.compile(r"(?<![A-Za-z0-9_\-])[A-Za-z0-9_\-]{32,}")
_DIGIT_RE = re.compile(r"[0-9]")








_PATH_ROOT_RE = re.compile(
    r"(?P<home><USER>)"
    r"|~(?<![\w.~]~)[\w.-]*(?=[\\/])"
    r"|%[A-Za-z_][\w()]*%(?=[\\/])"
    r"|\$\{?[A-Za-z_]\w*\}?(?=[\\/])"
    r"|(?P<drive>:)(?<=[A-Za-z]:)(?<![^\W\d_][A-Za-z]:)(?=\\|/(?!/))"
    r"|\\\\[?.]\\(?:UNC\\)?"
    r"|\\\\"
    r"|/(?<![^\W_]/)(?<![:/]/)"
    r"|/(?<=:/)(?!/)"
)






_FILE_EXTENSIONS = (

    "tif", "tiff", "geotiff", "gpkg", "shp", "shx", "dbf", "prj", "cpg", "qpj", "sbn",
    "sbx", "qgz", "qgs", "qml", "qlr", "sld", "vrt", "ecw", "jp2", "j2k", "sid", "mbtiles",
    "pmtiles", "kml", "kmz", "gml", "gpx", "geojson", "geojsonl", "topojson", "fgb", "gdb",
    "mdb", "dxf", "dwg", "dgn", "las", "laz", "e57", "ply", "obj", "xyz", "asc", "grd",
    "nc", "hdf", "h5", "hdf5", "img", "dem", "bil", "bip", "bsq", "ovr", "aux", "msk",
    "wld", "tfw", "jgw", "pgw", "osm", "pbf", "parquet", "geoparquet", "copc",

    "jpg", "jpeg", "png", "gif", "bmp", "webp", "svg", "ico", "heic",

    "pdf", "doc", "docx", "xls", "xlsx", "xlsm", "ods", "odt", "odp", "ppt", "pptx",
    "rtf", "txt", "md",

    "zip", "7z", "rar", "tar", "gz", "tgz", "bz2", "xz", "zst",

    "py", "pyc", "pyd", "pyw", "js", "ts", "html", "htm", "css", "sh", "bat", "ps1",
    "cmd", "exe", "dll", "so", "dylib", "whl", "ipynb", "sql",

    "csv", "tsv", "json", "jsonl", "ndjson", "log", "db", "sqlite", "sqlite3", "xml",
    "yaml", "yml", "toml", "ini", "cfg", "conf", "dat", "bin", "npy", "npz", "pkl",
    "pickle", "part", "tmp", "bak", "old", "lock",
)
_FILE_EXTENSION_RE = re.compile(
    r"(?i)\.(?:" + "|".join(_FILE_EXTENSIONS) + r")(?![\w\\/])"
)



_APOSTROPHE_CLOSE_RE = re.compile(r"'(?![^\W\d_])")
_LINE_BREAK_RE = re.compile(r"(\r\n|\r|\n)")
_SOURCE_FILE_RE = re.compile(r"(?i)^[\w.-]+\.py$")

_VSI_URL_RE = re.compile(r"(?i)/vsi[a-z0-9_]+/+[a-z][a-z0-9+.-]*://[^\s'\"<>]*")


def _account_name() -> str:
    try:
        import getpass

        name = getpass.getuser() or ""
    except Exception:  # noqa: BLE001
        return ""
    return name if len(name) >= 3 else ""


def _path_replacement(last: str) -> str:

    return f"<path>/{last}" if _SOURCE_FILE_RE.match(last) else "<path>"


def _scrub_paths_in_line(text: str) -> str:
    out: list[str] = []
    pos = 0
    closers = None
    last_ext = -1
    for found in _FILE_EXTENSION_RE.finditer(text):
        last_ext = found.end()
    while True:
        root = _PATH_ROOT_RE.search(text, pos)
        if root is None:
            break
        start = root.start() - 1 if root.group("drive") else root.start()
        if start < pos:

            start = root.start()
        quote = text[start - 1] if start > 0 and text[start - 1] in "'\"" else ""
        close = -1
        if quote == '"':
            close = text.find('"', root.end())
        elif quote:
            if closers is None:
                closers = [m.start() for m in _APOSTROPHE_CLOSE_RE.finditer(text)]
            at = bisect.bisect_left(closers, root.end())
            close = closers[at] if at < len(closers) else -1
        out.append(text[pos:start])
        if close >= 0:
            inside = text[start:close]
            cut = max(inside.rfind("/"), inside.rfind("\\")) + 1
            out.append(_path_replacement(inside[cut:]))
            pos = close
            continue
        after = root.end()
        if root.group("home") and after < len(text) and text[after] not in " \t\\/":


            out.append("<path>")
            pos = after
            continue
        end = last_ext if last_ext > after else len(text)
        span = text[start:end]
        cut = max(span.rfind("/"), span.rfind("\\")) + 1
        out.append(_path_replacement(span[cut:]) if end == last_ext else "<path>")
        pos = end
    out.append(text[pos:])
    return "".join(out)


def _scrub_paths(text: str) -> str:
    if "\n" not in text and "\r" not in text:
        return _scrub_paths_in_line(text)
    parts = _LINE_BREAK_RE.split(text)
    return "".join(part if i % 2 else _scrub_paths_in_line(part) for i, part in enumerate(parts))


def _replace_url(match: re.Match) -> str:
    return f"{match.group('pre')}<url>"


def _replace_opaque(match: re.Match) -> str:
    run = match.group(0)

    if run != run.lower() and run != run.upper() and _DIGIT_RE.search(run):
        return "<token>"
    return run


def scrub_report_line(line: str, account: str | None = None) -> str:

    if not line:
        return ""
    text = str(line)[:MAX_LINE_CHARS]
    text = anonymize_paths(text)
    if "@" in text:
        text = _EMAIL_RE.sub("<email>", text)
    if "://" in text:
        if "vsi" in text.lower():
            text = _VSI_URL_RE.sub("<USER>", text)
        text = _ANY_URL_RE.sub(_replace_url, text)
    if "eyJ" in text:
        text = _JWT_RE.sub("<token>", text)
    if _PREFIXED_SECRET_HINT_RE.search(text):
        text = _PREFIXED_SECRET_RE.sub("<token>", text)
    text = scrub_sensitive(text)
    if ":" in text or "=" in text:
        text = _ASSIGNED_SECRET_RE.sub("<auth>", text)
    text = _TOKEN_RUN_RE.sub(_replace_opaque, text)
    if "/" in text or "\\" in text or "<USER>" in text:
        text = _scrub_paths(text)
    text = text.replace("<USER>", "<path>")
    name = _account_name() if account is None else account
    if name and name.lower() in text.lower():

        text = re.sub(r"(?i)(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])",
                      "<user>", text)
    return text


def scrub_report_lines(lines: list[str], account: str | None = None) -> list[str]:
    name = _account_name() if account is None else account
    return [scrub_report_line(line, account=name) for line in lines]


def build_payload(lines: list[str], context: dict) -> dict:

    kept = list(lines[-LOG_LINES:])
    total = len(kept)

    def body(kept_lines: list[str]) -> dict:
        return {
            "product_id": "ai-segmentation",
            "context": context,
            "log_lines": kept_lines,
            "line_count": total,
            "truncated": len(kept_lines) < total,
        }

    while kept and _encoded_size(body(kept)) > MAX_BODY_BYTES:

        kept = kept[max(1, len(kept) // 10):]
    return body(kept)


def _encoded_size(payload: dict) -> int:
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))





def _wall_clock() -> float:
    return time.time()


def throttle_allows(last_sent_at: float, now: float) -> bool:


    if last_sent_at <= 0 or last_sent_at > now + 60:
        return True
    return now - last_sent_at >= MIN_INTERVAL_S


def counts_as_sent(status: int | None) -> bool:





    return status is not None and (200 <= status < 300 or status == 429)


def _read_last_sent() -> float:
    try:
        from qgis.PyQt.QtCore import QSettings

        return float(QSettings().value(_LAST_SENT_KEY, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _write_last_sent(at: float) -> None:
    from qgis.PyQt.QtCore import QSettings

    settings = QSettings()
    settings.setValue(_LAST_SENT_KEY, float(at))
    settings.sync()





def _context() -> dict:
    from .telemetry import _base_properties

    base = _base_properties()
    out = {key: str(base[key])[:64] for key in (
        "plugin_version", "qgis_version", "os", "os_version", "arch",
        "python_version", "session_id") if base.get(key)}
    try:
        from .telemetry_run_context import run_stage_for_report

        out.update(run_stage_for_report())
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    if "run_id" not in out:
        try:
            from .telemetry import get_last_run_id

            run_id = get_last_run_id()
            if run_id:
                out["run_id"] = str(run_id)[:64]
        except Exception:  # noqa: BLE001  # nosec B110
            pass
    return out


_inflight_lock = threading.Lock()
_inflight: set = set()

_inflight_since = [0.0]
_INFLIGHT_HOLD_S = 120


def send_problem_report_on_open() -> str:






    try:
        from . import telemetry as _tm

        _tm._forget_enabled_cache()
        if not _tm.is_telemetry_enabled() or not _tm._has_consent():
            return OFF
        auth = _tm._get_auth_header()
        if not auth:
            return OFF
        now = _wall_clock()
        if not throttle_allows(_read_last_sent(), now):
            return THROTTLED
        with _inflight_lock:




            if _inflight and 0 <= now - _inflight_since[0] < _INFLIGHT_HOLD_S:
                return THROTTLED
        url = f"{_tm._build_base_url().rstrip('/')}{ROUTE}"
        from .server_dials import cleartext_remote_url

        if cleartext_remote_url(url):
            return OFF
        from .log_scrub import get_recent_log_lines

        payload = build_payload(scrub_report_lines(get_recent_log_lines(LOG_LINES)), _context())
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        _start_task(url, data, auth, now)
        return SENT
    except Exception:  # noqa: BLE001
        return OFF


def _start_task(url: str, data: bytes, auth: dict, started_at: float) -> None:
    from qgis.core import QgsApplication

    task = _make_task(url, data, auth, started_at)
    with _inflight_lock:
        _inflight.add(task)
        _inflight_since[0] = started_at
    QgsApplication.taskManager().addTask(task)


def _make_task(url: str, data: bytes, auth: dict, started_at: float):
    from qgis.core import QgsFeedback, QgsNetworkAccessManager, QgsTask
    from qgis.PyQt.QtCore import QByteArray, QUrl
    from qgis.PyQt.QtNetwork import QNetworkRequest

    from .gil_safe_qobject import prime
    from .qt_compat import reply_http_status, silent_task_flags

    class _ProblemReportTask(QgsTask):


        def __init__(self) -> None:
            super().__init__("AI Segmentation problem report", silent_task_flags())
            self._feedback = prime(QgsFeedback())
            self._sent = False

        def cancel(self) -> None:
            try:
                self._feedback.cancel()
            except Exception:  # noqa: BLE001  # nosec B110
                pass
            super().cancel()

        def run(self) -> bool:
            try:
                from .telemetry import is_telemetry_enabled

                if self.isCanceled() or not is_telemetry_enabled():
                    return True
                req = QNetworkRequest(QUrl(url))
                req.setRawHeader(b"Content-Type", b"application/json")
                if hasattr(req, "setTransferTimeout"):
                    req.setTransferTimeout(_TIMEOUT_MS)
                for key, value in auth.items():
                    req.setRawHeader(key.encode("utf-8"), str(value).encode("utf-8"))
                reply = QgsNetworkAccessManager.blockingPost(
                    req, QByteArray(data), "", True, self._feedback)
                self._sent = counts_as_sent(reply_http_status(reply))
            except Exception:  # noqa: BLE001  # nosec B110
                pass
            return True

        def finished(self, result: bool) -> None:

            try:
                if self._sent:
                    _write_last_sent(started_at)
            except Exception:  # noqa: BLE001  # nosec B110
                pass
            finally:
                with _inflight_lock:
                    _inflight.discard(self)

    return _ProblemReportTask()
