











from __future__ import annotations

from .terralab_client_account import TerraLabAccountMixin
from .terralab_client_detection import TerraLabDetectionMixin



from .terralab_client_errors import (
    _CONNECT_FAILURE_ERRORS,
    _PROXY_ERRORS,
    _classify_network_error,
    _classify_qt_error,
    _error_shaped,
    _named_error_text,
    _unreadable_answer,
)
from .terralab_client_nam_pool import (
    _CONNECTIONS_PER_MANAGER,
    TerraLabManagerPoolMixin,
    _drop_thread_nam,
    _predict_manager_count,
    _qobject_alive,
)
from .terralab_client_primitives import (
    _HTTP_STATUS_ATTR,
    _NE,
    _NO_LESS_SAFE_REDIRECT,
    _REDIRECT_ATTR,
    _SAME_ORIGIN_REDIRECT,
    _SERVER_CONTACT_TTL_S,
    _TIMEOUT_API,
    _TIMEOUT_CHECKOUT_LINK,
    _TIMEOUT_INTERACTIVE,
    _TIMEOUT_POLL_DETECTION,
    _TIMEOUT_RUN_EXPORT,
    _TIMEOUT_SUBMIT_DETECTION,
    _TIMEOUT_SUBMIT_DETECTION_DIRECT,
    _TIMEOUT_TRANSLATE,
    _TIMEOUT_WARMUP,
    _WALL_CLOCK_GUARD_MS,
    _apply_redirect_policy,
    _Attr,
    _AuthRequired,
    _ConnRefused,
    _ContentDenied,
    _HostNotFound,
    _http_status_of,
    _log_warning,
    _NoError,
    _OpCanceled,
    _parse_json_body,
    _RedirectPolicy,
    _reply_was_packed,
    _SslFailed,
    _Timeout,
    _UnknownNetwork,
    _WallClockGuard,
    note_server_contact,
    server_reached_recently,
)
from .terralab_client_retry import (
    _HANDOFF_STATUSES,
    _RATE_LIMITED_STATUS,
    _RETRY_AFTER_HINT_MAX_S,
    _RETRY_AFTER_MAX_S,
    _RETRY_PAUSE_MAX_S,
    _RETRY_PAUSE_MIN_S,
    _WINDOW_HINT_MAX,
    _WINDOW_HINT_MIN,
    _WORTH_ASKING_AGAIN_CODES,
    _note_retry_after,
    _note_skipped_tuning,
    _note_window_hint,
    _parse_retry_after_header,
    _retry_after_hint,
    _retry_after_s,
    _retry_pause_s,
    _window_hint,
    _worth_asking_again,
)
from .terralab_client_transport import (
    TerraLabTransportMixin,
    _may_pack_body,
)

__all__ = [
    "TerraLabClient",
    "_Attr",
    "_AuthRequired",
    "_CONNECTIONS_PER_MANAGER",
    "_CONNECT_FAILURE_ERRORS",
    "_ConnRefused",
    "_ContentDenied",
    "_HANDOFF_STATUSES",
    "_HTTP_STATUS_ATTR",
    "_HostNotFound",
    "_NE",
    "_NO_LESS_SAFE_REDIRECT",
    "_NoError",
    "_OpCanceled",
    "_PROXY_ERRORS",
    "_RATE_LIMITED_STATUS",
    "_REDIRECT_ATTR",
    "_RETRY_AFTER_HINT_MAX_S",
    "_RETRY_AFTER_MAX_S",
    "_RETRY_PAUSE_MAX_S",
    "_RETRY_PAUSE_MIN_S",
    "_RedirectPolicy",
    "_SAME_ORIGIN_REDIRECT",
    "_SERVER_CONTACT_TTL_S",
    "_SslFailed",
    "_TIMEOUT_API",
    "_TIMEOUT_CHECKOUT_LINK",
    "_TIMEOUT_INTERACTIVE",
    "_TIMEOUT_POLL_DETECTION",
    "_TIMEOUT_RUN_EXPORT",
    "_TIMEOUT_SUBMIT_DETECTION",
    "_TIMEOUT_SUBMIT_DETECTION_DIRECT",
    "_TIMEOUT_TRANSLATE",
    "_TIMEOUT_WARMUP",
    "_Timeout",
    "_UnknownNetwork",
    "_WALL_CLOCK_GUARD_MS",
    "_WINDOW_HINT_MAX",
    "_WINDOW_HINT_MIN",
    "_WORTH_ASKING_AGAIN_CODES",
    "_WallClockGuard",
    "_apply_redirect_policy",
    "_classify_network_error",
    "_classify_qt_error",
    "_drop_thread_nam",
    "_error_shaped",
    "_http_status_of",
    "_log_warning",
    "_may_pack_body",
    "_named_error_text",
    "_note_retry_after",
    "_note_skipped_tuning",
    "_note_window_hint",
    "_parse_json_body",
    "_parse_retry_after_header",
    "_predict_manager_count",
    "_qobject_alive",
    "_reply_was_packed",
    "_retry_after_hint",
    "_retry_after_s",
    "_retry_pause_s",
    "_unreadable_answer",
    "_window_hint",
    "_worth_asking_again",
    "note_server_contact",
    "server_reached_recently",
]


class TerraLabClient(
    TerraLabManagerPoolMixin,
    TerraLabTransportMixin,
    TerraLabDetectionMixin,
    TerraLabAccountMixin,
):



    def __init__(self, base_url: str | None = None):
        if base_url is None:
            base_url = self._read_base_url()
        self.base_url = base_url.rstrip("/")





        direct = self._read_detection_base_url()
        if direct and not self._direct_detection_allowed():
            direct = ""
        self.detection_direct = bool(direct)
        self.detection_base_url = (direct or self.base_url).rstrip("/")


        self._pending_retry_after_s = 0.0

    @staticmethod
    def _direct_detection_allowed() -> bool:






        try:
            from ..core.server_dials import feature_enabled

            return feature_enabled("direct_inference")
        except Exception:  # noqa: BLE001  # nosec B110
            return True

    @staticmethod
    def _read_base_url() -> str:
        from ..core.env_local import terralab_base_url

        return terralab_base_url()

    @staticmethod
    def _read_detection_base_url() -> str:














        _DEFAULT_DETECTION_DIRECT_URL = "https://inference.terra-lab.ai"
        return TerraLabClient._read_env_value("TERRALAB_DETECTION_URL", _DEFAULT_DETECTION_DIRECT_URL)

    @staticmethod
    def _read_env_value(name: str, default: str) -> str:
        from ..core.env_local import env_local_value

        return env_local_value(name, default)
