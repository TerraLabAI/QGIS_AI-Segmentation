










from __future__ import annotations

from .env_setup_account import (
    EnvSetupAccountMixin,
    _drop_untagged_account_history,
    _notify_ui,
)
from .env_setup_activation import EnvSetupActivationMixin
from .env_setup_install import (
    _INSTALL_ATTEMPT_KEY,
    _INSTALL_PIPE_WAIT_MAX,
    EnvSetupInstallMixin,
    _bump_install_attempt,
    _clear_install_attempts,
)
from .env_setup_startup import (
    _VCREDIST_URL,
    EnvSetupStartupMixin,
)


class EnvSetupMixin(
    EnvSetupStartupMixin,
    EnvSetupInstallMixin,
    EnvSetupActivationMixin,
    EnvSetupAccountMixin,
):
    pass


__all__ = [
    "EnvSetupMixin",
    "_notify_ui",
    "_INSTALL_ATTEMPT_KEY",
    "_INSTALL_PIPE_WAIT_MAX",
    "_VCREDIST_URL",
    "_bump_install_attempt",
    "_clear_install_attempts",
    "_drop_untagged_account_history",
]
