














from __future__ import annotations

from .server_dials import dial_bool, dial_str, feature_enabled

_DEFAULT_CONTACT_EMAIL = "yvann.barbot@terra-lab.ai"
_MAX_CHARS = 120


def pro_ceiling_enabled() -> bool:


    return feature_enabled("pro_ceiling") and dial_bool("pro_ceiling.enabled", True)


def _looks_like_email(value: str) -> bool:


    if not value or len(value) > _MAX_CHARS or value.count("@") != 1:
        return False
    local, _, domain = value.partition("@")
    if not local or "." not in domain or domain.startswith(".") or domain.endswith("."):
        return False
    return all(ch.isprintable() and not ch.isspace() for ch in value)


def pro_ceiling_contact_email() -> str:


    served = dial_str("pro_ceiling.contact_email", _DEFAULT_CONTACT_EMAIL)
    return served if _looks_like_email(served) else _DEFAULT_CONTACT_EMAIL
