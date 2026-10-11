





from __future__ import annotations

from .i18n import tr


def is_computer_cap(payload) -> bool:

    if not isinstance(payload, dict):
        return False
    device = payload.get("free_device")
    if isinstance(device, dict):
        return device.get("reason") == "account_cap"
    return payload.get("free_device_reason") == "account_cap"


def computer_cap_message(server_message: str = "") -> str:

    text = (server_message or "").strip()
    if text:
        return text
    return (tr("Free AI Segmentation was already used by other accounts on "
               "this computer.") + " "
            + tr("Upgrade to Pro to keep going. Shared computer, or a "
                 "mistake? Contact yvann.barbot@terra-lab.ai and we will unlock "
                 "it."))
