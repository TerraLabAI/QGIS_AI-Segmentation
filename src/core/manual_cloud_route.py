






















from __future__ import annotations



MANUAL_CLOUD_ROUTE_FEATURE = "manual_cloud_route"



_cloud_route_picked = True


def manual_cloud_route_enabled() -> bool:







    return _cloud_route_picked


def set_manual_cloud_route_enabled(enabled: bool) -> None:

    global _cloud_route_picked
    _cloud_route_picked = bool(enabled)


def manual_cloud_route_offered() -> bool:














    if _dev_cloud_route_opt_in():
        return True
    try:
        from .server_dials import dial_bool

        return dial_bool(f"features.{MANUAL_CLOUD_ROUTE_FEATURE}", False)
    except Exception:  # nosec B110
        return False







_MANUAL_CLOUD_ENV = "TERRALAB_MANUAL_CLOUD"


def _dev_cloud_route_opt_in() -> bool:





    from .env_local import env_local_flag

    return env_local_flag(_MANUAL_CLOUD_ENV)
