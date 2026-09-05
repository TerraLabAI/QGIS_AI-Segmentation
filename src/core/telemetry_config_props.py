



from __future__ import annotations


def config_provenance_props() -> dict:

    try:
        from .served_config import served_config_age_s, served_config_state

        source = served_config_state()
        age = served_config_age_s()
    except Exception:  # noqa: BLE001
        return {"config_source": "missing", "config_age_s": None}
    return {
        "config_source": source,
        "config_age_s": int(round(age)) if age is not None else None,
    }
