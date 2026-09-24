








from __future__ import annotations

import os
import xml.etree.ElementTree as ET  # nosec B405


try:
    from defusedxml.ElementTree import parse as _safe_parse
except ImportError:
    _safe_parse = ET.parse

from qgis.PyQt.QtCore import QSettings


CONTEXT = "AISegmentation"


_translations: dict[str, str] = {}


_loaded = False



LANGUAGE_FALLBACKS = {
    "pt": "pt_BR",
    "pt_PT": "pt_BR",
    "es_MX": "es",
    "es_AR": "es",



    "zh": "zh_CN",
    "zh_Hans": "zh_CN",
    "zh_SG": "zh_CN",
    "zh_Hant": "zh_TW",
    "zh_HK": "zh_TW",
    "zh_MO": "zh_TW",
}


def locale_variants(locale: str) -> list[str]:






    variants: list[str] = []
    normalized = (locale or "").replace("-", "_")
    if not normalized:
        return variants
    if "_" in normalized:

        variants.append(normalized)



        parts = normalized.split("_")
        if len(parts) > 2:
            script = f"{parts[0]}_{parts[1]}"
            variants.append(script)
            if script in LANGUAGE_FALLBACKS:
                variants.append(LANGUAGE_FALLBACKS[script])
        variants.append(normalized[:2])
        if normalized in LANGUAGE_FALLBACKS:
            variants.append(LANGUAGE_FALLBACKS[normalized])
        if normalized[:2] in LANGUAGE_FALLBACKS:
            variants.append(LANGUAGE_FALLBACKS[normalized[:2]])
    else:
        variants.append(normalized[:2])
        if normalized[:2] in LANGUAGE_FALLBACKS:
            variants.append(LANGUAGE_FALLBACKS[normalized[:2]])
    return variants






_user_locale: str | None = None


def current_locale() -> str:

    global _user_locale
    if _user_locale is None:
        try:
            _user_locale = str(QSettings().value("locale/userLocale", "en_US") or "")
        except Exception:  # noqa: BLE001  # nosec B110
            return ""
    return _user_locale


def resolve_language(supported) -> str | None:





    try:
        for variant in locale_variants(current_locale()):
            if variant in supported:
                return variant
    except Exception:  # noqa: BLE001  # nosec B110
        return None
    return None


def _load_translations():






    global _loaded

    if _loaded:
        return

    _loaded = True

    ts_path = None
    try:


        locale = current_locale()
        if not locale:
            return


        if locale.startswith("en"):
            return


        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        for variant in locale_variants(locale):
            candidate = os.path.join(plugin_dir, "i18n", f"ai_segmentation_{variant}.ts")
            if os.path.exists(candidate):
                ts_path = candidate
                break

        if ts_path is None:
            return

        tree = _safe_parse(ts_path)
        root = tree.getroot()




        parsed: dict[str, str] = {}


        for context in root.findall("context"):
            context_name = context.find("name")
            if context_name is None or context_name.text != CONTEXT:
                continue


            for message in context.findall("message"):
                source = message.find("source")
                translation = message.find("translation")

                if source is None or translation is None:
                    continue

                source_text = source.text or ""



                translation_text = translation.text
                if translation_text is None:
                    forms = translation.findall("numerusform")
                    if forms:
                        translation_text = forms[0].text or ""
                    else:
                        translation_text = "".join(translation.itertext())


                if translation_text and translation.get("type") != "unfinished":
                    parsed[source_text] = translation_text

        _translations.update(parsed)

    except Exception as e:  # noqa: BLE001
        try:
            from qgis.core import Qgis, QgsMessageLog
            QgsMessageLog.logMessage(
                f"Failed to load translations from {ts_path}: {e}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )
        except Exception:
            pass  # nosec B110


def tr(message: str) -> str:










    if not _loaded:
        _load_translations()


    return _translations.get(message, message)
