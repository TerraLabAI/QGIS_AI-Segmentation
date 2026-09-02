"""
Internationalization (i18n) support for AI Segmentation plugin.

Parses .ts XML files directly at runtime - no binary .qm files needed.
This ensures compliance with QGIS plugin repository rules (no binaries).

Security: Uses defusedxml for safe XML parsing (no global monkey-patch).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET  # nosec B405

# Prefer defusedxml for safe XML parsing (no global monkey-patch)
try:
    from defusedxml.ElementTree import parse as _safe_parse
except ImportError:
    _safe_parse = ET.parse  # fallback: .ts files are local trusted plugin files

from qgis.PyQt.QtCore import QSettings

# Translation context - must match the context in .ts files
CONTEXT = "AISegmentation"

# Translation dictionary: {source_text: translated_text}
_translations = {}

# Flag to track if translations have been loaded
_loaded = False

# Language fallbacks: map language variants to available translations
# e.g., pt_PT (European Portuguese) -> pt_BR (Brazilian Portuguese)
LANGUAGE_FALLBACKS = {
    "pt": "pt_BR",      # Portuguese -> Brazilian Portuguese
    "pt_PT": "pt_BR",   # European Portuguese -> Brazilian Portuguese
    "es_MX": "es",      # Mexican Spanish -> Spanish
    "es_AR": "es",      # Argentine Spanish -> Spanish
    # Chinese: route bare/script/region variants to Simplified or Traditional.
    # Order in locale_variants ensures Hant/HK/MO resolve to zh_TW before the
    # generic "zh" -> zh_CN fallback is reached.
    "zh": "zh_CN",      # bare Chinese -> Simplified
    "zh_Hans": "zh_CN",  # Simplified script -> Simplified
    "zh_SG": "zh_CN",   # Singapore -> Simplified
    "zh_Hant": "zh_TW",  # Traditional script -> Traditional
    "zh_HK": "zh_TW",   # Hong Kong -> Traditional
    "zh_MO": "zh_TW",   # Macau -> Traditional
}


def locale_variants(locale: str) -> list[str]:
    """Ordered language codes to try for a QGIS locale string.

    Full code first (e.g. pt_BR), then the bare language code (pt), then the
    fallbacks above. The single source for "which language is this user in",
    shared by the translation loader and the server request context.
    """
    variants: list[str] = []
    normalized = (locale or "").replace("-", "_")  # normalize to underscore
    if not normalized:
        return variants
    if "_" in normalized:
        # e.g., "pt_BR" -> try "pt_BR" first, then "pt", then fallback
        variants.append(normalized)
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


# Session memo of the locale string. QGIS applies a language change only after
# a restart, so this cannot move under us. It matters because building a
# QSettings costs far more than reading a value off an existing one, and the
# locale is read on paths that run hundreds of times per panel build.
_user_locale: str | None = None


def current_locale() -> str:
    """The QGIS UI locale, or an empty string when it cannot be read."""
    global _user_locale
    if _user_locale is None:
        try:
            _user_locale = str(QSettings().value("locale/userLocale", "en_US") or "")
        except Exception:  # noqa: BLE001 -- locale is best-effort  # nosec B110
            return ""
    return _user_locale


def reset_locale_cache() -> None:
    """Drop the memoized locale and everything loaded from it.

    The memo is valid for as long as the QGIS locale is, which is the whole
    session. Call this only when the locale itself changes under the plugin.
    """
    global _user_locale, _loaded
    _user_locale = None
    _loaded = False
    _translations.clear()


def resolve_language(supported) -> str | None:
    """The first ``supported`` language code matching the QGIS UI locale.

    None when the user's language is not in ``supported``, so a caller can omit
    the value rather than guess one.
    """
    try:
        for variant in locale_variants(current_locale()):
            if variant in supported:
                return variant
    except Exception:  # noqa: BLE001 -- locale is best-effort  # nosec B110
        return None
    return None


def _load_translations():
    """Load translations from .ts XML file based on QGIS locale.

    Never raises. Every caller reaches this through ``tr()``, which sits on the
    first line of most of the plugin, so anything that escapes here takes the
    whole plugin down at startup rather than costing one language.
    """
    global _loaded

    if _loaded:
        return

    _loaded = True

    ts_path = None
    try:
        # str(): QSettings hands back whatever the profile holds, and a value
        # written as a list or a number has no startswith.
        locale = current_locale()
        if not locale:
            return

        # English is the source language - no translation needed
        if locale.startswith("en"):
            return

        # Find the translation file
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

        # Built aside and swapped in at the end: a file that parses and then
        # breaks part way through leaves the plugin all-English rather than
        # half translated, which is the harder thing to report.
        parsed: dict[str, str] = {}

        # Parse all contexts
        for context in root.findall("context"):
            context_name = context.find("name")
            if context_name is None or context_name.text != CONTEXT:
                continue

            # Parse all messages in this context
            for message in context.findall("message"):
                source = message.find("source")
                translation = message.find("translation")

                if source is None or translation is None:
                    continue

                source_text = source.text or ""
                # A plain <translation> carries its text directly. A plural one
                # nests <numerusform> children and leaves .text at None, which
                # dropped the whole entry in silence.
                translation_text = translation.text
                if translation_text is None:
                    forms = translation.findall("numerusform")
                    if forms:
                        translation_text = forms[0].text or ""
                    else:
                        translation_text = "".join(translation.itertext())

                # Skip unfinished/empty translations
                if translation_text and translation.get("type") != "unfinished":
                    parsed[source_text] = translation_text

        _translations.update(parsed)

    except Exception as e:  # noqa: BLE001 -- English is always a valid answer
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
    """
    Translate a string using the plugin's translation files.

    Args:
        message: The string to translate (English source text)

    Returns:
        The translated string, or the original if no translation is available
    """
    # Ensure translations are loaded
    if not _loaded:
        _load_translations()

    # Return translation if available, otherwise original text
    return _translations.get(message, message)
