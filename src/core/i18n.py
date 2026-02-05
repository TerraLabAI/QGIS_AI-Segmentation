# -*- coding: utf-8 -*-
"""
Internationalization (i18n) support for AI Segmentation plugin.

Parses .ts XML files directly at runtime - no binary .qm files needed.
This ensures compliance with QGIS plugin repository rules (no binaries).

Security: Uses defusedxml to patch stdlib for secure XML parsing.
"""

import os

# Patch stdlib XML modules to protect against XML attacks (XXE, billion laughs, etc.)
# This must be done before any XML parsing occurs
try:
    import defusedxml
    defusedxml.defuse_stdlib()
except ImportError:
    pass  # defusedxml not available, .ts files are local trusted plugin files

import xml.etree.ElementTree as ET  # noqa: E402 - must import after defuse_stdlib()

from qgis.PyQt.QtCore import QSettings

# Translation context - must match the context in .ts files
CONTEXT = "AISegmentation"

# Translation dictionary: {source_text: translated_text}
_translations = {}

# Flag to track if translations have been loaded
_loaded = False


def _load_translations():
    """Load translations from .ts XML file based on QGIS locale."""
    global _loaded

    if _loaded:
        return

    _loaded = True

    # Get the locale from QGIS settings
    locale = QSettings().value("locale/userLocale", "en_US")
    if not locale:
        return

    # English is the source language - no translation needed
    if locale.startswith("en"):
        return

    # Find the translation file
    plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Language fallbacks: map language variants to available translations
    # e.g., pt_PT (European Portuguese) -> pt_BR (Brazilian Portuguese)
    language_fallbacks = {
        "pt": "pt_BR",      # Portuguese -> Brazilian Portuguese
        "pt_PT": "pt_BR",   # European Portuguese -> Brazilian Portuguese
        "es_MX": "es",      # Mexican Spanish -> Spanish
        "es_AR": "es",      # Argentine Spanish -> Spanish
    }

    # Try full locale code first (e.g., pt_BR), then fall back to language code (e.g., pt)
    locale_variants = []
    normalized_locale = locale.replace("-", "_")  # normalize to underscore

    if "_" in normalized_locale:
        # e.g., "pt_BR" -> try "pt_BR" first, then "pt", then fallback
        locale_variants.append(normalized_locale)
        locale_variants.append(normalized_locale[:2])
        # Add fallback if defined
        if normalized_locale in language_fallbacks:
            locale_variants.append(language_fallbacks[normalized_locale])
        if normalized_locale[:2] in language_fallbacks:
            locale_variants.append(language_fallbacks[normalized_locale[:2]])
    else:
        locale_variants.append(normalized_locale[:2])
        # Add fallback if defined
        if normalized_locale[:2] in language_fallbacks:
            locale_variants.append(language_fallbacks[normalized_locale[:2]])

    ts_path = None
    for variant in locale_variants:
        candidate = os.path.join(plugin_dir, "i18n", f"ai_segmentation_{variant}.ts")
        if os.path.exists(candidate):
            ts_path = candidate
            break

    if ts_path is None:
        return

    try:
        tree = ET.parse(ts_path)  # nosec B314 - defuse_stdlib() called at module load
        root = tree.getroot()

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
                translation_text = translation.text

                # Skip unfinished/empty translations
                if translation_text and translation.get("type") != "unfinished":
                    _translations[source_text] = translation_text

    except Exception:
        # Silently fail - fall back to English
        pass


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
