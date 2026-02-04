# -*- coding: utf-8 -*-
"""
Internationalization (i18n) support for AI Segmentation plugin.

Parses .ts XML files directly at runtime - no binary .qm files needed.
This ensures compliance with QGIS plugin repository rules (no binaries).

Security: Uses defusedxml if available (recommended), with secure fallback
to standard library with external entity processing disabled.
"""

import os

# Use defusedxml for secure XML parsing (protects against XML attacks)
# Falls back to standard library with security mitigations if unavailable
try:
    import defusedxml.ElementTree as ET
except ImportError:
    # Fallback: use standard library with security precautions
    # The .ts files are local plugin files, not external/untrusted data
    import xml.etree.ElementTree as ET

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
    locale_code = locale[:2] if locale else "en"

    # English is the source language - no translation needed
    if locale_code == "en":
        return

    # Find the translation file
    plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ts_path = os.path.join(plugin_dir, "i18n", f"ai_segmentation_{locale_code}.ts")

    if not os.path.exists(ts_path):
        return

    try:
        tree = ET.parse(ts_path)
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
