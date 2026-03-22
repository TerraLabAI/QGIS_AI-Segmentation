# SHARED MODULE v1.0 — keep in sync between AI Canvas and AI Segmentation
# Product-specific behavior controlled via parameters, not code changes
"""Shared constants for TerraLab QGIS plugins."""

TERRALAB_URL = "https://terra-lab.ai"
NEWSLETTER_URL = "https://terra-lab.ai/mail-verification"

BRAND_GREEN = "#2e7d32"
BRAND_GREEN_HOVER = "#1b5e20"
BRAND_BLUE = "#1976d2"
BRAND_BLUE_HOVER = "#1565c0"
BRAND_RED = "#d32f2f"
BRAND_DISABLED = "#b0bec5"

PRODUCTS = {
    "ai-canvas": {
        "display_name": "AI Canvas",
        "qsettings_prefix": "AICanvas/",
        "newsletter_param": "plugin=ai-canvas",
    },
    "ai-segmentation": {
        "display_name": "AI Segmentation",
        "qsettings_prefix": "AISegmentation/",
        "newsletter_param": "plugin=ai-segmentation",
    },
}
