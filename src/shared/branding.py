# SHARED MODULE v1.0 — keep in sync between AI Canvas and AI Segmentation
"""Shared branding utilities for TerraLab QGIS plugins."""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QLabel, QPushButton

from .constants import (
    BRAND_BLUE,
    BRAND_BLUE_HOVER,
    BRAND_DISABLED,
    BRAND_GREEN,
    BRAND_GREEN_HOVER,
    BRAND_RED,
)


def create_terralab_title_bar(product_name: str) -> QLabel:
    """Create a styled title label for a TerraLab plugin panel."""
    label = QLabel(f"Unlock {product_name}")
    label.setAlignment(Qt.AlignCenter)
    label.setStyleSheet("font-weight: bold; font-size: 13px;")
    return label


def create_primary_button(text: str) -> QPushButton:
    """Create a green primary action button."""
    btn = QPushButton(text)
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {BRAND_GREEN};
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {BRAND_GREEN_HOVER};
        }}
        QPushButton:disabled {{
            background-color: {BRAND_DISABLED};
        }}
    """)
    return btn


def create_secondary_button(text: str) -> QPushButton:
    """Create a blue secondary action button."""
    btn = QPushButton(text)
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {BRAND_BLUE};
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
        }}
        QPushButton:hover {{
            background-color: {BRAND_BLUE_HOVER};
        }}
        QPushButton:disabled {{
            background-color: {BRAND_DISABLED};
        }}
    """)
    return btn


def style_error_label(label: QLabel):
    """Apply error styling to a label."""
    label.setStyleSheet(f"color: {BRAND_RED}; font-size: 11px;")


def style_success_label(label: QLabel):
    """Apply success styling to a label."""
    label.setStyleSheet(f"color: {BRAND_GREEN}; font-size: 11px;")
