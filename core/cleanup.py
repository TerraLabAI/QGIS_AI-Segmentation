"""
Cleanup Utilities for AI Segmentation

Provides functions to clean up cache files and settings when the plugin
is uninstalled or when the user wants to reset everything.

Note: The main plugin files, dependencies, and models are automatically
deleted when uninstalling via QGIS Plugin Manager because they are stored
inside the plugin folder. This module handles files stored OUTSIDE the
plugin folder.
"""

import os
import shutil
from pathlib import Path
from typing import Tuple, List

from qgis.core import QgsMessageLog, Qgis, QgsSettings


# Cache locations
WEB_CACHE_DIR = Path.home() / ".qgis_ai_segmentation_cache"
FILE_CACHE_NAME = ".ai_segmentation_cache"

# Settings keys used by the plugin
SETTINGS_KEYS = [
    "AI_Segmentation/dependencies_dismissed",
]


def get_web_cache_size() -> Tuple[int, int]:
    """
    Get the size and count of web layer cache.
    
    Returns:
        Tuple of (total_bytes, file_count)
    """
    if not WEB_CACHE_DIR.exists():
        return 0, 0
    
    total_size = 0
    file_count = 0
    
    for root, dirs, files in os.walk(WEB_CACHE_DIR):
        for file in files:
            file_path = Path(root) / file
            try:
                total_size += file_path.stat().st_size
                file_count += 1
            except OSError:
                pass
    
    return total_size, file_count


def clear_web_cache() -> Tuple[bool, str]:
    """
    Clear the cache for web-based layers (XYZ, WMS, etc.).
    
    This cache is stored in ~/.qgis_ai_segmentation_cache/
    
    Returns:
        Tuple of (success, message)
    """
    try:
        if not WEB_CACHE_DIR.exists():
            return True, "No web cache to clear"
        
        size, count = get_web_cache_size()
        size_mb = size / (1024 * 1024)
        
        shutil.rmtree(WEB_CACHE_DIR, ignore_errors=True)
        
        QgsMessageLog.logMessage(
            f"Cleared web cache: {count} files, {size_mb:.1f} MB",
            "AI Segmentation",
            level=Qgis.Info
        )
        
        return True, f"Cleared {count} cached files ({size_mb:.1f} MB)"
        
    except Exception as e:
        QgsMessageLog.logMessage(
            f"Failed to clear web cache: {str(e)}",
            "AI Segmentation",
            level=Qgis.Warning
        )
        return False, f"Failed: {str(e)}"


def clear_settings() -> Tuple[bool, str]:
    """
    Clear all plugin settings from QgsSettings.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        settings = QgsSettings()
        cleared = 0
        
        for key in SETTINGS_KEYS:
            if settings.contains(key):
                settings.remove(key)
                cleared += 1
        
        QgsMessageLog.logMessage(
            f"Cleared {cleared} plugin settings",
            "AI Segmentation",
            level=Qgis.Info
        )
        
        return True, f"Cleared {cleared} settings"
        
    except Exception as e:
        QgsMessageLog.logMessage(
            f"Failed to clear settings: {str(e)}",
            "AI Segmentation",
            level=Qgis.Warning
        )
        return False, f"Failed: {str(e)}"


def find_file_caches(search_paths: List[str] = None) -> List[Path]:
    """
    Find .ai_segmentation_cache directories next to GeoTIFF files.
    
    Note: This only searches common locations. The user may have cached
    files in other locations that won't be found.
    
    Args:
        search_paths: Optional list of paths to search. If None, searches
                      common locations (Documents, Desktop, etc.)
    
    Returns:
        List of cache directory paths found
    """
    cache_dirs = []
    
    if search_paths is None:
        # Search common user directories
        home = Path.home()
        search_paths = [
            home / "Documents",
            home / "Desktop",
            home / "Downloads",
        ]
        # Add Windows-specific paths
        if os.name == 'nt':
            search_paths.extend([
                Path("C:/Users") / os.getenv("USERNAME", "") / "Documents",
            ])
    
    for search_path in search_paths:
        search_path = Path(search_path)
        if not search_path.exists():
            continue
        
        try:
            for root, dirs, files in os.walk(search_path):
                if FILE_CACHE_NAME in dirs:
                    cache_dirs.append(Path(root) / FILE_CACHE_NAME)
                
                # Don't recurse too deep
                if root.count(os.sep) - str(search_path).count(os.sep) > 5:
                    dirs.clear()
        except PermissionError:
            continue
    
    return cache_dirs


def clear_file_caches(cache_dirs: List[Path] = None) -> Tuple[bool, str]:
    """
    Clear .ai_segmentation_cache directories next to GeoTIFF files.
    
    Args:
        cache_dirs: List of cache directories to clear. If None, searches
                    and clears all found caches.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        if cache_dirs is None:
            cache_dirs = find_file_caches()
        
        if not cache_dirs:
            return True, "No file caches found"
        
        cleared = 0
        failed = 0
        
        for cache_dir in cache_dirs:
            try:
                if cache_dir.exists():
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    cleared += 1
                    QgsMessageLog.logMessage(
                        f"Cleared cache: {cache_dir}",
                        "AI Segmentation",
                        level=Qgis.Info
                    )
            except Exception:
                failed += 1
        
        if failed > 0:
            return False, f"Cleared {cleared}, failed {failed} caches"
        
        return True, f"Cleared {cleared} cache directories"
        
    except Exception as e:
        return False, f"Failed: {str(e)}"


def full_cleanup() -> Tuple[bool, str]:
    """
    Perform a full cleanup of all plugin data outside the plugin folder.
    
    This includes:
    - Web layer cache (~/.qgis_ai_segmentation_cache/)
    - File caches (.ai_segmentation_cache/ next to GeoTIFFs)
    - Plugin settings in QgsSettings
    
    Note: The plugin folder itself (including dependencies and models)
    is automatically deleted by QGIS when uninstalling the plugin.
    
    Returns:
        Tuple of (all_success, combined_message)
    """
    messages = []
    all_success = True
    
    # Clear web cache
    success, msg = clear_web_cache()
    messages.append(f"Web cache: {msg}")
    if not success:
        all_success = False
    
    # Clear settings
    success, msg = clear_settings()
    messages.append(f"Settings: {msg}")
    if not success:
        all_success = False
    
    # Note about file caches
    messages.append(
        "File caches: Search manually for '.ai_segmentation_cache' folders "
        "next to your GeoTIFF files if needed."
    )
    
    QgsMessageLog.logMessage(
        f"Full cleanup completed: {'; '.join(messages)}",
        "AI Segmentation",
        level=Qgis.Info
    )
    
    return all_success, "\n".join(messages)


def get_cleanup_info() -> dict:
    """
    Get information about what would be cleaned up.
    
    Returns:
        Dictionary with cleanup information
    """
    web_size, web_count = get_web_cache_size()
    
    return {
        "web_cache": {
            "path": str(WEB_CACHE_DIR),
            "exists": WEB_CACHE_DIR.exists(),
            "size_bytes": web_size,
            "size_mb": web_size / (1024 * 1024),
            "file_count": web_count,
        },
        "settings": {
            "keys": SETTINGS_KEYS,
        },
        "note": (
            "Plugin folder (dependencies, models, code) is automatically "
            "deleted when uninstalling via QGIS Plugin Manager."
        ),
    }
