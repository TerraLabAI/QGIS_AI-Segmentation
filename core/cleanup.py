

import os
import shutil
from pathlib import Path
from typing import Tuple, List

from qgis.core import QgsMessageLog, Qgis, QgsSettings


WEB_CACHE_DIR = Path.home() / ".qgis_ai_segmentation_cache"
FILE_CACHE_NAME = ".ai_segmentation_cache"

SETTINGS_KEYS = [
    "AI_Segmentation/dependencies_dismissed",
]


def get_web_cache_size() -> Tuple[int, int]:
    
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
    
    cache_dirs = []
    
    if search_paths is None:
        home = Path.home()
        search_paths = [
            home / "Documents",
            home / "Desktop",
            home / "Downloads",
        ]
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
                
                if root.count(os.sep) - str(search_path).count(os.sep) > 5:
                    dirs.clear()
        except PermissionError:
            continue
    
    return cache_dirs


def clear_file_caches(cache_dirs: List[Path] = None) -> Tuple[bool, str]:
    
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
    
    messages = []
    all_success = True
    
    success, msg = clear_web_cache()
    messages.append(f"Web cache: {msg}")
    if not success:
        all_success = False
    
    success, msg = clear_settings()
    messages.append(f"Settings: {msg}")
    if not success:
        all_success = False
    
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
