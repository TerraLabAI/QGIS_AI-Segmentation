#!/usr/bin/env python3
"""
Install AI Segmentation plugin to QGIS plugins directory.

Usage:
    python install_plugin.py          # Copy plugin
    python install_plugin.py --dev     # Create symlink (development mode)
    python install_plugin.py --remove  # Remove installed plugin
"""

import argparse
import os
import platform
import shutil
import sys
import time
from pathlib import Path


def get_qgis_plugins_dir() -> Path:
    """Get QGIS plugins directory for current platform."""
    system = platform.system()
    
    if system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        return Path(appdata) / "QGIS" / "QGIS3" / "profiles" / "default" / "python" / "plugins"
    
    elif system == "Darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "QGIS" / "QGIS3" / "profiles" / "default" / "python" / "plugins"
    
    else:  # Linux
        return Path.home() / ".local" / "share" / "QGIS" / "QGIS3" / "profiles" / "default" / "python" / "plugins"


def get_plugin_source_dir() -> Path:
    """Get plugin source directory."""
    return Path(__file__).parent.parent


def safe_rmtree(path: Path, max_retries: int = 3, delay: float = 0.5):
    """Safely remove directory tree with retry logic for Windows file locking."""
    for attempt in range(max_retries):
        try:
            if path.is_symlink():
                path.unlink()
            else:
                # On Windows, remove read-only attributes first
                if platform.system() == "Windows":
                    def handle_remove_readonly(func, path, exc):
                        """Handle read-only files on Windows."""
                        os.chmod(path, 0o777)
                        func(path)
                    shutil.rmtree(path, onerror=handle_remove_readonly)
                else:
                    shutil.rmtree(path)
            return
        except (OSError, PermissionError) as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries} after {delay}s...")
                time.sleep(delay)
            else:
                print(f"\nError: Cannot remove {path}")
                print("This usually happens when QGIS is running and has locked the files.")
                print("\nPlease try one of the following:")
                print("  1. Close QGIS completely and try again")
                print("  2. Restart your computer if QGIS processes are still running")
                print("  3. Manually delete the directory: " + str(path))
                raise


def install_copy(source: Path, dest: Path):
    """Install by copying files."""
    if dest.exists():
        print(f"Removing existing installation: {dest}")
        safe_rmtree(dest)
    
    print(f"Copying plugin to: {dest}")
    shutil.copytree(source, dest)
    print("Done! Restart QGIS and enable the plugin.")


def install_symlink(source: Path, dest: Path):
    """Install by creating symlink (development mode)."""
    if dest.exists():
        if dest.is_symlink():
            print(f"Removing existing symlink: {dest}")
            dest.unlink()
        else:
            print(f"Removing existing installation: {dest}")
            safe_rmtree(dest)
    
    print(f"Creating symlink: {dest} -> {source}")
    
    # On Windows, need admin or developer mode for symlinks
    if platform.system() == "Windows":
        try:
            dest.symlink_to(source, target_is_directory=True)
        except OSError:
            print("Error: Cannot create symlink. Try one of:")
            print("  1. Run as Administrator")
            print("  2. Enable Developer Mode in Windows Settings")
            print("  3. Use --copy instead of --dev")
            sys.exit(1)
    else:
        dest.symlink_to(source)
    
    print("Done! Restart QGIS and enable the plugin.")
    print("Changes to plugin code will be reflected immediately (restart QGIS to reload).")


def remove_plugin(dest: Path):
    """Remove installed plugin."""
    if not dest.exists():
        print(f"Plugin not installed at: {dest}")
        return
    
    if dest.is_symlink():
        print(f"Removing symlink: {dest}")
        dest.unlink()
    else:
        print(f"Removing plugin: {dest}")
        safe_rmtree(dest)
    
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Install AI Segmentation QGIS plugin")
    parser.add_argument("--dev", action="store_true", help="Development mode (symlink)")
    parser.add_argument("--copy", action="store_true", help="Copy files (default)")
    parser.add_argument("--remove", action="store_true", help="Remove installed plugin")
    parser.add_argument("--plugins-dir", type=str, help="Custom QGIS plugins directory")
    
    args = parser.parse_args()
    
    # Determine directories
    source = get_plugin_source_dir()
    
    if args.plugins_dir:
        plugins_dir = Path(args.plugins_dir)
    else:
        plugins_dir = get_qgis_plugins_dir()
    
    # Plugin directory name in QGIS (typically lowercase with underscores)
    dest = plugins_dir / "ai_segmentation"
    
    # Ensure plugins directory exists
    plugins_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Plugin source: {source}")
    print(f"QGIS plugins dir: {plugins_dir}")
    print()
    
    if not source.exists():
        print(f"Error: Plugin source not found: {source}")
        sys.exit(1)
    
    if args.remove:
        remove_plugin(dest)
    elif args.dev:
        install_symlink(source, dest)
    else:
        install_copy(source, dest)


if __name__ == "__main__":
    main()
