"""Safe archive extraction utilities with path traversal protection."""

import os
import sys
import tarfile
import zipfile


def safe_extract_tar(tar: tarfile.TarFile, dest_dir: str) -> None:
    """Safely extract tar archive with path traversal + symlink protection."""
    dest_dir = os.path.realpath(dest_dir)
    # Python 3.12+ supports the filter parameter (suppresses DeprecationWarning)
    use_filter = sys.version_info >= (3, 12)
    for member in tar.getmembers():
        if member.issym() or member.islnk():
            continue
        member_path = os.path.realpath(os.path.join(dest_dir, member.name))
        if not member_path.startswith(dest_dir + os.sep) and member_path != dest_dir:
            raise ValueError(f"Attempted path traversal in tar archive: {member.name}")
        if use_filter:
            try:
                tar.extract(member, dest_dir, filter="data")
            except (AttributeError, TypeError):
                # Anaconda-patched Python may break filter='data' internally
                # (e.g. ntpath.ALLOW_MISSING missing). Fall back to plain extract.
                tar.extract(member, dest_dir)
        else:
            tar.extract(member, dest_dir)


def safe_extract_zip(zip_file: zipfile.ZipFile, dest_dir: str) -> None:
    """Safely extract zip archive with path traversal protection."""
    dest_dir = os.path.realpath(dest_dir)
    for member in zip_file.namelist():
        member_path = os.path.realpath(os.path.join(dest_dir, member))
        if not member_path.startswith(dest_dir + os.sep) and member_path != dest_dir:
            raise ValueError(f"Attempted path traversal in zip archive: {member}")
        zip_file.extract(member, dest_dir)
