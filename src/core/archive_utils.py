"""Safe archive extraction utilities with path traversal protection."""

from __future__ import annotations

import os
import tarfile
import zipfile

# Ceiling on what one archive may write. The two archives this module opens are
# a Python runtime and an installer binary, both well under a gigabyte, so this
# only ever trips on an archive that is not what it claims to be. Without it a
# few kilobytes of nested compression can fill the user's disk before any
# per-member check runs.
MAX_EXTRACTED_BYTES = 4 * 1024 * 1024 * 1024


def _reject_oversized(total: int, name: str) -> None:
    """Raise once the declared uncompressed size passes the ceiling."""
    if total > MAX_EXTRACTED_BYTES:
        raise ValueError(
            f"Archive declares more than {MAX_EXTRACTED_BYTES} bytes uncompressed, "
            f"refusing to extract: {name}"
        )


def safe_extract_tar(tar: tarfile.TarFile, dest_dir: str) -> None:
    """Safely extract tar archive with path traversal + symlink protection."""
    dest_dir = os.path.realpath(dest_dir)
    dest_cmp = os.path.normcase(dest_dir)
    # The extraction filter was backported to 3.9.17, 3.10.12 and 3.11.4, so
    # test for the feature rather than for 3.12: gating on the version number
    # silently dropped most 3.11 users onto the plain-extract branch.
    use_filter = hasattr(tarfile, "data_filter")
    total = 0
    for member in tar.getmembers():
        if member.issym() or member.islnk():
            continue
        total += max(0, int(member.size or 0))
        _reject_oversized(total, member.name)
        member_path = os.path.normcase(os.path.realpath(os.path.join(dest_dir, member.name)))
        if not member_path.startswith(dest_cmp + os.sep) and member_path != dest_cmp:
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
    dest_cmp = os.path.normcase(dest_dir)
    total = 0
    for info in zip_file.infolist():
        member = info.filename
        total += max(0, int(info.file_size or 0))
        _reject_oversized(total, member)
        member_path = os.path.normcase(os.path.realpath(os.path.join(dest_dir, member)))
        if not member_path.startswith(dest_cmp + os.sep) and member_path != dest_cmp:
            raise ValueError(f"Attempted path traversal in zip archive: {member}")
        zip_file.extract(member, dest_dir)
