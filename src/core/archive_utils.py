

from __future__ import annotations

import ntpath
import os
import stat
import tarfile
import zipfile






MAX_EXTRACTED_BYTES = 4 * 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 100_000


def _reject_oversized(total: int, name: str) -> None:

    if total > MAX_EXTRACTED_BYTES:
        raise ValueError(
            f"Archive declares more than {MAX_EXTRACTED_BYTES} bytes uncompressed, "
            f"refusing to extract: {name}"
        )


def _archive_member_path(directory: str, name: str, is_directory: bool) -> str:

    normalized = name.replace("\\", "/")
    if ("\0" in name or normalized.startswith("/") or ntpath.splitdrive(name)[0]
            or any(part == ".." or ":" in part for part in normalized.split("/"))):
        raise ValueError(f"Attempted path traversal in archive: {name}")
    target = os.path.normcase(os.path.realpath(os.path.join(directory, name)))
    root = os.path.normcase(directory)
    if ((target != root and not target.startswith(root + os.sep))
            or (target == root and not is_directory)):
        raise ValueError(f"Attempted path traversal in archive: {name}")
    return target


def safe_extract_tar(tar: tarfile.TarFile, dest_dir: str) -> None:

    dest_dir = os.path.realpath(dest_dir)



    use_filter = hasattr(tarfile, "data_filter")
    total = 0
    members = []
    for number, member in enumerate(tar, 1):
        if number > MAX_ARCHIVE_MEMBERS:
            raise ValueError("Archive contains too many members")
        if member.issym() or member.islnk():
            continue
        if not member.isfile() and not member.isdir():
            raise ValueError(f"Unsupported archive member: {member.name}")
        if member.size < 0:
            raise ValueError(f"Invalid archive member size: {member.name}")
        total += member.size
        _reject_oversized(total, member.name)
        _archive_member_path(dest_dir, member.name, member.isdir())
        members.append(member)

    for member in members:
        _archive_member_path(dest_dir, member.name, member.isdir())
        member.mode &= 0o777
        if use_filter:
            try:
                tar.extract(member, dest_dir, filter="data")
            except (AttributeError, TypeError):


                tar.extract(member, dest_dir)
        else:
            tar.extract(member, dest_dir)


def safe_extract_zip(zip_file: zipfile.ZipFile, dest_dir: str) -> None:

    dest_dir = os.path.realpath(dest_dir)
    total = 0
    members = zip_file.infolist()
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise ValueError("Archive contains too many members")
    for info in members:
        member = info.filename
        mode = info.external_attr >> 16
        if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError(f"Unsupported archive member: {member}")
        if info.flag_bits & 1:
            raise ValueError(f"Encrypted archive member: {member}")
        total += max(0, int(info.file_size or 0))
        _reject_oversized(total, member)
        _archive_member_path(dest_dir, member, info.is_dir())
    for info in members:
        _archive_member_path(dest_dir, info.filename, info.is_dir())
        zip_file.extract(info, dest_dir)
