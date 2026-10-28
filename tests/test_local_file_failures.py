"""Filesystem race and I/O failure regressions for the Linux transfer boundary."""

from __future__ import annotations

import errno
import os
import stat
from functools import partial
from typing import TYPE_CHECKING

import pytest

from llm_sandbox.exceptions import SandboxError, SecurityError
from llm_sandbox.local import files

if TYPE_CHECKING:
    from pathlib import Path


def test_copy_refuses_a_different_regular_source_inode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A replacement regular file is as unsafe as a symlink after the initial stat."""
    source, saved, destination = (tmp_path / name for name in ("source", "saved", "destination"))
    source.write_bytes(b"original")
    opened: list[int] = []
    original_open = os.open

    def replace_before_open(path: str, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if path == source.name:
            source.rename(saved)
            source.write_bytes(b"replacement")
            fd = original_open(path, flags, mode, dir_fd=dir_fd)
            opened.append(fd)
            return fd
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(files.os, "open", replace_before_open)
    with pytest.raises(SecurityError, match="changed while opening"):
        files.safe_copy(source, destination)
    assert saved.read_bytes() == b"original"
    assert not destination.exists()
    assert len(opened) == 1
    with pytest.raises(OSError, match="Bad file descriptor") as error:
        os.fstat(opened[0])
    assert error.value.errno == errno.EBADF


@pytest.mark.parametrize("cleanup", [False, True])
def test_open_permission_failure_preserves_data_and_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup: bool
) -> None:
    """Permission failures propagate without replacing output or discarding a cleanup root."""
    source, destination = tmp_path / "source", tmp_path / "destination"
    if cleanup:
        source.mkdir()
        (source / "sentinel").write_bytes(b"keep")
    else:
        source.write_bytes(b"input")
        destination.write_bytes(b"keep")
    original_open = os.open
    denied = PermissionError(errno.EACCES, "fixture denies open")

    def deny_open(path: str, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if path == source.name:
            raise denied
        return original_open(path, flags, mode, dir_fd=dir_fd)

    operation = partial(files.safe_remove_tree, source) if cleanup else partial(files.safe_copy, source, destination)
    with monkeypatch.context() as patcher:
        patcher.setattr(files.os, "open", deny_open)
        with pytest.raises(PermissionError) as error:
            operation()
        assert error.value is denied
    if cleanup:
        assert (source / "sentinel").read_bytes() == b"keep"
        files.safe_remove_tree(source)
        assert not source.exists()
    else:
        assert destination.read_bytes() == b"keep"
        files.safe_copy(source, destination)
        assert destination.read_bytes() == b"input"


def test_cleanup_refuses_replaced_directory_before_chmod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A new directory at the same name must not inherit private-root permissions or deletion."""
    root, saved = tmp_path / "private-root", tmp_path / "saved-root"
    root.mkdir()
    (root / "sentinel").write_bytes(b"original")
    original_open = os.open

    def replace_before_open(path: str, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if path == root.name and flags & os.O_PATH:
            root.rename(saved)
            root.mkdir()
            root.chmod(0o755)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(files.os, "open", replace_before_open)
    with pytest.raises(SecurityError, match="changed while opening"):
        files.safe_remove_tree(root)
    assert stat.S_IMODE(root.stat().st_mode) == 0o755
    assert (saved / "sentinel").read_bytes() == b"original"


def test_cleanup_refuses_replaced_directory_before_removal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An inode pinned for cleanup does not authorize removing a later directory at its name."""
    root, saved = tmp_path / "private-root", tmp_path / "saved-root"
    root.mkdir()
    original_stat = os.stat
    inspected = 0

    def replace_before_stat(path: str, *, dir_fd: int | None = None, follow_symlinks: bool = True) -> os.stat_result:
        nonlocal inspected
        if path == root.name and dir_fd is not None:
            inspected += 1
            if inspected == 2:
                root.rename(saved)
                root.mkdir()
                (root / "sentinel").write_bytes(b"unrelated")
        return original_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(files.os, "stat", replace_before_stat)
    with pytest.raises(SecurityError, match="changed before removal"):
        files.safe_remove_tree(root)
    assert (root / "sentinel").read_bytes() == b"unrelated"
    assert saved.is_dir()


def test_cleanup_unlinks_flat_entries_without_touching_targets(tmp_path: Path) -> None:
    """Files, links and FIFOs directly in the root need no traversal or target chmod."""
    root, outside = tmp_path / "private-root", tmp_path / "outside"
    root.mkdir()
    outside.write_bytes(b"private")
    outside.chmod(0o400)
    (root / "link").symlink_to(outside)
    (root / "file").write_bytes(b"discard")
    os.mkfifo(root / "pipe")
    files.safe_remove_tree(root)
    assert not root.exists()
    assert outside.read_bytes() == b"private"
    assert stat.S_IMODE(outside.stat().st_mode) == 0o400


def test_read_refuses_growth_after_eof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The final metadata check catches growth after read reported EOF."""
    source = tmp_path / "source"
    source.write_bytes(b"small")
    original_read = os.read

    def grow_after_eof(fd: int, length: int) -> bytes:
        chunk = original_read(fd, length)
        if not chunk:
            with source.open("ab") as stream:
                stream.write(b"now too large")
        return chunk

    monkeypatch.setattr(files.os, "read", grow_after_eof)
    with pytest.raises(SandboxError, match="exceeds"):
        files.safe_read_file(source, 8)


@pytest.mark.parametrize("change", ["create", "remove", "replace"])
def test_copy_refuses_destination_changes_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Atomic publication refuses a created, removed, or different destination inode."""
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"input")
    if change != "create":
        destination.write_bytes(b"old")
    replacement = tmp_path / "replacement"
    if change == "replace":
        # Allocate while the old file exists so the filesystem cannot reuse its inode.
        replacement.write_bytes(b"competing")
    original_write = os.write

    def change_during_write(fd: int, content: bytes) -> int:
        written = original_write(fd, content)
        if change == "replace":
            replacement.replace(destination)
        elif change == "remove":
            destination.unlink()
        else:
            destination.write_bytes(b"competing")
        return written

    monkeypatch.setattr(files.os, "write", change_during_write)
    with pytest.raises(SecurityError, match="Destination changed"):
        files.safe_copy(source, destination)
    if change == "remove":
        assert not destination.exists()
    else:
        assert destination.read_bytes() == b"competing"
    assert not list(tmp_path.glob(".llm-sandbox-copy-*"))


def test_copy_file_cannot_replace_directory(tmp_path: Path) -> None:
    """A destination directory and its retained entries survive a file transfer failure."""
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"input")
    destination.mkdir()
    (destination / "sentinel").write_bytes(b"keep")
    with pytest.raises(SecurityError, match="Cannot replace a directory"):
        files.safe_copy(source, destination)
    assert (destination / "sentinel").read_bytes() == b"keep"
