"""Security checks for host-side transfers into local sandbox workspaces."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from llm_sandbox.exceptions import SandboxError, SecurityError
from llm_sandbox.local import files
from llm_sandbox.local.files import runtime_relative, safe_copy, safe_mkdir, safe_read_file


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("hello.py", "hello.py"),
        ("folder/data", "folder/data"),
        ("/sandbox/data", "data"),
        ("/sandbox", "."),
        (".", "."),
    ],
)
def test_runtime_relative(path: str, expected: str) -> None:
    """Absolute runtime paths stay under the virtual workspace."""
    assert runtime_relative(path, "/sandbox") == Path(expected)


@pytest.mark.parametrize(
    "path", ["..", "../secret", "dir/../secret", "/sandbox/../secret", "/sandbox-other/data", "/etc/passwd", "a\x00b"]
)
def test_runtime_relative_rejects_escapes(path: str) -> None:
    """Parent traversal is forbidden even when normalization would stay inside."""
    with pytest.raises(SecurityError):
        runtime_relative(path, "/sandbox")


def test_safe_mkdir_creates_private_parents(tmp_path: Path) -> None:
    """Every new directory uses restrictive permissions."""
    destination = tmp_path / "one" / "two"
    safe_mkdir(destination)
    safe_mkdir(destination)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    assert stat.S_IMODE(destination.parent.stat().st_mode) == 0o700


@pytest.mark.parametrize("suffix", ["", "child"])
def test_safe_mkdir_rejects_links(tmp_path: Path, suffix: str) -> None:
    """Neither a directory leaf nor its parents may be a symlink."""
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(SecurityError):
        safe_mkdir(link / suffix)
    assert list(outside.iterdir()) == []


def test_safe_mkdir_rejects_file_parent(tmp_path: Path) -> None:
    """A regular file in the parent chain is an unsafe path."""
    parent = tmp_path / "file"
    parent.write_text("untouched")
    with pytest.raises(SecurityError):
        safe_mkdir(parent / "child")
    assert parent.read_text() == "untouched"


@pytest.mark.parametrize("mode", [0o666, 0o777])
def test_copy_file_creates_parents_and_private_output(tmp_path: Path, mode: int) -> None:
    """Transfers preserve content and only the owner's executable bit."""
    source = tmp_path / "source"
    source.write_bytes(b"sandbox\x00payload")
    source.chmod(mode)
    destination = tmp_path / "outputs" / "nested" / "result"
    safe_copy(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600 | (mode & stat.S_IXUSR)
    assert stat.S_IMODE(destination.parent.stat().st_mode) == 0o700
    assert list(destination.parent.iterdir()) == [destination]


def test_copy_replaces_file_with_a_new_inode(tmp_path: Path) -> None:
    """Replacing output never writes through an existing destination inode."""
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_text("new")
    destination.write_text("old")
    with destination.open() as previous:
        safe_copy(source, destination)
        assert previous.read() == "old"
    assert destination.read_text() == "new"


def test_copy_directory_merges_at_exact_destination(tmp_path: Path) -> None:
    """Directory copying handles nested and empty directories without adding a basename."""
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "empty").mkdir()
    (source / "nested" / "script").write_text("content")
    (source / "nested" / "script").chmod(0o755)
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "retained").write_text("keep")
    safe_copy(source, destination)
    assert (destination / "nested" / "script").read_text() == "content"
    assert (destination / "empty").is_dir()
    assert (destination / "retained").read_text() == "keep"
    assert stat.S_IMODE((destination / "nested" / "script").stat().st_mode) == 0o700
    assert stat.S_IMODE((destination / "nested").stat().st_mode) == 0o700
    assert stat.S_IMODE(destination.stat().st_mode) == 0o700


def test_missing_source_does_not_create_destination(tmp_path: Path) -> None:
    """Missing sources keep the standard exception and do not create output parents."""
    destination = tmp_path / "missing-parent" / "output"
    with pytest.raises(FileNotFoundError):
        safe_copy(tmp_path / "absent", destination)
    assert not destination.parent.exists()


@pytest.mark.parametrize("side", ["source", "destination"])
@pytest.mark.parametrize("parent", [False, True])
def test_copy_rejects_symlink_leaf_and_parent(tmp_path: Path, side: str, parent: bool) -> None:
    """Symlink rejection applies to both sides and every parent."""
    source = tmp_path / "source"
    source.write_text("input")
    destination = tmp_path / "destination"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_text("private")
    link = tmp_path / "link"
    link.symlink_to(outside if parent else outside / "file", target_is_directory=parent)
    unsafe_path = link / "file" if parent else link
    if side == "source":
        source = unsafe_path
    else:
        destination = unsafe_path
    with pytest.raises(SecurityError):
        safe_copy(source, destination)
    assert (outside / "file").read_text() == "private"


def test_copy_rejects_dangling_destination_symlink(tmp_path: Path) -> None:
    """A dangling symlink is unsafe even though Path.exists would return false."""
    source = tmp_path / "source"
    source.write_text("input")
    destination = tmp_path / "link"
    destination.symlink_to(tmp_path / "absent")
    with pytest.raises(SecurityError):
        safe_copy(source, destination)
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("side", ["source", "destination"])
@pytest.mark.parametrize("unsafe_kind", ["symlink", "fifo", "hardlink"])
def test_copy_rejects_unsafe_directory_children(tmp_path: Path, side: str, unsafe_kind: str) -> None:
    """Both complete trees are checked, including destination-only children."""
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "normal").write_text("input")
    outside = tmp_path / "outside"
    outside.write_text("private")
    child = (source if side == "source" else destination) / "unsafe"
    if unsafe_kind == "symlink":
        child.symlink_to(outside)
    elif unsafe_kind == "hardlink":
        child.hardlink_to(outside)
    else:
        os.mkfifo(child)
    with pytest.raises(SecurityError):
        safe_copy(source, destination)
    assert outside.read_text() == "private"
    assert not (destination / "normal").exists()


@pytest.mark.parametrize("side", ["source", "destination"])
def test_copy_rejects_hardlinked_files(tmp_path: Path, side: str) -> None:
    """An exported or imported hardlink must not expose or alter another file."""
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    outside = tmp_path / "outside"
    outside.write_text("private")
    if side == "source":
        source.hardlink_to(outside)
    else:
        source.write_text("input")
        destination.hardlink_to(outside)
    with pytest.raises(SecurityError):
        safe_copy(source, destination)
    assert outside.read_text() == "private"


@pytest.mark.parametrize("side", ["source", "destination"])
def test_copy_rejects_special_file_leaf(tmp_path: Path, side: str) -> None:
    """FIFO input and output are rejected without opening a blocking stream."""
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    if side == "source":
        os.mkfifo(source)
    else:
        source.write_text("input")
        os.mkfifo(destination)
    with pytest.raises(SecurityError):
        safe_copy(source, destination)


@pytest.mark.parametrize("destination", ["source", "source/child"])
def test_copy_rejects_overlapping_paths(tmp_path: Path, destination: str) -> None:
    """Copying a directory into itself must not recurse or overwrite its inputs."""
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(SecurityError):
        safe_copy(source, tmp_path / destination)


def test_host_paths_reject_traversal_and_relative_paths(tmp_path: Path) -> None:
    """Host path validation never normalizes a parent traversal away."""
    with pytest.raises(SecurityError):
        safe_mkdir(tmp_path / "child" / ".." / "output")
    with pytest.raises(SecurityError):
        safe_mkdir(Path("relative"))


def test_copy_rejects_source_symlink_swap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A source replaced after lstat cannot bypass the O_NOFOLLOW open."""
    source = tmp_path / "source"
    source.write_text("input")
    outside = tmp_path / "outside"
    outside.write_text("private")
    destination = tmp_path / "destination"
    original_open = os.open

    def swap_before_open(path: str, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if path == source.name:
            source.unlink()
            source.symlink_to(outside)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(files.os, "open", swap_before_open)
    with pytest.raises(SecurityError):
        safe_copy(source, destination)
    assert not destination.exists()


def test_copy_rejects_destination_symlink_swap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed output is rejected before publication without touching its target."""
    source = tmp_path / "source"
    source.write_text("input")
    destination = tmp_path / "destination"
    destination.write_text("old")
    outside = tmp_path / "outside"
    outside.write_text("private")
    original_read = os.read

    def swap_during_read(fd: int, length: int) -> bytes:
        chunk = original_read(fd, length)
        if chunk:
            destination.unlink()
            destination.symlink_to(outside)
        return chunk

    monkeypatch.setattr(files.os, "read", swap_during_read)
    with pytest.raises(SecurityError):
        safe_copy(source, destination)
    assert outside.read_text() == "private"
    assert not list(tmp_path.glob(".llm-sandbox-copy-*"))


@pytest.mark.parametrize("content", [b"", b"artifact", b"binary\x00\xff"])
def test_safe_read_file(tmp_path: Path, content: bytes) -> None:
    """Binary artifacts, including empty files and the exact limit, are returned intact."""
    source = tmp_path / "artifact"
    source.write_bytes(content)
    assert safe_read_file(source, max(1, len(content))) == content


@pytest.mark.parametrize("max_bytes", [0, -1])
def test_safe_read_file_requires_positive_limit(tmp_path: Path, max_bytes: int) -> None:
    """A nonpositive limit is a caller error even when the file is empty."""
    source = tmp_path / "artifact"
    source.touch()
    with pytest.raises(ValueError, match="max_bytes must be positive"):
        safe_read_file(source, max_bytes)


@pytest.mark.parametrize("unsafe_kind", ["symlink", "hardlink", "directory", "fifo"])
def test_safe_read_file_rejects_unsafe_leaf(tmp_path: Path, unsafe_kind: str) -> None:
    """Artifacts must be singly linked regular files."""
    source = tmp_path / "artifact"
    outside = tmp_path / "outside"
    outside.write_bytes(b"private")
    if unsafe_kind == "symlink":
        source.symlink_to(outside)
    elif unsafe_kind == "hardlink":
        source.hardlink_to(outside)
    elif unsafe_kind == "directory":
        source.mkdir()
    else:
        os.mkfifo(source)
    with pytest.raises(SecurityError):
        safe_read_file(source, 1024)


def test_safe_read_file_rejects_symlink_parent(tmp_path: Path) -> None:
    """Artifact reads cannot follow a parent symlink to another directory."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "artifact").write_bytes(b"private")
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(SecurityError):
        safe_read_file(link / "artifact", 1024)


def test_safe_read_file_rejects_oversized_before_reading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Known oversized artifacts fail before allocating or reading their contents."""
    source = tmp_path / "artifact"
    source.write_bytes(b"too large")

    def unexpected_read(_fd: int, _length: int) -> bytes:
        pytest.fail("An oversized artifact must not be read")

    monkeypatch.setattr(files.os, "read", unexpected_read)
    with pytest.raises(SandboxError, match="exceeds the 4-byte limit"):
        safe_read_file(source, 4)


def test_safe_read_file_bounds_growing_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Growth after fstat cannot make the reader consume more than the limit plus one."""
    source = tmp_path / "artifact"
    source.write_bytes(b"initial")
    original_read = os.read
    consumed: list[int] = []

    def grow_before_read(fd: int, length: int) -> bytes:
        with source.open("ab") as growing:
            growing.write(b"more data" * 100)
        chunk = original_read(fd, length)
        consumed.append(len(chunk))
        return chunk

    monkeypatch.setattr(files.os, "read", grow_before_read)
    with pytest.raises(SandboxError, match="exceeds the 8-byte limit"):
        safe_read_file(source, 8)
    assert sum(consumed) == 9
