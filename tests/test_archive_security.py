"""Host-side archive extraction must not trust sandbox metadata."""

import io
import os
import stat
import tarfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from llm_sandbox.core.mixins import FileOperationsMixin
from llm_sandbox.exceptions import SecurityError


@pytest.fixture
def session() -> FileOperationsMixin:
    """Use real file operations with only the container transport stubbed."""
    mixin = FileOperationsMixin()
    mixin.container = object()
    mixin.container_api = Mock()
    mixin.verbose = False
    return mixin


def archive(*members: tarfile.TarInfo) -> bytes:
    """Build archives without creating unsafe source files."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        for member in members:
            tar.addfile(member, io.BytesIO(b"hello") if member.isfile() else None)
    return stream.getvalue()


def regular_file(name: str = "result.txt", mode: int = 0o644) -> tarfile.TarInfo:
    """Create a small regular-file member."""
    member = tarfile.TarInfo(name)
    member.size = 5
    member.mode = mode
    return member


@pytest.mark.parametrize("mode", [0o4777, 0o2777, 0o1777, 0o666, 0o755, 0o644])
def test_copy_from_runtime_sanitizes_permissions(
    session: FileOperationsMixin, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: int
) -> None:
    """Dangerous metadata is removed even if the process default trusts archives."""
    monkeypatch.setattr(tarfile.TarFile, "extraction_filter", staticmethod(tarfile.fully_trusted_filter))
    bits = archive(regular_file(mode=mode))
    session.container_api.copy_from_container.return_value = (bits, {"size": len(bits)})
    dest = tmp_path / "renamed.txt"

    session.copy_from_runtime("/sandbox/result.txt", str(dest))

    assert dest.read_bytes() == b"hello"
    actual_mode = stat.S_IMODE(dest.stat().st_mode)
    assert actual_mode & 0o7022 == 0
    assert actual_mode & 0o600 == 0o600
    if mode == 0o755:
        assert actual_mode == 0o755


@pytest.mark.parametrize("member_type", [tarfile.FIFOTYPE, tarfile.BLKTYPE, tarfile.CHRTYPE, b"Z"])
def test_rejects_special_members_before_writing(
    session: FileOperationsMixin, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, member_type: bytes
) -> None:
    """No FIFO, device, unknown type, or earlier member reaches the filesystem."""

    def unexpected_mknod(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Extraction attempted to create a device")

    monkeypatch.setattr(os, "mknod", unexpected_mknod, raising=False)
    member = tarfile.TarInfo("special")
    member.type = member_type
    dest = tmp_path / "results"

    with pytest.raises(SecurityError, match="Unsupported archive member type"):
        session._extract_archive_safely(archive(regular_file(), member), str(dest))

    assert not dest.exists()


def test_archive_ownership_is_ignored(
    session: FileOperationsMixin, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A privileged extractor never applies the archive's user or group identity."""
    ownership_changes = []
    monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(os, "chown", lambda _path, uid, gid: ownership_changes.append((uid, gid)))
    member = regular_file()
    member.uid, member.gid = 12345, 23456
    member.uname, member.gname = "root", "root"
    dest = tmp_path / "result.txt"

    session._extract_archive_safely(archive(member), str(dest))

    assert dest.read_bytes() == b"hello"
    assert all(change == (-1, -1) for change in ownership_changes)


def test_missing_data_filter_fails_before_writing(
    session: FileOperationsMixin, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Old Python builds cannot silently fall back to fully trusted extraction."""
    monkeypatch.delattr(tarfile, "data_filter")
    dest = tmp_path / "new" / "result.txt"

    with pytest.raises(SecurityError, match=r"Python.*data filter"):
        session._extract_archive_safely(archive(regular_file()), str(dest))

    assert not dest.parent.exists()


def test_directory_copy_preserves_structure_without_unsafe_modes(session: FileOperationsMixin, tmp_path: Path) -> None:
    """Directory output stays usable without restoring attacker permissions."""
    directory = tarfile.TarInfo("output")
    directory.type = tarfile.DIRTYPE
    directory.mode = 0o7777
    dest = tmp_path / "results"

    session._extract_archive_safely(archive(directory, regular_file("output/result.txt", 0o666)), str(dest))

    assert (dest / "output/result.txt").read_bytes() == b"hello"
    assert stat.S_IMODE((dest / "output").stat().st_mode) & 0o7000 == 0
    assert stat.S_IMODE((dest / "output/result.txt").stat().st_mode) & 0o7022 == 0


@pytest.mark.parametrize("member_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_links_remain_excluded(session: FileOperationsMixin, tmp_path: Path, member_type: bytes) -> None:
    """The data filter does not relax the existing no-links policy."""
    link = tarfile.TarInfo("link")
    link.type = member_type
    link.linkname = "result.txt"

    with pytest.raises(FileNotFoundError, match="No safe content"):
        session._extract_archive_safely(archive(link), str(tmp_path / "results"))


@pytest.mark.parametrize("name", ["/absolute", "../escape", "output/../../escape"])
def test_unsafe_paths_remain_excluded(session: FileOperationsMixin, tmp_path: Path, name: str) -> None:
    """Existing absolute-path and traversal protections remain in force."""
    with pytest.raises(FileNotFoundError, match="No safe content"):
        session._extract_archive_safely(archive(regular_file(name)), str(tmp_path / "results"))


def test_existing_destination_symlink_cannot_escape(session: FileOperationsMixin, tmp_path: Path) -> None:
    """Renaming a single result cannot overwrite a file through an existing link."""
    outside = tmp_path / "outside.txt"
    outside.write_text("unchanged")
    dest = tmp_path / "results" / "result.txt"
    dest.parent.mkdir()
    dest.symlink_to(outside)

    with pytest.raises(SecurityError):
        session._extract_archive_safely(archive(regular_file()), str(dest))

    assert outside.read_text() == "unchanged"
