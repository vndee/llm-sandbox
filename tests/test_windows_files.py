"""Windows path policy, handle guards, and genuine native filesystem regressions."""

from __future__ import annotations

import ctypes
import gc
import os
import stat
import subprocess
import weakref
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

from llm_sandbox.exceptions import MissingDependencyError, SandboxError, SecurityError
from llm_sandbox.windows import files

if TYPE_CHECKING:
    from collections.abc import Iterator

_WINDOWS = pytest.mark.skipif(os.name != "nt", reason="Requires native Windows filesystem handles")


@pytest.mark.parametrize("path", [r"C:\data\file.py", "d:/a/b", "C:/", r"C:\Users\user name\é.txt"])
def test_local_drive_paths(path: str) -> None:
    """Only the local drive namespace is accepted, including ordinary spaces and Unicode."""
    assert files.validate_host_path(path) == PureWindowsPath(path)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "relative.txt",
        "C:relative.txt",
        r"\root-relative",
        r"\\server\share\file",
        r"\\?\C:\file",
        r"\\.\C:\file",
        r"\??\C:\file",
        r"\\?\GLOBALROOT\Device\HarddiskVolume1\file",
        r"C:\one\..\file",
        r"C:\file:stream",
        r"C:\file::$DATA",
        r"C:\file.",
        "C:\\file ",
        r"C:\one?two",
        r"C:\one*two",
        r'C:\one"two',
        r"C:\one|two",
        r"C:\one<two",
        r"C:\one>two",
        "C:\\one\x00two",
        "C:\\one\ntwo",
        r"C:\NUL",
        r"C:\aux.txt",
        r"C:\con .txt",
        r"C:\COM1.log",
        r"C:\Lpt9",
        r"C:\COM¹",
        r"C:\LPT².txt",
        r"C:\CONIN$",
        r"C:\CONOUT$",
        r"C:\CLOCK$",
    ],
)
def test_reject_windows_aliases(path: str) -> None:
    """Device aliases, alternate streams, ambiguous names, and traversal fail before I/O."""
    with pytest.raises(SecurityError):
        files.validate_host_path(path)


@pytest.mark.parametrize(
    ("path", "expected"),
    [("file", "file"), ("nested/file", r"nested\file"), (".", "."), (r"c:\WORK\File", "File"), (r"C:\work", ".")],
)
def test_runtime_relative(path: str, expected: str) -> None:
    """Windows containment is component-aware and case-insensitive."""
    assert files.runtime_relative(path, r"C:\work") == PureWindowsPath(expected)


@pytest.mark.parametrize(
    "path",
    [
        r"..\secret",
        r"folder\..\file",
        r"C:\work-other\file",
        r"D:\work\file",
        "C:file",
        r"\file",
        "file:ads",
        "NUL.txt",
    ],
)
def test_runtime_relative_rejects_escape(path: str) -> None:
    """Runtime paths cannot escape the workspace or select a different namespace."""
    with pytest.raises(SecurityError):
        files.runtime_relative(path, r"C:\work")


@pytest.mark.skipif(os.name == "nt", reason="Exercises lazy non-Windows imports")
def test_native_operations_fail_closed_on_other_platforms() -> None:
    """Importing Windows adapters on Linux does not load DLLs or use POSIX path fallbacks."""
    with pytest.raises(MissingDependencyError, match="native Windows"):
        files.PrivateWorkspace()
    with pytest.raises(MissingDependencyError, match="native Windows"):
        files.safe_read_file(Path(r"C:\data"), 100)


def _mock_kernel(monkeypatch: pytest.MonkeyPatch, *, attributes: int = 0, links: int = 1, file_type: int = 1) -> Mock:
    """Supply handle metadata so security decisions run identically on every platform."""
    kernel = Mock()
    kernel.CreateFileW.return_value = 17
    kernel.GetFileType.return_value = file_type

    def get_info(_handle: int, destination: Any) -> int:
        info = ctypes.cast(destination, ctypes.POINTER(files._FileInfo)).contents
        info.attributes, info.links = attributes, links
        info.volume, info.index_high, info.index_low = 3, 4, 5
        return 1

    kernel.GetFileInformationByHandle.side_effect = get_info
    monkeypatch.setattr(files, "_api", lambda: SimpleNamespace(kernel=kernel))
    return kernel


def test_open_pins_without_write_or_delete_sharing(monkeypatch: pytest.MonkeyPatch) -> None:
    """No-follow flags and sharing restrictions protect the opened identity from substitution."""
    kernel = _mock_kernel(monkeypatch)
    with files._open_entry(PureWindowsPath(r"C:\file")) as entry:
        assert entry.identity == (3, 4, 5)
        arguments = kernel.CreateFileW.call_args.args
        assert arguments[0] == r"\\?\C:\file"
        assert arguments[1] & 1  # READ_DATA/LIST_DIRECTORY activates Windows sharing checks.
        assert arguments[2] == 1
        assert arguments[3] is None
        assert arguments[5] & 0x00200000
        assert arguments[5] & 0x02000000
        assert arguments[6] is None
        kernel.CloseHandle.assert_not_called()
    kernel.CloseHandle.assert_called_once_with(17)


@pytest.mark.parametrize(("attributes", "links", "file_type"), [(0x400, 1, 1), (0x410, 1, 1), (0, 2, 1), (0, 1, 3)])
def test_open_rejects_reparse_hardlinks_and_special_handles(
    monkeypatch: pytest.MonkeyPatch, attributes: int, links: int, file_type: int
) -> None:
    """Every rejection closes its acquired handle, including directory junctions and pipes."""
    kernel = _mock_kernel(monkeypatch, attributes=attributes, links=links, file_type=file_type)
    with pytest.raises(SecurityError), files._open_entry(PureWindowsPath(r"C:\file")):
        pytest.fail("Unsafe entry was accepted")
    kernel.CloseHandle.assert_called_once_with(17)


@pytest.mark.parametrize(("attributes", "links"), [(0x410, 1), (0, 2)])
def test_cleanup_opens_link_itself(monkeypatch: pytest.MonkeyPatch, attributes: int, links: int) -> None:
    """Cleanup can unlink rejected entries through no-follow handles without visiting targets."""
    kernel = _mock_kernel(monkeypatch, attributes=attributes, links=links)
    with files._open_entry(PureWindowsPath(r"C:\link"), cleanup=True) as entry:
        assert entry.handle == 17
    assert kernel.CreateFileW.call_args.args[5] & 0x00200000


def test_cleanup_uses_ignore_readonly_disposition(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deleting a readonly hardlink never clears attributes shared with its outside target."""
    kernel = _mock_kernel(monkeypatch)
    seen = []

    def set_info(handle: int, information: int, value: Any, size: int) -> int:
        seen.append((handle, information, ctypes.cast(value, ctypes.POINTER(ctypes.c_uint32))[0], size))
        return 1

    kernel.SetFileInformationByHandle.side_effect = set_info
    files._delete(files._Entry(17, files._FileInfo()))
    assert seen == [(17, 21, 0x13, 4)]
    kernel.SetFileAttributesW.assert_not_called()


@pytest.fixture
def native_root() -> Iterator[Path]:
    """Use the real protected workspace with guaranteed native cleanup for fixture contents."""
    workspace = files.PrivateWorkspace()
    try:
        yield Path(workspace.name)
    finally:
        workspace.cleanup()


@pytest.mark.integration
@_WINDOWS
def test_native_copy_merge_read_and_guarded_write(native_root: Path) -> None:
    """Real CreateFile/ReadFile/WriteFile handles support nested copy and bounded reads."""
    source, destination = native_root / "source", native_root / "destination"
    files.safe_mkdir(source / "empty")
    files.safe_write_file(source / "nested" / "payload", b"first\x00payload")
    files.safe_mkdir(destination)
    files.safe_write_file(destination / "retained", b"keep")
    files.safe_copy(source, destination)
    assert files.safe_read_file(destination / "nested" / "payload", 100) == b"first\x00payload"
    assert (destination / "empty").is_dir()
    assert files.safe_read_file(destination / "retained", 100) == b"keep"
    files.safe_write_file(destination / "nested" / "payload", b"replacement")
    assert files.safe_read_file(destination / "nested" / "payload", 100) == b"replacement"
    with pytest.raises(SandboxError, match="limit"):
        files.safe_read_file(destination / "nested" / "payload", 2)
    with pytest.raises(SecurityError, match="regular file"):
        files.safe_read_file(destination, 100)


@pytest.mark.integration
@_WINDOWS
def test_native_large_and_empty_writes(native_root: Path) -> None:
    """Writes span several native chunks and publish an empty file correctly."""
    path = native_root / "payload"
    content = b"abc" * 60000
    files.safe_write_file(path, content)
    assert files.safe_read_file(path, len(content)) == content
    files.safe_write_file(path, b"")
    assert files.safe_read_file(path, 1) == b""


@pytest.mark.integration
@_WINDOWS
def test_native_hardlinks_rejected_on_both_sides(native_root: Path) -> None:
    """Copying or reading a multiply linked file never exposes or changes its other name."""
    outside = native_root / "outside"
    outside.write_bytes(b"outside")
    source = native_root / "source"
    source.hardlink_to(outside)
    with pytest.raises(SecurityError, match="hardlinked"):
        files.safe_read_file(source, 100)
    with pytest.raises(SecurityError, match="hardlinked"):
        files.safe_copy(source, native_root / "copy")
    with pytest.raises(SecurityError, match="hardlinked"):
        files.safe_write_file(source, b"changed")
    files.validate_existing(source, directory=False, allow_hardlinks=True)
    with pytest.raises(SecurityError, match="hardlinked"):
        files.validate_existing(source, directory=False)
    assert outside.read_bytes() == b"outside"
    assert not (native_root / "copy").exists()


def _junction(link: Path, target: Path) -> None:
    """Create a genuine Windows junction without requiring symlink privileges."""
    command = Path(os.environ["SYSTEMROOT"]) / "System32" / "cmd.exe"
    subprocess.run(  # noqa: S603 # Native Windows command and test-controlled absolute paths.
        [str(command), "/d", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True
    )


@pytest.mark.integration
@_WINDOWS
def test_native_junctions_rejected_and_cleaned_without_following(native_root: Path) -> None:
    """Junction leaves and ancestors cannot carry transfers outside; cleanup only removes the link."""
    outside = native_root / "outside"
    outside.mkdir()
    marker = outside / "marker"
    marker.write_bytes(b"untouched")
    sandbox = native_root / "sandbox"
    files.safe_mkdir(sandbox)
    link = sandbox / "junction"
    _junction(link, outside)
    with pytest.raises(SecurityError, match="Reparse"):
        files.safe_remove_tree(link)
    with pytest.raises(SecurityError, match="Reparse"):
        files.safe_read_file(link / "marker", 100)
    with pytest.raises(SecurityError, match="Reparse"):
        files.safe_mkdir(link / "new")
    with pytest.raises(SecurityError, match="Reparse"):
        files.safe_copy(sandbox, native_root / "copy")
    with pytest.raises(SecurityError, match="Reparse"):
        files.safe_copy(outside, link)
    files.safe_remove_tree(sandbox)
    assert marker.read_bytes() == b"untouched"
    assert list(outside.iterdir()) == [marker]
    assert not sandbox.exists()
    assert not (native_root / "copy").exists()


@pytest.mark.integration
@_WINDOWS
def test_native_cleanup_preserves_outside_readonly_hardlink(native_root: Path) -> None:
    """Deletion ignores readonly flags on the removed name without mutating the outside inode."""
    outside = native_root / "outside"
    outside.write_bytes(b"untouched")
    sandbox = native_root / "sandbox"
    files.safe_mkdir(sandbox / "nested")
    (sandbox / "nested" / "linked").hardlink_to(outside)
    outside.chmod(stat.S_IREAD)
    before = outside.stat().st_file_attributes
    files.safe_remove_tree(sandbox)
    assert outside.read_bytes() == b"untouched"
    assert outside.stat().st_file_attributes == before
    assert not sandbox.exists()


@pytest.mark.integration
@_WINDOWS
def test_native_pinned_ancestors_cannot_be_replaced(native_root: Path) -> None:
    """The actual Windows sharing mode prevents swapping an ancestor after validation."""
    directory = native_root / "directory"
    files.safe_mkdir(directory)
    with files._pin_directory(files.validate_host_path(directory)), pytest.raises(PermissionError):
        directory.rename(native_root / "replaced")
    assert directory.is_dir()
    assert not (native_root / "replaced").exists()


@pytest.mark.integration
@_WINDOWS
def test_native_cleanup_rejects_replaced_root() -> None:
    """A workspace owner cannot accidentally remove an unrelated replacement directory."""
    workspace = files.PrivateWorkspace()
    root = Path(workspace.name)
    moved = root.with_name(root.name + "-moved")
    root.rename(moved)
    root.mkdir()
    marker = root / "marker"
    marker.write_bytes(b"not-owned")
    try:
        with pytest.raises(SecurityError, match="expected directory"):
            workspace.cleanup()
        assert marker.read_bytes() == b"not-owned"
        assert workspace._finalizer.alive
    finally:
        marker.unlink()
        root.rmdir()
        moved.rename(root)
        workspace.cleanup()
    workspace.cleanup()
    assert not root.exists()


@pytest.mark.integration
@_WINDOWS
def test_native_finalizer_cleans_workspace() -> None:
    """Abandoned workspace owners release their private directories via the guarded finalizer."""
    workspace = files.PrivateWorkspace()
    root = Path(workspace.name)
    files.safe_write_file(root / "nested" / "file", b"content")
    reference = weakref.ref(workspace)
    del workspace
    gc.collect()
    assert reference() is None
    assert not root.exists()


@pytest.mark.integration
@_WINDOWS
def test_native_cleanup_deep_long_paths(native_root: Path) -> None:
    """Iterative cleanup handles paths beyond MAX_PATH without Python recursive traversal."""
    tree = native_root / "deep"
    deepest = tree.joinpath(*(["nested"] * 60))
    files.safe_write_file(deepest / "payload", b"content")
    files.safe_remove_tree(tree)
    assert not tree.exists()


def _private_acl(path: Path) -> set[str]:
    """Read the real protected DACL and return the SIDs of its full-access allow ACEs."""
    api = files._api()
    pointer = ctypes.c_void_p
    security = api.security
    security.GetNamedSecurityInfoW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        pointer,
        pointer,
        ctypes.POINTER(pointer),
        pointer,
        ctypes.POINTER(pointer),
    ]
    security.GetNamedSecurityInfoW.restype = ctypes.c_uint32
    security.GetSecurityDescriptorControl.argtypes = [
        pointer,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    security.GetSecurityDescriptorControl.restype = ctypes.c_int32
    security.GetAce.argtypes = [pointer, ctypes.c_uint32, ctypes.POINTER(pointer)]
    security.GetAce.restype = ctypes.c_int32
    descriptor, dacl = pointer(), pointer()
    assert (
        security.GetNamedSecurityInfoW(str(path), 1, 4, None, None, ctypes.byref(dacl), None, ctypes.byref(descriptor))
        == 0
    )
    try:
        control, revision = ctypes.c_uint16(), ctypes.c_uint32()
        assert security.GetSecurityDescriptorControl(descriptor, ctypes.byref(control), ctypes.byref(revision))
        assert control.value & 0x1000  # SE_DACL_PROTECTED: no broader permissions inherited from the temp parent.
        assert dacl.value is not None
        count = ctypes.c_uint16.from_address(dacl.value + 4).value
        assert count == 2
        sids = set()
        for index in range(count):
            ace = pointer()
            assert security.GetAce(dacl, index, ctypes.byref(ace))
            assert ace.value is not None
            assert ctypes.c_uint8.from_address(ace.value).value == 0  # ACCESS_ALLOWED_ACE_TYPE.
            assert ctypes.c_uint32.from_address(ace.value + 4).value == 0x1F01FF  # FILE_ALL_ACCESS.
            text = pointer()
            assert security.ConvertSidToStringSidW(ace.value + 8, ctypes.byref(text))
            try:
                sids.add(ctypes.wstring_at(text))
            finally:
                api.kernel.LocalFree(text)
        return sids
    finally:
        api.kernel.LocalFree(descriptor)


@pytest.mark.integration
@_WINDOWS
def test_native_new_entries_have_protected_private_acl(native_root: Path) -> None:
    """Workspace roots and transferred children grant access only to the owner and SYSTEM."""
    nested = native_root / "nested"
    artifact = nested / "artifact"
    files.safe_write_file(artifact, b"private")
    expected = {"S-1-5-18", files._current_user_sid()}
    for path in (native_root, nested, artifact):
        assert _private_acl(path) == expected


def test_trusted_hardlink_opt_in_never_allows_reparse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Trusted executable validation permits Windows store hardlinks but still refuses junctions."""
    _mock_kernel(monkeypatch, links=2)
    with files._open_entry(PureWindowsPath(r"C:\Windows\System32\cmd.exe"), allow_hardlinks=True) as entry:
        assert entry.info.links == 2
    _mock_kernel(monkeypatch, attributes=0x400)
    with pytest.raises(SecurityError), files._open_entry(PureWindowsPath(r"C:\link"), allow_hardlinks=True):
        pytest.fail("Trusted executable opt-in followed a reparse point")


@pytest.mark.parametrize(
    ("source", "destination", "aliases"),
    [
        (r"C:\source-long", r"C:\SOURCE~1\new", {r"C:\source-long": 2, r"C:\SOURCE~1": 2}),
        (
            r"C:\DESTIN~1\source",
            r"C:\destination-long",
            {r"C:\DESTIN~1": 2, r"C:\destination-long": 2, r"C:\DESTIN~1\source": 3},
        ),
        (r"C:\long-file", r"C:\LONG-F~1", {r"C:\long-file": 2, r"C:\LONG-F~1": 2}),
        ("C:\\", r"D:\output", {"D:\\": 1}),
    ],
)
def test_native_identities_reject_alias_overlap(
    monkeypatch: pytest.MonkeyPatch, source: str, destination: str, aliases: dict[str, int]
) -> None:
    """Short-name aliases cannot turn a copy into an endless self-copy or replace its own source."""
    identities = {PureWindowsPath("C:/"): 1, **{PureWindowsPath(name): identity for name, identity in aliases.items()}}

    @contextmanager
    def open_entry(path: PureWindowsPath, **_options: Any) -> Iterator[files._Entry]:
        if path not in identities:
            raise FileNotFoundError(str(path))
        info = files._FileInfo(attributes=0x10, links=1, volume=1, index_low=identities[path])
        yield files._Entry(identities[path], info)

    monkeypatch.setattr(files, "_open_entry", open_entry)
    walk = Mock(side_effect=AssertionError("A copy must reject aliases before traversing or writing"))
    monkeypatch.setattr(files, "_tree_paths", walk)
    with pytest.raises(SecurityError, match="overlap"):
        files.safe_copy(Path(source), Path(destination))
    walk.assert_not_called()


def _directory_write_handle(path: Path) -> int:
    """Try the access required by FSCTL_SET_REPARSE_POINT against an existing directory."""
    return files._api().kernel.CreateFileW(
        files._native_path(files.validate_host_path(path)), 0x40000000, 7, None, 3, 0x00200000 | 0x02000000, None
    )


@pytest.mark.integration
@_WINDOWS
def test_native_pins_exclude_reparse_writers(native_root: Path) -> None:
    """The actual directory lock rejects both new and already-open reparse mutation handles."""
    directory = native_root / "directory"
    files.safe_mkdir(directory)
    invalid = ctypes.c_void_p(-1).value
    with files._pin_directory(files.validate_host_path(directory)):
        handle = _directory_write_handle(directory)
        if handle != invalid:
            files._api().kernel.CloseHandle(handle)
            pytest.fail("A directory reparse writer opened while a strong pin was held")
        assert files._windows_error().winerror == 32  # ERROR_SHARING_VIOLATION.
    handle = _directory_write_handle(directory)
    assert handle != invalid
    try:
        with pytest.raises(PermissionError) as error, files._pin_directory(files.validate_host_path(directory)):
            pytest.fail("Strong pin ignored a pre-existing directory writer")
        assert error.value.winerror == 32
    finally:
        files._api().kernel.CloseHandle(handle)


@pytest.mark.integration
@_WINDOWS
def test_native_failed_replacement_removes_partial_file(native_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A write failure cannot leave the partial replacement, and never mutates other files."""
    target, retained = native_root / "target", native_root / "retained"
    files.safe_write_file(target, b"old")
    files.safe_write_file(retained, b"untouched")
    write_all = files._write_all

    def fail_after_partial(entry: files._Entry, _chunks: Any) -> None:
        write_all(entry, [b"partial"])
        message = "injected write failure"
        raise OSError(message)

    monkeypatch.setattr(files, "_write_all", fail_after_partial)
    with pytest.raises(OSError, match="injected write failure"):
        files.safe_write_file(target, b"replacement")
    assert not target.exists()
    assert retained.read_bytes() == b"untouched"


@pytest.mark.integration
@_WINDOWS
def test_native_competing_hardlink_cannot_receive_output(native_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Strong parent pins block hardlink insertion before the guarded output open."""
    target, outside = native_root / "target", native_root / "outside"
    files.safe_write_file(target, b"old")
    files.safe_write_file(outside, b"untouched")
    open_entry = files._open_entry

    @contextmanager
    def insert_link(path: PureWindowsPath, **options: Any) -> Iterator[files._Entry]:
        if options.get("create") and path == files.validate_host_path(target):
            target.hardlink_to(outside)
        with open_entry(path, **options) as entry:
            yield entry

    monkeypatch.setattr(files, "_open_entry", insert_link)
    with pytest.raises(PermissionError) as error:
        files.safe_write_file(target, b"replacement")
    assert error.value.winerror == 32  # The parent pin blocks the hardlink's directory-write open.
    assert outside.read_bytes() == b"untouched"
    assert not target.exists()


@pytest.mark.integration
@_WINDOWS
def test_native_competing_file_is_never_overwritten(native_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CREATE_NEW refuses an ordinary file inserted after removing the previous output."""
    target = native_root / "target"
    files.safe_write_file(target, b"old")
    open_entry = files._open_entry

    @contextmanager
    def insert_file(path: PureWindowsPath, **options: Any) -> Iterator[files._Entry]:
        if options.get("create") and path == files.validate_host_path(target):
            target.write_bytes(b"competing file")
        with open_entry(path, **options) as entry:
            yield entry

    monkeypatch.setattr(files, "_open_entry", insert_file)
    with pytest.raises(FileExistsError):
        files.safe_write_file(target, b"replacement")
    assert target.read_bytes() == b"competing file"


@pytest.mark.skipif(os.name == "nt", reason="Checks the native library loader guard")
def test_direct_library_loader_rejects_non_windows() -> None:
    """Calling the loader directly cannot bypass the native-host requirement."""
    with pytest.raises(MissingDependencyError, match="native Windows"):
        files._windows_library("kernel32")


@pytest.mark.parametrize("max_bytes", [0, -1])
def test_read_requires_a_positive_byte_budget(max_bytes: int) -> None:
    """An ineffective artifact budget is refused before opening any host file."""
    with pytest.raises(ValueError, match="positive"):
        files.safe_read_file(Path(r"C:\artifact"), max_bytes)


@pytest.mark.parametrize(
    ("source", "destination"),
    [(r"C:\tree", r"c:\TREE"), (r"C:\tree", r"C:\tree\child"), (r"C:\tree\child", r"C:\tree")],
)
def test_copy_rejects_textual_overlap_without_io(source: str, destination: str) -> None:
    """Same paths and ancestor relationships cannot delete the input or recurse into output."""
    with pytest.raises(SecurityError, match="overlap"):
        files.safe_copy(Path(source), Path(destination))


def test_cleanup_cannot_remove_a_drive_root() -> None:
    """Even an otherwise valid host path cannot authorize volume-root cleanup."""
    with pytest.raises(SecurityError, match="drive root"):
        files.safe_remove_tree(Path("C:\\"))


@pytest.mark.parametrize("prefix", ["", ".", r"nested\prefix", "nested/prefix"])
def test_private_workspace_prefix_cannot_select_another_directory(monkeypatch: pytest.MonkeyPatch, prefix: str) -> None:
    """Workspace names cannot become empty names, current directories, or nested paths."""
    monkeypatch.setattr(files, "_api", Mock())
    creator = Mock()
    monkeypatch.setattr(files, "_create_directory", creator)
    with pytest.raises(SecurityError, match="single Windows filename"):
        files.PrivateWorkspace(prefix=prefix)
    creator.assert_not_called()


@pytest.mark.parametrize("failure", ["query_size", "read_token", "convert_sid"])
def test_token_query_failure_closes_acquired_token(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """Failures while constructing a private ACL release the token without using a guessed SID."""
    kernel, security = Mock(), Mock()
    kernel.GetCurrentProcess.return_value = 1
    expected = PermissionError("injected token query failure")

    def open_token(_process: int, _access: int, destination: Any) -> int:
        ctypes.cast(destination, ctypes.POINTER(ctypes.c_void_p))[0] = 77
        return 1

    def query_token(_token: Any, _kind: int, buffer: Any, _size: Any, size_out: Any) -> int:
        if buffer is None:
            ctypes.cast(size_out, ctypes.POINTER(ctypes.c_uint32))[0] = 0 if failure == "query_size" else 16
            return 0
        if failure == "read_token":
            return 0
        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0] = 123
        return 1

    security.OpenProcessToken.side_effect = open_token
    security.GetTokenInformation.side_effect = query_token
    security.ConvertSidToStringSidW.return_value = 0
    monkeypatch.setattr(files, "_api", lambda: SimpleNamespace(kernel=kernel, security=security))
    monkeypatch.setattr(files, "_windows_error", lambda: expected)
    with pytest.raises(PermissionError, match="token query") as caught:
        files._current_user_sid()
    assert caught.value is expected
    assert kernel.CloseHandle.call_args.args[0].value == 77
    kernel.LocalFree.assert_not_called()
    if failure != "convert_sid":
        security.ConvertSidToStringSidW.assert_not_called()


def test_private_acl_failure_never_creates_unprotected_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Failure to build the protected DACL prevents creation rather than falling back to inheritance."""
    kernel, security = Mock(), Mock()
    security.ConvertStringSecurityDescriptorToSecurityDescriptorW.return_value = 0
    monkeypatch.setattr(files, "_api", lambda: SimpleNamespace(kernel=kernel, security=security))
    monkeypatch.setattr(files, "_current_user_sid", lambda: "S-1-5-21-1-2-3-1000")
    expected = PermissionError("injected ACL conversion failure")
    monkeypatch.setattr(files, "_windows_error", lambda: expected)
    with pytest.raises(PermissionError, match="ACL conversion"):
        files._create_directory(PureWindowsPath(r"C:\private"))
    kernel.CreateDirectoryW.assert_not_called()


def test_metadata_query_failure_releases_open_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """A handle whose identity cannot be checked is closed and never exposed to callers."""
    kernel = _mock_kernel(monkeypatch)
    kernel.GetFileInformationByHandle.side_effect = None
    kernel.GetFileInformationByHandle.return_value = 0
    with pytest.raises(OSError), files._open_entry(PureWindowsPath(r"C:\unverified")):  # noqa: PT011
        pytest.fail("An unverified handle was returned")
    kernel.CloseHandle.assert_called_once_with(17)


@pytest.mark.parametrize("progress", [0, 2])
def test_native_write_progress_is_required_and_partial_writes_are_completed(
    monkeypatch: pytest.MonkeyPatch, progress: int
) -> None:
    """A short WriteFile result cannot lose bytes, and a zero-byte result cannot spin indefinitely."""
    kernel = _mock_kernel(monkeypatch)
    received = bytearray()

    def write_file(_handle: int, data: bytes, requested: int, count: Any, _overlapped: Any) -> int:
        written = min(progress, requested)
        received.extend(data[:written])
        ctypes.cast(count, ctypes.POINTER(ctypes.c_uint32))[0] = written
        return 1

    kernel.WriteFile.side_effect = write_file
    entry = files._Entry(17, files._FileInfo())
    if not progress:
        with pytest.raises(OSError, match="no progress"):
            files._write_all(entry, [b"payload"])
        assert not received
    else:
        files._write_all(entry, [b"first", b"", b"second"])
        assert received == b"firstsecond"


def test_read_budget_is_enforced_beyond_initial_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Extra data returned after the size check still cannot escape the artifact byte limit."""
    kernel = _mock_kernel(monkeypatch)
    chunks = iter([b"abc", b"def"])

    @contextmanager
    def pinned_parent(_path: PureWindowsPath) -> Iterator[None]:
        yield None

    def read_file(_handle: int, buffer: Any, _requested: int, count: Any, _overlapped: Any) -> int:
        data = next(chunks)
        ctypes.memmove(buffer, data, len(data))
        ctypes.cast(count, ctypes.POINTER(ctypes.c_uint32))[0] = len(data)
        return 1

    kernel.ReadFile.side_effect = read_file
    monkeypatch.setattr(files, "_pin_directory", pinned_parent)
    with pytest.raises(SandboxError, match="4-byte limit"):
        files.safe_read_file(Path(r"C:\artifact"), 4)
    kernel.CloseHandle.assert_called_once_with(17)


@pytest.mark.integration
@_WINDOWS
def test_native_file_parent_and_directory_output_refused(native_root: Path) -> None:
    """Transfers never traverse a regular-file parent or replace a directory with file bytes."""
    parent = native_root / "file-parent"
    parent.write_bytes(b"untouched")
    with pytest.raises(SecurityError, match="Expected a directory"):
        files.safe_mkdir(parent / "child")
    with pytest.raises(SecurityError, match="expected directory"):
        files.safe_remove_tree(parent)
    directory = native_root / "directory-output"
    files.safe_mkdir(directory)
    with pytest.raises(SecurityError, match="replace a directory"):
        files.safe_write_file(directory, b"replacement")
    assert parent.read_bytes() == b"untouched"
    assert directory.is_dir()
    assert list(directory.iterdir()) == []


@pytest.mark.integration
@_WINDOWS
@pytest.mark.parametrize("is_directory", [False, True])
def test_native_existing_type_must_match_requested_type(native_root: Path, is_directory: bool) -> None:
    """Trusted validation still rejects an entry of the wrong requested kind."""
    path = native_root / "entry"
    if is_directory:
        files.safe_mkdir(path)
    else:
        files.safe_write_file(path, b"untouched")
    with pytest.raises(SecurityError, match="Unexpected filesystem entry type"):
        files.validate_existing(path, directory=not is_directory)
    files.validate_existing(path, directory=is_directory)
    assert path.is_dir() == is_directory


@pytest.mark.integration
@_WINDOWS
def test_native_destination_tree_links_refused_before_copy(native_root: Path) -> None:
    """Merging cannot retain an unsafe junction already present in the destination tree."""
    source, destination, outside = (native_root / name for name in ("source", "destination", "outside"))
    files.safe_write_file(source / "new", b"source")
    files.safe_write_file(outside / "marker", b"untouched")
    files.safe_mkdir(destination)
    _junction(destination / "retained-link", outside)
    with pytest.raises(SecurityError, match="Reparse"):
        files.safe_copy(source, destination)
    assert not (destination / "new").exists()
    assert (outside / "marker").read_bytes() == b"untouched"


@pytest.mark.integration
@_WINDOWS
def test_native_ambiguous_child_name_refused_before_copy(native_root: Path) -> None:
    """Literal NTFS trailing-dot entries cannot become different Win32 destination names."""
    source, destination = native_root / "source", native_root / "destination"
    files.safe_mkdir(source)
    ambiguous = Path(files._native_path(PureWindowsPath(str(source / "trailing."))))
    ambiguous.write_bytes(b"payload")
    with pytest.raises(SecurityError, match="Unsafe Windows path component"):
        files.safe_copy(source, destination)
    assert ambiguous.read_bytes() == b"payload"
    assert not destination.exists()


@pytest.mark.integration
@_WINDOWS
def test_native_cleanup_rechecks_identity_before_final_unlink(
    native_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replacement after walking the root is preserved by the final handle identity check."""
    root, moved = native_root / "root", native_root / "moved"
    files.safe_write_file(root / "payload", b"content")
    remove_entry = files._remove_entry

    def replace_before_unlink(path: PureWindowsPath, expected: tuple[int, int, int] | None = None) -> None:
        if path == files.validate_host_path(root) and expected is not None:
            root.rename(moved)
            root.mkdir()
            (root / "marker").write_bytes(b"not-owned")
        remove_entry(path, expected)

    with monkeypatch.context() as patch:
        patch.setattr(files, "_remove_entry", replace_before_unlink)
        with pytest.raises(SecurityError, match="replaced before cleanup"):
            files.safe_remove_tree(root)
    assert (root / "marker").read_bytes() == b"not-owned"
    assert moved.is_dir()
    assert not (moved / "payload").exists()


@pytest.mark.integration
@_WINDOWS
def test_native_workspace_name_collision_preserves_existing_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    """A random-name collision retries creation without adopting or deleting the existing owner."""
    prefix = "llm-sandbox-collision-"
    original = files.PrivateWorkspace(prefix=prefix)
    existing = Path(original.name)
    marker = existing / "marker"
    files.safe_write_file(marker, b"untouched")
    fresh_token = files.secrets.token_hex(16)
    tokens = Mock(side_effect=[existing.name[len(prefix) :], fresh_token])
    replacement = None
    try:
        with monkeypatch.context() as patch:
            patch.setattr(files.secrets, "token_hex", tokens)
            replacement = files.PrivateWorkspace(prefix=prefix)
        assert replacement.name != original.name
        assert marker.read_bytes() == b"untouched"
        assert tokens.call_count == 2
        replacement.cleanup()
        assert marker.read_bytes() == b"untouched"
    finally:
        if replacement is not None:
            replacement.cleanup()
        original.cleanup()


@pytest.mark.parametrize("invalid", [None, ctypes.c_void_p(-1).value])
def test_invalid_native_handle_never_reaches_metadata_or_close(
    monkeypatch: pytest.MonkeyPatch, invalid: int | None
) -> None:
    """A failed native open cannot become a usable entry or an invalid CloseHandle call."""
    kernel = _mock_kernel(monkeypatch)
    kernel.CreateFileW.return_value = invalid
    with pytest.raises(OSError), files._open_entry(PureWindowsPath(r"C:\missing")):  # noqa: PT011
        pytest.fail("An invalid native handle was returned")
    kernel.GetFileInformationByHandle.assert_not_called()
    kernel.CloseHandle.assert_not_called()


@pytest.mark.integration
@_WINDOWS
def test_native_partial_cleanup_failure_keeps_original_write_error(
    native_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup failure never masks the failed transfer or writes through a different host object."""
    target, retained = native_root / "target", native_root / "retained"
    files.safe_write_file(target, b"old")
    files.safe_write_file(retained, b"untouched")
    write_all = files._write_all
    original = OSError("injected original write failure")
    cleanup_error = PermissionError("injected cleanup refusal")
    with monkeypatch.context() as patch:

        def fail_after_partial(entry: files._Entry, _chunks: Any) -> None:
            write_all(entry, [b"partial"])
            patch.setattr(files, "_delete", Mock(side_effect=cleanup_error))
            raise original

        patch.setattr(files, "_write_all", fail_after_partial)
        with pytest.raises(OSError, match="original write failure") as caught:
            files.safe_write_file(target, b"replacement")
    assert caught.value is original
    assert target.read_bytes() == b"partial"
    assert _private_acl(target) == {"S-1-5-18", files._current_user_sid()}
    assert retained.read_bytes() == b"untouched"
