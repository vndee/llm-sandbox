"""Windows file transfers with pinned paths and explicit private-directory ACLs.

All filesystem operations require Windows and run only after sandbox processes
have stopped. Ancestor handles deny write/delete sharing, preventing directory
replacement or reparse-point changes while an operation follows the path. Reads
and copies reject every reparse point and multiply linked file. Cleanup removes
links themselves through handles and never modifies a linked target's attributes.
"""

from __future__ import annotations

import ctypes
import os
import re
import secrets
import sys
import tempfile
import weakref
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from functools import cache
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any

from llm_sandbox.exceptions import MissingDependencyError, SandboxError, SecurityError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

_DWORD = ctypes.c_uint32
_HANDLE = ctypes.c_void_p
_BOOL = ctypes.c_int32
_READ_ATTRIBUTES = 0x80
_READ_DATA = 0x01  # FILE_LIST_DIRECTORY for directories; participates in sharing checks.
_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_DELETE = 0x10000
_SHARE_READ = 1
_OPEN_EXISTING = 3
_CREATE_NEW = 1
_OPEN_REPARSE_POINT = 0x00200000
_BACKUP_SEMANTICS = 0x02000000
_DIRECTORY = 0x10
_REPARSE_POINT = 0x400
_FILE_TYPE_DISK = 1
_FILE_DISPOSITION_INFO_EX = 21
_DISPOSITION_DELETE = 0x01
_DISPOSITION_POSIX_SEMANTICS = 0x02
_DISPOSITION_IGNORE_READONLY = 0x10
_CHUNK_SIZE = 65536
_FIRST_PRINTABLE_CHARACTER = 32
_RESERVED_NAMES = frozenset({"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$", "CLOCK$"})
_DEVICE_NUMBER = re.compile(r"(?:COM|LPT)[1-9¹²³]\Z", re.IGNORECASE)
_NATIVE_WINDOWS_REQUIRED = "Windows filesystem helpers require a native Windows host"


def _validate_components(path: PureWindowsPath) -> None:
    """Reject Win32 aliases, alternate streams, and ambiguous path components."""
    components = path.parts[1:] if path.anchor else path.parts
    for component in components:
        basename = component.split(".", 1)[0].rstrip(" ").upper()
        if (
            component == ".."
            or component.endswith((".", " "))
            or any(ord(character) < _FIRST_PRINTABLE_CHARACTER or character in '<>:"|?*' for character in component)
            or basename in _RESERVED_NAMES
            or _DEVICE_NUMBER.fullmatch(basename)
        ):
            message = f"Unsafe Windows path component: {component!r}"
            raise SecurityError(message)


def validate_host_path(path: str | Path) -> PureWindowsPath:
    """Accept only absolute local drive paths, excluding UNC and device namespaces."""
    parsed = PureWindowsPath(str(path))
    if not parsed.is_absolute() or not re.fullmatch(r"[A-Za-z]:", parsed.drive):
        message = "Windows host paths must be absolute local drive paths, without UNC or device prefixes"
        raise SecurityError(message)
    _validate_components(parsed)
    return parsed


def runtime_relative(path: str, workdir: str) -> PureWindowsPath:
    """Map a Windows runtime path inside the workspace without accepting traversal or ADS."""
    root = validate_host_path(workdir)
    candidate = PureWindowsPath(path)
    _validate_components(candidate)
    if candidate.is_absolute():
        candidate = validate_host_path(path)
        try:
            return candidate.relative_to(root)
        except ValueError as exc:
            message = f"Runtime path must be inside {workdir}: {path}"
            raise SecurityError(message) from exc
    if candidate.drive or candidate.root:
        message = "Runtime paths cannot use drive-relative or root-relative Windows syntax"
        raise SecurityError(message)
    return candidate


class _FileInfo(ctypes.Structure):
    _fields_ = (
        ("attributes", _DWORD),
        ("creation", _DWORD * 2),
        ("access", _DWORD * 2),
        ("write", _DWORD * 2),
        ("volume", _DWORD),
        ("size_high", _DWORD),
        ("size_low", _DWORD),
        ("links", _DWORD),
        ("index_high", _DWORD),
        ("index_low", _DWORD),
    )


class _SecurityAttributes(ctypes.Structure):
    _fields_ = (("length", _DWORD), ("descriptor", ctypes.c_void_p), ("inherit", _BOOL))


def _windows_library(name: str) -> ctypes.CDLL:
    if sys.platform != "win32":
        raise MissingDependencyError(_NATIVE_WINDOWS_REQUIRED)
    return ctypes.WinDLL(name, use_last_error=True)


def _windows_error() -> OSError:
    if sys.platform != "win32":
        return OSError(_NATIVE_WINDOWS_REQUIRED)
    return ctypes.WinError(ctypes.get_last_error())


class _WindowsAPI:
    """Load the narrow Win32 API surface without affecting non-Windows imports."""

    def __init__(self) -> None:
        self.kernel = _windows_library("kernel32")
        self.security = _windows_library("advapi32")
        pointer = ctypes.c_void_p
        bindings: tuple[tuple[str, list[Any], Any], ...] = (
            ("CreateFileW", [ctypes.c_wchar_p, _DWORD, _DWORD, pointer, _DWORD, _DWORD, _HANDLE], _HANDLE),
            ("CreateDirectoryW", [ctypes.c_wchar_p, pointer], _BOOL),
            ("CloseHandle", [_HANDLE], _BOOL),
            ("GetFileType", [_HANDLE], _DWORD),
            ("GetFileInformationByHandle", [_HANDLE, ctypes.POINTER(_FileInfo)], _BOOL),
            ("SetFileInformationByHandle", [_HANDLE, _DWORD, pointer, _DWORD], _BOOL),
            ("ReadFile", [_HANDLE, pointer, _DWORD, ctypes.POINTER(_DWORD), pointer], _BOOL),
            ("WriteFile", [_HANDLE, pointer, _DWORD, ctypes.POINTER(_DWORD), pointer], _BOOL),
            ("GetCurrentProcess", [], _HANDLE),
            ("LocalFree", [pointer], pointer),
        )
        for name, arguments, result in bindings:
            function = getattr(self.kernel, name)
            function.argtypes, function.restype = arguments, result
        security_bindings = (
            ("OpenProcessToken", [_HANDLE, _DWORD, ctypes.POINTER(_HANDLE)]),
            ("GetTokenInformation", [_HANDLE, _DWORD, pointer, _DWORD, ctypes.POINTER(_DWORD)]),
            ("ConvertSidToStringSidW", [pointer, ctypes.POINTER(pointer)]),
            (
                "ConvertStringSecurityDescriptorToSecurityDescriptorW",
                [ctypes.c_wchar_p, _DWORD, ctypes.POINTER(pointer), ctypes.POINTER(_DWORD)],
            ),
        )
        for name, arguments in security_bindings:
            function = getattr(self.security, name)
            function.argtypes, function.restype = arguments, _BOOL


@cache
def _api() -> _WindowsAPI:
    if os.name != "nt":
        raise MissingDependencyError(_NATIVE_WINDOWS_REQUIRED)
    return _WindowsAPI()


def _check(result: Any) -> None:
    if not result:
        raise _windows_error()


def _native_path(path: PureWindowsPath) -> str:
    """Add the long-path prefix internally, after rejecting user-supplied namespaces."""
    return "\\\\?\\" + str(path)


def _current_user_sid() -> str:
    api = _api()
    token = _HANDLE()
    _check(api.security.OpenProcessToken(api.kernel.GetCurrentProcess(), 8, ctypes.byref(token)))
    try:
        size = _DWORD()
        api.security.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))
        if not size.value:
            raise _windows_error()
        buffer = ctypes.create_string_buffer(size.value)
        _check(api.security.GetTokenInformation(token, 1, buffer, size, ctypes.byref(size)))
        sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
        text = ctypes.c_void_p()
        _check(api.security.ConvertSidToStringSidW(sid, ctypes.byref(text)))
        try:
            return ctypes.wstring_at(text)
        finally:
            api.kernel.LocalFree(text)
    finally:
        api.kernel.CloseHandle(token)


@contextmanager
def _private_security() -> Iterator[_SecurityAttributes]:
    """Grant the current user and SYSTEM full access with protected, inheritable ACEs."""
    api = _api()
    descriptor = ctypes.c_void_p()
    sddl = f"D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;{_current_user_sid()})"
    _check(api.security.ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, 1, ctypes.byref(descriptor), None))
    try:
        yield _SecurityAttributes(length=ctypes.sizeof(_SecurityAttributes), descriptor=descriptor, inherit=False)
    finally:
        api.kernel.LocalFree(descriptor)


@dataclass(frozen=True)
class _Entry:
    handle: int
    info: _FileInfo

    @property
    def identity(self) -> tuple[int, int, int]:
        return int(self.info.volume), int(self.info.index_high), int(self.info.index_low)

    @property
    def directory(self) -> bool:
        return bool(self.info.attributes & _DIRECTORY)

    @property
    def reparse(self) -> bool:
        return bool(self.info.attributes & _REPARSE_POINT)


@contextmanager
def _open_entry(
    path: PureWindowsPath,
    *,
    access: int = _READ_ATTRIBUTES | _READ_DATA,
    create: bool = False,
    cleanup: bool = False,
    allow_hardlinks: bool = False,
) -> Iterator[_Entry]:
    api = _api()
    with ExitStack() as stack:
        security = stack.enter_context(_private_security()) if create else None
        handle = api.kernel.CreateFileW(
            _native_path(path),
            access,
            _SHARE_READ,
            ctypes.byref(security) if security is not None else None,
            _CREATE_NEW if create else _OPEN_EXISTING,
            _OPEN_REPARSE_POINT | _BACKUP_SEMANTICS,
            None,
        )
        if handle is None or handle == ctypes.c_void_p(-1).value:
            raise _windows_error()
        stack.callback(api.kernel.CloseHandle, handle)
        info = _FileInfo()
        _check(api.kernel.GetFileInformationByHandle(handle, ctypes.byref(info)))
        entry = _Entry(int(handle), info)
        if api.kernel.GetFileType(handle) != _FILE_TYPE_DISK:
            message = f"Only disk files and directories are supported: {path}"
            raise SecurityError(message)
        if not cleanup and (entry.reparse or (not allow_hardlinks and not entry.directory and info.links != 1)):
            message = f"Reparse points and hardlinked files are not permitted: {path}"
            raise SecurityError(message)
        yield entry


def _create_directory(path: PureWindowsPath) -> None:
    with _private_security() as security:
        _check(_api().kernel.CreateDirectoryW(_native_path(path), ctypes.byref(security)))


@contextmanager
def _pin_directory(
    path: PureWindowsPath, *, create: bool = False, forbidden_identity: tuple[int, int, int] | None = None
) -> Iterator[_Entry]:
    """Keep every ancestor open without write/delete sharing until the operation ends."""
    with ExitStack() as stack:
        current = PureWindowsPath(path.anchor)
        entry = stack.enter_context(_open_entry(current))
        _reject_identity(entry, forbidden_identity)
        for component in path.parts[1:]:
            current /= component
            if create:
                with suppress(FileExistsError):
                    _create_directory(current)
            entry = stack.enter_context(_open_entry(current))
            _reject_identity(entry, forbidden_identity)
            if not entry.directory:
                message = f"Expected a directory: {current}"
                raise SecurityError(message)
        yield entry


def safe_mkdir(path: Path) -> None:
    """Create private directories while refusing reparse points in all components."""
    with _pin_directory(validate_host_path(path), create=True):
        # Entering the context creates and pins every required directory.
        pass


def validate_existing(path: Path, *, directory: bool | None = None, allow_hardlinks: bool = False) -> None:
    """Validate an existing host path through pinned no-reparse handles.

    Trusted OS executables may opt into hardlinks (Windows component-store
    binaries use them). Untrusted transfer/read operations never opt into this.
    """
    parsed = validate_host_path(path)
    with _pin_directory(parsed.parent), _open_entry(parsed, allow_hardlinks=allow_hardlinks) as entry:
        if directory is not None and entry.directory != directory:
            message = f"Unexpected filesystem entry type: {path}"
            raise SecurityError(message)


def _read_chunks(entry: _Entry) -> Iterator[bytes]:
    buffer = ctypes.create_string_buffer(_CHUNK_SIZE)
    count = _DWORD()
    while True:
        _check(_api().kernel.ReadFile(entry.handle, buffer, _CHUNK_SIZE, ctypes.byref(count), None))
        if not count.value:
            return
        yield buffer.raw[: count.value]


def safe_read_file(path: Path, max_bytes: int) -> bytes:
    """Read one bounded regular file, rejecting reparse points and hardlinks."""
    if max_bytes <= 0:
        message = "max_bytes must be positive"
        raise ValueError(message)
    parsed = validate_host_path(path)
    with _pin_directory(parsed.parent), _open_entry(parsed, access=_GENERIC_READ) as entry:
        if entry.directory:
            message = f"Expected a regular file: {path}"
            raise SecurityError(message)
        size = (entry.info.size_high << 32) | entry.info.size_low
        message = f"Artifact file exceeds the {max_bytes}-byte limit: {path}"
        if size > max_bytes:
            raise SandboxError(message)
        content = bytearray()
        for chunk in _read_chunks(entry):
            if len(content) + len(chunk) > max_bytes:
                raise SandboxError(message)
            content.extend(chunk)
        return bytes(content)


def _delete(entry: _Entry) -> None:
    flags = _DWORD(_DISPOSITION_DELETE | _DISPOSITION_POSIX_SEMANTICS | _DISPOSITION_IGNORE_READONLY)
    _check(
        _api().kernel.SetFileInformationByHandle(
            entry.handle, _FILE_DISPOSITION_INFO_EX, ctypes.byref(flags), ctypes.sizeof(flags)
        )
    )


def _write_all(entry: _Entry, chunks: Iterable[bytes]) -> None:
    for chunk in chunks:
        remaining = chunk
        while remaining:
            written = _DWORD()
            _check(_api().kernel.WriteFile(entry.handle, remaining, len(remaining), ctypes.byref(written), None))
            if not written.value:
                message = "Windows file write made no progress"
                raise OSError(message)
            remaining = remaining[written.value :]


def _write_chunks(path: PureWindowsPath, chunks: Iterable[bytes]) -> None:
    with _pin_directory(path.parent, create=True):
        with suppress(FileNotFoundError), _open_entry(path, access=_DELETE | _READ_ATTRIBUTES) as existing:
            if existing.directory:
                message = f"Cannot replace a directory with a file: {path}"
                raise SecurityError(message)
            _delete(existing)
        # CREATE_NEW never follows a competing name inserted after the unlink.
        # Strong directory sharing locks remain held throughout publication.
        with _open_entry(path, access=_GENERIC_WRITE | _DELETE, create=True) as output:
            try:
                _write_all(output, chunks)
            except BaseException:
                with suppress(OSError):
                    _delete(output)
                raise


def safe_write_file(path: Path, content: bytes) -> None:
    """Publish private bytes without writing through any existing inode or link.

    Replacement is not atomic: a failed write may remove the previous file.
    The newly created partial file is removed on a write failure when possible.
    """
    _write_chunks(
        validate_host_path(path),
        (content[start : start + _CHUNK_SIZE] for start in range(0, len(content), _CHUNK_SIZE)),
    )


def _reject_identity(entry: _Entry, forbidden: tuple[int, int, int] | None) -> None:
    if entry.identity == forbidden:
        message = "Source and destination paths must not overlap, including Windows short-name aliases"
        raise SecurityError(message)


def _reject_copy_overlap(source: PureWindowsPath, destination: PureWindowsPath) -> None:
    """Check native identities as short names and drive mappings can alias textual paths."""
    with (
        _pin_directory(source.parent),
        _open_entry(source) as original,
        suppress(FileNotFoundError),
        _pin_directory(destination.parent, forbidden_identity=original.identity),
        _open_entry(destination) as target,
    ):
        _reject_identity(target, original.identity)
        with _pin_directory(source.parent, forbidden_identity=target.identity):
            # Checking the complete ancestor chain rejects the reverse overlap too.
            pass


def _tree_paths(root: PureWindowsPath) -> Iterator[tuple[PureWindowsPath, bool]]:
    pending = [root]
    while pending:
        path = pending.pop()
        _validate_components(path)
        with _pin_directory(path.parent), _open_entry(path) as entry:
            yield path, entry.directory
            if entry.directory:
                with os.scandir(_native_path(path)) as children:
                    pending.extend(path / child.name for child in children)


def safe_copy(src: Path, dest: Path) -> None:
    """Copy or merge a tree, rejecting links throughout both trees before writing.

    File replacements use fresh private files, but are not atomic: a failed
    transfer can remove a previous destination file.
    """
    source, destination = validate_host_path(src), validate_host_path(dest)
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        message = "Source and destination paths must not overlap"
        raise SecurityError(message)
    _reject_copy_overlap(source, destination)
    for _path, _directory in _tree_paths(source):
        # Exhaustion validates the complete source tree before creating output.
        pass
    with suppress(FileNotFoundError):
        for _path, _directory in _tree_paths(destination):
            # Retained destination entries must satisfy the same no-link policy.
            pass
    for path, directory in _tree_paths(source):
        target = destination / path.relative_to(source)
        if directory:
            safe_mkdir(Path(str(target)))
        else:
            with _pin_directory(path.parent), _open_entry(path, access=_GENERIC_READ) as entry:
                _write_chunks(target, _read_chunks(entry))


def _remove_entry(path: PureWindowsPath, expected: tuple[int, int, int] | None = None) -> None:
    with _pin_directory(path.parent), _open_entry(path, access=_DELETE | _READ_ATTRIBUTES, cleanup=True) as entry:
        if expected is not None and entry.identity != expected:
            message = f"Private workspace was replaced before cleanup: {path}"
            raise SecurityError(message)
        _delete(entry)


def _remove_tree(path: PureWindowsPath, expected: tuple[int, int, int] | None = None) -> None:
    if not path.name:
        message = "A drive root cannot be removed as a private workspace"
        raise SecurityError(message)
    with ExitStack() as stack:
        try:
            stack.enter_context(_pin_directory(path.parent))
            root = stack.enter_context(_open_entry(path))
        except FileNotFoundError:
            return
        if not root.directory or (expected is not None and root.identity != expected):
            message = f"Private workspace is not the expected directory: {path}"
            raise SecurityError(message)
        identity = root.identity
        pending: list[tuple[PureWindowsPath, bool]] = []
        with os.scandir(_native_path(path)) as children:
            pending.extend((path / child.name, False) for child in children)
        while pending:
            current, visited = pending.pop()
            if visited:
                _remove_entry(current)
                continue
            with _pin_directory(current.parent), _open_entry(current, cleanup=True) as entry:
                if entry.directory and not entry.reparse:
                    pending.append((current, True))
                    with os.scandir(_native_path(current)) as children:
                        pending.extend((current / child.name, False) for child in children)
                    continue
            _remove_entry(current)
    _remove_entry(path, identity)


def safe_remove_tree(path: Path) -> None:
    """Remove a private tree iteratively; delete reparse entries without following them.

    Requires modern Windows FileDispositionInfoEx support. Read-only files are
    removed without clearing shared inode attributes, including hardlinked files.
    All sandbox processes must be stopped before cleanup begins.
    """
    _remove_tree(validate_host_path(path))


class PrivateWorkspace:
    """Own a protected per-user temporary directory and verify its identity on cleanup."""

    def __init__(self, *, prefix: str = "llm-sandbox-") -> None:
        """Create a private root under the trusted, canonical host temporary directory."""
        _api()
        _validate_components(PureWindowsPath(prefix))
        if not prefix or PureWindowsPath(prefix).name != prefix:
            message = "Workspace prefix must be a single Windows filename component"
            raise SecurityError(message)
        parent = validate_host_path(Path(tempfile.gettempdir()).resolve())
        with _pin_directory(parent):
            while True:
                path = parent / f"{prefix}{secrets.token_hex(16)}"
                try:
                    _create_directory(path)
                except FileExistsError:
                    continue
                break
            with _open_entry(path) as entry:
                self._identity = entry.identity
        self.name = str(path)
        self._finalizer = weakref.finalize(self, _remove_tree, path, self._identity)

    def cleanup(self) -> None:
        """Keep the finalizer registered until identity-checked deletion succeeds."""
        _remove_tree(validate_host_path(self.name), self._identity)
        self._finalizer.detach()
