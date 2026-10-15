"""Copy local sandbox files without following links or exposing host paths."""

from __future__ import annotations

import errno
import os
import secrets
import stat
from collections import deque
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING

from llm_sandbox.exceptions import SandboxError, SecurityError

if TYPE_CHECKING:
    from collections.abc import Iterator


def runtime_relative(path: str, workdir: str) -> Path:
    """Map a runtime path into the workspace, rejecting parent traversal."""
    if "\x00" in path or ".." in Path(path).parts:
        msg = "Runtime paths must not contain NUL bytes or parent traversal"
        raise SecurityError(msg)
    root = _host_path(Path(workdir))
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate
    try:
        return candidate.relative_to(root)
    except ValueError as exc:
        msg = f"Runtime path must be inside {workdir}: {path}"
        raise SecurityError(msg) from exc


def _host_path(path: Path) -> Path:
    """Validate a host path lexically without resolving symlinks."""
    if not path.is_absolute() or ".." in path.parts or "\x00" in str(path):
        msg = f"Expected an absolute path without parent traversal: {path}"
        raise SecurityError(msg)
    return path


def _check_entry(info: os.stat_result, name: str) -> None:
    """Allow directories and singly linked regular files only."""
    if stat.S_ISDIR(info.st_mode):
        return
    if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
        return
    msg = f"Links and special files are not permitted in sandbox copies: {name}"
    raise SecurityError(msg)


def _identity(info: os.stat_result) -> tuple[int, int, int]:
    """Return the identity and kind of an entry for race detection."""
    return info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode)


def _entry_stat(parent_fd: int, name: str) -> os.stat_result | None:
    """Inspect an entry without following links, including dangling links."""
    try:
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    _check_entry(info, name)
    return info


@contextmanager
def _open_entry(parent_fd: int, name: str, *, directory: bool = False) -> Iterator[int]:
    """Open a checked entry relative to a pinned, already checked directory."""
    before = _entry_stat(parent_fd, name)
    if before is None:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), name)
    if directory and not stat.S_ISDIR(before.st_mode):
        msg = f"Expected a directory: {name}"
        raise SecurityError(msg)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    if stat.S_ISDIR(before.st_mode):
        flags |= os.O_DIRECTORY
    try:
        fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            msg = f"Path changed into a link or non-directory: {name}"
            raise SecurityError(msg) from exc
        raise
    try:
        after = os.fstat(fd)
        _check_entry(after, name)
        if _identity(before) != _identity(after):
            msg = f"Path changed while opening: {name}"
            raise SecurityError(msg)
        yield fd
    finally:
        os.close(fd)


def _mkdir_at(parent_fd: int, name: str) -> bool:
    """Create a private directory and report whether it was newly created."""
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    except FileExistsError:
        return False
    return True


@contextmanager
def _directory(path: Path, *, create: bool = False) -> Iterator[int]:
    """Walk an absolute path with O_NOFOLLOW on every component."""
    path = _host_path(path)
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        for component in path.parts[1:]:
            created = create and _mkdir_at(fd, component)
            with _open_entry(fd, component, directory=True) as child:
                if created:
                    os.fchmod(child, 0o700)
                next_fd = os.dup(child)
            os.close(fd)
            fd = next_fd
        yield fd
    finally:
        os.close(fd)


def safe_mkdir(path: Path) -> None:
    """Create directories with mode 0700, refusing links in any component."""
    with _directory(path, create=True):
        # Entering the context validates every component and creates missing directories.
        pass


@contextmanager
def _cleanup_directory(parent_fd: int, name: str, expected: os.stat_result) -> Iterator[int]:
    """Pin a directory before restoring owner permissions, including mode-000 entries."""
    if not stat.S_ISDIR(expected.st_mode):
        msg = f"Cleanup root must be a directory, not a link or special file: {name}"
        raise SecurityError(msg)
    flags = os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        path_fd = os.open(name, os.O_PATH | flags, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            msg = f"Cleanup directory changed into a link or non-directory: {name}"
            raise SecurityError(msg) from exc
        raise
    try:
        if _identity(os.fstat(path_fd)) != _identity(expected):
            msg = f"Cleanup directory changed while opening: {name}"
            raise SecurityError(msg)
        # Linux cannot fchmod an O_PATH descriptor. This trusted procfs reference
        # reaches the pinned inode even when mode 000 prevents opening it for reading.
        Path(f"/proc/self/fd/{path_fd}").chmod(0o700)
        fd = os.open(".", os.O_RDONLY | flags, dir_fd=path_fd)
    finally:
        os.close(path_fd)
    try:
        yield fd
    finally:
        os.close(fd)


def _remove_cleanup_directory(parent_fd: int, name: str, expected: os.stat_result) -> None:
    """Remove only the directory inode that was opened for cleanup."""
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if _identity(current) != _identity(expected):
        msg = f"Cleanup directory changed before removal: {name}"
        raise SecurityError(msg)
    os.rmdir(name, dir_fd=parent_fd)


def _lift_cleanup_directory(root_fd: int, parent_fd: int, name: str) -> str:
    """Move a nested directory directly under the private root without allocating an inode."""
    info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    # Moving a directory between parents also requires write permission on the
    # directory itself so the kernel can update its '..' entry.
    with _cleanup_directory(parent_fd, name, info):
        while True:
            destination = f".llm-sandbox-cleanup-{secrets.token_hex(16)}"
            try:
                os.stat(destination, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                break
        # The private root has no concurrent writers once the payload is reaped.
        # Avoid mkdir reservations: cleanup must also work when inode quotas are full.
        os.rename(name, destination, src_dir_fd=parent_fd, dst_dir_fd=root_fd)
    return destination


def _empty_private_tree(root_fd: int) -> None:
    """Flatten and remove a private tree using a constant number of descriptors."""
    pending = deque(os.listdir(root_fd))
    while pending:
        name = pending.popleft()
        info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode):
            os.unlink(name, dir_fd=root_fd)
            continue
        with _cleanup_directory(root_fd, name, info) as directory_fd:
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(_lift_cleanup_directory(root_fd, directory_fd, entry.name))
                    else:
                        os.unlink(entry.name, dir_fd=directory_fd)
            _remove_cleanup_directory(root_fd, name, info)


def safe_remove_tree(path: Path) -> None:
    """Delete a private Linux sandbox tree without recursion or depth-dependent FDs.

    The caller must first stop all sandbox processes and serialize cleanup with
    other workspace operations. Directories are pinned before restoring owner
    permissions; links and special files inside the tree are only unlinked.
    Nested directories move to temporary names under the same private root, so
    traversal never needs a pathname longer than a single component. Missing
    roots are harmless; a symlink root or changed directory fails closed.
    """
    path = _host_path(path)
    if not path.name:
        msg = "The filesystem root cannot be removed as a private sandbox tree"
        raise SecurityError(msg)
    with ExitStack() as stack:
        try:
            parent_fd = stack.enter_context(_directory(path.parent))
            info = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        with _cleanup_directory(parent_fd, path.name, info) as root_fd:
            _empty_private_tree(root_fd)
            _remove_cleanup_directory(parent_fd, path.name, info)


def safe_read_file(path: Path, max_bytes: int) -> bytes:
    """Read a bounded regular file without following links in any component.

    The initial size is checked before reading. At most ``max_bytes + 1`` bytes
    are consumed, so a file growing during the read cannot bypass the limit.
    """
    if max_bytes <= 0:
        msg = "max_bytes must be positive"
        raise ValueError(msg)
    path = _host_path(path)
    with _directory(path.parent) as parent_fd, _open_entry(parent_fd, path.name) as source_fd:
        info = os.fstat(source_fd)
        if not stat.S_ISREG(info.st_mode):
            msg = f"Expected a regular artifact file: {path}"
            raise SecurityError(msg)
        size_error = f"Artifact file exceeds the {max_bytes}-byte limit: {path}"
        if info.st_size > max_bytes:
            raise SandboxError(size_error)
        content = bytearray()
        while chunk := os.read(source_fd, min(65536, max_bytes - len(content) + 1)):
            content.extend(chunk)
            if len(content) > max_bytes:
                raise SandboxError(size_error)
        after = os.fstat(source_fd)
        _check_entry(after, str(path))
        if after.st_size > max_bytes:
            raise SandboxError(size_error)
        return bytes(content)


def _validate_tree(fd: int) -> None:
    """Check every existing entry, including children a directory merge would retain."""
    info = os.fstat(fd)
    _check_entry(info, "open file")
    if stat.S_ISDIR(info.st_mode):
        for name in os.listdir(fd):
            with _open_entry(fd, name) as child:
                _validate_tree(child)


def _assert_unchanged(parent_fd: int, name: str, before: os.stat_result | None) -> None:
    """Reject changes to the destination observed before atomic publication."""
    after = _entry_stat(parent_fd, name)
    if (before is None) != (after is None) or (
        before is not None and after is not None and _identity(before) != _identity(after)
    ):
        msg = f"Destination changed during copy: {name}"
        raise SecurityError(msg)


def _copy_file(source_fd: int, parent_fd: int, name: str) -> None:
    """Write a new private inode, then replace the target without following it."""
    source_info = os.fstat(source_fd)
    _check_entry(source_info, "source file")
    before = _entry_stat(parent_fd, name)
    if before is not None and not stat.S_ISREG(before.st_mode):
        msg = f"Cannot replace a directory with a file: {name}"
        raise SecurityError(msg)
    temporary = f".llm-sandbox-copy-{secrets.token_hex(16)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    output_fd = os.open(temporary, flags, 0o600, dir_fd=parent_fd)
    try:
        while chunk := os.read(source_fd, 1024 * 1024):
            remaining = memoryview(chunk)
            while remaining:
                written = os.write(output_fd, remaining)
                remaining = remaining[written:]
        _check_entry(os.fstat(source_fd), "source file")
        os.fchmod(output_fd, 0o600 | (source_info.st_mode & stat.S_IXUSR))
        _assert_unchanged(parent_fd, name, before)
        _assert_unchanged(parent_fd, temporary, os.fstat(output_fd))
        os.rename(temporary, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
    finally:
        os.close(output_fd)
        with suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=parent_fd)


def _copy_entry(source_fd: int, parent_fd: int, name: str) -> None:
    """Copy one already opened source into a destination directory."""
    if not stat.S_ISDIR(os.fstat(source_fd).st_mode):
        _copy_file(source_fd, parent_fd, name)
        return
    created = _mkdir_at(parent_fd, name)
    with _open_entry(parent_fd, name, directory=True) as destination_fd:
        if created:
            os.fchmod(destination_fd, 0o700)
        for child_name in os.listdir(source_fd):
            with _open_entry(source_fd, child_name) as child_fd:
                _copy_entry(child_fd, destination_fd, child_name)


def safe_copy(src: Path, dest: Path) -> None:
    """Copy a file or merge a directory at the exact destination path.

    Both paths must be absolute. Symlinks, hardlinked regular files and special
    files are rejected in either tree and in all parent components. Source paths
    must exist; newly created directories use mode 0700, while existing
    destination directories retain their permissions. The caller must serialize
    sandbox execution and file transfers.
    """
    src, dest = _host_path(src), _host_path(dest)
    if src.is_relative_to(dest) or dest.is_relative_to(src):
        msg = "Source and destination paths must not overlap"
        raise SecurityError(msg)
    with _directory(src.parent) as source_parent, _open_entry(source_parent, src.name) as source_fd:
        _validate_tree(source_fd)
        with _directory(dest.parent, create=True) as destination_parent:
            if _entry_stat(destination_parent, dest.name) is not None:
                with _open_entry(destination_parent, dest.name) as destination_fd:
                    _validate_tree(destination_fd)
            _copy_entry(source_fd, destination_parent, dest.name)
