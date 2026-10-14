"""Compile optional Linux syscall hardening without filtering the calling process.

The default policy is a denylist, not a complete syscall allowlist or a standalone
sandbox. Filesystem, network, resource and process isolation remain the backend's
responsibility. Only native x86-64 and AArch64 executables are supported; foreign
syscall ABIs are rejected by the generated filter.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import re
import sys
from contextlib import contextmanager
from tempfile import TemporaryFile
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from llm_sandbox.exceptions import MissingDependencyError, SecurityError

if TYPE_CHECKING:
    from collections.abc import Iterator

# These operations expose kernel administration, cross-process introspection,
# namespace changes and kernel interfaces that ordinary language runtimes do not
# need. This does not prohibit all namespace creation (clone has other uses),
# nor does it constrain ordinary file access or socket operations.
DEFAULT_BLOCKED_SYSCALLS: tuple[str, ...] = (
    "ptrace",
    "process_vm_readv",
    "process_vm_writev",
    "mount",
    "umount2",
    "pivot_root",
    "fsopen",
    "fsconfig",
    "fsmount",
    "fspick",
    "open_tree",
    "move_mount",
    "mount_setattr",
    "unshare",
    "setns",
    "bpf",
    "perf_event_open",
    "keyctl",
    "add_key",
    "request_key",
    "init_module",
    "finit_module",
    "delete_module",
    "reboot",
    "kexec_load",
    "kexec_file_load",
    "open_by_handle_at",
    "userfaultfd",
    "io_uring_setup",
    "io_uring_enter",
    "io_uring_register",
)

_SCMP_ACT_ALLOW = 0x7FFF0000
_SCMP_ACT_ERRNO = 0x00050000
_SCMP_ACT_KILL_PROCESS = 0x80000000
_SCMP_FLTATR_ACT_BADARCH = 2
_SCMP_ARCH_X86_64 = 0xC000003E
_SUPPORTED_NATIVE_ARCHES = frozenset({_SCMP_ARCH_X86_64, 0xC00000B7})  # x86-64, AArch64
_NATIVE_POINTER_SIZE = 8
_BPF_INSTRUCTION_SIZE = 8
_MAX_BPF_INSTRUCTIONS = 4096


class SeccompPolicy(BaseModel):
    """Immutable syscall denylist returning EPERM for blocked operations.

    Custom policies replace the default list. Extend ``DEFAULT_BLOCKED_SYSCALLS``
    when adding restrictions instead of replacing existing hardening accidentally.
    Compilation rejects names unknown to the installed libseccomp or native ABI;
    requested protections are never silently omitted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    blocked_syscalls: tuple[str, ...] = Field(default=DEFAULT_BLOCKED_SYSCALLS, min_length=1)

    @field_validator("blocked_syscalls")
    @classmethod
    def validate_syscall_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject malformed names, including embedded NULs at the C boundary."""
        for name in value:
            if re.fullmatch(r"[a-z_][a-z0-9_]*", name) is None:
                message = f"Invalid seccomp syscall name: {name!r}"
                raise ValueError(message)
        return tuple(dict.fromkeys(value))


def _load_libseccomp() -> ctypes.CDLL:
    """Load the system library and declare every C function signature used."""
    if sys.platform != "linux":
        message = "Seccomp is available only on Linux."
        raise MissingDependencyError(message)
    library_name = ctypes.util.find_library("seccomp")
    if library_name is None:
        message = "Seccomp requires the system libseccomp library (for example, the libseccomp2 package)."
        raise MissingDependencyError(message)
    try:
        library = ctypes.CDLL(library_name, use_errno=True)
        library.seccomp_arch_native.argtypes = []
        library.seccomp_arch_native.restype = ctypes.c_uint32
        library.seccomp_init.argtypes = [ctypes.c_uint32]
        library.seccomp_init.restype = ctypes.c_void_p
        library.seccomp_release.argtypes = [ctypes.c_void_p]
        library.seccomp_release.restype = None
        library.seccomp_attr_set.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint32]
        library.seccomp_attr_set.restype = ctypes.c_int
        library.seccomp_syscall_resolve_name_arch.argtypes = [ctypes.c_uint32, ctypes.c_char_p]
        library.seccomp_syscall_resolve_name_arch.restype = ctypes.c_int
        # The final argument counts the variadic argument comparisons; we use zero.
        library.seccomp_rule_add_exact.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
        library.seccomp_rule_add_exact.restype = ctypes.c_int
        library.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
        library.seccomp_export_bpf.restype = ctypes.c_int
    except (OSError, AttributeError) as exc:
        message = f"Cannot load a compatible system libseccomp library: {exc}"
        raise MissingDependencyError(message) from exc
    return library


def seccomp_available() -> bool:
    """Return whether the library dependency loads; do not test kernel enforcement.

    Compilation separately validates the native ABI and every requested syscall.
    A backend must still check that the kernel permits applying the filter.
    """
    try:
        _load_libseccomp()
    except MissingDependencyError:
        return False
    return True


def _check_result(result: int, operation: str) -> None:
    """Convert a failing libseccomp return code into an explicit security error."""
    if result != 0:
        message = f"Seccomp {operation} failed: {os.strerror(abs(result))} (code {result})."
        raise SecurityError(message)


@contextmanager
def compile_seccomp(policy: SeccompPolicy) -> Iterator[int]:
    """Yield a readable native BPF file descriptor, closing it on context exit.

    Pass the descriptor to bubblewrap with ``--seccomp`` and preserve it with
    ``subprocess.Popen(..., pass_fds=(fd,))``. Keep this context open until the
    child has inherited the descriptor. This function never loads a filter into
    the caller. The filter allows unspecified native syscalls and returns EPERM
    for blocked ones; other syscall architectures terminate the process.

    Raises:
        MissingDependencyError: Linux or a compatible libseccomp is unavailable.
        SecurityError: The native ABI, a syscall name or filter export is unsupported.

    """
    library = _load_libseccomp()
    native_arch = int(library.seccomp_arch_native())
    if native_arch not in _SUPPORTED_NATIVE_ARCHES or ctypes.sizeof(ctypes.c_void_p) != _NATIVE_POINTER_SIZE:
        message = f"Seccomp supports native Linux x86-64 and AArch64 ABIs; detected architecture {native_arch:#x}."
        raise SecurityError(message)

    context = library.seccomp_init(_SCMP_ACT_ALLOW)
    if not context:
        message = "Cannot allocate the seccomp filter context."
        raise SecurityError(message)
    try:
        _check_result(
            int(library.seccomp_attr_set(context, _SCMP_FLTATR_ACT_BADARCH, _SCMP_ACT_KILL_PROCESS)),
            "foreign-ABI protection",
        )
        for name in policy.blocked_syscalls:
            syscall_number = int(library.seccomp_syscall_resolve_name_arch(native_arch, name.encode("ascii")))
            # Negative pseudo-numbers can represent syscalls absent from this ABI.
            # Do not let libseccomp silently omit or rewrite requested protections.
            if syscall_number < 0:
                message = f"Seccomp syscall {name!r} is unknown or unavailable for the native ABI; update libseccomp."
                raise SecurityError(message)
            _check_result(
                int(library.seccomp_rule_add_exact(context, _SCMP_ACT_ERRNO | errno.EPERM, syscall_number, 0)),
                f"rule for {name!r}",
            )
        if native_arch == _SCMP_ARCH_X86_64:
            # Before Linux 5.4, numbers 512..547 reached x32 handlers even
            # without __X32_SYSCALL_BIT (for example, 521 invoked ptrace).
            # The usual libseccomp x32-bit guard does not cover these aliases.
            # Reserve the entire range independently of the requested denylist.
            for syscall_number in range(512, 548):
                _check_result(
                    int(library.seccomp_rule_add_exact(context, _SCMP_ACT_ERRNO | errno.EPERM, syscall_number, 0)),
                    f"legacy x32 protection for syscall {syscall_number}",
                )
        with TemporaryFile(mode="w+b") as bpf_file:
            _check_result(int(library.seccomp_export_bpf(context, bpf_file.fileno())), "BPF export")
            size = os.fstat(bpf_file.fileno()).st_size
            if not 0 < size <= _MAX_BPF_INSTRUCTIONS * _BPF_INSTRUCTION_SIZE or size % _BPF_INSTRUCTION_SIZE:
                message = f"Seccomp exported an invalid or oversized BPF program ({size} bytes)."
                raise SecurityError(message)
            bpf_file.seek(0)
            yield bpf_file.fileno()
    finally:
        library.seccomp_release(context)
