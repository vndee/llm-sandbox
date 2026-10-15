"""Test seccomp compilation and enforce exported filters only in child processes."""

from __future__ import annotations

import ctypes
import errno
import os
import platform
import signal
import struct
import subprocess
import sys
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from llm_sandbox.exceptions import MissingDependencyError, SecurityError
from llm_sandbox.security import seccomp
from llm_sandbox.security.seccomp import DEFAULT_BLOCKED_SYSCALLS, SeccompPolicy, compile_seccomp, seccomp_available

# Loading a filter is intentionally confined to an expendable Python subprocess.
_FILTER_RUNNER = """
import ctypes
import os
import sys

class SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte),
                ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]

class SockFprog(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ushort), ("filter", ctypes.POINTER(SockFilter))]

fd = int(sys.argv[1])
data = os.read(fd, 32768)
os.close(fd)
instructions = (SockFilter * (len(data) // ctypes.sizeof(SockFilter))).from_buffer_copy(data)
program = SockFprog(len(instructions), instructions)
libc = ctypes.CDLL(None, use_errno=True)
libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
libc.prctl.restype = ctypes.c_int
if libc.prctl(38, 1, 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")
if libc.prctl(22, 2, ctypes.addressof(program), 0, 0) != 0:
    raise OSError(ctypes.get_errno(), "PR_SET_SECCOMP")
exec(sys.argv[2])
"""


@pytest.fixture
def native_library() -> ctypes.CDLL:
    """Require the dependency and supported ABI for tests that compile filters."""
    if not seccomp_available():
        pytest.skip("System libseccomp is unavailable")
    library = seccomp._load_libseccomp()
    if library.seccomp_arch_native() not in seccomp._SUPPORTED_NATIVE_ARCHES or ctypes.sizeof(ctypes.c_void_p) != 8:
        pytest.skip("Seccomp compilation supports native x86-64 and AArch64")
    return library


def _run_filter(policy: SeccompPolicy, script: str) -> subprocess.CompletedProcess[str]:
    """Run a script with a real exported filter without affecting pytest itself."""
    with compile_seccomp(policy) as fd:
        return subprocess.run(  # noqa: S603
            [sys.executable, "-c", _FILTER_RUNNER, str(fd), script],
            pass_fds=(fd,),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )


def test_policy_is_frozen_and_keeps_default_hardening() -> None:
    """Do not let policy mutation silently remove the configured protections."""
    policy = SeccompPolicy()
    assert {"ptrace", "mount", "bpf", "unshare", "process_vm_readv", "io_uring_setup"} <= set(policy.blocked_syscalls)
    assert policy.blocked_syscalls == DEFAULT_BLOCKED_SYSCALLS
    with pytest.raises(ValidationError, match="frozen"):
        policy.blocked_syscalls = ("getpid",)


@pytest.mark.parametrize("names", [(), ("",), ("getpid\x00mount",), ("mount;reboot",), ("móunt",)])
def test_policy_rejects_empty_or_malformed_names(names: tuple[str, ...]) -> None:
    """Reject misleading policies and syscall names truncated at the C boundary."""
    with pytest.raises(ValidationError):
        SeccompPolicy(blocked_syscalls=names)


def test_missing_library_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional dependency is reported without producing an unfiltered result."""
    monkeypatch.setattr(seccomp.sys, "platform", "linux")
    monkeypatch.setattr(seccomp.ctypes.util, "find_library", lambda _: None)
    assert not seccomp_available()
    with pytest.raises(MissingDependencyError, match="libseccomp"), compile_seccomp(SeccompPolicy()):
        pytest.fail("Missing libseccomp must not yield a descriptor")


def test_library_load_error_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken or incompatible dynamic library is an explicit dependency error."""
    monkeypatch.setattr(seccomp.sys, "platform", "linux")
    monkeypatch.setattr(seccomp.ctypes.util, "find_library", lambda _: "libseccomp.so.2")
    monkeypatch.setattr(seccomp.ctypes, "CDLL", Mock(side_effect=OSError("invalid library")))
    assert not seccomp_available()
    with pytest.raises(MissingDependencyError, match="invalid library"), compile_seccomp(SeccompPolicy()):
        pytest.fail("A library load failure must not yield a descriptor")


def test_non_linux_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid presenting a Linux mechanism as an available cross-platform backend."""
    monkeypatch.setattr(seccomp.sys, "platform", "darwin")
    assert not seccomp_available()


def test_unknown_syscall_releases_context(native_library: ctypes.CDLL, monkeypatch: pytest.MonkeyPatch) -> None:
    """A misspelled syscall must abort compilation and free the native context."""
    release = Mock(wraps=native_library.seccomp_release)
    monkeypatch.setattr(native_library, "seccomp_release", release)
    monkeypatch.setattr(seccomp, "_load_libseccomp", lambda: native_library)
    with (
        pytest.raises(SecurityError, match="unknown or unavailable"),
        compile_seccomp(SeccompPolicy(blocked_syscalls=("not_a_real_syscall",))),
    ):
        pytest.fail("An unknown rule must not yield an incomplete filter")
    release.assert_called_once()


def test_unsupported_abi_fails_closed(native_library: ctypes.CDLL, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unsupported ABI combinations are rejected before creating a context."""
    monkeypatch.setattr(native_library, "seccomp_arch_native", Mock(return_value=0x40000003))
    monkeypatch.setattr(seccomp, "_load_libseccomp", lambda: native_library)
    with pytest.raises(SecurityError, match="native Linux x86-64 and AArch64"), compile_seccomp(SeccompPolicy()):
        pytest.fail("An unsupported ABI must not yield a filter")


def test_rule_error_releases_context(native_library: ctypes.CDLL, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never return a partially compiled filter after a rule is rejected."""
    release = Mock(wraps=native_library.seccomp_release)
    monkeypatch.setattr(native_library, "seccomp_release", release)
    monkeypatch.setattr(native_library, "seccomp_rule_add_exact", Mock(return_value=-errno.EINVAL))
    monkeypatch.setattr(seccomp, "_load_libseccomp", lambda: native_library)
    with pytest.raises(SecurityError, match="rule for 'ptrace' failed"), compile_seccomp(SeccompPolicy()):
        pytest.fail("A rule failure must not yield an incomplete filter")
    release.assert_called_once()


def test_context_allocation_failure_never_yields_a_filter(
    native_library: ctypes.CDLL, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Out-of-memory initialization cannot degrade into an empty or absent filter."""
    release = Mock(wraps=native_library.seccomp_release)
    monkeypatch.setattr(native_library, "seccomp_init", Mock(return_value=None))
    monkeypatch.setattr(native_library, "seccomp_release", release)
    monkeypatch.setattr(seccomp, "_load_libseccomp", lambda: native_library)
    policy = SeccompPolicy()
    with pytest.raises(SecurityError, match="allocate"), compile_seccomp(policy):
        pytest.fail("Allocation failure must not yield a descriptor")
    release.assert_not_called()


@pytest.mark.parametrize("size", [0, 7, 32776])
def test_invalid_filter_export_releases_native_memory_and_descriptor(
    native_library: ctypes.CDLL, monkeypatch: pytest.MonkeyPatch, size: int
) -> None:
    """Empty, misaligned and over-limit programs are refused even after successful native export."""
    descriptors: list[int] = []

    def invalid_export(_context: int, fd: int) -> int:
        descriptors.append(fd)
        os.ftruncate(fd, size)
        return 0

    release = Mock(wraps=native_library.seccomp_release)
    monkeypatch.setattr(native_library, "seccomp_release", release)
    monkeypatch.setattr(native_library, "seccomp_export_bpf", invalid_export)
    monkeypatch.setattr(seccomp, "_load_libseccomp", lambda: native_library)
    policy = SeccompPolicy()
    with pytest.raises(SecurityError, match="invalid or oversized"), compile_seccomp(policy):
        pytest.fail("An invalid filter must not reach a launcher")
    release.assert_called_once()
    assert len(descriptors) == 1
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(descriptors[0])


@pytest.mark.usefixtures("native_library")
def test_descriptor_closes_on_consumer_exception() -> None:
    """Descriptors are rewound for the child and closed even if launch fails."""
    descriptors: list[int] = []

    def failed_launch() -> None:
        with compile_seccomp(SeccompPolicy()) as descriptor:
            descriptors.append(descriptor)
            assert os.lseek(descriptor, 0, os.SEEK_CUR) == 0
            assert len(os.read(descriptor, 8)) == 8
            message = "launch failed"
            raise RuntimeError(message)

    with pytest.raises(RuntimeError, match="launch failed"):
        failed_launch()
    assert len(descriptors) == 1
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(descriptors[0])


def test_export_failure_closes_file_and_context(native_library: ctypes.CDLL, monkeypatch: pytest.MonkeyPatch) -> None:
    """An export failure cleans both native memory and the intermediate file."""
    descriptors: list[int] = []
    original_temporary_file = seccomp.TemporaryFile

    def track_file(*, mode: str) -> object:
        file = original_temporary_file(mode=mode)
        descriptors.append(file.fileno())
        return file

    release = Mock(wraps=native_library.seccomp_release)
    monkeypatch.setattr(seccomp, "TemporaryFile", track_file)
    monkeypatch.setattr(native_library, "seccomp_release", release)
    monkeypatch.setattr(native_library, "seccomp_export_bpf", Mock(return_value=-errno.EIO))
    monkeypatch.setattr(seccomp, "_load_libseccomp", lambda: native_library)
    with pytest.raises(SecurityError, match="BPF export failed"), compile_seccomp(SeccompPolicy()):
        pytest.fail("An export failure must not yield a descriptor")
    release.assert_called_once()
    assert len(descriptors) == 1
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(descriptors[0])


def test_filter_denies_a_normally_permitted_syscall(native_library: ctypes.CDLL) -> None:
    """A raw syscall that works on the host must return EPERM in the child."""
    number = int(native_library.seccomp_syscall_resolve_name_arch(native_library.seccomp_arch_native(), b"getppid"))
    libc = ctypes.CDLL(None, use_errno=True)
    assert libc.syscall(number) > 0
    script = f"ctypes.set_errno(0); result = libc.syscall({number}); print(result, ctypes.get_errno())"
    result = _run_filter(SeccompPolicy(blocked_syscalls=(*DEFAULT_BLOCKED_SYSCALLS, "getppid")), script)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"-1 {errno.EPERM}"
    # Compiling and exporting must not change the parent process's permissions.
    assert libc.syscall(number) > 0


@pytest.mark.usefixtures("native_library")
def test_default_filter_allows_normal_python_and_child_processes() -> None:
    """Normal filesystem access, threads and subprocesses survive default hardening."""
    script = """
import pathlib
import subprocess
import tempfile
import threading

values = []
thread = threading.Thread(target=lambda: values.append(42))
thread.start()
thread.join()
with tempfile.TemporaryDirectory() as directory:
    path = pathlib.Path(directory) / 'result'
    path.write_text(str(values[0]))
    print(path.read_text())
subprocess.run([sys.executable, '-c', "print('child-ok')"], check=True)
"""
    result = _run_filter(SeccompPolicy(), script)
    assert result.returncode == 0, result.stderr
    assert set(result.stdout.splitlines()) == {"42", "child-ok"}


@pytest.mark.skipif(platform.machine() not in {"x86_64", "AMD64"}, reason="x32 is an x86-64 compatibility ABI")
@pytest.mark.usefixtures("native_library")
def test_x32_syscall_abi_is_rejected() -> None:
    """The x32 syscall bit cannot bypass a native x86-64 denylist."""
    # The library emits a guard against x32 syscall numbers even on hosts whose
    # kernels do not enable x32. KILL_PROCESS proves the guard executes first.
    result = _run_filter(SeccompPolicy(), "libc.syscall(0x40000027)")
    assert result.returncode == -signal.SIGSYS, result.stderr


def _filter_action(program: bytes, syscall_number: int, architecture: int) -> int:
    """Evaluate the exported classic BPF without depending on host kernel dispatch."""
    instructions = list(struct.iter_unpack("=HBBI", program))
    syscall_data = struct.pack("=iI7Q", syscall_number, architecture, 0, 0, 0, 0, 0, 0, 0)
    accumulator = 0
    position = 0
    for _ in instructions:
        code, jump_true, jump_false, operand = instructions[position]
        position += 1
        if code == 0x20:  # BPF_LD | BPF_W | BPF_ABS
            accumulator = struct.unpack_from("=I", syscall_data, operand)[0]
        elif code == 0x06:  # BPF_RET | BPF_K
            return operand
        elif code == 0x05:  # BPF_JMP | BPF_JA
            position += operand
        else:
            comparisons = {
                0x15: accumulator == operand,  # BPF_JMP | BPF_JEQ | BPF_K
                0x25: accumulator > operand,  # BPF_JMP | BPF_JGT | BPF_K
                0x35: accumulator >= operand,  # BPF_JMP | BPF_JGE | BPF_K
                0x45: bool(accumulator & operand),  # BPF_JMP | BPF_JSET | BPF_K
            }
            assert code in comparisons, f"Unsupported BPF instruction: {code:#x}"
            position += jump_true if comparisons[code] else jump_false
    pytest.fail("BPF program did not return an action")


@pytest.mark.skipif(platform.machine() not in {"x86_64", "AMD64"}, reason="Legacy x32 aliases are x86-64 specific")
@pytest.mark.usefixtures("native_library")
@pytest.mark.parametrize(
    "policy", [SeccompPolicy(), SeccompPolicy(blocked_syscalls=("getppid",))], ids=["default", "custom"]
)
def test_legacy_x32_aliases_are_denied_in_exported_filter(policy: SeccompPolicy) -> None:
    """The real BPF must reject raw 521 even when the host kernel would return ENOSYS."""
    with compile_seccomp(policy) as descriptor:
        program = os.read(descriptor, 32768)
    for syscall_number in range(512, 548):
        assert _filter_action(program, syscall_number, 0xC000003E) == 0x00050000 | errno.EPERM
    # Native read remains permitted, while both compatibility ABIs are rejected.
    assert _filter_action(program, 0, 0xC000003E) == 0x7FFF0000
    assert _filter_action(program, 0x40000065, 0xC000003E) == 0x80000000
    assert _filter_action(program, 26, 0x40000003) == 0x80000000


def test_aarch64_does_not_receive_x86_specific_reserved_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    """The x32 compatibility range must not constrain a different native ABI."""
    library = Mock()
    library.seccomp_arch_native.return_value = 0xC00000B7
    library.seccomp_init.return_value = 1
    library.seccomp_attr_set.return_value = 0
    library.seccomp_syscall_resolve_name_arch.return_value = 173  # AArch64 getppid.
    library.seccomp_rule_add_exact.return_value = 0

    def export_filter(context: int, descriptor: int) -> int:
        del context
        os.write(descriptor, struct.pack("=HBBI", 0x06, 0, 0, 0x7FFF0000))
        return 0

    library.seccomp_export_bpf.side_effect = export_filter
    monkeypatch.setattr(seccomp, "_load_libseccomp", lambda: library)
    with compile_seccomp(SeccompPolicy(blocked_syscalls=("getppid",))):
        pass
    library.seccomp_rule_add_exact.assert_called_once_with(1, 0x00050000 | errno.EPERM, 173, 0)
