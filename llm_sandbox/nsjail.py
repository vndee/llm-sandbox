"""Linux process sandbox backed by NsJail and optional Kafel/seccomp policy.

NsJail supplies user, mount, PID, IPC, UTS and network namespaces, drops
capabilities and sets no-new-privileges. Its default root is an empty, read-only
tmpfs: the host root is never mounted. Resource limits and deadlines belong to
``LocalProcessSession``; NsJail's restrictive defaults are explicitly disabled.
"""

from __future__ import annotations

import platform
import re
import shutil
import signal
import subprocess  # nosec B404 # Used for process typing and timeout handling; launching stays in LocalProcessSession.
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm_sandbox.exceptions import MissingDependencyError, SecurityError, UnsupportedBackendError
from llm_sandbox.local.launch import read_only_mounts
from llm_sandbox.local.session import LocalProcessSession

if TYPE_CHECKING:
    from collections.abc import Iterator

_NATIVE_ARCHITECTURES = frozenset({"x86_64", "amd64", "aarch64", "arm64"})
_X86_64_ARCHITECTURES = frozenset({"x86_64", "amd64"})
_SYSCALL_NAME = re.compile(r"[a-zA-Z_]\w*\Z", re.ASCII)
_NATIVE_SYSCALL_LIMIT = 1024
_LEGACY_X32_SYSCALLS = range(512, 548)
_KAFEL_SYSCALL_ALIASES = {"umount2": "umount"}


def _mount_argument(source: str, destination: str) -> str:
    """Reject NsJail's mount separators instead of reinterpreting user paths."""
    if any(":" in value or "\0" in value for value in (source, destination)):
        message = "NsJail mount paths cannot contain ':' or NUL characters"
        raise SecurityError(message)
    return f"{source}:{destination}"


def _seccomp_policy(blocked_syscalls: tuple[str, ...]) -> str:
    """Render a native-ABI policy without allowing Kafel source injection.

    Kafel rejects non-native audit architectures. x32 shares x86_64's audit
    architecture, so a default-allow policy alone would let x32 bypass named
    deny rules. Explicitly allow only the native number range 0..1023, after
    the deny rules, and reject everything else. On x86_64, also exclude the
    untagged x32 syscall range 512..547 accepted by kernels before Linux 5.4.
    All currently assigned native x86_64 and aarch64 syscalls are below this
    ceiling. New syscall numbers at or above it fail closed until supported.
    """
    architecture = platform.machine().lower()
    if architecture not in _NATIVE_ARCHITECTURES:
        message = "NsJail seccomp policies currently support native x86_64 and aarch64 only"
        raise SecurityError(message)
    if any(not _SYSCALL_NAME.fullmatch(name) for name in blocked_syscalls):
        message = "Seccomp syscall names must be plain Linux syscall identifiers"
        raise SecurityError(message)
    rules = []
    if blocked_syscalls:
        # Kafel calls the two-argument unmount syscall 'umount' on both supported
        # architectures; libseccomp and the public policy call it 'umount2'.
        kafel_names = dict.fromkeys(_KAFEL_SYSCALL_ALIASES.get(name, name) for name in blocked_syscalls)
        rules.append("ERRNO(1) { " + ", ".join(kafel_names) + " }")
    native_calls = ", ".join(
        f"SYSCALL[{number}]"
        for number in range(_NATIVE_SYSCALL_LIMIT)
        if architecture not in _X86_64_ARCHITECTURES or number not in _LEGACY_X32_SYSCALLS
    )
    rules.append("ALLOW { " + native_calls + " }")
    return "POLICY sandbox { " + ", ".join(rules) + " } USE sandbox DEFAULT ERRNO(1)"


class NsjailSession(LocalProcessSession):
    """Run local tools through NsJail, with no daemon or container image.

    Install ``nsjail`` separately and enable unprivileged user namespaces on
    the Linux host. The launcher must support ``--experimental_mnt new`` and
    the host must permit the new mount API, which enforces recursive read-only
    grants. Missing dependencies or rejected isolation setup fail closed; there
    is no legacy-mount or unsandboxed fallback. ``network='host'`` explicitly
    grants the host's network namespace.
    """

    def __init__(self, *, nsjail_path: str = "nsjail", **kwargs: Any) -> None:
        """Resolve the trusted launcher before creating any session resources."""
        if platform.system() != "Linux":
            backend = "nsjail (Linux only)"
            raise UnsupportedBackendError(backend)
        executable = shutil.which(nsjail_path)
        if executable is None:
            message = f"NsJail executable {nsjail_path!r} was not found; install nsjail or set nsjail_path"
            raise MissingDependencyError(message)
        self.nsjail_path = str(Path(executable).resolve())
        super().__init__(**kwargs)

    @staticmethod
    def _terminate_process(process: subprocess.Popen[bytes]) -> None:
        """Let NsJail kill its namespace before terminating its supervisor.

        NsJail responds to SIGTERM by killing and reaping the jail's PID 1
        with SIGKILL from the host PID namespace. Killing the supervisor
        immediately would instead rely on a parent-death signal that hostile
        code can clear with prctl. Do not enable NsJail's --forward_signals.

        A wedged supervisor is forcibly reaped after two seconds. That fallback
        cannot guarantee descendant cleanup if the supervisor is broken and
        the payload has disabled its parent-death signal; aggregate cgroup
        supervision is needed to cover that case.
        """
        with suppress(ProcessLookupError):
            process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            LocalProcessSession._terminate_process(process)  # noqa: SLF001

    @contextmanager
    def _launch_command(self, command: str, workdir: str) -> Iterator[tuple[list[str], tuple[int, ...]]]:
        """Build argv for a fresh namespace without exposing the host root."""
        # These flags are documented in google/nsjail/cmdline.cc. In particular,
        # omitting --chroot keeps the fresh tmpfs root instead of binding /.
        argv = [
            self.nsjail_path,
            "--mode",
            "o",
            "--quiet",
            "--time_limit",
            "0",
            "--disable_rlimits",
            # Older NsJail legacy mounts leave nested bind mounts writable.
            # Force the recursive new API; never fall back to old or auto.
            "--experimental_mnt",
            "new",
            "--cwd",
            workdir,
        ]
        # --env values are installed only in the jailed child. Inheriting the
        # payload environment would expose the host launcher to LD_PRELOAD and
        # other loader hooks before namespace isolation has been established.
        for key, value in self._environment().items():
            argv.extend(("--env", f"{key}={value}"))
        if self.sandbox_config.network == "host":
            argv.append("--disable_clone_newnet")
        elif self.sandbox_config.network == "none":
            argv.append("--iface_no_lo")
        else:
            message = f"Unsupported NsJail network policy: {self.sandbox_config.network!r}"
            raise SecurityError(message)

        for source, destination in read_only_mounts(self.sandbox_config, self.config.workdir):
            argv.extend(("--bindmount_ro", _mount_argument(source, destination)))
        argv.extend(("--bindmount", _mount_argument(str(self.workspace), self.config.workdir)))

        # A separate private directory preserves artifacts written to /tmp
        # across commands. Its host parent is never exposed inside the jail.
        argv.extend(("--bindmount", _mount_argument(str(self.tempdir), "/tmp")))  # NOSONAR(S5443) Private mount target.
        for device in ("null", "zero", "random", "urandom"):
            path = f"/dev/{device}"
            argv.extend(("--bindmount_ro", _mount_argument(path, path)))
        argv.extend(("--symlink", "/proc/self/fd:/dev/fd"))
        # POSIX shared memory and semaphores need writable /dev/shm. A fresh
        # tmpfs supports multiprocessing without exposing the host's shm.
        argv.extend(("--mount", "none:/dev/shm:tmpfs:size=67108864,mode=1777"))

        if self.sandbox_config.seccomp is not None:
            argv.extend(("--seccomp_string", _seccomp_policy(self.sandbox_config.seccomp.blocked_syscalls)))
        argv.extend(("--", "/bin/sh", "-c", command))
        yield argv, ()
