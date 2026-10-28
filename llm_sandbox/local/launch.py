"""Validated mount planning and inherited resource limits for local launchers."""

from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from llm_sandbox.exceptions import MissingDependencyError, SecurityError
from llm_sandbox.local.config import DEFAULT_READ_ONLY_PATHS

if TYPE_CHECKING:
    from llm_sandbox.local.config import LocalSandboxConfig

_HOST_NETWORK_PATHS = (
    "/etc/resolv.conf",
    "/etc/hosts",
    "/etc/nsswitch.conf",
    "/etc/services",
    "/etc/protocols",
    "/etc/ssl/certs",
    "/etc/ssl/openssl.cnf",
    "/etc/pki/ca-trust",
    "/etc/pki/tls/certs",
    "/etc/pki/tls/cert.pem",
    "/etc/pki/tls/openssl.cnf",
)


def _resolve_read_only_source(source: str) -> Path | None:
    """Resolve one grant, allowing absent distro defaults but rejecting unsafe targets."""
    path = Path(source)
    if not path.exists():
        if source in DEFAULT_READ_ONLY_PATHS or source in _HOST_NETWORK_PATHS:
            return None
        msg = f"Read-only mount source does not exist: {source}"
        raise FileNotFoundError(msg)
    resolved = path.resolve(strict=True)
    if resolved == Path("/") or resolved.parts[1] in {"proc", "sys", "dev", "tmp"}:
        msg = f"Read-only mount resolves to a host virtual or temporary filesystem: {source}"
        raise SecurityError(msg)
    if not resolved.is_dir() and not resolved.is_file():
        msg = f"Read-only mount source must be a regular file or directory: {source}"
        raise SecurityError(msg)
    return resolved


def read_only_mounts(config: LocalSandboxConfig, workdir: str) -> tuple[tuple[str, str], ...]:
    """Resolve runtime mounts without exposing the host's virtual filesystems.

    Optional system paths may differ by distribution (for example ``/lib64``).
    A missing caller-supplied path is an error, never a silently dropped grant.
    """
    workspace = PurePosixPath(workdir)
    if not workspace.is_absolute() or workspace == PurePosixPath("/") or ".." in workspace.parts:
        msg = "workdir must be an absolute non-root sandbox path without traversal"
        raise SecurityError(msg)
    if workspace.parts[1] in {"proc", "sys", "dev", "tmp"}:
        msg = "workdir cannot overlap a sandbox virtual or temporary filesystem"
        raise SecurityError(msg)
    mounts: list[tuple[str, str]] = []
    sources = list(config.read_only_paths)
    if config.network == "host":
        sources.extend(path for path in _HOST_NETWORK_PATHS if path not in sources)
    for source in sources:
        destination = PurePosixPath(source)
        if workspace.is_relative_to(destination) or destination.is_relative_to(workspace):
            msg = f"Read-only mount {source!r} overlaps the writable workspace {workdir!r}"
            raise SecurityError(msg)
        resolved = _resolve_read_only_source(source)
        if resolved is None:
            continue
        mounts.append((str(resolved), str(destination)))
    return tuple(mounts)


def apply_resource_limits(argv: list[str], config: LocalSandboxConfig) -> list[str]:
    """Apply requested hard and soft rlimits before starting the sandbox."""
    values = {
        "as": config.memory_limit_bytes,
        "cpu": config.cpu_time_limit,
        "fsize": config.file_size_limit,
        "nofile": config.open_files_limit,
    }
    flags = [f"--{name}={value}:{value}" for name, value in values.items() if value is not None]
    if not flags:
        return argv
    executable = shutil.which("prlimit")
    if executable is None:
        msg = "Local resource limits require the 'prlimit' executable from util-linux"
        raise MissingDependencyError(msg)
    # Popen changes cwd to the writable workspace. Resolve relative PATH entries
    # here so an untrusted workspace executable cannot replace the host helper.
    return [str(Path(executable).resolve()), *flags, "--", *argv]
