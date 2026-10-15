"""OS-enforced local sandbox settings, separate from source-code screening."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from llm_sandbox.security.seccomp import SeccompPolicy

DEFAULT_READ_ONLY_PATHS = ("/usr", "/bin", "/lib", "/lib64", "/etc/ld.so.cache", "/etc/alternatives")


class LocalSandboxConfig(BaseModel):
    """Compose filesystem, network, syscall and resource restrictions.

    Only the session's private workspace and temporary directory are writable on the host. Runtime
    directories are explicitly exposed read-only. Network access is disabled
    unless the caller chooses ``host``; that choice permits unrestricted host
    networking and does not implement domain filtering.

    Resource limits are POSIX per-process limits inherited by descendants,
    not aggregate cgroup quotas. ``memory_limit_bytes`` limits virtual address
    space, which can require a large value for JVM and JavaScript runtimes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    network: Literal["none", "host"] = "none"
    read_only_paths: tuple[str, ...] = DEFAULT_READ_ONLY_PATHS
    seccomp: SeccompPolicy | None = Field(default_factory=SeccompPolicy)
    python_executable: str = "/usr/bin/python3"
    env: dict[str, str] = Field(default_factory=dict)
    max_output_bytes: int = Field(default=8 * 1024 * 1024, gt=0, strict=True)
    max_artifact_bytes: int = Field(default=8 * 1024 * 1024, gt=0, strict=True)
    memory_limit_bytes: int | None = Field(default=None, gt=0, strict=True)
    cpu_time_limit: int | None = Field(default=None, gt=0, strict=True)
    file_size_limit: int | None = Field(default=None, gt=0, strict=True)
    open_files_limit: int | None = Field(default=None, ge=16, strict=True)

    @field_validator("read_only_paths")
    @classmethod
    def validate_read_paths(cls, paths: tuple[str, ...]) -> tuple[str, ...]:
        """Reject ambiguous paths and host virtual filesystems."""
        from pathlib import PurePosixPath

        for path in paths:
            parsed = PurePosixPath(path)
            if not parsed.is_absolute() or ".." in parsed.parts or "\x00" in path or path.startswith("//"):
                msg = f"Read-only mounts require absolute paths without traversal: {path!r}"
                raise ValueError(msg)
            if parsed == PurePosixPath("/") or parsed.parts[1] in {"proc", "sys", "dev", "tmp"}:
                msg = f"Host root and virtual/temporary filesystems cannot be exposed: {path!r}"
                raise ValueError(msg)
        return tuple(dict.fromkeys(str(PurePosixPath(path)) for path in paths))

    @field_validator("python_executable")
    @classmethod
    def validate_python_executable(cls, executable: str) -> str:
        """Require an absolute executable path inside the sandbox."""
        from pathlib import PurePosixPath

        path = PurePosixPath(executable)
        if not path.is_absolute() or ".." in path.parts or "\x00" in executable:
            msg = "python_executable must be an absolute sandbox path without traversal"
            raise ValueError(msg)
        return executable

    @field_validator("env")
    @classmethod
    def validate_environment(cls, environment: dict[str, str]) -> dict[str, str]:
        """Validate explicitly supplied environment entries."""
        for key, value in environment.items():
            if not key or "=" in key or "\x00" in key or "\x00" in value:
                msg = "Environment keys must be nonempty and cannot contain '=' or NUL; values cannot contain NUL"
                raise ValueError(msg)
        return dict(environment)
