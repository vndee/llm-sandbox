"""Configuration for native Windows process isolation through Microsoft MXC."""

from __future__ import annotations

import re
from pathlib import PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_HOST_ENVIRONMENT_KEYS = frozenset({"systemroot", "windir", "systemdrive", "localappdata"})


def windows_policy_path(value: str) -> str:
    """Validate a local absolute Windows path without opening the host filesystem."""
    path = PureWindowsPath(value)
    components = value.replace("/", "\\").split("\\")
    if (
        "\0" in value
        or not path.is_absolute()
        or re.fullmatch(r"[A-Za-z]:", path.drive) is None
        or any(part in {".", ".."} for part in components)
    ):
        msg = "MXC paths must be absolute local drive paths without traversal or device namespaces"
        raise ValueError(msg)
    for part in path.parts[1:]:
        stem = part.split(".", 1)[0].rstrip(" ").casefold()
        if (
            any(character < " " or character in '<>:"|?*' for character in part)
            or part.endswith((".", " "))
            or PureWindowsPath(part).is_reserved()
            or stem in {"con", "prn", "aux", "nul", "conin$", "conout$", "clock$"}
            or re.fullmatch(r"(?:com|lpt)[1-9¹²³]", stem) is not None
        ):
            msg = "MXC paths cannot contain alternate data streams, reserved names, or trailing dots/spaces"
            raise ValueError(msg)
    return str(path)


def explicit_environment(environment: dict[str, str]) -> dict[str, str]:
    """Validate payload-only variables while preserving host-owned MXC bootstrap values."""
    seen: set[str] = set()
    for key, value in environment.items():
        if not isinstance(key, str) or not key or "=" in key or "\0" in key:
            msg = "Environment keys must be nonempty strings without '=' or NUL"
            raise ValueError(msg)
        if not isinstance(value, str) or "\0" in value:
            msg = "Environment values must be strings without NUL"
            raise ValueError(msg)
        normalized = key.casefold()
        if normalized in seen or normalized in _HOST_ENVIRONMENT_KEYS:
            msg = f"Environment key is duplicated or reserved for trusted Windows bootstrap: {key}"
            raise ValueError(msg)
        seen.add(normalized)
    return dict(environment)


class WindowsMxcConfig(BaseModel):
    """Native PSEC policy with a private writable workspace and no network by default.

    Runtime directories and Windows itself are exposed read-only. Additional
    grants are explicit; neither a container image nor the complete host user
    environment is inherited. ``network='host'`` permits unrestricted host
    networking, including local services and inbound connections. Domain
    allowlists, automatic package installation, and resource quotas are not
    implemented by this adapter. An explicitly selected Python virtualenv may
    require its base installation in ``read_only_paths``; unlike the default
    current interpreter, its external dependencies cannot be inferred here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    network: Literal["none", "host"] = "none"
    read_only_paths: tuple[str, ...] = ()
    deny_paths: tuple[str, ...] = ()
    python_executable: str | None = None
    javascript_executable: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    max_output_bytes: int = Field(default=8 * 1024 * 1024, gt=0, strict=True)
    max_artifact_bytes: int = Field(default=16 * 1024 * 1024, gt=0, strict=True)

    @field_validator("read_only_paths", "deny_paths")
    @classmethod
    def validate_policy_paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Reject ambiguous grants, including nonrecursive volume-root grants."""
        validated: dict[str, str] = {}
        for value in values:
            normalized = windows_policy_path(value)
            if PureWindowsPath(normalized).parent == PureWindowsPath(normalized):
                msg = "MXC policy paths cannot be a whole volume root"
                raise ValueError(msg)
            validated.setdefault(normalized.casefold(), normalized)
        return tuple(validated.values())

    @field_validator("python_executable", "javascript_executable")
    @classmethod
    def validate_executable(cls, value: str | None) -> str | None:
        """Select real executables rather than batch files or command strings."""
        if value is None:
            return None
        normalized = windows_policy_path(value)
        if PureWindowsPath(normalized).suffix.casefold() != ".exe":
            msg = "Windows runtime paths must identify an .exe executable"
            raise ValueError(msg)
        return normalized

    @field_validator("env")
    @classmethod
    def validate_environment(cls, value: dict[str, str]) -> dict[str, str]:
        """Snapshot explicit payload environment values."""
        return explicit_environment(value)
