"""Bounded, versioned control messages shared by the Windows client and WSL worker."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import IO, Any

from llm_sandbox.exceptions import SandboxError, SecurityError
from llm_sandbox.local.config import DEFAULT_READ_ONLY_PATHS, LocalSandboxConfig
from llm_sandbox.security.seccomp import DEFAULT_BLOCKED_SYSCALLS

PROTOCOL_VERSION = 1
MAX_CONTENT_BYTES = 8 * 1024 * 1024
MAX_FRAME_BYTES = 64 * 1024 * 1024
HEARTBEAT_INTERVAL = 1.0
LEASE_SECONDS = 6.0


def identity() -> dict[str, Any]:
    """Reject a guest with an older protocol, package, or security implementation."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py"), key=lambda source: source.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        # Git can install CRLF on Windows; line endings do not change Python semantics.
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return {"protocol": PROTOCOL_VERSION, "package": version("llm-sandbox"), "implementation": digest.hexdigest()}


def encode_frame(message: dict[str, Any]) -> bytes:
    """Encode a single control record; payload output is always a JSON string field."""
    data = json.dumps(message, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii") + b"\n"
    if len(data) > MAX_FRAME_BYTES:
        msg = "WSL control message exceeds the transport limit"
        raise SandboxError(msg)
    return data


def read_frame(pipe: IO[bytes]) -> dict[str, Any] | None:
    """Read one bounded record, rejecting truncation and unstructured stdout."""
    data = pipe.readline(MAX_FRAME_BYTES + 1)
    if not data:
        return None
    if len(data) > MAX_FRAME_BYTES or not data.endswith(b"\n"):
        msg = "Oversized or truncated WSL control message"
        raise SandboxError(msg)
    try:
        message = json.loads(data)
    except ValueError as exc:
        msg = "Invalid WSL control message; verify the trusted guest worker installation"
        raise SandboxError(msg) from exc
    if not isinstance(message, dict):
        msg = "WSL control messages must be objects"
        raise SandboxError(msg)
    return message


def validate_policy(config: LocalSandboxConfig, env: dict[str, str]) -> None:
    """Keep WSL host integration inaccessible even when callers customize policy."""
    if config.seccomp is None or not set(DEFAULT_BLOCKED_SYSCALLS).issubset(config.seccomp.blocked_syscalls):
        msg = "WSL2 requires the default seccomp restrictions; custom policies may only add restrictions"
        raise SecurityError(msg)
    if not set(config.read_only_paths).issubset(DEFAULT_READ_ONLY_PATHS):
        msg = "WSL2 permits only the built-in Linux runtime read-only paths; custom mounts are unsupported"
        raise SecurityError(msg)
    if any(key.upper() in {"WSL_INTEROP", "WSLENV"} for key in (*config.env, *env)):
        msg = "WSL_INTEROP and WSLENV cannot be forwarded into a sandbox"
        raise SecurityError(msg)
    if max(config.max_artifact_bytes, config.max_output_bytes) > MAX_CONTENT_BYTES:
        msg = f"WSL2 output and artifact limits cannot exceed {MAX_CONTENT_BYTES} bytes"
        raise ValueError(msg)
