"""Trusted WSL2 control worker. Only BubblewrapSession executes user commands.

Run using an installed guest package and ``python -I -u -m`` from ``/``.
The host sends heartbeats even during execution. EOF, a missed lease, or a
protocol failure closes the session, killing its PID namespace before cleanup.
"""

from __future__ import annotations

import base64
import os
import platform
import queue
import re
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from llm_sandbox.bubblewrap import BubblewrapSession
from llm_sandbox.exceptions import SandboxError, SecurityError
from llm_sandbox.local.config import LocalSandboxConfig
from llm_sandbox.local.files import safe_read_file
from llm_sandbox.local.launch import read_only_mounts
from llm_sandbox.local.wsl_protocol import LEASE_SECONDS, encode_frame, identity, read_frame, validate_policy
from llm_sandbox.security import SecurityPolicy

if TYPE_CHECKING:
    from collections.abc import Iterator


def _verify_wsl2() -> None:
    release = platform.release().lower()
    if sys.platform != "linux" or "microsoft" not in release or "wsl2" not in release:
        msg = "The guest worker requires the Microsoft WSL2 Linux kernel; WSL1 is unsupported"
        raise SandboxError(msg)


def _unescape_mount(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def _wsl_driver_directory() -> Path | None:
    """Mask only the standard real directory, never follow a surprising runtime alias."""
    path = Path("/usr/lib/wsl")
    if path.is_symlink() or (path.exists() and (not path.is_dir() or path.resolve(strict=True) != path)):
        msg = "The WSL driver directory must be a canonical real directory"
        raise SecurityError(msg)
    return path if path.exists() else None


def _validate_runtime_source(source: str, destination: str, masked: Path | None) -> Path:
    """Permit the exact Linux resolver file while rejecting integration directories."""
    path = Path(source)
    resolver = destination == "/etc/resolv.conf" and path.is_relative_to("/mnt/wsl") and path.is_file()
    if (
        path == Path("/")
        or (masked is not None and path.is_relative_to(masked))
        or (not resolver and (path.parts[1] in {"mnt", "run", "media"} or path == Path("/init")))
    ):
        msg = f"WSL host integration path cannot be mounted: {source}"
        raise SecurityError(msg)
    return path


def _validate_mounts(config: LocalSandboxConfig, workdir: str) -> None:
    """Refuse indirect Windows mounts, including nested mounts beneath /usr."""
    mounts = read_only_mounts(config, workdir)
    records = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    windows_mounts: list[Path] = []
    all_mounts: list[Path] = []
    for record in records:
        fields = record.split()
        separator = fields.index("-")
        fs_type = fields[separator + 1]
        mountpoint = Path(_unescape_mount(fields[4]))
        all_mounts.append(mountpoint)
        if fs_type in {"9p", "drvfs", "cifs", "smb3"} or fs_type.startswith("fuse"):
            windows_mounts.append(mountpoint)
    # tempfile's root must also be backed by the guest, not a symlink or
    # administrator-provided bind mount of a Windows drive.
    guest_temp = "/tmp"  # NOSONAR(S5443) Validated below before mkdtemp allocates private guest state.
    private_temp = str(Path(guest_temp).resolve(strict=True))
    masked = _wsl_driver_directory()
    for source, destination in [*mounts, (private_temp, guest_temp)]:
        path = _validate_runtime_source(source, destination, masked)
        if any(
            path.is_relative_to(mount)
            or (mount.is_relative_to(path) and (masked is None or not mount.is_relative_to(masked)))
            for mount in windows_mounts
        ):
            msg = f"Runtime mount exposes a Windows or shared filesystem: {source}"
            raise SecurityError(msg)
        # Nested tmpfs/bind mounts can expose interop sockets even when their
        # filesystem type is not drvfs. WSL's injected GPU libraries are masked.
        if any(
            mount != path and mount.is_relative_to(path) and (masked is None or not mount.is_relative_to(masked))
            for mount in all_mounts
        ):
            msg = f"Nested runtime mounts are unsupported in WSL2: {source}"
            raise SecurityError(msg)


class Worker:
    """Own one private session and a bounded, heartbeat-aware control channel."""

    def __init__(self, incoming: BinaryIO, outgoing: BinaryIO) -> None:
        """Bind trusted protocol pipes without starting payload execution."""
        self.incoming = incoming
        self.outgoing = outgoing
        self.session: BubblewrapSession | None = None
        self.requests: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self.stopped = threading.Event()
        self._state_lock = threading.Lock()
        self._output_lock = threading.Lock()
        self._last_heartbeat = time.monotonic()
        try:
            self._output_fd: int | None = outgoing.fileno()
        except (OSError, ValueError):
            self._output_fd = None
        if self._output_fd is not None:
            # A dead or stalled Windows transport must never trap execution in
            # a callback and prevent the lease watcher from cleaning the workspace.
            os.set_blocking(self._output_fd, False)

    def send(self, message: dict[str, Any]) -> None:
        """Keep control records intact if startup/cleanup runs concurrently."""
        data = memoryview(encode_frame(message))
        with self._output_lock:
            while data:
                if self.stopped.is_set():
                    msg = "WSL transport closed while writing output"
                    raise SandboxError(msg)
                try:
                    written = (
                        os.write(self._output_fd, data) if self._output_fd is not None else self.outgoing.write(data)
                    )
                except BlockingIOError:
                    self.stopped.wait(0.05)
                    continue
                if not written:
                    msg = "WSL output pipe closed"
                    raise SandboxError(msg)
                data = data[written:]
            if self._output_fd is None:
                self.outgoing.flush()

    def stop(self) -> None:
        """Cancel without waiting for the request-processing thread."""
        self.stopped.set()
        with self._state_lock:
            session = self.session
        if session is not None:
            session.close()

    def _read_requests(self) -> None:
        try:
            while not self.stopped.is_set():
                message = read_frame(self.incoming)
                if message is None:
                    break
                if message.get("op") == "heartbeat":
                    self._last_heartbeat = time.monotonic()
                    continue
                if message.get("op") == "cancel":
                    break
                self.requests.put_nowait(message)
        except (OSError, ValueError, SandboxError, queue.Full):
            # Protocol failures must never keep an executing payload alive.
            pass
        finally:
            self.stop()

    def _watch_lease(self) -> None:
        while not self.stopped.wait(0.2):
            if time.monotonic() - self._last_heartbeat > LEASE_SECONDS:
                self.stop()
                return

    def serve(self) -> None:
        """Process one request at a time; output from the payload is never parsed."""
        _verify_wsl2()
        os.environ.pop("WSL_INTEROP", None)
        os.environ.pop("WSLENV", None)
        self.send({"type": "hello", **identity()})
        threading.Thread(target=self._read_requests, daemon=True).start()
        threading.Thread(target=self._watch_lease, daemon=True).start()
        try:
            while not self.stopped.is_set():
                try:
                    request = self.requests.get(timeout=0.1)
                except queue.Empty:
                    continue
                request_id = request.get("id")
                try:
                    value = self.dispatch(request)
                    self.send({"type": "result", "id": request_id, "value": value})
                except Exception as exc:  # noqa: BLE001 # Serialize trusted worker failures, never tracebacks or stdout.
                    self.send({
                        "type": "error",
                        "id": request_id,
                        "error": type(exc).__name__,
                        "message": str(exc)[:8192],
                    })
        finally:
            self.stop()

    def dispatch(self, request: dict[str, Any]) -> Any:
        """Expose an explicit operation allowlist, not arbitrary method dispatch."""
        operation = request.get("op")
        params = request.get("params", {})
        if not isinstance(params, dict):
            msg = "Request parameters must be an object"
            raise SandboxError(msg)
        if operation == "open":
            return self._open_session(params)
        session = self.session
        if session is None or self.stopped.is_set():
            msg = "Guest session is not open"
            raise SandboxError(msg)
        callbacks = {
            name: self._callback(request.get("id"), name.removeprefix("on_")) for name in ("on_stdout", "on_stderr")
        }
        if operation == "execute":
            return asdict(session.execute_command(**params, **callbacks))
        if operation == "run":
            return asdict(session.run(**params, **callbacks))
        if operation == "install":
            session.install(**params)
            return None
        return self._transfer(operation, params, session)

    def _open_session(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.session is not None:
            msg = "A worker cannot open a second session"
            raise SandboxError(msg)
        options = dict(params)
        config = LocalSandboxConfig.model_validate(options.pop("sandbox_config"))
        validate_policy(config, options.get("env_vars", {}))
        _validate_mounts(config, options.get("workdir", "/sandbox"))
        if options.get("security_policy") is not None:
            options["security_policy"] = SecurityPolicy.model_validate(options["security_policy"])
        tempfile.tempdir = "/tmp"  # NOSONAR(S5443) mkdtemp creates random private directories within the guest.
        session = _GuestSession(self.stopped, sandbox_config=config, **options)
        with self._state_lock:
            if self.stopped.is_set():
                msg = "WSL transport closed during startup"
                raise SandboxError(msg)
            self.session = session
        session.open()
        if self.stopped.is_set():
            session.close()
            msg = "WSL transport closed during startup"
            raise SandboxError(msg)
        return {"workspace": str(session.workspace), "pid": os.getpid()}

    @staticmethod
    def _transfer(operation: Any, params: dict[str, Any], session: BubblewrapSession) -> Any:
        if operation == "put":
            content = base64.b64decode(params["data"], validate=True)
            if len(content) > session.sandbox_config.max_artifact_bytes:
                msg = "Upload exceeds the artifact limit"
                raise SandboxError(msg)
            with tempfile.NamedTemporaryFile(dir="/tmp") as source:  # NOSONAR(S5443) Private guest staging file.
                source.write(content)
                source.flush()
                session.copy_to_runtime(source.name, params["path"])
            return None
        if operation == "get":
            # The session's mapping and nofollow reader restrict reads to its workspace and private /tmp.
            content = safe_read_file(
                session._runtime_host_path(params["path"]),  # noqa: SLF001 # Reuse the session's nofollow boundary.
                session.sandbox_config.max_artifact_bytes,
            )
            return base64.b64encode(content).decode("ascii")
        if operation == "mkdir":
            session._ensure_directory_exists(params["path"])  # noqa: SLF001
            return None
        msg = f"Unsupported WSL control operation: {operation!r}"
        raise SandboxError(msg)

    def _callback(self, request_id: Any, stream: str) -> Any:
        def emit(chunk: str) -> None:
            if self.stopped.is_set():
                msg = "WSL transport closed during execution"
                raise SandboxError(msg)
            self.send({"type": "event", "id": request_id, "stream": stream, "data": chunk})

        return emit


class _GuestSession(BubblewrapSession):
    """Prevent a late open/spawn after EOF races with session initialization."""

    def __init__(self, stopped: threading.Event, **kwargs: Any) -> None:
        self._transport_stopped = stopped
        super().__init__(**kwargs)

    def _require_open(self) -> None:
        if self._transport_stopped.is_set():
            msg = "WSL transport has closed"
            raise SandboxError(msg)
        super()._require_open()

    @contextmanager
    def _launch_command(self, command: str, workdir: str) -> Iterator[tuple[list[str], tuple[int, ...]]]:
        """Hide WSL's injected Windows driver mounts after mounting the Linux runtime."""
        _validate_mounts(self.sandbox_config, self.config.workdir)
        with super()._launch_command(command, workdir) as (argv, descriptors):
            masked = _wsl_driver_directory()
            if masked is not None:
                boundary = argv.index("--")
                targets = set()
                for index, option in enumerate(argv[:boundary]):
                    if option == "--ro-bind" and masked.is_relative_to(argv[index + 1]):
                        targets.add(str(Path(argv[index + 2]) / masked.relative_to(argv[index + 1])))
                # /lib may be a distinct bind of /usr/lib; every visible alias
                # needs its own mask or Windows driver mounts remain reachable.
                masks = [
                    argument for target in sorted(targets) for argument in ("--tmpfs", target, "--remount-ro", target)
                ]
                argv[boundary:boundary] = masks
            yield argv, descriptors


def main() -> None:
    """Start the installed trusted worker; no caller text appears in executable arguments."""
    worker = Worker(sys.stdin.buffer, sys.stdout.buffer)
    try:
        worker.serve()
    except (OSError, SandboxError) as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
