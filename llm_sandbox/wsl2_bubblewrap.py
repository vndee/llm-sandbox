"""Windows client for a persistent bubblewrap session in an explicit WSL2 distro.

Install a matching llm-sandbox pure Python build in the guest interpreter, plus bubblewrap
and libseccomp. The client never runs payload code with a Windows shell or in
the unsandboxed WSL worker. File transfers support bounded regular files only.
"""

# Raising inside these try blocks is intentional: every error must close the remote session.
# ruff: noqa: TRY301

from __future__ import annotations

import base64
import ctypes
import io
import math
import queue
import re
import subprocess  # nosec B404 # Used only for the trusted Windows transport's timeout exception.
import sys
import tarfile
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any

from llm_sandbox.core.config import SessionConfig
from llm_sandbox.core.session_base import BaseSession
from llm_sandbox.data import ConsoleOutput, StreamCallback
from llm_sandbox.exceptions import (
    CommandEmptyError,
    MissingDependencyError,
    NotOpenSessionError,
    SandboxError,
    SandboxTimeoutError,
    SecurityError,
    SecurityViolationError,
    ValidationError,
)
from llm_sandbox.local.config import LocalSandboxConfig
from llm_sandbox.local.wsl_protocol import (
    HEARTBEAT_INTERVAL,
    LEASE_SECONDS,
    encode_frame,
    identity,
    read_frame,
    validate_policy,
)

if TYPE_CHECKING:
    from llm_sandbox.windows.process import ProcessSupervisor

_STARTUP_TIMEOUT = 30.0
_DIAGNOSTIC_LIMIT = 8192
_32_BIT_POINTER_BYTES = 4
_WSL2_VERSION = 2
_COMPLETED_READ_ERRORS: dict[str, type[SecurityError | FileNotFoundError]] = {
    "SecurityError": SecurityError,
    "FileNotFoundError": FileNotFoundError,
}


def _wsl_executable() -> Path:
    """Resolve the trusted Windows system launcher without PATH or environment lookup."""
    if sys.platform != "win32":
        msg = "The wsl2-bubblewrap backend requires a Windows Python host"
        raise ValidationError(msg)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetSystemDirectoryW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint]
    kernel.GetSystemDirectoryW.restype = ctypes.c_uint
    buffer = ctypes.create_unicode_buffer(32768)
    count = kernel.GetSystemDirectoryW(buffer, len(buffer))
    if not count or count >= len(buffer):
        msg = "Cannot resolve the trusted Windows system directory"
        raise MissingDependencyError(msg)
    system = Path(buffer.value)
    # A 32-bit client needs the native system directory instead of WOW64 redirection.
    native = system.parent / "Sysnative" / "wsl.exe"
    executable = (
        native if ctypes.sizeof(ctypes.c_void_p) == _32_BIT_POINTER_BYTES and native.is_file() else system / "wsl.exe"
    )
    if not executable.is_file():
        msg = "WSL is not installed; enable WSL2 and install an explicit Linux distribution"
        raise MissingDependencyError(msg)
    return executable


def _distribution_version(output: str, distribution: str) -> int:
    """Read the final numeric version column without depending on translated headers."""
    matches = []
    for line in output.splitlines():
        match = re.fullmatch(r"\s*\*?\s*" + re.escape(distribution) + r"\s+.+?\s+([12])\s*", line)
        if match:
            matches.append(int(match[1]))
    if len(matches) != 1:
        msg = f"Cannot verify WSL version for distribution {distribution!r}; check 'wsl --list --verbose'"
        raise MissingDependencyError(msg)
    return matches[0]


def _launcher_environment(executable: Path) -> dict[str, str]:
    """Supply only Windows loader necessities derived from the trusted system executable."""
    root = PureWindowsPath(str(executable)).parent.parent
    return {"SystemRoot": str(root), "WINDIR": str(root), "SystemDrive": root.drive}


def _runtime_path(path: str, workdir: str) -> str:
    candidate = PurePosixPath(path)
    if not path or "\0" in path or ".." in candidate.parts or "\\" in path or path.startswith("//"):
        msg = "Runtime paths must be Linux paths without traversal, backslashes or NUL"
        raise SecurityError(msg)
    private_temp = PurePosixPath("/tmp")  # NOSONAR(S5443) Guest virtual path backed by the session's private directory.
    if candidate.is_absolute() and not (
        candidate.is_relative_to(PurePosixPath(workdir)) or candidate.is_relative_to(private_temp)
    ):
        msg = "Runtime path must stay inside the private workspace or /tmp"
        raise SecurityError(msg)
    return str(candidate)


def _validate_session_options(config: SessionConfig, environment: dict[str, str]) -> None:
    """Reject ambiguous guest paths, invalid deadlines and malformed explicit environment entries."""
    workdir = PurePosixPath(config.workdir)
    if (
        re.fullmatch(r"/[A-Za-z0-9_./-]+", config.workdir) is None
        or ".." in workdir.parts
        or workdir == PurePosixPath("/")
        or workdir.parts[1] in {"mnt", "run", "media", "proc", "dev", "sys", "tmp"}
    ):
        msg = "workdir must be an absolute private Linux path without traversal"
        raise ValueError(msg)
    for timeout in (config.default_timeout, config.execution_timeout, config.session_timeout):
        if timeout is not None and (timeout <= 0 or not math.isfinite(timeout)):
            msg = "Timeouts must be finite and positive, or None"
            raise ValueError(msg)
    for key, value in environment.items():
        if (
            not isinstance(key, str)
            or not key
            or "=" in key
            or "\0" in key
            or not isinstance(value, str)
            or "\0" in value
        ):
            msg = "Invalid sandbox environment entry"
            raise ValueError(msg)


@dataclass
class _OutboundFrame:
    data: bytes
    completed: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


class _Transport:
    """Persistent framed RPC; heartbeats and EOF cleanup are independent of execution."""

    def __init__(self, argv: list[str]) -> None:
        from llm_sandbox.windows.process import ManagedProcess

        # WSLENV is deliberately absent: do not forward host loader settings or secrets into Linux.
        self.managed = ManagedProcess(argv, env=_launcher_environment(Path(argv[0])), stdin_pipe=True)
        self.process = self.managed.process
        self.messages: queue.Queue[dict[str, Any] | BaseException | None] = queue.Queue(maxsize=8)
        self.stopped = threading.Event()
        self._outbound: queue.Queue[_OutboundFrame] = queue.Queue(maxsize=1)
        self._writing = threading.Event()
        self._write_error: BaseException | None = None
        self.diagnostics = bytearray()
        self._threads = [
            threading.Thread(target=self._read, daemon=True),
            threading.Thread(target=self._drain_stderr, daemon=True),
            threading.Thread(target=self._write, daemon=True),
            threading.Thread(target=self._heartbeat, daemon=True),
        ]
        try:
            for thread in self._threads:
                thread.start()
        except BaseException:
            self.close()
            raise

    def _enqueue(self, value: dict[str, Any] | BaseException | None) -> None:
        while not self.stopped.is_set():
            try:
                self.messages.put(value, timeout=0.1)
            except queue.Full:
                continue
            else:
                return

    def _read(self) -> None:
        pipe = self.process.stdout
        if pipe is None:
            return
        try:
            # Keep draining after close begins. The worker may be blocked writing
            # an output event and must finish that write before workspace cleanup.
            while True:
                message = read_frame(pipe)
                self._enqueue(message)
                if message is None:
                    return
        except (OSError, ValueError, SandboxError) as exc:
            self._enqueue(exc)

    def _drain_stderr(self) -> None:
        pipe = self.process.stderr
        if pipe is None:
            return
        try:
            while chunk := pipe.read(4096):
                remaining = _DIAGNOSTIC_LIMIT - len(self.diagnostics)
                if remaining > 0:
                    self.diagnostics.extend(chunk[:remaining])
        except (OSError, ValueError):
            return

    def _heartbeat(self) -> None:
        while not self.stopped.wait(HEARTBEAT_INTERVAL):
            try:
                self.send({"op": "heartbeat"}, deadline=time.monotonic() + LEASE_SECONDS)
            except (OSError, ValueError, SandboxError):
                self._enqueue(SandboxError("WSL worker heartbeat transport failed"))
                return

    def _write(self) -> None:
        while not self.stopped.is_set():
            try:
                frame = self._outbound.get(timeout=0.1)
            except queue.Empty:
                continue
            # Publish busy state before checking cancellation so close() never
            # closes a Windows CRT descriptor while its synchronous write holds it.
            self._writing.set()
            try:
                self._write_frame(frame)
            except (OSError, ValueError, SandboxError) as exc:
                frame.error = self._write_error = exc
                return
            finally:
                self._writing.clear()
                frame.completed.set()

    def _write_frame(self, frame: _OutboundFrame) -> None:
        """Finish partial pipe writes while the writer owns the published busy state."""
        pipe = self.process.stdin
        if self.stopped.is_set() or pipe is None:
            raise NotOpenSessionError
        data = memoryview(frame.data)
        while data:
            if self.stopped.is_set():
                raise NotOpenSessionError
            written = pipe.write(data)
            if not written:
                msg = "WSL worker input closed"
                raise SandboxError(msg)
            data = data[written:]
        pipe.flush()

    def _check_send(self, deadline: float | None) -> None:
        if self.stopped.is_set():
            raise NotOpenSessionError
        if self._write_error is not None:
            raise self._write_error
        if deadline is not None and time.monotonic() >= deadline:
            msg = "WSL sandbox operation exceeded its deadline while sending a request"
            raise SandboxTimeoutError(msg)

    def send(self, message: dict[str, Any], *, deadline: float | None = None) -> None:
        """Queue a whole frame, keeping blocked pipe writes off deadline-owning callers."""
        frame = _OutboundFrame(encode_frame(message))
        while True:
            self._check_send(deadline)
            try:
                self._outbound.put(frame, timeout=0.05)
            except queue.Full:
                continue
            break
        while not frame.completed.wait(0.05):
            self._check_send(deadline)
        self._check_send(deadline)
        if frame.error is not None:
            raise frame.error

    def receive(self, deadline: float | None) -> dict[str, Any]:
        """Wait in bounded increments so concurrent close always unblocks callers."""
        while not self.stopped.is_set():
            remaining = deadline - time.monotonic() if deadline is not None else 0.1
            if remaining <= 0:
                msg = "WSL sandbox operation exceeded its deadline"
                raise SandboxTimeoutError(msg)
            try:
                message = self.messages.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                continue
            if isinstance(message, BaseException):
                raise message
            if message is None:
                detail = self.diagnostics.decode("utf-8", errors="replace").strip()
                msg = f"WSL worker exited. Install the matching llm-sandbox package in guest_python. {detail}"
                raise SandboxError(msg)
            return message
        raise NotOpenSessionError

    def close(self) -> None:
        """Close stdin first; allow guest cleanup before killing the Windows transport job."""
        self.stopped.set()
        # An active writer cannot be interrupted by closing a synchronous Windows
        # CRT pipe descriptor. Stop heartbeats and allow the guest lease to expire;
        # the final managed.close() kills the reader before closing that descriptor.
        if self.process.stdin is not None and not self._writing.is_set():
            with suppress(OSError, ValueError):
                self.process.stdin.close()
        try:
            self.process.wait(timeout=LEASE_SECONDS + 2)
        except subprocess.TimeoutExpired:
            # The lease remains the guest-side backstop if WSL fails to forward EOF.
            pass
        finally:
            self.managed.close()
            for thread in self._threads:
                if thread.ident is not None and thread is not threading.current_thread():
                    thread.join(timeout=0.5)


class Wsl2BubblewrapSession(BaseSession):
    """Delegate to a guest BubblewrapSession without exposing Windows drives or interop.

    ``distribution`` is mandatory. ``guest_python`` must identify an interpreter
    with the same installed package; it is not the interpreter used for payloads
    (see ``sandbox_config.python_executable``). Arbitrary read-only mounts and
    directory copies are intentionally unsupported. Stream callbacks are
    synchronous and must return promptly; reentering session operations is rejected.
    """

    def __init__(
        self,
        *,
        distribution: str,
        guest_python: str = "/usr/bin/python3",
        sandbox_config: LocalSandboxConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Validate policy before starting WSL or creating guest state."""
        self.wsl_executable = _wsl_executable()
        if (
            not isinstance(distribution, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_. -]{0,127}", distribution) is None
        ):
            msg = "distribution must be an explicit registered WSL distribution name"
            raise ValueError(msg)
        executable = PurePosixPath(guest_python)
        if (
            not executable.is_absolute()
            or executable == PurePosixPath("/")
            or ".." in executable.parts
            or "\0" in guest_python
            or "\\" in guest_python
            or guest_python.startswith("//")
            or executable.parts[1] in {"mnt", "run", "media", "tmp"}
        ):
            msg = "guest_python must be an absolute trusted Linux interpreter path outside shared mounts"
            raise ValueError(msg)
        self.distribution = distribution
        self.guest_python = str(executable)
        self.sandbox_config = (sandbox_config or LocalSandboxConfig()).model_copy(deep=True)
        self.stream = kwargs.pop("stream", False)
        self._explicit_env = dict(kwargs.pop("env_vars", None) or {})
        libraries = kwargs.pop("libraries", None)
        for option in ("image", "dockerfile", "container_id", "runtime_configs", "keep_template", "commit_container"):
            if kwargs.pop(option, None):
                msg = f"WSL2 sandboxes do not support {option}; use sandbox_config instead"
                raise ValueError(msg)
        unknown = set(kwargs).difference(SessionConfig.model_fields)
        if unknown:
            msg = f"Unknown WSL2 sandbox options: {', '.join(sorted(unknown))}"
            raise TypeError(msg)
        if kwargs.get("workdir") is None:
            kwargs["workdir"] = "/sandbox"
        config = SessionConfig(**kwargs)
        _validate_session_options(config, self._explicit_env)
        validate_policy(self.sandbox_config, self._explicit_env)
        super().__init__(config, libraries=libraries)
        self._transport: _Transport | None = None
        self._operation_lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._request_id = 0
        self._in_callback = False
        self._opening_cancelled: threading.Event | None = None
        self._startup_probe: ProcessSupervisor | None = None

    def _verify_distribution(self) -> None:
        from llm_sandbox.windows.process import ProcessSupervisor

        probe = ProcessSupervisor(max_output_bytes=65536, output_encoding="utf-16-le")
        with self._lifecycle_lock:
            if self._opening_cancelled is not None and self._opening_cancelled.is_set():
                raise NotOpenSessionError
            self._startup_probe = probe
        try:
            result = probe.run(
                [str(self.wsl_executable), "--list", "--verbose"],
                env=_launcher_environment(self.wsl_executable),
                timeout=10,
            )
        finally:
            probe.close()
            with self._lifecycle_lock:
                if self._startup_probe is probe:
                    self._startup_probe = None
        if result.exit_code or _distribution_version(result.stdout, self.distribution) != _WSL2_VERSION:
            msg = f"Distribution {self.distribution!r} must use WSL2; WSL1 and fallback execution are unsupported"
            raise MissingDependencyError(msg)

    def _worker_argv(self) -> list[str]:
        return [
            str(self.wsl_executable),
            "--distribution",
            self.distribution,
            "--cd",
            "/",
            "--exec",
            self.guest_python,
            "-I",
            "-u",
            "-m",
            "llm_sandbox.local.wsl_worker",
        ]

    def open(self) -> None:
        """Verify WSL2 and an identical trusted worker before opening the private guest session."""
        with self._operation_lock:
            self._check_reentry()
            if self.is_open:
                return
            cancelled = threading.Event()
            with self._lifecycle_lock:
                self._opening_cancelled = cancelled
            try:
                self._open_worker(cancelled)
            finally:
                with self._lifecycle_lock:
                    if self._opening_cancelled is cancelled:
                        self._opening_cancelled = None

    def _open_worker(self, cancelled: threading.Event) -> None:
        self._verify_distribution()
        with self._lifecycle_lock:
            if cancelled.is_set():
                raise NotOpenSessionError
            # Publish process creation atomically against close(). A concurrent
            # close either prevents startup or owns the newly created transport.
            transport = _Transport(self._worker_argv())
            self._transport = transport
        try:
            hello = transport.receive(time.monotonic() + _STARTUP_TIMEOUT)
            if hello != {"type": "hello", **identity()}:
                msg = (
                    "WSL guest worker version mismatch; install matching llm-sandbox builds, "
                    "using the pure Python wheel in WSL"
                )
                raise MissingDependencyError(msg)
            options = self.config.model_dump(
                mode="json", exclude={"image", "dockerfile", "container_id", "runtime_configs"}
            )
            options.update(
                sandbox_config=self.sandbox_config.model_dump(mode="json"),
                env_vars=self._explicit_env,
                libraries=self._initial_libraries,
                verbose=False,
                stream=False,
            )
            self._guest_state = self._rpc(
                "open", options, timeout=max(_STARTUP_TIMEOUT, self.config.get_execution_timeout() or 0)
            )
            with self._lifecycle_lock:
                if self._transport is not transport:
                    raise NotOpenSessionError
                self.container = self.distribution
                self.is_open = True
                self._session_start_time = time.time()
                if self.config.session_timeout is not None:
                    self._session_timer = threading.Timer(
                        self.config.session_timeout, self._close_transport, (transport,)
                    )
                    self._session_timer.daemon = True
                    self._session_timer.start()
        except BaseException:
            self._close_transport(transport)
            raise

    def _close_transport(self, expected: _Transport) -> None:
        with self._lifecycle_lock:
            if self._transport is not expected:
                return
            self._transport = None
            self.is_open = False
            self.container = None
            self._stop_session_timer()
        expected.close()

    def close(self) -> None:
        """Cancel guest execution through EOF and the lease before reaping wsl.exe."""
        with self._lifecycle_lock:
            if self._opening_cancelled is not None:
                self._opening_cancelled.set()
            probe = self._startup_probe
            transport = self._transport
        if probe is not None:
            probe.close()
        if transport is not None:
            self._close_transport(transport)

    def _check_reentry(self) -> None:
        if self._in_callback:
            msg = "WSL operations cannot be reentered from a stream callback"
            raise SandboxError(msg)

    def _require_open(self) -> None:
        self._check_reentry()
        if not self.is_open or self._transport is None:
            raise NotOpenSessionError
        self._check_session_timeout()

    def _rpc(
        self,
        operation: str,
        params: dict[str, Any],
        *,
        timeout: float | None = None,
        on_stdout: StreamCallback | None = None,
        on_stderr: StreamCallback | None = None,
    ) -> Any:
        transport = self._transport
        if transport is None:
            raise NotOpenSessionError
        self._request_id += 1
        request_id = self._request_id
        deadline = time.monotonic() + timeout if timeout is not None else None
        rejection: SecurityError | FileNotFoundError
        try:
            transport.send({"id": request_id, "op": operation, "params": params}, deadline=deadline)
            output_bytes = 0
            while True:
                message = self._receive_response(transport, request_id, deadline)
                if message.get("type") == "result":
                    return message.get("value")
                if message.get("type") == "error":
                    if self._is_completed_read_rejection(operation, message):
                        rejection = _COMPLETED_READ_ERRORS[message["error"]](message["message"])
                        break
                    exceptions = {
                        "SandboxTimeoutError": SandboxTimeoutError,
                        "SecurityError": SecurityError,
                        "SecurityViolationError": SecurityViolationError,
                        "ValueError": ValueError,
                        "FileNotFoundError": FileNotFoundError,
                        "MissingDependencyError": MissingDependencyError,
                    }
                    raise exceptions.get(str(message.get("error")), SandboxError)(
                        message.get("message", "WSL worker failed")
                    )
                chunk = self._event_chunk(message)
                output_bytes += len(chunk.encode("utf-8", errors="surrogatepass"))
                if output_bytes > self.sandbox_config.max_output_bytes * 4:
                    msg = "WSL output events exceeded the transport limit"
                    raise SandboxError(msg)
                callback = on_stdout if message["stream"] == "stdout" else on_stderr
                self._emit_output(chunk, callback)
        except BaseException:
            self._close_transport(transport)
            raise
        # A completed, validated read rejection leaves no command running. Keep
        # the worker available so artifact collection can continue with safe files.
        raise rejection

    @staticmethod
    def _receive_response(transport: _Transport, request_id: int, deadline: float | None) -> dict[str, Any]:
        message = transport.receive(deadline)
        if type(message.get("id")) is not int or message["id"] != request_id:
            msg = "WSL worker returned an unexpected request identifier"
            raise SandboxError(msg)
        return message

    @staticmethod
    def _is_completed_read_rejection(operation: str, message: dict[str, Any]) -> bool:
        return (
            operation == "get"
            and set(message) == {"type", "id", "error", "message"}
            and isinstance(message["error"], str)
            and message["error"] in _COMPLETED_READ_ERRORS
            and isinstance(message["message"], str)
            and len(message["message"]) <= _DIAGNOSTIC_LIMIT
        )

    @staticmethod
    def _event_chunk(message: dict[str, Any]) -> str:
        """Validate output records before counting or delivering their payload text."""
        chunk = message.get("data")
        if (
            message.get("type") != "event"
            or message.get("stream") not in {"stdout", "stderr"}
            or not isinstance(chunk, str)
        ):
            msg = "Unexpected WSL worker response"
            raise SandboxError(msg)
        return chunk

    def _emit_output(self, chunk: str, callback: StreamCallback | None) -> None:
        """Run stream callbacks with the reentry guard restored even after an exception."""
        if callback is not None:
            self._in_callback = True
            try:
                callback(chunk)
            finally:
                self._in_callback = False
        elif self.stream and self.verbose:
            self._log(chunk)

    def execute_command(
        self,
        command: str,
        workdir: str | None = None,
        on_stdout: StreamCallback | None = None,
        on_stderr: StreamCallback | None = None,
    ) -> ConsoleOutput:
        """Send command text as data; only the guest bubblewrap driver invokes a shell."""
        if not command:
            raise CommandEmptyError
        with self._operation_lock:
            self._require_open()
            if workdir is not None:
                _runtime_path(workdir, self.config.workdir)
            result = self._rpc(
                "execute",
                {"command": command, "workdir": workdir},
                timeout=self.config.get_execution_timeout(),
                on_stdout=on_stdout,
                on_stderr=on_stderr,
            )
            return ConsoleOutput(**result)

    def run(
        self,
        code: str,
        libraries: list | None = None,
        timeout: float | None = None,
        on_stdout: StreamCallback | None = None,
        on_stderr: StreamCallback | None = None,
    ) -> ConsoleOutput:
        """Delegate the complete language-handler lifecycle to Linux, including code upload."""
        with self._operation_lock:
            self._require_open()
            if timeout is not None and (timeout <= 0 or not math.isfinite(timeout)):
                msg = "timeout must be finite and positive"
                raise ValueError(msg)
            result = self._rpc(
                "run",
                {"code": code, "libraries": libraries, "timeout": timeout},
                timeout=timeout or self.config.get_execution_timeout(),
                on_stdout=on_stdout,
                on_stderr=on_stderr,
            )
            return ConsoleOutput(**result)

    def install(self, libraries: list[str] | None = None) -> None:
        """Install inside the guest sandbox only when its network policy permits it."""
        with self._operation_lock:
            self._require_open()
            self._rpc("install", {"libraries": libraries}, timeout=self.config.get_execution_timeout())

    def copy_to_runtime(self, src: str, dest: str) -> None:
        """Transfer one bounded regular Windows file, rejecting reparse points and hardlinks."""
        from llm_sandbox.windows.files import safe_read_file

        with self._operation_lock:
            self._require_open()
            destination = _runtime_path(dest, self.config.workdir)
            data = safe_read_file(Path(src).absolute(), self.sandbox_config.max_artifact_bytes)
            self._rpc(
                "put",
                {"path": destination, "data": base64.b64encode(data).decode("ascii")},
                timeout=self.config.get_execution_timeout(),
            )

    def _get_file(self, src: str) -> bytes:
        result = self._rpc(
            "get", {"path": _runtime_path(src, self.config.workdir)}, timeout=self.config.get_execution_timeout()
        )
        data = base64.b64decode(result, validate=True)
        if len(data) > self.sandbox_config.max_artifact_bytes:
            msg = "WSL download exceeds the artifact limit"
            raise SandboxError(msg)
        return data

    def copy_from_runtime(self, src: str, dest: str) -> None:
        """Publish a bounded regular file through the Windows no-reparse-point boundary."""
        from llm_sandbox.windows.files import safe_write_file

        with self._operation_lock:
            self._require_open()
            safe_write_file(Path(dest).absolute(), self._get_file(src))

    def get_archive(self, path: str) -> tuple[bytes, dict]:
        """Provide the regular-file archive adapter expected by artifact handlers."""
        with self._operation_lock:
            self._require_open()
            data = self._get_file(path)
            name = PurePosixPath(path).name
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as archive:
                member = tarfile.TarInfo(name)
                member.size, member.mode = len(data), 0o600
                archive.addfile(member, io.BytesIO(data))
            return buffer.getvalue(), {"name": name, "size": len(data), "mode": 0o100600}

    def _ensure_directory_exists(self, path: str) -> None:
        with self._operation_lock:
            self._require_open()
            self._rpc(
                "mkdir", {"path": _runtime_path(path, self.config.workdir)}, timeout=self.config.get_execution_timeout()
            )

    def _ensure_ownership(self, paths: list[str]) -> None:
        for path in paths:
            _runtime_path(path, self.config.workdir)

    def _handle_timeout(self) -> None:
        self.close()

    def _connect_to_existing_container(self, container_id: str) -> None:
        msg = "WSL2 sessions cannot attach to containers"
        raise NotImplementedError(msg)

    def _process_non_stream_output(self, output: Any) -> tuple[str, str]:
        msg = "WSL output uses the framed worker protocol"
        raise NotImplementedError(msg)

    def _process_stream_output(
        self,
        output: Any,
        on_stdout: StreamCallback | None = None,  # noqa: ARG002 # Preserve the BaseSession method signature.
        on_stderr: StreamCallback | None = None,  # noqa: ARG002
    ) -> tuple[str, str]:
        return self._process_non_stream_output(output)
