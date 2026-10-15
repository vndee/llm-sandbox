"""Bounded supervision of trusted host launchers, including Windows process jobs.

This module does not sandbox payloads. Backend launchers establish their own
security boundary. The POSIX process-group path exists for trusted launcher
tests; Windows jobs prevent detached host descendants from surviving cleanup.
"""

from __future__ import annotations

import codecs
import ctypes
import math
import os
import queue
import signal
import subprocess  # nosec B404 # Trusted executable launch uses an explicit argv and shell=False.
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from llm_sandbox.data import ConsoleOutput, StreamCallback
from llm_sandbox.exceptions import MissingDependencyError, SandboxError, SandboxTimeoutError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import TracebackType

_WINDOWS = sys.platform == "win32"
_CREATE_SUSPENDED = 0x00000004
_CREATE_NO_WINDOW = 0x08000000
_READ_SIZE = 65536
_POLL_SECONDS = 0.05


def _windows_library(name: str) -> ctypes.CDLL:
    if sys.platform != "win32":
        msg = "Windows process jobs require Windows"
        raise OSError(msg)
    return ctypes.WinDLL(name, use_last_error=True)


def _windows_error() -> OSError:
    if sys.platform != "win32":
        return OSError("Windows process jobs require Windows")
    return ctypes.WinError(ctypes.get_last_error())


def _executable_candidates(path: Path, search_path: str | None) -> list[Path]:
    """Plan absolute candidates without treating the current directory as a search grant."""
    if path.is_absolute():
        return [path]
    if path.parent != Path():
        msg = "An explicit launcher path must be absolute"
        raise ValueError(msg)
    names = [str(path)]
    if _WINDOWS and not path.suffix:
        names = [str(path) + ".exe", str(path) + ".com"]
    return [
        Path(directory) / name
        for directory in (search_path if search_path is not None else os.environ.get("PATH", "")).split(os.pathsep)
        if directory and Path(directory).is_absolute()
        for name in names
    ]


def resolve_executable(executable: str | Path, *, search_path: str | None = None) -> Path:
    """Resolve a regular executable without implicitly searching the current directory.

    Windows batch files are rejected because Windows may pass them through a
    shell even when ``Popen(shell=False)`` is requested. Only absolute PATH
    entries participate in bare-name resolution.
    """
    value = os.fspath(executable)
    if not value or "\0" in value:
        msg = "Launcher executable must be a nonempty path without NUL"
        raise ValueError(msg)
    for candidate in _executable_candidates(Path(value), search_path):
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.suffix.lower() in {".bat", ".cmd"}:
            msg = "Batch files cannot be used as trusted native launchers"
            raise ValueError(msg)
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    msg = f"Trusted launcher executable was not found: {value}"
    raise MissingDependencyError(msg)


class _WindowsJob:
    """Create a kill-on-close job and assign a suspended process before resuming it."""

    def __init__(self) -> None:
        # WinDLL and Windows HANDLE layouts are evaluated only on Windows.
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [  # noqa: RUF012 # ctypes requires class-level layout declarations.
                ("per_process_user_time", ctypes.c_longlong),
                ("per_job_user_time", ctypes.c_longlong),
                ("limit_flags", wintypes.DWORD),
                ("minimum_working_set", ctypes.c_size_t),
                ("maximum_working_set", ctypes.c_size_t),
                ("active_process_limit", wintypes.DWORD),
                ("affinity", ctypes.c_size_t),
                ("priority_class", wintypes.DWORD),
                ("scheduling_class", wintypes.DWORD),
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [  # noqa: RUF012
                ("basic", BasicLimits),
                ("io_counters", ctypes.c_ulonglong * 6),
                ("process_memory_limit", ctypes.c_size_t),
                ("job_memory_limit", ctypes.c_size_t),
                ("peak_process_memory", ctypes.c_size_t),
                ("peak_job_memory", ctypes.c_size_t),
            ]

        self._kernel = _windows_library("kernel32")
        self._ntdll = _windows_library("ntdll")
        self._kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self._kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self._kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self._kernel.SetInformationJobObject.restype = wintypes.BOOL
        self._kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self._kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self._kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel.CloseHandle.restype = wintypes.BOOL
        self._ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
        self._ntdll.NtResumeProcess.restype = wintypes.LONG
        self._handle = self._kernel.CreateJobObjectW(None, None)
        if not self._handle:
            raise _windows_error()
        limits = ExtendedLimits()
        limits.basic.limit_flags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self._kernel.SetInformationJobObject(self._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = _windows_error()
            self.close()
            raise error

    def assign_and_resume(self, process: subprocess.Popen[bytes]) -> None:
        """Fail closed if job assignment or resuming the suspended launcher fails."""
        handle = int(getattr(process, "_handle"))  # noqa: B009 # CPython exposes this attribute only on Windows.
        if not self._kernel.AssignProcessToJobObject(self._handle, handle):
            raise _windows_error()
        status = self._ntdll.NtResumeProcess(handle)
        if status < 0:
            msg = f"Could not resume the launcher after job assignment (NTSTATUS {status:#x})"
            raise OSError(msg)

    def close(self) -> None:
        """Close the sole job handle to terminate every assigned process."""
        if self._handle:
            if not self._kernel.CloseHandle(self._handle):
                raise _windows_error()
            self._handle = None


def _validated_launch(argv: Sequence[str], env: Mapping[str, str]) -> tuple[list[str], dict[str, str]]:
    if not argv or any(not isinstance(value, str) or "\0" in value for value in argv):
        msg = "Launcher argv must contain strings without NUL and an executable"
        raise ValueError(msg)
    environment = dict(env)
    if any(
        not isinstance(key, str) or not key or "=" in key or "\0" in key or not isinstance(value, str) or "\0" in value
        for key, value in environment.items()
    ):
        msg = "Launcher environment must contain valid explicit names and string values"
        raise ValueError(msg)
    return [str(resolve_executable(argv[0])), *argv[1:]], environment


class ManagedProcess:
    """A trusted launcher whose Windows descendants belong to a kill-on-close job.

    ``process`` exposes unbuffered binary pipes for persistent backend protocols.
    A backend should close stdin and await guest cleanup before calling ``close``
    when its actual payload lives outside the host process tree, such as WSL.
    """

    def __init__(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        cwd: str | Path | None = None,
        stdin_pipe: bool = False,
    ) -> None:
        """Start the trusted launcher, suspending Windows execution until job assignment."""
        argv, environment = _validated_launch(argv, env)
        self._lock = threading.Lock()
        self._terminated = False
        self._job = _WindowsJob() if _WINDOWS else None
        try:
            self.process: subprocess.Popen[bytes] = subprocess.Popen(  # noqa: S603 # nosec B603
                argv,
                shell=False,
                stdin=subprocess.PIPE if stdin_pipe else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                cwd=cwd,
                bufsize=0,
                close_fds=True,
                start_new_session=not _WINDOWS,
                creationflags=(_CREATE_SUSPENDED | _CREATE_NO_WINDOW) if _WINDOWS else 0,
            )
            if self._job is not None:
                self._job.assign_and_resume(self.process)
        except BaseException:
            if hasattr(self, "process"):
                self.process.kill()
                self.process.wait()
                self._close_pipes()
            if self._job is not None:
                self._job.close()
            raise

    def terminate(self) -> None:
        """Kill the host job/process group before closing potentially busy input."""
        with self._lock:
            if self._terminated:
                return
            # A Windows synchronous pipe write can hold the CRT descriptor lock.
            # Terminate its reader before close() can wait on that writer's lock.
            if self._job is not None:
                self._job.close()
            elif sys.platform != "win32":
                with suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
            if self.process.stdin is not None:
                with suppress(OSError):
                    self.process.stdin.close()
            self._terminated = True

    def _close_pipes(self) -> None:
        for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
            if pipe is not None:
                with suppress(OSError):
                    pipe.close()

    def close(self) -> None:
        """Terminate the host process tree and release all pipe handles."""
        self.terminate()
        self._close_pipes()

    def __enter__(self) -> ManagedProcess:  # noqa: PYI034 # Keep runtime compatibility with Python 3.10.
        """Return the managed process for a bounded lifetime."""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        """Release the process tree and pipes when leaving the context."""
        self.close()


@dataclass
class _RunState:
    cancelled: threading.Event = field(default_factory=threading.Event)
    created: threading.Event = field(default_factory=threading.Event)
    process: ManagedProcess | None = None


class _PipePump:
    """Transfer pipe bytes through a fixed-size queue without blocking the supervisor."""

    def __init__(self, managed: ManagedProcess, payload: bytes | None, cancelled: threading.Event) -> None:
        self.managed = managed
        self.payload = payload
        self.cancelled = cancelled
        self.events: queue.Queue[tuple[str, bytes | BaseException]] = queue.Queue(maxsize=8)
        self.stopped = threading.Event()
        process = managed.process
        self.threads = [
            threading.Thread(target=self._read, args=(name, pipe.fileno()), daemon=True)
            for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr))
            if pipe is not None
        ]
        if payload is not None:
            self.threads.append(threading.Thread(target=self._write, daemon=True))

    def _publish(self, name: str, value: bytes | BaseException) -> None:
        while not self.stopped.is_set():
            try:
                self.events.put((name, value), timeout=_POLL_SECONDS)
            except queue.Full:
                continue
            else:
                return

    def _read(self, name: str, descriptor: int) -> None:
        try:
            while not self.stopped.is_set():
                chunk = os.read(descriptor, _READ_SIZE)
                self._publish(name, chunk)
                if not chunk:
                    return
        except OSError as exc:
            self._publish(name, exc)

    def _write(self) -> None:
        pipe = self.managed.process.stdin
        if pipe is None or self.payload is None:
            return
        try:
            view = memoryview(self.payload)
            while view and not self.stopped.is_set():
                count = os.write(pipe.fileno(), view[:_READ_SIZE])
                view = view[count:]
        except BrokenPipeError:
            pass  # The launcher's exit status is authoritative when it stops consuming input.
        except (OSError, ValueError) as exc:
            if not self.stopped.is_set() and not self.cancelled.is_set():
                self._publish("stdin", exc)
        finally:
            with suppress(OSError):
                pipe.close()

    def start(self) -> None:
        """Start readers before writing input so full-duplex protocols cannot deadlock."""
        for thread in self.threads:
            thread.start()

    def close(self) -> None:
        """Stop publishers, release process pipes, and join all worker threads."""
        self.stopped.set()
        try:
            self.managed.close()
        finally:
            for thread in self.threads:
                if thread.ident is not None:
                    thread.join(timeout=1)
        if any(thread.is_alive() for thread in self.threads):
            msg = "Native process pipe workers did not stop after process cleanup"
            raise SandboxError(msg)


class ProcessSupervisor:
    """Capture bounded output using pipe reader threads that also work on Windows.

    Callbacks run synchronously on the calling thread and must return promptly.
    ``cancel`` aborts the current command. ``close`` also permanently rejects new
    commands, allowing sessions to close without a late-start race.
    """

    def __init__(
        self, max_output_bytes: int = 8 * 1024 * 1024, encoding_errors: str = "strict", output_encoding: str = "utf-8"
    ) -> None:
        """Configure bounded byte capture and incremental decoding."""
        if max_output_bytes <= 0:
            msg = "max_output_bytes must be positive"
            raise ValueError(msg)
        codecs.lookup_error(encoding_errors)
        codecs.getincrementaldecoder(output_encoding)(errors=encoding_errors)
        self.max_output_bytes = max_output_bytes
        self.encoding_errors = encoding_errors
        self.output_encoding = output_encoding
        self._lock = threading.Lock()
        self._state: _RunState | None = None
        self._closed = False

    def _cancel(self, *, permanent: bool) -> None:
        with self._lock:
            self._closed = self._closed or permanent
            state = self._state
            if state is not None:
                state.cancelled.set()
        if state is not None:
            if not state.created.wait(timeout=5):
                msg = "Launcher creation did not finish during cancellation"
                raise SandboxError(msg)
            if state.process is not None:
                state.process.terminate()

    def cancel(self) -> None:
        """Cancel the active command without closing this reusable supervisor."""
        self._cancel(permanent=False)

    def close(self) -> None:
        """Permanently reject new commands and cancel any current command."""
        self._cancel(permanent=True)

    @staticmethod
    def _check_run(state: _RunState, deadline: float | None) -> None:
        if state.cancelled.is_set():
            msg = "Native process execution was cancelled"
            raise SandboxError(msg)
        if deadline is not None and time.monotonic() >= deadline:
            msg = "Native process execution exceeded its deadline"
            raise SandboxTimeoutError(msg)

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        cwd: str | Path | None = None,
        timeout: float | None = 60,
        on_stdout: StreamCallback | None = None,
        on_stderr: StreamCallback | None = None,
        input: bytes | None = None,  # noqa: A002 # Match subprocess.run's familiar input argument.
    ) -> ConsoleOutput:
        """Run one exact argv with explicit environment and optional input bytes."""
        if input is not None and not isinstance(input, bytes):
            msg = "Launcher input must be bytes, or None"
            raise TypeError(msg)
        if timeout is not None and (timeout <= 0 or not math.isfinite(timeout)):
            msg = "timeout must be finite and positive, or None"
            raise ValueError(msg)
        state = _RunState()
        with self._lock:
            if self._closed or self._state is not None:
                msg = "Process supervisor is closed or already running a command"
                raise SandboxError(msg)
            self._state = state
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            self._check_run(state, deadline)
            managed = ManagedProcess(argv, env=env, cwd=cwd, stdin_pipe=input is not None)
            with self._lock:
                state.process = managed
            state.created.set()
            self._check_run(state, deadline)
            return self._collect(state, deadline, input, on_stdout, on_stderr)
        finally:
            state.created.set()
            try:
                if state.process is not None:
                    state.process.close()
            finally:
                with self._lock:
                    self._state = None

    def _collect(
        self,
        state: _RunState,
        deadline: float | None,
        payload: bytes | None,
        on_stdout: StreamCallback | None,
        on_stderr: StreamCallback | None,
    ) -> ConsoleOutput:
        managed = state.process
        if managed is None:
            msg = "Launcher process has not started"
            raise SandboxError(msg)
        process = managed.process
        pump = _PipePump(managed, payload, state.cancelled)
        buffers: dict[str, list[str]] = {"stdout": [], "stderr": []}
        decoders = {
            name: codecs.getincrementaldecoder(self.output_encoding)(errors=self.encoding_errors) for name in buffers
        }
        callbacks = {"stdout": on_stdout, "stderr": on_stderr}
        remaining = len(buffers)
        total = 0
        try:
            pump.start()
            while remaining or process.poll() is None:
                self._check_run(state, deadline)
                try:
                    name, chunk = pump.events.get(timeout=_POLL_SECONDS)
                except queue.Empty:
                    continue
                if isinstance(chunk, BaseException):
                    raise chunk
                total += len(chunk)
                if total > self.max_output_bytes:
                    msg = f"Native process output exceeded {self.max_output_bytes} bytes"
                    raise SandboxError(msg)
                self._append_output(chunk, decoders[name], buffers[name], callbacks[name])
                if not chunk:
                    remaining -= 1
            self._check_run(state, deadline)
            return ConsoleOutput(process.wait(), stderr="".join(buffers["stderr"]), stdout="".join(buffers["stdout"]))
        finally:
            pump.close()

    @staticmethod
    def _append_output(
        chunk: bytes, decoder: codecs.IncrementalDecoder, buffer: list[str], callback: StreamCallback | None
    ) -> None:
        decoded = decoder.decode(chunk, final=not chunk)
        if decoded:
            buffer.append(decoded)
            if callback is not None:
                callback(decoded)
