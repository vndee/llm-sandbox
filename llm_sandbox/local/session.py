"""Shared lifecycle for native sandbox drivers; there is no unsandboxed fallback."""

from __future__ import annotations

import codecs
import io
import math
import os
import re
import selectors
import shlex
import signal
import subprocess  # nosec B404 # Required to supervise native sandbox launchers; no host shell is used.
import tarfile
import tempfile
import threading
import time
import weakref
from abc import abstractmethod
from contextlib import AbstractContextManager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from llm_sandbox.const import EncodingErrorsType, SupportedLanguage
from llm_sandbox.core.config import SessionConfig
from llm_sandbox.core.session_base import BaseSession
from llm_sandbox.data import ConsoleOutput, StreamCallback
from llm_sandbox.exceptions import (
    CommandEmptyError,
    CommandFailedError,
    LibraryInstallationNotSupportedError,
    NotOpenSessionError,
    SandboxError,
    SandboxTimeoutError,
    SecurityViolationError,
)
from llm_sandbox.language_handlers.runtime_context import RuntimeContext
from llm_sandbox.local.config import LocalSandboxConfig
from llm_sandbox.local.files import runtime_relative, safe_copy, safe_mkdir, safe_read_file, safe_remove_tree
from llm_sandbox.local.launch import apply_resource_limits

if TYPE_CHECKING:
    from llm_sandbox.security import SecurityPolicy

_READ_SIZE = 65536
_POLL_INTERVAL = 0.1


def _validate_workdir(workdir: str | None) -> str:
    """Normalize the private workspace's sandbox path before constructing a session."""
    workdir = workdir or "/sandbox"
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", workdir) or ".." in Path(workdir).parts:
        msg = "workdir must be an absolute path without traversal, whitespace or shell metacharacters"
        raise ValueError(msg)
    if Path(workdir) == Path("/"):
        msg = "The native workspace cannot be the filesystem root"
        raise ValueError(msg)
    return str(Path(workdir))


def _validate_environment(env_vars: dict[str, str] | None) -> dict[str, str]:
    """Validate and snapshot session-level environment overrides."""
    explicit_env = dict(env_vars or {})
    for key, value in explicit_env.items():
        if not isinstance(key, str) or not key or "=" in key or "\0" in key:
            msg = "Environment variable names must be nonempty strings without '=' or NUL"
            raise ValueError(msg)
        if not isinstance(value, str) or "\0" in value:
            msg = "Environment variable values must be strings without NUL"
            raise ValueError(msg)
    return explicit_env


class _PrivateWorkspace:
    """Own a temporary tree whose explicit and implicit cleanup use bounded descriptors."""

    def __init__(self) -> None:
        # Host temp ancestors are trusted; resolve them before no-follow workspace operations.
        self.name = str(Path(tempfile.mkdtemp(prefix="llm-sandbox-")).resolve())
        self._finalizer = weakref.finalize(self, safe_remove_tree, Path(self.name))

    def cleanup(self) -> None:
        """Keep ownership and the finalizer available if removal needs to be retried."""
        safe_remove_tree(Path(self.name))
        self._finalizer.detach()


class LocalProcessSession(BaseSession):
    """Run each command through a native driver with a persistent private workspace.

    Commands and copies are serialized. The driver must isolate its PID namespace:
    killing the outer process group alone cannot terminate children that call setsid.
    Stream callbacks execute synchronously and must return promptly.
    """

    def __init__(
        self,
        *,
        sandbox_config: LocalSandboxConfig | None = None,
        lang: str = SupportedLanguage.PYTHON,
        workdir: str | None = "/sandbox",
        verbose: bool = False,
        stream: bool = False,
        image: str | None = None,
        dockerfile: str | None = None,
        keep_template: bool = False,
        commit_container: bool = False,
        container_id: str | None = None,
        runtime_configs: dict | None = None,
        env_vars: dict[str, str] | None = None,
        libraries: list[str] | None = None,
        default_timeout: float | None = 60,
        execution_timeout: float | None = None,
        session_timeout: float | None = None,
        skip_environment_setup: bool = False,
        encoding_errors: EncodingErrorsType = "strict",
        security_policy: SecurityPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        """Validate native options and accept only inactive container wrapper defaults."""
        unsupported = {
            "image": image,
            "dockerfile": dockerfile,
            "keep_template": keep_template,
            "commit_container": commit_container,
            "container_id": container_id,
            "runtime_configs": runtime_configs,
        }
        if active := [name for name, value in unsupported.items() if value]:
            msg = f"Native sandboxes do not support: {', '.join(active)}; use sandbox_config instead"
            raise ValueError(msg)
        if kwargs:
            msg = f"Unknown native sandbox options: {', '.join(sorted(kwargs))}"
            raise TypeError(msg)
        workdir = _validate_workdir(workdir)
        for name, value in (
            ("default_timeout", default_timeout),
            ("execution_timeout", execution_timeout),
            ("session_timeout", session_timeout),
        ):
            if value is not None and (value <= 0 or not math.isfinite(value)):
                msg = f"{name} must be finite and positive, or None"
                raise ValueError(msg)
        self.sandbox_config = sandbox_config or LocalSandboxConfig()
        self._explicit_env = _validate_environment(env_vars)
        config = SessionConfig(
            lang=SupportedLanguage(lang),
            workdir=workdir,
            verbose=verbose,
            default_timeout=default_timeout,
            execution_timeout=execution_timeout,
            session_timeout=session_timeout,
            skip_environment_setup=skip_environment_setup,
            encoding_errors=encoding_errors,
            security_policy=security_policy,
        )
        super().__init__(config, libraries=libraries)
        self.stream = stream
        self.workspace: Path
        self.tempdir: Path
        self._workspace_temp: _PrivateWorkspace | None = None
        self._operation_lock = threading.RLock()
        self._process_lock = threading.Lock()
        self._active_process: subprocess.Popen[bytes] | None = None
        self._generation = 0
        self._closing = False
        self._deadline: float | None = None

    @abstractmethod
    def _launch_command(self, command: str, workdir: str) -> AbstractContextManager[tuple[list[str], tuple[int, ...]]]:
        """Provide the sandbox launcher argv and inherited policy file descriptors."""
        raise NotImplementedError

    def _environment(self) -> dict[str, str]:
        """Return payload environment settings for the driver to apply after isolation."""
        return {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": f"{self.config.workdir}/.home",
            "TMPDIR": "/tmp",  # NOSONAR(S5443) Sandbox mount target backed by the private session directory.
            "LANG": "C.UTF-8",
            "PYTHONUNBUFFERED": "1",
            **self.sandbox_config.env,
            **self._explicit_env,
        }

    @staticmethod
    def _launcher_environment() -> dict[str, str]:
        """Keep payload loader settings away from trusted host launcher executables."""
        return {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"}

    def open(self) -> None:
        """Create a private workspace and prove the configured sandbox can launch."""
        with self._operation_lock:
            with self._process_lock:
                if self.is_open and not self._closing:
                    return
                self._require_idle()
                previous_generation = self._generation
            if self._workspace_temp is not None:
                self._close_generation(previous_generation)
            with self._process_lock:
                self._generation += 1
                self._closing = False
            self._workspace_temp = _PrivateWorkspace()
            private_root = Path(self._workspace_temp.name)
            private_root.chmod(0o700)
            self.workspace = private_root / "workspace"
            self.tempdir = private_root / "tmp"
            safe_mkdir(self.workspace / ".home")
            safe_mkdir(self.tempdir)
            self.container = self.workspace
            self.is_open = True
            self._start_session_timer()
            try:
                self._require_success("/bin/true")
                self.environment_setup()
            except BaseException:
                self.close()
                raise

    @staticmethod
    def _terminate_process(process: subprocess.Popen[bytes]) -> None:
        """Kill the launcher process group and reap its leader."""
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()

    def close(self) -> None:
        """Cancel active execution before waiting for the operation lock."""
        with self._process_lock:
            generation = self._generation
        self._close_generation(generation)

    def _close_generation(self, generation: int) -> None:
        """Finish only the lifecycle that requested cleanup, even after a reopen."""
        with self._process_lock:
            if generation != self._generation:
                return
            self._closing = True
            process = self._active_process
            if process is not None:
                # Do not let another closer finish and reopen while this one
                # still holds a process handle from the previous lifecycle.
                self._terminate_process(process)
        with self._operation_lock:
            with self._process_lock:
                if generation != self._generation:
                    return
            self._stop_session_timer()
            self.is_open = False
            self.container = None
            if self._workspace_temp is not None:
                self._workspace_temp.cleanup()
                self._workspace_temp = None

    def _start_session_timer(self) -> None:
        """Bind each timeout callback to its originating session lifecycle."""
        self._session_start_time = time.time()
        if self.config.session_timeout is None:
            return
        generation = self._generation

        def timeout_handler() -> None:
            try:
                self._close_generation(generation)
            except Exception as exc:  # noqa: BLE001 # Timer cleanup errors cannot propagate to the caller.
                self._log(f"Error during timeout cleanup: {exc}", "error")

        self._session_timer = threading.Timer(self.config.session_timeout, timeout_handler)
        self._session_timer.daemon = True
        self._session_timer.start()

    def _require_open(self) -> None:
        if not self.is_open or self._closing:
            raise NotOpenSessionError
        self._check_session_timeout()

    def _require_idle(self) -> None:
        if self._active_process is not None:
            msg = "Sandbox operations cannot be reentered from a stream callback while a command is active"
            raise SandboxError(msg)

    def _execute_with_timeout(
        self,
        func: Any,
        *args: Any,
        timeout: float | None = None,
        force_kill_on_timeout: bool = True,  # noqa: ARG002
        **kwargs: Any,
    ) -> Any:
        """Propagate a synchronous deadline without creating an uncancellable worker."""
        with self._operation_lock:
            previous = self._deadline
            if timeout is not None:
                if timeout <= 0 or not math.isfinite(timeout):
                    msg = "timeout must be finite and positive"
                    raise ValueError(msg)
                deadline = time.monotonic() + timeout
                self._deadline = min(previous, deadline) if previous is not None else deadline
            try:
                result = func(*args, **kwargs)
                self._remaining_time()
                return result
            finally:
                self._deadline = previous

    def _remaining_time(self) -> float | None:
        if self._deadline is None:
            return None
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            msg = "Native sandbox execution exceeded its deadline"
            raise SandboxTimeoutError(msg)
        return remaining

    def run(
        self,
        code: str,
        libraries: list | None = None,
        timeout: float | None = None,
        on_stdout: StreamCallback | None = None,
        on_stderr: StreamCallback | None = None,
    ) -> ConsoleOutput:
        """Serialize code upload, optional installation and execution as one operation."""
        with self._operation_lock:
            self._require_open()
            if timeout is not None and (timeout <= 0 or not math.isfinite(timeout)):
                msg = "timeout must be finite and positive"
                raise ValueError(msg)
            if self.language_handler.name == SupportedLanguage.PYTHON:
                executable = shlex.quote(self.sandbox_config.python_executable).replace("{", "{{").replace("}", "}}")
                self.language_handler.config.execution_commands = [f"{executable} {{file}}"]
            return super().run(code, libraries, timeout, on_stdout, on_stderr)

    def execute_command(
        self,
        command: str,
        workdir: str | None = None,
        on_stdout: StreamCallback | None = None,
        on_stderr: StreamCallback | None = None,
    ) -> ConsoleOutput:
        """Execute through the driver, capturing bounded output and enforcing a deadline."""
        if not command:
            raise CommandEmptyError
        with self._operation_lock:
            self._require_open()
            relative = runtime_relative(workdir or self.config.workdir, self.config.workdir)
            sandbox_workdir = str(Path(self.config.workdir) / relative)
            return cast(
                "ConsoleOutput",
                self._execute_with_timeout(
                    self._execute_command,
                    command,
                    sandbox_workdir,
                    on_stdout,
                    on_stderr,
                    timeout=self.config.get_execution_timeout() if self._deadline is None else None,
                ),
            )

    def _execute_command(
        self,
        command: str,
        workdir: str,
        on_stdout: StreamCallback | None,
        on_stderr: StreamCallback | None,
    ) -> ConsoleOutput:
        with self._launch_command(command, workdir) as (argv, pass_fds):
            launch_argv = apply_resource_limits(argv, self.sandbox_config)
            with self._process_lock:
                self._require_open()
                self._require_idle()
                self._remaining_time()
                # Drivers provide absolute trusted launcher paths, including the optional prlimit wrapper.
                # The payload shell runs only inside the sandbox; the host launcher gets a clean environment.
                process = subprocess.Popen(  # noqa: S603 # nosec B603
                    launch_argv,
                    shell=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=self._launcher_environment(),
                    cwd=self.workspace,
                    close_fds=True,
                    pass_fds=pass_fds,
                    start_new_session=True,
                )
                self._active_process = process
            try:
                return self._collect_output(process, on_stdout, on_stderr)
            finally:
                self._terminate_process(process)
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()
                with self._process_lock:
                    self._active_process = None

    def _collect_output(
        self,
        process: subprocess.Popen[bytes],
        on_stdout: StreamCallback | None,
        on_stderr: StreamCallback | None,
    ) -> ConsoleOutput:
        buffers: dict[str, list[str]] = {"stdout": [], "stderr": []}
        callbacks = {"stdout": on_stdout, "stderr": on_stderr}
        decoders = {name: codecs.getincrementaldecoder("utf-8")(errors=self.config.encoding_errors) for name in buffers}
        total_bytes = 0
        with selectors.DefaultSelector() as selector:
            for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
                if pipe is not None:
                    os.set_blocking(pipe.fileno(), False)
                    selector.register(pipe, selectors.EVENT_READ, name)
            while selector.get_map():
                remaining = self._remaining_time()
                wait = min(remaining, _POLL_INTERVAL) if remaining is not None else _POLL_INTERVAL
                for key, _ in selector.select(wait):
                    name = key.data
                    chunk = os.read(key.fd, _READ_SIZE)
                    total_bytes += len(chunk)
                    if total_bytes > self.sandbox_config.max_output_bytes:
                        msg = f"Native sandbox output exceeded {self.sandbox_config.max_output_bytes} bytes"
                        raise SandboxError(msg)
                    self._append_output(chunk, decoders[name], buffers[name], callbacks[name])
                    if not chunk:
                        selector.unregister(key.fileobj)
            try:
                return_code = process.wait(timeout=self._remaining_time())
            except subprocess.TimeoutExpired as exc:
                msg = "Native sandbox execution exceeded its deadline"
                raise SandboxTimeoutError(msg) from exc
        return ConsoleOutput(return_code, stderr="".join(buffers["stderr"]), stdout="".join(buffers["stdout"]))

    def _append_output(
        self,
        chunk: bytes,
        decoder: codecs.IncrementalDecoder,
        buffer: list[str],
        callback: StreamCallback | None,
    ) -> None:
        """Decode an already bounded chunk and deliver it to its buffer and callback."""
        decoded = decoder.decode(chunk, final=not chunk)
        if not decoded:
            return
        buffer.append(decoded)
        if callback is not None:
            callback(decoded)
        elif self.stream and self.verbose:
            self._log(decoded)

    def _require_success(self, command: str) -> None:
        output = self.execute_command(command)
        if output.exit_code:
            raise CommandFailedError(command, output.exit_code, output.stderr or output.stdout)

    @property
    def python_executable_path(self) -> str:
        """Return the explicitly selected Python or the session's offline virtualenv."""
        if self.config.skip_environment_setup:
            return self.sandbox_config.python_executable
        return super().python_executable_path

    def environment_setup(self) -> None:
        """Create language state offline; package installation requires explicit network permission."""
        if self.config.skip_environment_setup:
            if self._initial_libraries:
                raise LibraryInstallationNotSupportedError(self.config.lang)
            return
        if self.language_handler.name == SupportedLanguage.PYTHON:
            self._require_success(
                f"{shlex.quote(self.sandbox_config.python_executable)} -m venv --system-site-packages "
                f"{shlex.quote(self._python_env_dir)}"
            )
            self._ensure_directory_exists(self._python_pip_cache_dir)
        elif self.language_handler.name == SupportedLanguage.GO:
            self._require_success("go mod init sandbox")
        self.install(self._initial_libraries)

    def install(self, libraries: list[str] | None = None) -> None:
        """Install requested packages explicitly and surface every installation failure."""
        if not libraries:
            return
        with self._operation_lock:
            self._require_open()
            if self.config.skip_environment_setup or not self.language_handler.is_support_library_installation:
                raise LibraryInstallationNotSupportedError(self.config.lang)
            if self.sandbox_config.network == "none":
                msg = "Library installation needs network='host'; for offline wheels use execute_command explicitly"
                raise ValueError(msg)
            if self.config.security_policy:
                restricted = self.config.security_policy.restricted_modules or []
                for library in libraries:
                    if any(library == module.name for module in restricted):
                        msg = f"Library {library} is not allowed to be installed"
                        raise SecurityViolationError(msg)
            context = RuntimeContext(
                workdir=self.config.workdir,
                python_executable_path=self.python_executable_path,
                pip_executable_path=self.pip_executable_path,
                pip_cache_dir=self.pip_cache_dir_path,
            )
            for library in libraries:
                self._require_success(self.language_handler.get_library_installation_command(library, context))

    def copy_to_runtime(self, src: str, dest: str) -> None:
        """Copy into the private workspace or /tmp; links are rejected."""
        with self._operation_lock:
            self._require_open()
            self._require_idle()
            safe_copy(Path(src).absolute(), self._runtime_host_path(dest))

    def copy_from_runtime(self, src: str, dest: str) -> None:
        """Copy private runtime files to an exact host destination without following links."""
        with self._operation_lock:
            self._require_open()
            self._require_idle()
            safe_copy(self._runtime_host_path(src), Path(dest).absolute())

    def _runtime_host_path(self, path: str) -> Path:
        """Map only workspace-relative paths or explicit workspace and /tmp paths."""
        if Path(path).is_absolute() and Path(path).is_relative_to("/tmp"):  # NOSONAR(S5443) Match only a sandbox path.
            return self.tempdir / runtime_relative(path, "/tmp")  # NOSONAR(S5443) Map into the private host directory.
        return self.workspace / runtime_relative(path, self.config.workdir)

    def get_archive(self, path: str) -> tuple[bytes, dict]:
        """Return one bounded regular-file artifact in the archive format used by handlers."""
        with self._operation_lock:
            self._require_open()
            self._require_idle()
            source = self._runtime_host_path(path)
            content = safe_read_file(source, self.sandbox_config.max_artifact_bytes)
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as archive:
                member = tarfile.TarInfo(source.name)
                member.size = len(content)
                member.mode = 0o600
                archive.addfile(member, io.BytesIO(content))
            return buffer.getvalue(), {"name": source.name, "size": len(content), "mode": 0o100600}

    def _ensure_directory_exists(self, path: str) -> None:
        with self._operation_lock:
            self._require_open()
            self._require_idle()
            safe_mkdir(self._runtime_host_path(path))

    def _ensure_ownership(self, paths: list[str]) -> None:
        """Validate workspace paths; native sessions retain the launching user's ownership."""
        for path in paths:
            self._runtime_host_path(path)

    def _handle_timeout(self) -> None:
        """Close timed-out sessions after synchronous process cleanup."""
        self.close()

    def _connect_to_existing_container(self, container_id: str) -> None:
        msg = "Native sandboxes cannot attach to existing containers"
        raise NotImplementedError(msg)

    def _process_non_stream_output(self, output: Any) -> tuple[str, str]:
        msg = "Native output is decoded incrementally by the process supervisor"
        raise NotImplementedError(msg)

    def _process_stream_output(
        self,
        output: Any,
        on_stdout: StreamCallback | None = None,
        on_stderr: StreamCallback | None = None,
    ) -> tuple[str, str]:
        msg = "Native streaming output is delivered directly by the process supervisor"
        raise NotImplementedError(msg)
