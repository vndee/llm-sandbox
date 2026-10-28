"""Native Windows sessions using a pinned Microsoft MXC/PSEC helper.

The helper is bundled with a Windows distribution or supplied explicitly. This
adapter never downloads executables, switches to AppContainer, or retries a
failed command without native isolation.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import sys
import tarfile
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.core.config import SessionConfig
from llm_sandbox.core.session_base import BaseSession
from llm_sandbox.exceptions import (
    CommandEmptyError,
    CommandFailedError,
    LanguageNotSupportedError,
    LibraryInstallationNotSupportedError,
    MissingDependencyError,
    NotOpenSessionError,
    SandboxError,
    SandboxTimeoutError,
)
from llm_sandbox.windows.config import WindowsMxcConfig, explicit_environment, windows_policy_path
from llm_sandbox.windows.files import (
    PrivateWorkspace,
    runtime_relative,
    safe_copy,
    safe_mkdir,
    safe_read_file,
    safe_write_file,
    validate_existing,
)
from llm_sandbox.windows.process import ProcessSupervisor

if TYPE_CHECKING:
    from llm_sandbox.data import ConsoleOutput, StreamCallback

_PROTOCOL_VERSION = 1
_MAX_REQUEST_BYTES = 1_000_000
_HOME_DIRECTORY = ".home"
_SUPPORTED_LANGUAGES = {SupportedLanguage.PYTHON, SupportedLanguage.JAVASCRIPT}
_SESSION_OPTIONS = {
    "verbose",
    "default_timeout",
    "execution_timeout",
    "session_timeout",
    "skip_environment_setup",
    "encoding_errors",
    "security_policy",
}


def _require_windows() -> None:
    """Reject unsupported hosts without importing or running another sandbox backend."""
    if sys.platform != "win32":
        msg = "The windows-mxc backend requires native Windows with PSEC support"
        raise MissingDependencyError(msg)


def _resolve_executable(value: str, label: str) -> Path:
    """Validate an explicitly selected executable before running trusted native code."""
    path = Path(windows_policy_path(value))
    if path.suffix.casefold() != ".exe":
        msg = f"{label} must be an absolute .exe path"
        raise ValueError(msg)
    try:
        validate_existing(path, directory=False, allow_hardlinks=True)
    except FileNotFoundError as exc:
        msg = f"{label} was not found: {path}"
        raise MissingDependencyError(msg) from exc
    return path


def _host_environment() -> dict[str, str]:
    """Copy only Windows bootstrap values needed by the trusted native helper."""
    result: dict[str, str] = {}
    for key in ("SystemRoot", "SystemDrive", "LOCALAPPDATA"):
        value = next((value for name, value in os.environ.items() if name.casefold() == key.casefold()), None)
        if not value or "\0" in value:
            msg = f"Windows MXC requires a nonempty host {key} environment variable"
            raise MissingDependencyError(msg)
        if key != "SystemDrive":
            value = windows_policy_path(value)
        elif re.fullmatch(r"[A-Za-z]:", value) is None:
            msg = "SystemDrive must identify a local Windows drive"
            raise ValueError(msg)
        result[key] = value
    result["WINDIR"] = result["SystemRoot"]
    result["PATH"] = str(Path(result["SystemRoot"]) / "System32")
    return result


def _session_config(lang: SupportedLanguage, options: dict[str, Any]) -> tuple[SessionConfig, bool, dict[str, str]]:
    """Validate compatibility options without pretending container settings apply."""
    for name in ("image", "dockerfile", "container_id", "runtime_configs", "keep_template", "commit_container"):
        if options.pop(name, None):
            msg = f"Windows MXC does not support {name}; configure sandbox_config instead"
            raise ValueError(msg)
    if options.pop("workdir", None) not in {None, "/sandbox"}:
        msg = "Windows MXC owns a private workspace; use session.workspace after opening it"
        raise ValueError(msg)
    if options.pop("libraries", None):
        raise LibraryInstallationNotSupportedError(lang)
    stream = options.pop("stream", False)
    if not isinstance(stream, bool):
        msg = "stream must be a boolean"
        raise TypeError(msg)
    environment = explicit_environment(options.pop("env_vars", None) or {})
    if unknown := set(options) - _SESSION_OPTIONS:
        msg = f"Unknown Windows MXC options: {', '.join(sorted(unknown))}"
        raise TypeError(msg)
    options.setdefault("default_timeout", 60.0)
    config = SessionConfig(lang=lang, workdir="/sandbox", **options)
    for name in ("default_timeout", "execution_timeout", "session_timeout"):
        value = getattr(config, name)
        if value is not None and (value <= 0 or not math.isfinite(value)):
            msg = f"{name} must be finite and positive, or None"
            raise ValueError(msg)
    return config, stream, environment


class WindowsMxcSession(BaseSession):
    """Run Python, JavaScript, or Windows commands inside native PSEC isolation.

    Files persist in one private workspace while every command receives a fresh
    native sandbox and Job Object. All descendants stop when that command exits.
    Commands and file operations are serialized; ``close`` cancels running work.
    Only regular files and directories are supported by the workspace APIs.
    """

    def __init__(
        self,
        *,
        sandbox_config: WindowsMxcConfig | None = None,
        mxc_path: str | None = None,
        lang: str = SupportedLanguage.PYTHON,
        **kwargs: Any,
    ) -> None:
        """Select installed native tools without executing payload code on the host."""
        _require_windows()
        language = SupportedLanguage(lang)
        if language not in _SUPPORTED_LANGUAGES:
            raise LanguageNotSupportedError(language)
        if sandbox_config is not None and not isinstance(sandbox_config, WindowsMxcConfig):
            msg = "sandbox_config must be a WindowsMxcConfig instance"
            raise TypeError(msg)
        config, stream, environment = _session_config(language, kwargs)
        super().__init__(config)
        self.sandbox_config = sandbox_config or WindowsMxcConfig()
        self.stream = stream
        self._explicit_env = environment
        self._bootstrap_env = _host_environment()
        helper_path = mxc_path or str(Path(__file__).parent / "bin" / "llm-sandbox-mxc.exe")
        try:
            self._helper = _resolve_executable(helper_path, "MXC helper")
        except MissingDependencyError as exc:
            msg = (
                "Install the Windows distribution containing llm-sandbox-mxc.exe, or build "
                "native/windows-mxc and pass mxc_path. Native helpers are never downloaded at runtime."
            )
            raise MissingDependencyError(msg) from exc
        self._system_root = Path(self._bootstrap_env["SystemRoot"])
        self._shell = _resolve_executable(str(self._system_root / "System32" / "cmd.exe"), "Windows command shell")
        self._runtime = self._resolve_runtime(language)
        self._read_roots = self._resolve_read_roots(language)
        self._operation_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._supervisor: ProcessSupervisor | None = None
        self._workspace_owner: PrivateWorkspace | None = None
        self.workspace: Path
        self._generation = 0
        self._closing = False
        self._executing = False
        self._session_deadline: float | None = None

    def _resolve_runtime(self, language: SupportedLanguage) -> Path:
        executable: str | None
        if language == SupportedLanguage.PYTHON:
            executable = self.sandbox_config.python_executable or sys.executable
        else:
            executable = self.sandbox_config.javascript_executable or shutil.which("node.exe")
            if executable is None:
                msg = "Install Node.js or configure javascript_executable with its absolute node.exe path"
                raise MissingDependencyError(msg)
        return _resolve_executable(executable, f"{language} runtime")

    def _resolve_read_roots(self, language: SupportedLanguage) -> tuple[str, ...]:
        paths = [self._system_root, self._runtime.parent]
        if language == SupportedLanguage.PYTHON and self.sandbox_config.python_executable is None:
            paths.extend([Path(sys.prefix), Path(sys.base_prefix)])
        paths.extend(Path(value) for value in self.sandbox_config.read_only_paths)
        roots: dict[str, str] = {}
        for path in paths:
            normalized = windows_policy_path(str(path))
            validate_existing(path)
            roots.setdefault(normalized.casefold(), normalized)
        return tuple(roots.values())

    def open(self) -> None:
        """Create the private workspace and verify native isolation before use."""
        with self._operation_lock:
            with self._state_lock:
                if self.is_open and not self._closing:
                    return
                if self._executing:
                    msg = "Windows MXC cannot reopen from an active command callback"
                    raise SandboxError(msg)
            if self._workspace_owner is not None:
                self.close()
            with self._state_lock:
                self._generation += 1
                self._closing = False
            owner = PrivateWorkspace()
            workspace = Path(owner.name)
            # Keep ownership before fallible setup so failed cleanup remains retryable.
            with self._state_lock:
                self._workspace_owner = owner
                self.workspace = workspace
            try:
                self._initialize_workspace(workspace)
                self._start_session_timer()
                self._probe()
                # Prove the selected toolchain actually starts through PSEC.
                self._check_runtime_startup()
            except BaseException as error:
                try:
                    self.close()
                except Exception as cleanup_error:
                    raise error from cleanup_error
                raise

    def _initialize_workspace(self, workspace: Path) -> None:
        """Prepare private directories and publish a supervisor unless opening was cancelled."""
        safe_mkdir(workspace / _HOME_DIRECTORY)
        safe_mkdir(workspace / ".tmp")
        with self._state_lock:
            if self._closing:
                raise NotOpenSessionError
            self.config.workdir = str(workspace)
            self.container = workspace
            self.is_open = True
            self._supervisor = ProcessSupervisor(
                max_output_bytes=self.sandbox_config.max_output_bytes,
                encoding_errors=self.config.encoding_errors,
            )

    def _check_runtime_startup(self) -> None:
        output = self.execute_argv([str(self._runtime), "--version"])
        if output.exit_code:
            command = "MXC runtime startup"
            raise CommandFailedError(command, output.exit_code, output.stderr or output.stdout)

    def _probe(self) -> None:
        supervisor = self._require_open()
        output = supervisor.run(
            [str(self._helper), "--probe"],
            env=self._launcher_environment(),
            cwd=self._helper.parent,
            timeout=self._execution_timeout(None),
        )
        try:
            result = json.loads(output.stdout)
        except (ValueError, TypeError) as exc:
            msg = "MXC helper returned an invalid capability response"
            raise SandboxError(msg) from exc
        if (
            output.exit_code
            or not isinstance(result, dict)
            or type(result.get("protocol")) is not int
            or result.get("protocol") != _PROTOCOL_VERSION
            or result.get("available") is not True
        ):
            msg = "Native MXC/PSEC is unavailable or the helper protocol is incompatible; execution is refused"
            raise MissingDependencyError(msg)
        if self.sandbox_config.deny_paths and result.get("deny_paths") is not True:
            msg = "This Windows host cannot enforce the requested native MXC deny paths"
            raise SandboxError(msg)

    def _launcher_environment(self) -> dict[str, str]:
        return {
            **self._bootstrap_env,
            "TEMP": str(self.workspace / ".tmp"),
            "TMP": str(self.workspace / ".tmp"),
        }

    def _payload_environment(self) -> dict[str, str]:
        environment = {
            **self._launcher_environment(),
            "PATH": f"{self._runtime.parent};{self._bootstrap_env['PATH']}",
            "HOME": str(self.workspace / _HOME_DIRECTORY),
            "USERPROFILE": str(self.workspace / _HOME_DIRECTORY),
            "APPDATA": str(self.workspace / _HOME_DIRECTORY),
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
        for overrides in (explicit_environment(self.sandbox_config.env), self._explicit_env):
            for key, value in overrides.items():
                environment = {name: item for name, item in environment.items() if name.casefold() != key.casefold()}
                environment[key] = value
        return environment

    def _require_open(self) -> ProcessSupervisor:
        with self._state_lock:
            if not self.is_open or self._closing or self._supervisor is None:
                raise NotOpenSessionError
            return self._supervisor

    def _require_idle(self) -> ProcessSupervisor:
        supervisor = self._require_open()
        if self._executing:
            msg = "Sandbox operations cannot be reentered from an active command callback"
            raise SandboxError(msg)
        return supervisor

    def _start_session_timer(self) -> None:
        self._session_start_time = time.time()
        self._session_deadline = None
        if self.config.session_timeout is not None:
            self._session_deadline = time.monotonic() + self.config.session_timeout
            generation = self._generation
            self._session_timer = threading.Timer(self.config.session_timeout, self._close_generation, [generation])
            self._session_timer.daemon = True
            self._session_timer.start()

    def close(self) -> None:
        """Permanently stop the active supervisor before deleting untrusted files."""
        with self._state_lock:
            generation = self._generation
        self._close_generation(generation)

    def _close_generation(self, generation: int) -> None:
        with self._state_lock:
            if generation != self._generation:
                return
            self._closing = True
            self.is_open = False
            supervisor = self._supervisor
        if supervisor is not None:
            supervisor.close()
        with self._operation_lock:
            with self._state_lock:
                if generation != self._generation:
                    return
                self._stop_session_timer()
                self._supervisor = None
                self.container = None
                owner = self._workspace_owner
            if owner is not None:
                owner.cleanup()
                self._workspace_owner = None

    def _execution_timeout(self, timeout: float | None) -> float | None:
        selected = timeout if timeout is not None else self.config.get_execution_timeout()
        if selected is not None and (selected <= 0 or not math.isfinite(selected)):
            msg = "Execution timeout must be finite and positive, or None"
            raise ValueError(msg)
        if self._session_deadline is not None:
            remaining = self._session_deadline - time.monotonic()
            if remaining <= 0:
                msg = "Windows MXC session exceeded its deadline"
                raise SandboxTimeoutError(msg)
            selected = remaining if selected is None else min(selected, remaining)
        return selected

    def _runtime_path(self, path: str) -> Path:
        relative = runtime_relative(path, str(self.workspace))
        return self.workspace.joinpath(*relative.parts)

    def execute_argv(
        self,
        argv: list[str],
        *,
        workdir: str | None = None,
        timeout: float | None = None,
        on_stdout: StreamCallback | None = None,
        on_stderr: StreamCallback | None = None,
    ) -> ConsoleOutput:
        """Execute exact arguments after PSEC applies, without an intervening shell."""
        if not argv:
            raise CommandEmptyError
        if any(not isinstance(value, str) or "\0" in value for value in argv):
            msg = "Command arguments must be strings without NUL"
            raise ValueError(msg)
        executable = windows_policy_path(argv[0])
        if Path(executable).suffix.casefold() != ".exe":
            msg = "execute_argv requires an absolute .exe executable; use execute_command for batch commands"
            raise ValueError(msg)
        with self._operation_lock:
            supervisor = self._require_idle()
            cwd = self.workspace if workdir is None else self._runtime_path(workdir)
            validate_existing(cwd, directory=True)
            request = {
                "protocol": _PROTOCOL_VERSION,
                "argv": [executable, *argv[1:]],
                "cwd": str(cwd),
                "read_only_paths": list(self._read_roots),
                "writable_paths": [str(self.workspace)],
                "deny_paths": list(self.sandbox_config.deny_paths),
                "network": self.sandbox_config.network,
                "env": self._payload_environment(),
            }
            encoded = json.dumps(request, ensure_ascii=False).encode("utf-8")
            if len(encoded) > _MAX_REQUEST_BYTES:
                msg = "MXC command request exceeds the native helper protocol limit"
                raise ValueError(msg)
            if self.stream and self.verbose:
                on_stdout = self._log if on_stdout is None else on_stdout
                on_stderr = self._log if on_stderr is None else on_stderr
            self._executing = True
            try:
                return supervisor.run(
                    [str(self._helper), "--run"],
                    input=encoded,
                    env=self._launcher_environment(),
                    cwd=self._helper.parent,
                    timeout=self._execution_timeout(timeout),
                    on_stdout=on_stdout,
                    on_stderr=on_stderr,
                )
            except BaseException:
                self.close()
                raise
            finally:
                self._executing = False

    def execute_command(
        self,
        command: str,
        workdir: str | None = None,
        on_stdout: StreamCallback | None = None,
        on_stderr: StreamCallback | None = None,
    ) -> ConsoleOutput:
        """Run a staged Windows batch script using cmd.exe inside native isolation."""
        if not command:
            raise CommandEmptyError
        if "\0" in command:
            msg = "Commands cannot contain NUL"
            raise ValueError(msg)
        with self._operation_lock:
            self._require_idle()
            cwd = self.workspace if workdir is None else self._runtime_path(workdir)
            validate_existing(cwd, directory=True)
            script = cwd / f"{uuid.uuid4().hex}.cmd"
            content = "@echo off\r\nchcp 65001 >nul\r\n" + command + "\r\n"
            safe_write_file(script, content.encode("utf-8"))
            return self.execute_argv(
                [str(self._shell), "/d", "/c", script.name],
                workdir=workdir,
                on_stdout=on_stdout,
                on_stderr=on_stderr,
            )

    def run(
        self,
        code: str,
        libraries: list | None = None,
        timeout: float | None = None,
        on_stdout: StreamCallback | None = None,
        on_stderr: StreamCallback | None = None,
    ) -> ConsoleOutput:
        """Stage Python or JavaScript source and launch the selected native interpreter."""
        with self._operation_lock:
            self._require_idle()
            self.install(libraries)
            extension = "py" if self.config.lang == SupportedLanguage.PYTHON else "js"
            script = self.workspace / f"{uuid.uuid4().hex}.{extension}"
            safe_write_file(script, code.encode("utf-8"))
            return self.execute_argv(
                [str(self._runtime), str(script)],
                timeout=timeout,
                on_stdout=on_stdout,
                on_stderr=on_stderr,
            )

    def environment_setup(self) -> None:
        """Use the installed runtime without changing its packages or host files."""

    def install(self, libraries: list[str] | None = None) -> None:
        """Reject automatic package installation until a native package workflow is implemented."""
        if libraries:
            raise LibraryInstallationNotSupportedError(self.config.lang)

    @property
    def python_executable_path(self) -> str:
        """Return the selected native runtime rather than a POSIX virtualenv path."""
        return str(self._runtime)

    def copy_to_runtime(self, src: str, dest: str) -> None:
        """Copy into the private workspace while no payload process is running."""
        with self._operation_lock:
            self._require_idle()
            safe_copy(Path(src).absolute(), self._runtime_path(dest))

    def copy_from_runtime(self, src: str, dest: str) -> None:
        """Copy regular workspace files without following Windows reparse points."""
        with self._operation_lock:
            self._require_idle()
            safe_copy(self._runtime_path(src), Path(dest).absolute())

    def get_archive(self, path: str) -> tuple[bytes, dict]:
        """Export one bounded regular file in the shared artifact archive format."""
        with self._operation_lock:
            self._require_idle()
            source = self._runtime_path(path)
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
            self._require_idle()
            safe_mkdir(self._runtime_path(path))

    def _ensure_ownership(self, paths: list[str]) -> None:
        """Validate workspace destinations; native PSEC retains the launching user's identity."""
        for path in paths:
            self._runtime_path(path)

    def _handle_timeout(self) -> None:
        self.close()

    def _connect_to_existing_container(self, container_id: str) -> None:
        msg = "Windows MXC cannot attach to existing containers"
        raise NotImplementedError(msg)

    def _process_non_stream_output(self, output: Any) -> tuple[str, str]:
        msg = "Windows MXC output is decoded by the process supervisor"
        raise NotImplementedError(msg)

    def _process_stream_output(
        self,
        output: Any,
        on_stdout: StreamCallback | None = None,
        on_stderr: StreamCallback | None = None,
    ) -> tuple[str, str]:
        msg = "Windows MXC streams are delivered directly by the process supervisor"
        raise NotImplementedError(msg)
