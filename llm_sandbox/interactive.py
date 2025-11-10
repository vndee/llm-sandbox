"""Interactive sandbox session backed by a persistent IPython runtime."""

from __future__ import annotations

import json
import shlex
import tempfile
import textwrap
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_sandbox.const import SandboxBackend, SupportedLanguage
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.docker import SandboxDockerSession
from llm_sandbox.exceptions import (
    ContainerError,
    LanguageNotSupportedError,
    NotOpenSessionError,
    SandboxTimeoutError,
)

from .const import StrEnum
from .exceptions import UnsupportedBackendError

RUNTIME_START_TIMEOUT = 30.0
RESULT_POLL_INTERVAL = 0.2


class KernelType(StrEnum):
    """Supported kernel types for interactive sessions."""

    IPYTHON = "ipython"


@dataclass(slots=True)
class InteractiveSettings:
    """Configuration for interactive execution.

    Attributes:
        kernel_type: Kernel backend used for execution (default: ``KernelType.IPYTHON``).
        max_memory: Optional memory limit passed to the runtime backend.
        history_size: Number of cached execution entries retained in the kernel (default: 1000).
        timeout: Default per-cell timeout in seconds; ``None`` means no timeout (default: 300).
        poll_interval: Interval in seconds for polling runner status files (default: 0.1).

    """

    kernel_type: KernelType = KernelType.IPYTHON
    max_memory: str | None = None
    history_size: int = 1000
    timeout: float | None = 300.0
    poll_interval: float = 0.1

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.history_size < 0:
            msg = "history_size must be non-negative"
            raise ValueError(msg)
        if self.poll_interval <= 0:
            msg = "poll_interval must be positive"
            raise ValueError(msg)


class InteractiveSandboxSession(SandboxDockerSession):
    """Interactive sandbox session that preserves interpreter state across runs."""

    def __init__(
        self,
        *,
        backend: SandboxBackend = SandboxBackend.DOCKER,
        lang: str | SupportedLanguage = SupportedLanguage.PYTHON,
        kernel_type: KernelType | str = KernelType.IPYTHON,
        max_memory: str | None = "1GB",
        history_size: int = 1000,
        timeout: float | None = 300.0,
        runtime_configs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the interactive session."""
        if backend != SandboxBackend.DOCKER:
            raise UnsupportedBackendError(backend=backend)

        lang_value = str(lang)
        if lang_value.lower() != SupportedLanguage.PYTHON:
            raise LanguageNotSupportedError(lang_value)

        kernel_enum = KernelType(kernel_type)
        runtime_configs = runtime_configs.copy() if runtime_configs else {}
        if max_memory:
            runtime_configs.setdefault("mem_limit", max_memory)

        if timeout is not None and "execution_timeout" not in kwargs:
            kwargs["execution_timeout"] = timeout

        self.settings = InteractiveSettings(
            kernel_type=kernel_enum,
            max_memory=max_memory,
            history_size=history_size,
            timeout=timeout,
        )

        kwargs.setdefault("lang", SupportedLanguage.PYTHON)
        super().__init__(runtime_configs=runtime_configs, **kwargs)

        workdir = self.config.workdir.rstrip("/")
        self._channel_dir = f"{workdir}/.interactive"
        self._commands_dir = f"{self._channel_dir}/commands"
        self._results_dir = f"{self._channel_dir}/results"
        self._runner_script_path = f"{self._channel_dir}/runner.py"
        self._ready_file = f"{self._channel_dir}/ready"
        self._pid_file = f"{self._channel_dir}/runner.pid"
        self._log_file = f"{self._channel_dir}/runner.log"
        self._runner_ready = False
        self._python_exec_path = self.python_executable_path
        self._pip_exec_path = self.pip_executable_path
        self._pip_cache_dir = self.pip_cache_dir_path

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def open(self) -> None:
        """Open interactive session and prepare runtime assets."""
        super().open()
        try:
            self._bootstrap_runtime()
        except Exception:
            super().close()
            raise

    def close(self) -> None:
        """Close the interactive session."""
        self._stop_runner_process()
        super().close()

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def run(self, code: str, libraries: list[str] | None = None, timeout: float | None = None) -> ConsoleOutput:
        """Execute code in the persistent interpreter context."""
        if not self.container or not self.is_open:
            raise NotOpenSessionError

        if not self._runner_ready:
            msg = "Interactive runtime is not ready"
            raise ContainerError(msg)

        self._check_session_timeout()
        actual_timeout = timeout or self.settings.timeout or self.config.get_execution_timeout()
        self.install(libraries)

        request_id = uuid.uuid4().hex
        command_path = f"{self._commands_dir}/command-{request_id}.json"
        result_path = f"{self._results_dir}/result-{request_id}.json"

        self._write_remote_command(command_path, request_id, code)

        if not self._wait_for_remote_file(result_path, actual_timeout):
            self._interrupt_runner()
            msg = f"Interactive execution timed out after {actual_timeout} seconds"
            raise SandboxTimeoutError(msg, timeout_duration=actual_timeout)

        payload = self._read_remote_result(result_path)
        self.execute_command(f"rm -f {result_path}")
        exit_code = 0 if payload.get("success") else 1
        stdout = payload.get("stdout", "")
        stderr = payload.get("stderr", "")
        return ConsoleOutput(exit_code=exit_code, stdout=stdout, stderr=stderr)

    # ------------------------------------------------------------------ #
    # Runtime bootstrap helpers
    # ------------------------------------------------------------------ #
    def _bootstrap_runtime(self) -> None:
        self._ensure_runtime_dependencies()
        self._upload_runner_script()
        self._start_runner_process()

    def _ensure_runtime_dependencies(self) -> None:
        packages = ["ipython"]
        install_command = (
            f"{self._pip_exec_path} install --quiet --disable-pip-version-check "
            f"--cache-dir {self._pip_cache_dir} {' '.join(packages)}"
        )
        result = self.execute_command(install_command)
        if result.exit_code:
            msg = f"Failed to install interactive dependencies: {result.stderr or result.stdout}"
            raise ContainerError(msg)

        self.execute_commands([
            (f"mkdir -p {self._commands_dir}", None),
            (f"mkdir -p {self._results_dir}", None),
        ])

    def _upload_runner_script(self) -> None:
        script_content = _INTERACTIVE_RUNNER_SCRIPT
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as script_file:
            script_file.write(script_content)
            temp_path = script_file.name

        try:
            self.copy_to_runtime(temp_path, self._runner_script_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _start_runner_process(self) -> None:
        args = [
            self._python_exec_path,
            self._runner_script_path,
            "--channel-dir",
            self._channel_dir,
            "--history-size",
            str(self.settings.history_size),
            "--poll-interval",
            str(self.settings.poll_interval),
            "--ready-file",
            self._ready_file,
        ]
        runner_cmd = " ".join(shlex.quote(arg) for arg in args)
        inner_command = (
            f"rm -f {self._ready_file} {self._pid_file}; "
            f"nohup {runner_cmd} > {self._log_file} 2>&1 & "
            f"echo $! > {self._pid_file}"
        )
        launch_command = f"/bin/sh -c {shlex.quote(inner_command)}"
        result = self.execute_command(launch_command)
        if result.exit_code:
            msg = "Failed to start interactive runtime"
            raise ContainerError(msg)

        if not self._wait_for_remote_file(self._ready_file, RUNTIME_START_TIMEOUT):
            msg = "Interactive runtime did not signal readiness in time"
            raise ContainerError(msg)

        self._runner_ready = True

    # ------------------------------------------------------------------ #
    # Runner process management
    # ------------------------------------------------------------------ #
    def _stop_runner_process(self) -> None:
        if not self._runner_ready:
            return

        with suppress(Exception):
            self.execute_command(
                f"if [ -f {self._pid_file} ]; then "
                f"kill $(cat {self._pid_file}) >/dev/null 2>&1 || true; "
                f"rm -f {self._pid_file}; "
                "fi"
            )
        self._runner_ready = False

    def _interrupt_runner(self) -> None:
        with suppress(Exception):
            self.execute_command(
                f"if [ -f {self._pid_file} ]; then kill -SIGINT $(cat {self._pid_file}) >/dev/null 2>&1 || true; fi"
            )

    # ------------------------------------------------------------------ #
    # Remote file helpers
    # ------------------------------------------------------------------ #
    def _write_remote_command(self, remote_path: str, request_id: str, code: str) -> None:
        payload = {"id": request_id, "code": code}
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp_file:
            json.dump(payload, tmp_file)
            tmp_path = tmp_file.name

        try:
            self.copy_to_runtime(tmp_path, remote_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _read_remote_result(self, remote_path: str) -> dict[str, Any]:
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp_file:
            local_path = tmp_file.name

        try:
            self.copy_from_runtime(remote_path, local_path)
            with Path(local_path).open("r", encoding="utf-8") as result_file:
                return json.load(result_file)
        finally:
            Path(local_path).unlink(missing_ok=True)

    def _wait_for_remote_file(self, remote_path: str, timeout: float | None) -> bool:
        deadline = time.monotonic() + timeout if timeout else None

        while True:
            status = self.execute_command(f"test -f {remote_path}")
            if status.exit_code == 0:
                return True

            if deadline and time.monotonic() >= deadline:
                return False

            time.sleep(self.settings.poll_interval or RESULT_POLL_INTERVAL)


_INTERACTIVE_RUNNER_SCRIPT = textwrap.dedent(
    """
    import argparse
    import io
    import json
    import sys
    import time
    import traceback
    from contextlib import redirect_stderr, redirect_stdout
    from pathlib import Path

    from IPython.core.interactiveshell import InteractiveShell


    def _safe_read(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


    def _write_json(path: Path, payload: dict) -> None:
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        tmp_path.replace(path)


    def main() -> None:
        parser = argparse.ArgumentParser(description="Interactive IPython runner")
        parser.add_argument("--channel-dir", required=True)
        parser.add_argument("--history-size", type=int, default=1000)
        parser.add_argument("--poll-interval", type=float, default=0.1)
        parser.add_argument("--ready-file", required=True)
        args = parser.parse_args()

        base_dir = Path(args.channel_dir)
        commands_dir = base_dir / "commands"
        results_dir = base_dir / "results"
        commands_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        shell = InteractiveShell.instance()
        shell.cache_size = max(shell.cache_size, args.history_size or 0)
        shell.user_ns.setdefault("__name__", "__main__")
        shell.user_ns.setdefault("__package__", None)

        Path(args.ready_file).write_text("ready", encoding="utf-8")

        while True:
            command_files = sorted(commands_dir.glob("command-*.json"))
            if not command_files:
                time.sleep(args.poll_interval)
                continue

            for command_file in command_files:
                try:
                    payload = _safe_read(command_file)
                except json.JSONDecodeError:
                    time.sleep(0.05)
                    continue

                code = payload.get("code", "")
                request_id = payload.get("id")
                stdout_buffer = io.StringIO()
                stderr_buffer = io.StringIO()
                success = True

                try:
                    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                        result = shell.run_cell(code, store_history=True, silent=False)
                    success = bool(result.success)
                except SystemExit:
                    raise
                except KeyboardInterrupt:
                    success = False
                    if not stderr_buffer.getvalue():
                        stderr_buffer.write("KeyboardInterrupt\\n")
                except Exception:  # noqa: BLE001
                    success = False
                    traceback.print_exc(file=stderr_buffer)

                result_payload = {
                    "id": request_id,
                    "success": success,
                    "stdout": stdout_buffer.getvalue(),
                    "stderr": stderr_buffer.getvalue(),
                }
                _write_json(results_dir / f"result-{request_id}.json", result_payload)
                command_file.unlink(missing_ok=True)


    if __name__ == "__main__":
        try:
            main()
        except KeyboardInterrupt:
            sys.exit(0)
    """
)
