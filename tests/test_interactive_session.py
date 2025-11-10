"""Tests for the InteractiveSandboxSession."""

import json
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_sandbox.const import SandboxBackend
from llm_sandbox.core.config import SessionConfig
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.exceptions import LanguageNotSupportedError, SandboxTimeoutError, UnsupportedBackendError
from llm_sandbox.interactive import (
    _INTERACTIVE_RUNNER_SCRIPT,
    InteractiveSandboxSession,
    KernelType,
)


def _stub_docker_session(self: InteractiveSandboxSession, *args: Any, **kwargs: Any) -> None:
    """Stub for SandboxDockerSession.__init__ to avoid Docker dependency."""
    del args, kwargs
    self.config = SessionConfig()
    self.container = MagicMock()
    self.is_open = True
    self.using_existing_container = False
    self.verbose = False
    self._python_env_dir = "/sandbox/.sandbox-venv"
    self._python_pip_cache_dir = "/sandbox/.sandbox-pip-cache"


def test_interactive_session_requires_docker_backend() -> None:
    """Interactive session only supports Docker backend."""
    with pytest.raises(UnsupportedBackendError):
        InteractiveSandboxSession(backend=SandboxBackend.KUBERNETES)


def test_interactive_session_requires_python_language() -> None:
    """Interactive session currently only supports Python."""
    with pytest.raises(LanguageNotSupportedError):
        InteractiveSandboxSession(lang="javascript")


@patch("llm_sandbox.interactive.SandboxDockerSession.__init__", new=_stub_docker_session)
def test_run_executes_code_and_returns_output() -> None:
    """Run should deliver stdout/stderr from the persistent interpreter."""
    session = InteractiveSandboxSession(kernel_type=KernelType.IPYTHON)
    session._commands_dir = "/sandbox/.interactive/commands"  # noqa: SLF001
    session._results_dir = "/sandbox/.interactive/results"  # noqa: SLF001
    session._runner_ready = True  # noqa: SLF001
    session.settings.timeout = 2
    session.settings.poll_interval = 0.01

    session.install = MagicMock()
    session._check_session_timeout = MagicMock()  # noqa: SLF001

    def fake_copy_from_runtime(src: str, dest: str) -> None:
        req_id = Path(src).stem.split("-")[-1]
        payload = {"id": req_id, "success": True, "stdout": "ok", "stderr": ""}
        Path(dest).write_text(json.dumps(payload), encoding="utf-8")

    session.copy_to_runtime = MagicMock()
    session.copy_from_runtime = MagicMock(side_effect=fake_copy_from_runtime)

    def fake_execute_command(command: str, **_: Any) -> ConsoleOutput:
        if command.startswith("test -f"):
            return ConsoleOutput(exit_code=0, stdout="", stderr="")
        return ConsoleOutput(exit_code=0, stdout="", stderr="")

    session.execute_command = MagicMock(side_effect=fake_execute_command)

    result = session.run("value = 42")

    assert result.stdout == "ok"
    assert result.exit_code == 0
    session.copy_to_runtime.assert_called_once()
    assert session.copy_from_runtime.call_count == 1


@patch("llm_sandbox.interactive.SandboxDockerSession.__init__", new=_stub_docker_session)
def test_run_times_out_when_result_missing() -> None:
    """Run should raise SandboxTimeoutError when the result never arrives."""
    session = InteractiveSandboxSession(kernel_type=KernelType.IPYTHON, timeout=0.2)
    session._commands_dir = "/sandbox/.interactive/commands"  # noqa: SLF001
    session._results_dir = "/sandbox/.interactive/results"  # noqa: SLF001
    session._runner_ready = True  # noqa: SLF001
    session.settings.poll_interval = 0.01

    session.install = MagicMock()
    session._check_session_timeout = MagicMock()  # noqa: SLF001
    session.copy_to_runtime = MagicMock()
    session.copy_from_runtime = MagicMock()
    session._interrupt_runner = MagicMock()  # noqa: SLF001

    def fake_execute_command(command: str, **_: Any) -> ConsoleOutput:
        if command.startswith("test -f"):
            return ConsoleOutput(exit_code=1, stdout="", stderr="")
        return ConsoleOutput(exit_code=0, stdout="", stderr="")

    session.execute_command = MagicMock(side_effect=fake_execute_command)

    with pytest.raises(SandboxTimeoutError):
        session.run("value = 1")

    session._interrupt_runner.assert_called_once()  # noqa: SLF001


def _start_local_runner(tmp_path: Path) -> tuple[Path, subprocess.Popen[bytes]]:
    runner_path = tmp_path / "runner.py"
    runner_path.write_text(_INTERACTIVE_RUNNER_SCRIPT, encoding="utf-8")

    channel_dir = tmp_path / "channel"
    channel_dir.mkdir(parents=True, exist_ok=True)
    ready_file = tmp_path / "ready"

    command = [
        sys.executable,
        str(runner_path),
        "--channel-dir",
        str(channel_dir),
        "--history-size",
        "100",
        "--poll-interval",
        "0.05",
        "--ready-file",
        str(ready_file),
    ]

    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # noqa: S603

    deadline = time.time() + 5
    while time.time() < deadline:
        if proc.poll() is not None:
            _stdout, stderr = proc.communicate()
            stderr_text = stderr.decode("utf-8", "ignore")
            message = f"Runner exited early: {stderr_text}"
            raise RuntimeError(message)
        if ready_file.exists():
            break
        time.sleep(0.05)
    else:
        proc.kill()
        _stdout, stderr = proc.communicate()
        stderr_text = stderr.decode("utf-8", "ignore")
        message = f"Runner did not start: {stderr_text}"
        raise TimeoutError(message)

    return channel_dir, proc


def _send_runner_command(channel_dir: Path, code: str, timeout: float = 5.0) -> dict[str, Any]:
    commands_dir = channel_dir / "commands"
    results_dir = channel_dir / "results"
    commands_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    request_id = uuid.uuid4().hex
    command_path = commands_dir / f"command-{request_id}.json"
    command_path.write_text(json.dumps({"id": request_id, "code": code}), encoding="utf-8")

    deadline = time.time() + timeout
    result_path = results_dir / f"result-{request_id}.json"
    while time.time() < deadline:
        if result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            result_path.unlink(missing_ok=True)
            return payload
        time.sleep(0.05)

    message = "Runner did not produce a result"
    raise TimeoutError(message)


def test_runner_persists_state_and_supports_magic(tmp_path: Path) -> None:
    """End-to-end test of the runner script to ensure state and magic commands work."""
    pytest.importorskip("IPython")
    channel_dir, proc = _start_local_runner(tmp_path)
    try:
        first = _send_runner_command(channel_dir, "value = 21 * 2")
        assert first["success"]

        second = _send_runner_command(channel_dir, "print(value)")
        assert "42" in second["stdout"]

        magic = _send_runner_command(channel_dir, "%who")
        assert "value" in magic["stdout"]
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.mark.integration
def test_interactive_session_state_with_docker() -> None:
    """Full integration test that starts a real Docker-backed interactive session."""
    docker = pytest.importorskip("docker")
    try:
        client = docker.from_env()
        client.ping()
    except docker.errors.DockerException as exc:  # type: ignore[attr-defined]
        pytest.skip(f"Docker daemon unavailable: {exc}")

    with InteractiveSandboxSession(lang="python", kernel_type="ipython", timeout=90) as session:
        first = session.run("value = 21 * 2")
        assert first.exit_code == 0

        second = session.run("print(value)")
        assert "42" in second.stdout

        magic = session.run("%who")
        assert "value" in magic.stdout

        pwd_result = session.run("%pwd")
        assert "/sandbox" in pwd_result.stdout, f"Unexpected pwd output: {pwd_result.stdout!r}"

        whoami = session.run("user = !whoami\nprint('\\n'.join(user))")
        assert "root" in whoami.stdout, f"Unexpected whoami output: {whoami.stdout!r}"
