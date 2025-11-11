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
from llm_sandbox.exceptions import (
    ContainerError,
    LanguageNotSupportedError,
    SandboxTimeoutError,
    UnsupportedBackendError,
)
from llm_sandbox.interactive import (
    _INTERACTIVE_RUNNER_SCRIPT,
    InteractiveSandboxSession,
    KernelType,
)


def _set_private(obj: Any, name: str, value: Any) -> None:
    """Set private attributes without referencing them directly."""
    object.__setattr__(obj, name, value)


def _call_private(obj: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    """Call private methods without referencing them directly."""
    return getattr(obj, name)(*args, **kwargs)


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


def test_interactive_settings_rejects_negative_history() -> None:
    """InteractiveSettings rejects negative history_size values."""
    from llm_sandbox.interactive import InteractiveSettings

    with pytest.raises(ValueError, match="history_size must be non-negative"):
        InteractiveSettings(history_size=-1)


def test_interactive_settings_rejects_non_positive_poll_interval() -> None:
    """InteractiveSettings rejects non-positive poll_interval values."""
    from llm_sandbox.interactive import InteractiveSettings

    with pytest.raises(ValueError, match="poll_interval must be positive"):
        InteractiveSettings(poll_interval=0)


@patch("llm_sandbox.interactive.SandboxDockerSession.__init__", new=_stub_docker_session)
def test_run_executes_code_and_returns_output() -> None:
    """Run should deliver stdout/stderr from the persistent interpreter."""
    session = InteractiveSandboxSession(kernel_type=KernelType.IPYTHON)
    _set_private(session, "_commands_dir", "/sandbox/.interactive/commands")
    _set_private(session, "_results_dir", "/sandbox/.interactive/results")
    _set_private(session, "_runner_ready", value=True)
    session.settings.timeout = 2
    session.settings.poll_interval = 0.01

    session.install = MagicMock()
    _set_private(session, "_check_session_timeout", MagicMock())

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
    _set_private(session, "_commands_dir", "/sandbox/.interactive/commands")
    _set_private(session, "_results_dir", "/sandbox/.interactive/results")
    _set_private(session, "_runner_ready", value=True)
    session.settings.poll_interval = 0.01

    session.install = MagicMock()
    _set_private(session, "_check_session_timeout", MagicMock())
    session.copy_to_runtime = MagicMock()
    session.copy_from_runtime = MagicMock()
    interrupt_mock = MagicMock()
    _set_private(session, "_interrupt_runner", interrupt_mock)

    def fake_execute_command(command: str, **_: Any) -> ConsoleOutput:
        if command.startswith("test -f"):
            return ConsoleOutput(exit_code=1, stdout="", stderr="")
        return ConsoleOutput(exit_code=0, stdout="", stderr="")

    session.execute_command = MagicMock(side_effect=fake_execute_command)

    with pytest.raises(SandboxTimeoutError):
        session.run("value = 1")

    interrupt_mock.assert_called_once()


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


@patch("llm_sandbox.interactive.SandboxDockerSession.__init__", new=_stub_docker_session)
@patch("llm_sandbox.interactive.SandboxDockerSession.open")
@patch("llm_sandbox.interactive.SandboxDockerSession.close")
def test_open_cleans_up_on_bootstrap_failure(mock_close: MagicMock, mock_open: MagicMock) -> None:
    """open() should tear down the container if bootstrap fails."""
    session = InteractiveSandboxSession()
    mock_open.return_value = None
    with (
        patch.object(session, "_bootstrap_runtime", side_effect=ContainerError("boom")),
        pytest.raises(ContainerError),
    ):
        session.open()
    mock_close.assert_called_once()


@patch("llm_sandbox.interactive.SandboxDockerSession.__init__", new=_stub_docker_session)
def test_run_requires_ready_runner() -> None:
    """run() should error when the interactive runner is not ready."""
    session = InteractiveSandboxSession()
    session.container = object()
    session.is_open = True
    _set_private(session, "_runner_ready", value=False)
    session.install = MagicMock()
    with pytest.raises(ContainerError):
        session.run("print('hello')")


@patch("llm_sandbox.interactive.SandboxDockerSession.__init__", new=_stub_docker_session)
def test_ensure_runtime_dependencies_failure() -> None:
    """_ensure_runtime_dependencies raises ContainerError when pip install fails."""
    session = InteractiveSandboxSession()
    session.execute_command = MagicMock(return_value=ConsoleOutput(exit_code=1, stderr="fail"))

    attr_name = "_ensure_runtime_dependencies"
    method = getattr(InteractiveSandboxSession, attr_name)
    with pytest.raises(ContainerError):
        method(session)


@patch("llm_sandbox.interactive.SandboxDockerSession.__init__", new=_stub_docker_session)
def test_wait_for_remote_file_timeout() -> None:
    """_wait_for_remote_file returns False when the timeout expires."""
    session = InteractiveSandboxSession()
    session.settings.poll_interval = 0.01
    session.execute_command = MagicMock(return_value=ConsoleOutput(exit_code=1))
    attr_name = "_wait_for_remote_file"
    wait_method = getattr(InteractiveSandboxSession, attr_name)
    assert wait_method(session, "missing", timeout=0.02) is False


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
