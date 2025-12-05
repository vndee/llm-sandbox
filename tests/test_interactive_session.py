# mypy: disable-error-code="method-assign"
"""Tests for the InteractiveSandboxSession."""

import json
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, cast
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
from llm_sandbox.interactive import _INTERACTIVE_RUNNER_SCRIPT, InteractiveSandboxSession, KernelType


def _set_private(obj: Any, name: str, value: Any) -> None:
    """Set private attributes without referencing them directly."""
    object.__setattr__(obj, name, value)


def _call_private(obj: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    """Call private methods without referencing them directly."""
    return getattr(obj, name)(*args, **kwargs)


def _stub_backend_session(*args: Any, **kwargs: Any) -> MagicMock:
    """Stub for _create_backend_session to avoid backend dependencies."""
    del args, kwargs
    mock_session = MagicMock()
    mock_session.config = SessionConfig()
    mock_session.container = MagicMock()
    mock_session.is_open = True
    mock_session.using_existing_container = False
    mock_session.verbose = False
    mock_session.python_executable_path = "/sandbox/.sandbox-venv/bin/python"
    mock_session.pip_executable_path = "/sandbox/.sandbox-venv/bin/pip"
    mock_session.pip_cache_dir_path = "/sandbox/.sandbox-pip-cache"
    mock_session.container_api = MagicMock()
    return mock_session


def test_interactive_session_requires_supported_backend() -> None:
    """Interactive session supports Docker, Podman, and Kubernetes backends."""
    # Micromamba is not supported for interactive sessions
    with pytest.raises(UnsupportedBackendError):
        InteractiveSandboxSession(backend=SandboxBackend.MICROMAMBA)


def test_interactive_session_requires_python_language() -> None:
    """Interactive session currently only supports Python."""
    with pytest.raises(LanguageNotSupportedError):
        InteractiveSandboxSession(lang="javascript")


def test_create_backend_session_filters_runtime_configs_for_kubernetes() -> None:
    """_create_backend_session should filter out runtime_configs for Kubernetes backend."""
    from llm_sandbox.interactive import _create_backend_session

    # Patch SandboxKubernetesSession to verify it's not called with runtime_configs
    with patch("llm_sandbox.kubernetes.SandboxKubernetesSession") as mock_k8s_class:
        mock_k8s_session = MagicMock()
        mock_k8s_session.config = SessionConfig()
        mock_k8s_class.return_value = mock_k8s_session

        # Simulate the case where runtime_configs might be in kwargs (e.g., if someone passes it incorrectly)
        # We'll manually put it in kwargs to test the filtering
        kwargs_with_runtime_configs: dict[str, Any] = {
            "lang": "python",
            "runtime_configs": {"mem_limit": "1GB"},  # This should be filtered out
        }

        # Call _create_backend_session with runtime_configs only in kwargs (not as separate param)
        # This tests that the filtering works when runtime_configs is in kwargs
        result = _create_backend_session(
            backend=SandboxBackend.KUBERNETES,
            **kwargs_with_runtime_configs,  # runtime_configs in here should be filtered out
        )

        # Verify SandboxKubernetesSession was called
        mock_k8s_class.assert_called_once()

        # Verify runtime_configs was NOT in the kwargs passed to SandboxKubernetesSession
        call_kwargs = mock_k8s_class.call_args[1] if mock_k8s_class.call_args else {}
        assert "runtime_configs" not in call_kwargs, "runtime_configs should be filtered out for Kubernetes"

        # Verify other kwargs were passed through
        assert call_kwargs.get("lang") == "python"
        assert result == mock_k8s_session


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_interactive_session_kubernetes_with_runtime_configs() -> None:
    """Interactive session with Kubernetes backend should handle runtime_configs gracefully."""
    # This should not raise a TypeError even though Kubernetes doesn't support runtime_configs
    session = InteractiveSandboxSession(
        backend=SandboxBackend.KUBERNETES,
        runtime_configs={"mem_limit": "1GB"},  # runtime_configs should be ignored for Kubernetes
    )

    # Verify the session was created successfully
    assert session._backend_session is not None


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


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_run_executes_code_and_returns_output() -> None:
    """Run should deliver stdout/stderr from the persistent interpreter."""
    session = InteractiveSandboxSession(kernel_type=KernelType.IPYTHON)
    # Set up session state as if it was opened
    session.container = session._backend_session.container
    session.is_open = True
    session.container_api = session._backend_session.container_api
    _set_private(session, "_commands_dir", "/sandbox/.interactive/commands")
    _set_private(session, "_results_dir", "/sandbox/.interactive/results")
    _set_private(session, "_runner_ready", value=True)
    session.settings.timeout = 2
    session.settings.poll_interval = 0.01

    session._backend_session.install = MagicMock()  # type: ignore[method-assign]
    _set_private(session, "_check_session_timeout", MagicMock())

    def fake_copy_from_runtime(src: str, dest: str) -> None:
        req_id = Path(src).stem.split("-")[-1]
        payload = {"id": req_id, "success": True, "stdout": "ok", "stderr": ""}
        Path(dest).write_text(json.dumps(payload), encoding="utf-8")

    session._backend_session.copy_to_runtime = MagicMock()  # type: ignore[method-assign]
    session._backend_session.copy_from_runtime = MagicMock(side_effect=fake_copy_from_runtime)  # type: ignore[method-assign]

    def fake_execute_command(command: str, **_: Any) -> ConsoleOutput:
        if command.startswith("test -f"):
            return ConsoleOutput(exit_code=0, stdout="", stderr="")
        return ConsoleOutput(exit_code=0, stdout="", stderr="")

    session._backend_session.execute_command = MagicMock(side_effect=fake_execute_command)  # type: ignore[method-assign]

    result = session.run("value = 42")

    assert result.stdout == "ok"
    assert result.exit_code == 0
    session._backend_session.copy_to_runtime.assert_called_once()
    assert session._backend_session.copy_from_runtime.call_count == 1


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_run_times_out_when_result_missing() -> None:
    """Run should raise SandboxTimeoutError when the result never arrives."""
    session = InteractiveSandboxSession(kernel_type=KernelType.IPYTHON, timeout=0.2)
    # Set up session state as if it was opened
    session.container = session._backend_session.container
    session.is_open = True
    session.container_api = session._backend_session.container_api
    _set_private(session, "_commands_dir", "/sandbox/.interactive/commands")
    _set_private(session, "_results_dir", "/sandbox/.interactive/results")
    _set_private(session, "_runner_ready", value=True)
    session.settings.poll_interval = 0.01

    session._backend_session.install = MagicMock()  # type: ignore[method-assign]
    _set_private(session, "_check_session_timeout", MagicMock())
    session._backend_session.copy_to_runtime = MagicMock()  # type: ignore[method-assign]
    session._backend_session.copy_from_runtime = MagicMock()  # type: ignore[method-assign]
    interrupt_mock = MagicMock()
    _set_private(session, "_interrupt_runner", interrupt_mock)

    def fake_execute_command(command: str, **_: Any) -> ConsoleOutput:
        if command.startswith("test -f"):
            return ConsoleOutput(exit_code=1, stdout="", stderr="")
        return ConsoleOutput(exit_code=0, stdout="", stderr="")

    session._backend_session.execute_command = MagicMock(side_effect=fake_execute_command)  # type: ignore[method-assign]

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


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_open_cleans_up_on_bootstrap_failure() -> None:
    """open() should tear down the container if bootstrap fails."""
    session = InteractiveSandboxSession()
    # Mock the backend session methods
    with (
        patch.object(session, "_bootstrap_runtime", side_effect=ContainerError("boom")),
        pytest.raises(ContainerError),
    ):
        session.open()
    session._backend_session.close.assert_called_once()  # type: ignore[attr-defined]


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_run_requires_ready_runner() -> None:
    """run() should error when the interactive runner is not ready."""
    session = InteractiveSandboxSession()
    session.container = session._backend_session.container
    session.is_open = True
    session.container_api = session._backend_session.container_api
    _set_private(session, "_runner_ready", value=False)
    session._backend_session.install = MagicMock()
    with pytest.raises(ContainerError):
        session.run("print('hello')")


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_ensure_runtime_dependencies_failure() -> None:
    """_ensure_runtime_dependencies raises ContainerError when pip install fails."""
    session = InteractiveSandboxSession()
    session._backend_session.execute_command = MagicMock(return_value=ConsoleOutput(exit_code=1, stderr="fail"))
    session._backend_session.execute_commands = MagicMock(return_value=ConsoleOutput(exit_code=0))

    attr_name = "_ensure_runtime_dependencies"
    method = getattr(InteractiveSandboxSession, attr_name)
    with pytest.raises(ContainerError):
        method(session)


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_wait_for_remote_file_timeout() -> None:
    """_wait_for_remote_file returns False when the timeout expires."""
    session = InteractiveSandboxSession()
    session.settings.poll_interval = 0.01
    session._backend_session.execute_command = MagicMock(return_value=ConsoleOutput(exit_code=1))
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
            payload = cast("dict[str, Any]", json.loads(result_path.read_text(encoding="utf-8")))
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
    except docker.errors.DockerException as exc:
        pytest.skip(f"Docker daemon unavailable: {exc}")

    with InteractiveSandboxSession(lang="python", kernel_type="ipython", timeout=90) as session:
        first = session.run("value = 21 * 2")
        assert first.exit_code == 0

        second = session.run("print(value)")
        assert "42" in second.stdout

        magic = session.run("%who")
        assert "value" in magic.stdout


# Multi-backend support tests
def test_create_backend_session_docker() -> None:
    """Test _create_backend_session creates Docker session."""
    from llm_sandbox.interactive import _create_backend_session

    with patch("llm_sandbox.docker.SandboxDockerSession") as mock_docker:
        mock_instance = MagicMock()
        mock_docker.return_value = mock_instance
        result = _create_backend_session(SandboxBackend.DOCKER, runtime_configs={})
        mock_docker.assert_called_once()
        assert result == mock_instance


def test_create_backend_session_podman() -> None:
    """Test _create_backend_session creates Podman session."""
    from llm_sandbox.interactive import _create_backend_session

    with patch("llm_sandbox.podman.SandboxPodmanSession") as mock_podman:
        mock_instance = MagicMock()
        mock_podman.return_value = mock_instance
        result = _create_backend_session(SandboxBackend.PODMAN, runtime_configs={})
        mock_podman.assert_called_once()
        assert result == mock_instance


def test_create_backend_session_kubernetes() -> None:
    """Test _create_backend_session creates Kubernetes session."""
    from llm_sandbox.interactive import _create_backend_session

    with patch("llm_sandbox.kubernetes.SandboxKubernetesSession") as mock_k8s:
        mock_instance = MagicMock()
        mock_k8s.return_value = mock_instance
        result = _create_backend_session(SandboxBackend.KUBERNETES)
        mock_k8s.assert_called_once()
        assert result == mock_instance


def test_create_backend_session_unsupported() -> None:
    """Test _create_backend_session raises error for unsupported backend."""
    from llm_sandbox.interactive import _create_backend_session

    with pytest.raises(UnsupportedBackendError):
        _create_backend_session(SandboxBackend.MICROMAMBA)


# Delegation methods tests
@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_execute_command_delegation() -> None:
    """Test execute_command delegates to backend session."""
    session = InteractiveSandboxSession()
    mock_output = ConsoleOutput(exit_code=0, stdout="test", stderr="")
    session._backend_session.execute_command = MagicMock(return_value=mock_output)

    result = session.execute_command("echo test")
    assert result == mock_output
    session._backend_session.execute_command.assert_called_once_with("echo test", workdir=None)


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_execute_commands_delegation() -> None:
    """Test execute_commands delegates to backend session."""
    session = InteractiveSandboxSession()
    mock_output = ConsoleOutput(exit_code=0, stdout="test", stderr="")
    session._backend_session.execute_commands = MagicMock(return_value=mock_output)

    commands: list[str | tuple[str, str | None]] = ["echo test1", "echo test2"]
    result = session.execute_commands(commands)
    assert result == mock_output
    session._backend_session.execute_commands.assert_called_once_with(commands, workdir=None)


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_copy_to_runtime_delegation() -> None:
    """Test copy_to_runtime delegates to backend session."""
    session = InteractiveSandboxSession()
    session._backend_session.copy_to_runtime = MagicMock()  # type: ignore[method-assign]

    session.copy_to_runtime("/tmp/src", "/sandbox/dest")
    session._backend_session.copy_to_runtime.assert_called_once_with("/tmp/src", "/sandbox/dest")


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_copy_from_runtime_delegation() -> None:
    """Test copy_from_runtime delegates to backend session."""
    session = InteractiveSandboxSession()
    session._backend_session.copy_from_runtime = MagicMock()

    session.copy_from_runtime("/sandbox/src", "/tmp/dest")
    session._backend_session.copy_from_runtime.assert_called_once_with("/sandbox/src", "/tmp/dest")


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_close_stops_runner() -> None:
    """Test close() stops the runner process and closes backend session."""
    session = InteractiveSandboxSession()
    session._backend_session.close = MagicMock()
    session._backend_session.execute_command = MagicMock(return_value=ConsoleOutput(exit_code=0))
    _set_private(session, "_runner_ready", value=True)

    session.close()
    session._backend_session.close.assert_called_once()
    # Verify stop runner was called (it calls execute_command)
    assert session._backend_session.execute_command.called


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_close_without_runner() -> None:
    """Test close() works when runner is not ready."""
    session = InteractiveSandboxSession()
    session._backend_session.close = MagicMock()
    _set_private(session, "_runner_ready", value=False)
    session.is_open = True

    # Verify that _stop_session_timer is NOT called on InteractiveSandboxSession
    # (backend session manages the timer, so we don't stop a timer that was never started)
    with patch.object(session, "_stop_session_timer") as mock_stop_timer:
        session.close()

        # Verify _stop_session_timer was NOT called on InteractiveSandboxSession
        # (backend session manages the timer cleanup)
        mock_stop_timer.assert_not_called()
        # Verify backend session.close() was called
        session._backend_session.close.assert_called_once()
        # Verify session state is set to closed
        assert session.is_open is False


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_open_sets_container_references() -> None:
    """Test open() sets container and container_api from backend session."""
    session = InteractiveSandboxSession()
    mock_container = MagicMock()
    mock_api = MagicMock()
    session._backend_session.container = mock_container
    session._backend_session.container_api = mock_api
    session._backend_session.open = MagicMock()
    session._bootstrap_runtime = MagicMock()
    # Set session_start_time on backend session to simulate timer initialization
    mock_start_time = time.time()
    session._backend_session._session_start_time = mock_start_time

    # Verify that _start_session_timer is NOT called on InteractiveSandboxSession
    # (only the backend session should start a timer to avoid duplicates)
    with patch.object(session, "_start_session_timer") as mock_start_timer:
        session.open()

        # Verify _start_session_timer was NOT called on InteractiveSandboxSession
        # (backend session manages the timer, so we don't start a duplicate)
        mock_start_timer.assert_not_called()
        # Verify backend session.open() was called
        session._backend_session.open.assert_called_once()
        # Verify session state is synced
        assert session.is_open is True
        assert session._session_start_time == mock_start_time
        assert session.container == mock_container
        assert session.container_api == mock_api
        session._bootstrap_runtime.assert_called_once()


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_run_with_custom_timeout() -> None:
    """Test run() uses custom timeout when provided."""
    session = InteractiveSandboxSession()
    session.container = session._backend_session.container
    session.is_open = True
    session.container_api = session._backend_session.container_api
    _set_private(session, "_runner_ready", value=True)
    session.settings.timeout = 300
    session.settings.poll_interval = 0.01

    session._backend_session.install = MagicMock()
    _set_private(session, "_check_session_timeout", MagicMock())

    def fake_copy_from_runtime(src: str, dest: str) -> None:
        req_id = Path(src).stem.split("-")[-1]
        payload = {"id": req_id, "success": True, "stdout": "ok", "stderr": ""}
        Path(dest).write_text(json.dumps(payload), encoding="utf-8")

    session._backend_session.copy_to_runtime = MagicMock()
    session._backend_session.copy_from_runtime = MagicMock(side_effect=fake_copy_from_runtime)

    def fake_execute_command(command: str, **_: Any) -> ConsoleOutput:
        if command.startswith("test -f"):
            return ConsoleOutput(exit_code=0, stdout="", stderr="")
        return ConsoleOutput(exit_code=0, stdout="", stderr="")

    session._backend_session.execute_command = MagicMock(side_effect=fake_execute_command)

    result = session.run("print('test')", timeout=60.0)
    assert result.exit_code == 0


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_run_with_failure() -> None:
    """Test run() handles execution failures correctly."""
    session = InteractiveSandboxSession()
    session.container = session._backend_session.container
    session.is_open = True
    session.container_api = session._backend_session.container_api
    _set_private(session, "_runner_ready", value=True)
    session.settings.timeout = 2
    session.settings.poll_interval = 0.01

    session._backend_session.install = MagicMock()
    _set_private(session, "_check_session_timeout", MagicMock())

    def fake_copy_from_runtime(src: str, dest: str) -> None:
        req_id = Path(src).stem.split("-")[-1]
        payload = {"id": req_id, "success": False, "stdout": "", "stderr": "Error occurred"}
        Path(dest).write_text(json.dumps(payload), encoding="utf-8")

    session._backend_session.copy_to_runtime = MagicMock()
    session._backend_session.copy_from_runtime = MagicMock(side_effect=fake_copy_from_runtime)

    def fake_execute_command(command: str, **_: Any) -> ConsoleOutput:
        if command.startswith("test -f"):
            return ConsoleOutput(exit_code=0, stdout="", stderr="")
        return ConsoleOutput(exit_code=0, stdout="", stderr="")

    session._backend_session.execute_command = MagicMock(side_effect=fake_execute_command)

    result = session.run("raise ValueError('test')")
    assert result.exit_code == 1
    assert "Error occurred" in result.stderr


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_wait_for_remote_file_success() -> None:
    """Test _wait_for_remote_file returns True when file exists."""
    session = InteractiveSandboxSession()
    session.settings.poll_interval = 0.01

    call_count = 0

    def fake_execute_command(_command: str, **_: Any) -> ConsoleOutput:
        nonlocal call_count
        call_count += 1
        # First call fails, second succeeds
        if call_count == 1:
            return ConsoleOutput(exit_code=1, stdout="", stderr="")
        return ConsoleOutput(exit_code=0, stdout="", stderr="")

    session._backend_session.execute_command = MagicMock(side_effect=fake_execute_command)

    attr_name = "_wait_for_remote_file"
    wait_method = getattr(InteractiveSandboxSession, attr_name)
    result = wait_method(session, "exists", timeout=1.0)
    assert result is True


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_wait_for_remote_file_no_timeout() -> None:
    """Test _wait_for_remote_file works without timeout."""
    session = InteractiveSandboxSession()
    session.settings.poll_interval = 0.01
    session._backend_session.execute_command = MagicMock(return_value=ConsoleOutput(exit_code=0, stdout="", stderr=""))

    attr_name = "_wait_for_remote_file"
    wait_method = getattr(InteractiveSandboxSession, attr_name)
    # This will run indefinitely, so we'll use a short timeout in the test
    # In practice, this would be used with a timeout or in a controlled environment
    result = wait_method(session, "exists", timeout=0.01)
    assert result is True


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_interrupt_runner() -> None:
    """Test _interrupt_runner sends SIGINT to runner process."""
    session = InteractiveSandboxSession()
    captured_commands: list[str] = []

    def fake_execute_command(command: str, **_: Any) -> ConsoleOutput:
        captured_commands.append(command)
        if "cat" in command and "pid" in command:
            return ConsoleOutput(exit_code=0, stdout="12345", stderr="")
        return ConsoleOutput(exit_code=0, stdout="", stderr="")

    session._backend_session.execute_command = MagicMock(side_effect=fake_execute_command)

    attr_name = "_interrupt_runner"
    interrupt_method = getattr(InteractiveSandboxSession, attr_name)
    interrupt_method(session)

    # Verify execute_command was called (to check for pid file and kill)
    assert session._backend_session.execute_command.called
    assert captured_commands
    assert captured_commands[-1].startswith("/bin/sh -c ")


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_stop_runner_process() -> None:
    """Test _stop_runner_process stops the runner."""
    session = InteractiveSandboxSession()
    captured_commands: list[str] = []

    def fake_execute_command(command: str, **_: Any) -> ConsoleOutput:
        captured_commands.append(command)
        return ConsoleOutput(exit_code=0)

    session._backend_session.execute_command = MagicMock(side_effect=fake_execute_command)
    _set_private(session, "_runner_ready", value=True)

    attr_name = "_stop_runner_process"
    stop_method = getattr(InteractiveSandboxSession, attr_name)
    stop_method(session)

    # Verify execute_command was called to stop the process
    assert session._backend_session.execute_command.called
    assert captured_commands
    assert captured_commands[0].startswith("/bin/sh -c ")
    # Runner should no longer be ready
    assert not session._runner_ready


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_stop_runner_process_not_ready() -> None:
    """Test _stop_runner_process does nothing when runner is not ready."""
    session = InteractiveSandboxSession()
    session._backend_session.execute_command = MagicMock()
    _set_private(session, "_runner_ready", value=False)

    attr_name = "_stop_runner_process"
    stop_method = getattr(InteractiveSandboxSession, attr_name)
    stop_method(session)

    # Should not call execute_command when runner is not ready
    session._backend_session.execute_command.assert_not_called()


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_delegation_methods_with_backend_none() -> None:
    """Test delegation methods handle None backend session gracefully."""
    session = InteractiveSandboxSession()
    session._backend_session = None  # type: ignore[assignment]

    # These should raise AttributeError or handle None gracefully
    with pytest.raises(AttributeError):
        session.execute_command("test")

    with pytest.raises(AttributeError):
        session.install(["test"])


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_ensure_directory_exists_delegation() -> None:
    """Test _ensure_directory_exists delegates to backend session."""
    session = InteractiveSandboxSession()
    session._backend_session._ensure_directory_exists = MagicMock()

    attr_name = "_ensure_directory_exists"
    method = getattr(InteractiveSandboxSession, attr_name)
    method(session, "/test/path")

    session._backend_session._ensure_directory_exists.assert_called_once_with("/test/path")


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_ensure_ownership_delegation() -> None:
    """Test _ensure_ownership delegates to backend session."""
    session = InteractiveSandboxSession()
    session._backend_session._ensure_ownership = MagicMock()

    attr_name = "_ensure_ownership"
    method = getattr(InteractiveSandboxSession, attr_name)
    method(session, ["/path1", "/path2"])

    session._backend_session._ensure_ownership.assert_called_once_with(["/path1", "/path2"])


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_process_output_delegation() -> None:
    """Test _process_non_stream_output and _process_stream_output delegate to backend."""
    session = InteractiveSandboxSession()
    session._backend_session._process_non_stream_output = MagicMock(return_value=("stdout", "stderr"))
    session._backend_session._process_stream_output = MagicMock(return_value=("stdout", "stderr"))

    attr_name = "_process_non_stream_output"
    method = getattr(InteractiveSandboxSession, attr_name)
    result = method(session, "test_output")
    assert result == ("stdout", "stderr")

    attr_name = "_process_stream_output"
    method = getattr(InteractiveSandboxSession, attr_name)
    result = method(session, "test_output")
    assert result == ("stdout", "stderr")


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_process_output_without_backend() -> None:
    """Test _process_non_stream_output returns empty strings when backend is None."""
    session = InteractiveSandboxSession()
    session._backend_session = None  # type: ignore[assignment]

    attr_name = "_process_non_stream_output"
    method = getattr(InteractiveSandboxSession, attr_name)
    result = method(session, "test_output")
    assert result == ("", "")

    attr_name = "_process_stream_output"
    method = getattr(InteractiveSandboxSession, attr_name)
    result = method(session, "test_output")
    assert result == ("", "")


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_connect_to_existing_container_delegation() -> None:
    """Test _connect_to_existing_container delegates to backend session."""
    session = InteractiveSandboxSession()
    session._backend_session._connect_to_existing_container = MagicMock()

    attr_name = "_connect_to_existing_container"
    method = getattr(InteractiveSandboxSession, attr_name)
    method(session, "container-id")

    session._backend_session._connect_to_existing_container.assert_called_once_with("container-id")


@patch("llm_sandbox.interactive._create_backend_session", new=_stub_backend_session)
def test_handle_timeout_delegation() -> None:
    """Test _handle_timeout delegates to backend session."""
    session = InteractiveSandboxSession()
    session._backend_session._handle_timeout = MagicMock()

    attr_name = "_handle_timeout"
    method = getattr(InteractiveSandboxSession, attr_name)
    method(session)

    session._backend_session._handle_timeout.assert_called_once()
